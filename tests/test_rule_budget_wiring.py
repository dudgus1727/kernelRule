"""★ 항 예산이 **프롬프트 전부**에 같은 값으로 간다 (D-105).

`--rule-budget 16` 으로 캠페인 하나를 돌렸는데, 검사기만 16 이었고
**시스템 프롬프트의 역할 파일과 사용자 프롬프트는 8 로 렌더링됐다.**
같은 프롬프트 안에서 `_rules_common.md` 는 "16 이하", `rule_editor.md` 는
"항 상한 8개" 라고 말하고 있었다. 규칙 36개 전부가 8항에서 멈췄고,
"예산을 늘려도 항이 안 는다" 로 읽힐 뻔했다.

원인은 자리가 여럿이었다는 것이다 (원칙 23):

    load_prompt(..., budget=)   `assemble_instructions` 만 넘기고 있었다
    load_prompt("role/...")     `_agent` 와 `_optimize_prompt` 는 안 넘겼다
    checks.BUDGET               사용자 프롬프트가 **직접 import** 했다

옛 시험 `test_rule_writers_get_the_budget` 은 `"8" in body` 였다 — 8 은
피처 설명에도 나오므로 16 으로 바꿔도 통과한다. **바뀌는 값을 상수로
찾으면 안 된다.**
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "kernelrule/agents/openai_client.py"


def test_every_load_prompt_call_passes_the_budget():
    """★ 호출부를 **전부** 센다 — 하나만 빠져도 프롬프트가 갈린다."""
    tree = ast.parse(SRC.read_text())
    bad = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
           and n.func.id == "load_prompt"
           and "budget" not in {k.arg for k in n.keywords}]
    assert not bad, (
        f"`budget=` 없이 부르는 자리가 있다: {SRC.name} 줄 {bad}. "
        "기본값으로 떨어지면 `rule_budget` 을 무시한다 (D-105).")


#: `checks.BUDGET` 을 읽어도 되는 함수. 그 밖에서 읽으면 `rule_budget`
#: 을 무시한다 — 조건이 조용히 8 로 돌아간다.
_MAY_READ_BUDGET = {"load_prompt", "__init__"}


def test_only_two_functions_read_the_module_constant():
    """★ `checks.BUDGET` 을 읽는 자리를 **센다**. 유효 예산은 하나다."""
    tree = ast.parse(SRC.read_text())
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if fn.name in _MAY_READ_BUDGET:
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and n.id in ("BUDGET", "_CHECK_BUDGET"):
                bad.append((fn.name, n.lineno))
    assert not bad, (
        f"`checks.BUDGET` 을 직접 읽는 자리: {bad}. "
        "`self._budget` 하나만 봐야 한다 (D-105).")


def _llm(budget: int | None):
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import FeatureRegistry

    os.environ.setdefault("OPENAI_API_KEY", "t")
    return OpenAILLM(LLMConfig(rule_budget=budget), feature_names=[],
                     shape_values=[], registry=FeatureRegistry("F1"))


@pytest.mark.parametrize("budget", [8, 16])
def test_user_and_system_prompts_agree_on_the_budget(budget):
    """시스템과 사용자 프롬프트가 **같은 숫자**를 말하는가."""
    from kernelrule.agents.openai_client import assemble_instructions

    llm = _llm(budget)
    other = 16 if budget == 8 else 8
    sys_p = assemble_instructions("rule_editor", objective="rank",
                                  budget=llm._budget)
    usr_p = llm._user_prompt("rule_editor", "", parent=None,
                             parent_n_terms=0, analyst=False)
    for name, txt in (("시스템", sys_p), ("사용자", usr_p)):
        assert f"항 상한 {budget}개" in txt or f"{budget} 이하" in txt, (
            f"{name} 프롬프트에 예산 {budget} 이 안 보인다")
        assert f"항 상한 {other}개" not in txt, (
            f"{name} 프롬프트가 {other} 를 말한다 — 조건이 갈렸다 (D-105)")


def test_budget_reaches_the_saturation_notice():
    """포화 문구도 유효 예산을 봐야 한다 — 8항짜리 부모가 16 예산에서
    '예산이 찼습니다' 를 받으면 항을 절대 못 늘린다."""
    llm = _llm(16)
    txt = llm._user_prompt("rule_editor", "", parent=None,
                           parent_n_terms=8, analyst=False)
    assert "예산이 찼습니다" not in txt, (
        "예산 16 인데 8항 부모에게 '예산이 찼습니다' 를 보냈다 (D-105)")
    assert "남은 예산: 8항" in txt, txt[:400]
