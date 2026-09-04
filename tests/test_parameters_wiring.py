"""★ 항 예산이 **프롬프트 전부**에 같은 값으로 간다 (D-105).

`--rule-budget 16` 으로 캠페인 하나를 돌렸는데, 검사기만 16 이었고
**시스템 프롬프트의 역할 파일과 사용자 프롬프트는 8 로 렌더링됐다.**
같은 프롬프트 안에서 `_rules_common.md` 는 "16 이하", `rule_editor.md` 는
"항 상한 8개" 라고 말하고 있었다 (문구는 D-128 에서 "파라미터 상한 N개" 로
바뀌었다 — 검사하는 것은 **네 면이 같은 숫자를 말하는가** 다). 규칙 36개 전부가 8항에서 멈췄고,
"예산을 늘려도 항이 안 는다" 로 읽힐 뻔했다.

원인은 자리가 여럿이었다는 것이다 (원칙 23):

    load_prompt(..., parameters=)   `assemble_instructions` 만 넘기고 있었다
    load_prompt("role/...")     `_agent` 와 `_optimize_prompt` 는 안 넘겼다
    checks.PARAMETERS               사용자 프롬프트가 **직접 import** 했다

옛 시험 `test_rule_writers_get_the_budget` 은 `"8" in body` 였다 — 8 은
피처 설명에도 나오므로 16 으로 바꿔도 통과한다. **바뀌는 값을 상수로
찾으면 안 된다.**
"""
from __future__ import annotations

import ast
import json
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
           and "parameters" not in {k.arg for k in n.keywords}]
    assert not bad, (
        f"`parameters=` 없이 부르는 자리가 있다: {SRC.name} 줄 {bad}. "
        "기본값으로 떨어지면 `parameters` 을 무시한다 (D-105).")


#: `checks.PARAMETERS` 을 읽어도 되는 함수. 그 밖에서 읽으면 `parameters`
#: 을 무시한다 — 조건이 조용히 8 로 돌아간다.
_MAY_READ_PARAMETERS = {"load_prompt", "__init__"}


def test_only_two_functions_read_the_module_constant():
    """★ `checks.PARAMETERS` 을 읽는 자리를 **센다**. 유효 예산은 하나다."""
    tree = ast.parse(SRC.read_text())
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if fn.name in _MAY_READ_PARAMETERS:
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and n.id in ("BUDGET", "_CHECK_BUDGET"):
                bad.append((fn.name, n.lineno))
    assert not bad, (
        f"`checks.PARAMETERS` 을 직접 읽는 자리: {bad}. "
        "`self._budget` 하나만 봐야 한다 (D-105).")


def _llm(budget: int | None):
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import FeatureRegistry

    os.environ.setdefault("OPENAI_API_KEY", "t")
    return OpenAILLM(LLMConfig(parameters=budget), feature_names=[],
                     shape_values=[], registry=FeatureRegistry("F1"))


@pytest.mark.parametrize("budget", [8, 16])
def test_user_and_system_prompts_agree_on_the_budget(budget):
    """시스템과 사용자 프롬프트가 **같은 숫자**를 말하는가."""
    from kernelrule.agents.openai_client import assemble_instructions

    llm = _llm(budget)
    other = 16 if budget == 8 else 8
    sys_p = assemble_instructions("rule_editor", objective="rank",
                                  parameters=llm._parameters)
    usr_p = llm._user_prompt("rule_editor", "", parent=None,
                             parent_n_terms=0, analyst=False)
    for name, txt in (("시스템", sys_p), ("사용자", usr_p)):
        assert (f"파라미터 상한 {budget}개" in txt
                or f"{budget} 이하" in txt), (
            f"{name} 프롬프트에 예산 {budget} 이 안 보인다")
        assert f"파라미터 상한 {other}개" not in txt, (
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


# ---------------------------------------------------------------------------
# ★ 예산에 **딸린** 상한들 (D-106)
# ---------------------------------------------------------------------------
#
# 8항 규칙의 AST 노드가 실측 중앙 271 / 최대 383 인데 상한이 400 이다.
# 예산만 16 으로 올리면 16항 규칙은 **노드 상한에서 거부된다** — "예산
# 16 이 효과가 없다" 가 아니라 "16항을 쓸 수 없었다" 를 재게 된다.


def test_limits_scale_with_the_budget():
    from kernelrule.rules.checks import LIMITS, limits_for

    assert limits_for(None) == LIMITS
    assert limits_for(8)["ast_nodes"] == LIMITS["ast_nodes"]
    assert limits_for(16)["ast_nodes"] == 2 * LIMITS["ast_nodes"]
    assert limits_for(16)["max_lines"] == 2 * LIMITS["max_lines"]


def test_prompt_states_the_scaled_node_cap():
    """프롬프트가 상수 400 을 말하면 검사기(800)와 갈린다."""
    from kernelrule.agents.openai_client import load_prompt
    from kernelrule.rules.checks import limits_for

    for b in (8, 16):
        txt = load_prompt("role/_rules_edit.md", parameters=b)
        n = limits_for(b)["ast_nodes"]
        assert f"AST 노드 {n}개" in txt, f"예산 {b} 에서 노드 상한이 안 맞는다"


def test_a_sixteen_term_rule_fits_only_under_the_raised_cap():
    """★ 실제 16항 규칙으로 확인한다 — 숫자만 맞춰 놓으면 소용없다."""
    from kernelrule.rules.checks import check_rule, limits_for

    terms = "\n".join(
        f"    s = s + np.where(p.is_memory_bound, f.log_dram_traffic, "
        f"f.log_inst_total) * w[{i}]" for i in range(1, 16))
    code = ("def score(f, p, hw, w):\n"
            "    s = f.reg_pressure * w[0]\n" + terms + "\n    return s")
    kw = {"feature_names": ["reg_pressure", "log_dram_traffic",
                            "log_inst_total"],
          "shape_value_names": ["is_memory_bound"], "n_weights": 16}
    lo = check_rule(code, limits=limits_for(8), **kw)
    hi = check_rule(code, limits=limits_for(16), **kw)
    assert not lo.ok, "예산 8 에서 16항이 통과하면 검사기가 안 걸러진다"
    assert hi.n_nodes > limits_for(8)["ast_nodes"], (
        f"노드가 {hi.n_nodes} 뿐이라 상한 검사를 못 건드린다 — "
        "이 시험이 재려던 것을 못 잰다")
    assert hi.ok, f"예산 16 에서도 거부됐다: {hi.violations}"


# ---------------------------------------------------------------------------
# ★ 출력 스키마 (D-107) — 네 번째 자리
# ---------------------------------------------------------------------------
#
# `RuleOutput` 의 필드 설명은 `pydantic-ai` 가 **도구 스키마로 모델에
# 보낸다.** 거기에 "★ 항은 최대 8개" 가 굳어 있어서, 프롬프트가 "상한
# 16개" 라고 말해도 예산 16 캠페인의 규칙 29개가 전부 8항이었다.
# 스키마 거부는 36라운드 내내 0 이었다 — 모델은 시도조차 안 했다.


def _twelve_terms() -> str:
    body = "".join(f"    s = s + f.edge_waste * w[{i}]\n" for i in range(1, 12))
    return ("def score(f, p, hw, w):\n"
            "    s = f.reg_pressure * w[0]\n" + body + "    return s")


@pytest.mark.parametrize("budget", [8, 16])
def test_output_schema_states_the_budget(budget):
    from kernelrule.agents.schemas import rule_output_for

    schema = rule_output_for(budget).model_json_schema()
    for fld in ("code", "w0"):
        d = schema["properties"][fld]["description"]
        assert f"최대 {budget}개" in d, (
            f"출력 스키마 {fld} 설명이 예산 {budget} 을 안 말한다 — "
            "모델은 이 문장을 보고 항 수를 정한다 (D-107)")


def test_output_schema_validation_follows_the_budget():
    """★ 설명만 고치고 검증이 8 이면 모델은 시도했다가 거부당한다."""
    from kernelrule.agents.schemas import rule_output_for

    kw = {"code": _twelve_terms(), "w0": [1.0] * 12, "changes": "",
          "hypothesis_id": ""}
    with pytest.raises(Exception, match="12"):
        rule_output_for(8)(**kw)
    rule_output_for(16)(**kw)          # 예산 16 에서는 통과해야 한다


def test_dict_path_validation_follows_the_budget():
    """MockLLM / 구조화 출력을 안 쓰는 경로도 같은 예산을 봐야 한다."""
    from kernelrule.agents.schemas import (
        SchemaViolation,
        validate_rule_proposal,
    )

    d = {"code": _twelve_terms(), "w0": [1.0] * 12}
    with pytest.raises(SchemaViolation):
        validate_rule_proposal(d)
    validate_rule_proposal(d, parameters=16)


#: ★ 예산 숫자가 나가는 **모든 면**. 하나라도 빠지면 조건이 갈린다.
#:
#:   D-105  검사기만 닿았다 (프롬프트 파일 / 사용자 프롬프트)
#:   D-106  딸린 상한(ast_nodes)이 안 따라왔다
#:   D-107  출력 스키마 설명이 8 로 굳어 있었다
def test_all_four_surfaces_say_the_same_budget():
    from kernelrule.agents.openai_client import assemble_instructions
    from kernelrule.agents.schemas import rule_output_for
    from kernelrule.rules.checks import limits_for

    for b in (8, 16):
        llm = _llm(b)
        surfaces = {
            "시스템 프롬프트": assemble_instructions(
                "rule_editor", objective="rank", parameters=llm._parameters),
            "사용자 프롬프트": llm._user_prompt(
                "rule_editor", "", parent=None, parent_n_terms=0,
                analyst=False),
            "출력 스키마": json.dumps(
                rule_output_for(b).model_json_schema(), ensure_ascii=False),
        }
        for name, txt in surfaces.items():
            assert (f"파라미터 상한 {b}개" in txt or f"{b} 이하" in txt
                    or f"최대 {b}개" in txt), f"{name} 가 예산 {b} 을 안 말한다"
        assert limits_for(b)["parameters"] == b


# ---------------------------------------------------------------------------
# ★ 목표 정의의 숫자 (k, lambda) 와 곱 항 (D-109 / D-110)
# ---------------------------------------------------------------------------
#
# `rank` 목표 블록이 "config 100개" 를 **상수로** 적고 있었다. `k` 스윕을
# 그대로 돌렸으면 프롬프트만 100 이라고 말하는 다섯 번째 면이 됐다.


@pytest.mark.parametrize("k", [10, 20, 100])
def test_objective_block_states_the_running_k(k):
    from kernelrule.agents.openai_client import assemble_instructions

    txt = assemble_instructions("rule_editor", objective="rank", parameters=8,
                                rank_top_k=k)
    assert f"config {k}개" in txt, f"목표 정의가 k={k} 를 안 말한다"
    for other in (10, 20, 100):
        if other != k:
            assert f"config {other}개" not in txt


def test_objective_block_states_lambda_only_when_set():
    from kernelrule.agents.openai_client import assemble_instructions

    kw = {"objective": "rank", "parameters": 8}
    assert "가중치 1 를" not in assemble_instructions("rule_editor", **kw)
    on = assemble_instructions("rule_editor", rank_lambda=1.0, **kw)
    assert "참 1등을 맞히는 것" in on


def test_product_hint_is_off_by_default_and_lands_on_every_surface():
    from kernelrule.agents.openai_client import assemble_instructions
    from kernelrule.agents.schemas import rule_output_for

    off = assemble_instructions("rule_editor", objective="rank", parameters=8)
    on = assemble_instructions("rule_editor", objective="rank", parameters=8,
                               product_hint=True)
    assert "{product_block}" not in off and "{product_note}" not in off
    assert "피처를 곱해도 됩니다" not in off
    assert "피처를 곱해도 됩니다" in on and "피처 둘을 곱한 항" in on
    for ph in (False, True):
        d = rule_output_for(8, product_hint=ph).model_json_schema()
        has = "곱해도 된다" in d["properties"]["code"]["description"]
        assert has is ph, f"스키마 product_hint={ph} 인데 곱 문장 {has}"
