"""프롬프트 문서에 표 유래 명제가 들어가지 않는가 (§12.3b / D-32).

## 왜 필요한가

§3 은 규칙 함수가 표를 못 보게 네 겹으로 막는다. 그런데 **그 표에서 나온
결론을 문장으로 옮겨 넣으면** 그 네 겹이 아무 일도 하지 않는다.

    "warp_m=128 은 최적을 낸 적이 없다"
      = 66형상 전부의 최적을 알아야 나오는 문장
      = 정답을 요약해 프롬프트에 넣은 것

학습 분할에서 세어도 마찬가지다 (§12.3b). 그리고 정정 이력을 프롬프트에
남기면 **뺐다고 적으면서 그 문장을 그대로 다시 쓰게 된다** (§12.3c) —
실제로 `hw/sm_86.md` 에서 그렇게 됐다.

## 한계

문자열 검사라 완벽하지 않다. 목적은 **다음에 누가 넣을 때 눈에 띄게**
하는 것이지 증명이 아니다. 구조적 방어는 `report/table_facts.py` 다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROMPTS = Path(__file__).resolve().parents[1] / "kernelrule/agents/prompts"

#: 표를 봐야만 쓸 수 있는 문장의 흔적.
_LEAK = (
    (re.compile(r"\d+\s*형상\s*중\s*\d+"), "N형상 중 M개 — 전수 집계"),
    (re.compile(r"\d+\s*/\s*66|\d+\s*/\s*61"), "N/66 · N/61 — 이 표의 분모"),
    (re.compile(r"최적(을)?\s*(낸 적이|한 적이)\s*없"), "'최적을 낸 적이 없다'"),
    (re.compile(r"최적\s*0\s*회|0\s*회\s*최적"), "'최적 0회'"),
    (re.compile(r"가설은?\s*기각"), "표로 기각한 가설"),
    (re.compile(r"이 표(에서|의)"), "'이 표에서/의' — 표를 가리킨다"),
    (re.compile(r"(정답 집합|answer set)에\s*(든|들어)"), "정답 집합 집계"),
)

#: 하드웨어 스펙은 표 없이 안다. 숫자가 있다고 누출이 아니다.
_HW_OK = re.compile(
    r"SM\s*84|101,?376|65,?536|1,?536|116\.1|729\.7|159\.1|6\s*MB|"
    r"1\.024|154\.8|768\s*GB|1350|7601|99KB")


def _prompt_files() -> list[Path]:
    return sorted(PROMPTS.rglob("*.md"))


def test_prompts_exist():
    assert _prompt_files(), f"프롬프트를 못 찾았다: {PROMPTS}"


@pytest.mark.parametrize("path", _prompt_files(), ids=lambda p: p.name)
def test_prompt_has_no_table_derived_claim(path: Path):
    """★ 표에서 나온 명제가 프롬프트에 있으면 안 된다 (§12.3b).

    걸리면 두 갈래로 처리한다.
      - 정말 표에서 나온 것   -> 뺀다. 정정 이력은 `docs/` 에 쓴다 (§12.3c)
      - 하드웨어/실행 모델    -> `_HW_OK` 에 추가하고 **왜 표 없이 아는지**
                               한 줄 남긴다
    """
    hits = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if _HW_OK.search(line):
            continue
        for pat, why in _LEAK:
            if pat.search(line):
                hits.append(f"  {path.name}:{i}  [{why}]\n    {line.strip()}")
                break
    assert not hits, (
        "프롬프트에 표 유래 명제가 있다 (§12.3b). 규칙 함수가 표를 못 보게 "
        "막아 놓고 그 표의 결론을 문장으로 넣으면 §3 이 아무 일도 하지 "
        "않는다:\n" + "\n".join(hits))


# ---------------------------------------------------------------------------
# ★ 스키마 필드 설명도 **프롬프트의 일부**다 (D-90)
# ---------------------------------------------------------------------------
#
#   구조화 출력 스키마의 `description` 은 모델에 그대로 전달된다. 그런데
#   프롬프트를 고칠 때 여기를 안 훑어서 **같은 요청 안에서 시스템 프롬프트와
#   필드 설명이 반대를 말하는 상태**가 됐다:
#
#     _rules_common.md  "w0 를 대충 내지는 마세요 ... 물리적 크기를 반영한"
#     schemas.py        "대략적이면 충분하다 — 수치 최적화기가 맞춘다"
#
#   원칙 26 의 짝이다 — 검사기와 프롬프트가 갈리면 모델은 프롬프트를 믿지만,
#   프롬프트와 스키마가 갈리면 **모델은 둘 다 본다.**


def _schema_descriptions() -> dict[str, str]:
    from kernelrule.agents.schemas import HAVE_PYDANTIC

    if not HAVE_PYDANTIC:
        pytest.skip("pydantic 없음")
    import kernelrule.agents.schemas as S

    out: dict[str, str] = {}
    for name in dir(S):
        cls = getattr(S, name)
        fields = getattr(cls, "model_fields", None)
        if not isinstance(fields, dict):
            continue
        for fname, f in fields.items():
            if getattr(f, "description", None):
                out[f"{name}.{fname}"] = f.description
    return out


def test_schema_descriptions_have_no_table_leak():
    """프롬프트에 건 누출 검사를 **스키마 설명에도** 건다."""
    bad = []
    for where, text in _schema_descriptions().items():
        for rx, why in _LEAK:
            if rx.search(text):
                bad.append(f"  {where}: {why}")
    assert not bad, ("스키마 필드 설명에 표 유래 문장이 있다 — 이것도 "
                     "모델에 간다:\n" + "\n".join(bad))


def test_shared_schema_does_not_mention_a_parent():
    """★ `RuleOutput` 은 RuleEditor 와 RuleWriter 가 **함께** 쓴다.

    설명에 부모 이야기를 넣으면 RuleWriter 가 없는 부모를 찾는다.
    `_rules_edit.md` 를 RuleWriter 에서 뺀 것과 같은 이유다 (§30.10).
    교체 지시는 RuleEditor 프롬프트의 `budget_note` 가 동적으로 넣는다.
    """
    from kernelrule.agents.schemas import HAVE_PYDANTIC

    if not HAVE_PYDANTIC:
        pytest.skip("pydantic 없음")
    import kernelrule.agents.schemas as S

    for fname in ("code", "w0"):
        d = S.RuleOutput.model_fields[fname].description or ""
        assert "부모" not in d, (
            f"RuleOutput.{fname} 설명이 부모를 말한다 — RuleWriter 에게는 "
            "부모가 없다")


def test_w0_description_agrees_with_the_prompt():
    """★ `w0` 설명이 §29 정정을 따라왔는가 (D-54, D-90).

    프롬프트는 "대충 내지 마라, 물리적 크기를 반영하라" 인데 스키마만
    "대략적이면 충분하다" 로 남아 있었다.
    """
    from kernelrule.agents.openai_client import load_prompt
    from kernelrule.agents.schemas import HAVE_PYDANTIC

    if not HAVE_PYDANTIC:
        pytest.skip("pydantic 없음")
    import kernelrule.agents.schemas as S

    d = S.RuleOutput.model_fields["w0"].description or ""
    assert "대략적이면 충분" not in d, "프롬프트와 반대를 말한다"
    assert "대충 내지 마라" in d
    assert "대충 내지는 마세요" in load_prompt("role/_rules_common.md")


# ---------------------------------------------------------------------------
# ★ 표에서 나온 것이 규칙으로 새는 좁은 길 (D-114)
# ---------------------------------------------------------------------------
#
# 정적 검사는 `p.M > 1024` 를 막는다. 그런데 **가설 문장은 그 검사를 안
# 거친다** — `claim` 이 `json.dumps` 로 RuleEditor 에 통째로 가고,
# 거기에 "M=4096 에서" 가 있으면 그것을 리터럴로 옮겨 적을 수 있다.


@pytest.mark.parametrize("claim", [
    "M=4096 에서 split-K 가 항상 진다",
    "4096x4096 형상에서 tail 이 커진다",
    "N = 11008 인 형상만 다르게 다뤄야 한다",
])
def test_hypothesis_claim_refuses_shape_sizes(claim):
    import kernelrule.agents.schemas as S

    if not S.HAVE_PYDANTIC:                              # pragma: no cover
        pytest.skip("pydantic 없음")
    with pytest.raises(Exception, match="형상 크기"):
        S.HypothesisOut(claim=claim, evidence_cases=[1])


@pytest.mark.parametrize("claim", [
    "waves < 1 인 형상에서 tail 이 커진다",
    "stages=3 인 config 이 짧은 형상에서 진다",
    "split_k=8 이면 리덕션이 지배한다",
])
def test_hypothesis_claim_allows_regimes_and_config_values(claim):
    """★ 오탐을 본다 — 세 자리 미만 config 값은 잡으면 안 된다."""
    import kernelrule.agents.schemas as S

    if not S.HAVE_PYDANTIC:                              # pragma: no cover
        pytest.skip("pydantic 없음")
    assert S.HypothesisOut(claim=claim, evidence_cases=[1]).claim == claim


def test_evidence_cases_do_not_reach_the_rule_editor():
    """사례 번호는 RuleEditor 가 볼 수 없는 것을 가리킨다."""
    import os

    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import FeatureRegistry

    os.environ.setdefault("OPENAI_API_KEY", "t")
    llm = OpenAILLM(LLMConfig(), feature_names=[], shape_values=[],
                    registry=FeatureRegistry("F1"))
    hyp = {"claim": "짧은 형상에서 진다", "evidence_cases": [17, 23, 41],
           "affected_regime": "waves < 1", "id": "H9",
           "proposed_direction": "tail 항을 세게",
           "measurable_with": ["tail_waste"]}
    txt = llm._user_prompt("rule_editor", "", parent=None, parent_n_terms=0,
                           hypothesis=hyp, analyst=True)
    block = txt.split("## 이번 가설")[1][:600]
    # ★ 허용 목록만 간다 (D-117)
    for keep in ("짧은 형상에서 진다", "tail 항을 세게", "tail_waste"):
        assert keep in block, keep
    # 나머지는 Analyst 기록용 — `hypotheses.jsonl` 에는 남는다
    for drop in ("evidence_cases", "affected_regime", "waves < 1", "H9"):
        assert drop not in block, drop


def test_prompt_no_longer_names_the_optimizer():
    """순위 경로는 L-BFGS-B 다. 이름을 적으면 또 갈린다 (pending 12)."""
    from kernelrule.agents.openai_client import load_prompt

    txt = load_prompt("role/_rules_common.md", parameters=8)
    assert "Nelder-Mead" not in txt
    assert "수치 최적화기가" in txt


def test_n_candidates_is_described_as_enumeration_not_performance():
    from kernelrule.agents.openai_client import load_prompt

    txt = load_prompt("role/_rules_common.md", parameters=8)
    assert "p.n_candidates" in txt and "성능이 아니라 열거 정보" in txt
