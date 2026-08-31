"""MockLLM 과 스키마 경계 (§24, §11.7)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from kernelrule.agents.mock import (
    ADVERSARIAL_CASES,
    MockLLM,
    _parse_terms,
    _render_rule,
)
from kernelrule.agents.schemas import (
    RuleProposal,
    SchemaViolation,
    validate_rule_proposal,
)

FEATS = ["traffic_amplification", "has_spill", "is_two_stage",
         "log_workspace_bytes", "sm_idle_cost", "split_k_cost",
         "smem_pressure"]


# ---------------------------------------------------------------------------
# 스키마 경계 — **부분 수용하지 않는다** (§26.4)
# ---------------------------------------------------------------------------
BAD_RESPONSES = [
    ("함수 없음", {"code": "x = 1", "w0": [1.0]}),
    ("정답 참조", {"code": "def score(f,p,hw,w): return time_ms", "w0": [1.0]}),
    ("난이도 참조", {"code": "def score(f,p,hw,w): return difficulty",
                     "w0": [1.0]}),
    ("import", {"code": "def score(f,p,hw,w):\n import os\n return 1",
                "w0": [1.0]}),
    ("w0 빔", {"code": "def score(f,p,hw,w): return 1", "w0": []}),
    ("w0 문자열", {"code": "def score(f,p,hw,w): return 1", "w0": ["a"]}),
    ("w0 폭주", {"code": "def score(f,p,hw,w): return 1", "w0": [1e9]}),
    ("dict 아님", ["code"]),
]


@pytest.mark.parametrize("name,obj", BAD_RESPONSES,
                         ids=[c[0] for c in BAD_RESPONSES])
def test_schema_violation_is_discarded(name, obj):
    with pytest.raises(SchemaViolation):
        validate_rule_proposal(obj)


def test_valid_response_passes():
    p = validate_rule_proposal(
        {"code": "def score(f, p, hw, w):\n    return f.waves * w[0]\n",
         "w0": [2.0], "changes": "x"})
    assert isinstance(p, RuleProposal) and p.w0 == [2.0]


# ---------------------------------------------------------------------------
# ★ adversarial — 하나라도 통과하면 방어에 구멍이 있다 (§24.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,code,w0", ADVERSARIAL_CASES,
                         ids=[c[0] for c in ADVERSARIAL_CASES])
def test_adversarial_case_is_blocked_somewhere(name, code, w0):
    """스키마 / 정적 검사 / 샌드박스 **어딘가에서** 반드시 걸린다."""
    from kernelrule.rules.checks import check_rule

    blocked = False
    try:
        validate_rule_proposal({"code": code, "w0": w0})
    except SchemaViolation:
        blocked = True
    if not blocked:
        rep = check_rule(code, feature_names=set(FEATS) | {"waves",
                         "tail_waste", "edge_waste"},
                         shape_value_names={"is_memory_bound", "M"},
                         n_weights=len(w0))
        blocked = not rep.ok
    if not blocked:
        from kernelrule.core.matrix import Feats, ShapeInfo
        from kernelrule.core.sandbox import run_isolated
        f = Feats({n: np.ones(4) for n in FEATS + ["waves"]})
        out = run_isolated(code, (f, ShapeInfo({"is_memory_bound": 0.0}),
                                  None, np.asarray(w0)), timeout=6.0)
        blocked = not out.ok
    assert blocked, f"{name} 이 모든 방어를 통과했다"


# ---------------------------------------------------------------------------
# mutate — ★ 구조를 섭동한다
# ---------------------------------------------------------------------------
def test_render_parse_roundtrip():
    code, w0 = _render_rule(["f.a", "f.b"], ("is_memory_bound", "f.c"))
    terms, branch = _parse_terms(code)
    assert terms == ["f.a", "f.b"] and branch == ("is_memory_bound", "f.c")
    assert len(w0) == 3


def test_mutate_changes_structure_not_just_weights():
    """★ 가중치만 흔들면 아무것도 시험하지 못한다 — `fit_weights` 가 맞춘다."""
    m = MockLLM("mutate", seed=1, feature_names=FEATS)
    parent = RuleProposal(code="def score(f, p, hw, w):\n"
                               "    s = f.traffic_amplification * w[0]\n"
                               "    return s\n", w0=[1.0])
    codes = set()
    for _ in range(12):
        out = m.complete("rule_editor", "x", parent=parent,
                         hypothesis={"measurable_with": ["has_spill"]})
        codes.add(out["code"])
    assert len(codes) > 1, "구조가 하나도 안 바뀐다"
    # 항 개수가 실제로 달라진다
    sizes = {len(_parse_terms(c)[0]) for c in codes}
    assert len(sizes) > 1, f"항 개수가 고정이다: {sizes}"


def test_mutate_follows_the_hypothesis():
    """가설이 지목한 피처를 우선 추가한다 — 진단이 기여하는 경로다."""
    m = MockLLM("mutate", seed=3, feature_names=FEATS)
    parent = RuleProposal(code="def score(f, p, hw, w):\n"
                               "    s = f.traffic_amplification * w[0]\n"
                               "    return s\n", w0=[1.0])
    hits = 0
    for _ in range(20):
        out = m.complete("rule_editor", "x", parent=parent,
                         hypothesis={"measurable_with": ["is_two_stage"]})
        if "f.is_two_stage" in out["code"]:
            hits += 1
    assert hits >= 8, f"가설을 따른 것이 {hits}/20 뿐이다"


def test_mutate_respects_the_literal_budget():
    m = MockLLM("mutate", seed=5, feature_names=FEATS * 3)
    p = None
    for _ in range(40):
        out = m.complete("rule_editor", "x", parent=p,
                         hypothesis={"measurable_with": FEATS})
        assert len(out["w0"]) <= 8, out["w0"]
        p = validate_rule_proposal(out)


def test_diagnose_reads_the_unused_column():
    """★ 진단이 리포트의 `★ 미사용` 열을 읽는다. 루프 배관의 시험이다."""
    m = MockLLM("mutate", seed=0, feature_names=FEATS)
    report = ("is_two_stage       1.0   0.0  ★ 미사용\n"
              "log_workspace_bytes 22.0 0.0  ★ 미사용\n"
              "split_k_cost       0.5   0.0  사용 중\n")
    out = m.complete("analyze", report)
    names = [h["measurable_with"][0] for h in out["hypotheses"]]
    assert "is_two_stage" in names and "log_workspace_bytes" in names
    assert "split_k_cost" not in names


# ---------------------------------------------------------------------------
# replay — 결정론적 재현
# ---------------------------------------------------------------------------
def test_replay_reproduces_exactly(tmp_path):
    a = MockLLM("mutate", seed=11, feature_names=FEATS)
    outs = [a.complete("rule_editor", f"p{i}") for i in range(6)]
    a.dump(tmp_path / "calls")
    b = MockLLM("replay", replay_dir=tmp_path / "calls")
    assert [b.complete("rule_editor", f"p{i}") for i in range(6)] == outs


def test_replay_missing_dir_is_an_error(tmp_path):
    """★ 조용히 canned 로 떨어지지 않는다 (§26.4)."""
    with pytest.raises(FileNotFoundError):
        MockLLM("replay", replay_dir=tmp_path / "nope")


def test_replay_detects_a_changed_loop(tmp_path):
    a = MockLLM("mutate", seed=2, feature_names=FEATS)
    a.complete("rule_editor", "x")
    a.dump(tmp_path / "c")
    b = MockLLM("replay", replay_dir=tmp_path / "c")
    b.complete("rule_editor", "x")
    with pytest.raises(SchemaViolation, match="replay"):
        b.complete("rule_editor", "y")


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError, match="알 수 없는 모드"):
        MockLLM("wishful")


def test_deterministic_across_instances():
    a = MockLLM("mutate", seed=9, feature_names=FEATS)
    b = MockLLM("mutate", seed=9, feature_names=FEATS)
    assert ([a.complete("rule_editor", "x") for _ in range(5)]
            == [b.complete("rule_editor", "x") for _ in range(5)])


# ---------------------------------------------------------------------------
# 강제 장치의 세 곳이 일치하는가 (D-26) — §30.8 패턴
# ---------------------------------------------------------------------------
# 설명 / validator / 에러 메시지가 셋 다 다른 채로 굴러갔다. 셋을 상수 하나로
# 묶었으므로 **그 사실을 테스트가 고정한다** — 나중에 한 곳만 고치는 것을
# 막는 것이 목적이다.

def test_hypothesis_count_desc_and_validator_share_one_constant():
    from kernelrule.agents import schemas as S
    if not S.HAVE_PYDANTIC:
        pytest.skip("pydantic 없음")
    desc = S.AnalysisOutput.model_fields["hypotheses"].description
    assert f"{S.N_HYP_MIN}~{S.N_HYP_MAX}" in desc

    def mk(n):
        return S.AnalysisOutput(hypotheses=[{"claim": f"가설 {i}"}
                                             for i in range(n)])

    mk(S.N_HYP_MIN)                                   # 하한은 통과
    for n in (S.N_HYP_MIN - 1, S.N_HYP_MAX + 1):      # 밖은 거부
        with pytest.raises(Exception) as ei:
            mk(n)
        # 에러 메시지도 같은 상수를 말해야 한다
        assert f"{S.N_HYP_MIN}~{S.N_HYP_MAX}" in str(ei.value)


def test_weight_cap_has_one_source_of_truth():
    from kernelrule.agents.schemas import MAX_WEIGHTS
    from kernelrule.rules.checks import LIMITS
    assert LIMITS["budget"] == MAX_WEIGHTS


def test_mock_and_real_paths_enforce_the_same_budget():
    """§24 — `validate_rule_proposal` 에만 w0 길이 검사가 없었다.

    목으로 개발하면 예산 초과가 안 잡히고 실제 LLM 에서만 잡혔다.
    """
    from kernelrule.agents.schemas import (
        MAX_WEIGHTS,
        SchemaViolation,
        validate_rule_proposal,
    )
    code = "def score(f, p, hw, w):\n    return f.waves * w[0]\n"
    validate_rule_proposal({"code": code, "w0": [1.0] * MAX_WEIGHTS})
    with pytest.raises(SchemaViolation, match="예산"):
        validate_rule_proposal({"code": code, "w0": [1.0] * (MAX_WEIGHTS + 1)})


# ---------------------------------------------------------------------------
# 금지어 부분 매칭이 주석을 잡지 않는가 (D-27)
# ---------------------------------------------------------------------------

_HDR = "def score(f, p, hw, w):\n"


@pytest.mark.parametrize("code,banned", [
    # 주석/문자열 안의 것은 실행되지 않는다 -> 잡지 않는다
    (_HDR + "    # 난이도(difficulty)가 높은 형상이다\n"
            "    return f.tail_waste * w[0]\n", None),
    (_HDR + "    note = 'import os 는 금지다'\n"
            "    return f.waves * w[0]\n", None),
    # 실제 코드는 여전히 잡는다
    (_HDR + "    return f.difficulty * w[0]\n", "difficulty"),
    ("import os\n" + _HDR + "    return f.waves * w[0]\n", "import "),
    (_HDR + "    return TABLE.time_ms * w[0]\n", "time_ms"),
])
def test_banned_check_ignores_comments_and_strings(code, banned):
    from kernelrule.agents.schemas import check_banned
    assert check_banned(code) == banned


def test_banned_check_never_skips_on_tokenize_failure():
    """문법 오류면 **원본을 보수적으로 검사한다** (§26.4)."""
    from kernelrule.agents.schemas import check_banned
    assert check_banned("def score(  # 안 닫힘\n  import os") == "import "


# ---------------------------------------------------------------------------
# Pydantic 부재를 쓰려는 순간 알리는가 (4-5)
# ---------------------------------------------------------------------------

def test_missing_pydantic_fails_loudly():
    from kernelrule.agents.schemas import _NoPydantic
    stub = _NoPydantic("AnalysisOutput")
    with pytest.raises(ImportError, match="비활성화된 상태"):
        stub()
    with pytest.raises(ImportError, match="비활성화된 상태"):
        _ = stub.model_validate       # 속성 접근만으로도 알린다


def test_changes_is_optional():
    """계보 추적용이다. 비었다고 규칙을 버리면 재시도만 소진한다 (4-4)."""
    from kernelrule.agents import schemas as S
    if not S.HAVE_PYDANTIC:
        pytest.skip("pydantic 없음")
    out = S.RuleOutput(code="def score(f, p, hw, w):\n    return f.waves*w[0]",
                       w0=[1.0])
    assert out.changes == ""


# ---------------------------------------------------------------------------
# FeatureWriter — F0~F3 조건 (§11.4)
# ---------------------------------------------------------------------------
# 근본 질문은 "LLM 이 물리량을 **만들** 수 있나" 다. 그러려면 조건마다
# 무엇을 주는지가 엄밀해야 한다 — F1 에 기존 피처 이름이 하나라도 새면
# 그 실험은 "조합만 했나" 를 다시 물은 것이 된다.

def _feat_client():
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import REGISTRY
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    return OpenAILLM(LLMConfig(model="m"), feature_names=[], shape_values=[],
                     registry=REGISTRY, cache=False), REGISTRY


@pytest.mark.parametrize("cond", ["F0", "F1"])
def test_f0_f1_leak_no_existing_feature_name(cond):
    """★ 원시 값 조건에 기존 피처 이름이 들어가면 안 된다."""
    c, reg = _feat_client()
    text = c._feature_prompt(condition=cond)
    leaked = [n for n in reg._items if n in text]
    assert not leaked, f"{cond} 프롬프트에 기존 피처가 샜다: {leaked}"


def test_f3_shows_the_existing_features():
    c, reg = _feat_client()
    text = c._feature_prompt(condition="F3")
    assert "tail_waste" in text and "has_spill" in text


def test_unknown_condition_is_rejected():
    c, _ = _feat_client()
    with pytest.raises(ValueError, match="알 수 없는 조건"):
        c._feature_prompt(condition="F9")


def test_feature_prompt_example_uses_no_real_feature(monkeypatch):
    """D-35 — **F0/F1 의** 형태 예시가 실제 피처를 건네주면 안 된다.

    공개 지식을 주는 조건(F1-K/F2/F3)은 실제 피처를 코드까지 보여준다 —
    그것이 조건의 정의다 (§30.17). 그쪽은 `examples/known5.md` 다.
    """
    from kernelrule.agents.openai_client import load_prompt
    from kernelrule.features import REGISTRY
    body = load_prompt("examples/other_domain.md")
    start = body.index("```python")
    example = body[start:body.index("```", start + 3)]
    leaked = [n for n in REGISTRY._items if n in example]
    assert not leaked, f"형태 예시가 실제 피처를 담고 있다: {leaked}"


def test_prompts_never_hardcode_the_budget_number(monkeypatch):
    """★ 예산 숫자의 출처는 `checks.BUDGET` **하나**다.

    프롬프트 다섯 파일과 스키마와 검사기가 각자 8 을 적고 있었다. 바꾸면
    하나를 빠뜨린다 — `is_reference` / `top_k` / `DEFAULT_MODEL` /
    `REGISTRY` / `load_generated` 에 이은 여섯 번째가 된다.

    `checks.BUDGET` 을 바꿨을 때 프롬프트가 따라 바뀌면 단일 출처다.
    """
    from kernelrule.agents.openai_client import load_prompt
    from kernelrule.rules import checks

    files = ["role/_rules_common.md", "role/_rules_edit.md",
             "role/rule_editor.md", "role/rule_writer.md"]
    monkeypatch.setattr(checks, "BUDGET", 16)
    for f in files:
        txt = load_prompt(f)
        assert "{budget}" not in txt, f"{f}: 치환이 안 됐다"
        assert "16" in txt, f"{f}: 예산이 프롬프트에 안 흘러간다"
        # 예산 문장에 옛 숫자가 남아 있으면 안 된다
        for line in txt.splitlines():
            if "예산" in line or "상한" in line or "리터럴" in line:
                assert " 8 " not in line and "8개" not in line \
                    and "<= 8" not in line, f"{f}: 굳은 8 이 남았다 — {line}"


def test_prompt_tells_the_model_branch_constants_are_free():
    """★ 규칙을 바꿨으면 **모델도 알아야 한다** (D-78).

    검사기만 풀고 프롬프트를 그대로 두면 모델은 계속 우회한다 — 제약이
    풀린 것을 모르기 때문이다.
    """
    from kernelrule.agents.openai_client import load_prompt

    for f in ("role/_rules_common.md", "role/rule_writer.md"):
        txt = load_prompt(f)
        assert "분기" in txt and "비교 상수" in txt, f"{f}: 면제 설명이 없다"
