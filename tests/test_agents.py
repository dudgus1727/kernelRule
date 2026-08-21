"""MockLLM 과 스키마 경계 (§24, §11.7)."""
from __future__ import annotations

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
        out = m.complete("optimize", "x", parent=parent,
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
        out = m.complete("optimize", "x", parent=parent,
                         hypothesis={"measurable_with": ["is_two_stage"]})
        if "f.is_two_stage" in out["code"]:
            hits += 1
    assert hits >= 8, f"가설을 따른 것이 {hits}/20 뿐이다"


def test_mutate_respects_the_literal_budget():
    m = MockLLM("mutate", seed=5, feature_names=FEATS * 3)
    p = None
    for _ in range(40):
        out = m.complete("optimize", "x", parent=p,
                         hypothesis={"measurable_with": FEATS})
        assert len(out["w0"]) <= 8, out["w0"]
        p = validate_rule_proposal(out)


def test_diagnose_reads_the_unused_column():
    """★ 진단이 리포트의 `★ 미사용` 열을 읽는다. 루프 배관의 시험이다."""
    m = MockLLM("mutate", seed=0, feature_names=FEATS)
    report = ("is_two_stage       1.0   0.0  ★ 미사용\n"
              "log_workspace_bytes 22.0 0.0  ★ 미사용\n"
              "split_k_cost       0.5   0.0  사용 중\n")
    out = m.complete("diagnose", report)
    names = [h["measurable_with"][0] for h in out["hypotheses"]]
    assert "is_two_stage" in names and "log_workspace_bytes" in names
    assert "split_k_cost" not in names


# ---------------------------------------------------------------------------
# replay — 결정론적 재현
# ---------------------------------------------------------------------------
def test_replay_reproduces_exactly(tmp_path):
    a = MockLLM("mutate", seed=11, feature_names=FEATS)
    outs = [a.complete("optimize", f"p{i}") for i in range(6)]
    a.dump(tmp_path / "calls")
    b = MockLLM("replay", replay_dir=tmp_path / "calls")
    assert [b.complete("optimize", f"p{i}") for i in range(6)] == outs


def test_replay_missing_dir_is_an_error(tmp_path):
    """★ 조용히 canned 로 떨어지지 않는다 (§26.4)."""
    with pytest.raises(FileNotFoundError):
        MockLLM("replay", replay_dir=tmp_path / "nope")


def test_replay_detects_a_changed_loop(tmp_path):
    a = MockLLM("mutate", seed=2, feature_names=FEATS)
    a.complete("optimize", "x")
    a.dump(tmp_path / "c")
    b = MockLLM("replay", replay_dir=tmp_path / "c")
    b.complete("optimize", "x")
    with pytest.raises(SchemaViolation, match="replay"):
        b.complete("optimize", "y")


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError, match="알 수 없는 모드"):
        MockLLM("wishful")


def test_deterministic_across_instances():
    a = MockLLM("mutate", seed=9, feature_names=FEATS)
    b = MockLLM("mutate", seed=9, feature_names=FEATS)
    assert ([a.complete("optimize", "x") for _ in range(5)]
            == [b.complete("optimize", "x") for _ in range(5)])
