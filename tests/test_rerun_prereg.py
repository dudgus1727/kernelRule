"""재실행 실험 계획서 — 문서와 코드가 달라지지 않는가 (원칙 2).

같은 판정이 두 곳에 있으면 달라진다. 실험 계획서는 **문서**로 읽히고
**코드**로 실행되므로 특히 위험하다 — 문서만 고치고 코드를 안 고치면
"실험 계획서대로 했다" 가 거짓이 된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

DOC = (Path(__file__).resolve().parents[1]
       / "docs" / "artifacts" / "rerun-preregistration.md")


@pytest.fixture(scope="module")
def rr():
    import rerun
    return rerun


def test_doc_exists_and_says_it_is_preregistered():
    body = DOC.read_text()
    assert "실행 **전**에 박는다" in body
    assert "결과를 보고 기준을 정하면 오염이다" in body


def test_numbers_match_between_doc_and_code(rr):
    """규모·비용 상한이 문서와 코드에서 같은가."""
    body = DOC.read_text()
    p = rr.PREREG
    for text in (f"{p['n_seeds']}시드", f"{p['rounds']}라운드",
                 f"{p['n_rules_per_round']}제안"):
        assert text in body, f"문서에 {text!r} 가 없다"
    assert p["feature_detail"] == "full"
    assert p["split_kind"] in body


def test_expected_result_is_written_down(rr):
    """★ 예상 결과를 미리 적어야 사후 합리화를 막는다."""
    assert "구분 불가" in rr.PREREG["expected"]
    assert "구분 불가" in DOC.read_text()
    assert "실패가 아니다" in rr.PREREG["expected"]


def test_primary_metric_is_per_shape(rr):
    """형상별이 주 지표다 — 실행 6개 부호검정은 p 하한 0.031 이라 약하다."""
    assert "형상" in rr.PREREG["primary_metric"]
    assert "중앙값" in rr.PREREG["primary_metric"]
    assert "0.031" in rr.PREREG["secondary_metric"]


def test_failure_policy_forbids_dropping_a_run(rr):
    """★ 실패한 실행을 결과에서 빼는 것이 가장 위험하다 (D-50)."""
    one = rr.PREREG["on_gate_failure"]["1건"]
    assert "빼지 말" in one
    assert "2건 이상" in rr.PREREG["on_gate_failure"]
    assert "0.03" in " ".join(rr.PREREG["on_gate_failure"])


def test_partial_completion_forbids_cherry_picking(rr):
    assert "고르지 않는다" in rr.PREREG["on_partial"]
    assert "설계는" in rr.PREREG["on_partial"]


def test_ab_comparison_is_explicitly_excluded(rr):
    joined = " ".join(rr.PREREG["not_doing"])
    assert "A/B" in joined
    assert "삭제한 값과의 비교" in joined
    # ★ 빼는 이유가 "결론이 났으니까" 여야 한다. "적합기가 실패하니까" 면
    #   그것은 기준을 결과에 맞춰 바꾼 것이다 (원칙 18).
    assert "이미 결론이 났다" in joined


def test_budget_is_bounded(rr):
    for k in ("max_calls", "max_input_tokens", "max_output_tokens"):
        assert rr.BUDGET[k] > 0
    # 기존 6실행 실측(호출 155/실행)보다는 커야 하고 터무니없이 크면 안 된다
    assert 900 < rr.BUDGET["max_calls"] < 3000
