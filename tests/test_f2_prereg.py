"""The F2 pre-registration — do the document and the code diverge
(principle 2)?

A pre-registration is read as **a document** and executed as **code**. Fixing
only the document and not the code makes "we did it as pre-registered" false.

⚠️ 2026-09-08 (D-146): **the asserted strings stay in Korean.** They are the
text of `docs/artifacts/f2-preregistration.md` and of the frozen `F2_PREREG`
that mirrors it, and `docs/` is not translated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

DOC = (Path(__file__).resolve().parents[1]
       / "docs" / "artifacts" / "f2-preregistration.md")


@pytest.fixture(scope="module")
def pre():
    import f1_pipeline
    return f1_pipeline.F2_PREREG


def test_doc_says_it_was_written_before_any_llm_call():
    body = DOC.read_text()
    assert "실행 **전**에 박는다" in body
    assert "LLM 호출 0회 상태에서 작성" in body
    # ★ the argument "just before the run is more dangerous" has to remain
    assert "실행 직전이 오히려 더 위험하다" in body


def test_numbers_match_between_doc_and_code(pre):
    body = DOC.read_text()
    for text in (f"{pre['start_library']}개", f"고정 {pre['areas']}개",
                 # ★ The pre-registration was written with the old name. Both
                 # the document and the code are a record of that time, so
                 # neither is changed (D-93, documentation rule 2).
                 f"Architect {pre['n_architect']}회",
                 f"{pre['n_seeds']}시드", f"{pre['rounds']}라운드"):
        assert text in body, f"{text!r} is not in the document"


def test_expected_result_is_written_down(pre):
    assert "실패가 아니다" in pre["expected"]
    assert "모른다" in pre["expected"]
    assert "실패가 아니다" in DOC.read_text()


def test_rediscovery_is_explicitly_not_a_criterion(pre):
    """★ Five were given, so of course rediscoveries go down."""
    assert "재발견" in pre["not_a_criterion"]
    assert "판정 기준이 **아닌** 것" in DOC.read_text()


def test_two_variables_are_acknowledged(pre):
    """It **states** that this is an exception to D-31 — it is not passed over
    silently."""
    assert "분리하지 않는다" in pre["two_variables"]
    assert "D-31" in pre["two_variables"]
    assert "못 가른다" in pre["two_variables"]


def test_discrimination_limit_is_stated(pre):
    assert "0.0274" in pre["discrimination_note"]
    assert "구분 불가" in pre["discrimination_note"]


def test_not_doing_covers_the_traps(pre):
    joined = " ".join(pre["not_doing"])
    for must in ("19개", "recategorize", "F1 결과", "모델", "프롬프트"):
        assert must in joined, must


def test_failure_policy_is_explicit(pre):
    for key in ("영역 3회 연속 거부", "채택 절반 미만",
                "RuleWriter 전부 거부", "3실행 연속 빈 아카이브"):
        assert key in pre["on_failure"], key


def test_stop_condition_says_what_to_do_after_stopping(pre):
    """★ "it stops" alone is not enough — what to look at is written down
    (principle 8)."""
    action = pre["on_failure"]["채택 절반 미만"]
    assert "거부 사유 분포" in action
    assert "원칙 8" in action
    body = DOC.read_text()
    assert "멈춘 뒤의 행동" in body
    assert "기준을 바꾸는 것이 아니라" in body


def test_threshold_rationale_is_not_calibrated_to_f1(pre):
    """The ground that the threshold was not tightened after seeing F1's
    measurement (D-50)."""
    assert "최소 요건" in pre["threshold_rationale"]
    assert "정상 동작" in pre["threshold_rationale"]
