"""F2 사전 등록 — 문서와 코드가 갈리지 않는가 (원칙 2).

사전 등록은 **문서**로 읽히고 **코드**로 실행된다. 문서만 고치고 코드를
안 고치면 "사전 등록대로 했다" 가 거짓이 된다.
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
    # ★ "실행 직전이 더 위험하다" 는 논지가 남아 있어야 한다
    assert "실행 직전이 오히려 더 위험하다" in body


def test_numbers_match_between_doc_and_code(pre):
    body = DOC.read_text()
    for text in (f"{pre['start_library']}개", f"고정 {pre['areas']}개",
                 # ★ 사전 등록은 옛 이름으로 쓰였다. 문서도 코드도 그때의
                 # 기록이므로 둘 다 안 고친다 (D-93, 문서 규칙 2).
                 f"Architect {pre['n_architect']}회",
                 f"{pre['n_seeds']}시드", f"{pre['rounds']}라운드"):
        assert text in body, f"문서에 {text!r} 가 없다"


def test_expected_result_is_written_down(pre):
    assert "실패가 아니다" in pre["expected"]
    assert "모른다" in pre["expected"]
    assert "실패가 아니다" in DOC.read_text()


def test_rediscovery_is_explicitly_not_a_criterion(pre):
    """★ 5개를 줬으니 재발견이 줄어드는 게 당연하다."""
    assert "재발견" in pre["not_a_criterion"]
    assert "판정 기준이 **아닌** 것" in DOC.read_text()


def test_two_variables_are_acknowledged(pre):
    """D-31 의 예외임을 **명시**한다 — 조용히 넘어가지 않는다."""
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
    """★ "멈춘다" 만으로는 부족하다 — 무엇을 볼지 적어 둔다 (원칙 8)."""
    action = pre["on_failure"]["채택 절반 미만"]
    assert "거부 사유 분포" in action
    assert "원칙 8" in action
    body = DOC.read_text()
    assert "멈춘 뒤의 행동" in body
    assert "기준을 바꾸는 것이 아니라" in body


def test_threshold_rationale_is_not_calibrated_to_f1(pre):
    """임계값이 F1 실측을 보고 조인 것이 아니라는 근거 (D-50)."""
    assert "최소 요건" in pre["threshold_rationale"]
    assert "정상 동작" in pre["threshold_rationale"]
