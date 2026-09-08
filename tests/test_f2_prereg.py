"""The F2 pre-registration — do the document and the code diverge
(principle 2)?

A pre-registration is read as **a document** and executed as **code**. Fixing
only the document and not the code makes "we did it as pre-registered" false.

⚠️ 2026-09-08 (D-146): the document and `F2_PREREG` were translated
together, so the asserted strings became English. Nothing was deleted — the
Korean original of both is at commit `ee53b4d`.
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
    assert "nailed down **before** the run" in body
    assert "Written with 0 LLM calls made" in body
    # ★ the argument "just before the run is more dangerous" has to remain
    assert "Just before the run is the more dangerous moment" in body


def test_numbers_match_between_doc_and_code(pre):
    body = DOC.read_text()
    for text in (f"the {pre['start_library']} public facts",
                 f"a fixed {pre['areas']}",
                 # ★ The pre-registration was written with the old name. Both
                 # the document and the code are a record of that time, so
                 # neither is changed (D-93, documentation rule 2).
                 f"Architect {pre['n_architect']} calls",
                 f"{pre['n_seeds']} seeds x {pre['rounds']} rounds"):
        assert text in body, f"{text!r} is not in the document"


def test_expected_result_is_written_down(pre):
    assert "not a failure" in pre["expected"]
    assert "is not known" in pre["expected"]
    assert "not a failure" in DOC.read_text()


def test_rediscovery_is_explicitly_not_a_criterion(pre):
    """★ Five were given, so of course rediscoveries go down."""
    assert "rediscoveries" in pre["not_a_criterion"]
    assert "What is **not** a criterion" in DOC.read_text()


def test_two_variables_are_acknowledged(pre):
    """It **states** that this is an exception to D-31 — it is not passed over
    silently."""
    assert "are not separated" in pre["two_variables"]
    assert "D-31" in pre["two_variables"]
    assert "cannot be separated" in pre["two_variables"]


def test_discrimination_limit_is_stated(pre):
    assert "0.0274" in pre["discrimination_note"]
    assert "indistinguishable" in pre["discrimination_note"]


def test_not_doing_covers_the_traps(pre):
    joined = " ".join(pre["not_doing"])
    for must in ("remaining 19", "recategorize", "F1 results", "model",
                 "prompt"):
        assert must in joined, must


def test_failure_policy_is_explicit(pre):
    for key in ("an area refused 3 times in a row", "acceptance under half",
                "all RuleWriter calls refused",
                "3 runs in a row with an empty archive"):
        assert key in pre["on_failure"], key


def test_stop_condition_says_what_to_do_after_stopping(pre):
    """★ "it stops" alone is not enough — what to look at is written down
    (principle 8)."""
    action = pre["on_failure"]["acceptance under half"]
    assert "refusal-reason distribution" in action
    assert "principle 8" in action
    body = DOC.read_text()
    assert "What to do after stopping" in body
    assert "does not change the criteria" in body


def test_threshold_rationale_is_not_calibrated_to_f1(pre):
    """The ground that the threshold was not tightened after seeing F1's
    measurement (D-50)."""
    assert "the minimum for the experiment to hold" in pre["threshold_rationale"]
    assert "normal behaviour" in pre["threshold_rationale"]
