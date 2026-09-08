"""The re-run pre-registration — do the document and the code diverge
(principle 2)?

The same judgement in two places diverges. A pre-registration is especially
dangerous because it is read as **a document** and executed as **code** —
fixing only the document and not the code makes "we did it as pre-registered"
false.

⚠️ 2026-09-08 (D-146): the document and `PREREG` were translated together,
so the asserted strings became English. Nothing was deleted — the Korean
original of both is at commit `ee53b4d`.
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
    assert "nailed down **before** the run" in body
    assert ("Setting the criteria after seeing the results is contamination"
            in body)


def test_numbers_match_between_doc_and_code(rr):
    """Do the scale and the cost cap match between the document and the
    code?"""
    body = DOC.read_text()
    p = rr.PREREG
    for text in (f"{p['n_seeds']} seeds", f"{p['rounds']} rounds",
                 f"{p['n_rules_per_round']} proposals"):
        assert text in body, f"{text!r} is not in the document"
    assert p["feature_detail"] == "full"
    assert p["split_kind"] in body


def test_expected_result_is_written_down(rr):
    """★ The expected result has to be written in advance to block
    rationalising afterwards."""
    assert "indistinguishable" in rr.PREREG["expected"]
    assert "indistinguishable" in DOC.read_text()
    assert "not a failure" in rr.PREREG["expected"]


def test_primary_metric_is_per_shape(rr):
    """Per shape is the main metric — a sign test over 6 runs has a p lower
    bound of 0.031 and is weak."""
    assert "shape" in rr.PREREG["primary_metric"]
    assert "median" in rr.PREREG["primary_metric"]
    assert "0.031" in rr.PREREG["secondary_metric"]


def test_failure_policy_forbids_dropping_a_run(rr):
    """★ Dropping a failed run from the results is the most dangerous thing
    (D-50)."""
    one = rr.PREREG["on_gate_failure"]["1 failure"]
    assert "Do not take that run out" in one
    assert "2 or more" in rr.PREREG["on_gate_failure"]
    assert "0.03" in " ".join(rr.PREREG["on_gate_failure"])


def test_partial_completion_forbids_cherry_picking(rr):
    assert "not cherry-picked" in rr.PREREG["on_partial"]
    assert "the design was" in rr.PREREG["on_partial"]


def test_ab_comparison_is_explicitly_excluded(rr):
    joined = " ".join(rr.PREREG["not_doing"])
    assert "A/B" in joined
    assert "a comparison with a deleted value" in joined
    # ★ The reason for excluding it has to be "because the conclusion is
    #   already in". "Because the fitter fails" would be changing the
    #   criterion to fit the result (principle 18).
    assert "the conclusion is already in" in joined


def test_budget_is_bounded(rr):
    for k in ("max_calls", "max_input_tokens", "max_output_tokens"):
        assert rr.BUDGET[k] > 0
    # It has to be larger than the existing 6 runs' measurement (155 calls per
    # run) and must not be absurdly large
    assert 900 < rr.BUDGET["max_calls"] < 3000
