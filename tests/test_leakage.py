"""★ The answer-leak defences (§3, §22.5, §30.7).

If this module is skipped, `conftest.py`'s watchdog fails the session
(§26.3).
"""
from __future__ import annotations

import dataclasses
import inspect
import warnings

import numpy as np
import pytest
from toy import constant_score_order

from kernelrule.core.scoring import evaluate
from kernelrule.core.types import CandidateSet, Config, Hardware, Problem

ANSWERISH = ("time", "ms", "difficulty", "cublas", "tflops", "regret",
             "outlier", "distinct", "peak", "elapsed")

#: Names that look like answers but are **measurement-condition
#: constants**. They are fixed before measurement and have a single value
#: across the table. The test below checks "is it really constant", so this
#: is not a plain whitelist — a mismatch is caught.
SAFE_CONDITION_COLS = {"peak_tflops_used", "locked_mhz", "ridge_point",
                       "ridge_point_spec", "build_seconds"}


# ---------------------------------------------------------------------------
# 1. At the data-structure level — the objects a rule can touch have no
#    times (§3.3)
# ---------------------------------------------------------------------------
def test_rule_facing_types_have_no_time_field():
    """Nowhere in `Problem` / `Config` / `CandidateSet` is there a measured
    time."""
    for cls in (Problem, Config, CandidateSet):
        names = [f.name for f in dataclasses.fields(cls)]
        bad = [n for n in names
               if any(k in n.lower() for k in ("time", "cublas", "difficulty",
                                               "regret", "tflops"))]
        assert not bad, f"answer-like fields on {cls.__name__}: {bad}"


def test_hardware_has_no_measurement():
    names = [f.name for f in dataclasses.fields(Hardware)]
    assert not any("time" in n or "difficulty" in n for n in names)


def test_candidate_set_cannot_reach_times(synth_table):
    """★ The times cannot be reached through `CandidateSet`.

    Writing `sorted(..., key=lambda c: (score, time))` requires an attribute
    that does not exist.
    """
    p = synth_table.shapes()[0]
    cand = synth_table.candidates(p)
    for attr in ("time_ms", "time", "times", "best_time", "difficulty"):
        assert not hasattr(cand, attr), f"CandidateSet.{attr} exists"


def test_order_fn_signature_excludes_the_table():
    """The scorer passes a rule only `(Problem, CandidateSet)`."""
    src = inspect.getsource(evaluate)
    assert "order_fn(p, cand)" in src, (
        "the order_fn call arguments changed — check that no table or time "
        "is being passed")


def test_times_of_is_read_only(synth_table):
    """The time array the scorer receives is read-only too. It blocks
    touching it by accident."""
    p = synth_table.shapes()[0]
    t = synth_table.times_of(p)
    with pytest.raises(ValueError):
        t[0] = 0.0


def test_perftable_has_no_best_config():
    """★ It does not provide "the optimal config per shape".

    In this table, 29 of 66 shapes have an **exact tie** at the best time,
    with up to 84 tied. Then "the optimal config" is a function of the
    tie-break rule, not a physical fact. What can be defined is only
    `best_time` and `answer_mask`.
    """
    from kernelrule.core.table import PerfTable
    assert not hasattr(PerfTable, "best_config")
    assert not hasattr(PerfTable, "argbest")


# ---------------------------------------------------------------------------
# 2. The tie-break does not look at the answer (§30.7)
# ---------------------------------------------------------------------------
def test_constant_score_gives_random_performance(synth_table):
    """★ With every config scored the same, regret must equal that of a
    random pick.

    If it comes out better, the tie-break is looking at the answer. It is a
    bug that really occurred in kernelTab's baseline experiment (§30.7).
    """
    ev = evaluate(constant_score_order, synth_table, ks=(1,), label="constant")
    got = ev.at(1)

    rng = np.random.default_rng(0)
    draws = []
    for _ in range(24):
        def rnd(p, cand, rng=rng):
            return rng.permutation(cand.n)
        draws.append(evaluate(rnd, synth_table, ks=(1,)).at(1))
    lo, hi = float(np.min(draws)), float(np.max(draws))
    assert lo * 0.7 <= got <= hi * 1.3, (
        f"the constant-score regret {got:.3f} is outside the random range "
        f"[{lo:.3f}, {hi:.3f}] — the tie-break is looking at the answer")


def test_tiebreak_is_independent_of_row_order(synth_table):
    """The tie-break does not depend on **the table's row order**.

    `groupby.idxmin()` does depend on row order. That really did make the
    axis distribution of "the optimal config per shape" differ per
    procedure.
    """
    from kernelrule.core.types import make_tiebreak

    p = synth_table.shapes()[0]
    c = synth_table.candidates(p)
    perm = np.random.default_rng(3).permutation(c.n)
    tb2 = make_tiebreak(c.kernel_id[perm], c.split_k[perm],
                        c.split_k_mode[perm])
    # The same config keeps the same relative rank under shuffling
    assert np.array_equal(np.argsort(c.tiebreak[perm]), np.argsort(tb2))


def test_order_by_rejects_nonfinite_scores(synth_table):
    """It does not push nan to the back and carry on silently. The rule is
    broken."""
    p = synth_table.shapes()[0]
    c = synth_table.candidates(p)
    s = np.zeros(c.n)
    s[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        c.order_by(s)


# ---------------------------------------------------------------------------
# 3. ★ The null preset — the only automatic leak detector (§22.5)
# ---------------------------------------------------------------------------
def test_null_preset_gives_no_improvement(null_table):
    """★ On a table where the config is unrelated to performance, **no rule
    can fall well below 1.0.**

    If one does, the answer is leaking somewhere. This is the most important
    test.
    """
    orders = {
        "constant": constant_score_order,
        "by_tiebreak": lambda p, c: c.order_by(np.zeros(c.n)),
        "by_splitk": lambda p, c: c.order_by(c.split_k.astype(float)),
        "random": lambda p, c: np.random.default_rng(5).permutation(c.n),
    }
    for name, fn in orders.items():
        ev = evaluate(fn, null_table, ks=(1,), label=name)
        assert ev.at(1) >= 0.999, (
            f"{name}: regret {ev.at(1):.4f} < 1.0 on the null table — the "
            f"answer is leaking")


def test_null_preset_difficulty_is_near_one(null_table):
    """On the null table the difficulty must be near 1 (§22.5)."""
    d = np.array([s.difficulty for s in null_table.all_stats()])
    assert d.max() < 1.15, (
        f"the null table's difficulty is {d.max():.3f} — structure remains")


def test_null_preset_best_equals_typical(null_table):
    """On the null table the gap between the best and the median is noise
    alone."""
    for s in null_table.all_stats():
        assert s.difficulty - 1.0 < 4.0 * s.noise_floor + 0.1


# ---------------------------------------------------------------------------
# 4. The loader contract (§3.2)
# ---------------------------------------------------------------------------
def test_ranking_loader_has_no_answers(real_bundle_path):
    from kerneltab.core.bundle import load_bundle
    from kerneltab.core.table import ANSWER_COLS, assert_no_answers

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = load_bundle(real_bundle_path).ranking(ok_only=False,
                                                  unknown_columns="ignore")
    assert_no_answers(X)
    assert not (set(X.columns) & set(ANSWER_COLS))


def test_perftable_feature_frame_has_no_answers(synth_table):
    """`PerfTable.frame_for` gives only features — it is the feature
    matrix's input."""
    from kerneltab.core.table import ANSWER_COLS, OUTCOME_COLS

    p = synth_table.shapes()[0]
    cols = set(synth_table.frame_for(p).columns)
    assert not (cols & set(ANSWER_COLS))
    assert not (cols & set(OUTCOME_COLS))
    suspect = sorted(c for c in cols
                     if any(k in c.lower() for k in ANSWERISH))
    unexplained = [c for c in suspect if c not in SAFE_CONDITION_COLS]
    assert not unexplained, (
        f"there is an answer-like column in the rule input: {unexplained}. "
        f"If it is a measurement-condition constant, put it in "
        f"SAFE_CONDITION_COLS and write down why.")
    # It checks that what is claimed to be a condition constant **really is
    # constant**.
    df = synth_table.frame_for(p)
    for c in suspect:
        assert df[c].nunique(dropna=False) == 1, (
            f"{c!r} is classified as a measurement-condition constant yet "
            f"takes {df[c].nunique()} values within a shape — it may be "
            f"derived from the answer.")


def test_env_hash_is_required(real_bundle_path):
    """`env_hash` is not a join key but an isolation boundary. It has no
    default (§3.4)."""
    from kernelrule.core.table import PerfTable

    sig = inspect.signature(PerfTable.from_bundle)
    assert sig.parameters["env_hash"].default is inspect.Parameter.empty
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception, match="env_hash"):
            PerfTable.from_bundle(real_bundle_path, env_hash="deadbeef")


# ---------------------------------------------------------------------------
# Does the frame a feature function receives contain no answers (the 5th
# layer of §3)
# ---------------------------------------------------------------------------
# `FeatureMatrix` passes `table.frame_for(p)` to the feature function as it
# is. If that frame contains the answers, **a feature can see the measured
# times.**
#
# Today's 24 were written by a human so they do not use them, but **the
# features the FeatureWriter builds are different** — that is code we have
# not seen. It has to be blocked structurally.

@pytest.mark.needs_bundle
def test_feature_frame_has_no_answer_column(real_bundle_path):
    """★ `frame_for` contains not one cell of `ANSWER_COLS`."""
    from kerneltab.core.table import ANSWER_COLS

    from kernelrule.core.table import PerfTable
    t = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                              ok_only=False)
    cols = set(t.frame_for(t.shapes()[0]).columns)
    leaked = cols & set(ANSWER_COLS)
    assert not leaked, (
        f"there is an answer column in the frame a feature function "
        f"receives: {sorted(leaked)}. Check that `PerfTable.from_bundle` "
        f"uses `bundle.ranking()` (§3)")


def test_generated_feature_touching_answers_is_rejected():
    """★ Does the checker catch a generated feature that references an
    answer column (§11.4)?

    It is absent from the frame, so running it blows up anyway, but the
    reason must be caught **clearly at the AST stage** — blowing up as a
    runtime exception reads as "the model produced a bad feature" (D-49).
    """
    from kernelrule.features.generated import FeatureRejected, check_feature_code

    for col in ("time_ms", "cublas_ms", "difficulty", "tflops"):
        code = (f"def peek(p, hw, cfg) -> float:\n"
                f"    return float(cfg.{col})\n")
        with pytest.raises(FeatureRejected):
            check_feature_code(code, known=frozenset())
