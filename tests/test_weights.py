"""Weight optimisation (§29). Separating structure from parameters."""
from __future__ import annotations

import numpy as np
import pytest
from toy import make_table

from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.scoring import evaluate
from kernelrule.core.splits import Split, SplitError, SplitSet, by_predicate
from kernelrule.core.weights import FitError, fit_weights, make_order_fn
from kernelrule.features import FeatureRegistry, feature

#: Builds a table whose generating coefficients are **known**.
#: time = exp(w_true . f).
W_TRUE = np.array([2.0, 0.5, 1.5])


@pytest.fixture(scope="module")
def known():
    """A table built from known coefficients + a rule of the same
    structure (§26.2)."""
    rng = np.random.default_rng(0)
    n_cfg, n_shape = 24, 12

    # ★ The feature values differ per shape. If they were the same, every
    #   shape would share one optimal config and regret@1 would become a
    #   step with only 24 values, unlike the real table.
    times, cols = {}, {"f0": [], "f1": [], "f2": []}
    for s in range(n_shape):
        F = rng.uniform(0.0, 1.0, size=(n_cfg, 3))
        base = 0.5 + 0.1 * s
        times[(128 * (s + 1), 4096, 4096)] = list(base * np.exp(F @ W_TRUE))
        for i in range(3):
            cols[f"f{i}"].extend(F[:, i].tolist())
    t = make_table(times, feature_cols=cols)

    r = FeatureRegistry("known")
    for i in range(3):
        def mk(i=i):
            @feature(registry=r, vec=lambda df, hw, p, i=i:
                     df[f"f{i}"].to_numpy(np.float64))
            def _f(p, hw, cfg, i=i) -> float:
                return 0.0
            _f.__name__ = f"f{i}"
            return _f
        # It has to be re-registered after renaming, so it is built
        # directly
        from kernelrule.features import Feature
        r.add(Feature(name=f"f{i}", fn=lambda p, hw, cfg: 0.0,
                      unit="dimensionless", expected_range=(0.0, 1.0),
                      direction="higher_is_worse",
                      vec=(lambda df, hw, p, i=i: df[f"f{i}"].to_numpy(np.float64)),
                      code_hash=f"h{i}"))
    m = FeatureMatrix(t, r)

    def score(f, p, hw, w):
        return f.f0 * w[0] + f.f1 * w[1] + f.f2 * w[2]

    return t, m, score


def _all_train(table) -> Split:
    return Split("train", tuple(table.shapes()), name="all")


def test_weight_fit_recovers_known_optimum(known):
    """★ Given the same structure as the known coefficients, the optimiser
    must recover that optimum.

    regret looks only at **order**, so `w` is identified only up to scale.
    The verdict is whether the direction (the normalised vector) matches and
    regret reaches 1.0.
    """
    t, m, score = known
    w0 = np.array([1.0, 1.0, 1.0])       # deliberately wrong initial values
    fr = fit_weights(score, m, t, _all_train(t), w0, max_evals=400)
    assert fr.fit_regret == pytest.approx(1.0, abs=1e-9), fr
    cos = float(fr.w @ W_TRUE / (np.linalg.norm(fr.w) * np.linalg.norm(W_TRUE)))
    assert cos > 0.99, (
        f"the recovered direction differs: {fr.w} vs {W_TRUE} "
        f"(cos={cos:.4f})")


def test_bad_initial_weights_would_have_lost_a_good_structure(known):
    """★ The rationale for §29.3. Scoring without optimising the weights
    throws away good structures."""
    t, m, score = known
    w_bad = np.array([1.0, 4.0, -3.0])
    before = evaluate(make_order_fn(score, m, w_bad), t, ks=(1,)).at(1)
    fr = fit_weights(score, m, t, _all_train(t), w_bad, max_evals=400)
    assert before > 1.05, (
        "the initial values are not bad enough for this test to hold")
    assert fr.fit_regret < before
    assert fr.fit_regret == pytest.approx(1.0, abs=1e-9)


def test_fit_never_worse_than_initial(known):
    """It is a step function, so Nelder-Mead can stop somewhere worse than
    the initial values.

    ★ `objective="regret"` is **stated explicitly** (D-122). This invariant
    is a property of "the objective being fitted", and fitting with the rank
    loss can make regret worse (`test_rank_fit_may_worsen_regret` below).
    """
    t, m, score = known
    w0 = W_TRUE.copy()
    fr = fit_weights(score, m, t, _all_train(t), w0, max_evals=30,
                     objective="regret")
    base = evaluate(make_order_fn(score, m, w0), t, ks=(1,)).at(1)
    assert fr.fit_regret <= base + 1e-12


def test_rank_fit_never_worse_than_initial_in_rank_loss(known):
    """★ The same invariant, held **on the rank-loss side** (D-122).

    While polish did not know the objective this check meant nothing — polish
    did nothing, so it held automatically (principle 38).
    """
    from kernelrule.core.weights import _Problem

    t, m, score = known
    w0 = np.array([1.0, 1.0, 1.0])
    fr = fit_weights(score, m, t, _all_train(t), w0, max_evals=120,
                     objective="rank", warn_invariants=False)
    pr = _Problem(m, t, _all_train(t).shapes, 1)
    pr.build_pairs(t, 100)
    assert pr.rank_loss(score, fr.w) <= pr.rank_loss(score, w0) + 1e-12


def test_rank_fit_may_worsen_regret(known):
    """★ Fitting with the rank loss **can make regret worse** (D-122).

    Starting from the true coefficients and lowering the rank loss raises
    regret above 1.0. It is the price of "score by regret, train by the rank
    loss", and it points the same way as the wall D-118 measured. This is
    held **by a test, not by documentation**.
    """
    t, m, score = known
    w0 = W_TRUE.copy()
    fr = fit_weights(score, m, t, _all_train(t), w0, max_evals=120,
                     objective="rank", warn_invariants=False)
    assert fr.fit_regret >= 1.0 - 1e-12


def test_weight_fit_uses_train_split_only(known):
    """★ There is no path by which the validation or final split enters
    the objective (§29.7)."""
    t, m, score = known
    shapes = t.shapes()
    for role in ("val", "test"):
        bad = Split(role, tuple(shapes))
        with pytest.raises(SplitError,
                           match="Only the training split is accepted"):
            fit_weights(score, m, t, bad, [1.0, 1.0, 1.0])


def test_weight_fit_refuses_a_bare_shape_list(known):
    """Not stating the split is an error. Which split it is cannot be known
    (§26.4)."""
    t, m, score = known
    with pytest.raises(SplitError, match="fit_weights takes a Split"):
        fit_weights(score, m, t, t.shapes(), [1.0, 1.0, 1.0])


def test_val_split_must_be_val(known):
    t, m, score = known
    tr = _all_train(t)
    with pytest.raises(SplitError, match="It must be 'val'"):
        fit_weights(score, m, t, tr, [1.0, 1.0, 1.0], val_split=tr)


def test_gap_is_recorded(known):
    """The training-validation gap is recorded every round (§29.4)."""
    t, m, score = known
    shapes = t.shapes()
    tr = Split("train", tuple(shapes[:6]))
    va = Split("val", tuple(shapes[6:]))
    fr = fit_weights(score, m, t, tr, [1.0, 1.0, 1.0], val_split=va,
                     max_evals=300)
    assert np.isfinite(fr.val_regret)
    assert np.isfinite(fr.gap)


def test_sensitivity_flags_dead_terms(known):
    """Terms that converge near 0 or are insensitive are candidates for
    feature cleanup (§29.6)."""
    t, m, score = known

    def score4(f, p, hw, w):
        # w[3] is used nowhere -> it must be completely insensitive
        return f.f0 * w[0] + f.f1 * w[1] + f.f2 * w[2] + 0.0 * w[3]

    fr = fit_weights(score4, m, t, _all_train(t), [1.0, 1.0, 1.0, 1.0],
                     max_evals=300)
    assert fr.sensitivity[3] == 0.0
    assert 3 in fr.dead_terms


def test_structure_that_never_scores_is_rejected(known):
    """Producing no valid score at any weights means **the structure is
    rejected** (§26.4)."""
    t, m, score = known

    def broken(f, p, hw, w):
        return np.full(len(f.f0), np.nan)

    with pytest.raises(FitError, match="The structure is rejected"):
        fit_weights(broken, m, t, _all_train(t), [1.0], max_evals=20)


def test_make_order_fn_uses_the_same_score_fn(known):
    """★ Training and deployment use the same `score()` (the §8.1
    replacement)."""
    t, m, score = known
    fr = fit_weights(score, m, t, _all_train(t), [1.0, 1.0, 1.0], max_evals=400)
    ev = evaluate(make_order_fn(score, m, fr.w), t, ks=(1,))
    assert ev.at(1) == pytest.approx(fr.fit_regret, abs=1e-12)


def test_split_refuses_empty_and_overlap():
    """An empty or overlapping split is an error (§26.4, §10)."""
    from kernelrule.core.types import Problem
    a = Problem(1024, 4096, 4096)
    b = Problem(2048, 4096, 4096)
    with pytest.raises(SplitError, match="is empty"):
        Split("train", ())
    with pytest.raises(SplitError, match="share"):
        SplitSet(train=Split("train", (a, b)), val=Split("val", (b,)))
    with pytest.raises(SplitError, match="left one side empty"):
        by_predicate([a, b], lambda p: False, name="none")


# ---------------------------------------------------------------- D-54/D-55
# The fitter must report "did I do nothing" itself. 13 of 24 stayed at the
# initial values and nobody knew — these tests block that silence.

def test_fitted_rule_reports_that_it_did_not_move():
    from kernelrule.core.weights import FittedRule

    fr = FittedRule(w=np.array([1.0, 2.0]), w0=np.array([1.0, 2.0]),
                    fit_regret=1.1, n_evals=305, n_infeasible=0,
                    sensitivity=np.zeros(2), seconds=1.0)
    assert not fr.moved
    assert any("did not move" in m for m in fr.invariants())


def test_fitted_rule_flags_dominance_and_negative_weights():
    """★ Dominance is caught by **effective contribution**, not by
    absolute magnitude (D-70)."""
    from kernelrule.core.weights import FittedRule

    fr = FittedRule(w=np.array([500.0, -3.0, 1.0]), w0=np.array([1.0, 2.0, 1.0]),
                    fit_regret=1.1, n_evals=10, n_infeasible=0,
                    sensitivity=np.zeros(3), seconds=1.0,
                    contrib=np.array([1000.0, 1.0, 1.0]))
    assert fr.moved
    msgs = " ".join(fr.invariants())
    assert "overwhelms" in msgs and "negative weights" in msgs

    # With even contributions, no warning however large |w| is — magnitude is
    # harmless
    ok = FittedRule(w=np.array([5e6, 4e6, 6e6]), w0=np.ones(3),
                    fit_regret=1.1, n_evals=10, n_infeasible=0,
                    sensitivity=np.zeros(3), seconds=1.0,
                    contrib=np.array([1.0, 0.9, 1.1]))
    assert not any("overwhelms" in m for m in ok.invariants())


def test_dead_term_is_flagged():
    """Zero effective contribution = a term that takes no part in the
    ranking (absolute rule 2)."""
    from kernelrule.core.weights import FittedRule

    fr = FittedRule(w=np.ones(3), w0=np.ones(3), fit_regret=1.1, n_evals=10,
                    n_infeasible=0, sensitivity=np.zeros(3), seconds=1.0,
                    contrib=np.array([1.0, 0.0, 1.2]))
    assert any("zero effective contribution" in m for m in fr.invariants())


def test_fit_weights_warns_about_its_own_invariants(known):
    """An odd fit **does not pass silently** (D-54).

    ★ This used to be checked through "the evaluation cap was reached".
    That warning was **always up**, so the test passed while holding
    nothing (D-76). It is now checked through a **dead term** — `w[2]` does
    not enter the score, so its effective contribution is 0, and that is a
    genuine anomaly signal.
    """
    import warnings

    from kernelrule.core.weights import FitWarning

    t, m, _score = known

    def dead_term(f, p, hw, w):
        return f.f0 * w[0] + f.f1 * w[1] + f.f2 * 0.0 * w[2]

    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        fit_weights(dead_term, m, t, _all_train(t), [1.0, 1.0, 1.0],
                    max_evals=200)
    msgs = [str(w.message) for w in got if issubclass(w.category, FitWarning)]
    assert any("zero effective contribution" in x for x in msgs), \
        f"the dead term was not reported: {msgs}"


def test_polish_never_worsens_training_regret(known):
    """The coordinate polish must **improve or match** the training regret
    (D-55).

    The acceptance condition is `v < base`, so it is structurally so. If this
    check breaks, polish is looking at something other than training
    (§29.7).

    ★ `objective="regret"` is stated explicitly (D-122) — called with the
    default (`rank`), this check passes **because polish does not look at
    regret**.
    """
    t, m, score = known
    tr = _all_train(t)
    a = fit_weights(score, m, t, tr, [1.0, 1.0, 1.0], max_evals=120,
                    objective="regret", warn_invariants=False, polish=False)
    b = fit_weights(score, m, t, tr, [1.0, 1.0, 1.0], max_evals=120,
                    objective="regret", warn_invariants=False, polish=True)
    assert b.fit_regret <= a.fit_regret + 1e-12


def test_polish_actually_runs_on_the_rank_path(known):
    """★ Does polish **do any work** on the rank-loss path (D-122)?

    `prob.regret` used to be nailed into `_polish`, so it was comparing a
    rank-loss reference value (0.24) against regret (1.2) — no step was ever
    accepted and it **spent 600 evaluations doing nothing.** Confirmed by
    measurement: the `w` of `polish=True` and `False` were exactly equal.

    This is a case of principle 38 — the test
    (`test_polish_never_worsens_training_regret`) passed while the thing
    under test was not running.
    """
    t, m, score = known
    tr = _all_train(t)
    a = fit_weights(score, m, t, tr, [1.0, 1.0, 1.0], max_evals=60,
                    objective="rank", polish=False, warn_invariants=False)
    b = fit_weights(score, m, t, tr, [1.0, 1.0, 1.0], max_evals=60,
                    objective="rank", polish=True, polish_budget=600,
                    warn_invariants=False)
    assert not np.array_equal(a.w, b.w), (
        "polish changed not one weight on the rank-loss path — the "
        "objective is not being handed over")


def test_polish_only_sees_the_training_split(known):
    """There is no path leaking the validation split into polish — the
    argument does not exist (§29.7)."""
    import inspect

    from kernelrule.core.weights import _polish

    names = set(inspect.signature(_polish).parameters)
    assert "val_split" not in names and "splits" not in names


def test_contributions_are_scale_invariant(known):
    """★ Effective contributions keep their ratios **even when every
    weight is scaled up**.

    The absolute magnitude (|w|/|w0|) changes by orders with feature scale,
    so the criterion becomes meaningless when the library changes — in F1
    (features [0,0.2]) max |w| was 770,164 and in the human 24 it was 45.1
    (D-70).
    """
    from kernelrule.core.weights import _contributions, _Problem

    t, m, score = known
    prob = _Problem(m, t, t.shapes(), 1)
    w = np.array([1.0, 2.0, 0.5])
    a = _contributions(prob, score, w)
    b = _contributions(prob, score, w * 1000.0)
    assert a is not None and b is not None
    # The absolute values are 1000x, the **ratios** are the same
    ra, rb = a / a.max(), b / b.max()
    assert np.allclose(ra, rb, atol=1e-9), (ra, rb)


def test_contribution_of_a_shape_constant_term_is_zero(known):
    """A shape-constant term does not change the ranking, so its
    contribution must be **exactly 0**.

    This catches the no-op term that absolute rule 2 of `_rules_common.md`
    speaks of.
    """
    from kernelrule.core.weights import _contributions, _Problem

    t, m, _ = known

    def score_with_noop(f, p, hw, w):
        # The w[1] term is a shape constant, so it cannot change the
        # ranking within that shape
        return f.f0 * w[0] + p.n_candidates * w[1]

    prob = _Problem(m, t, t.shapes(), 1)
    c = _contributions(prob, score_with_noop, np.array([1.0, 5.0]))
    assert c is not None
    assert c[0] > 0.0
    assert c[1] == 0.0, (
        f"the shape-constant term's contribution is not 0: {c[1]}")


def test_contributions_never_touch_the_answer():
    """★ There is no path that looks at the times (§3)."""
    import ast
    import inspect

    from kernelrule.core.weights import _contributions

    # ★ It looks at **the body only**, excluding the docstring. Writing "it
    #   does not call prob.regret" in the documentation would be caught by a
    #   string check (principle 14 — false positives created by the
    #   instrument).
    tree = ast.parse(inspect.getsource(_contributions).strip())
    fn = tree.body[0]
    body = fn.body[1:] if (isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)
                           ) else fn.body
    src = "\n".join(ast.unparse(n) for n in body)
    assert "prob.regret" not in src, (
        "it calls regret, which goes through the answer")
    assert "_times" in src and "_best" in src, (
        "the answer slots are not bound to `_` — without names they cannot "
        "be touched")
    # It calls only `score_fn`
    assert src.count("score_fn(") == 2

def test_cap_warning_ignores_polish_evals(known):
    """★ The cap warning is judged on the evaluations **before polish**.

    `n_evals` includes polish. Comparing that against `max_evals` makes the
    polish budget (600) always exceed the cap (300), so "the evaluation cap
    was reached — cut before convergence" fires on **every fit**. After
    polish became the default (D-56) this warning was always on and was
    therefore not a signal (principle 11).
    """
    import warnings as _w

    from kernelrule.core.weights import FitWarning

    t, m, score = known
    sp = _all_train(t)
    with _w.catch_warnings(record=True) as got:
        _w.simplefilter("always")
        fr = fit_weights(score, m, t, sp, [1.0, 1.0, 1.0], max_evals=3000,
                         polish=True, polish_budget=400)
    assert fr.n_fit_evals < fr.n_evals, (
        "polish did not run — the test is meaningless")
    assert fr.n_fit_evals < 3000, (
        "the fit alone reached the cap — raise the budget")
    caps = [str(x.message) for x in got
            if issubclass(x.category, FitWarning)
            and "evaluation cap" in str(x.message)]
    assert not caps, f"the polish evaluations raised the cap warning: {caps}"


def test_cap_warning_needs_actual_improvement_at_cutoff(known):
    """★ Exhausting the budget alone does not warn.

    ⚠️ `objective="regret"` is **stated explicitly** (2026-09-01). This test
    pins the restart schedule of the regret path (D-76), and when the
    default changed to `rank` it started measuring a different path.

    The restart schedule is **designed to spend all of** `max_evals`, so
    "the cap was reached" is always true. Something always true is not a
    watchdog (principle 11). The warning fires only **when it was still
    improving at the moment it was cut**.

    Starting from the generating coefficients `W_TRUE` there is nowhere
    better to go — the budget is spent but there is no improvement, so there
    must be no warning.
    """
    import warnings as _w

    from kernelrule.core.weights import FitWarning

    t, m, score = known
    with _w.catch_warnings(record=True) as got:
        _w.simplefilter("always")
        fr = fit_weights(score, m, t, _all_train(t), W_TRUE.tolist(),
                         max_evals=120, objective="regret", polish=False)
    assert fr.n_fit_evals >= 120, (
        "the budget was not spent — the test is meaningless")
    caps = [str(x.message) for x in got
            if issubclass(x.category, FitWarning)
            and "cap" in str(x.message)]
    assert not caps, (
        f"a warning fired on budget exhaustion alone: {caps}")


# ---------------------------------------------------------------------------
# D-101 — the rank loss
# ---------------------------------------------------------------------------
def test_objective_default_is_regret(known):
    """★ The default is `"regret"` again (D-128).

    ```
    ~09-01  regret   the path every result so far passed through
     09-01  rank     the experiment being run then was the rank loss (D-101)
    ★09-04  regret   the rank loss was concluded to be the wrong objective
                     (D-118 · D-121)
    ```

    `"rank"` **remains as a function** — used as a metric, and for
    reproducing old runs.
    """
    t, m, score = known
    a = fit_weights(score, m, t, _all_train(t), [1.0, 1.0, 1.0], max_evals=60)
    b = fit_weights(score, m, t, _all_train(t), [1.0, 1.0, 1.0], max_evals=60,
                    objective="regret")
    assert np.array_equal(a.w, b.w)
    r = fit_weights(score, m, t, _all_train(t), [1.0, 1.0, 1.0], max_evals=60,
                    objective="rank", warn_invariants=False)
    assert not np.array_equal(a.w, r.w), (
        "the objective branch does not run")


def test_loop_refuses_the_rank_objective():
    """★ The evolution path **refuses** the rank loss (D-128). It does not
    pass silently."""
    import pytest as _pytest

    from kernelrule.core.loop import LoopConfig, RoundLoop

    with _pytest.raises(ValueError, match="only regret"):
        RoundLoop(cfg=LoopConfig(run_id="x", objective="rank"),
                  table=None, matrix=None, splits=None, llm=None)


def test_explicit_regret_still_reproduces_the_old_path(known):
    """★ The reverse check — is `objective="regret"` still the old path?

    The default was changed, so this is where **the path that retraces old
    results** is kept alive. If this breaks, none of the numbers so far can
    be reproduced.
    """
    t, m, score = known
    a = fit_weights(score, m, t, _all_train(t), [1.0, 1.0, 1.0], max_evals=60,
                    objective="regret")
    b = fit_weights(score, m, t, _all_train(t), [1.0, 1.0, 1.0], max_evals=60,
                    objective="regret")
    assert np.array_equal(a.w, b.w) and a.fit_regret == b.fit_regret
    # Deterministic, and it must differ from rank — equal means the branch
    # does not run
    r = fit_weights(score, m, t, _all_train(t), [1.0, 1.0, 1.0], max_evals=60,
                    objective="rank")
    assert not np.array_equal(a.w, r.w), (
        "the objective branch does not run")


def test_rank_pairs_drop_the_noise_indistinguishable(known):
    """★ Pairs the noise floor cannot separate drop out of the loss.

    Without dropping them it fits noise. If not one pair drops,
    `resolvable` is not running.
    """
    from kernelrule.core.weights import _Problem

    t, m, _score = known
    pr = _Problem(m, t, _all_train(t).shapes, 1)
    pr.build_pairs(t, 100)
    assert pr.n_pairs > 0
    assert pr.n_dropped >= 0
    assert pr.n_pairs + pr.n_dropped > 0


def test_rank_objective_still_records_regret(known):
    """Even under `objective="rank"`, `fit_regret` is **regret**.

    "score by regret, train by the rank loss" — changing the scoring
    criterion makes it impossible to place alongside existing results
    (`rank-evo-prereg.md` §3).
    """
    from kernelrule.core.weights import _Problem

    t, m, score = known
    fr = fit_weights(score, m, t, _all_train(t), [1.0, 1.0, 1.0],
                     max_evals=60, objective="rank", rank_top_k=100)
    pr = _Problem(m, t, _all_train(t).shapes, 1)
    assert fr.fit_regret == pytest.approx(pr.regret(score, fr.w), abs=1e-12)


def test_rank_loss_prefers_the_true_order(known):
    """★ At the true coefficients the rank loss must be **smaller**. A
    sign check.

    The correct order is `s_i < s_j` (lower is better), and with the sign
    flipped the loss silently learns the opposite.
    """
    from kernelrule.core.weights import _Problem

    t, m, score = known
    pr = _Problem(m, t, _all_train(t).shapes, 1)
    pr.build_pairs(t, 100)
    good = pr.rank_loss(score, W_TRUE)
    bad = pr.rank_loss(score, -W_TRUE)
    assert good < bad, (
        f"the sign is flipped: true {good:.4f} vs reversed {bad:.4f}")


def test_unknown_objective_is_refused(known):
    t, m, score = known
    with pytest.raises(FitError, match="unknown objective"):
        fit_weights(score, m, t, _all_train(t), [1.0, 1.0, 1.0],
                    objective="nope")


def test_restarts_actually_run(known):
    """★ Blocks writing down `n_restarts` while only 1 runs.

    It really happened — on the rank path, giving L-BFGS
    `maxfun=max_evals` meant it **used the whole budget alone and 0 restarts
    ran**. The comment said "the restarts are left as they are" and it was
    false (principle 1).
    """
    import warnings as _w

    from kernelrule.core.weights import FitWarning

    t, m, score = known
    for obj in ("regret", "rank"):
        with _w.catch_warnings(record=True) as got:
            _w.simplefilter("always")
            fit_weights(score, m, t, _all_train(t), [1.0, 1.0, 1.0],
                        max_evals=200, n_restarts=4, objective=obj)
        bad = [str(x.message) for x in got
               if issubclass(x.category, FitWarning)
               and "restarts ran" in str(x.message)]
        assert not bad, f"{obj}: {bad}"


def test_canonical_scoring_pins_regret():
    """★ Final scoring is **always regret** — and it must be stated.

    The default of `fit_weights` changed to `rank` (D-101). If
    `canonical.py` does not state it, **every number in this project
    silently becomes something else.** It is checked directly in the source.
    """
    import inspect

    from kernelrule.core import canonical

    src = inspect.getsource(canonical)
    i = src.index("fit_weights(")
    depth, k = 1, i + len("fit_weights(")
    while depth:
        depth += {"(": 1, ")": -1}.get(src[k], 0)
        k += 1
    assert 'objective="regret"' in src[i:k], (
        "canonical does not state the objective — if the default changes, "
        "final scoring silently changes with it")


def test_history_experiments_pin_their_objective():
    """★ Do the experiment scripts that reproduce old conditions state the
    objective?

    The moment the default changed, the 20 scripts that simply called
    `fit_weights` **all started measuring something else.** It is the kind
    that changes silently, so it is pinned by a test.
    """
    from pathlib import Path

    bad = []
    for f in sorted(Path("experiments").glob("*.py")):
        s = f.read_text()
        i = 0
        while True:
            j = s.find("fit_weights(", i)
            if j < 0:
                break
            depth, k = 1, j + len("fit_weights(")
            while depth and k < len(s):
                depth += {"(": 1, ")": -1}.get(s[k], 0)
                k += 1
            if "objective=" not in s[j:k]:
                bad.append(f"{f.name}:{s[:j].count(chr(10)) + 1}")
            i = k
    assert not bad, f"calls that do not state the objective: {bad}"
