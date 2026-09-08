"""Scoring — hand-computed pinned values (§26.2)."""
from __future__ import annotations

import numpy as np
import pytest
from toy import make_table, order_by_index

from kernelrule.core.scoring import Strata, evaluate, geomean, is_significant


@pytest.fixture
def t3():
    """3 shapes x 5 configs. The times are specified by hand."""
    return make_table({
        (1024, 4096, 4096): [1.0, 1.2, 2.0, 4.0, 8.0],
        (2048, 4096, 4096): [2.0, 2.5, 3.0, 3.5, 10.0],
        (4096, 4096, 4096): [4.0, 4.4, 4.8, 20.0, 40.0],
    })


def test_best_and_difficulty_hand_computed(t3):
    shapes = t3.shapes()
    assert [t3.best_time(p) for p in shapes] == [1.0, 2.0, 4.0]
    # difficulty = median/best. The medians are [2.0, 3.0, 4.8]
    assert [t3.difficulty(p) for p in shapes] == [2.0, 1.5, 1.2]


def test_regret_hand_computed(t3):
    """The rule produces the index order [2,1,0,3,4] on every shape.

        shape1 top1 = idx2 -> 2.0 / 1.0 = 2.0   top3 = min(2.0,1.2,1.0) = 1.0
        shape2 top1 = idx2 -> 3.0 / 2.0 = 1.5   top3 = min(3.0,2.5,2.0)/2.0 = 1.0
        shape3 top1 = idx2 -> 4.8 / 4.0 = 1.2   top3 = min(4.8,4.4,4.0)/4.0 = 1.0
    """
    ev = evaluate(order_by_index([2, 1, 0, 3, 4]), t3, ks=(1, 3))
    assert list(ev.regret[:, 0]) == [2.0, 1.5, 1.2]
    assert list(ev.regret[:, 1]) == [1.0, 1.0, 1.0]
    assert ev.at(1) == pytest.approx((2.0 * 1.5 * 1.2) ** (1 / 3))
    assert ev.at(3) == pytest.approx(1.0)


def test_perfect_rule_gives_exactly_one(t3):
    """A rule that returns the answer exactly has regret **exactly**
    1.0."""
    ev = evaluate(order_by_index([0, 1, 2, 3, 4]), t3, ks=(1, 3, 5))
    assert ev.at(1) == 1.0
    assert np.all(ev.regret == 1.0) is np.True_ or ev.regret[:, 0].tolist() == [1.0] * 3
    assert ev.hit_rate(1) == 1.0


def test_worst_rule_matches_max_over_best(t3):
    """A reversed rule's regret equals (worst/best)."""
    ev = evaluate(order_by_index([4, 3, 2, 1, 0]), t3, ks=(1,))
    assert list(ev.regret[:, 0]) == [8.0, 5.0, 10.0]
    assert ev.at(1) == pytest.approx(geomean([8.0, 5.0, 10.0]))


def test_geomean_not_arithmetic():
    """regret is a ratio scale. With an arithmetic mean the large values
    dominate."""
    assert geomean([1.0, 4.0]) == pytest.approx(2.0)
    assert geomean([1.0, 4.0]) != pytest.approx(2.5)


def test_order_must_be_a_permutation(t3):
    """★ Dropping candidates favours top-k. It does not pass silently
    (§26.4)."""
    with pytest.raises(ValueError, match="order length"):
        evaluate(order_by_index([0, 1, 2]), t3, ks=(1,))
    with pytest.raises(ValueError, match="not a permutation"):
        evaluate(order_by_index([0, 0, 1, 2, 3]), t3, ks=(1,))


def test_empty_shape_set_is_an_error(t3):
    with pytest.raises(ValueError, match="no shapes at all"):
        evaluate(order_by_index([0, 1, 2, 3, 4]), t3, shapes=[], ks=(1,))


def test_strata_splits_by_difficulty_and_size(t3):
    s = Strata.build(t3)
    # difficulty [2.0, 1.5, 1.2] -> above the median 1.5 is the hard half
    assert list(s.hard) == [True, False, False]
    # best times [1.0, 2.0, 4.0] -> all at or above 0.5ms
    assert not s.small.any()


def test_stratified_report_always_has_all_axes(t3):
    ev = evaluate(order_by_index([1, 0, 2, 3, 4]), t3, ks=(1,))
    st = ev.stratified(1)
    assert {"all", "large(>=0.5ms)", "small(<0.5ms)", "hard", "easy"} <= set(st)
    # ★ The size stratification comes first (§30.5)
    keys = list(st)
    assert keys.index("large(>=0.5ms)") < keys.index("hard")


def test_significance_uses_shape_noise_not_fixed_threshold():
    """★ It does not use a fixed threshold. It differs per shape (§7.4)."""
    from kernelrule.core.noise import NoiseModel
    m = NoiseModel.a6000_reference()
    big = make_table({(4096, 4096, 4096): [4.0, 4.1]}, noise=m)
    small = make_table({(512, 512, 512): [0.012, 0.013]}, noise=m)
    ev_b = evaluate(order_by_index([0, 1]), big, ks=(1,))
    ev_s = evaluate(order_by_index([0, 1]), small, ks=(1,))
    # The same 0.5% difference is significant on the large shape and not on
    # the small one
    assert is_significant(0.005, ev_b)
    assert not is_significant(0.005, ev_s)


def test_significance_fails_closed():
    """When it cannot be computed it is taken as **not significant**
    (§26.4)."""
    t = make_table({(1024, 4096, 4096): [1.0, 2.0]})
    ev = evaluate(order_by_index([0, 1]), t, ks=(1,))
    ev.tol[:] = np.nan
    assert is_significant(999.0, ev) is False


def test_answer_mask_uses_noise_floor_not_one_percent():
    """The answer set is built from the per-shape noise floor (§30.3)."""
    from kernelrule.core.noise import NoiseModel
    m = NoiseModel.a6000_reference()
    # 11.3µs shape: the floor is 9%, so everything within 2σ = 18% is an
    # answer
    t = make_table({(512, 512, 512): [0.0113, 0.0120, 0.0130, 0.0200]}, noise=m)
    p = t.shapes()[0]
    assert t.answer_mask(p).tolist() == [True, True, True, False]
    # 4ms shape: the floor is 0.053%, so a 1% difference is clearly wrong
    t2 = make_table({(4096, 4096, 4096): [4.0, 4.04]}, noise=m)
    assert t2.answer_mask(t2.shapes()[0]).tolist() == [True, False]


# ---------------------------------------------------------------------------
# ★ Comparing two methods — won/lost is not declared from a geomean
# difference alone (§7.4)
# ---------------------------------------------------------------------------
def test_comparison_uses_shape_noise_not_geomean():
    """★ "1.085 > 1.080, so it lost" is imprecise.

    In a 61-shape geomean, 0.5% can appear when a few shapes move by a single
    tick. **"It lost significantly on N shapes" is the precise statement.**
    """
    from kernelrule.core.noise import NoiseModel
    from kernelrule.core.scoring import compare

    m = NoiseModel.a6000_reference()
    t = make_table({
        # 4ms shape: floor 0.053%. A 1% difference is clear
        (4096, 4096, 4096): [4.0, 4.04],
        # 11us shape: floor 9.1%. Even a 6% difference is indistinguishable
        (512, 512, 512): [0.0113, 0.0120],
    }, noise=m)
    a = evaluate(order_by_index([1, 0]), t, ks=(1,), label="A")
    b = evaluate(order_by_index([0, 1]), t, ks=(1,), label="B")
    c = compare(a, b, t, name_a="A", name_b="B")
    assert int(c.a_loses.sum()) == 1        # the large shape only
    assert int(c.tied.sum()) == 1           # the small one is indistinguishable
    assert "significantly" in c.report()


def test_comparison_reports_magnitude_not_just_significance():
    """★ Sigma says "is it real", not "how large".

    On long shapes the noise floor is 0.05%, so even a small difference is
    hundreds of sigma. The size (the regret difference) has to be shown
    alongside or it gets misread.
    """
    from kernelrule.core.noise import NoiseModel
    from kernelrule.core.scoring import compare

    # 4ms shape, floor 0.053%. A 1% difference -> about 19 sigma, yet the
    # regret difference is only 0.01. (Tried 0.1% first and got 1.87 sigma —
    # not significant, which was correct. The expectation was wrong, not the
    # code.)
    t = make_table({(4096, 4096, 4096): [4.0, 4.04]},
                   noise=NoiseModel.a6000_reference())
    a = evaluate(order_by_index([1, 0]), t, ks=(1,))
    b = evaluate(order_by_index([0, 1]), t, ks=(1,))
    c = compare(a, b, t)
    assert c.sigma[0] > 10, (
        "a 1% difference must be significant on a large shape")
    assert abs(c.delta[0]) < 0.02, "and yet the magnitude is small"
    rep = c.report()
    assert "regret" in rep and "sigma" in rep


def test_comparison_refuses_mismatched_shape_sets():
    """A comparison only holds when measured on the same shape set
    (§30.8)."""
    from kernelrule.core.scoring import compare

    t = make_table({(1024, 4096, 4096): [1.0, 2.0],
                    (2048, 4096, 4096): [1.0, 2.0]})
    sh = t.shapes()
    a = evaluate(order_by_index([0, 1]), t, sh, ks=(1,))
    b = evaluate(order_by_index([0, 1]), t, sh[:1], ks=(1,))
    with pytest.raises(ValueError, match="different shape sets"):
        compare(a, b, t)
