"""★ D-195 — term caps (`kernelrule/rules/caps.py`) and the time main term
(`kernelrule/rules/time_term.py`)."""

import numpy as np
import pytest

from kernelrule.rules.caps import CapUnavailable, shape_range, term_caps
from kernelrule.rules.time_term import main_weight_indices

FIXED = """def score(f, p, hw, w):
    sm = np.maximum(f.tm_crit_ratio, f.tm_l2_ratio)
    t = np.sqrt(np.square(sm) + np.square(f.tm_dram_ratio))
    s = 10.0 * np.log2(t)
    s = s + f.corr * w[0]
    return s
"""

WEIGHTED = """def score(f, p, hw, w):
    t = np.sqrt(np.square(np.maximum(f.tm_crit_ratio, f.tm_l2_ratio))
                + np.square(f.tm_dram_ratio))
    s = np.log2(t) * w[0]
    s = s + f.corr * w[1]
    s = s + f.tm_dram_ratio * f.corr * w[2]
    return s
"""


def test_main_weights_are_the_ones_on_the_time_estimate():
    assert main_weight_indices(FIXED) == set()
    # w[2] mixes a path time with another axis: a correction
    assert main_weight_indices(WEIGHTED) == {0}


def test_shape_range_falls_back_to_the_full_range():
    v = np.zeros(1000)
    v[3] = 5.0                       # one config moves: p1..p99 is flat
    assert shape_range(v) == 5.0
    assert shape_range(np.arange(101.0)) == pytest.approx(98.0)


class _Feats:
    def __init__(self, d):
        self.__dict__.update(d)


class _Matrix:
    """Two shapes of four configs; the time estimate spans 10*log2(4) = 20
    score units in each, `corr` spans 2 in shape 0 and 4 in shape 1."""

    hw = None

    def __init__(self):
        crit = np.array([1.0, 2.0, 3.0, 4.0])
        self._f = [
            _Feats({"tm_crit_ratio": crit, "tm_l2_ratio": crit * 0.5,
                    "tm_dram_ratio": np.zeros(4),
                    "corr": np.array([0.0, 1.0, 2.0, 2.0])}),
            _Feats({"tm_crit_ratio": crit, "tm_l2_ratio": crit * 0.5,
                    "tm_dram_ratio": np.zeros(4),
                    "corr": np.array([0.0, 4.0, 1.0, 2.0])})]

    def feature_names(self):
        return ["tm_crit_ratio", "tm_l2_ratio", "tm_dram_ratio", "corr"]

    def for_shape(self, p):
        return self._f[p], None


def _fn(code):
    ns = {"np": np}
    exec(code, ns)                   # noqa: S102 — a test rule
    return ns["score"]


def test_a_correction_cannot_outweigh_the_time_estimate():
    m = _Matrix()
    caps = term_caps(_fn(FIXED), FIXED, [1.0], m, [0, 1], lam=1.0)
    lo, hi = caps[0]
    # ratios R(main)/R(corr) per shape are ~ 20/2 and 20/4; the 10th
    # percentile is near the smaller — the shape where corr moves most
    assert lo == -hi and 5.0 <= hi <= 6.0


def test_a_term_that_moves_no_shape_is_held_at_zero():
    code = FIXED.replace("f.corr * w[0]", "f.tm_dram_ratio * w[0]")
    caps = term_caps(_fn(code), code, [1.0], _Matrix(), [0, 1], lam=1.0)
    assert caps == [(0.0, 0.0)]


def test_caps_need_the_time_features():
    class _NoTime(_Matrix):
        def feature_names(self):
            return ["corr"]
    with pytest.raises(CapUnavailable):
        term_caps(_fn(FIXED), FIXED, [1.0], _NoTime(), [0, 1], lam=1.0)
