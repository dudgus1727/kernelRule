"""The time-axis check (D-192): a time axis must be lowest where the time
is. Pure functions here; the calibration on real axes is
`experiments/time_gate_calib.py`."""
from __future__ import annotations

import numpy as np

from kernelrule.features.time_gate import (
    INTERIOR_SHARE,
    RANK_MIN,
    slice_stats,
    verdict,
)


def test_a_monotone_axis_misses_an_interior_optimum():
    sk = np.array([1, 2, 3, 4, 8])
    t = np.array([5.0, 3.0, 2.0, 2.5, 4.0])          # lowest at 3
    mono = 1.0 / sk                                   # lowest at 8
    s = slice_stats(sk, t, mono)
    assert s == {"hit": False, "interior": True, "end_miss": True}
    bowl = (sk - 3.0) ** 2
    assert slice_stats(sk, t, bowl) == {"hit": True, "interior": True,
                                        "end_miss": False}


def test_an_axis_that_ignores_the_field_makes_no_claim():
    sk = np.array([1, 2, 4])
    assert slice_stats(sk, np.array([3.0, 1.0, 2.0]),
                       np.array([7.0, 7.0, 7.0])) is None
    assert slice_stats(np.array([1, 2]), np.array([1.0, 2.0]),
                       np.array([1.0, 2.0])) is None


def _diag(rank, interior, end_miss, n):
    return {"n_shapes": n, "rank": rank, "pick_gm": 1.1,
            "slices": {"split_k": {"n": n, "interior": interior,
                                   "end_miss": end_miss, "hit": 0},
                       "stages": {"n": n, "interior": 0, "end_miss": 0,
                                  "hit": 0},
                       "raster_width": {"n": 0, "interior": 0,
                                        "end_miss": 0, "hit": 0}}}


def test_verdict_rank_and_slices():
    assert verdict(_diag(0.8, 32, 5, 47)) is None
    m = verdict(_diag(0.8, 32, 32, 47))
    assert m and "cfg.split_k" in m and "32 of 47" in m
    assert verdict(_diag(RANK_MIN - 0.01, 0, 0, 47)) is not None
    # a field where the time is rarely lowest inside is not judged
    few = int(INTERIOR_SHARE * 44) - 1
    assert verdict(_diag(0.8, few, few, 44)) is None


def test_the_gate_is_a_recorded_condition_that_defaults_off():
    from kernelrule.core.loop import LoopConfig
    from kernelrule.core.runset import _OLD_DEFAULTS, KEYS

    assert LoopConfig(run_id="x").time_gate is False
    assert "time_gate" in KEYS and _OLD_DEFAULTS["time_gate"] is False
