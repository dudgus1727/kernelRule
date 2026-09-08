"""Core types — ordering and tie-break (§types, §30.7)."""
from __future__ import annotations

import numpy as np
import pytest

from kernelrule.core.types import (
    CandidateSet,
    Hardware,
    Problem,
    hardware_from_env,
    make_tiebreak,
)


def _cand(n=5):
    kid = np.array([f"k{i}" for i in range(n)], dtype=object)
    sk = np.arange(1, n + 1, dtype=np.int64)
    mode = np.array(["serial"] * n, dtype=object)
    return CandidateSet(n=n, kernel_id=kid, split_k=sk, split_k_mode=mode,
                        tiebreak=make_tiebreak(kid, sk, mode))


def test_order_by_sorts_ascending():
    c = _cand()
    order = c.order_by(np.array([5.0, 1.0, 3.0, 2.0, 4.0]))
    assert order.tolist() == [1, 3, 2, 4, 0]


def test_ties_broken_by_config_identity_only():
    """Ties break on config identity alone, independent of table row order."""
    c = _cand()
    order = c.order_by(np.zeros(5))
    assert order.tolist() == sorted(range(5), key=lambda i: c.tiebreak[i])


def test_order_by_rejects_wrong_length():
    c = _cand()
    with pytest.raises(ValueError, match="number of candidates"):
        c.order_by(np.zeros(3))


def test_candidate_set_length_mismatch_is_an_error():
    kid = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="length"):
        CandidateSet(n=3, kernel_id=kid, split_k=np.array([1, 2]),
                     split_k_mode=np.array(["serial", "serial"], dtype=object),
                     tiebreak=np.array([0, 1]))


def test_problem_and_config_are_frozen():
    """`pytest.raises(Exception)` will not do — a typo raising AttributeError
    would also pass. Narrowing to what a frozen dataclass raises is what
    actually verifies it."""
    import dataclasses

    p = Problem(1024, 4096, 4096)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)) as e:
        p.M = 2048
    assert "M" in str(e.value) or "frozen" in str(e.value).lower()


def test_ridge_point_uses_effective_values():
    hw = Hardware(name="A6000", arch="sm_86", sm_count=84,
                  smem_per_block=101376, max_threads_per_sm=1536,
                  regs_per_sm=65536, peak_tflops_f16=116.1,
                  bandwidth_gbps=729.7, l2_bytes=6291456)
    assert hw.ridge_point == pytest.approx(159.1, abs=0.2)


def test_hardware_from_env_warns_without_effective_values():
    """★ Spec values put the ridge point 26% high. Do not let it pass."""
    env = {"hardware": {
        "name": "A6000", "arch": "sm_86", "sm_count": 84,
        "smem_per_block": 101376, "max_threads_per_sm": 1536,
        "regs_per_sm": 65536, "peak_tflops_f16": 154.8,
        "bandwidth_gbps": 768.0, "l2_bytes": 6291456}}
    with pytest.warns(UserWarning, match="effective"):
        hw = hardware_from_env(env)
    assert hw.peak_tflops_f16 == 154.8      # cannot correct — hence the warning


def test_hardware_from_env_applies_effective_values(real_bundle_path):
    import json
    env = json.loads((real_bundle_path / "env.json").read_text())
    hw = hardware_from_env(env)
    assert hw.peak_tflops_f16 == pytest.approx(116.1)
    assert hw.bandwidth_gbps == pytest.approx(729.7)
    assert hw.ridge_point == pytest.approx(159.1, abs=0.2)


def test_tiebreak_is_a_permutation():
    c = _cand(7)
    assert sorted(c.tiebreak.tolist()) == list(range(7))
