"""The noise floor (§30). **Going back to a fixed 1% is caught here.**"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from kernelrule.core.noise import NoiseMismatchError, NoiseModel


def test_15us_kernel_cannot_resolve_one_percent(noise_a6000):
    """On a 15µs kernel the noise floor exceeds 1% (6.8% measured).

    Going back to a fixed 1% tolerance trips this test (§30.3).
    """
    assert noise_a6000.floor(0.015) > 0.01
    assert noise_a6000.floor(0.014) > 0.07     # one tick is 7.3%


def test_tick_dominates_below_1_5ms(noise_a6000):
    """1.5ms is the boundary. Below it **resolution exceeds statistics**
    (§30.2)."""
    m = noise_a6000
    assert m.tick_pct(0.014) > m.sigma(0.014)
    assert m.tick_pct(0.5) > m.sigma(0.5)
    assert m.tick_pct(4.0) < m.sigma(4.0)
    # Is the boundary roughly near 1.5ms
    lo, hi = 1.0, 2.5
    assert m.tick_pct(lo) > m.sigma(lo) * 0.9
    assert m.tick_pct(hi) < m.sigma(hi)


def test_floor_is_max_not_sum(noise_a6000):
    """It is `max`, not a sum, because the two terms are of different
    character (§30.2)."""
    m = noise_a6000
    for t in (0.011, 0.1, 1.0, 5.0):
        assert m.floor(t) == pytest.approx(max(m.sigma(t), m.tick_pct(t)))


def test_floor_scales_35x_across_shape_sizes(noise_a6000):
    """The floor varies greatly with shape size. Why a single threshold is
    wrong."""
    small = noise_a6000.floor(0.0113)     # this table's smallest best time
    large = noise_a6000.floor(9.74)       # the largest
    assert small / large > 30


def test_floor_fails_conservatively(noise_a6000):
    """★ When it cannot be computed it is a **conservatively large** value,
    not 0 (§26.4).

    Returning 0 would mean "every difference is significant", and all the
    noise would be learned as signal.
    """
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        assert noise_a6000.floor(bad) == 1.0
    arr = noise_a6000.floor(np.array([0.0, np.nan, 1.0]))
    assert arr[0] == 1.0 and arr[1] == 1.0 and arr[2] < 0.01


def test_answer_tol_is_two_sigma(noise_a6000):
    """The answer tolerance is 2σ. Overestimating is safer than
    underestimating (§30.3)."""
    assert noise_a6000.answer_tol(0.5) == pytest.approx(2.0 * noise_a6000.floor(0.5))


def test_from_bundle_records_tick_fallback(real_bundle_path):
    """A schema_version 1 bundle has no `tick_ms`. **That fact is
    recorded** (§30.3b)."""
    from kerneltab.core.bundle import load_bundle

    b = load_bundle(real_bundle_path)
    assert b.schema_version == 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = NoiseModel.from_bundle(b)
    assert m.tick_is_fallback is True
    assert any("tick_ms" in str(x.message) for x in w), (
        "the tick_ms fallback passed silently — another GPU's bundle would "
        "use the wrong tick")
    assert "tick_ms=fallback" in m.source


def test_from_bundle_records_coefficient_mismatch(real_bundle_path, monkeypatch):
    """A coefficient mismatch is **recorded**. It does not block.

    ## ★ Correction history (2026-08-31)

    This test originally demanded an **error**. The rationale was
    "`answer_set()` uses the module globals, so scoring in a mismatched
    state builds the answer set with the wrong coefficients".

    **kernelTab closed that hole** — `answer_set`/`answer_tolerance` now
    demand coefficient injection and raise `NoiseCoefRequired` without it.
    Then comparing values becomes **a road closure that blocks other GPUs
    forever** (the 5090 tick is 1/64 of the A6000's, so it differs by
    definition).

    Guarding the danger moved to `_require_injected_noise` — see
    `test_guard_catches_a_revived_default_tolerance` below.
    """
    from kerneltab.core import noise as kt
    from kerneltab.core.bundle import load_bundle

    monkeypatch.setattr(kt, "SIGMA_ABS_MS", 0.999)
    b = load_bundle(real_bundle_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = NoiseModel.from_bundle(b)
    assert m.sigma_abs_ms != 0.999, "it must use the bundle coefficients"
    assert any("A6000 reference values" in str(x.message) for x in caught), (
        "it must not pass silently — it has to be recorded as a fact")
    assert pytest is not None


def test_from_bundle_requires_coefficients():
    """With no coefficients it errors instead of papering over with a
    default (§26.4)."""
    class FakeBundle:
        info = {"bundle_id": "X"}
        tick_ms = 0.001
    with pytest.raises(NoiseMismatchError):
        NoiseModel.from_bundle(FakeBundle())


# ---------------------------------------------------------------------------
# The capability check — not a value comparison but "is the dangerous path
# closed" (2026-08-31)
# ---------------------------------------------------------------------------
def test_guard_catches_a_revived_default_tolerance(monkeypatch):
    """★ If `answer_tolerance` returns a value without injection, this
    **blocks**.

    The old check (bundle coefficients == module globals) could not catch
    this — on the development A6000 bundle the values agreed so it passed.
    And it blocked other GPUs forever, where there was no danger at all.
    Wrong in both directions.
    """
    import pytest

    from kernelrule.core.noise import NoiseMismatchError, _require_injected_noise

    _require_injected_noise()           # it must pass as things stand
    import kerneltab.core.table as ktt
    monkeypatch.setattr(ktt, "answer_tolerance", lambda *a, **k: 0.01)
    with pytest.raises(NoiseMismatchError, match="a default has come back"):
        _require_injected_noise()


def test_foreign_gpu_bundle_is_not_blocked_by_coefficient_drift():
    """For another GPU's bundle, **differing coefficients are normal**. It
    records rather than blocks.

    The 5090's tick is 1/64 of the A6000's. Demanding equal values blocks
    every non-A6000 bundle forever — not a safeguard but a road closure.
    """
    import warnings
    from pathlib import Path

    import pytest

    b = Path("datasets/rtx-5090-sm_120-5bb6f403")
    if not b.exists():
        pytest.skip("no 5090 bundle")
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = PerfTable.from_bundle(str(b), env_hash="5bb6f403")
    assert t.noise.tick_ms < 1e-4 and not t.noise.tick_is_fallback
    msgs = [str(x.message) for x in caught]
    assert any("A6000 reference values" in m for m in msgs), (
        "the mismatch must be **recorded as a fact** — it must not pass "
        "silently")
