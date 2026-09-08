"""★ The bundle checker's **digit tolerance** (D-125).

The release note writes a rounded value — the 4090 says
`sigma_abs 0.000743` while the bundle has `0.0007433368963633708`.
Comparing with `tol=1e-12` **rejected a perfectly good bundle.**

It is not loosening things but looking only as far as **the announced
precision**. Giving more digits makes it stricter — that is checked by
turning it back (principle 38).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from bundle_guard import (  # noqa: E402
    A6000_SIGMA_ABS_MS,
    A6000_TICK_MS,
    _close,
    _close_at,
)


@pytest.mark.parametrize(("value", "expect", "ok"), [
    # Where the 4090 was caught — it matches the rounded announced value
    (0.0007433368963633708, "0.000743", True),
    (3.2e-05, "0.000032", True),
    (3.2e-05, "3.2e-05", True),
    # ★ More digits makes it stricter
    (0.0007438, "0.0007433", False),
    (0.00074333, "0.0007433", True),
    # ★ A genuinely different value has to be caught — the A6000 coefficients
    #   riding in
    (A6000_SIGMA_ABS_MS, "0.000743", False),
    (A6000_TICK_MS, "0.000032", False),
])
def test_close_at_respects_the_announced_precision(value, expect, ok):
    got, _tol = _close_at(value, expect)
    assert got is ok


def test_a6000_check_stays_exact():
    """★ The A6000 coefficient check **did not get looser** — it is a
    different comparator."""
    assert _close(A6000_SIGMA_ABS_MS, A6000_SIGMA_ABS_MS)
    assert not _close(0.000374001, A6000_SIGMA_ABS_MS)
