"""★ 번들 검사기의 **자릿수 허용오차** (D-125).

릴리즈 노트는 반올림값을 적는다 — 4090 은 `sigma_abs 0.000743` 인데
번들은 `0.0007433368963633708` 이다. `tol=1e-12` 로 견주다가 **멀쩡한
번들을 거부했다.**

느슨하게 푼 것이 아니라 **예고된 정밀도까지만** 본다. 자릿수를 더 주면
더 엄격해진다 — 그것을 되돌려 확인한다 (원칙 38).
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
    # 4090 이 걸렸던 자리 — 반올림한 예고값과 맞는다
    (0.0007433368963633708, "0.000743", True),
    (3.2e-05, "0.000032", True),
    (3.2e-05, "3.2e-05", True),
    # ★ 자릿수를 더 주면 더 엄격하다
    (0.0007438, "0.0007433", False),
    (0.00074333, "0.0007433", True),
    # ★ 진짜로 다른 값은 걸러야 한다 — A6000 계수가 실린 경우
    (A6000_SIGMA_ABS_MS, "0.000743", False),
    (A6000_TICK_MS, "0.000032", False),
])
def test_close_at_respects_the_announced_precision(value, expect, ok):
    got, _tol = _close_at(value, expect)
    assert got is ok


def test_a6000_check_stays_exact():
    """★ A6000 계수 검사는 **느슨해지지 않았다** — 다른 비교자다."""
    assert _close(A6000_SIGMA_ABS_MS, A6000_SIGMA_ABS_MS)
    assert not _close(0.000374001, A6000_SIGMA_ABS_MS)
