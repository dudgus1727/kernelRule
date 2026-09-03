"""★ §3 보고서의 형태 계수기 — **제곱과 교차곱을 따로** 센다 (D-110·D-123).

D-110 에서 두 쪽을 합집합으로 접었더니 `f.a * f.a` 와 `f.a * f.b` 가 같은
"쌍" 이 되고, 최빈 쌍이 실은 `reg_pressure^3` 였다. 계수기는 **되돌려서**
확인한다 (원칙 38) — 세야 할 것과 세면 안 될 것을 둘 다 넣는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from expressive_report import _n_power, _squares_and_crosses  # noqa: E402


def _f(body: str) -> str:
    return f"def score(f, p, hw, w):\n    s = {body}\n    return s\n"


@pytest.mark.parametrize(("body", "sq", "cr"), [
    ("f.a * f.a * w[0]", 1, 0),                 # 제곱
    ("(f.a * f.b) * w[0]", 0, 1),               # 교차곱
    ("np.square(f.a) * w[0]", 1, 0),            # np.square 도 제곱이다
    ("np.square(f.a) * w[0] + (f.a * f.b) * w[1]", 1, 1),
    ("f.a * w[0] + f.b * w[1]", 0, 0),          # ★ 곱이 아니다
    ("f.a * w[0] * f.b", 0, 0),                 # ★ 가중치가 낀 곱은 빼다
    ("np.power(f.a, w[0])", 0, 0),              # 지수는 곱이 아니다
])
def test_squares_and_crosses(body, sq, cr):
    assert _squares_and_crosses(_f(body)) == (sq, cr)


@pytest.mark.parametrize(("body", "n"), [
    ("np.power(f.a, w[0])", 1),
    ("f.a ** w[0]", 1),
    ("f.a ** w[0] + np.power(f.b, w[1]) * w[2]", 2),
    ("np.power(f.a, 2.0) * w[0]", 0),           # ★ 상수 지수는 자유도가 아니다
    ("np.square(f.a) * w[0]", 0),
    ("f.a * w[0]", 0),
])
def test_n_power_counts_only_weights_in_the_exponent(body, n):
    """★ 지수 **자리에 가중치**가 든 항만 센다 (D-112 가 준 자유도다).

    `np.power(f.a, 2.0)` 은 상수 지수라 적합기가 맞출 것이 없다 — 세면
    "지수 자리를 썼다" 가 거짓이 된다.
    """
    assert _n_power(_f(body)) == n
