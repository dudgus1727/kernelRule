"""★ The form counters of the §3 report — squares and cross products are
counted **separately** (D-110·D-123).

In D-110 the two sides were folded into a union, so `f.a * f.a` and
`f.a * f.b` became the same "pair" and the most frequent pair was in fact
`reg_pressure^3`. The counters are checked **by turning them back**
(principle 38) — both what should be counted and what should not are put in.
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
    ("f.a * f.a * w[0]", 1, 0),                 # a square
    ("(f.a * f.b) * w[0]", 0, 1),               # a cross product
    ("np.square(f.a) * w[0]", 1, 0),            # np.square is a square too
    ("np.square(f.a) * w[0] + (f.a * f.b) * w[1]", 1, 1),
    ("f.a * w[0] + f.b * w[1]", 0, 0),          # ★ not a product
    ("f.a * w[0] * f.b", 0, 0),                 # ★ a product with a weight in
                                                #   it is left out
    ("np.power(f.a, w[0])", 0, 0),              # an exponent is not a product
])
def test_squares_and_crosses(body, sq, cr):
    assert _squares_and_crosses(_f(body)) == (sq, cr)


@pytest.mark.parametrize(("body", "n"), [
    ("np.power(f.a, w[0])", 1),
    ("f.a ** w[0]", 1),
    ("f.a ** w[0] + np.power(f.b, w[1]) * w[2]", 2),
    ("np.power(f.a, 2.0) * w[0]", 0),           # ★ a constant exponent is not
                                                #   a degree of freedom
    ("np.square(f.a) * w[0]", 0),
    ("f.a * w[0]", 0),
])
def test_n_power_counts_only_weights_in_the_exponent(body, n):
    """★ Only terms with **a weight in the exponent slot** are counted (that
    is the degree of freedom D-112 gave).

    `np.power(f.a, 2.0)` has a constant exponent, so there is nothing for the
    fitter to fit — counting it would make "the exponent slot was used"
    false.
    """
    assert _n_power(_f(body)) == n
