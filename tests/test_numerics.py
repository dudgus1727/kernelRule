"""Are the floating-point comparisons gathered into one place (§30.13,
principle 2)?

`abs(a - b) < tol` falls apart at `inf` — `abs(inf-inf)` is `nan` and
`nan < x` is False, so **it is judged "different".** The `expected_range`
check really fell into that trap, and `expected_range` really does contain
`inf`
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from kernelrule.core.numerics import DEFAULT_TOL, approx_equal, approx_zero

INF, NAN = float("inf"), float("nan")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("a", "b", "want"), [
    (INF, INF, True),          # ★ abs(inf-inf) is nan
    (-INF, -INF, True),
    (INF, -INF, False),
    (INF, 1e300, False),
    (NAN, NAN, False),         # ★ nan differs even from itself
    (NAN, 1.0, False),
    (1.0, 1.0 + 1e-12, True),
    (1.0, 2.0, False),
    (0.0, 0.0, True),
])
def test_approx_equal_handles_inf_and_nan(a, b, want):
    assert approx_equal(a, b) is want


def test_nan_is_not_zero():
    """Passing a broken computation off as "it is 0" makes that fact
    disappear (§26.4)."""
    assert approx_zero(0.0)
    assert approx_zero(1e-12)
    assert not approx_zero(NAN)
    assert not approx_zero(INF)


def test_naive_comparison_would_have_failed():
    """★ It records why this check is needed — what the naive comparison
    actually does."""
    assert math.isnan(abs(INF - INF))
    assert not (abs(INF - INF) < DEFAULT_TOL)   # this was D-71's false negative
    assert approx_equal(INF, INF)               # the shared function is not fooled


# ---------------------------------------------------------------------------
# ★ It fails if a place using a direct comparison is left
# ---------------------------------------------------------------------------

#: Where a direct comparison is allowed.
#:   numerics.py itself — it is the implementation
#:   noise.py          — it handles finite values only and the comparison is
#:                       a **relative** error
_MAY_COMPARE = {
    "kernelrule/core/numerics.py",
    "kernelrule/core/noise.py",
}


#: A literal on the right smaller than this is taken as "a tolerance". Larger
#: than that is a threshold comparison — `abs(auc - 0.5) > 0.05` is not an
#: equality test but "how far from the threshold", and `inf` cannot arrive
#: there.
_TOL_LITERAL = 1e-3


def _is_tolerance_compare(node: ast.AST) -> bool:
    """Is it of the form `abs(a - b) <op> <tolerance>`?

    **A threshold comparison is excluded.** What this check is after is the
    place that asks "are the two values the same" in floating point, and only
    there do `inf`/`nan` become a problem.
    """
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return False
    if not isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
        return False
    left = node.left
    if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
            and left.func.id == "abs" and len(left.args) == 1
            and isinstance(left.args[0], ast.BinOp)
            and isinstance(left.args[0].op, ast.Sub)):
        return False
    rhs = node.comparators[0]
    if isinstance(rhs, ast.Constant) and isinstance(rhs.value, (int, float)):
        return abs(float(rhs.value)) <= _TOL_LITERAL
    # A name like `TOL` / `tol` / `_EPS` is taken as a tolerance
    txt = ast.unparse(rhs).lower()
    return "tol" in txt or "eps" in txt


def test_no_raw_float_comparison_outside_numerics():
    bad: list[str] = []
    for f in sorted([*(ROOT / "kernelrule").rglob("*.py"),
                     *(ROOT / "experiments").glob("*.py")]):
        rel = f.relative_to(ROOT).as_posix()
        if rel in _MAY_COMPARE:
            continue
        for node in ast.walk(ast.parse(f.read_text(), filename=rel)):
            if _is_tolerance_compare(node):
                bad.append(f"  {rel}:{node.lineno}  {ast.unparse(node)[:60]}")
    assert not bad, (
        "`abs(a - b) < tol` is used directly — it falls apart at inf (D-71).\n"
        "Use `kernelrule.core.numerics.approx_equal`:\n" + "\n".join(bad))
