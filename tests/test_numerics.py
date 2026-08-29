"""부동소수 비교를 한 곳으로 모았는가 (§30.13, 원칙 2).

`abs(a - b) < tol` 은 `inf` 에서 무너진다 — `abs(inf-inf)` 가 `nan` 이고
`nan < x` 는 False 라 **"다르다" 로 판정된다.** 실제로 `expected_range`
대조가 그 함정에 빠졌고, `expected_range` 에는 `inf` 가 실제로 들어 있다
(D-71).
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
    (INF, INF, True),          # ★ abs(inf-inf) 는 nan 이다
    (-INF, -INF, True),
    (INF, -INF, False),
    (INF, 1e300, False),
    (NAN, NAN, False),         # ★ nan 은 자기 자신과도 다르다
    (NAN, 1.0, False),
    (1.0, 1.0 + 1e-12, True),
    (1.0, 2.0, False),
    (0.0, 0.0, True),
])
def test_approx_equal_handles_inf_and_nan(a, b, want):
    assert approx_equal(a, b) is want


def test_nan_is_not_zero():
    """계산이 깨진 것을 "0 이다" 로 넘기면 그 사실이 사라진다 (§26.4)."""
    assert approx_zero(0.0)
    assert approx_zero(1e-12)
    assert not approx_zero(NAN)
    assert not approx_zero(INF)


def test_naive_comparison_would_have_failed():
    """★ 이 검사가 왜 필요한지 남긴다 — 순진한 비교의 실제 동작."""
    assert math.isnan(abs(INF - INF))
    assert not (abs(INF - INF) < DEFAULT_TOL)   # 이것이 D-71 의 오탐이었다
    assert approx_equal(INF, INF)               # 공용 함수는 안 속는다


# ---------------------------------------------------------------------------
# ★ 직접 비교를 쓰는 곳이 남아 있으면 실패한다
# ---------------------------------------------------------------------------

#: 직접 비교가 허용되는 곳.
#:   numerics.py 자신 — 구현이다
#:   noise.py       — 유한값만 다루고 비교가 **상대** 오차다
_MAY_COMPARE = {
    "kernelrule/core/numerics.py",
    "kernelrule/core/noise.py",
}


#: 오른쪽이 이보다 작은 리터럴이면 "허용 오차" 로 본다. 그보다 크면
#: 임계값 비교다 — `abs(auc - 0.5) > 0.05` 는 동등 비교가 아니라
#: "임계에서 얼마나 떨어졌나" 이고 `inf` 가 들어올 수 없다.
_TOL_LITERAL = 1e-3


def _is_tolerance_compare(node: ast.AST) -> bool:
    """`abs(a - b) <op> <허용오차>` 꼴인가.

    **임계값 비교는 제외한다.** 이 검사가 잡으려는 것은 "두 값이 같은가" 를
    부동소수로 묻는 자리이고, 거기서만 `inf`/`nan` 이 문제가 된다.
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
    # `TOL` / `tol` / `_EPS` 같은 이름이면 허용 오차로 본다
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
        "`abs(a - b) < tol` 을 직접 쓴다 — inf 에서 무너진다 (D-71).\n"
        "`kernelrule.core.numerics.approx_equal` 을 써라:\n" + "\n".join(bad))
