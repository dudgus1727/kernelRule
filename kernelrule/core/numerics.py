"""Float comparison — done **in exactly one place** (§30.13).

## Why it is centralised

`abs(a - b) < tol` breaks on `inf` and `nan`.

```python
abs(inf - inf)          # nan
nan < 1e-9              # False  -> judged "different"
```

The `expected_range` comparison actually fell into that trap —
`global_tile_traffic_amplification` had `[1, inf]` in its declaration, the
registry and the prompt alike, yet came out "different" (D-71). **`inf` really
does appear in `expected_range`, so this is not a coincidence.**

And this repository has several such comparisons.

```
verify_rules       recorded value vs recomputed value
physics_coverage   correlation threshold
expected_range     declaration comparison
contribution       zero test
answer set         noise floor
shape_level        relative variance
```

**The same judgement in several places drifts apart** (principle 2). This is
the sixth, after `is_reference` / `top_k` / `DEFAULT_MODEL` / `REGISTRY` /
`load_generated`.
"""

from __future__ import annotations

import math

__all__ = ["approx_equal", "approx_zero", "DEFAULT_TOL"]

#: Default tolerance. Where a site needs a different one, it says so.
DEFAULT_TOL = 1e-9


def approx_equal(a: float, b: float, tol: float = DEFAULT_TOL) -> bool:
    """`a == b` within a tolerance. **`inf`/`nan` are handled explicitly.**

    ```
    approx_equal(inf, inf)    True    ★ abs(inf-inf) is nan
    approx_equal(inf, -inf)   False
    approx_equal(nan, nan)    False   ★ nan differs even from itself
    ```

    The last one is deliberate — a `nan` means the computation broke, and
    passing it as "equal" erases that fact (§26.4).
    """
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return False
    if a == b:                  # catches inf == inf and -inf == -inf here
        return True
    if math.isinf(a) or math.isinf(b):
        return False            # only one of them infinite means different
    return abs(a - b) <= tol


def approx_zero(x: float, tol: float = DEFAULT_TOL) -> bool:
    """`x == 0` within a tolerance. `nan` is not zero."""
    return approx_equal(float(x), 0.0, tol)
