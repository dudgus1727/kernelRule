"""★ Term caps — a correction may not outweigh the time estimate (D-195).

## Why

D-193's loop rules beat K1a on inner-CV and then picked, on one holdout
shape, a spilling 256x256 kernel 16x slower than the best: the time main
term scored that config about 52 units worse, and the corrections together
moved it about 68 units the other way (every feature value was inside the
training range — not an extrapolation, a weight size). The fit is
scale-free (`fit_weights(space="u")`), so nothing ever limited how far one
correction can move the score relative to the time estimate.

## What is capped — each term, normalised by its own range, shape by shape

```
R_s(x)     in training shape s: the p1..p99 range of x over the configs
           (the full range when p1..p99 is flat — an effect on a few configs)
main       the canonical time estimate 10*log2(sqrt(max(tm_crit, tm_l2)^2
           + tm_dram^2)) — the yardstick, read from the features, not from
           the rule
term i     the score change when w[i] goes 0 -> 1 (the others held)
ratio_i    the 10th percentile, over the shapes where term i moves at all,
           of R_s(main) / R_s(term i)
correction |w[i]| <= lam * ratio_i — in (almost) every training shape the
           term moves the configs by at most lam x the time estimate
never      a term that moves no training shape has no evidence for any
moves      weight: capped at 0
main term  a weight on the time estimate itself (`rules/time_term`) stays
           within 0.5x .. 2x of the median ratio (the template's scale)
exponent   untouched — `checks.weight_bounds` already bounds it
```

Normalising the term to its range and bounding the weight is the user's
"normalise the correction features" (2026-10-06), applied to what is added
to the score rather than to the raw feature — rules take log2 of features,
multiply them, and a feature-level rescale would change those meanings.

⚠️ Reads feature values and the rule only — **no times**. The caller passes
the **training** shapes (or the inner-CV training parts).
"""

from __future__ import annotations

import numpy as np

from kernelrule.rules.checks import exponent_indices
from kernelrule.rules.time_term import main_weight_indices

__all__ = ["CapUnavailable", "MAIN_BAND", "shape_range", "main_ranges",
           "term_caps"]

#: A weight on the time estimate itself stays within this band of the
#: template's scale (10 per log2 of time).
MAIN_BAND = (0.5, 2.0)
#: Within one shape: the p1..p99 range of a term over the configs.
_WITHIN = (1.0, 99.0)
#: Across shapes: the low quantile of R(main) / R(term) — a near-minimum
#: that one odd shape cannot set alone.
_RATIO_Q = 10.0


class CapUnavailable(ValueError):
    """The registry has no time estimate to measure against."""


def shape_range(col) -> float:
    """R_s: the p1..p99 range of one shape's column, or its full range when
    that is flat (a term that moves only a few configs)."""
    v = np.asarray(col, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0
    lo, hi = np.percentile(v, _WITHIN)
    r = float(hi - lo)
    return r if r > 0 else float(np.ptp(v))


def main_ranges(matrix, shapes) -> np.ndarray:
    """R_s(main) per shape: the canonical time estimate, in score units."""
    need = ("tm_crit_ratio", "tm_l2_ratio", "tm_dram_ratio")
    names = set(matrix.feature_names())
    if not set(need) <= names:
        raise CapUnavailable(
            f"term caps measure against the time estimate, and this library "
            f"lacks {sorted(set(need) - names)} (condition F4 carries them)")
    out = []
    for p in shapes:
        f, _info = matrix.for_shape(p)
        sm = np.maximum(np.asarray(f.tm_crit_ratio, np.float64),
                        np.asarray(f.tm_l2_ratio, np.float64))
        t = np.sqrt(np.square(sm)
                    + np.square(np.asarray(f.tm_dram_ratio, np.float64)))
        out.append(shape_range(10.0 * np.log2(np.maximum(t, 1e-300))))
    return np.asarray(out)


def term_caps(score_fn, code: str, w, matrix, shapes, *, lam: float
              ) -> list[tuple[float, float]]:
    """Per-weight `(lo, hi)` for `fit_weights(caps=...)`; ±inf where nothing
    is capped (exponent slots)."""
    if not lam > 0:
        raise ValueError(f"lam must be positive, got {lam}")
    w = np.asarray(w, dtype=np.float64)
    shapes = list(shapes)
    r_main = main_ranges(matrix, shapes)
    main = main_weight_indices(code)
    expo = exponent_indices(code)
    hw = matrix.hw
    inf = float("inf")
    views = [matrix.for_shape(p) for p in shapes]
    caps: list[tuple[float, float]] = []
    for i in range(w.size):
        if i in expo:
            caps.append((-inf, inf))
            continue
        on, off = w.copy(), w.copy()
        on[i], off[i] = 1.0, 0.0
        ratios = []
        for (f, info), rm in zip(views, r_main, strict=True):
            a = np.asarray(score_fn(f, info, hw, on), dtype=np.float64)
            b = np.asarray(score_fn(f, info, hw, off), dtype=np.float64)
            if a.shape != b.shape or a.ndim != 1:
                continue
            ri = shape_range(a - b)
            if ri > 0 and rm > 0:
                ratios.append(rm / ri)
        if not ratios:
            caps.append((0.0, 0.0))             # never moves a shape
        elif i in main:
            med = float(np.median(ratios))
            caps.append((MAIN_BAND[0] * med, MAIN_BAND[1] * med))
        else:
            b = lam * float(np.percentile(ratios, _RATIO_Q))
            caps.append((-b, b))
    return caps
