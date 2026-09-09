"""★ Can the features separate the candidates within the top 100 — **the
ceiling**. 0 LLM calls.

    python3 experiments/top100_ceiling.py

## Why it is measured **before** the evolution

However well the loss is designed, there is a wall it cannot cross.

```
the feature values are nearly the same within the top 100
                                    ★ no loss can learn it
the feature values differ within the top 100
                                    it can be learned — designing the loss
                                    is worth it
```

And this value becomes **the ground for reading the verdict**. When tau does
not rise, it separates "is the loss bad" from "can the features not measure
it".

## ★ A shape-level feature cannot enter the ceiling

```
the 5 p. features   they are **constant** within one shape -> 0 contribution
                    to separating candidates
                    they are only used as a branch,
                    `np.where(p.is_memory_bound, ...)`
the 19 f. features  ★ this is all of it
```

⚠️ 2026-09-08 (D-146): the two block labels were translated together with the
top-level keys of `top100-ceiling.json`.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
TOP_N = 100
AXES = ("tile_m", "tile_n", "tile_k", "ext_stages", "split_k",
        "ext_warp_m", "ext_warp_n", "ext_warp_k", "ext_swizzle_n",
        "split_k_mode")


def _splits(table: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def _report(label, table, matrix, shapes, out: dict) -> None:
    feats = REGISTRY.names(shape_level=False)
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")

    tops = {}
    for p in shapes:
        t = np.asarray(table.times_of(p))
        tops[p.key] = np.argsort(t, kind="stable")[:TOP_N]

    # -- the times themselves ---------------------------------------------
    span, uniq_t = [], []
    for p in shapes:
        t = np.asarray(table.times_of(p))[tops[p.key]]
        span.append(float(t.max() / t.min() - 1.0))
        uniq_t.append(int(len(np.unique(t))))
    print(f"  {len(shapes)} shapes, the top {TOP_N}")
    print(f"  the time span (worst/best - 1)   median {np.median(span):.1%}  "
          f"range {min(span):.1%}~{max(span):.1%}")
    print(f"  distinct time values             median "
          f"{int(np.median(uniq_t))}  "
          f"range {min(uniq_t)}~{max(uniq_t)}")

    # -- do the f. features separate --------------------------------------
    print(f"\n  ★ the {len(feats)} config-level features — within the top "
          f"{TOP_N}")
    print(f"  {'feature':24s} {'uniq median':>12} {'cv median':>11} "
          f"{'constant in':>12}")
    frows = {}
    for name in feats:
        u, cv, const = [], [], 0
        for p in shapes:
            # ★ The per-shape column is used directly. The global `column()`
            #   is concatenated, so the shape boundaries would have to be
            #   recomputed, and then the definition would be in two places.
            f_, _info = matrix.for_shape(p)
            v = np.asarray(getattr(f_, name))[tops[p.key]]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            nu = int(len(np.unique(v)))
            u.append(nu)
            m = float(np.mean(np.abs(v)))
            cv.append(float(np.std(v) / m) if m > 1e-12 else 0.0)
            const += int(nu <= 1)
        frows[name] = {"uniq_median": float(np.median(u)),
                       "cv_median": float(np.median(cv)),
                       "n_const": const}
        print(f"  {name:24s} {np.median(u):12.1f} {np.median(cv):11.4f} "
              f"{const:9d}/{len(shapes)}")

    n_dead = sum(1 for v in frows.values() if v["uniq_median"] <= 1.0)
    n_flat = sum(1 for v in frows.values() if v["cv_median"] < 0.01)
    print(f"\n  ★ features that are **constant** at the median within the top "
          f"{TOP_N}: {n_dead}/{len(feats)}")
    print(f"  ★ features with a cv under 1% (effectively flat): "
          f"{n_flat}/{len(feats)}")

    # -- the distribution of the config axes -------------------------------
    print(f"\n  ★ the config axes — how many distinct values within the top "
          f"{TOP_N} (median per shape)")
    arows = {}
    for ax in AXES:
        u = []
        for p in shapes:
            df = table.frame_for(p)
            if ax not in df:
                continue
            v = df[ax].to_numpy()[tops[p.key]]
            u.append(int(len(np.unique(v))))
        if u:
            arows[ax] = float(np.median(u))
            print(f"    {ax:16s} {np.median(u):5.1f} kinds")

    # -- the axis distance between the optimum and places 2~10 -------------
    dist = []
    for p in shapes:
        df = table.frame_for(p)
        cols = [ax for ax in AXES if ax in df]
        arr = np.stack([df[c].to_numpy().astype(str) for c in cols], axis=1)
        top = tops[p.key]
        base = arr[top[0]]
        for r in top[1:10]:
            dist.append(int((arr[r] != base).sum()))
    print(f"\n  ★ **how many axes** the optimum and places 2~10 differ in: "
          f"median {int(np.median(dist))}  range {min(dist)}~{max(dist)}  "
          f"(out of {len(AXES)} axes)")
    print(f"     0 axes different (identical) "
          f"{sum(1 for d in dist if d == 0)}"
          f"/{len(dist)} pairs")

    out[label] = {"n_shapes": len(shapes), "top_n": TOP_N,
                  "span_median": float(np.median(span)),
                  "uniq_time_median": int(np.median(uniq_t)),
                  "features": frows, "axes": arows,
                  "axis_dist_median": int(np.median(dist)),
                  "n_const_features": n_dead, "n_flat_features": n_flat}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/top100-ceiling.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    out: dict = {}
    for lbl, (d, h) in (("A6000 training 41 shapes", A6000),
                        ("5090 training 41 shapes", G5090)):
        T = PerfTable.from_bundle(d, env_hash=h, ok_only=False)
        M = FeatureMatrix(T, REGISTRY)
        _report(lbl, T, M, list(_splits(T).train.shapes), out)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
