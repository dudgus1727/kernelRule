"""★ The top-rank ordering ability — measured again **with the resolution
confound controlled**. 0 LLM calls.

    python3 experiments/top_tau.py

## Why it is measured again

`degeneracy.py` gave 0.141 for the tau-b within the true top 100. The
pre-registration's decision line is `>= 0.30 it ranks / <= 0.10 it cannot`,
so **0.141 is "in between"** — it must not be written down as "it cannot".

And **there is a confound.**

```
the distinct time values in the true top 100: 24 at the median, ★ 10 or
fewer in 10/41 shapes
the A6000 tick is 1.024 µs — at the top the resolution dominates
-> a low tau does not separate the rule's fault from the tick's
```

## The control

```
tau only on the shapes with N or more distinct time values in the top 100
(N = 30, 50)
★ the number of shapes left is printed alongside
★ so is the random floor of that subset (20 draws — the floor is a sample
  too, principle 7)
```

## ★ It is measured on the 5090 too — the confound is far weaker there

```
the 5090 tick is 16 ns  vs  the A6000's 1.024 µs   (1/64)
```

⚠️ The tables differ, so the comparison is **against the random floor, not on
absolute values** (principle 4). The answer-set size differs too (A6000
median 5 / 5090 median 11).

⚠️ 2026-09-08 (D-146): **the three block labels stay in Korean.** They are the
top-level keys of `top-tau.json`, and `docs/` is not translated.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
SRC_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
TOP_N = 100
MIN_UNIQ = (1, 30, 50)      # 1 = no control (only what is defined)
N_DRAWS = 20


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


def _uniq_top(table, p) -> int:
    t = np.asarray(table.times_of(p))
    return int(len(np.unique(t[np.argsort(t, kind="stable")[:TOP_N]])))


def _tau_top(sc, t) -> float:
    top = np.argsort(t, kind="stable")[:TOP_N]
    if len(np.unique(t[top])) < 2:
        return float("nan")
    return float(kendalltau(sc[top], t[top], variant="b").statistic)


def _one(label, table, matrix, fit_shapes, eval_shapes, out: dict) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    uniq = np.array([_uniq_top(table, p) for p in eval_shapes])
    print(f"  {len(eval_shapes)} shapes   the distinct time values in the top "
          f"{TOP_N}: median {int(np.median(uniq))}  "
          f"range {uniq.min()}~{uniq.max()}")
    print(f"  the tick is {table.noise.tick_ms} ms")
    print(f"\n  {'structure':22s} " + "  ".join(
        f"{'no control' if n == 1 else f'uniq>={n}':>12}" for n in MIN_UNIQ))

    rows: dict = {}
    for run in SRC_RUNS:
        f = Path("runs") / run / "archive.jsonl"
        e = sorted((json.loads(x) for x in f.read_text().splitlines()
                    if x.strip()), key=lambda z: z["regret"])[0]
        fn = compile_rule(e["code"])
        ws = {}
        for nm in ("short", "long"):
            g = [q for q in fit_shapes if regime_of(q, table.hw) == nm]
            ws[nm] = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                                 e["w"], max_evals=300,
                          objective="regret").w
        taus = {}
        for p in eval_shapes:
            cand = table.candidates(p)
            sc = np.asarray(make_score_of(fn, matrix, ws[regime_of(
                p, table.hw)])(p, cand), dtype=float)
            taus[p.key] = _tau_top(sc, np.asarray(table.times_of(p)))
        row = {}
        for n in MIN_UNIQ:
            v = [taus[p.key] for p, u in zip(eval_shapes, uniq, strict=True)
                 if u >= n and np.isfinite(taus[p.key])]
            row[n] = (float(np.median(v)) if v else float("nan"), len(v))
        rows[run] = row
        print(f"  {run:22s} " + "  ".join(
            f"{row[n][0]:12.3f}" for n in MIN_UNIQ))

    # ★ The random floor — on the same subset, 20 draws
    rng = np.random.default_rng(0)
    floor: dict = {n: [] for n in MIN_UNIQ}
    for _ in range(N_DRAWS):
        taus = {}
        for p in eval_shapes:
            t = np.asarray(table.times_of(p))
            taus[p.key] = _tau_top(rng.random(len(t)), t)
        for n in MIN_UNIQ:
            v = [taus[p.key] for p, u in zip(eval_shapes, uniq, strict=True)
                 if u >= n and np.isfinite(taus[p.key])]
            if v:
                floor[n].append(float(np.median(v)))
    print(f"  {'★ random floor':22s} " + "  ".join(
        f"{np.mean(floor[n]):12.3f}" if floor[n] else f"{'—':>12}"
        for n in MIN_UNIQ))
    print(f"  {'shapes left':22s} " + "  ".join(
        f"{rows[SRC_RUNS[0]][n][1]:12d}" for n in MIN_UNIQ))

    med = {n: float(np.median([rows[r][n][0] for r in SRC_RUNS
                               if np.isfinite(rows[r][n][0])]))
           for n in MIN_UNIQ}
    print("\n  ★ the median of the 6 structures: " + "   ".join(
        f"{'no control' if n == 1 else f'uniq>={n}'} {med[n]:.3f}"
        for n in MIN_UNIQ))
    out[label] = {"rows": {r: {str(n): rows[r][n] for n in MIN_UNIQ}
                           for r in SRC_RUNS},
                  "floor": {str(n): (float(np.mean(floor[n]))
                                     if floor[n] else None)
                            for n in MIN_UNIQ},
                  "median": {str(n): med[n] for n in MIN_UNIQ},
                  "n_shapes": len(eval_shapes),
                  "uniq_median": int(np.median(uniq))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/top-tau.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    out: dict = {}

    A = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    mA = FeatureMatrix(A, REGISTRY)
    spA = _splits(A)
    # ★ Exactly the pre-registration condition (the training 41) — it
    #   continues from the earlier numbers
    _one("A6000 training 41 shapes (the pre-registration condition)", A, mA,
         list(spA.train.shapes), list(spA.train.shapes), out)
    # For comparison across tables — holdout against holdout
    _one("A6000 holdout 20 shapes", A, mA,
         list(spA.train.shapes), list(spA.val.shapes), out)

    B = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    mB = FeatureMatrix(B, REGISTRY)
    spB = _splits(B)
    # ★ The (b) refit weights — fitted on the 5090 training split
    _one("5090 holdout 20 shapes — (b) refit", B, mB,
         list(spB.train.shapes), list(spB.val.shapes), out)

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ the tables differ — compare **against the random floor**, not "
          "on absolute values (principle 4)")


if __name__ == "__main__":
    main()
