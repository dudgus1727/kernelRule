"""★ Does the answer fall in the top k% of the (a) full-transplant rule? 0 LLM
calls.

    python3 experiments/hit_at_k.py

## Why

**It is the upper bound on whether config-axis sampling is possible.** So far
only the shape axis has been shrunk (`refit_sample.py`), and shrinking the
config axis is far cheaper.

```
12 shapes x all configs     230k jobs
41 shapes x 5% of configs   39k jobs   ★ and 3.4x the shape diversity
```

## What is measured — two things. **The second matters more**

```
the fraction of shapes whose **true optimum** is in the top k%
the fraction where at least one member of the **answer set** (within 2σ of
the noise floor) is in the top k%
```

If even one answer-set member is in, the best can be chosen for that shape.
And this table has many ties (the 5090's answer set is 9 at the median and up
to 724).

## What it is compared against

```
(a) the 6 rules   the A6000 structure + the A6000 weights. ★ only the hw
                  constants are the 5090's
static top-k      ★ fixed axis coordinates chosen on the A6000 table — usable
                  straight away on a new GPU
random            the floor
vendor            ⛔ nvMatmulHeuristics is not in this environment. See the
                  comment below
```
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.crosstable import AXIS_FIELDS
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
SRC_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
PCTS = (0.5, 1.0, 2.0, 5.0, 10.0)


def _splits(table: PerfTable):
    from kernelrule.core.splits import SplitSet

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(train=Split("train", tuple(p for p in shapes
                                               if p not in held)),
                    val=Split("val", tuple(held)), kind="nk11008")


def _axis_keys(table: PerfTable, p) -> list[tuple]:
    """★ It returns a list. Building it with `np.array(..., dtype=object)`
    flattens the tuples into a 2-D array and **they stop being hashable** —
    that really happened."""
    df = table.frame_for(p)
    cols = [df[f].to_numpy() for f in AXIS_FIELDS]
    return [tuple(c[i] for c in cols) for i in range(len(df))]


def _hits(order: np.ndarray, table: PerfTable, p, pcts) -> dict:
    """`order` is the order it judged good (indices). It counts the hits
    within the top k%."""
    t = table.times_of(p)
    best = int(np.argmin(t))
    ans = np.flatnonzero(table.answer_mask(p))
    n = len(t)
    out = {}
    for q in pcts:
        k = max(1, int(np.ceil(n * q / 100.0)))
        top = set(order[:k].tolist())
        out[q] = (best in top, bool(top & set(ans.tolist())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/hit-at-k.json")
    ap.add_argument("--weights", choices=("a6000", "refit"), default="a6000",
                    help="a6000 = (a) the full transplant / refit = ★ (b) the "
                         "5090 refit. Comparing the two separates 'does the "
                         "ranking ability transfer'")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    A = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    B = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    mA, mB = FeatureMatrix(A, REGISTRY), FeatureMatrix(B, REGISTRY)
    spA, spB = _splits(A), _splits(B)
    shapes = list(spB.val.shapes)      # ★ it is measured on the holdout

    print("=" * 78)
    print("hit@k% of the (a) full-transplant rule  —  the upper bound of "
          "config sampling")
    print("=" * 78)
    n_c = [len(B.times_of(p)) for p in shapes]
    print(f"  5090 holdout {len(shapes)} shapes   candidates, median "
          f"{int(np.median(n_c))}")
    print(f"  answer-set size, median "
          f"{int(np.median([int(B.answer_mask(p).sum()) for p in shapes]))}")
    print("  ⛔ vendor: nvMatmulHeuristics is not in this environment (the "
          "import fails).")
    print("     A 5090 vendor recommendation cannot be made, so it is left "
          "out of this table — 'absent' is not written down as 'bad'\n")

    res: dict = {"pcts": list(PCTS), "arms": {}}

    def record(name: str, orders: dict) -> None:
        rows = {q: [0, 0] for q in PCTS}
        for p in shapes:
            h = _hits(orders[p.key], B, p, PCTS)
            for q in PCTS:
                rows[q][0] += int(h[q][0])
                rows[q][1] += int(h[q][1])
        res["arms"][name] = {str(q): {"best": rows[q][0],
                                      "answer": rows[q][1],
                                      "n": len(shapes)} for q in PCTS}
        print(f"  {name:28s} " + "  ".join(
            f"{q}%: {rows[q][1]}/{len(shapes)}={rows[q][1] / len(shapes):.0%}"
            for q in PCTS))

    # -- (a) the 6 full-transplant rules -----------------------------------
    print("  ★ the fraction of shapes with at least one answer-set member in "
          "the top k%")
    print("  " + "-" * 74)
    for run in SRC_RUNS:
        f = Path("runs") / run / "archive.jsonl"
        e = sorted((json.loads(x) for x in f.read_text().splitlines()
                    if x.strip()), key=lambda z: z["regret"])[0]
        fn = compile_rule(e["code"])
        ws = {}
        for nm in ("short", "long"):
            if a.weights == "a6000":   # (a) — the weights fitted on the A6000
                g = [q for q in spA.train.shapes if regime_of(q, A.hw) == nm]
                ws[nm] = fit_weights(fn, mA, A, Split("train", tuple(g)),
                                     e["w"], max_evals=300,
                          objective="regret").w
            else:                      # ★ (b) — refitted on the 5090 training
                g = [q for q in spB.train.shapes if regime_of(q, B.hw) == nm]
                ws[nm] = fit_weights(fn, mB, B, Split("train", tuple(g)),
                                     e["w"], max_evals=300,
                          objective="regret").w
        orders = {}
        for p in shapes:
            cand = B.candidates(p)
            sc = np.asarray(make_score_of(fn, mB, ws[regime_of(p, B.hw)])(
                p, cand), dtype=float)
            orders[p.key] = np.argsort(sc, kind="stable")
        tag = "(a)" if a.weights == "a6000" else "(b)"
        record(f"{tag} {run.split('-')[-1]}", orders)

    # -- static top-k: fixed axis coordinates chosen on the A6000 table -----
    # ★ It joins on the axis coordinates. kernel_id is compiled differently
    #   per architecture
    rank_a: dict = {}
    for p in spA.train.shapes:
        t = A.times_of(p)
        r = t / t.min()
        for key, v in zip(_axis_keys(A, p), r, strict=True):
            rank_a.setdefault(key, []).append(float(v))
    score_a = {k: float(np.exp(np.mean(np.log(v)))) for k, v in rank_a.items()}
    orders = {}
    for p in shapes:
        keys = _axis_keys(B, p)
        # Coordinates absent from the A6000 go to the very back (they are not
        # silently dropped)
        s = np.array([score_a.get(k, 1e9) for k in keys], dtype=float)
        orders[p.key] = np.argsort(s, kind="stable")
    record("static (A6000 fixed config)", orders)

    # -- the random floor ---------------------------------------------------
    rng = np.random.default_rng(0)
    orders = {p.key: rng.permutation(len(B.times_of(p))) for p in shapes}
    record("random", orders)

    # -- the fraction where the true optimum lands (strict) -----------------
    print("\n  ★ the fraction of shapes with the **true optimum** in the "
          "top k% (strict)")
    print("  " + "-" * 74)
    for name, d in res["arms"].items():
        print(f"  {name:28s} " + "  ".join(
            f"{q}%: {d[str(q)]['best']}/{len(shapes)}"
            f"={d[str(q)]['best'] / len(shapes):.0%}" for q in PCTS))

    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
