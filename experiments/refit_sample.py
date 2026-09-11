"""★ The **cost** claim of §29.5(b) — how many % of the sample fits the
weights? 0 LLM calls.

    python3 experiments/refit_sample.py --workers 6

## What the claim is

> If §29.5(b) holds, **"15 hours of table" becomes "5% of a sample + a few
> seconds"**.

`transfer_29_5.py` refitted with **all 41 shapes** of the training split.
That measured **"does the structure transfer"**, and what is measured here is
**"is that cheap"**. They are different questions.

## How it is measured

```
k shapes are drawn from the 5090's 41 training shapes and the per-regime
weights are fitted **from those alone**
-> scored on the 5090's 20 structural holdout shapes  (the holdout is always
   all of them)
k = 2(5%) 4 8 12 20 41(100%)
```

★ **Which k are drawn is also a result.** The draw is repeated many times to
give a distribution — "5% is enough" must not be one lucky draw.

## ★ The nature of the catastrophe is recorded too (2026-09-01)

The worst was 2.35 — that is not "a bit bad", something broke. The response
differs, so the cause is recorded.

```
good on the fitting set but bad on the holdout  -> ★ overfitting. It is a
                                                   sample problem. Raising
                                                   the fitting budget does
                                                   not help
bad on both                                     -> the fitter did not find it
```

★ And **the shapes / parameters ratio** is looked at alongside. `k=2` is 1
shape per regime against 8 weights — however they are chosen it is
underdetermined. "how many %" may not be the right unit.

★ **The draw is stratified by regime.** The weights are fitted separately per
regime, so if one regime gets 0 the fit does not happen at all. At k=2 it is
1 per regime. **This stratification is a condition, and it has to be done
that way in practice too** — it is not "any 2".

## What is not done

It is not put up against `(c)`. What it is put up against here is **the
41-shape refit of the same structure** — the question is "how much is lost by
shrinking the sample".
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, SplitSet, experiment_shapes, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
SRC_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
#: ★ The vertex is between 8 and 12, so 10/14/16 fill it in (2026-09-01)
KS = (2, 4, 8, 10, 12, 14, 16, 20, 41)

#: The workers inherit it by fork (the 4.2 GB matrix is not copied).
_W: dict = {}


def _splits(table: PerfTable) -> SplitSet:
    shapes = experiment_shapes(table)
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def _stratified(rng, by_regime: dict, k: int) -> list:
    """k evenly across the regimes. It guarantees **at least 1 from each
    regime**."""
    names = sorted(by_regime)
    take = dict.fromkeys(names, 1)
    left = k - len(names)
    if left < 0:
        raise ValueError(f"k={k} is smaller than the number of regimes "
                         f"{len(names)}")
    # The remaining slots go in proportion to the regime sizes
    sizes = np.array([len(by_regime[n]) for n in names], dtype=float)
    extra = np.floor(sizes / sizes.sum() * left).astype(int)
    for i in range(left - int(extra.sum())):
        extra[i % len(names)] += 1
    out = []
    for n, e in zip(names, extra, strict=True):
        pool = by_regime[n]
        idx = rng.choice(len(pool), size=min(len(pool), take[n] + int(e)),
                         replace=False)
        out.extend(pool[i] for i in idx)
    return out


def _job(arg):
    """(rule index, k, draw seed) -> the holdout regret."""
    ri, k, seed = arg
    e = _W["rules"][ri]
    table, matrix = _W["table"], _W["matrix"]
    by_regime, hold = _W["by_regime"], _W["hold"]
    rng = np.random.default_rng(seed)
    sample = (list(_W["train"]) if k >= len(_W["train"])
              else _stratified(rng, by_regime, k))
    fn = compile_rule(e["code"])
    ws, fits, moved = {}, [], []
    for name in ("short", "long"):
        g = [p for p in sample if regime_of(p, table.hw) == name]
        if not g:                    # ★ it is not passed over silently
            return {"ri": ri, "k": k, "seed": seed, "holdout": float("nan"),
                    "empty_regime": name}
        fr = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                         e["w"], max_evals=300,
                          objective="regret")
        ws[name] = fr.w
        fits.append(fr.fit_regret)
        moved.append(bool(fr.moved))
    regs = []
    for p in hold:
        cand = table.candidates(p)
        sc = make_score_of(fn, matrix, ws[regime_of(p, table.hw)])(p, cand)
        t = table.times_of(p)
        regs.append(float(t[cand.top_k(sc, 1)[0]] / t.min()))
    return {"ri": ri, "run": _W["runs"][ri], "k": k, "seed": seed,
            "n_sample": len(sample),
            # ★ The value on the fitting set. Diverging from the holdout is
            #   overfitting
            "fit_regret": float(np.exp(np.mean(np.log(fits)))),
            "moved": all(moved),
            "n_terms": _W["n_terms"][ri],
            "n_weights": len(e["w"]),
            "holdout": float(geomean(np.array(regs)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10,
                    help="how many draws per k. k=41 is deterministic, so 1")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="docs/artifacts/refit-sample.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    table = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    sp = _splits(table)
    train = list(sp.train.shapes)
    by_regime: dict = {}
    for p in train:
        by_regime.setdefault(regime_of(p, table.hw), []).append(p)

    rules = []
    for r in SRC_RUNS:
        f = Path("runs") / r / "archive.jsonl"
        arc = sorted((json.loads(x) for x in f.read_text().splitlines()
                      if x.strip()), key=lambda e: e["regret"])
        rules.append(arc[0])

    # ★ Is the catastrophe related to the number of terms — the question
    #   continues from the budget experiment (8 vs 16)
    from kernelrule.rules.checks import check_rule
    n_terms = []
    for e in rules:
        try:
            n_terms.append(check_rule(
                e["code"], feature_names=REGISTRY.names(shape_level=False),
                shape_value_names=REGISTRY.names(shape_level=True),
                n_weights=len(e["w"])).n_terms)
        except Exception:                                   # noqa: BLE001
            n_terms.append(-1)
    _W.update(table=table, matrix=matrix, train=train, by_regime=by_regime,
              hold=list(sp.val.shapes), rules=rules, runs=SRC_RUNS,
              n_terms=n_terms)

    print("=" * 76)
    print("§29.5(b) the cost — how many % of the sample fits the weights")
    print("=" * 76)
    print(f"  5090 training {len(train)} shapes  "
          + "  ".join(f"{n} {len(v)}" for n, v in sorted(by_regime.items())))
    print(f"  holdout {len(sp.val.shapes)} shapes — **always all of them**")
    print(f"  {len(rules)} structures (the training best of the 6 A6000 F3 "
          f"seeds)")
    print(f"  {a.draws} draws per k, stratified by regime\n")

    jobs = [(ri, k, 1000 * ri + 7 * k + d)
            for ri in range(len(rules)) for k in KS
            for d in range(1 if k >= len(train) else a.draws)]
    if a.workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=a.workers,
                                 mp_context=get_context("fork")) as ex:
            out = list(ex.map(_job, jobs, chunksize=4))
    else:
        out = [_job(j) for j in jobs]

    by_k: dict = {k: [] for k in KS}
    for r in out:
        if np.isfinite(r["holdout"]):
            by_k[r["k"]].append(r["holdout"])

    print(f"  {'k':>4} {'sample%':>8}  {'median':>8} {'range':>19} "
          f"{'width':>8}  n")
    rows = []
    for k in KS:
        v = np.array(by_k[k])
        pct = 100.0 * k / len(train)
        print(f"  {k:4d} {pct:7.1f}%  {np.median(v):8.4f}  "
              f"{v.min():.4f}~{v.max():.4f}  {v.max() - v.min():8.4f}  "
              f"{len(v)}")
        rows.append({"k": k, "pct": pct, "median": float(np.median(v)),
                     "min": float(v.min()), "max": float(v.max()),
                     "n": len(v), "values": [float(x) for x in v]})

    full = np.median(by_k[len(train)]) if by_k.get(len(train)) else float("nan")
    print(f"\n  ★ the loss against the 41-shape (100%) median {full:.4f}")
    for k in KS:
        if k >= len(train):
            continue
        v = np.array(by_k[k])
        print(f"     k={k:2d} ({100.0 * k / len(train):4.1f}%)  "
              f"median +{np.median(v) - full:.4f}   "
              f"worst +{v.max() - full:.4f}")
    # ------------------------------------------------------------------
    # ★ The nature of the catastrophe
    # ------------------------------------------------------------------
    ok = [r for r in out if np.isfinite(r["holdout"])]
    cat = [r for r in ok if r["holdout"] > 1.15]
    print("\n" + "=" * 76)
    print(f"★ catastrophes (holdout > 1.15): {len(cat)} of {len(ok)}")
    print("=" * 76)
    if cat:
        fr = np.array([r["fit_regret"] for r in cat])
        fr_ok = np.array([r["fit_regret"] for r in ok if r not in cat])
        print(f"  the regret on the fitting set   catastrophes median "
              f"{np.median(fr):.4f}   the rest median {np.median(fr_ok):.4f}")
        print("  ★ if the fitting regret on the catastrophe side is **low** "
              "it is overfitting (a sample problem) / if it is high it is a "
              "fitting failure")
        print(f"  the fraction where the fitter moved   catastrophes "
              f"{np.mean([r['moved'] for r in cat]):.0%}"
              f"   the rest "
              f"{np.mean([r['moved'] for r in ok if r not in cat]):.0%}")
        print("\n  catastrophes per structure (are they clustered or "
              "scattered)")
        for i, run in enumerate(SRC_RUNS):
            n_c = sum(1 for r in cat if r["ri"] == i)
            n_o = sum(1 for r in ok if r["ri"] == i)
            print(f"    {run:26s} terms {n_terms[i]:2d}  "
                  f"catastrophes {n_c:3d}/{n_o:3d}"
                  f" = {n_c / max(n_o, 1):5.1%}")
        print("\n  ★ the shape/parameter ratio — this, not 'k', may be the "
              "unit")
        print(f"  {'k':>4} {'shapes per regime (min)':>24} {'weights':>8} "
              f"{'ratio':>7} {'catastr':>8}")
        for k in KS:
            rows_k = [r for r in ok if r["k"] == k]
            if not rows_k:
                continue
            per = k // len(by_regime)
            w = int(np.median([r["n_weights"] for r in rows_k]))
            c = np.mean([r["holdout"] > 1.15 for r in rows_k])
            print(f"  {k:4d} {per:24d} {w:8d} {per / max(w, 1):7.2f} "
                  f"{c:8.0%}")
    print("\n  ⚠️ no significance is attached — the 5090 σ confidence "
          "interval is wide")
    Path(a.out).write_text(json.dumps(
        {"bundle": G5090[0], "n_train": len(train),
         "n_holdout": len(sp.val.shapes), "src_runs": SRC_RUNS,
         "draws": a.draws, "stratified_by_regime": True, "rows": rows,
         "n_terms": n_terms, "raw": out},
        ensure_ascii=False, indent=1))
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
