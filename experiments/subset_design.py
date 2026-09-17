"""★ Is the problem the sample size or the sample **selection**? 0 LLM calls.

    python3 experiments/subset_design.py --workers 6

The **pre-registration** is `docs/artifacts/subset-design-prereg.md`

## A random draw measures the worst case

`refit_sample.py` stratifies by regime and then draws **at random**. In
practice **we decide** which shapes to measure on a new GPU — designing the
shape grid is what kernelTab does every campaign.

## ★ Only what can be computed before the table is measured is used

```
used       the SOL lower bound / arith_intensity / the position relative to
           ridge / the layer label
★ not used difficulty / best_ms / distinct_time_frac — they need the table
           to be measured
```

Choosing by a table-derived value **is itself a leak** and breaks the
"choosing without a table" scenario. `_features_without_table` below is that
boundary.

## The strategies

```
sol        it covers the log(SOL) quantiles evenly within each regime
ridge      ★ nearest to the ridge boundary first — that is where the
           classification flips
sol+ridge  half and half
layer      evenly from kernelTab's layers (★ d_alignment has 0 in the
           training 41)
```
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


#: ⛔ 2026-09-17 (D-179) — the SOL lower-bound value this script used was
#: **removed from the code**. The 0.5 ms boundary was ours, chosen by looking
#: at the a6000 table, and the scoring path used it to fit two weight vectors
#: while the loop evolved one. See `docs/decisions.md` D-179.
#:
#: ⚠️ This script is kept because the numbers it produced are on the record
#: and this is how they were produced. ⛔ It **cannot be re-run** — and it
#: says so rather than quietly substituting another cut, which would put
#: different numbers under the same name.
def _sol_removed(where: str):
    raise SystemExit(
        f"{where}: the SOL lower-bound value was removed at D-179. This "
        f"script cannot be re-run. Its recorded numbers stay in "
        f"docs/decisions.md; ⛔ do not substitute another boundary.")


G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
SRC_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
KS = (2, 4, 8, 10, 12)
STRATEGIES = ("sol", "ridge", "sol+ridge", "layer")
#: The catastrophe threshold. Nailed down in the pre-registration.
CATASTROPHE = 1.15

_W: dict = {}


def _splits(table: PerfTable) -> SplitSet:
    shapes = experiment_shapes(table)
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def _features_without_table(p, hw) -> dict:
    """★ Only values computable **before the table is measured**. This is the
    boundary of this experiment.

    `difficulty` / `best_ms` / `distinct_time_frac` are not here — they need
    the table to be known, and using them is a leak (pre-registration §3-2).
    """
    from kernelrule.core.splits import _DUMMY_CFG
    from kernelrule.features.physical import arith_intensity

    _sol_removed("subset_design._axes")
    ai = arith_intensity(p, hw, _DUMMY_CFG)
    return {"log_sol": None,
            "arith_intensity": ai,
            # The position relative to ridge. The closer to 1, the closer to
            # the regime boundary
            "roofline_ratio": ai / hw.ridge_point}


def _spread_pick(vals: np.ndarray, n: int) -> list[int]:
    """n values covering the `vals` distribution evenly. **The nearest to each
    quantile centre.**

    It is deterministic — the same input gives the same answer. On a tie, the
    smaller index.
    """
    if n >= len(vals):
        return list(range(len(vals)))
    qs = [(i + 0.5) / n for i in range(n)]
    targets = np.quantile(vals, qs)
    out: list[int] = []
    for t in targets:
        order = np.argsort(np.abs(vals - t), kind="stable")
        for j in order:
            if int(j) not in out:
                out.append(int(j))
                break
    return out


def _select(strategy: str, k: int) -> list:
    """k by the strategy. **The per-regime stratification is a condition
    shared by both arms.**"""
    by_regime, feats, layers = _W["by_regime"], _W["feats"], _W["layers"]
    names = sorted(by_regime)
    sizes = np.array([len(by_regime[n]) for n in names], dtype=float)
    take = np.ones(len(names), dtype=int)
    left = k - len(names)
    extra = np.floor(sizes / sizes.sum() * left).astype(int)
    for i in range(left - int(extra.sum())):
        extra[i % len(names)] += 1
    take = take + extra

    out = []
    for name, want in zip(names, take, strict=True):
        pool = by_regime[name]
        n = int(min(want, len(pool)))
        if strategy == "layer":
            # One at a time round the layers. Within a layer, the nearest to
            # the SOL centre
            buckets: dict = {}
            for p in pool:
                buckets.setdefault(layers.get((p.M, p.N, p.K), "?"), []).append(p)
            keys = sorted(buckets)
            picked: list = []
            r = 0
            while len(picked) < n:
                b = buckets[keys[r % len(keys)]]
                rest = [q for q in b if q not in picked]
                if rest:
                    v = np.array([feats[q.key]["log_sol"] for q in rest])
                    picked.append(rest[_spread_pick(v, 1)[0]])
                r += 1
                if r > 100 * n:
                    break
            out.extend(picked)
            continue
        if strategy == "ridge":
            # Nearest to the boundary first (ascending
            # |roofline_ratio - 1|)
            d = np.array([abs(feats[q.key]["roofline_ratio"] - 1.0)
                          for q in pool])
            idx = list(np.argsort(d, kind="stable")[:n])
        elif strategy == "sol":
            v = np.array([feats[q.key]["log_sol"] for q in pool])
            idx = _spread_pick(v, n)
        elif strategy == "sol+ridge":
            n_r = n // 2
            d = np.array([abs(feats[q.key]["roofline_ratio"] - 1.0)
                          for q in pool])
            idx = list(np.argsort(d, kind="stable")[:n_r])
            v = np.array([feats[q.key]["log_sol"] for q in pool])
            for j in _spread_pick(v, n):
                if len(idx) >= n:
                    break
                if j not in idx:
                    idx.append(j)
        else:
            raise ValueError(strategy)
        out.extend(pool[int(i)] for i in idx)
    return out


def _job(arg):
    ri, k, strategy = arg
    e = _W["rules"][ri]
    table, matrix, hold = _W["table"], _W["matrix"], _W["hold"]
    sample = _select(strategy, k)
    fn = compile_rule(e["code"])
    ws, fits = {}, []
    for name in ("short", "long"):
        g = [p for p in sample if regime_of(p, table.hw) == name]
        if not g:
            return {"ri": ri, "k": k, "strategy": strategy,
                    "holdout": float("nan"), "empty_regime": name}
        fr = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                         e["w"], max_evals=300,
                          objective="regret")
        ws[name] = fr.w
        fits.append(fr.fit_regret)
    regs = []
    for p in hold:
        cand = table.candidates(p)
        sc = make_score_of(fn, matrix, ws[regime_of(p, table.hw)])(p, cand)
        t = table.times_of(p)
        regs.append(float(t[cand.top_k(sc, 1)[0]] / t.min()))
    return {"ri": ri, "run": _W["runs"][ri], "k": k, "strategy": strategy,
            "n_sample": len(sample),
            "shapes": [[p.M, p.N, p.K] for p in sample],
            "fit_regret": float(np.exp(np.mean(np.log(fits)))),
            "holdout": float(geomean(np.array(regs)))}


def _wilson_hi(k: int, n: int, z: float = 1.96) -> float:
    """The 95% upper bound, so that 0/n is not used as 'no catastrophe'
    (principle 27)."""
    if n == 0:
        return 1.0
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    r = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return float(min(1.0, (c + r) / d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="docs/artifacts/subset-design.json")
    ap.add_argument("--random-json", default="docs/artifacts/refit-sample.json",
                    help="the random arm. Only the same k are compared")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    table = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    sp = _splits(table)
    train = list(sp.train.shapes)

    # ★ The join key is matched up. `p.key` is (M,N,K,dtype) and
    #   `shape_layers` is [M,N,K] — putting it in as is matches **nothing**.
    layers: dict = {}
    for name, shs in (table.meta.get("shape_layers") or {}).items():
        for sh in shs:
            layers[(sh[0], sh[1], sh[2])] = name
    feats = {p.key: _features_without_table(p, table.hw) for p in train}
    by_regime: dict = {}
    for p in train:
        by_regime.setdefault(regime_of(p, table.hw), []).append(p)

    rules = []
    for r in SRC_RUNS:
        f = Path("runs") / r / "archive.jsonl"
        arc = sorted((json.loads(x) for x in f.read_text().splitlines()
                      if x.strip()), key=lambda e: e["regret"])
        rules.append(arc[0])

    _W.update(table=table, matrix=matrix, train=train, by_regime=by_regime,
              hold=list(sp.val.shapes), rules=rules, runs=SRC_RUNS,
              feats=feats, layers=layers)

    print("=" * 78)
    print("sample size or sample selection — the designed subset")
    print("=" * 78)
    print(f"  5090 training {len(train)} shapes  "
          + "  ".join(f"{n} {len(v)}" for n, v in sorted(by_regime.items())))
    lay_n: dict = {}
    for p in train:
        key = layers.get((p.M, p.N, p.K), "?")
        lay_n[key] = lay_n.get(key, 0) + 1
    print(f"  layer distribution {dict(sorted(lay_n.items()))}")
    # ★ If the labels do not attach, the `layer` strategy becomes **drawing
    #   from one bucket**. The numbers come out but the strategy did not run
    #   (principle 1). It stops.
    if lay_n.get("?", 0):
        raise SystemExit(
            f"★ {lay_n['?']} shapes have no layer label. Check the "
            "`shape_layers` join key — `p.key` is (M,N,K,dtype) and the "
            "bundle is [M,N,K].\n  Leaving it silent makes the `layer` "
            "strategy draw from one bucket while the result still looks "
            "plausible.")
    absent = sorted(set(table.meta.get("shape_layers") or {}) - set(lay_n))
    if absent:
        print(f"  ★ layers **absent** from the training 41 shapes: {absent}  "
              "— 'layer-even' means even over the remaining layers")
    print(f"  {len(rules)} structures   strategies {list(STRATEGIES)}")
    print("  ★ the design arm is deterministic — there is no draw luck\n")

    jobs = [(ri, k, st) for ri in range(len(rules)) for k in KS
            for st in STRATEGIES]
    if a.workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=a.workers,
                                 mp_context=get_context("fork")) as ex:
            out = list(ex.map(_job, jobs, chunksize=2))
    else:
        out = [_job(j) for j in jobs]

    rnd: dict = {}
    rp = Path(a.random_json)
    if rp.exists():
        for row in json.loads(rp.read_text())["rows"]:
            rnd[row["k"]] = row["values"]

    print(f"  {'k':>4} {'strategy':>10} {'median':>8} {'worst':>8} "
          f"{'catastr':>8} {'95% hi':>8}   {'random catastr':>15}")
    rows = []
    for k in KS:
        for st in STRATEGIES:
            v = np.array([r["holdout"] for r in out
                          if r["k"] == k and r["strategy"] == st
                          and np.isfinite(r["holdout"])])
            if not len(v):
                continue
            nc = int((v > CATASTROPHE).sum())
            hi = _wilson_hi(nc, len(v))
            rv = np.array(rnd.get(k, []))
            rtxt = (f"{(rv > CATASTROPHE).mean():14.0%}" if len(rv)
                    else f"{'—':>14}")
            print(f"  {k:4d} {st:>10} {np.median(v):8.4f} {v.max():8.4f} "
                  f"{nc:3d}/{len(v):<4d} {hi:8.0%}   {rtxt}")
            rows.append({"k": k, "strategy": st, "median": float(np.median(v)),
                         "max": float(v.max()), "n_catastrophe": nc,
                         "n": len(v), "wilson_hi": hi,
                         "values": [float(x) for x in v]})
        print()

    print("  ⚠️ the design arm has only 6 cases per k (6 structures x 1 "
          "draw). 0/6 is not 'no catastrophe' but **a 95% upper bound of "
          "39%** (principle 27).")
    print("  ⚠️ no significance is attached.")
    Path(a.out).write_text(json.dumps(
        {"bundle": G5090[0], "n_train": len(train), "src_runs": SRC_RUNS,
         "strategies": list(STRATEGIES), "catastrophe": CATASTROPHE,
         "layer_counts": lay_n, "rows": rows, "raw": out},
        ensure_ascii=False, indent=1))
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
