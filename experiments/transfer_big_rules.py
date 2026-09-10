"""★ Do the **big** rules transfer? (a) transplant / (b) refit. 0 LLM calls.

    python3 -m experiments.transfer_big_rules

★ It is run with `-m` — it imports the tables and the split from
`experiments.transfer_29_5` so that the procedure has one definition
(principle 2), and that needs the repository root on `sys.path`.

D-156 grew the rules — 7 terms at the start, 29 terms and 4 paths at r11.
Whether that moves to another GPU was never measured. Two things make it
worth measuring now:

```
r11 rule    29 terms · 38 weights · 4 paths
target train  ★ 20~27 shapes
-> more parameters than shapes. The refit may overfit, or the fitter may
   not reach at all
```

```
 r    train      val    terms  len(w0)  paths
 5   1.0698   ★ 1.0472    17      21      2    <- best val
11   1.0559     1.0509    29      38      4    <- best train
```

After r5 only the training score improves. ⚠️ One seed — this is a reason to
measure, not a conclusion.

## What this does not do

```
[ ] (c) regrow      already measured, and this must not overwrite it —
                    the output file is its own (`transfer-big-rules.json`)
[ ] (d) seeded      the target-table rounds seeded from (b). Decided after
                    reading this
```

## ⚠️ The fitter differs from `transfer_29_5.py`

That script fits **everything** with Nelder-Mead / 300. Here the fitter is
`fitter_for(len(w0))` — the standing rule since D-144 — because Nelder-Mead
in 38 dimensions measures the fitter, not the transfer (D-77). So **the
numbers here do not go beside the old transfer numbers** (principle 4).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401

# ★ The tables and the split come from `transfer_29_5` — one definition
#   (principle 2).
from experiments.transfer_29_5 import TABLES, _score_on, _splits
from kernelrule.baselines.static_topk import StaticTopK
from kernelrule.core.crosstable import common_shapes
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights
from kernelrule.features import REGISTRY
from kernelrule.rules.checks import check_rule, fitter_for, limits_for

SRC = "a6000"
DSTS = ("5090", "4090", "h100")
OUT = "docs/artifacts/transfer-big-rules.json"


def _rules() -> list[dict]:
    """The three rules on the ladder. **Fixed here, not picked later.**"""
    out = []
    for tag, run, rnd in (("r5 (D-156)", "noregime-s0", 5),
                          ("r11 (D-156)", "noregime-s0", 11)):
        b = [json.loads(x) for x in
             (Path("runs") / run / "bests.jsonl").read_text().splitlines()
             if x.strip()]
        e = [x for x in b if x["round"] <= rnd][-1]
        out.append({"tag": tag, "run": run, "round": e["round"],
                    "code": e["code"], "w": list(e["w"])})
    # ★ The old 8-parameter rule (D-140). One seed is taken, and **which one
    #   is decided without looking at the holdout**: the median of the six by
    #   training regret.
    six = []
    for i in range(6):
        b = [json.loads(x) for x in
             (Path("runs") / f"F3rw-p8-nan-s{i}"
              / "bests.jsonl").read_text().splitlines() if x.strip()]
        e = [x for x in b if x["round"] <= 11][-1]
        six.append((e["regret"], i, e))
    six.sort()
    _, i, e = six[len(six) // 2 - 1]        # the lower median of six
    out.append({"tag": f"old 8p (D-140, s{i})", "run": f"F3rw-p8-nan-s{i}",
                "round": e["round"], "code": e["code"], "w": list(e["w"])})
    return out


def _fit(code, w0, table, matrix, train):
    """Per-regime refit. ★ The fitter follows `fitter_for(len(w0))`
    (D-144), and **whether it moved is reported**."""
    fn = compile_rule(code)
    ws, moved, tr = {}, {}, {}
    fit = fitter_for(len(w0))
    for name in ("short", "long"):
        g = [p for p in train if regime_of(p, table.hw) == name]
        if not g:
            continue
        fr = fit_weights(fn, matrix, table, Split("train", tuple(g)), w0,
                         method=fit["fit_method"],
                         n_restarts=fit["fit_restarts"],
                         max_evals=max(300, fit["max_evals"]),
                         objective="regret")
        ws[name] = fr.w
        moved[name] = bool(fr.moved)
        tr[name] = float(fr.fit_regret)
    return fn, ws, moved, tr, fit["fit_method"]


def _shape_of(code, w) -> dict:
    rep = check_rule(code, feature_names=REGISTRY.names(shape_level=False),
                     shape_value_names=REGISTRY.names(shape_level=True),
                     n_weights=len(w), limits=limits_for())
    return {"terms": rep.n_terms, "weights": rep.n_weights,
            "paths": rep.n_paths, "nodes": rep.n_nodes}


def main() -> None:
    warnings.simplefilter("ignore")
    A = PerfTable.from_bundle(TABLES[SRC]["bundle"],
                              env_hash=TABLES[SRC]["env_hash"], ok_only=False)
    mA, spA = FeatureMatrix(A, REGISTRY), _splits(
        PerfTable.from_bundle(TABLES[SRC]["bundle"],
                              env_hash=TABLES[SRC]["env_hash"],
                              ok_only=False))
    rules = _rules()
    res: dict = {"src": SRC, "rules": [], "targets": {}}
    print("=" * 88)
    print("Do the big rules transfer — (a) transplant / (b) refit. 0 LLM "
          "calls")
    print("=" * 88)
    for r in rules:
        sh = _shape_of(r["code"], r["w"])
        r["shape"] = sh
        res["rules"].append({k: r[k] for k in
                             ("tag", "run", "round", "shape")})
        print(f"  {r['tag']:22s} terms {sh['terms']:2d}  weights "
              f"{sh['weights']:2d}  paths {sh['paths']}  nodes {sh['nodes']}")

    for dst in DSTS:
        D = TABLES[dst]
        B = PerfTable.from_bundle(D["bundle"], env_hash=D["env_hash"],
                                  ok_only=False)
        mB, spB = FeatureMatrix(B, REGISTRY), _splits(B)
        common = {(p.M, p.N, p.K) for p in common_shapes(A, B)}
        hold = [p for p in spB.val.shapes if (p.M, p.N, p.K) in common]
        n_train = len(spB.train.shapes)
        print("\n" + "=" * 88)
        print(f"{SRC} -> {dst}   ridge {A.hw.ridge_point:.1f} -> "
              f"{B.hw.ridge_point:.1f}   SM {A.hw.sm_count} -> "
              f"{B.hw.sm_count}")
        print(f"  holdout {len(hold)} of the {len(common)} common shapes   "
              f"★ target training {n_train} shapes")
        print("=" * 88)
        t: dict = {"n_train": n_train, "n_holdout": len(hold), "cells": []}

        # -- baselines ------------------------------------------------------
        st1 = StaticTopK(B, hold, coverage="union").run(ks=(1,))
        t["static_top1"] = float(st1.by_k[1]["all"])
        print(f"  {'baseline static top-1':30s} {t['static_top1']:.4f}")
        vend = Path(f"datasets/baselines/vendor-{dst}-{D['env_hash']}.json")
        if vend.exists():
            # ★ The same call as `vendor_compare.py` — one procedure
            #   (principle 2).
            from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
            from kernelrule.core.scoring import evaluate_scores
            ev = evaluate_scores(
                vendor_order_fn(B, load_vendor(vend), mapping="nearest"),
                B, hold, ks=(1,), label="vendor")
            t["vendor"] = float(geomean(ev.regret[:, 0]))
            print(f"  {'baseline vendor':30s} {t['vendor']:.4f}")
        else:
            t["vendor"] = None
            print(f"  {'baseline vendor':30s} ★ none for this table")
        from kernelrule.rules.human_guided import CODE as PS_CODE
        from kernelrule.rules.human_guided import W0 as PS_W0
        fnh, wsh, _mh, _trh, _fh = _fit(PS_CODE, list(PS_W0), B, mB,
                                        list(spB.train.shapes))
        t["human_guided_refit"] = float(_score_on(fnh, wsh, B, mB, hold))
        print(f"  {'baseline human_guided refit':30s} "
              f"{t['human_guided_refit']:.4f}")

        print(f"\n  {'rule':22s} {'par/train':>10} {'(a)':>8} {'(b)':>8} "
              f"{'b train':>8} {'gap':>8}  moved  fitter")
        for r in rules:
            fn = compile_rule(r["code"])
            _, wsA, _, _, _ = _fit(r["code"], r["w"], A, mA,
                                   list(spA.train.shapes))
            va = _score_on(fn, wsA, B, mB, hold)
            fnB, wsB, moved, tr, meth = _fit(r["code"], r["w"], B, mB,
                                             list(spB.train.shapes))
            vb = _score_on(fnB, wsB, B, mB, hold)
            tr_g = geomean(np.array(list(tr.values()))) if tr else float("nan")
            mv = f"{sum(moved.values())}/{len(moved)}"
            print(f"  {r['tag']:22s} "
                  f"{r['shape']['weights']:4d}/{n_train:<5d} {va:8.4f} "
                  f"{vb:8.4f} {tr_g:8.4f} {vb - tr_g:+8.4f}  {mv:>5}  {meth}")
            t["cells"].append({
                "rule": r["tag"], "a": float(va), "b": float(vb),
                "b_train": float(tr_g), "gap": float(vb - tr_g),
                "moved": mv, "fitter": meth,
                "weights": r["shape"]["weights"],
                "terms": r["shape"]["terms"]})
        res["targets"][dst] = t

    Path(OUT).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {OUT}")
    print("  ⚠️ one seed produced r5 and r11. No verdict is attached.")


if __name__ == "__main__":
    main()
