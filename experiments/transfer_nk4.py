"""★ The transfer matrix of the 48-run campaign (D-171 §4). **0 LLM calls.**

    python3 -m experiments.transfer_nk4

```
source   the best-by-★training rule of each (table, fold) — the holdout is
         not looked at (§10.2)
target   the ★ same fold of the other three tables
method   (a) carried over as it is   (b) weights refitted on the target's
                                         training split
```

## ★ The rule's axes travel with it (§4-1, the D-165 wiring)

A source run's registry is the shared `k7-1` library **plus the axes that
run built during the loop** — and, because `stage3` runs a fold's three
seeds in one process with one matrix, **plus the axes the earlier seeds of
that fold built**. All of them are re-derived on the *target* table with
`load_generated(..., table=<target>)`, and two things are counted:

```
★ shape_level flips   an axis that is shape level on the source table and
                      config level on the target (or the reverse)
★ constant on target  the axis takes one value over the target's shapes
                      -> any branch on it is dead there
```

## What each cell carries

Beside (a) and (b): the **vendor heuristic**, **static top-1**, and the
**native** rule — the one grown on that very table and fold. ★ The native
column is the answer to "is transfer as good as growing it there".
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.f1_pipeline import _splits
from experiments.transfer_29_5 import TABLES, _fit_per_regime, _score_on
from kernelrule.baselines.static_topk import StaticTopK
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.scoring import evaluate, geomean
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY, FeatureRegistry
from kernelrule.features.loader import load_generated, run_registry

GPUS = ("a6000", "5090", "4090", "h100")
FOLDS = (0, 1, 2, 3)
SEEDS = (0, 1, 2)
OUT = Path("docs/artifacts/transfer-nk4.json")
VENDOR = {g: f"datasets/baselines/vendor-{g}-{TABLES[g]['env_hash'][:8]}.json"
          for g in GPUS}


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _best_of_fold(gpu: str, fold: int) -> tuple[dict, int]:
    """The best rule of a (table, fold) **by training score**, over its three
    seeds. ⛔ The holdout is not read (§10.2)."""
    best, best_seed = None, None
    for s in SEEDS:
        arc = Path(f"runs/nk4-{gpu}-f{fold}-s{s}/archive.jsonl")
        if not arc.exists():
            continue
        e = min(_rows(arc), key=lambda x: x["regret"])
        if best is None or e["regret"] < best["regret"]:
            best, best_seed = e, s
    return best, best_seed


def _registry_for(gpu: str, fold: int, seed: int, table):
    """The source run's axes, re-derived **on `table`** (§4-1).

    Returns `(registry, notes)` where notes records the axes whose
    `shape_level` verdict differs from the source table's and the axes that
    are constant here.
    """
    reg, origin = run_registry(f"nk4-{gpu}-f{fold}", table=table, seed=seed,
                               human=REGISTRY)
    for earlier in range(seed):
        fp = Path(f"runs/nk4-{gpu}-f{fold}-s{earlier}/features.jsonl")
        if fp.exists():
            for f in load_generated(fp, table=table):
                if f.name not in reg._items:
                    reg.add(f)
                    origin[f.name] = f"loop s{earlier} (inherited)"
    return reg, origin


def _shape_level_map(gpu: str, fold: int, seed: int, table) -> dict:
    reg, _ = _registry_for(gpu, fold, seed, table)
    return {n: bool(reg[n].shape_level) for n in sorted(reg._items)}


def _constant_axes(reg: FeatureRegistry, table, shapes) -> list[str]:
    """Axes that take a single value over `shapes` — a branch on one of them
    is dead."""
    matrix = FeatureMatrix(table, reg, shapes=shapes)
    out = []
    for n in sorted(reg._items):
        if not reg[n].shape_level:
            continue
        v = np.array([float(getattr(matrix.for_shape(p)[1], n))
                      for p in shapes])
        if len(np.unique(np.round(v, 12))) == 1:
            out.append(n)
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    # ★ One cell costs two per-regime refits at 300 evals on rules of up to
    #   83 weights — measured at ~19 min, so 48 cells in one process is 15
    #   hours. Cells are independent, so they are split by (fold, source) and
    #   merged. ⛔ The computation per cell is unchanged.
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    a = ap.parse_args()

    if a.merge:
        cells = []
        for fn_ in a.merge:
            cells.extend(json.loads(Path(fn_).read_text())["cells"])
        cells.sort(key=lambda c: (c["fold"], c["src"], c["dst"]))
        Path(a.out).write_text(json.dumps(
            {"design": "D-171 §4", "n_cells": len(cells), "cells": cells},
            ensure_ascii=False, indent=1))
        print(f"merged {len(cells)} cells -> {a.out}")
        return

    tables = {g: PerfTable.from_bundle(TABLES[g]["bundle"],
                                       env_hash=TABLES[g]["env_hash"],
                                       ok_only=False) for g in GPUS}
    splits = {(g, f): _splits(tables[g], fold=f, k=4, design="nkgroup")
              for g in GPUS for f in FOLDS}
    best = {(g, f): _best_of_fold(g, f) for g in GPUS for f in FOLDS}

    cells: list[dict] = []
    print("=" * 100)
    print("★ the transfer matrix — 4 sources x 3 targets x 4 folds "
          "(D-171 §4). 0 LLM calls")
    print("=" * 100)
    #: ★ The native fit of a (table, fold) is the same for every source, so
    #: it is computed once per process rather than once per cell.
    native_cache: dict = {}
    for f in (FOLDS if a.fold is None else (a.fold,)):
        for src in (GPUS if a.src is None else (a.src,)):
            e, seed = best[(src, f)]
            if e is None:
                print(f"  fold{f} {src:6s} ★ no run")
                continue
            sA, tA = splits[(src, f)], tables[src]
            regA, _ = _registry_for(src, f, seed, tA)
            mA = FeatureMatrix(tA, regA, cache_dir=CACHE_DIR)
            fnA, wsA = _fit_per_regime(e["code"], e["w"], tA, mA,
                                       list(sA.train.shapes))
            slA = _shape_level_map(src, f, seed, tA)
            for dst in GPUS:
                if dst == src:
                    continue
                tB, sB = tables[dst], splits[(dst, f)]
                regB, _ = _registry_for(src, f, seed, tB)
                mB = FeatureMatrix(tB, regB, cache_dir=CACHE_DIR)
                hold = list(sB.val.shapes)
                slB = {n: bool(regB[n].shape_level) for n in regB._items}
                flips = sorted(n for n in slA
                               if n in slB and slA[n] != slB[n])
                const = _constant_axes(regB, tB, hold)
                from kernelrule.core.sandbox import compile_rule
                fnB = compile_rule(e["code"])
                ga = _score_on(fnB, wsA, tB, mB, hold)
                _, wsB = _fit_per_regime(e["code"], e["w"], tB, mB,
                                         list(sB.train.shapes))
                gb = _score_on(fnB, wsB, tB, mB, hold)
                moved = {k: [round(x, 6) for x in v] != list(e["w"])
                         for k, v in wsB.items()}
                # native: the target's own rule for this fold
                ne, nseed = best[(dst, f)]
                gn = None
                if (dst, f) in native_cache:
                    gn = native_cache[(dst, f)]
                elif ne is not None:
                    regN, _ = _registry_for(dst, f, nseed, tB)
                    mN = FeatureMatrix(tB, regN, cache_dir=CACHE_DIR)
                    fnN, wsN = _fit_per_regime(ne["code"], ne["w"], tB, mN,
                                               list(sB.train.shapes))
                    gn = _score_on(fnN, wsN, tB, mN, hold)
                    native_cache[(dst, f)] = gn
                vend = load_vendor(VENDOR[dst])
                ev = evaluate(vendor_order_fn(tB, vend, mapping="nearest"),
                              tB, hold, ks=(1,), label="vendor")
                gv = float(geomean(ev.regret[:, 0]))
                gs = float(StaticTopK(tB, hold,
                                      coverage="union").run(ks=(1,)
                                                            ).by_k[1]["all"])
                cells.append({
                    "fold": f, "src": src, "dst": dst, "src_seed": seed,
                    "n_holdout": len(hold),
                    "a_as_is": round(float(ga), 6),
                    "b_refit": round(float(gb), 6),
                    "native": (round(float(gn), 6) if gn is not None
                               else None),
                    "vendor": round(gv, 6), "static_top1": round(gs, 6),
                    "fit_moved": moved,
                    "shape_level_flips": flips,
                    "constant_on_target": const,
                    "src_train_regret": e["regret"]})
                print(f"  fold{f} {src:6s} -> {dst:6s}  (a) {ga:.4f}  "
                      f"(b) {gb:.4f}  native "
                      + (f"{gn:.4f}" if gn is not None else "  —  ")
                      + f"  vendor {gv:.4f}  top1 {gs:.4f}"
                      + (f"  ★ flips {len(flips)}" if flips else "")
                      + (f"  ★ const {len(const)}" if const else ""))
    Path(a.out).write_text(json.dumps(
        {"design": "D-171 §4", "n_cells": len(cells), "cells": cells},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
