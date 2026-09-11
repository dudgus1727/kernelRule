"""★ The vendor comparison — per shape is the main metric. 0 LLM calls.

    python3 experiments/vendor_compare.py

## Why per shape

```
secondary  a sign test over the 6 runs.  The p lower bound is 0.031 — at 5/6
           it is p=0.22 and says nothing
★ main     the **median** of the 6 runs at each shape vs the vendor, a sign
           test over the 20 shapes
           The p lower bound is ~1e-6.  The variance enters once (the
           between-run variance is absorbed by the median)
```

Per shape also avoids §30.4's problem that "the geomean is dragged by a few
shapes". **Both are reported, and per shape is the main metric** (the re-run
pre-registration §3).
"""

from __future__ import annotations

import argparse
import json
import warnings
from math import comb
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate, evaluate_scores, geomean
from kernelrule.core.splits import Split, experiment_shapes, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY, FeatureRegistry
from kernelrule.features.loader import extended_registry, load_generated

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"

#: The two arms. `(the run prefix, how the library is built)`
ARMS = {
    "F1, 21 features": ("F1rw-p8-s",
                        "runs/F1rw-p8/stage1-features/proposals.jsonl"),
    "human 24": ("F3rw-p8-s", None),
}


def _sign_test(wins: int, losses: int) -> float:
    """A two-sided sign test. Ties are dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _pick(pre: str, s: int, rnd: int | None) -> dict | None:
    """One rule from that run. If `rnd` is given it uses **the best of that
    round**.

    ★ `bests.jsonl` holds the best per round. `archive.jsonl` is **the final
    state** and cannot give an intermediate round (D-139).
    """
    d = Path("runs") / f"{pre}{s}"
    if rnd is None:
        rows = [json.loads(ln) for ln in (d / "archive.jsonl").open()
                if ln.strip()] if (d / "archive.jsonl").exists() else []
        return min(rows, key=lambda e: e["regret"]) if rows else None
    f = d / "bests.jsonl"
    if not f.exists():
        raise SystemExit(f"{f} does not exist — a round-pinned comparison "
                         f"cannot be made (D-139)")
    rows = [json.loads(ln) for ln in f.open() if ln.strip()]
    at = [e for e in rows if e["round"] <= rnd]
    return max(at, key=lambda e: e["round"]) if at else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", metavar="NAME=PREFIX[:LIBRARY]",
                    help="give the arms directly. Without it, ARMS is used")
    ap.add_argument("--round", type=int, default=None,
                    help="the rule as if it had stopped at that round "
                         "(bests.jsonl). The default is the last archive best")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = ARMS
    if a.arm:
        arms = {}
        for spec in a.arm:
            name, _, rest = spec.partition("=")
            pre, _, lib = rest.partition(":")
            arms[name] = (pre, lib or None)
    warnings.simplefilter("ignore")
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    shapes = experiment_shapes(table)
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]

    # -- the vendor: per-shape regret ------------------------------------
    vend = load_vendor(VENDOR)
    ev = evaluate(vendor_order_fn(table, vend, mapping="nearest"),
                  table, held, ks=(1,), label="vendor")
    v_by_shape = {p: float(ev.regret[i, 0]) for i, p in enumerate(ev.shapes)}
    print("=" * 78)
    print("the vendor comparison — ★ per shape is the main metric")
    print("=" * 78)
    print(f"  structural holdout {len(held)} shapes   "
          f"vendor geomean "
          f"{geomean(np.array(list(v_by_shape.values()))):.4f}\n")

    out_json: dict = {"bundle": BUNDLE, "round": a.round, "arms": {}}
    for tag, (pre, lib) in arms.items():
        if lib:
            reg = extended_registry(FeatureRegistry("F1-empty"),
                                    load_generated(lib, table=table,
                                                   exclude=set()),
                                    name="F1-lib")
        else:
            reg = FeatureRegistry("human24")
            for n in sorted(REGISTRY._items):
                reg.add(REGISTRY[n])
        matrix = FeatureMatrix(table, reg)

        # Per-shape regret per run. **Refitted per regime**, as in the final
        # scoring procedure.
        per_run: list[dict] = []
        used: list[str] = []
        for s in range(6):
            best = _pick(pre, s, a.round)
            if best is None:
                continue
            used.append(f"{pre}{s}")
            fn = compile_rule(best["code"])
            reg_by_shape: dict = {}
            for name in ("short", "long"):
                g_tr = [p for p in train if regime_of(p, table.hw) == name]
                g_ho = [p for p in held if regime_of(p, table.hw) == name]
                fr = fit_weights(fn, matrix, table, Split("train", tuple(g_tr)),
                                 best["w"], max_evals=300,
                                 warn_invariants=False,
                          objective="regret")
                e = evaluate_scores(make_score_of(fn, matrix, fr.w), table,
                                    g_ho, ks=(1,))
                for i, p in enumerate(e.shapes):
                    reg_by_shape[p] = float(e.regret[i, 0])
            per_run.append(reg_by_shape)

        print(f"  runs: {used}"
              + (f"   ★ at round {a.round}" if a.round is not None
                 else "   (the last archive best)"))
        out_json["arms"][tag] = {
            "runs": used,
            "per_shape_median": {str(p): float(np.median(
                [r[p] for r in per_run if p in r]))
                for p in held if any(p in r for r in per_run)},
        }
        _report(tag, per_run, v_by_shape, held, table)
    out_json["vendor_per_shape"] = {str(p): v for p, v in v_by_shape.items()}
    if a.out:
        Path(a.out).write_text(json.dumps(out_json, ensure_ascii=False,
                                          indent=1))
        print(f"\n  -> {a.out}")


def _report(tag, per_run, v_by_shape, held, table) -> None:
    print("=" * 78)
    print(f"{tag}   {len(per_run)} runs")
    print("=" * 78)

    # ★ The main metric — the per-shape median vs the vendor
    med = {p: float(np.median([r[p] for r in per_run if p in r]))
           for p in held if any(p in r for r in per_run)}
    rows = [(p, med[p], v_by_shape[p]) for p in med if p in v_by_shape]
    w = sum(1 for _, m, v in rows if m < v - 1e-9)
    lo = sum(1 for _, m, v in rows if m > v + 1e-9)
    print(f"  ★ per shape  wins {w} / losses {lo} / ties {len(rows)-w-lo}"
          f"   sign test p = {_sign_test(w, lo):.2e}")
    print(f"     geomean  ours {geomean(np.array([m for _, m, _ in rows])):.4f}"
          f"   vendor {geomean(np.array([v for _, _, v in rows])):.4f}")

    # Secondary — the 6 runs
    vg = geomean(np.array([v_by_shape[p] for p in held if p in v_by_shape]))
    runs = [geomean(np.array([r[p] for p in held if p in r])) for r in per_run]
    rw = sum(1 for x in runs if x < vg)
    print(f"  secondary, per run  wins {rw}/{len(runs)}"
          f"   sign test p = {_sign_test(rw, len(runs)-rw):.3f}")

    # ★ The per-regime decomposition
    print(f"\n  {'regime':22s} {'shapes':>7} {'ours':>8} {'vendor':>8} "
          f"{'win/loss':>10} {'p':>9}")
    for name, label in (("short", "fast (SOL<0.5ms)"),
                        ("long", "slow (SOL>=0.5ms)")):
        g = [(p, m, v) for p, m, v in rows
             if regime_of(p, table.hw) == name]
        if not g:
            continue
        gw = sum(1 for _, m, v in g if m < v - 1e-9)
        gl = sum(1 for _, m, v in g if m > v + 1e-9)
        print(f"  {label:22s} {len(g):7d} "
              f"{geomean(np.array([m for _, m, _ in g])):8.4f} "
              f"{geomean(np.array([v for _, _, v in g])):8.4f} "
              f"{gw:5d}/{gl:<4d} {_sign_test(gw, gl):9.3f}")
    print()


if __name__ == "__main__":
    main()
