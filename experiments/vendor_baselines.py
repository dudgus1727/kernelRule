"""★ The vendor baseline on **all four tables**, whole and per split.
0 LLM calls.

    python3 -m experiments.vendor_baselines

Until now only the A6000 and the 5090 had a vendor file, so the transfer
ladder (D-157) could only compare against static top-1 — the lowest bar
there is. `vendor_extract.py` computes from a preset without a GPU, so the
4090 and the H100 were simply never extracted.

## ⚠️ The H100 preset

The bundle is an **H100 NVL** and `GPU_PRESETS` mapped every "h100" to
`H100_SXM` — a different card. `H100_NVL` exists in the library, so the
mapping was fixed (D-158). Nothing was substituted.

## What is reported

```
whole table   vendor geomean · strict-match rate · shapes with no
              recommendation · static top-1 · random
per split     the transfer holdout (nk11008 val) of each table
★ mapping     "nearest" and "strict" side by side — the mapping method is a
              choice and it is shown, not hidden
```
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.transfer_29_5 import TABLES, _splits
from kernelrule.baselines.static_topk import StaticTopK
from kernelrule.baselines.vendor import (
    load_vendor,
    match_report,
    vendor_order_fn,
)
from kernelrule.core.scoring import evaluate, geomean
from kernelrule.core.splits import aligned_shapes, experiment_shapes
from kernelrule.core.table import PerfTable

OUT = "docs/artifacts/vendor-baselines.json"
FILES = {
    "a6000": "datasets/baselines/vendor-a6000-c63710df.json",
    "5090": "datasets/baselines/vendor-5090-5bb6f403.json",
    "4090": "datasets/baselines/vendor-4090-ad95d455.json",
    "h100": "datasets/baselines/vendor-h100-63684546.json",
}


def _vendor(table, vend, shapes, mapping):
    # ★ `vendor_order_fn` returns an **order**, so it goes to `evaluate`.
    #   Handing it to `evaluate_scores` reads the index array as scores and
    #   the number comes out meaningless (D-158 §2 — that is where D-157's
    #   5090 vendor 1.2562 came from).
    ev = evaluate(vendor_order_fn(table, vend, mapping=mapping),
                  table, shapes, ks=(1,), label="vendor")
    return ev, geomean(ev.regret[:, 0])


def _random(table, shapes) -> float:
    """The random baseline. **One place** — it was written inline twice
    below once the population block was added (principle 2)."""
    rng = np.random.default_rng(0)
    rr = []
    for p in shapes:
        t = table.times_of(p)
        rr.append(float(np.median([t[i] / t.min() for i in
                                   rng.integers(len(t), size=64)])))
    return float(geomean(np.array(rr)))


def _scope(table, vend, shapes) -> dict:
    """vendor / strict / static top-1 / random over one shape set."""
    _, g_near = _vendor(table, vend, shapes, "nearest")
    _, g_strict = _vendor(table, vend, shapes, "strict")
    st = StaticTopK(table, shapes, coverage="union").run(ks=(1,))
    return {"n": len(shapes), "vendor_nearest": float(g_near),
            "vendor_strict": float(g_strict),
            "static_top1": float(st.by_k[1]["all"]),
            "random": _random(table, shapes)}


def main() -> None:
    warnings.simplefilter("ignore")
    res: dict = {}
    print("=" * 92)
    print("The vendor baseline on four tables. 0 LLM calls")
    print("=" * 92)
    print(f"  {'gpu':7s} {'preset':10s} {'shapes':>6} {'vendor':>8} "
          f"{'strict':>8} {'no rec':>7} {'exact%':>7} {'top-1':>8} "
          f"{'random':>8}")
    per_shape: dict = {}
    for name, f in FILES.items():
        T = TABLES[name]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        vend = load_vendor(f)
        shapes = list(table.shapes())
        ev, g_near = _vendor(table, vend, shapes, "nearest")
        _, g_strict = _vendor(table, vend, shapes, "strict")
        mr = match_report(table, vend)
        st = StaticTopK(table, shapes, coverage="union").run(ks=(1,))
        per_shape[name] = {f"{p.M}x{p.N}x{p.K}": float(ev.regret[i, 0])
                           for i, p in enumerate(ev.shapes)}
        res[name] = {
            "preset": vend["meta"].get("preset"),
            "lib_version": vend["meta"].get("lib_version", "?"),
            "n_shapes": len(shapes),
            "vendor_nearest": float(g_near), "vendor_strict": float(g_strict),
            "exact_frac": mr["frac"],
            "shapes_without_vendor": mr["shapes_without_vendor"],
            "static_top1": float(st.by_k[1]["all"]),
            "random": _random(table, shapes),
            # ★ 2026-09-13 (D-170 §1): the row above is the **whole table**
            #   (66 / 64 shapes) and always was. The experiments never ran on
            #   that set. These two are the populations that are actually
            #   used — the current one and the one every pre-2026-09-13
            #   number has as its denominator.
            "population_family": _scope(table, vend,
                                        experiment_shapes(table)),
            "population_align8": _scope(table, vend, aligned_shapes(table))}
        r = res[name]
        print(f"  {name:7s} {r['preset']:10s} {r['n_shapes']:6d} "
              f"{r['vendor_nearest']:8.4f} {r['vendor_strict']:8.4f} "
              f"{len(r['shapes_without_vendor']):7d} {r['exact_frac']:7.1%} "
              f"{r['static_top1']:8.4f} {r['random']:8.4f}")

    # -- ★ the two populations (D-170 §1) ----------------------------------
    print("\n  ★ the shape populations — current (kernel family) vs the "
          "pre-2026-09-13 one (alignment 8)")
    print(f"  {'gpu':7s} {'n':>4} {'vendor':>8} {'top-1':>8} {'random':>8}"
          f"    {'n':>4} {'vendor':>8} {'top-1':>8} {'random':>8}")
    for name in FILES:
        f_, o_ = res[name]["population_family"], res[name]["population_align8"]
        print(f"  {name:7s} {f_['n']:4d} {f_['vendor_nearest']:8.4f} "
              f"{f_['static_top1']:8.4f} {f_['random']:8.4f}    "
              f"{o_['n']:4d} {o_['vendor_nearest']:8.4f} "
              f"{o_['static_top1']:8.4f} {o_['random']:8.4f}")

    # -- per split ---------------------------------------------------------
    print("\n  the transfer holdout of each table (nk11008 val)")
    print(f"  {'gpu':7s} {'shapes':>6} {'vendor':>8} {'strict':>8} "
          f"{'top-1':>8}")
    for name, f in FILES.items():
        T = TABLES[name]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        vend = load_vendor(f)
        hold = list(_splits(table).val.shapes)
        _, g_near = _vendor(table, vend, hold, "nearest")
        _, g_strict = _vendor(table, vend, hold, "strict")
        st = StaticTopK(table, hold, coverage="union").run(ks=(1,))
        res[name]["holdout"] = {
            "n": len(hold), "vendor_nearest": float(g_near),
            "vendor_strict": float(g_strict),
            "static_top1": float(st.by_k[1]["all"])}
        print(f"  {name:7s} {len(hold):6d} {g_near:8.4f} {g_strict:8.4f} "
              f"{st.by_k[1]['all']:8.4f}")

    # -- ★ the 5090 recheck (D-158 §2) -------------------------------------
    print("\n" + "=" * 92)
    print("★ the 5090 recheck — is the vendor really that bad there?")
    print("=" * 92)
    common = set(per_shape["a6000"]) & set(per_shape["5090"])
    pairs = sorted(((per_shape["5090"][k], per_shape["a6000"][k], k)
                    for k in common), reverse=True)
    print(f"  {len(common)} shapes are in both tables")
    print(f"  {'shape':22s} {'5090':>8} {'a6000':>8}")
    for v5, va, k in pairs[:8]:
        print(f"  {k:22s} {v5:8.4f} {va:8.4f}")
    a5 = np.array([per_shape["5090"][k] for k in common])
    aa = np.array([per_shape["a6000"][k] for k in common])
    res["recheck_5090"] = {
        "n_common": len(common),
        "5090_geomean": float(geomean(a5)),
        "a6000_geomean": float(geomean(aa)),
        "5090_median": float(np.median(a5)),
        "5090_over_1_5": int((a5 > 1.5).sum()),
        "5090_worst": [[k, per_shape["5090"][k]] for _, _, k in pairs[:5]]}
    print(f"\n  geomean on the common shapes   5090 {geomean(a5):.4f}   "
          f"a6000 {geomean(aa):.4f}")
    print(f"  5090 median {np.median(a5):.4f}   over 1.5: "
          f"{int((a5 > 1.5).sum())}/{len(a5)}   max {a5.max():.4f}")
    res["per_shape"] = per_shape
    Path(OUT).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
