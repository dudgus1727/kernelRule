"""★ The autotuning curve, merged — and where it crosses us (D-176 §1-4~6).
**0 LLM calls · 0 GPU.**

    python3 -m experiments.autotune_report --merge

Takes the per-`(gpu, fold)` shards `experiments.autotune_curve` writes and
answers the three questions the curve exists for:

```
★ 1-4  the curve         k -> regret, per fold and pooled per table
★ 1-5  the crossing      ★ the k at which autotuning first matches
                         ★ our rule · ★ the vendor
★ 1-6  in time           k kernels -> ★ seconds, from the table's own
                         `n_reps` · `build_seconds` · the bundle protocol
```

⛔ **The k=0 points come from the same shapes as the curve.** Our rule's
point is the c2 loop's holdout on that fold (median over its 4 seeds), the
vendor's is `vendor-baselines.json`'s per-shape regret restricted to that
fold's validation shapes. A fold whose shapes are not all in the vendor
artefact reports `null` — ⛔ not a partial geomean.

⚠️ The vendor artefact carries no H100 per-shape regret, so the H100 has no
vendor crossing. It is left empty rather than filled from another table.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import warnings
from pathlib import Path

import numpy as np

from experiments.autotune_curve import KS
from experiments.f1_pipeline import _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.core.scoring import geomean
from kernelrule.core.table import PerfTable

GPUS = ("a6000", "5090", "4090", "h100")
ARMS = ("random", "random_pruned", "random_cond", "tpe")
OUT = Path("docs/artifacts/autotune-curve.json")
OUT_MD = Path("docs/artifacts/autotune-curve.md")
OUT_CSV = Path("docs/artifacts/autotune-curve.csv")
VENDOR = Path("docs/artifacts/vendor-baselines.json")
CAMP = Path("docs/artifacts/campaign2.json")


def _cross(curve: dict, target: float) -> dict:
    """★ The first k whose curve value is at or below `target`, and a
    log-interpolated k between the bracketing points.

    ⚠️ The interpolated number is **not** a measured k — the curve is only
    known at the ks that were run. It is labelled as interpolated wherever
    it is printed."""
    ks = [k for k in KS if str(k) in curve]
    vals = [curve[str(k)] for k in ks]
    for i, (k, v) in enumerate(zip(ks, vals, strict=True)):
        if v <= target:
            if i == 0:
                return {"k": k, "k_interp": float(k), "reached": True}
            k0, v0 = ks[i - 1], vals[i - 1]
            # log-linear in k, linear in regret
            f = (v0 - target) / (v0 - v) if v0 != v else 1.0
            lk = np.log(k0) + f * (np.log(k) - np.log(k0))
            return {"k": k, "k_interp": round(float(np.exp(lk)), 1),
                    "reached": True}
    return {"k": None, "k_interp": None, "reached": False,
            "best_at_kmax": vals[-1] if vals else None}


def _sec_per_candidate(table, shapes, bundle: str) -> dict:
    """★ What one measured candidate cost the people who built the table.

    ```
    measure   time_ms x (n_reps + warmup),  warmup = max(min_warmup,
                                            warmup_frac x n_reps)
    build     `build_seconds` — the compile cost of that kernel
    ```

    ⚠️ `n_reps` is **not** in `frame_for` (the scoring frame keeps only what
    a rule may see), so this reads the bundle's `table.parquet` directly —
    the same file the table was loaded from. The protocol constants come
    from the bundle's `BUNDLE.json`.

    ⛔ Nothing here is estimated. A field that is missing is reported
    missing and the curve is then quoted in kernel runs only (§1-6).
    """
    import pandas as pd

    bj = Path(bundle) / "BUNDLE.json"
    proto = (json.loads(bj.read_text()).get("protocol") or {}
             if bj.exists() else {})
    pq = Path(bundle) / "table.parquet"
    if not pq.exists():
        return {"available": False, "why": f"no {pq}"}
    want = ["M", "N", "K", "time_ms", "n_reps", "build_seconds"]
    try:
        d = pd.read_parquet(pq, columns=want)
    except Exception as e:                       # noqa: BLE001
        return {"available": False,
                "why": f"{want} not all in the table: {e}"}
    wf, mw = proto.get("warmup_frac"), proto.get("min_warmup")
    if wf is None or mw is None:
        return {"available": False,
                "why": "the bundle protocol block has no warmup setting"}
    keys = {(p.M, p.N, p.K) for p in shapes}
    d = d[[(m, n, k) in keys for m, n, k in
           zip(d["M"], d["N"], d["K"], strict=True)]]
    if d.empty:
        return {"available": False, "why": "no rows for these shapes"}
    n = d["n_reps"].to_numpy(dtype=float)
    t = d["time_ms"].to_numpy(dtype=float)
    ms = t * (n + np.maximum(mw, wf * n))
    b = d["build_seconds"].to_numpy(dtype=float)
    return {"available": True, "source": str(pq),
            "warmup_frac": wf, "min_warmup": mw,
            "n_rows": int(len(d)),
            "measure_sec_median": round(float(np.median(ms)) / 1000.0, 4),
            "measure_sec_mean": round(float(np.mean(ms)) / 1000.0, 4),
            "build_sec_median": round(float(np.median(b)), 3),
            "build_sec_mean": round(float(np.mean(b)), 3),
            "build_available": True,
            "n_reps_median": float(np.median(n))}


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--shard-dir", default="docs/artifacts")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    vend = json.loads(VENDOR.read_text())["per_shape"]
    camp = json.loads(CAMP.read_text())
    loop: dict = {}
    for r in camp["runs"]:
        if not r.get("missing"):
            loop.setdefault((r["gpu"], r["fold"]), []).append(r["holdout"])

    res: dict = {"ks": list(KS), "arms": list(ARMS), "gpus": {},
                 "k0_note": ("⛔ the k=0 points are computed on the same "
                             "fold validation shapes as the curve"),
                 "interp_note": ("⚠️ k_interp is log-interpolated between "
                                 "the ks that were actually run — it is not "
                                 "a measured k")}
    print("=" * 104)
    print("★ 오토튜닝 곡선 — 병합 · 교차점 · 시간 환산 (D-176 §1). 0 LLM · 0 GPU")
    print("=" * 104)
    for gpu in GPUS:
        # ★ Shards are per (fold, shape-slice); a shape appears in exactly
        #   one of them. Merge by shape key so a slice is never averaged as
        #   if it were a whole fold.
        byfold: dict = {}
        meta: dict = {}
        for sp in sorted(Path(a.shard_dir).glob(f"autotune-{gpu}-f*.json")):
            sh = json.loads(sp.read_text())
            meta = {k: sh[k] for k in ("tpe_seeds", "tpe_multivariate",
                                       "tpe_group", "repeats")
                    if k in sh}
            for f, d in sh["folds"].items():
                for row in d["shapes"]:
                    byfold.setdefault(int(f), {})[tuple(row["shape"])] = row
        if not byfold:
            continue
        T = TABLES[gpu]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        g: dict = {"folds": {}, "n_folds_present": len(byfold), **meta}
        allrows: list[dict] = []
        for f in sorted(byfold):
            rows = list(byfold[f].values())
            allrows += rows
            want = list(_splits(table, fold=f, k=4,
                                design="nkband").val.shapes)
            keys = [f"{p.M}x{p.N}x{p.K}" for p in want]
            vv = [vend.get(gpu, {}).get(k) for k in keys]
            v0 = (float(geomean(np.array(vv, float)))
                  if all(x is not None for x in vv) else None)
            lp = loop.get((gpu, f))
            r0 = round(st.median(lp), 6) if lp else None
            cur = _agg(rows)
            row = {"n_shapes": len(rows), "n_shapes_expected": len(want),
                   # ⚠️ a fold whose shards are not all in is aggregated over
                   #    what is there, and says so.
                   "complete": len(rows) == len(want),
                   "curves": cur, **_stats(rows),
                   "k0_our_rule": r0, "k0_our_rule_seeds": lp,
                   "k0_vendor": (round(v0, 6) if v0 is not None else None),
                   "k0_vendor_missing": sum(1 for x in vv if x is None),
                   "crossing": {}}
            for m in ARMS:
                row["crossing"][m] = {
                    "vs_our_rule": (_cross(cur[m], r0)
                                    if r0 is not None else None),
                    "vs_vendor": (_cross(cur[m], v0)
                                  if v0 is not None else None)}
            g["folds"][str(f)] = row
        g["pooled"] = _agg(allrows)
        g["n_shapes"] = len(allrows)
        g.update(_stats(allrows))
        want = [p for f in sorted(byfold)
                for p in _splits(table, fold=f, k=4,
                                 design="nkband").val.shapes]
        vals = [vend.get(gpu, {}).get(f"{p.M}x{p.N}x{p.K}") for p in want]
        g["k0_vendor_pooled"] = (round(float(geomean(np.array(vals, float))),
                                       6)
                                 if all(x is not None for x in vals) else None)
        lp = [x for f in sorted(byfold) for x in loop.get((gpu, f), [])]
        g["k0_our_rule_pooled"] = round(st.median(lp), 6) if lp else None
        g["time"] = _sec_per_candidate(table, want, T["bundle"])
        for m in ARMS:
            g.setdefault("crossing_pooled", {})[m] = {
                "vs_our_rule": (_cross(g["pooled"][m], g["k0_our_rule_pooled"])
                                if g["k0_our_rule_pooled"] else None),
                "vs_vendor": (_cross(g["pooled"][m], g["k0_vendor_pooled"])
                              if g["k0_vendor_pooled"] else None)}
        res["gpus"][gpu] = g
        _print(gpu, g)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    OUT_MD.write_text(_md(res))
    _csv(res)
    print(f"\n  -> {a.out}\n  -> {OUT_MD}\n  -> {OUT_CSV}")


def _agg(rows: list[dict]) -> dict:
    """Geomean over shapes, per arm and k.

    ⚠️ Only the ks an arm was **actually run at** in every shape. The TPE
    arm stops at `autotune_curve.TPE_KMAX`; its higher cells are absent, ⛔
    not extrapolated from the last one."""
    out: dict = {}
    for m in ARMS:
        out[m] = {}
        for k in KS:
            v = [r["curves"][m].get(str(k)) for r in rows]
            if any(x is None for x in v):
                continue
            out[m][str(k)] = float(geomean(np.array(v, float)))
    return out


def _stats(rows: list[dict]) -> dict:
    nv = [r["n_valid"] for r in rows]
    return {"n_valid_median": int(np.median(nv)),
            "n_valid_min": int(min(nv)),
            "n_pruned_median": int(np.median([r["n_pruned"] for r in rows])),
            "prune_keeps": round(float(np.mean(
                [r["n_pruned"] / r["n_valid"] for r in rows])), 4),
            "tpe_duplicate_rate": round(float(np.mean(
                [r["tpe_duplicate_rate"] for r in rows])), 4),
            "cond_random_duplicate_rate": round(float(np.mean(
                [r["cond_random_duplicate_rate"] for r in rows])), 4),
            "tpe_hit_cap": sum(r["tpe_hit_cap"] for r in rows),
            # ★ D-176 — shapes whose whole candidate list fits inside k_max,
            #   so the curve there is exhaustive rather than a search.
            "n_shapes_exhausted_at_kmax": sum(1 for n in nv if n <= max(KS))}


def _print(gpu: str, g: dict) -> None:
    print(f"\n  ── {gpu}  {g['n_shapes']} 형상 "
          f"· TPE 중복률 {g['tpe_duplicate_rate']:.1%} "
          f"· k=4096 에서 전수인 형상 {g['n_shapes_exhausted_at_kmax']}"
          f"/{g['n_shapes']} (최소 후보 {g['n_valid_min']:,})")
    print(f"    {'k':>6} " + " ".join(f"{m:>13}" for m in ARMS))
    for k in (x for x in (1, 8, 64, 512, 1024, 4096)
              if str(x) in g["pooled"][ARMS[0]]):
        print(f"    {k:>6} " + " ".join(
            (f"{g['pooled'][m][str(k)]:13.4f}"
             if str(k) in g["pooled"][m] else f"{'—':>13}") for m in ARMS))
    r0, v0 = g["k0_our_rule_pooled"], g["k0_vendor_pooled"]
    print(f"    {'k=0':>6} 우리 {r0}  벤더 {v0}")
    for m in ARMS:
        c = g["crossing_pooled"][m]
        def s(x):
            if x is None:
                return "—"
            return (f"k={x['k']} (보간 {x['k_interp']})" if x["reached"]
                    else f"★ {max(KS)} 까지 못 넘음 "
                         f"({x['best_at_kmax']:.4f})")
        print(f"    {m:>14}  우리 {s(c['vs_our_rule'])}  "
              f"벤더 {s(c['vs_vendor'])}")
    t = g["time"]
    if t.get("available"):
        print(f"    ★ 후보 하나 측정 {t['measure_sec_median']:.3f}s "
              f"(중앙) · 빌드 "
              + (f"{t['build_sec_median']:.2f}s" if t["build_available"]
                 else "표에 없다"))
    else:
        print(f"    ⛔ 시간 환산 불가 — {t.get('why')}")


def _csv(res: dict) -> None:
    """★ §4 — the plot's own data. x = kernel runs per shape (log), y =
    regret. One row per (table, fold-or-pooled, arm, k), plus the ★ k=0
    rows so our rule and the vendor sit on the same axes."""
    L = ["gpu,fold,arm,k,regret"]
    for gpu, g in res["gpus"].items():
        for f, row in g["folds"].items():
            for m in res["arms"]:
                for k in res["ks"]:
                    if str(k) in row["curves"][m]:
                        L.append(f"{gpu},{f},{m},{k},"
                                 f"{row['curves'][m][str(k)]:.6f}")
            if row["k0_our_rule"] is not None:
                L.append(f"{gpu},{f},our_rule,0,{row['k0_our_rule']:.6f}")
            if row["k0_vendor"] is not None:
                L.append(f"{gpu},{f},vendor,0,{row['k0_vendor']:.6f}")
        for m in res["arms"]:
            for k in res["ks"]:
                if str(k) in g["pooled"][m]:
                    L.append(f"{gpu},pooled,{m},{k},"
                             f"{g['pooled'][m][str(k)]:.6f}")
        if g["k0_our_rule_pooled"] is not None:
            L.append(f"{gpu},pooled,our_rule,0,{g['k0_our_rule_pooled']:.6f}")
        if g["k0_vendor_pooled"] is not None:
            L.append(f"{gpu},pooled,vendor,0,{g['k0_vendor_pooled']:.6f}")
    OUT_CSV.write_text("\n".join(L) + "\n")


def _md(res: dict) -> str:
    L = ["# The autotuning curve (D-176 §1)", "",
         ("> **reproduce** `python3 -m experiments.autotune_curve --gpu "
          "<g> --fold <f>` then `python3 -m experiments.autotune_report "
          "--merge`"),
         "> 0 LLM calls · 0 GPU · the table is the measurement", "",
         ("| table | shapes | k=0 ours | k=0 vendor | random k=64 | "
          "pruned k=64 | cond-random k=64 | TPE k=64 | TPE dup |"),
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for gpu, g in res["gpus"].items():
        v = g["k0_vendor_pooled"]
        L.append(f"| {gpu} | {g['n_shapes']} | {g['k0_our_rule_pooled']} | "
                 + (f"{v}" if v else "—") + " | "
                 + " | ".join(f"{g['pooled'][m]['64']:.4f}" for m in ARMS)
                 + f" | {g['tpe_duplicate_rate']:.1%} |")
    L += ["", "## ★ Where autotuning catches us", "",
          "| table | arm | vs our rule | vs vendor |", "|---|---|---|---|"]
    for gpu, g in res["gpus"].items():
        for m in res["arms"]:
            c = g["crossing_pooled"][m]

            def s(x):
                if x is None:
                    return "—"
                return (f"k={x['k']} (interp {x['k_interp']})"
                        if x["reached"]
                        else f"**not by 4096** ({x['best_at_kmax']:.4f})")
            L.append(f"| {gpu} | {m} | {s(c['vs_our_rule'])} | "
                     f"{s(c['vs_vendor'])} |")
    L += ["", ("⚠️ `interp` is log-interpolated between the ks actually run "
               "(1, 2, 4, … 4096). It is not a measured k."),
          "", ("⚠️ **The TPE arm stops at k=1024**, the three random arms go "
               "to 4096. TPE's cost grows with both the duplicate rate and "
               "the number of completed trials (measured: 22.5 s at k=512, "
               "163 s at k=1024, 1,065 s at k=2048 for one shape and one "
               "seed). It is not a cut where we win — TPE reaches regret "
               "1.0000 by k=1024 on that shape and the crossing against our "
               "rule is near k=50. ⛔ The missing cells are left empty, not "
               "extrapolated. See `runs/x-autotune-aborted-k4096/`."),
          "", "## ★ In seconds", "",
          "| table | measure / candidate | build / candidate |",
          "|---|--:|--:|"]
    for gpu, g in res["gpus"].items():
        t = g["time"]
        if not t.get("available"):
            L.append(f"| {gpu} | — | — |")
            continue
        L.append(f"| {gpu} | {t['measure_sec_median']:.3f} s | "
                 + (f"{t['build_sec_median']:.2f} s" if t["build_available"]
                    else "**not in the table**") + " |")
    L += ["", ("★ Both numbers are the table's own: measure = `time_ms` × "
               "(`n_reps` + warmup) with the bundle `protocol` block's "
               "`warmup_frac` / `min_warmup`, build = `build_seconds`. "
               "⛔ Nothing is estimated."), ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
