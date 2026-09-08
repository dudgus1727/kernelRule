"""★ **The selection cost** — how many µs does choosing take? 0 LLM calls ·
0 GPU.

    python3 experiments/select_cost.py
    python3 experiments/select_cost.py --shapes 6 --reps 50

§2's first constraint is "it has to choose within µs", but **we had no number
of our own.** Only regret was measured. It is measured here.

## The three arms — the same process, the same Python

```
A ours    compute the features -> evaluate the rule -> argmin
          ★ it **includes** the feature computation
B vendor  one nvMatmulHeuristics.get_with_mnk  (the Python binding included)
C GBDT    predict every candidate of that shape with the trained model
          -> argmin
```

The pre-registration is `docs/artifacts/select-cost-prereg.md`.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
ENV_HASH = "c63710df"
#: The representative run. It uses the rule at r11 (D-140).
RUNS = [f"F3rw-p8-nan-s{i}" for i in range(6)]
ROUND = 11


def _rule(run: str, rnd: int) -> dict:
    f = Path("runs") / run / "bests.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    at = [e for e in rows if e["round"] <= rnd]
    return max(at, key=lambda e: e["round"])


def _timed(fn, reps: int, warmup: int = 5) -> tuple[float, float, float]:
    """(median µs, worst µs, mean µs). ★ The worst is always reported — that
    is the deployment criterion."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    a = np.array(ts)
    return float(np.median(a)), float(a.max()), float(a.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=int, default=6,
                    help="how many shapes to measure (spread evenly by "
                         "candidate count)")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="docs/artifacts/select-cost.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(BUNDLE, env_hash=ENV_HASH, ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    hw = T.hw
    rule = _rule(RUNS[0], ROUND)
    fn = compile_rule(rule["code"])
    w = np.asarray(rule["w"], dtype=np.float64)

    # ★ Only the features the rule actually uses are computed — there is no
    #   reason to compute what deployment does not use. The names are taken
    #   from the code.
    cfg_feats = [f for f in REGISTRY.items(shape_level=False)
                 if f"f.{f.name}" in rule["code"]]
    shp_feats = [f for f in REGISTRY.items(shape_level=True)
                 if f"p.{f.name}" in rule["code"]]
    print(f"rule {RUNS[0]}@r{ROUND}  features {len(cfg_feats)} "
          f"(config) + {len(shp_feats)} (shape)  weights {len(w)}")

    shapes = sorted(T.shapes(), key=lambda p: len(T.frame_for(p)))
    pick = [shapes[i] for i in
            np.linspace(0, len(shapes) - 1, a.shapes).astype(int)]

    # -- the catalogue (the exhaustive upper bound) -------------------------
    X = T._X
    key = ["kernel_id", "split_k", "split_k_mode"]
    cat = X.drop_duplicates(key).reset_index(drop=True)
    print(f"catalogue {len(cat):,} combinations · candidates per shape "
          f"{len(T.frame_for(shapes[0])):,}~{len(T.frame_for(shapes[-1])):,}")

    # A representative filter. ⚠️ It is not kernelTab's real predicate — it
    # measures **the order of magnitude of the cost**. Four vector comparisons
    # against static arrays.
    c_tile_k = cat["tile_k"].to_numpy(dtype=np.float64)
    c_split = cat["split_k"].to_numpy(dtype=np.float64)
    c_align = cat[["align_a", "align_b", "align_c"]].to_numpy(dtype=np.float64)
    c_smem = cat["smem_bytes"].to_numpy(dtype=np.float64)
    c_regs = cat["regs_total_per_block"].to_numpy(dtype=np.float64)
    SMEM_LIMIT, REGS_LIMIT = 101376.0, 65536.0

    def _filter(p):
        ok = (np.ceil(p.K / c_split) >= c_tile_k)
        ok &= (p.K % c_align[:, 0] == 0) & (p.N % c_align[:, 1] == 0)
        ok &= (p.N % c_align[:, 2] == 0)
        ok &= (c_smem <= SMEM_LIMIT) & (c_regs <= REGS_LIMIT)
        return ok

    # -- the vendor ---------------------------------------------------------
    vend = None
    try:
        import nvMatmulHeuristics as nv
        from kernelrule.baselines.vendor import preset_for
        h = nv.NvMatmulHeuristicsInterface(nv.NvMatmulHeuristicsTarget.CUTLASS,
                                           precision="HSS")
        hd = h.createHardwareDescriptor()
        h.setHardwarePredefinedGpu(hd, getattr(
            nv.NvMatmulHeuristicsNvidiaGpu, preset_for(hw.name)))
        layout = nv.NvMatmulHeuristicsMatmulLayout.TN_ROW_MAJOR
        vend = (h, hd, layout)
        print(f"vendor preset {preset_for(hw.name)}")
    except Exception as e:                                  # noqa: BLE001
        print(f"⚠️ no vendor arm: {type(e).__name__}: {e}")

    # -- GBDT ---------------------------------------------------------------
    gbdt = None
    try:
        from kernelrule.baselines.gbdt import build_xy
        from lightgbm import LGBMRegressor
        Xg, yg, g, cols, gshapes = build_xy(T)
        m = LGBMRegressor(n_estimators=200, num_leaves=63, verbose=-1)
        m.fit(Xg, yg)
        gbdt = (m, Xg, g, gshapes)
        print(f"GBDT trained — {len(cols)} features")
    except Exception as e:                                  # noqa: BLE001
        print(f"⚠️ no GBDT arm: {type(e).__name__}: {e}")

    print("\n" + "=" * 100)
    print(f"the selection cost (µs) — {a.reps} repetitions, 5 warm-ups. "
          f"★ median / worst")
    print("=" * 100)
    print(f"  {'shape':22s} {'cand':>7s} {'ours, filtered':>16s} "
          f"{'ours, catalogue':>17s} {'vendor':>14s} {'GBDT':>14s}")
    out: dict = {"bundle": BUNDLE, "reps": a.reps, "rule_run": RUNS[0],
                 "rule_round": ROUND, "n_catalogue": int(len(cat)),
                 "shapes": {}}

    for p in pick:
        df = T.frame_for(p)
        info = {k: v for k, v in M._info[p.key].items()}

        def ours(d=df, i=info):
            cols_ = {f.name: M._vector(f, p, d, i) for f in cfg_feats}
            from kernelrule.core.matrix import Feats, ShapeInfo
            s = fn(Feats(cols_), ShapeInfo(i), hw, w)
            return int(np.argmin(np.asarray(s)))

        def ours_full(i=info):
            ok = _filter(p)                # ★ the filter cost is included
            d = cat.loc[ok]
            cols_ = {f.name: M._vector(f, p, d, i) for f in cfg_feats}
            from kernelrule.core.matrix import Feats, ShapeInfo
            s = fn(Feats(cols_), ShapeInfo(i), hw, w)
            return int(np.argmin(np.asarray(s)))

        # ★ The breakdown — where the time goes. A diagnostic, not a verdict.
        from kernelrule.core.matrix import Feats, ShapeInfo
        d_np = {f.name: None for f in cfg_feats}
        _feat = _timed(lambda: {f.name: M._vector(f, p, df, info)
                                for f in cfg_feats}, a.reps)[0]
        _cols = {f.name: M._vector(f, p, df, info) for f in cfg_feats}
        _eval = _timed(lambda: int(np.argmin(np.asarray(
            fn(Feats(_cols), ShapeInfo(info), hw, w)))), a.reps)[0]
        _filt = _timed(lambda: int(_filter(p).sum()), a.reps)[0]
        # ★ The deployment form — the static columns are **converted to numpy
        #   in advance** (the catalogue is known at deployment time). The
        #   features are still computed every time.
        df_np = df.reset_index(drop=True)
        _deploy = _timed(lambda: int(np.argmin(np.asarray(fn(
            Feats({f.name: M._vector(f, p, df_np, info) for f in cfg_feats}),
            ShapeInfo(info), hw, w)))), a.reps)[0]
        med, mx, _ = _timed(ours, a.reps)
        n_filt = int(_filter(p).sum())
        med2, mx2, _ = _timed(ours_full, a.reps)
        row = {"n_cand": int(len(df)), "n_after_filter": n_filt,
               "ours_med": med, "ours_max": mx,
               "part_feat": _feat, "part_eval": _eval, "part_filter": _filt,
               "ours_deploy_med": _deploy,
               "ours_cat_med": med2, "ours_cat_max": mx2}
        vs = vm = float("nan")
        if vend:
            h, hd, layout = vend
            vs, vm, _ = _timed(
                lambda: h.get_with_mnk(p.M, p.N, p.K, layout, 1, hd), a.reps)
            row["vendor_med"], row["vendor_max"] = vs, vm
        gs = gm = float("nan")
        if gbdt:
            m, Xg, g, gshapes = gbdt
            idx = [i for i, q in enumerate(gshapes) if q.key == p.key]
            rows_ = np.flatnonzero(g == idx[0]) if idx else np.array([], int)
            # `build_xy` gives a DataFrame — it is taken by positional index
            Xs = Xg.iloc[rows_] if hasattr(Xg, "iloc") else Xg[rows_]
            gs, gm, _ = _timed(
                lambda: int(np.argmin(m.predict(Xs))), a.reps)
            row["gbdt_med"], row["gbdt_max"] = gs, gm
        out["shapes"][f"{p.M}x{p.N}x{p.K}"] = row
        print(f"  {f'{p.M}x{p.N}x{p.K}':22s} {len(df):7,d} "
              f"{med:8.0f}/{mx:<7.0f} {med2:8.0f}/{mx2:<7.0f} "
              f"{vs:6.0f}/{vm:<7.0f} {gs:6.0f}/{gm:<7.0f}", flush=True)

    def agg(k):
        v = [r[k] for r in out["shapes"].values() if k in r]
        return (float(np.median(v)), float(np.max(v))) if v else (
            float("nan"), float("nan"))

    print("\n  the breakdown (median, µs) — where the time goes")
    print(f"  {'shape':22s} {'filter':>9s} {'features':>11s} "
          f"{'eval+argmin':>12s}")
    for k, r in out["shapes"].items():
        print(f"  {k:22s} {r['part_filter']:9.0f} {r['part_feat']:11.0f} "
              f"{r['part_eval']:12.0f}")
    print("\n  " + "-" * 96)
    for lab, a_, b_ in (("ours (after the filter)", "ours_med", "ours_max"),
                        ("ours (whole catalogue + filter)", "ours_cat_med",
                         "ours_cat_max"),
                        ("ours (deployment: static cols numpy)",
                         "ours_deploy_med", "ours_deploy_med"),
                        ("  of which, the features", "part_feat", "part_feat"),
                        ("  of which, eval+argmin", "part_eval", "part_eval"),
                        ("vendor", "vendor_med", "vendor_max"),
                        ("GBDT", "gbdt_med", "gbdt_max")):
        m1, _ = agg(a_)
        _, m2 = agg(b_)
        print(f"  {lab:36s} shape median {m1:9.0f} µs   "
              f"shape worst {m2:9.0f} µs")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ the filter is **a representative one**, not kernelTab's "
          "predicate")


if __name__ == "__main__":
    main()
