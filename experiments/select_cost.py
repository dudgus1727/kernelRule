"""★ **선택 비용** — 고르는 데 몇 µs 걸리나. LLM 0회 · GPU 0회.

    python3 experiments/select_cost.py
    python3 experiments/select_cost.py --shapes 6 --reps 50

§2 의 첫 제약이 "µs 안에 골라야 한다" 인데 **우리 쪽 수치가 없었다.**
regret 만 쟀다. 여기서 잰다.

## 세 팔 — 같은 프로세스, 같은 파이썬

```
A 우리    피처 계산 -> 규칙 평가 -> argmin   ★ 피처 계산을 **포함**한다
B 벤더    nvMatmulHeuristics.get_with_mnk 한 번  (파이썬 바인딩 포함)
C GBDT    학습된 모델로 그 형상 후보 전부 예측 -> argmin
```

실험 계획서 `docs/artifacts/select-cost-prereg.md`.
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
#: 대표값 실행. r11 의 규칙을 쓴다 (D-140).
RUNS = [f"F3rw-p8-nan-s{i}" for i in range(6)]
ROUND = 11


def _rule(run: str, rnd: int) -> dict:
    f = Path("runs") / run / "bests.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    at = [e for e in rows if e["round"] <= rnd]
    return max(at, key=lambda e: e["round"])


def _timed(fn, reps: int, warmup: int = 5) -> tuple[float, float, float]:
    """(중앙 µs, 최악 µs, 평균 µs). ★ 최악을 반드시 낸다 — 배포 기준이다."""
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
                    help="형상 몇 개를 잴 것인가 (후보 수 순으로 고루)")
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

    # ★ 규칙이 실제로 쓰는 피처만 계산한다 — 배포에서 안 쓰는 것을 계산할
    #   이유가 없다. 이름을 코드에서 뽑는다.
    cfg_feats = [f for f in REGISTRY.items(shape_level=False)
                 if f"f.{f.name}" in rule["code"]]
    shp_feats = [f for f in REGISTRY.items(shape_level=True)
                 if f"p.{f.name}" in rule["code"]]
    print(f"규칙 {RUNS[0]}@r{ROUND}  피처 {len(cfg_feats)}개 "
          f"(config) + {len(shp_feats)}개 (형상)  가중치 {len(w)}")

    shapes = sorted(T.shapes(), key=lambda p: len(T.frame_for(p)))
    pick = [shapes[i] for i in
            np.linspace(0, len(shapes) - 1, a.shapes).astype(int)]

    # -- 카탈로그 (전수 상한) ------------------------------------------------
    X = T._X
    key = ["kernel_id", "split_k", "split_k_mode"]
    cat = X.drop_duplicates(key).reset_index(drop=True)
    print(f"카탈로그 {len(cat):,} 조합 · 형상별 후보 "
          f"{len(T.frame_for(shapes[0])):,}~{len(T.frame_for(shapes[-1])):,}")

    # 대표적인 필터. ⚠️ kernelTab 의 실제 판정식이 아니다 — **비용의 자릿수**를
    # 재는 것이다. 정적 배열에 대한 벡터 비교 넷.
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

    # -- 벤더 ---------------------------------------------------------------
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
        print(f"벤더 프리셋 {preset_for(hw.name)}")
    except Exception as e:                                  # noqa: BLE001
        print(f"⚠️ 벤더 팔 없음: {type(e).__name__}: {e}")

    # -- GBDT ---------------------------------------------------------------
    gbdt = None
    try:
        from kernelrule.baselines.gbdt import build_xy
        from lightgbm import LGBMRegressor
        Xg, yg, g, cols, gshapes = build_xy(T)
        m = LGBMRegressor(n_estimators=200, num_leaves=63, verbose=-1)
        m.fit(Xg, yg)
        gbdt = (m, Xg, g, gshapes)
        print(f"GBDT 학습 완료 — 피처 {len(cols)}개")
    except Exception as e:                                  # noqa: BLE001
        print(f"⚠️ GBDT 팔 없음: {type(e).__name__}: {e}")

    print("\n" + "=" * 100)
    print(f"선택 비용 (µs) — 반복 {a.reps}회, 워밍업 5회. ★ 중앙 / 최악")
    print("=" * 100)
    print(f"  {'형상':22s} {'후보':>7s} {'우리 필터후':>16s} "
          f"{'우리 전수':>16s} {'벤더':>14s} {'GBDT':>14s}")
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
            ok = _filter(p)                       # ★ 필터 비용 포함
            d = cat.loc[ok]
            cols_ = {f.name: M._vector(f, p, d, i) for f in cfg_feats}
            from kernelrule.core.matrix import Feats, ShapeInfo
            s = fn(Feats(cols_), ShapeInfo(i), hw, w)
            return int(np.argmin(np.asarray(s)))

        # ★ 내역 — 어디에 시간이 가나. 판정이 아니라 진단이다.
        from kernelrule.core.matrix import Feats, ShapeInfo
        d_np = {f.name: None for f in cfg_feats}
        _feat = _timed(lambda: {f.name: M._vector(f, p, df, info)
                                for f in cfg_feats}, a.reps)[0]
        _cols = {f.name: M._vector(f, p, df, info) for f in cfg_feats}
        _eval = _timed(lambda: int(np.argmin(np.asarray(
            fn(Feats(_cols), ShapeInfo(info), hw, w)))), a.reps)[0]
        _filt = _timed(lambda: int(_filter(p).sum()), a.reps)[0]
        # ★ 배포형 — 정적 열은 numpy 로 **미리 변환**해 둔다 (카탈로그는
        #   배포 시점에 알려져 있다). 피처는 여전히 매번 계산한다.
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
            # `build_xy` 가 DataFrame 을 준다 — 위치 색인으로 잡는다
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

    print("\n  내역 (중앙, µs) — 어디에 시간이 가나")
    print(f"  {'형상':22s} {'필터':>9s} {'피처 계산':>11s} {'평가+argmin':>12s}")
    for k, r in out["shapes"].items():
        print(f"  {k:22s} {r['part_filter']:9.0f} {r['part_feat']:11.0f} "
              f"{r['part_eval']:12.0f}")
    print("\n  " + "-" * 96)
    for lab, a_, b_ in (("우리 (필터 후)", "ours_med", "ours_max"),
                        ("우리 (카탈로그 전수 + 필터)", "ours_cat_med",
                         "ours_cat_max"),
                        ("우리 (배포형: 정적열 numpy)", "ours_deploy_med",
                         "ours_deploy_med"),
                        ("  그중 피처 계산", "part_feat", "part_feat"),
                        ("  그중 평가+argmin", "part_eval", "part_eval"),
                        ("벤더", "vendor_med", "vendor_max"),
                        ("GBDT", "gbdt_med", "gbdt_max")):
        m1, _ = agg(a_)
        _, m2 = agg(b_)
        print(f"  {lab:28s} 형상 중앙 {m1:9.0f} µs   형상 최악 {m2:9.0f} µs")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 필터는 **대표적인 것**이지 kernelTab 의 판정식이 아니다")


if __name__ == "__main__":
    main()
