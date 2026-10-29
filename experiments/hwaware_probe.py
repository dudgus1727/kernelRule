"""D-200 — Chandran 외 (arXiv 2609.35587) 의 하드웨어 인지 피처 + XGBoost 를 ★ 우리 표 · 우리 fold ·
우리 채점 함수로 돌린다. **LLM 0회 · GPU 0** (XGBoost 는 CPU).

    python3 -m experiments.hwaware_probe --gpu a6000 --out docs/artifacts/hwaware/a6000.json
    python3 -m experiments.hwaware_probe --smoke     # 학습 쪽만 — ⛔ holdout 을 읽지 않는다

## 가져온 것 — `spcl/cutlass-selector` 8a243cb (BSD-3, 코드는 실행하지 않고 공식을 옮겼다)

```
피처      src/model/features.py featurize() 의 G0~G7 — 수치 40 개 + 범주
XGBoost   src/model/train_xgb.py DEFAULT_PARAMS · 916 트리 · hist · seed 42 · 목적 MSE (y_norm)
          — 논문 표 12 의 "XGBoost (full) 6.4%" 가 이 팔이다
Ridge     benchmarks/baselines/learned_and_analytical/train_linear.py RidgeLinear — 닫힌 꼴,
          수치 열 표준화 (nan · inf -> 0) + 범주 one-hot
```

## 우리 공간에 맞춘 것 (A1~A6, D-200)

```
A1 split-K    그들의 Stream-K 자리. 격자 = tiles_m·tiles_n·split_k, CTA 하나의 k 반복
              = ceil(ceil(K/split_k)/tile_k). streamk_applicability 는 쪼개지 않은 격자의
              1 - 마지막 wave 효율, is_streamk 는 split_k > 1
A2 스레드      32·(tm/wm)(tn/wn)(tk/wk) — 2.x 는 warp 타일이 정한다. 소비자 warpgroup = 1
A3 smem       그들 smem_total 은 mainloop 와 epilogue 저장소의 union (WS 스케줄) — 2.x 도 union 이고,
              그 값이 표의 정적 열 smem_computed 다 (mainloop 공식 stages·(tm·tk + tn·tk)·2 와 a6000 행의
              96% 에서 같고, 나머지는 epilogue 가 더 크다). bytes_per_stage 는 공식 그대로
A4 cluster    2.x 에는 없다 — 1×1 (그들 공식 그대로, 상수 열이 된다)
A5 손잡이      그들의 구조 핵심은 "config 그대로" — 우리 손잡이를 더한다. 수치 warp_m·n·k · split_k ·
              swizzle_n · align_a·b·c, 범주 pipeline_kind · split_k_mode · swizzle_type. 두 집합 모두에
A6 GPU 상수    우리 Hardware (실효 피크 · 실효 대역폭 · SM · smem/블록 · L2 · 레지스터). 블록 상한 32 는 그대로
```

⛔ 컴파일 결과 (레지스터 수 · spill · SASS) 와 kerneltab 의 파생 열 (waves · tail_waste · occupancy ...) 은
넣지 않는다 — 그들 피처가 아니다. A2 는 표의 정적 열 threads 와 같은지 대조한다.

## 절차 — D-199 와 같은 자

```
표 · fold   네 GPU · nkband k=4 (f1_pipeline._splits) — holdout 은 a6000 · 5090 65, 4090 · H100 63 형상
학습        fold 의 train 형상 전부의 유효 config 전부. 하이퍼파라미터 고정 — 튜닝 · inner 분할 없음
채점        evaluate_scores (규칙과 같은 함수 · 같은 동률 깨기). 점수 = -예측
집계        4 fold holdout 을 모은 기하평균 (a6000_probe 의 holdout_gm_65 와 같은 정의)
```
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = (0, 1, 2, 3)
DESIGN = "nkband"
SOURCE = "spcl/cutlass-selector 8a243cb"

# ★ src/model/train_xgb.py — Optuna trial #202 에서 잠근 값 그대로
XGB_PARAMS = dict(eta=0.14994338035621416, max_depth=7,
                  min_child_weight=2.129510939594175,
                  subsample=0.8006253365599527,
                  colsample_bytree=0.905985078442277,
                  reg_lambda=0.0329689343120563,
                  reg_alpha=0.08879582214664276,
                  gamma=0.019394121795509252)
XGB_TREES = 916
RIDGE_ALPHA = 1.0
MAX_CTA_PER_SM = 32         # 그들 상수 그대로 (A6)
F16 = 2.0                   # 피연산자 · 출력 바이트 — 표가 f16 뿐이다 (확인한다)

# 그들 NUMERIC_FEATURES 의 순서 그대로
THEIR_NUMERIC = [
    "log2_M", "log2_N", "log2_K", "tile_m", "tile_n", "tile_k", "stages",
    "cluster_m_f", "cluster_n_f", "sm_subscription", "last_wave_eff",
    "tiles_m", "tiles_n", "k_iters", "m_quant_waste", "n_quant_waste",
    "mainloop_compute_intensity", "tile_aspect_ratio",
    "problem_arith_intensity", "restream_factor", "working_set_vs_L2",
    "fits_L2", "resident_panel_vs_L2", "smem_total", "smem_frac",
    "blocks_per_sm", "reg_pressure_proxy", "reg_limited_blocks_per_sm",
    "true_blocks_per_sm", "bytes_per_stage", "producer_consumer_ratio",
    "pipeline_fill_frac", "stage_amortization", "stages_vs_smem",
    "streamk_applicability", "is_streamk", "cluster_size", "cluster_fits_m",
    "cluster_fits_n", "cluster_overshoots"]
THEIR_STRUCTURAL = ["log2_M", "log2_N", "log2_K", "tile_m", "tile_n",
                    "tile_k", "stages", "cluster_m_f", "cluster_n_f"]
# A5 — 우리 공간의 손잡이 (두 집합 모두에)
OUR_KNOBS = ["warp_m", "warp_n", "warp_k", "split_k", "swizzle_n",
             "align_a", "align_b", "align_c"]
CATEGORICAL = {"pipeline_kind": ["multistage", "pipelined"],
               "split_k_mode": ["parallel", "serial"],
               "swizzle_type": ["horizontal", "identity"]}
FEATURE_SETS = {"full": THEIR_NUMERIC + OUR_KNOBS,
                "structural": THEIR_STRUCTURAL + OUR_KNOBS}
ARMS = {"xgb_full": ("xgb", "full"), "xgb_struct": ("xgb", "structural"),
        "ridge_full": ("ridge", "full")}


def _gm(v) -> float:
    v = np.asarray(v, float)
    return float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")


def featurize(df: pd.DataFrame, hw) -> pd.DataFrame:
    """그들 featurize() 를 우리 config 열에 — 시간 열은 읽지 않는다."""
    if set(df["dtype"].astype(str)) != {"f16"}:
        raise ValueError("f16 표만 다룬다 (바이트 폭 2 를 가정한다).")
    ba = bb = bc = F16
    n_sms = float(hw.sm_count)
    smem_cap = float(hw.smem_per_block)
    l2 = float(hw.l2_bytes)
    peak = float(hw.peak_tflops_f16) * 1e12
    bw = float(hw.bandwidth_gbps) * 1e9
    regs = float(hw.regs_per_sm)

    M, N, K = (df[c].to_numpy(np.float64) for c in ("M", "N", "K"))
    tm, tn, tk = (df[c].to_numpy(np.float64)
                  for c in ("tile_m", "tile_n", "tile_k"))
    wm, wn, wk = (df[c].to_numpy(np.float64)
                  for c in ("ext_warp_m", "ext_warp_n", "ext_warp_k"))
    st = df["ext_stages"].to_numpy(np.float64)
    sk = df["split_k"].to_numpy(np.float64)
    cm = cn = np.ones_like(M)                                  # A4

    threads = 32.0 * (tm / wm) * (tn / wn) * (tk / wk)         # A2
    consumer_wgs = 1.0
    tiles_m, tiles_n = np.ceil(M / tm), np.ceil(N / tn)
    num_tiles = tiles_m * tiles_n * sk                         # A1
    k_iters = np.ceil(np.ceil(K / sk) / tk)                    # A1
    smem_stage = tm * tk * ba + tn * tk * bb
    smem_total = df["smem_computed"].to_numpy(np.float64)      # A3 — union
    if np.any(smem_total < st * smem_stage):
        raise ValueError("smem_computed 가 mainloop 공식보다 작다 — A3 의 전제가 틀렸다.")

    blocks_per_sm = np.minimum(np.floor(smem_cap / smem_total), MAX_CTA_PER_SM)
    reg_pressure_proxy = (tm * tn) / threads
    reg_limited = np.floor(regs / (reg_pressure_proxy * threads))
    true_blocks = np.maximum(1.0, np.minimum(blocks_per_sm, reg_limited))
    slots = n_sms * true_blocks
    sm_subscription = num_tiles / slots
    last_wave_eff = num_tiles / (np.ceil(sm_subscription) * slots)
    dp_tiles = tiles_m * tiles_n                               # A1 — 쪼개기 전
    dp_eff = dp_tiles / (np.ceil(dp_tiles / slots) * slots)
    ab = ba * M * K + bb * K * N
    smem_frac = smem_total / smem_cap
    consumer_time = (2.0 * tm * tn * tk) / (consumer_wgs * (peak / n_sms))
    producer_time = smem_stage / bw

    out = pd.DataFrame({
        "log2_M": np.log2(M), "log2_N": np.log2(N), "log2_K": np.log2(K),
        "tile_m": tm, "tile_n": tn, "tile_k": tk, "stages": st,
        "cluster_m_f": cm, "cluster_n_f": cn,
        "sm_subscription": sm_subscription, "last_wave_eff": last_wave_eff,
        "tiles_m": tiles_m, "tiles_n": tiles_n, "k_iters": k_iters,
        "m_quant_waste": (tiles_m * tm - M) / M,
        "n_quant_waste": (tiles_n * tn - N) / N,
        "mainloop_compute_intensity": (2.0 * tm * tn) / (tm * ba + tn * bb),
        "tile_aspect_ratio": np.log2(tm / tn),
        "problem_arith_intensity": (2.0 * M * N * K)
        / (ba * M * K + bb * K * N + bc * M * N),
        "restream_factor": (tiles_n * ba * M * K + tiles_m * bb * K * N) / ab,
        "working_set_vs_L2": ab / l2,
        "fits_L2": (ab < l2).astype(np.float64),
        "resident_panel_vs_L2": np.minimum(ba * tm * K, bb * tn * K) / l2,
        "smem_total": smem_total, "smem_frac": smem_frac,
        "blocks_per_sm": blocks_per_sm,
        "reg_pressure_proxy": reg_pressure_proxy,
        "reg_limited_blocks_per_sm": reg_limited,
        "true_blocks_per_sm": true_blocks, "bytes_per_stage": smem_stage,
        "producer_consumer_ratio": consumer_time / producer_time,
        "pipeline_fill_frac": np.minimum(st, k_iters) / st,
        "stage_amortization": k_iters / st,
        "stages_vs_smem": st * smem_frac,
        "streamk_applicability": np.maximum(0.0, 1.0 - dp_eff),
        "is_streamk": (sk > 1).astype(np.float64),
        "cluster_size": cm * cn, "cluster_fits_m": tiles_m / cm,
        "cluster_fits_n": tiles_n / cn,
        "cluster_overshoots": ((cm > tiles_m) | (cn > tiles_n)).astype(np.float64),
        "warp_m": wm, "warp_n": wn, "warp_k": wk, "split_k": sk,
        "swizzle_n": df["ext_swizzle_n"].to_numpy(np.float64),
        "align_a": df["align_a"].to_numpy(np.float64),
        "align_b": df["align_b"].to_numpy(np.float64),
        "align_c": df["align_c"].to_numpy(np.float64),
    }, index=df.index).astype(np.float32)
    for c, levels in CATEGORICAL.items():
        src = "ext_swizzle_type" if c == "swizzle_type" else c
        vals = df[src].astype(str)
        if not set(vals) <= set(levels):
            raise ValueError(f"{c}: 고정 수준 밖의 값 {sorted(set(vals) - set(levels))}")
        out[c] = pd.Categorical(vals, categories=levels)
    # A2 대조 — 표의 정적 열과 다르면 옮긴 공식이 틀린 것이다. A3 은 epilogue 가 mainloop 보다 큰 행의 비율
    out.attrs["threads_match"] = float(np.mean(threads == df["threads"].to_numpy()))
    out.attrs["smem_epi_larger"] = float(np.mean(smem_total > st * smem_stage))
    return out


class _Data:
    """한 GPU 의 표 · fold · 형상별 피처 (시간은 채점과 y 에만)."""

    def __init__(self, gpu: str):
        from experiments import a6000_probe as P
        from experiments.f1_pipeline import _splits

        P._G.clear()
        P._G["gpu"] = gpu
        self.gpu = gpu
        self.t = P._table()
        self.splits = {f: _splits(self.t, fold=f, k=4, design=DESIGN)
                       for f in FOLDS}
        used = {p.key: p for f in FOLDS for p in
                (*self.splits[f].train.shapes, *self.splits[f].val.shapes)}
        self.X, self.y = {}, {}
        tm, sm = [], []
        for k, p in used.items():
            x = featurize(self.t.frame_for(p), self.t.hw)
            tm.append(x.attrs["threads_match"])
            sm.append(x.attrs["smem_epi_larger"])
            self.X[k] = x
            tt = np.asarray(self.t.times_of(p), np.float64)
            self.y[k] = (self.t.best_time(p) / tt).astype(np.float32)
        self.threads_match = float(np.mean(tm))
        self.smem_epi_larger = float(np.mean(sm))

    def stack(self, shapes, cols):
        X = pd.concat([self.X[p.key][cols] for p in shapes], ignore_index=True)
        y = np.concatenate([self.y[p.key] for p in shapes])
        return X, y


def _ridge_design(X: pd.DataFrame, num: list[str], ref=None):
    """그들 build_X — 수치 표준화 (nan · inf -> 0), 범주 one-hot (고정 수준)."""
    A = X[num].to_numpy(np.float64)
    A[~np.isfinite(A)] = 0.0
    if ref is None:
        mu, sd = A.mean(0), A.std(0)
        sd[sd == 0] = 1.0
        ref = (mu, sd)
    A = (A - ref[0]) / ref[1]
    hot = [(X[c].to_numpy() == lv).astype(np.float64)[:, None]
           for c, levels in CATEGORICAL.items() for lv in levels]
    return np.hstack([A, *hot]), ref


def _fit_predict(kind, X_tr, y_tr, preds, num, seed, jobs):
    """적합 하나 -> {이름: 예측}. preds = {이름: X}."""
    cols = num + list(CATEGORICAL)
    if kind == "xgb":
        import xgboost as xgb

        m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                             device="cpu", enable_categorical=True,
                             n_estimators=XGB_TREES, random_state=seed,
                             n_jobs=jobs, **XGB_PARAMS)
        m.fit(X_tr[cols], y_tr)
        gain = m.get_booster().get_score(importance_type="gain")
        top = dict(sorted(gain.items(), key=lambda kv: -kv[1])[:15])
        return {k: m.predict(v[cols]) for k, v in preds.items()}, top
    A, ref = _ridge_design(X_tr, num)
    mu_y = float(y_tr.mean())
    Am = A.mean(0)
    Ac = A - Am
    coef = np.linalg.solve(Ac.T @ Ac + RIDGE_ALPHA * np.eye(A.shape[1]),
                           Ac.T @ (y_tr - mu_y))
    b = mu_y - float(Am @ coef)
    return {k: _ridge_design(v, num, ref)[0] @ coef + b
            for k, v in preds.items()}, None


def _score(d: _Data, shapes, pred: np.ndarray):
    """예측을 형상별 점수로 나눠 evaluate_scores 로 채점 (점수 낮을수록 먼저)."""
    from kernelrule.core.scoring import evaluate_scores

    by, at = {}, 0
    for p in shapes:
        n = len(d.y[p.key])
        by[p.key] = -np.asarray(pred[at:at + n], np.float64)
        at += n
    assert at == len(pred)
    ev = evaluate_scores(lambda p, c: by[p.key], d.t, list(shapes), ks=(1,))
    reg = {f"{p.M}x{p.N}x{p.K}": float(r)
           for p, r in zip(ev.shapes, ev.regret[:, 0], strict=True)}
    hit = {f"{p.M}x{p.N}x{p.K}": bool(h)
           for p, h in zip(ev.shapes, ev.hit[:, 0], strict=True)}
    picks = {}
    for p in shapes:
        c = d.t.candidates(p)
        i = int(c.top_k(by[p.key], 1)[0])
        picks[f"{p.M}x{p.N}x{p.K}"] = {
            "kernel_id": str(c.kernel_id[i]), "split_k": int(c.split_k[i]),
            "split_k_mode": str(c.split_k_mode[i])}
    return reg, hit, picks


def run_arm(d: _Data, arm: str, seed: int, jobs: int, *, smoke=False) -> dict:
    kind, fs = ARMS[arm]
    num = FEATURE_SETS[fs]
    cols = num + list(CATEGORICAL)
    per_fold, pooled, pooled_tr = {}, {}, []
    for f in (FOLDS[:1] if smoke else FOLDS):
        sp = d.splits[f]
        X_tr, y_tr = d.stack(sp.train.shapes, cols)
        preds = {"train": X_tr}
        if not smoke:
            preds["hold"] = d.stack(sp.val.shapes, cols)[0]
        t0 = time.perf_counter()
        out, top = _fit_predict(kind, X_tr, y_tr, preds, num, seed, jobs)
        secs = time.perf_counter() - t0
        tr_reg, _, _ = _score(d, sp.train.shapes, out["train"])
        pf = {"train_gm": round(_gm(list(tr_reg.values())), 6),
              "n_train_rows": len(y_tr), "n_train_shapes": len(sp.train.shapes),
              "fit_seconds": round(secs, 1), "importance_gain_top15": top}
        if not smoke:
            reg, hit, picks = _score(d, sp.val.shapes, out["hold"])
            pf.update({"holdout_gm": round(_gm(list(reg.values())), 6),
                       "n_hold": len(reg), "regret": reg, "hit": hit,
                       "picks": picks})
            pooled.update(reg)
        pooled_tr += list(tr_reg.values())
        per_fold[f] = pf
        print(f"  {d.gpu} {arm} seed {seed} fold {f}: train {pf['train_gm']:.4f} "
              f"({pf['n_train_rows']:,} 행, {secs:.0f}s)", flush=True)
    agg = {"train_gm": round(_gm(pooled_tr), 6)}
    if not smoke:
        r = np.asarray(list(pooled.values()))
        agg.update({
            "holdout_gm": round(_gm(r), 6), "n_hold": len(r),
            # 그들 지표 — 1 - 처리량 비의 산술평균, 5% 안 비율
            "their_regret_mean": round(float(np.mean(1.0 - 1.0 / r)), 6),
            "their_within5": round(float(np.mean(1.0 / r >= 0.95)), 6),
            "hit_rate": round(float(np.mean([v for fr in per_fold.values()
                                             for v in fr["hit"].values()])), 6)})
    return {"arm": arm, "model": kind, "feature_set": fs, "seed": seed,
            "aggregate": agg, "per_fold": per_fold}


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="a6000")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="a6000 fold 0, 학습 쪽만 (holdout 을 읽지 않는다), 시간 재기")
    a = ap.parse_args()

    import xgboost

    t0 = time.perf_counter()
    d = _Data(a.gpu)
    print(f"{a.gpu}: 피처 {time.perf_counter() - t0:.0f}s — threads 일치 "
          f"{d.threads_match:.4f} · epilogue smem 이 더 큰 행 {d.smem_epi_larger:.4f}",
          flush=True)
    if a.smoke:
        r = run_arm(d, "xgb_full", 42, a.jobs, smoke=True)
        print(json.dumps(r["per_fold"][0], ensure_ascii=False, indent=1))
        return
    plan = [("xgb_full", 42), ("xgb_struct", 42), ("ridge_full", 0),
            ("xgb_full", 0), ("xgb_full", 1)]
    res = []
    for arm, seed in plan:
        res.append(run_arm(d, arm, seed, a.jobs))
        r = res[-1]["aggregate"]
        print(f"{a.gpu} {arm} seed {seed}: train {r['train_gm']:.4f}", flush=True)
    out = {"gpu": a.gpu, "source": SOURCE, "xgboost": xgboost.__version__,
           "xgb_params": XGB_PARAMS, "xgb_trees": XGB_TREES,
           "ridge_alpha": RIDGE_ALPHA, "feature_sets": FEATURE_SETS,
           "categorical": CATEGORICAL,
           "checks": {"threads_match": d.threads_match,
                      "smem_epi_larger": d.smem_epi_larger},
           "arms": res}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {a.out}  ({math.ceil(time.perf_counter() - t0)}s)")


if __name__ == "__main__":
    main()
