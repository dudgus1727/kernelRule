"""GBDT 랭커 — **학습 모델 상한** (§9, §30.6).

규칙이 못 담는 것이 얼마인지를 재는 자리다. 손규칙과 GBDT 사이의 격차가
이 연구의 내용이므로, 이 값은 **낙관적일수록 정직하다** — GBDT 에 최대한
유리하게 준다 (원시 컬럼 전부, 넉넉한 트리 수).

## 무엇을 학습하는가

목표값은 `log(t / t_best_of_shape)` — **형상 안의 상대 순위만** 배운다.
절대 시간을 배우면 GPU 가 바뀔 때 전부 다시 배워야 하지만 무차원 량은
전이된다 (§8.1).

## ★ 측정 시간과 그로부터 유도된 값은 피처에 넣지 않는다

`load_for_ranking` 을 쓰므로 `ANSWER_COLS` 가 구조적으로 빠져 있다.
`difficulty` / `distinct_time_frac` 도 거기 있다 — 배포 시점에 알 수 없는
값이라 GBDT 에도 주지 않는다. 목표값에만 정답이 들어간다.

## 분할이 값을 크게 바꾼다

    블록 (M > 2048)   홀드아웃 11형상   **주 지표.** 형상 일반화를 실제로 시험
    형상 단위 5-fold  전 66형상         낙관적 상한. 사실상 보간이다

5-fold 는 M=1024 가 학습에, M=1000 이 검증에 들어간다. 두 값의 격차가
**형상 일반화가 얼마나 어려운가**의 척도다. 둘 다 보고한다.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_xy", "fit_predict_block", "fit_predict_kfold",
           "order_fn_from_scores", "GBDT_PARAMS"]

GBDT_PARAMS = dict(objective="regression", n_estimators=600,
                   learning_rate=0.05, num_leaves=63, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   reg_lambda=1.0, n_jobs=8, verbose=-1, random_state=0)

#: 피처에서 뺄 것. 식별자와 상수 메타데이터.
_DROP = {"kernel_id", "arch", "dtype", "acc_dtype", "layout_a", "layout_b",
         "layout_c", "split_k_mode", "pipeline_kind", "ext_swizzle_type",
         "workspace_dtype", "partials_dtype", "env_hash", "bundle_id",
         "gpu_name", "cutlass_commit", "nvcc_arch", "clock_locked"}


def build_xy(table, shapes=None):
    """(X, y, group, columns). `y = log(t / best_of_shape)`."""
    import pandas as pd

    shapes = list(shapes if shapes is not None else table.shapes())
    frames, ys, groups = [], [], []
    for gi, p in enumerate(shapes):
        df = table.frame_for(p)
        t = np.asarray(table.times_of(p), dtype=np.float64)
        best = table.best_time(p)
        frames.append(df)
        ys.append(np.log(t / best))
        groups.append(np.full(len(df), gi, dtype=np.int64))
    X = pd.concat(frames, ignore_index=True)
    # 범주형은 코드로, 나머지는 수치로. 문자열 식별자는 버린다.
    for c in list(X.columns):
        if c in _DROP:
            if str(X[c].dtype) == "category" or X[c].dtype == object:
                if c in ("split_k_mode", "pipeline_kind", "ext_swizzle_type"):
                    X[c] = X[c].astype("category").cat.codes
                else:
                    X = X.drop(columns=[c])
            continue
        if str(X[c].dtype) in ("bool",):
            X[c] = X[c].astype(np.int8)
    X = X.select_dtypes(include=[np.number, "bool"]).astype(np.float32)
    return (X, np.concatenate(ys), np.concatenate(groups),
            list(X.columns), shapes)


def _fit(Xtr, ytr):
    from lightgbm import LGBMRegressor

    m = LGBMRegressor(**GBDT_PARAMS)
    m.fit(Xtr, ytr)
    return m


def fit_predict_block(table, holdout_pred, **kw):
    """블록 분할. `holdout_pred(Problem) -> bool` 이 홀드아웃을 정한다."""
    X, y, g, cols, shapes = build_xy(table)
    held = np.asarray([holdout_pred(p) for p in shapes])
    if not held.any() or held.all():
        raise ValueError("블록 분할이 한쪽을 비웠다 (§26.4).")
    mask_tr = ~held[g]
    m = _fit(X[mask_tr], y[mask_tr])
    pred = np.full(len(y), np.nan)
    pred[~mask_tr] = m.predict(X[~mask_tr])
    imp = dict(sorted(zip(cols, m.feature_importances_, strict=True),
                      key=lambda kv: -kv[1]))
    return pred, g, shapes, held, imp


def fit_predict_kfold(table, n_folds: int = 5, seed: int = 0, **kw):
    """형상 단위 k-fold. **낙관적 상한**이다 (사실상 보간)."""
    X, y, g, cols, shapes = build_xy(table)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(shapes)) % n_folds
    pred = np.full(len(y), np.nan)
    imp_acc = np.zeros(len(cols))
    for k in range(n_folds):
        te = fold == k
        mask_te = te[g]
        m = _fit(X[~mask_te], y[~mask_te])
        pred[mask_te] = m.predict(X[mask_te])
        imp_acc += m.feature_importances_
    imp = dict(sorted(zip(cols, imp_acc, strict=True), key=lambda kv: -kv[1]))
    return pred, g, shapes, np.ones(len(shapes), dtype=bool), imp


def order_fn_from_scores(pred: np.ndarray, g: np.ndarray, shapes):
    """형상별 예측 점수 -> `order_fn`. **tie-break 는 config 정체성만** (§30.7)."""
    by_shape = {}
    for gi, p in enumerate(shapes):
        by_shape[p.key] = pred[g == gi]

    def order_fn(p, cand):
        s = by_shape[p.key]
        if not np.all(np.isfinite(s)):
            # 학습에 쓰인 형상은 예측이 없다. 채점 대상에서 빼야 한다.
            raise ValueError(f"{p.key}: 예측이 없다 (학습 형상이다)")
        return cand.order_by(s)

    return order_fn
