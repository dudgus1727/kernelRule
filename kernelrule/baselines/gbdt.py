"""GBDT ranker — **the ceiling of a learned model** (§9, §30.6).

This is where we measure how much a rule cannot hold. The gap between the
hand rule and GBDT is the content of this research, so this number is
**more honest the more optimistic it is** — give GBDT every advantage (all
raw columns, plenty of trees).

## What it learns

The target is `log(t / t_best_of_shape)` — **relative order within a shape
only**. Learning absolute time means relearning everything when the GPU
changes, whereas dimensionless quantities transfer (§8.1).

## ★ Measured times and anything derived from them are not features

It uses `load_for_ranking`, so `ANSWER_COLS` is structurally excluded.
`difficulty` and `distinct_time_frac` are in there too — unknown at
deployment time, so GBDT does not get them either. The answer enters only
through the target.

## The split changes the number a lot

    block (M > 2048)     11 held-out shapes   **primary.** Really tests
                                              shape generalisation
    shape-wise 5-fold    all 66 shapes        optimistic ceiling; effectively
                                              interpolation

In 5-fold, M=1024 is in training and M=1000 in validation. The gap between
the two numbers measures **how hard shape generalisation is**. Report both.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_xy", "fit_predict_block", "fit_predict_kfold",
           "order_fn_from_scores", "GBDT_PARAMS"]

GBDT_PARAMS = dict(objective="regression", n_estimators=600,
                   learning_rate=0.05, num_leaves=63, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   reg_lambda=1.0, n_jobs=8, verbose=-1, random_state=0)

#: Excluded from features: identifiers and constant metadata.
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
    # Categoricals as codes, the rest numeric. String identifiers are dropped.
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
    """Block split. `holdout_pred(Problem) -> bool` decides the holdout."""
    X, y, g, cols, shapes = build_xy(table)
    held = np.asarray([holdout_pred(p) for p in shapes])
    if not held.any() or held.all():
        raise ValueError("the block split emptied one side (§26.4).")
    mask_tr = ~held[g]
    m = _fit(X[mask_tr], y[mask_tr])
    pred = np.full(len(y), np.nan)
    pred[~mask_tr] = m.predict(X[~mask_tr])
    imp = dict(sorted(zip(cols, m.feature_importances_, strict=True),
                      key=lambda kv: -kv[1]))
    return pred, g, shapes, held, imp


def fit_predict_kfold(table, n_folds: int = 5, seed: int = 0, **kw):
    """Shape-wise k-fold. An **optimistic ceiling** (effectively
    interpolation)."""
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
    """Per-shape predicted scores -> `order_fn`. **The tie-break uses config
    identity only** (§30.7)."""
    by_shape = {}
    for gi, p in enumerate(shapes):
        by_shape[p.key] = pred[g == gi]

    def order_fn(p, cand):
        s = by_shape[p.key]
        if not np.all(np.isfinite(s)):
            # Shapes used in training have no prediction. They must be
            # excluded from scoring.
            raise ValueError(f"{p.key}: no prediction (it is a training "
                             "shape)")
        return cand.order_by(s)

    return order_fn
