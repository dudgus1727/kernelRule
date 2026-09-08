"""A toy table for hand calculation. **The times are given by a human.**

The "known-answer test" of §26.2 runs on this rather than on the synthetic
generator — so that a wrong generator does not take the scorer down with it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kernelrule.core.noise import NoiseModel
from kernelrule.core.table import PerfTable
from kernelrule.core.types import Hardware

HW = Hardware(name="TOY", arch="sm_86", sm_count=84, smem_per_block=101376,
              max_threads_per_sm=1536, regs_per_sm=65536,
              peak_tflops_f16=116.1, bandwidth_gbps=729.7, l2_bytes=6291456)

#: A model with no tick and no noise. For tests where the hand calculation has
#: to come out exactly.
EXACT = NoiseModel(sigma_abs_ms=0.0, sigma_rel_coef=0.0, tick_ms=0.0,
                   source="toy: no noise")


def make_table(times_by_shape: dict[tuple[int, int, int], list[float]], *,
               noise: NoiseModel | None = None,
               feature_cols: dict[str, list] | None = None) -> PerfTable:
    """`{(M,N,K): [t0, t1, ...]}` -> PerfTable.

    The configs are named `k0, k1, ...` and split_k is fixed at 1.
    `feature_cols` is an array in **whole-row order** (for the weight tests).
    """
    rows = []
    for (M, N, K), ts in times_by_shape.items():
        for i, t in enumerate(ts):
            row = {
                "M": M, "N": N, "K": K, "dtype": "f16",
                "tile_m": 128, "tile_n": 128, "tile_k": 32,
                "align_a": 8, "align_b": 8, "align_c": 8,
                "split_k": 1, "split_k_mode": "serial",
                "kernel_id": f"k{i:03d}", "arch": "sm_86",
                "regs_per_thread": 100 + i, "threads": 256,
                "max_blocks_per_sm": 2, "pipeline_kind": "multistage",
                "smem_dynamic": 32768, "spill_stores": 0, "spill_loads": 0,
                "_t": float(t),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if feature_cols:
        for name, vals in feature_cols.items():
            if len(vals) != len(df):
                raise ValueError(
                    f"feature_cols[{name!r}] has length {len(vals)} != "
                    f"{len(df)} rows")
            df[name] = list(vals)
    t = df.pop("_t").to_numpy(np.float64)
    y = df[["kernel_id", "M", "N", "K", "split_k", "split_k_mode"]].copy()
    y["time_ms"] = t
    return PerfTable.from_frames(df, y, hw=HW, noise=noise or EXACT,
                                 env_hash="toy0000000000000",
                                 meta={"bundle_id": "TOY", "ok_only": False},
                                 unexpected="ignore")


def order_by_index(indices):
    """An `order_fn` that returns a fixed order. For verifying the scorer."""
    idx = np.asarray(indices, dtype=np.int64)

    def fn(p, cand):
        return idx

    return fn


def constant_score_order(p, cand):
    """★ Every candidate has the same score. The order is decided by the
    tie-break alone (§30.7).

    If this rule beats a random pick, **the tie-break is looking at the
    answer.**
    """
    return cand.order_by(np.zeros(cand.n))
