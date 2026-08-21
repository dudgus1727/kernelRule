"""손계산용 장난감 표. **시간을 사람이 직접 지정한다.**

§26.2 의 "알려진 답 테스트" 는 합성 생성기가 아니라 이 위에서 돈다 —
생성기가 틀렸을 때 채점기까지 같이 틀리는 것을 막기 위해서다.
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

#: 눈금/노이즈가 개입하지 않는 모델. 손계산이 정확히 맞아야 하는 테스트용.
EXACT = NoiseModel(sigma_abs_ms=0.0, sigma_rel_coef=0.0, tick_ms=0.0,
                   source="toy: 노이즈 없음")


def make_table(times_by_shape: dict[tuple[int, int, int], list[float]], *,
               noise: NoiseModel | None = None,
               feature_cols: dict[str, list] | None = None) -> PerfTable:
    """`{(M,N,K): [t0, t1, ...]}` -> PerfTable.

    config 는 `k0, k1, ...` 로 이름 붙고 split_k 는 1 로 고정한다.
    `feature_cols` 는 **전체 행 순서**의 배열이다 (가중치 테스트용).
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
                    f"feature_cols[{name!r}] 길이 {len(vals)} != 행 수 {len(df)}")
            df[name] = list(vals)
    t = df.pop("_t").to_numpy(np.float64)
    y = df[["kernel_id", "M", "N", "K", "split_k", "split_k_mode"]].copy()
    y["time_ms"] = t
    return PerfTable.from_frames(df, y, hw=HW, noise=noise or EXACT,
                                 env_hash="toy0000000000000",
                                 meta={"bundle_id": "TOY", "ok_only": False},
                                 unexpected="ignore")


def order_by_index(indices):
    """고정된 순서를 내는 `order_fn`. 채점기 검증용."""
    idx = np.asarray(indices, dtype=np.int64)

    def fn(p, cand):
        return idx

    return fn


def constant_score_order(p, cand):
    """★ 모든 후보의 점수가 같다. tie-break 만으로 순서가 정해진다 (§30.7).

    이 규칙이 무작위 선택보다 좋으면 **tie-break 가 정답을 보고 있다.**
    """
    return cand.order_by(np.zeros(cand.n))
