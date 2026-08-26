"""lunaNAMES-s0 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      gpt-5.6-luna / responses
추론      medium
피처 표시  names

구조 홀드아웃 1.0688  (표본내 1.0657)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = f.log_dram_traffic * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + f.spill_magnitude * w[2]
    s = s + f.log_workspace_bytes * np.exp(p.log_sol_ms) * w[3]
    s = s + np.where(p.is_memory_bound, f.edge_waste, np.sqrt(np.square(f.edge_waste) + np.square(f.tile_aspect_imbalance))) * w[4]
    s = s + np.where(p.is_memory_bound, f.split_k_cost * np.log2(p.K), f.split_k_cost / np.log2(p.K)) * w[5]
    s = s + np.maximum(f.reg_pressure, f.smem_pressure) * f.occupancy_deficit / f.log_inst_total * w[6]
    s = s + (f.pipeline_warmup_frac + f.is_two_stage) * w[7]
    return s


W_FITTED = {
    'long': [1.728, 2.46, 5.11, 1.27, 0.57, 2.624, 2.47, 4.22],
    'short': [1.08, 2.46, 5.11, 1.27, 0.57, 2.624, 2.47, 4.22],
}
