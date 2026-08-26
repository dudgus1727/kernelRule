"""luna-s0 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      ? / ?
추론      ?
피처 표시  ?

구조 홀드아웃 1.0573  (표본내 1.0482)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = f.has_spill * (f.spill_magnitude + f.reg_pressure) * w[0]
    s = s + np.where(p.is_memory_bound, np.log2(f.traffic_amplification), f.log_inst_total) * w[1]
    s = s + np.where(p.is_memory_bound, f.sm_idle_cost, f.occupancy_deficit + f.sm_idle_cost) * w[2]
    s = s + np.log2(np.maximum(f.edge_waste, f.traffic_amplification)) * w[3]
    s = s + np.where(f.is_two_stage, f.pipeline_warmup_frac, np.square(f.pipeline_warmup_frac) * (f.smem_pressure + f.reg_pressure)) * w[4]
    s = s + (f.split_k_cost + f.log_workspace_bytes - f.waves * f.log_grid_tiles / np.maximum(f.log_mainloop_iters, f.waves)) * w[5]
    s = s + f.is_two_stage * (p.can_use_cp_async + f.smem_pressure) * w[6]
    s = s + f.tile_aspect_imbalance * np.log2(f.traffic_amplification) / (f.traffic_amplification + np.maximum(f.edge_waste, f.sm_idle_cost)) * w[7]
    return s


W_FITTED = {
    'long': [12.039729, -0.489001, 1.209401, 0.873762, 6.875513, 0.028664, 6.83006, 1.390639],
    'short': [15.624592, -0.378271, 0.980876, 1.077807, 7.266435, 0.003582, 10.37789, 2.084212],
}
