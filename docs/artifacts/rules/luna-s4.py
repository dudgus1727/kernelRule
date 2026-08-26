"""luna-s4 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      gpt-5.6-luna / responses
추론      medium
피처 표시  ?

구조 홀드아웃 1.0821  (표본내 1.0806)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = np.where(p.is_memory_bound, f.log_dram_traffic, f.log_inst_total) * w[0]
    s = s + (f.traffic_amplification + f.traffic_amplification * f.tile_aspect_imbalance * np.nan_to_num(f.edge_waste / (f.edge_waste + f.sm_idle_cost + f.tail_waste))) * w[1]
    s = s + f.has_spill * (f.spill_magnitude + f.has_spill) * w[2]
    if p.can_use_cp_async:
        s = s + f.is_two_stage * f.pipeline_warmup_frac * w[3]
    s = s + f.sm_idle_cost * w[4]
    s = s + f.split_k_cost * (f.waves / (f.waves + f.sm_idle_cost)) * (f.log_workspace_bytes + f.split_k_cost) * w[5]
    s = s + np.where(p.is_memory_bound, np.nan_to_num(f.occupancy_deficit * f.log_grid_tiles / (f.waves + f.log_grid_tiles) * np.exp(p.log_sol_ms) / (np.exp(p.log_sol_ms) + f.sm_idle_cost + f.occupancy_deficit)), f.smem_pressure) * w[6]
    s = s + np.square(f.reg_pressure) * w[7]
    return s


W_FITTED = {
    'long': [-0.105245, 0.594614, 30.430908, 1.341023, 0.9868, 2.624279, 1.276466, 9.590145],
    'short': [-0.08, 0.58, 29.5, 1.3, 0.83, 2.544, 1.291709, 7.16],
}
