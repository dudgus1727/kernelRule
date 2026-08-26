"""luna-s3 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      gpt-5.6-luna / responses
추론      medium
피처 표시  ?

구조 홀드아웃 1.1020  (표본내 1.0566)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + f.has_spill * (f.spill_magnitude + f.has_spill) * w[2]
    s = s + np.where(p.is_memory_bound, f.occupancy_deficit, f.log_inst_total - f.reg_pressure) * w[3]
    s = s + (f.edge_waste + f.reg_pressure + f.smem_pressure) * w[4]
    s = s + f.pipeline_warmup_frac * w[5]
    s = s + (f.split_k_cost * np.nan_to_num(f.waves / (f.waves + f.log_grid_tiles + f.occupancy_deficit)) - f.log_grid_tiles * np.nan_to_num(f.log_grid_tiles / (f.log_grid_tiles + f.waves + f.occupancy_deficit)) + f.log_workspace_bytes) * w[6]
    s = s + f.is_two_stage * w[7]
    return s


W_FITTED = {
    'long': [2.656028, 17.689044, 6.210052, -0.087361, 0.298633, 12.60577, 1.233187, 2.290497],
    'short': [2.515815, 10.493451, 3.826086, -0.191222, 0.452789, 11.911457, 0.994371, 4.350195],
}
