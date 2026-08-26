"""luna-s5 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      gpt-5.6-luna / responses
추론      medium
피처 표시  ?

구조 홀드아웃 1.1378  (표본내 1.0987)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = f.has_spill * (f.spill_magnitude + f.log_dram_traffic) * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + f.edge_waste * w[2]
    s = s + np.where(p.is_memory_bound, np.log2(f.traffic_amplification), f.log_inst_total) * w[3]
    s = s + (f.log_mainloop_iters - np.log2(np.maximum(f.waves, f.is_two_stage + p.can_use_cp_async))) * w[4]
    s = s + np.square(f.reg_pressure) * w[5]
    s = s + (f.split_k_cost * (f.log_mainloop_iters - np.log2(np.maximum(f.waves, f.is_two_stage + p.can_use_cp_async))) + f.log_workspace_bytes) * w[6]
    s = s + np.log2(f.traffic_amplification) * f.log_grid_tiles * w[7]
    return s


W_FITTED = {
    'long': [90.236116, 4.036275, 2.371666, 0.405109, -7.939687, 2.000755, 2.838171, 0.114574],
    'short': [89.013196, 3.981573, 2.339524, 0.39137, -8.050287, 1.611173, 2.799707, 0.100835],
}
