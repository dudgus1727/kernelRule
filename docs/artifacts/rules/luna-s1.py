"""luna-s1 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      ? / ?
추론      ?
피처 표시  ?

구조 홀드아웃 1.1018  (표본내 1.0875)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = f.has_spill * (f.spill_magnitude + f.has_spill) * w[0]
    s = s + np.log2(f.traffic_amplification) * w[1]
    s = s + f.edge_waste * w[2]
    s = s + f.sm_idle_cost * w[3]
    s = s + np.where(p.is_memory_bound, f.log_dram_traffic, f.log_inst_total) * w[4]
    s = s + np.square(np.maximum(f.reg_pressure, f.smem_pressure)) * w[5]
    s = s + f.log_grid_tiles * np.log2(f.traffic_amplification) * w[6]
    s = s + (f.split_k_cost - f.tail_waste) * w[7]
    return s


W_FITTED = {
    'long': [21.6, 1.1, 0.35, 3.3325, -0.12, 1.0, 0.009, 4.88],
    'short': [21.6, 1.1, 0.35, 3.3325, -0.12, 1.0, 0.009, 7.808],
}
