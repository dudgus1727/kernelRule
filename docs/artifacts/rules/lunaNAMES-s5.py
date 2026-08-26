"""lunaNAMES-s5 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      gpt-5.6-luna / responses
추론      medium
피처 표시  names

구조 홀드아웃 1.1106  (표본내 1.0849)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = f.traffic_amplification * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + (f.log_grid_tiles + f.log_dram_traffic + f.log_inst_total) * np.exp(p.log_sol_ms) * w[2]
    s = s + -np.log2(f.waves) * w[3]
    s = s + f.spill_magnitude * w[4]
    s = s + f.reg_pressure * w[5]
    if p.is_memory_bound:
        s = s + f.smem_pressure * (f.pipeline_warmup_frac - np.log2(f.traffic_amplification)) * w[6]
    s = s + f.split_k_cost * np.exp(-f.occupancy_deficit) * w[7]
    return s


W_FITTED = {
    'long': [0.314751, 1.379876, -0.051336, -0.871805, 4.696548, 7.140853, 0.607984, 7.059545],
    'short': [0.598277, 1.268255, 0.276163, -0.864222, 2.680459, 6.587092, 1.008837, 13.075421],
}
