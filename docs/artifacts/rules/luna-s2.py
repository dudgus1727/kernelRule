"""luna-s2 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      ? / ?
추론      ?
피처 표시  ?

구조 홀드아웃 1.1137  (표본내 1.0934)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.log_dram_traffic * w[1]
    s = s + np.maximum(f.spill_magnitude, f.has_spill) * w[2]
    s = s + f.log_inst_total * w[3]
    s = s + f.sm_idle_cost * w[4]
    s = s + p.can_use_cp_async * f.is_two_stage * (f.log_mainloop_iters - f.pipeline_warmup_frac) * w[5]
    s = s + f.split_k_cost * w[6]
    s = s + np.where(p.is_memory_bound, np.maximum(f.has_spill, f.reg_pressure * f.smem_pressure), np.maximum(f.reg_pressure, f.smem_pressure)) * w[7]
    return s


W_FITTED = {
    'long': [3.216384, 0.5051, 8.8011, -0.4924, 1.0301, 0.3559, 10.0897, 3.1614],
    'short': [2.01024, 0.5051, 8.8011, -0.4924, 1.0301, 0.3559, 10.0897, 3.1614],
}
