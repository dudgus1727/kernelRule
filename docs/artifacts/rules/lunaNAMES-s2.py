"""lunaNAMES-s2 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      gpt-5.6-luna / responses
추론      medium
피처 표시  names

구조 홀드아웃 1.1249  (표본내 1.0927)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + f.occupancy_deficit * w[2]
    s = s + f.reg_pressure * w[3]
    s = s + f.spill_magnitude * w[4]
    s = s + (f.is_two_stage * f.log_mainloop_iters / f.pipeline_warmup_frac) * w[5]
    s = s + f.split_k_cost * w[6]
    s = s + (-f.log_grid_tiles * f.sm_idle_cost) * w[7]
    return s


W_FITTED = {
    'long': [2.368, 2.704, 1.89, 2.19, 1.04, 0.5, 8.58, 0.33],
    'short': [2.368, 2.704, 1.89, 2.19, 1.04, 0.5, 8.58, 0.33],
}
