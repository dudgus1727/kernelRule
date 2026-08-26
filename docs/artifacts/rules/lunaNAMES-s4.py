"""lunaNAMES-s4 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      gpt-5.6-luna / responses
추론      medium
피처 표시  names

구조 홀드아웃 1.0841  (표본내 1.0820)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = f.traffic_amplification * w[0]
    s = s + f.smem_pressure * w[1]
    s = s + f.has_spill * w[2]
    s = s + f.reg_pressure * w[3]
    s = s + (f.log_mainloop_iters + f.log_inst_total - f.log_grid_tiles) * w[4]
    s = s + f.split_k_cost * w[5]
    s = s + f.sm_idle_cost * w[6]
    s = s + np.log2(f.waves) * w[7]
    return s


W_FITTED = {
    'long': [0.987587, 2.990088, 24.459459, 22.426959, -1.977602, 46.638878, 4.869551, 2.410129],
    'short': [0.987587, 2.990088, 24.459459, 22.426959, -1.977602, 46.638878, 4.869551, 2.410129],
}
