"""lunaNAMES-s1 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      gpt-5.6-luna / responses
추론      medium
피처 표시  names

구조 홀드아웃 1.1281  (표본내 1.0921)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + f.has_spill * w[2]
    s = s + f.tile_aspect_imbalance * (f.edge_waste + f.tail_waste) * w[3]
    s = s + f.occupancy_deficit * w[4]
    s = s + f.reg_pressure * w[5]
    s = s + f.split_k_cost * w[6]
    s = s + np.log2(f.waves) * w[7]
    return s


W_FITTED = {
    'long': [1.593116, 1.712757, 2.834336, 1.021711, 1.476512, 1.333517, 7.749283, 0.896875],
    'short': [1.593116, 1.712757, 2.834336, 1.021711, 1.476512, 1.333517, 7.749283, 0.896875],
}
