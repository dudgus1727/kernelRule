"""lunaNAMES-s3 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.

모델      gpt-5.6-luna / responses
추론      medium
피처 표시  names

구조 홀드아웃 1.1367  (표본내 1.1156)

★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.
재현:  python3 experiments/verify_rules.py
"""

import numpy as np  # noqa: F401

def score(f, p, hw, w):
    s = f.tile_aspect_imbalance * w[0]
    s = s + f.has_spill * w[1]
    s = s + f.log_dram_traffic * w[2]
    s = s + f.occupancy_deficit * w[3]
    s = s + f.pipeline_warmup_frac * np.exp(-f.log_mainloop_iters) * w[4]
    s = s + f.split_k_cost * w[5]
    s = s + f.tail_waste * w[6]
    s = s + f.reg_pressure * w[7]
    return s


W_FITTED = {
    'long': [0.790854, 41.992166, 4.095975, 0.726483, 1.348129, 15.845192, 10.114663, 0.581117],
    'short': [0.790854, 41.992166, 4.095975, 0.726483, 1.348129, 15.845192, 10.114663, 0.581117],
}
