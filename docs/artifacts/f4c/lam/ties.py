"""본항이 가장 좋다고 보는 config 들 (본항 최소에서 0.5 단위 = 시간 3.5% 안) 은 몇 개이고, 그 안에 정답이
있나. LLM 0 · GPU 0. 판정 아님 — influence.py 의 '본항만' 숫자를 읽기 위한 것.

    python3 docs/artifacts/f4c/lam/ties.py > docs/artifacts/f4c/lam/ties.txt

```
묶음       본항 점수가 최소 + 0.5 안인 config (holdout 형상마다, fold 마다 그 fold 의 규칙의 본항)
묶음 최선   묶음 안에서 가장 빠른 것의 regret — 본항이 거른 묶음에 정답이 있나
묶음 최악   묶음 안에서 가장 느린 것의 regret — 묶음 안에서 아무거나 고르면 얼마나 나쁠 수 있나
```
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

D = Path(__file__).resolve().parents[1]
GPUS = ("a6000", "5090", "4090", "h100")
BAND = 0.5


def _gm(v) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(v, float)))))


def main() -> None:
    warnings.simplefilter("ignore")
    os.chdir(ROOT)
    import kernelrule.features.physical  # noqa: F401
    from experiments import a6000_probe as P
    from experiments.f1_pipeline import _splits
    from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.rules.time_term import main_weight_indices

    print(f"{'GPU':6s} {'config 수 중앙':>12s} {'묶음 크기 중앙':>12s} {'90%':>6s} "
          f"{'묶음 최선':>9s} {'묶음 최악':>9s} {'본항만':>7s}")
    for g in GPUS:
        P._G.clear()
        P._G.update(gpu=g, prefix="f4c")
        t = P._table()
        rules = D / "rules" if g == "a6000" else D / g / "rules"
        spec = json.loads((rules / "f4c_evolved_s0.json").read_text())["per_fold"]
        w1 = json.loads(((D / "refit_cap1.json") if g == "a6000" else
                         (D / g / "refit_cap1.json")).read_text())["per_fold"]
        n_all, n_band, best_in, worst_in, main_pick = [], [], [], [], []
        for f in "0123":
            code = spec[f]["code"]
            fn = compile_rule(code)
            w = np.asarray(w1[f]["w"], float)
            keep = sorted(main_weight_indices(code))
            w0 = np.zeros_like(w)
            w0[keep] = w[keep]
            sp = _splits(t, fold=int(f), k=4, design="nkband")
            m = FeatureMatrix(t, P._registry_fold(t, int(f), spec[f].get(
                "extra_features", [])), cache_dir=CACHE_DIR)
            for p in sp.val.shapes:
                feats, info = m.for_shape(p)
                tt = np.asarray(t.times_of(p), float)
                s0 = np.asarray(fn(feats, info, m.hw, w0), float)
                band = s0 <= s0.min() + BAND
                n_all.append(len(tt))
                n_band.append(int(band.sum()))
                best_in.append(tt[band].min() / tt.min())
                worst_in.append(tt[band].max() / tt.min())
                main_pick.append(tt[int(t.candidates(p).top_k(s0, 1)[0])] / tt.min())
        print(f"{g:6s} {np.median(n_all):12.0f} {np.median(n_band):12.0f} "
              f"{np.percentile(n_band, 90):6.0f} {_gm(best_in):9.4f} "
              f"{_gm(worst_in):9.4f} {_gm(main_pick):7.4f}")


if __name__ == "__main__":
    main()
