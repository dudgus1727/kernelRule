"""시간 본항만으로는 얼마나 맞히나, 그리고 보정항을 합친 힘은 본항에 견줘 얼마나 큰가. LLM 0 · GPU 0.
판정이 아니다 — 사용자 질문 (2026-10-07) "시간 본항만 해도 어느정도 맞추지 않아? 보정항이 여러 개 달리면
본항의 영향이 사실 적어지는 거 아니야?" 에 숫자로 답한다.

    python3 docs/artifacts/f4c/lam/influence.py > docs/artifacts/f4c/lam/influence.txt

```
본항만     보정 가중치를 모두 0 으로 둔 점수 (본항에 가중치가 있으면 그것은 둔다) 로 고른 regret,
           holdout 65 형상 (fold 마다 그 fold 의 규칙). 적합 없음
보정의 합   점수(w) - 점수(0). 형상 안 폭 R_s (p1~p99, caps.shape_range) 를 본항의 R_s 와 견준다
뒤집힘     규칙이 고른 config 가 본항만으로 고른 것과 다른 holdout 형상의 비율, 그리고 그때 본항이 본
           고른 것의 손해 (단위: 본항 점수, 10 = 시간 2 배) — 중앙 · 90% · 최대 · 10 단위 넘는 형상 수
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
import re  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

D = Path(__file__).resolve().parents[1]
GPUS = ("a6000", "5090", "4090", "h100")
LAMS = ("1", "0.5", "0.1")


def _weights(g: str, lam: str) -> dict:
    if lam == "1":
        p = D / "refit_cap1.json" if g == "a6000" else D / g / "refit_cap1.json"
    else:
        p = D / "lam" / f"{g}_cap{lam}.json"
    return {f: np.asarray(v["w"], float)
            for f, v in json.loads(p.read_text())["per_fold"].items()}


def _q(v, k: float) -> float:
    return float(np.percentile(v, k)) if np.size(v) else float("nan")


def _gm(v) -> float:
    v = np.asarray(v, float)
    return float(np.exp(np.mean(np.log(v))))


def main() -> None:
    warnings.simplefilter("ignore")
    os.chdir(ROOT)
    import kernelrule.features.physical  # noqa: F401
    from experiments import a6000_probe as P
    from experiments.f1_pipeline import _splits
    from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.rules.caps import shape_range
    from kernelrule.rules.checks import exponent_indices
    from kernelrule.rules.time_term import main_weight_indices

    print(f"{'GPU':6s} {'lam':>4s} {'본항만':>7s} {'규칙':>7s} {'보정 가중치':>10s} "
          f"{'R(보정합)/R(본항) 중앙':>20s} {'90%':>6s} {'R(보정합) 단위':>12s} "
          f"{'뒤집힌 형상':>10s} {'본항이 본 손해 중앙':>18s} {'90%':>6s} "
          f"{'최대':>6s} {'>10 단위':>8s}")
    for g in GPUS:
        P._G.clear()
        P._G.update(gpu=g, prefix="f4c")
        t = P._table()
        rules = D / "rules" if g == "a6000" else D / g / "rules"
        spec = json.loads((rules / "f4c_evolved_s0.json").read_text())["per_fold"]
        W = {lam: _weights(g, lam) for lam in LAMS}
        rows = {lam: {"main": [], "rule": [], "ratio": [], "flip": [], "loss": [],
                      "ncorr": [], "units": []} for lam in LAMS}
        for f in "0123":
            code = spec[f]["code"]
            fn = compile_rule(code)
            sp = _splits(t, fold=int(f), k=4, design="nkband")
            m = FeatureMatrix(t, P._registry_fold(t, int(f), spec[f].get(
                "extra_features", [])), cache_dir=CACHE_DIR)
            used = {int(i) for i in re.findall(r"w\[(\d+)\]", code)}
            keep = sorted(main_weight_indices(code))
            corr = used - set(keep) - exponent_indices(code)
            for p in sp.val.shapes:
                feats, info = m.for_shape(p)
                tt = np.asarray(t.times_of(p), float)
                best = float(tt.min())
                cands = t.candidates(p)
                for lam in LAMS:
                    w = W[lam][f]
                    w0 = np.zeros_like(w)
                    w0[keep] = w[keep]          # a weighted main term stays
                    s0 = np.asarray(fn(feats, info, m.hw, w0), float)
                    s1 = np.asarray(fn(feats, info, m.hw, w), float)
                    i0 = int(cands.top_k(s0, 1)[0])
                    i1 = int(cands.top_k(s1, 1)[0])
                    r = rows[lam]
                    r["main"].append(tt[i0] / best)
                    r["rule"].append(tt[i1] / best)
                    rm = shape_range(s0)
                    r["ratio"].append(shape_range(s1 - s0) / rm if rm > 0 else np.nan)
                    r["units"].append(shape_range(s1 - s0))
                    r["flip"].append(i1 != i0)
                    if i1 != i0:
                        r["loss"].append(float(s0[i1] - s0.min()))
                    r["ncorr"].append(sum(abs(w[k]) > 1e-3 for k in corr))
        for lam in LAMS:
            r = rows[lam]
            ratio = np.asarray(r["ratio"], float)
            loss = np.asarray(r["loss"], float)
            print(f"{g:6s} {lam:>4s} {_gm(r['main']):7.4f} {_gm(r['rule']):7.4f} "
                  f"{np.median(r['ncorr']):10.0f} {np.nanmedian(ratio):20.2f} "
                  f"{np.nanpercentile(ratio, 90):6.2f} "
                  f"{np.median(r['units']):12.1f} "
                  f"{np.mean(r['flip']) * 100:9.0f}% "
                  f"{_q(loss, 50):18.1f} {_q(loss, 90):6.1f} {_q(loss, 100):6.1f} "
                  f"{int(np.sum(loss > 10)):8d}")


if __name__ == "__main__":
    main()
