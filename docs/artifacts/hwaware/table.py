"""D-200 결과표 — 미리 정한 읽기 (xgb_full seed 42 대 D-199 규칙, holdout 차이 ±0.01) 그대로. LLM 0 · GPU 0.

    python3 docs/artifacts/hwaware/table.py > docs/artifacts/hwaware/table.txt

```
holdout     4 fold holdout 을 모은 regret 기하평균 (a6000 · 5090 65, 4090 · H100 63 형상)
그들 지표    1 - 처리량 비 (= 1 - 1/regret) 의 산술평균 · 최적의 95% 안 비율 — 논문 표 12 의 정의
이김 · 짐    형상마다 1% 를 넘게 빠르면 이김, 1% 안이면 비김 (vs_vendor.py 와 같은 띠)
```
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

D = Path(__file__).resolve().parent
F4P = D.parent / "f4p"
GPUS = ("a6000", "5090", "4090", "h100")
K1A_HOLDOUT = {"a6000": 1.0479, "5090": 1.0354, "4090": 1.0380, "h100": 1.0635}
BAND = 0.01      # 읽기 — holdout 기하평균 차이
TIE = 0.01       # 형상별 이김 · 비김 · 짐


def _gm(v) -> float:
    v = np.asarray(list(v), float)
    return float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")


def _theirs(v) -> tuple[float, float]:
    r = np.asarray(list(v), float)
    return float(np.mean(1.0 - 1.0 / r)), float(np.mean(1.0 / r >= 0.95))


def _wtl(a: dict, b: dict) -> str:
    w = sum(a[k] < b[k] * (1 - TIE) for k in a)
    lo = sum(a[k] > b[k] * (1 + TIE) for k in a)
    return f"{w}/{len(a) - w - lo}/{lo}"


def _pooled(arm: dict) -> dict:
    return {s: r for fr in arm["per_fold"].values() for s, r in fr["regret"].items()}


def _verdict(x: float, rule: float) -> str:
    if x < rule - BAND:
        return "XGB 나음"
    if x > rule + BAND:
        return "규칙 나음"
    return "비슷"


def main() -> None:
    warnings.simplefilter("ignore")
    os.chdir(ROOT)
    from experiments import a6000_probe as P

    vv = json.loads((F4P / "vs_vendor.json").read_text())
    print("D-200 — Chandran 외 피처 + XGBoost/Ridge, 우리 표 · nkband fold · evaluate_scores. holdout 기하평균")
    print(f"{'GPU':6s} {'xgb_full':>9s} {'seed 0·1':>15s} {'xgb_struct':>10s} {'ridge':>7s} "
          f"{'D-199 규칙':>10s} {'K1a':>7s} {'D-196':>7s} {'벤더':>7s}  읽기 (xgb_full 42 대 D-199)")
    count = {"XGB 나음": 0, "규칙 나음": 0, "비슷": 0}
    rows = {}
    for g in GPUS:
        r = json.loads((D / f"{g}.json").read_text())
        arms = {(a["arm"], a["seed"]): a for a in r["arms"]}
        xf = arms[("xgb_full", 42)]
        cur = {x["shape"]: x["current"] for x in vv[g]["shapes"]}
        ven = {x["shape"]: x["vendor"] for x in vv[g]["shapes"]}
        hx = _pooled(xf)
        assert set(hx) == set(cur), f"{g}: holdout 형상이 D-199 와 다르다"
        h = {k: a["aggregate"]["holdout_gm"] for k, a in arms.items()}
        rule, vend = _gm(cur.values()), _gm(ven.values())
        v = _verdict(h[("xgb_full", 42)], rule)
        count[v] += 1
        seeds = f"{h[('xgb_full', 0)]:.4f}·{h[('xgb_full', 1)]:.4f}"
        print(f"{g:6s} {h[('xgb_full', 42)]:9.4f} {seeds:>15s} {h[('xgb_struct', 42)]:10.4f} "
              f"{h[('ridge_full', 0)]:7.4f} {rule:10.4f} {K1A_HOLDOUT[g]:7.4f} "
              f"{vv[g]['holdout']['d196']:7.4f} {vend:7.4f}  {v} ({h[('xgb_full', 42)] - rule:+.4f})")
        rows[g] = (r, arms, hx, cur, ven)
    n = len(GPUS)
    reading = ("(a) 우리 표에서도 그 방법이 더 정확하다" if count["XGB 나음"] >= 3 else
               "(b) 그 피처 위의 비선형 모델이 우리 규칙을 이기지 못한다"
               if count["규칙 나음"] + count["비슷"] >= 3 else "(c) 섞였다")
    print(f"\n읽기: XGB 나음 {count['XGB 나음']} · 비슷 {count['비슷']} · 규칙 나음 "
          f"{count['규칙 나음']} (/{n}) -> {reading}")

    print("\n그들 지표 — 1 - 처리량 비의 평균 · 최적의 95% 안 비율")
    print(f"{'GPU':6s} {'xgb_full':>15s} {'xgb_struct':>15s} {'ridge':>15s} {'D-199 규칙':>15s} {'벤더':>15s}")
    for g in GPUS:
        r, arms, hx, cur, ven = rows[g]
        cells = [_theirs(_pooled(arms[k]).values()) for k in
                 (("xgb_full", 42), ("xgb_struct", 42), ("ridge_full", 0))]
        cells += [_theirs(cur.values()), _theirs(ven.values())]
        print(f"{g:6s} " + " ".join(f"{m * 100:6.1f}% {w * 100:5.0f}%  " for m, w in cells))

    print("\n형상별 이김/비김/짐 (1%) · 학습 regret · 적합 시간")
    print(f"{'GPU':6s} {'xgb 대 벤더':>12s} {'xgb 대 규칙':>12s} {'규칙 대 벤더':>12s} "
          f"{'xgb 학습':>9s} {'규칙 학습':>9s} {'적합 s/fold':>11s}")
    for g in GPUS:
        r, arms, hx, cur, ven = rows[g]
        xf = arms[("xgb_full", 42)]
        refit = json.loads((F4P / g / "refit_cap0.1.json").read_text())
        secs = np.mean([fr["fit_seconds"] for fr in xf["per_fold"].values()])
        print(f"{g:6s} {_wtl(hx, ven):>12s} {_wtl(hx, cur):>12s} {_wtl(cur, ven):>12s} "
              f"{xf['aggregate']['train_gm']:9.4f} {refit['aggregate']['train_gm']:9.4f} {secs:11.0f}")

    print("\n분류별 holdout 기하평균 — xgb_full / D-199 규칙 / 벤더")
    for g in GPUS:
        r, arms, hx, cur, ven = rows[g]
        P._G.clear()
        P._G["gpu"] = g
        t = P._table()
        by = {p.key: p for p in t.shapes()}
        cls = P._classes(t, None, [p for p in by.values()
                                   if f"{p.M}x{p.N}x{p.K}" in hx])
        parts = []
        for c in ("big", "hungry", "mid", "small"):
            ks = [k for k in hx if cls.get(k) == c]
            parts.append(f"{c} {len(ks)}: {_gm(hx[k] for k in ks):.3f} / "
                         f"{_gm(cur[k] for k in ks):.3f} / {_gm(ven[k] for k in ks):.3f}")
        print(f"  {g:6s} " + "  ".join(parts))

    print("\nXGBoost (full, seed 42) gain 상위 — fold 넷의 순위 합이 작은 것부터")
    for g in GPUS:
        r, arms, hx, cur, ven = rows[g]
        tops = [list(fr["importance_gain_top15"])
                for fr in arms[("xgb_full", 42)]["per_fold"].values()]
        rank = {k: sum(t.index(k) if k in t else 15 for t in tops)
                for k in set().union(*tops)}
        print(f"  {g:6s} " + ", ".join(k for k, _ in sorted(rank.items(), key=lambda kv: kv[1])[:8]))


if __name__ == "__main__":
    main()
