"""★ 피처 하나하나를 ★ 학습 형상에서만 잰다 (D-189). **LLM 0회 · GPU 0.**

    python3 -m experiments.a6000_featdiag --features F.jsonl --out OUT.json
    python3 -m experiments.a6000_featdiag --ext-contract E.jsonl --out OUT.json
    python3 -m experiments.a6000_featdiag --library --out OUT.json   # 기존 27축

## 무엇을 재나

fold 마다 **그 fold 의 학습 형상만** 쓴다. 홀드아웃 형상의 시간은 ⛔ 읽지 않는다.

```
spearman     형상 안에서 피처 값과 log(시간) 의 순위 상관 — 형상 평균
             (+ 면 "클수록 느리다" 를 잘 나타낸다)
pick_lo      피처가 가장 작은 config 를 고르면 regret (기하평균)
pick_hi      가장 큰 config 를 고르면 regret
by_class     pick_lo 를 형상 분류별로 (hungry · small · big · mid)
```

⚠️ 네 fold 의 학습 형상을 합치면 65형상 전부다 (한 형상은 세 fold 의 학습에
들어 있다). 그래서 fold 평균을 보고 설계를 정하면 **모든 형상을 본 셈**이 된다.
이 도구는 fold 마다 따로 재서 그 사실을 숨기지 않고, 규칙의 **가중치와 문턱**은
여전히 fold 학습에서만 정한다. 설계(피처의 모양)가 본 것은 결과 보고에 적는다.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from experiments import a6000_probe as P

FOLDS = P.FOLDS


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def diagnose(names: list[str], m, t, splits, cls) -> dict:
    out = {}
    for n in names:
        per_fold = {}
        for f in FOLDS:
            sp, lo, hi, bycls = [], [], [], {}
            for p in splits[f].train.shapes:
                F, _ = m.for_shape(p)
                v = np.asarray(getattr(F, n), float)
                tt = np.asarray(t.times_of(p), float)
                if v.ndim == 0 or len(v) != len(tt):
                    continue
                v = np.nan_to_num(v, nan=np.inf)
                sp.append(_spearman(v[np.isfinite(v)],
                                    np.log(tt[np.isfinite(v)])))
                cand = t.candidates(p)
                i_lo = int(cand.top_k(v, 1)[0])
                i_hi = int(cand.top_k(-v, 1)[0])
                r_lo = tt[i_lo] / tt.min()
                lo.append(r_lo)
                hi.append(tt[i_hi] / tt.min())
                bycls.setdefault(cls[f"{p.M}x{p.N}x{p.K}"], []).append(r_lo)
            per_fold[f] = {
                "spearman": round(float(np.nanmean(sp)), 4) if sp else None,
                "pick_lo": round(P._gm(lo), 4) if lo else None,
                "pick_hi": round(P._gm(hi), 4) if hi else None,
                "by_class": {c: round(P._gm(x), 4) for c, x in bycls.items()}}
        ok = [v for v in per_fold.values() if v["spearman"] is not None]
        out[n] = {
            "spearman_mean": round(float(np.mean([v["spearman"] for v in ok])), 4)
            if ok else None,
            "pick_lo_mean": round(float(np.mean([v["pick_lo"] for v in ok])), 4)
            if ok else None,
            "pick_hi_mean": round(float(np.mean([v["pick_hi"] for v in ok])), 4)
            if ok else None,
            "per_fold": per_fold}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--ext-contract", type=Path, default=None)
    ap.add_argument("--library", action="store_true")
    ap.add_argument("--gpu", default=P.GPU)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    from experiments.f1_pipeline import _splits
    from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
    from kernelrule.features.loader import load_generated

    P._G["gpu"] = a.gpu
    t = P._table()
    extra = [str(a.features)] if a.features else []
    P._G["ext_contract"] = [str(a.ext_contract)] if a.ext_contract else []
    reg = P._registry(t, extra)
    m = FeatureMatrix(t, reg, cache_dir=CACHE_DIR)
    splits = {f: _splits(t, fold=f, k=4, design=P.DESIGN) for f in FOLDS}
    cls = P._classes(t, m, t.shapes())
    names: list[str] = []
    if a.library:
        names += sorted(n for n in reg._items if not reg[n].shape_level)
    if a.features:
        names += [f.name for f in load_generated(a.features, table=t)
                  if not f.shape_level]
    if a.ext_contract:
        names += [f.name for f in P._load_ext_contract(a.ext_contract)]
    res = diagnose(names, m, t, splits, cls)
    a.out.write_text(json.dumps({"gpu": a.gpu, "holdout": "not read",
                                 "features": res}, ensure_ascii=False,
                                indent=1))
    print(f"{'축':<40}{'spearman':>9}{'pick_lo':>9}{'pick_hi':>9}   "
          "(학습 형상만 · fold 평균)")
    for n, v in sorted(res.items(), key=lambda x: (x[1]["pick_lo_mean"] or 9)):
        print(f"  {n:<38}{v['spearman_mean'] or float('nan'):9.3f}"
              f"{v['pick_lo_mean'] or float('nan'):9.4f}"
              f"{v['pick_hi_mean'] or float('nan'):9.4f}")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
