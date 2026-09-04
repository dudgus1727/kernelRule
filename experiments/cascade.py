"""★ Cascade — regret 규칙이 영역을, 순위 규칙이 순서를. LLM 0회.

    python3 experiments/cascade.py

실험 계획서 `docs/artifacts/cascade-prereg.md`.

```
1단계   regret 규칙으로 상위 k 를 추린다      (영역 선택)
2단계   순위 규칙으로 그 안에서 1등을 고른다   (순서)
```

벽의 두 진술이 상보적이라 이어 붙인다 (D-118). **벽을 없애는 것이
아니라 배포 시점에 우회하는 것이다.**
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from two_stage import A6000, _fit, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.runset import assert_same_condition
from kernelrule.core.splits import regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY

KS = (10, 20, 50)
N_DRAWS = 20
REG_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
RANK_RUNS = [f"x-rank-rankevo-s{i}" for i in range(3)]


def _best(run: str, by: str) -> dict:
    arc = [json.loads(x) for x in
           (Path("runs") / run / "archive.jsonl").read_text().splitlines()
           if x.strip()]
    key = (lambda e: e.get("rank_loss", 1e9)) if by == "rank" \
        else (lambda e: e["regret"])
    return sorted(arc, key=key)[0]


def _scores(fn, ws, table, matrix, shapes) -> dict:
    """형상별 점수 배열. 적합을 한 번만 하려고 미리 뽑아 둔다."""
    out = {}
    for p in shapes:
        cand = table.candidates(p)
        w = ws[regime_of(p, table.hw)]
        out[p.key] = np.asarray(make_score_of(fn, matrix, w)(p, cand),
                                dtype=np.float64)
    return out


def _geo(v) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=np.float64)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/cascade.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    assert_same_condition(REG_RUNS, label="regret 팔")
    assert_same_condition(RANK_RUNS, label="순위 팔")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    times = {p.key: np.asarray(T.times_of(p), dtype=np.float64) for p in hold}
    cands = {p.key: T.candidates(p) for p in hold}
    out: dict = {"ks": list(KS), "n_holdout": len(hold)}

    print("=" * 86)
    print("Cascade — 1단계 regret 규칙(영역) -> 2단계 순위 규칙(순서)")
    print("=" * 86)
    print(f"  A6000 홀드아웃 {len(hold)}형상 · regret 팔 {len(REG_RUNS)}구조 "
          f"x 순위 팔 {len(RANK_RUNS)}구조 = {len(REG_RUNS) * len(RANK_RUNS)} 조합\n")

    # -- 적합은 구조마다 한 번 -------------------------------------------
    S1 = {}
    for r in REG_RUNS:
        e = _best(r, "regret")
        S1[r] = _scores(*_fit(e["code"], e["w"], T, M, train, "regret"),
                        T, M, hold)
    S2 = {}
    for r in RANK_RUNS:
        e = _best(r, "rank")
        S2[r] = _scores(*_fit(e["code"], e["w"], T, M, train, "rank"),
                        T, M, hold)

    # -- 기준선: regret 단독 --------------------------------------------
    solo = []
    for r in REG_RUNS:
        solo.append(_geo([times[p.key][int(cands[p.key].top_k(S1[r][p.key],
                                                              1)[0])]
                          / times[p.key].min() for p in hold]))
    print(f"  기준선  regret 단독 regret@1   중앙 {np.median(solo):.4f}   "
          f"범위 {min(solo):.4f}~{max(solo):.4f}")
    out["solo"] = solo

    rng = np.random.default_rng(0)
    print(f"\n  {'k':>4} {'★ cascade 중앙':>14} {'범위':>17} "
          f"{'천장 oracle@k':>14} {'바닥 무작위in-k':>15} {'hit@k':>7}")
    for k in KS:
        # 천장 / 바닥 / hit@k 는 1단계 구조마다
        ceil_, floor_, hit_ = [], [], []
        casc = []
        for r1 in REG_RUNS:
            picks = {p.key: np.asarray(cands[p.key].top_k(S1[r1][p.key], k))
                     for p in hold}
            ceil_.append(_geo([times[p.key][picks[p.key]].min()
                               / times[p.key].min() for p in hold]))
            hit_.append(float(np.mean(
                [bool(times[p.key][picks[p.key]].min()
                      <= times[p.key].min() + 0.0) for p in hold])))
            draws = []
            for _ in range(N_DRAWS):
                draws.append(_geo(
                    [times[p.key][rng.choice(picks[p.key])]
                     / times[p.key].min() for p in hold]))
            floor_.append(float(np.mean(draws)))
            for r2 in RANK_RUNS:
                v = []
                for p in hold:
                    idx = picks[p.key]
                    j = idx[int(np.argmin(S2[r2][p.key][idx]))]
                    v.append(times[p.key][j] / times[p.key].min())
                casc.append(_geo(v))
        print(f"  {k:>4} {np.median(casc):14.4f} "
              f"{min(casc):8.4f}~{max(casc):<8.4f} {np.median(ceil_):14.4f} "
              f"{np.median(floor_):15.4f} {np.median(hit_):7.1%}")
        out.setdefault("cascade", {})[str(k)] = casc
        out.setdefault("ceiling", {})[str(k)] = ceil_
        out.setdefault("floor_in_k", {})[str(k)] = floor_
        out.setdefault("hit_at_k", {})[str(k)] = hit_

    # -- 판정 ------------------------------------------------------------
    print("\n" + "=" * 86)
    print("판정 — 실험 계획서에 박은 선")
    print("=" * 86)
    lo, hi = min(solo), max(solo)
    print(f"  regret 단독 시드 범위  {lo:.4f}~{hi:.4f}")
    for k in KS:
        m = float(np.median(out["cascade"][str(k)]))
        v = ("★ 우회된다" if m < lo else
             "★ 안 된다 (범위 안)" if m <= hi else "★ 2단계가 해를 끼친다")
        f = float(np.median(out["floor_in_k"][str(k)]))
        c = float(np.median(out["ceiling"][str(k)]))
        pos = (m - c) / (f - c) if f > c else float("nan")
        print(f"  k={k:<3d} cascade {m:.4f}   {v}")
        print(f"        천장 {c:.4f} · 바닥 {f:.4f} → "
              f"2단계가 그 사이의 {1 - pos:.0%} 를 메운다")

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 18조합은 독립 표본이 아니다 (구조 9개에서 나온다). "
          "유의성을 안 낸다 (원칙 27)")


if __name__ == "__main__":
    main()
