"""★ 상위권 순위 능력 — **분해능 교락을 통제하고** 다시 잰다. LLM 0회.

    python3 experiments/top_tau.py

## 왜 다시 재나

`degeneracy.py` 가 참 상위 100 안의 tau-b 를 0.141 로 냈다. 사전 등록의
판정선은 `>= 0.30 매긴다 / <= 0.10 못 매긴다` 이므로 **0.141 은
"가운데"** 다 — "못 매긴다" 로 쓰면 안 된다.

그리고 **교락이 있다.**

```
참 상위 100 의 고유 시간값: 중앙 24개, ★ 10개 이하인 형상이 10/41
A6000 눈금 1.024 µs — 상위권은 분해능이 지배한다
-> tau 가 낮은 것이 규칙 탓인지 눈금 탓인지 안 갈린다
```

## 통제

```
상위 100 의 고유 시간값이 N 개 이상인 형상에서만 tau   (N = 30, 50)
★ 남은 형상 수를 함께 찍는다
★ 그 부분집합의 무작위 바닥도 함께 (20뽑기 — 바닥도 표본이다, 원칙 7)
```

## ★ 5090 에서도 잰다 — 거기는 교락이 훨씬 덜하다

```
5090 눈금 16 ns  vs  A6000 1.024 µs   (1/64)
```

⚠️ 표가 다르므로 **절대값이 아니라 무작위 바닥 대비**로 견준다
(원칙 4). 정답 집합 크기도 다르다 (A6000 중앙 5 / 5090 중앙 11).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
SRC_RUNS = [f"f1pipe-F3-arch24-s{i}" for i in range(6)]
TOP_N = 100
MIN_UNIQ = (1, 30, 50)      # 1 = 통제 없음(정의 가능한 것만)
N_DRAWS = 20


def _splits(table: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def _uniq_top(table, p) -> int:
    t = np.asarray(table.times_of(p))
    return int(len(np.unique(t[np.argsort(t, kind="stable")[:TOP_N]])))


def _tau_top(sc, t) -> float:
    top = np.argsort(t, kind="stable")[:TOP_N]
    if len(np.unique(t[top])) < 2:
        return float("nan")
    return float(kendalltau(sc[top], t[top], variant="b").statistic)


def _one(label, table, matrix, fit_shapes, eval_shapes, out: dict) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    uniq = np.array([_uniq_top(table, p) for p in eval_shapes])
    print(f"  {len(eval_shapes)}형상   상위 {TOP_N} 의 고유 시간값: "
          f"중앙 {int(np.median(uniq))}  범위 {uniq.min()}~{uniq.max()}")
    print(f"  눈금 {table.noise.tick_ms} ms")
    print(f"\n  {'구조':22s} " + "  ".join(
        f"{'통제없음' if n == 1 else f'고유>={n}':>10}" for n in MIN_UNIQ))

    rows: dict = {}
    for run in SRC_RUNS:
        f = Path("runs") / run / "archive.jsonl"
        e = sorted((json.loads(x) for x in f.read_text().splitlines()
                    if x.strip()), key=lambda z: z["regret"])[0]
        fn = compile_rule(e["code"])
        ws = {}
        for nm in ("short", "long"):
            g = [q for q in fit_shapes if regime_of(q, table.hw) == nm]
            ws[nm] = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                                 e["w"], max_evals=300).w
        taus = {}
        for p in eval_shapes:
            cand = table.candidates(p)
            sc = np.asarray(make_score_of(fn, matrix, ws[regime_of(
                p, table.hw)])(p, cand), dtype=float)
            taus[p.key] = _tau_top(sc, np.asarray(table.times_of(p)))
        row = {}
        for n in MIN_UNIQ:
            v = [taus[p.key] for p, u in zip(eval_shapes, uniq, strict=True)
                 if u >= n and np.isfinite(taus[p.key])]
            row[n] = (float(np.median(v)) if v else float("nan"), len(v))
        rows[run] = row
        print(f"  {run:22s} " + "  ".join(
            f"{row[n][0]:10.3f}" for n in MIN_UNIQ))

    # ★ 무작위 바닥 — 같은 부분집합에서, 20뽑기
    rng = np.random.default_rng(0)
    floor: dict = {n: [] for n in MIN_UNIQ}
    for _ in range(N_DRAWS):
        taus = {}
        for p in eval_shapes:
            t = np.asarray(table.times_of(p))
            taus[p.key] = _tau_top(rng.random(len(t)), t)
        for n in MIN_UNIQ:
            v = [taus[p.key] for p, u in zip(eval_shapes, uniq, strict=True)
                 if u >= n and np.isfinite(taus[p.key])]
            if v:
                floor[n].append(float(np.median(v)))
    print(f"  {'★ 무작위 바닥':22s} " + "  ".join(
        f"{np.mean(floor[n]):10.3f}" if floor[n] else f"{'—':>10}"
        for n in MIN_UNIQ))
    print(f"  {'남은 형상':22s} " + "  ".join(
        f"{rows[SRC_RUNS[0]][n][1]:10d}" for n in MIN_UNIQ))

    med = {n: float(np.median([rows[r][n][0] for r in SRC_RUNS
                               if np.isfinite(rows[r][n][0])]))
           for n in MIN_UNIQ}
    print("\n  ★ 6구조 중앙: " + "   ".join(
        f"{'통제없음' if n == 1 else f'고유>={n}'} {med[n]:.3f}"
        for n in MIN_UNIQ))
    out[label] = {"rows": {r: {str(n): rows[r][n] for n in MIN_UNIQ}
                           for r in SRC_RUNS},
                  "floor": {str(n): (float(np.mean(floor[n]))
                                     if floor[n] else None)
                            for n in MIN_UNIQ},
                  "median": {str(n): med[n] for n in MIN_UNIQ},
                  "n_shapes": len(eval_shapes),
                  "uniq_median": int(np.median(uniq))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/top-tau.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    out: dict = {}

    A = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    mA = FeatureMatrix(A, REGISTRY)
    spA = _splits(A)
    # ★ 사전 등록 조건 그대로 (학습 41) — 앞 수치와 이어진다
    _one("A6000 학습 41형상 (사전 등록 조건)", A, mA,
         list(spA.train.shapes), list(spA.train.shapes), out)
    # 표 사이 비교용 — 홀드아웃끼리
    _one("A6000 홀드아웃 20형상", A, mA,
         list(spA.train.shapes), list(spA.val.shapes), out)

    B = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    mB = FeatureMatrix(B, REGISTRY)
    spB = _splits(B)
    # ★ (b) 재적합 가중치 — 5090 학습 분할로 맞춘다
    _one("5090 홀드아웃 20형상 — (b) 재적합", B, mB,
         list(spB.train.shapes), list(spB.val.shapes), out)

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 표가 다르다 — 절대값이 아니라 **무작위 바닥 대비**로 "
          "견줘라 (원칙 4)")


if __name__ == "__main__":
    main()
