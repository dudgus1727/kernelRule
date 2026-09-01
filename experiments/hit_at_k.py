"""★ (a) 완전 이식 규칙의 상위 k% 안에 정답이 드는가. LLM 0회.

    python3 experiments/hit_at_k.py

## 왜

**config 축 샘플링이 가능한지의 상한이다.** 지금까지 형상 축만 줄여
봤는데(`refit_sample.py`), config 축을 줄이는 쪽이 훨씬 싸다.

```
형상 12 x config 전부   23만 작업
형상 41 x config 5%     3.9만 작업   ★ 그리고 형상 다양성이 3.4배
```

## 재는 것 — 두 가지. **두 번째가 더 중요하다**

```
상위 k% 안에 **진짜 최적**이 든 형상 비율
상위 k% 안에 **정답 집합**(노이즈 바닥 2σ 이내) 원소가 하나라도 든 비율
```

정답 집합에 하나라도 들면 그 형상에서 최선을 고를 수 있다. 그리고 이
표는 동률이 많다 (5090 정답 집합 중앙 9개, 최대 724개).

## 비교 대상

```
(a) 규칙 6개   A6000 구조 + A6000 가중치. ★ hw 상수만 5090 것
정적 top-k    ★ A6000 표에서 고른 고정 축 좌표 — 새 GPU 에서 바로 쓸 수 있다
무작위        바닥
벤더          ⛔ nvMatmulHeuristics 가 이 환경에 없다. 아래 주석 참고
```
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.crosstable import AXIS_FIELDS
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
SRC_RUNS = [f"f1pipe-F3-arch24-s{i}" for i in range(6)]
PCTS = (0.5, 1.0, 2.0, 5.0, 10.0)


def _splits(table: PerfTable):
    from kernelrule.core.splits import SplitSet

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(train=Split("train", tuple(p for p in shapes
                                               if p not in held)),
                    val=Split("val", tuple(held)), kind="nk11008")


def _axis_keys(table: PerfTable, p) -> list[tuple]:
    """★ 리스트로 돌려준다. `np.array(..., dtype=object)` 로 만들면
    튜플이 2차원 배열로 펴져서 **해시가 안 된다** — 실제로 걸렸다."""
    df = table.frame_for(p)
    cols = [df[f].to_numpy() for f in AXIS_FIELDS]
    return [tuple(c[i] for c in cols) for i in range(len(df))]


def _hits(order: np.ndarray, table: PerfTable, p, pcts) -> dict:
    """`order` 는 좋다고 본 순서(인덱스). 상위 k% 안의 적중을 센다."""
    t = table.times_of(p)
    best = int(np.argmin(t))
    ans = np.flatnonzero(table.answer_mask(p))
    n = len(t)
    out = {}
    for q in pcts:
        k = max(1, int(np.ceil(n * q / 100.0)))
        top = set(order[:k].tolist())
        out[q] = (best in top, bool(top & set(ans.tolist())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/hit-at-k.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    A = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    B = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    mA, mB = FeatureMatrix(A, REGISTRY), FeatureMatrix(B, REGISTRY)
    spA, spB = _splits(A), _splits(B)
    shapes = list(spB.val.shapes)          # ★ 홀드아웃에서 잰다

    print("=" * 78)
    print("(a) 완전 이식 규칙의 hit@k%  —  config 샘플링의 상한")
    print("=" * 78)
    n_c = [len(B.times_of(p)) for p in shapes]
    print(f"  5090 홀드아웃 {len(shapes)}형상   후보 중앙 {int(np.median(n_c))}개")
    print(f"  정답 집합 크기 중앙 "
          f"{int(np.median([int(B.answer_mask(p).sum()) for p in shapes]))}개")
    print("  ⛔ 벤더: nvMatmulHeuristics 가 이 환경에 없다 (import 실패).")
    print("     5090 벤더 추천을 만들 수 없어 이 표에서 뺀다 — "
          "'없다' 를 '나쁘다' 로 적지 않는다\n")

    res: dict = {"pcts": list(PCTS), "arms": {}}

    def record(name: str, orders: dict) -> None:
        rows = {q: [0, 0] for q in PCTS}
        for p in shapes:
            h = _hits(orders[p.key], B, p, PCTS)
            for q in PCTS:
                rows[q][0] += int(h[q][0])
                rows[q][1] += int(h[q][1])
        res["arms"][name] = {str(q): {"best": rows[q][0],
                                      "answer": rows[q][1],
                                      "n": len(shapes)} for q in PCTS}
        print(f"  {name:28s} " + "  ".join(
            f"{q}%: {rows[q][1]}/{len(shapes)}={rows[q][1] / len(shapes):.0%}"
            for q in PCTS))

    # -- (a) 완전 이식 규칙 6개 --------------------------------------------
    print("  ★ 정답 집합 원소가 상위 k% 에 하나라도 드는 형상 비율")
    print("  " + "-" * 74)
    for run in SRC_RUNS:
        f = Path("runs") / run / "archive.jsonl"
        e = sorted((json.loads(x) for x in f.read_text().splitlines()
                    if x.strip()), key=lambda z: z["regret"])[0]
        fn = compile_rule(e["code"])
        ws = {}
        for nm in ("short", "long"):     # ★ A6000 에서 맞춘 가중치
            g = [q for q in spA.train.shapes if regime_of(q, A.hw) == nm]
            ws[nm] = fit_weights(fn, mA, A, Split("train", tuple(g)),
                                 e["w"], max_evals=300).w
        orders = {}
        for p in shapes:
            cand = B.candidates(p)
            sc = np.asarray(make_score_of(fn, mB, ws[regime_of(p, B.hw)])(
                p, cand), dtype=float)
            orders[p.key] = np.argsort(sc, kind="stable")
        record(f"(a) {run.split('-')[-1]}", orders)

    # -- 정적 top-k: A6000 표에서 고른 고정 축 좌표 ------------------------
    # ★ 축 좌표로 조인한다. kernel_id 는 아키텍처마다 다르게 컴파일된다
    rank_a: dict = {}
    for p in spA.train.shapes:
        t = A.times_of(p)
        r = t / t.min()
        for key, v in zip(_axis_keys(A, p), r, strict=True):
            rank_a.setdefault(key, []).append(float(v))
    score_a = {k: float(np.exp(np.mean(np.log(v)))) for k, v in rank_a.items()}
    orders = {}
    for p in shapes:
        keys = _axis_keys(B, p)
        # A6000 에 없던 좌표는 맨 뒤로 (조용히 빼지 않는다)
        s = np.array([score_a.get(k, 1e9) for k in keys], dtype=float)
        orders[p.key] = np.argsort(s, kind="stable")
    record("정적 (A6000 고정 config)", orders)

    # -- 무작위 바닥 --------------------------------------------------------
    rng = np.random.default_rng(0)
    orders = {p.key: rng.permutation(len(B.times_of(p))) for p in shapes}
    record("무작위", orders)

    # -- 진짜 최적이 드는 비율 (엄격) ---------------------------------------
    print("\n  ★ **진짜 최적**이 상위 k% 에 드는 형상 비율 (엄격)")
    print("  " + "-" * 74)
    for name, d in res["arms"].items():
        print(f"  {name:28s} " + "  ".join(
            f"{q}%: {d[str(q)]['best']}/{len(shapes)}"
            f"={d[str(q)]['best'] / len(shapes):.0%}" for q in PCTS))

    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
