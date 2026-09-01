"""★ 상위 100 안에서 피처가 후보를 가를 수 있나 — **상한**. LLM 0회.

    python3 experiments/top100_ceiling.py

## 왜 진화 **전에** 재나

손실을 아무리 잘 설계해도 넘을 수 없는 벽이 있다.

```
상위 100 안에서 피처값이 거의 같다   ★ 어떤 손실로도 못 배운다
상위 100 안에서 피처값이 다르다      배울 수 있다 — 손실 설계가 값한다
```

그리고 이 값이 **판정을 읽는 근거**가 된다. tau 가 안 오를 때
"손실이 나쁜가" 와 "피처가 못 재는가" 를 이것으로 가른다.

## ★ 형상 수준 피처는 상한에 못 들어간다

```
p. 피처 5개   한 형상 안에서 **상수**다 -> 후보를 가르는 데 0 기여
             `np.where(p.is_memory_bound, ...)` 분기로만 쓰인다
f. 피처 19개  ★ 여기가 전부다
```
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
TOP_N = 100
AXES = ("tile_m", "tile_n", "tile_k", "ext_stages", "split_k",
        "ext_warp_m", "ext_warp_n", "ext_warp_k", "ext_swizzle_n",
        "split_k_mode")


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


def _report(label, table, matrix, shapes, out: dict) -> None:
    feats = REGISTRY.names(shape_level=False)
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")

    tops = {}
    for p in shapes:
        t = np.asarray(table.times_of(p))
        tops[p.key] = np.argsort(t, kind="stable")[:TOP_N]

    # -- 시간 자체 --------------------------------------------------------
    span, uniq_t = [], []
    for p in shapes:
        t = np.asarray(table.times_of(p))[tops[p.key]]
        span.append(float(t.max() / t.min() - 1.0))
        uniq_t.append(int(len(np.unique(t))))
    print(f"  {len(shapes)}형상, 상위 {TOP_N}")
    print(f"  시간 폭 (최악/최적 - 1)   중앙 {np.median(span):.1%}  "
          f"범위 {min(span):.1%}~{max(span):.1%}")
    print(f"  고유 시간값               중앙 {int(np.median(uniq_t))}  "
          f"범위 {min(uniq_t)}~{max(uniq_t)}")

    # -- f. 피처가 가르나 --------------------------------------------------
    print(f"\n  ★ config 수준 피처 {len(feats)}개 — 상위 {TOP_N} 안에서")
    print(f"  {'피처':24s} {'고유값 중앙':>10} {'변동계수 중앙':>12} "
          f"{'상수인 형상':>11}")
    frows = {}
    for name in feats:
        u, cv, const = [], [], 0
        for p in shapes:
            # ★ 형상별 열을 직접 쓴다. 전역 `column()` 은 이어붙인 것이라
            #   형상 경계를 다시 계산해야 하고, 그러면 정의가 둘이 된다.
            f_, _info = matrix.for_shape(p)
            v = np.asarray(getattr(f_, name))[tops[p.key]]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            nu = int(len(np.unique(v)))
            u.append(nu)
            m = float(np.mean(np.abs(v)))
            cv.append(float(np.std(v) / m) if m > 1e-12 else 0.0)
            const += int(nu <= 1)
        frows[name] = {"uniq_median": float(np.median(u)),
                       "cv_median": float(np.median(cv)),
                       "n_const": const}
        print(f"  {name:24s} {np.median(u):10.1f} {np.median(cv):12.4f} "
              f"{const:8d}/{len(shapes)}")

    n_dead = sum(1 for v in frows.values() if v["uniq_median"] <= 1.0)
    n_flat = sum(1 for v in frows.values() if v["cv_median"] < 0.01)
    print(f"\n  ★ 상위 {TOP_N} 안에서 중앙적으로 **상수**인 피처 "
          f"{n_dead}/{len(feats)}")
    print(f"  ★ 변동계수 1% 미만(사실상 평평)인 피처 {n_flat}/{len(feats)}")

    # -- config 축 분포 ---------------------------------------------------
    print(f"\n  ★ config 축 — 상위 {TOP_N} 안의 값 종류 (형상별 중앙)")
    arows = {}
    for ax in AXES:
        u = []
        for p in shapes:
            df = table.frame_for(p)
            if ax not in df:
                continue
            v = df[ax].to_numpy()[tops[p.key]]
            u.append(int(len(np.unique(v))))
        if u:
            arows[ax] = float(np.median(u))
            print(f"    {ax:16s} {np.median(u):5.1f}종")

    # -- 최적과 2~10등의 축 거리 -------------------------------------------
    dist = []
    for p in shapes:
        df = table.frame_for(p)
        cols = [ax for ax in AXES if ax in df]
        arr = np.stack([df[c].to_numpy().astype(str) for c in cols], axis=1)
        top = tops[p.key]
        base = arr[top[0]]
        for r in top[1:10]:
            dist.append(int((arr[r] != base).sum()))
    print(f"\n  ★ 최적과 2~10등이 **몇 축** 다른가: 중앙 "
          f"{int(np.median(dist))}  범위 {min(dist)}~{max(dist)}  "
          f"(축 {len(AXES)}개 중)")
    print(f"     0축 다름(완전 동일) {sum(1 for d in dist if d == 0)}"
          f"/{len(dist)}쌍")

    out[label] = {"n_shapes": len(shapes), "top_n": TOP_N,
                  "span_median": float(np.median(span)),
                  "uniq_time_median": int(np.median(uniq_t)),
                  "features": frows, "axes": arows,
                  "axis_dist_median": int(np.median(dist)),
                  "n_const_features": n_dead, "n_flat_features": n_flat}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/top100-ceiling.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    out: dict = {}
    for lbl, (d, h) in (("A6000 학습 41형상", A6000),
                        ("5090 학습 41형상", G5090)):
        T = PerfTable.from_bundle(d, env_hash=h, ok_only=False)
        M = FeatureMatrix(T, REGISTRY)
        _report(lbl, T, M, list(_splits(T).train.shapes), out)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
