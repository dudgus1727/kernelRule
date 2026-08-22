"""씨앗 절제 채점 — 정준 절차 + 시드별 최악값 병기.

    python3 experiments/score_ablation.py

## 무엇을 가르는가

```
(나) - 기존(1.0652)  =  리포트 정화 효과   ★ 음수일 수 있다
(가) - (나)          =  씨앗의 질
(다)                 =  씨앗이 없을 때의 바닥
```

## 왜 시드 3개인가

20라운드는 확률적이라 궤적 하나로는 못 가린다 (F-10). **중앙과 최악을
함께** 낸다 — 최고만 보면 시드를 늘릴수록 좋아진다.

## 절차 (§30.8b / D-36)

`core.canonical.canonical_score` 하나만 쓴다 — **루프의 `SplitSet` 을 받아야
돌고**, 형상을 따로 뽑는 경로가 없다. 체제(SOL 2분할)마다 `splits.train` 에서
가중치를 적합하고 `splits.val`(루프가 구조 진화에 안 쓴 20형상)에서 잰다.

★ 전에는 61형상 전체에서 19를 뽑았고 그중 11이 루프의 학습 형상이었다.
그 상태로 "관문 통과" 를 잘못 보고했다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.scoring import compare, evaluate
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.rules.physics_seeded import CODE as PS
from kernelrule.rules.physics_seeded import W0 as PS_W0

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
CONDS = [("가-architect", "Architect 씨앗"), ("나-physics", "physics 씨앗"),
         ("다-noseed", "씨앗 없음"),
         # (나') — 프롬프트 상수를 뺀 뒤 같은 조건 (§12.3b). 차이가 곧
         # "그 누출이 얼마나 도움이 됐는가" 다.
         ("clean-나-physics", "physics 씨앗(정화)"),
         # 피처의 물리적 정의를 Optimizer 에게도 준 뒤 (D-34). (다)와의
         # 차이가 "설명이 실제로 기여했나" 의 측정이다.
         ("desc-다-noseed", "씨앗 없음(설명)")]


def main() -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")

    vendor = load_vendor(VENDOR)
    v_ho = evaluate(vendor_order_fn(table, vendor, mapping="nearest"),
                    table, list(splits.val.shapes), ks=(1,), label="벤더")

    print("=" * 78)
    print("씨앗 절제 — 정준 채점 (루프 분할 그대로, D-36)")
    print("=" * 78)
    print(f"  학습 {len(splits.train.shapes)} (구조+가중치) / "
          f"★구조 홀드아웃 {len(splits.val.shapes)} (루프가 구조 진화에 안 씀)")
    print(f"\n  {'조건':18s} {'시드':>4} {'표본내':>9} {'★구조HO':>9} {'라운드':>6}")

    rows: dict[str, list] = {}
    trajectories: dict[str, list] = {}
    for key, label in CONDS:
        for s in range(3):
            d = Path("runs") / f"seedabl-{key}-s{s}"
            if not (d / "archive.jsonl").exists():
                continue
            with (d / "archive.jsonl").open() as fh:
                arc = [json.loads(ln) for ln in fh if ln.strip()]
            with (d / "rounds.jsonl").open() as fh:
                rds = [json.loads(ln) for ln in fh if ln.strip()]
            best = min(arc, key=lambda e: e["regret"])
            try:
                r = canonical_score(best["code"], best["w"], table=table,
                                    matrix=matrix, splits=splits)
            except Exception as exc:                        # noqa: BLE001
                print(f"  {label:18s} {s:4d}  실패 {type(exc).__name__}")
                continue
            rows.setdefault(label, []).append(r)
            trajectories.setdefault(label, []).append(
                [x["best_regret"] for x in sorted(rds, key=lambda x: x["round"])])
            print(f"  {label:18s} {s:4d} {r.in_sample:9.4f} "
                  f"{r.holdout:9.4f} {len(rds):6d}")
            for w in r.warnings:
                print(f"       ⚠️ {w}")

    print(f"\n  {'조건':18s} {'표본내 중앙':>11} {'★구조HO 중앙':>13} {'최악':>8}")
    for label, rs in rows.items():
        ins = sorted(x.in_sample for x in rs)
        hos = sorted(x.holdout for x in rs)
        print(f"  {label:18s} {ins[len(ins) // 2]:11.4f} "
              f"{hos[len(hos) // 2]:13.4f} {hos[-1]:8.4f}")
    print(f"  {'벤더 nearest ★관문':18s} {'':11s} {v_ho.at(1):13.4f}")
    ps = canonical_score(PS, PS_W0, table=table, matrix=matrix, splits=splits)
    print(f"  {'physics_seeded':18s} {ps.in_sample:11.4f} {ps.holdout:13.4f}")

    print(f"\n{'=' * 78}")
    print("유의성 — 구조 홀드아웃, 조건별 **중앙** 시드")
    print("=" * 78)
    for label, rs in rows.items():
        mid = sorted(rs, key=lambda x: x.holdout)[len(rs) // 2]
        c = compare(mid.evaluation, v_ho, table, name_a="A", name_b="벤더")
        print(f"  {label:18s} {c.geo_a:.4f} vs {c.geo_b:.4f}   "
              f"이김 {int(c.a_wins.sum()):2d} / 짐 {int(c.a_loses.sum()):2d}"
              f" / 구분불가 {int(c.tied.sum()):2d}")

    print(f"\n{'=' * 78}")
    print("라운드별 학습 최고 (시드 중앙값)")
    print("=" * 78)
    print(f"  {'':18s} " + " ".join(f"r{i:<4d}" for i in range(12)))
    for label, trs in trajectories.items():
        n = min(len(t) for t in trs)
        med = [float(np.median([t[i] for t in trs])) for i in range(n)]
        print(f"  {label:18s} " + " ".join(f"{x:5.3f}" for x in med))


if __name__ == "__main__":
    main()
