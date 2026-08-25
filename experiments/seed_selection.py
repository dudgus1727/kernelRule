"""★ 시드 선택이 실력인가 선택 편향인가.

    python3 experiments/seed_selection.py [n_seeds] [seed_base] [tag]

## 무엇을 확인하나

같은 조건의 시드 6개에서 **표본내 점수가 구조 홀드아웃을 잘 예측했다**
(스피어만 0.943). 표본내 최소 시드를 고르면 홀드아웃 1.0518 로, 무작위
시드의 중앙 1.0817 보다 훨씬 낫고 벤더 1.0737 도 앞선다.

**그런데 그 1.0518 은 편향 없는 추정치가 아니다.** 6개 중 최소를 골랐으니
그 홀드아웃 값도 6개 중 좋은 쪽일 확률이 높다. 상관이 높을수록 더 그렇다.

그래서 **절차를 고정하고 새 시드 묶음에서 다시 잰다.**

```
새 시드 6개 -> 표본내 최소 시드 하나 -> 그 홀드아웃 값
  1.05 근처면  실력. 절차로 확정
  1.08 근처면  선택 편향이었다
```

## 절차가 §10.2 를 어기지 않는 이유

선택 신호가 **전부 학습 분할에서** 나온다. 하이퍼파라미터 탐색과 같은
구조다 — 학습에서 고르고 홀드아웃에서 한 번 잰다.

## 왜 시드 사이에서만 예측이 되는가

아카이브 **안** 규칙들은 서로 비슷해 학습 점수 0.001 차이가 노이즈다
(상위 10개 중 홀드아웃 최고를 골라도 1.0817 -> 1.0778 뿐이다).
**시드가 다르면 진화 궤적 자체가 달라 품질 차이가 실재한다.**
"아카이브 안에서 고르는 것" 과 "실행을 고르는 것" 은 다른 문제다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.agents.openai_client import DEFAULT_MODEL, Budget, LLMConfig, OpenAILLM
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.canonical import canonical_score
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.scoring import evaluate
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
MODEL = DEFAULT_MODEL   # ★ 단일 출처 (D-45)

#: 앞선 두 묶음. 12시드 합산 분포에 쓴다.
PRIOR = [f"seedabl-desc-다-noseed-s{s}" for s in range(3)] + \
        [f"newaxes-A-base-s{s}" for s in range(3)]


def _setup(table):
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def main(n_seeds: int = 6, seed_base: int = 20260823,
         tag: str = "selB") -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    splits = _setup(table)

    budget = Budget(max_calls=3000, max_input_tokens=60_000_000,
                    max_output_tokens=8_000_000)
    print("=" * 76)
    print(f"시드 선택 확인 — 새 시드 {n_seeds}개  [{MODEL}]  tag={tag}")
    print("=" * 76)
    print("  조건: 씨앗 없음 + 피처 설명 + 기본 24개 (= A-base 와 동일)")
    print(f"  학습 {len(splits.train.shapes)} / 구조 홀드아웃 "
          f"{len(splits.val.shapes)}\n")

    t0 = time.perf_counter()
    for s in range(n_seeds):
        run_id = f"{tag}-s{s}"
        if (Path("runs") / run_id / "archive.jsonl").exists():
            print(f"  [{run_id}] 이미 있다. 건너뛴다")
            continue
        llm = OpenAILLM(LLMConfig(model=MODEL, temperature=0.7, concurrency=6,
                                  seed=seed_base + s),
                        feature_names=matrix.feature_names(),
                        shape_values=matrix.shape_value_names(),
                        registry=REGISTRY, budget=budget, cache=False)
        loop = RoundLoop(cfg=LoopConfig(run_id=run_id, max_rounds=12,
                                        n_rules_per_round=12, seed=100 + s),
                         table=table, matrix=matrix, splits=splits, llm=llm)
        print(f"\n  --- {run_id} ---", flush=True)
        try:
            loop.run(12)
        except Exception as e:                              # noqa: BLE001
            print(f"  ★ 중단: {type(e).__name__}: {str(e)[:100]}")
        print(f"  누적 호출 {budget.calls}  {time.perf_counter() - t0:.0f}s",
              flush=True)

    # -- 채점 -------------------------------------------------------------
    def score(run_id: str):
        f = Path("runs") / run_id / "archive.jsonl"
        if not f.exists():
            return None
        with f.open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        if not arc:
            # ★ 빈 아카이브는 "나쁜 실행" 이 아니라 **실행이 안 된 것**이다.
            #   채점에서 조용히 0 으로 넣으면 분포가 오염된다 (§26.4).
            print(f"  {run_id:16s} ⚠️ 아카이브가 비었다 — 채점에서 제외")
            return None
        best = min(arc, key=lambda e: e["regret"])
        return canonical_score(best["code"], best["w"], table=table,
                               matrix=matrix, splits=splits)

    v = evaluate(vendor_order_fn(table, load_vendor(VENDOR),
                                 mapping="nearest"),
                 table, list(splits.val.shapes), ks=(1,))

    print(f"\n{'=' * 76}")
    print(f"새 묶음 {n_seeds}시드 — ★ 표본내로 고르고 홀드아웃은 한 번만 본다")
    print("=" * 76)
    print(f"  {'실행':16s} {'표본내':>9} {'구조HO':>9}")
    new = []
    for s in range(n_seeds):
        r = score(f"{tag}-s{s}")
        if r is None:
            continue
        new.append((f"{tag}-s{s}", r.in_sample, r.holdout))
        print(f"  {f'{tag}-s{s}':16s} {r.in_sample:9.4f} {r.holdout:9.4f}")
    if not new:
        return
    pick = min(new, key=lambda x: x[1])
    ho = np.array([x[2] for x in new])
    print(f"\n  ★ 표본내 최소 시드: {pick[0]}   구조HO {pick[2]:.4f}")
    print(f"     무작위 시드 중앙 {np.median(ho):.4f}  최악 {ho.max():.4f}")
    print(f"     벤더 {v.at(1):.4f}   기존 묶음의 선택값 1.0518")
    verdict = ("실력 — 절차로 확정" if pick[2] < 1.065
               else "선택 편향이었다" if pick[2] > 1.075 else "중간. 판단 보류")
    print(f"     ★ 판정: {verdict}")

    # -- 12시드 합산 --------------------------------------------------------
    print(f"\n{'=' * 76}")
    print("두 묶음 합산 — 폭이 안정적인가")
    print("=" * 76)
    allr = []
    for run in PRIOR:
        r = score(run)
        if r:
            allr.append((run, r.in_sample, r.holdout))
    allr += new
    a = np.array([[x[1], x[2]] for x in allr])
    from kernelrule.features.validate import _pearson, _spearman
    print(f"  n={len(allr)}   구조HO 중앙 {np.median(a[:, 1]):.4f}  "
          f"최소 {a[:, 1].min():.4f}  최대 {a[:, 1].max():.4f}  "
          f"폭 {a[:, 1].max() - a[:, 1].min():.4f}")
    print(f"  표본내 vs 홀드아웃  스피어만 {_spearman(a[:, 0], a[:, 1]):.3f}"
          f"  피어슨 {_pearson(a[:, 0], a[:, 1]):.3f}")
    print(f"  전체에서 표본내 최소를 고르면: "
          f"{min(allr, key=lambda x: x[1])[0]} "
          f"-> {min(allr, key=lambda x: x[1])[2]:.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6,
         int(sys.argv[2]) if len(sys.argv) > 2 else 20260823,
         sys.argv[3] if len(sys.argv) > 3 else "selB")
