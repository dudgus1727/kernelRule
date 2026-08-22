"""★ 씨앗 절제 — 진화 결과에서 "씨앗의 질" 과 "리포트 정화" 를 가른다.

    python3 experiments/seed_ablation.py [rounds] [n_seeds]

## 설계

`evolved` 1.0652 는 **두 요인이 섞여** 있다: `physics_seeded` 씨앗 +
**오염된 블록 3.5**(전수 표 집계가 프롬프트에 있었다, D-28). (가)만 돌리면
결과가 어느 쪽이든 원인을 모른다.

```
              오염 리포트      정화 리포트
physics 씨앗   1.0652 (기존)    (나)
Architect 씨앗  —               (가)
씨앗 없음      —                (다)

(나) - 기존  =  정화 효과   ★ 음수일 수 있다 (오염이 홀드아웃 정보를 줬다)
(가) - (나)  =  씨앗의 질
(다)         =  씨앗이 없을 때의 바닥
```

## 시드 3개인 이유

20라운드는 확률적 과정이라 궤적 하나로는 못 가린다. 앞선 경계 탐색에서
같은 조건인데 어떤 궤적은 3.75, 어떤 궤적은 1.10 이 나왔다 (F-10).
**최악값을 병기한다.**

라운드는 12 로 줄였다 — 첫 실행에서 r11 이후 학습 최고가 정체했다.

## 채점

**정준 절차로만 비교한다** (§30.8b): SOL 2분할 -> 체제별 적합 -> 61형상
결합. 유의성은 홀드아웃 19형상에서.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.agents.openai_client import Budget, LLMConfig, OpenAILLM
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.rules.physics_seeded import CODE as PS
from kernelrule.rules.physics_seeded import W0 as PS_W0

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
MODEL = "gpt-5.4"
ARCH_A = Path("runs/architect-A-gpt-5.4/tries.jsonl")

#: 세 조건 전체의 상한. 넘으면 멈춘다 (§4-1).
MAX_CALLS = 2000
MAX_IN, MAX_OUT = 30_000_000, 4_000_000


def _arch_seed():
    with ARCH_A.open() as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    ok = [r for r in rows if "code" in r]
    best = min(ok, key=lambda r: r["train"])
    return best["code"], best["w"]


def main(rounds: int = 12, n_seeds: int = 3, *, only: str = "",
         tag: str = "") -> None:
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

    arch_code, arch_w = _arch_seed()
    conditions = [("가-architect", arch_code, arch_w),
                  ("나-physics", PS, PS_W0),
                  ("다-noseed", None, None)]

    budget = Budget(max_calls=MAX_CALLS, max_input_tokens=MAX_IN,
                    max_output_tokens=MAX_OUT)
    print("=" * 76)
    print(f"씨앗 절제 — 3조건 x 시드 {n_seeds} x {rounds}라운드  [{MODEL}]")
    print("=" * 76)
    print(f"  학습 {len(splits.train.shapes)} / 검증 {len(splits.val.shapes)}"
          f"   빠른 {sum(1 for p in shapes if regime_of(p, table.hw) == 'short')}"
          f" / 느린 {sum(1 for p in shapes if regime_of(p, table.hw) == 'long')}")
    print("  ★ 리포트는 전부 정화판 (TableFacts, 학습 분할에서만) — D-28\n")

    t0 = time.perf_counter()
    if only:
        conditions = [c for c in conditions if c[0] == only]
        if not conditions:
            raise SystemExit(f"알 수 없는 조건: {only!r}")
    for name, code, w0 in conditions:
        for s in range(n_seeds):
            run_id = f"seedabl{tag}-{name}-s{s}"
            if (Path("runs") / run_id / "archive.jsonl").exists():
                print(f"  [{run_id}] 이미 있다. 건너뛴다")
                continue
            llm = OpenAILLM(LLMConfig(model=MODEL, temperature=0.7,
                                      concurrency=6, seed=20260821 + s),
                            feature_names=matrix.feature_names(),
                            shape_values=matrix.shape_value_names(),
                            registry=REGISTRY, budget=budget, cache=False)
            loop = RoundLoop(cfg=LoopConfig(run_id=run_id, max_rounds=rounds,
                                            n_rules_per_round=12, seed=7 + s),
                             table=table, matrix=matrix, splits=splits, llm=llm)
            if code is not None:
                loop.seed(code, w0)
            print(f"\n  --- {run_id} ---")
            try:
                loop.run(rounds)
                loop.dump()
            except Exception as e:                          # noqa: BLE001
                print(f"  ★ 중단: {type(e).__name__}: {str(e)[:100]}")
            print(f"  누적 호출 {budget.calls} (실패 {budget.failed_calls}) "
                  f"입력 {budget.input_tokens:,} 출력 {budget.output_tokens:,}"
                  f"  {time.perf_counter() - t0:.0f}s")

    print(f"\n총 {time.perf_counter() - t0:.0f}s  호출 {budget.calls}"
          f"  입력 {budget.input_tokens:,}  출력 {budget.output_tokens:,}")
    print("\n채점: python3 experiments/rescore_canonical.py "
          "(runs/seedabl-* 를 읽도록 확장한 뒤)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12,
         int(sys.argv[2]) if len(sys.argv) > 2 else 3,
         only=sys.argv[3] if len(sys.argv) > 3 else "",
         tag=sys.argv[4] if len(sys.argv) > 4 else "")
