"""★ 2번 — 생성된 새 축이 **쓸모 있는가**. F1 의 위상을 정한다.

    python3 experiments/new_axes.py [rounds] [n_seeds]

## 왜 이것이 먼저인가

F1 은 "만들 수 있다" 를 보였다. **"쓸모 있다" 는 다른 질문이고, 후자가
없으면 전자가 "가능성 증명" 에 머문다.**

```
조건 A   사람이 쓴 24개              (= 지금 최고, 씨앗 없음 + 설명)
조건 B   24 + 생성된 새 축 N개
같은 시드 3개 / 같은 라운드 / 씨앗 없음 / 설명 있음
```

## 무엇을 "새 축" 으로 보나

생성 피처 중 **기존과 스피어만 > 0.95 인 것을 뺀다** (§8.4 의 중복 판정).
재발견된 것을 넣으면 같은 정보를 두 번 주는 것이라 "새 축이 쓸모 있나" 가
흐려진다. 목록을 손으로 적지 않고 **여기서 계산한다** — 적어 두면 다음에
F1 을 다시 돌렸을 때 어긋난다.

## 판정

```
표본내 regret 비교
★ 생성 피처가 최종 규칙에 실제로 쓰이는가 (몇 개, 어느 것)
구조 홀드아웃은 이 실험이 끝난 뒤 한 번만 (§12.3d)
```

**"쓰이는가" 가 regret 만큼 중요하다.** 37개를 줬는데 최종 규칙이 기존
24개만 쓰면, 생성 피처가 유용하지 않다는 뜻이다.
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.agents.openai_client import Budget, LLMConfig, OpenAILLM
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY, FeatureRegistry
from kernelrule.features.loader import extended_registry, load_generated
from kernelrule.features.validate import _spearman

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
MODEL = "gpt-5.4"
PROPOSALS = Path("runs/featwriter-F1-gpt-5.4/proposals.jsonl")
DUP_RHO = 0.95


def _columns(reg: FeatureRegistry, table, shapes) -> dict:
    mat = FeatureMatrix(table, reg)
    out: dict[str, list] = {n: [] for n in reg._items}
    for p in shapes:
        fe, info = mat.for_shape(p)
        for n, acc in out.items():
            f = reg[n]
            acc.append(np.full(int(info.n_candidates), float(getattr(info, n)))
                       if f.shape_level else np.asarray(getattr(fe, n), float))
    return {n: np.concatenate(v) for n, v in out.items()}


def novel_axes(table, shapes) -> list:
    """생성 피처 중 **기존과 중복이 아닌 것**. 목록을 손으로 적지 않는다."""
    gen = load_generated(PROPOSALS)
    ref = _columns(REGISTRY, table, shapes)
    tmp = FeatureRegistry("gen-probe")
    for f in gen:
        tmp.add(f)
    mine = _columns(tmp, table, shapes)
    novel, dup = [], []
    for f in gen:
        gv = mine[f.name]
        hit = next((rn for rn, rv in ref.items()
                    if len(rv) == len(gv) and abs(_spearman(gv, rv)) > DUP_RHO),
                   None)
        (dup if hit else novel).append((f, hit))
    print(f"  생성 {len(gen)}개 -> 새 축 {len(novel)} / 중복 {len(dup)}")
    for f, hit in dup:
        print(f"    중복  {f.name:30s} ~ {hit}")
    return [f for f, _ in novel]


def used_features(code: str) -> set[str]:
    return {n.attr for n in ast.walk(ast.parse(code.strip()))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id in ("f", "p")}


def main(rounds: int = 12, n_seeds: int = 3) -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")

    print("=" * 78)
    print(f"2번 — 새 축의 쓸모.  시드 {n_seeds} x {rounds}라운드  [{MODEL}]")
    print("=" * 78)
    novel = novel_axes(table, list(table.shapes())[:12])
    ext = extended_registry(REGISTRY, novel)
    new_names = {f.name for f in novel}
    print(f"  조건 A 피처 {len(REGISTRY._items)}  /  조건 B 피처 "
          f"{len(ext._items)}\n")

    budget = Budget(max_calls=2000, max_input_tokens=40_000_000,
                    max_output_tokens=5_000_000)
    t0 = time.perf_counter()
    for cond, reg in (("A-base", REGISTRY), ("B-extended", ext)):
        matrix = FeatureMatrix(table, reg)
        for s in range(n_seeds):
            run_id = f"newaxes-{cond}-s{s}"
            if (Path("runs") / run_id / "archive.jsonl").exists():
                print(f"  [{run_id}] 이미 있다. 건너뛴다")
                continue
            llm = OpenAILLM(LLMConfig(model=MODEL, temperature=0.7,
                                      concurrency=6, seed=20260822 + s),
                            feature_names=matrix.feature_names(),
                            shape_values=matrix.shape_value_names(),
                            registry=reg, budget=budget, cache=False)
            loop = RoundLoop(cfg=LoopConfig(run_id=run_id, max_rounds=rounds,
                                            n_rules_per_round=12, seed=7 + s),
                             table=table, matrix=matrix, splits=splits,
                             llm=llm)
            print(f"\n  --- {run_id} (피처 {len(reg._items)}) ---", flush=True)
            try:
                loop.run(rounds)
            except Exception as e:                          # noqa: BLE001
                print(f"  ★ 중단: {type(e).__name__}: {str(e)[:100]}")
            print(f"  누적 호출 {budget.calls} 입력 {budget.input_tokens:,}"
                  f"  {time.perf_counter() - t0:.0f}s", flush=True)

    # -- 생성 피처가 실제로 쓰였는가 ---------------------------------------
    print(f"\n{'=' * 78}")
    print("생성 피처가 최종 규칙에 쓰였는가 — regret 만큼 중요하다")
    print("=" * 78)
    for s in range(n_seeds):
        d = Path("runs") / f"newaxes-B-extended-s{s}" / "archive.jsonl"
        if not d.exists():
            continue
        with d.open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        best = min(arc, key=lambda e: e["regret"])
        used_best = used_features(best["code"]) & new_names
        used_any: set[str] = set()
        for e in arc:
            used_any |= used_features(e["code"]) & new_names
        print(f"  s{s}  최고 규칙 {len(used_best)}개 {sorted(used_best)}")
        print(f"      아카이브 전체 {len(used_any)}/{len(new_names)}개 "
              f"{sorted(used_any)}")

    print(f"\n  총 {time.perf_counter() - t0:.0f}s  호출 {budget.calls}")
    print("  채점: python3 experiments/score_new_axes.py")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12,
         int(sys.argv[2]) if len(sys.argv) > 2 else 3)
