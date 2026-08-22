"""★ F1 — LLM 이 물리량을 **만들** 수 있는가. 이 프로젝트의 근본 질문.

    python3 experiments/feature_writer.py [n_proposals] [condition] [model]

## 왜 근본 질문인가

지금까지의 모든 실행에서 LLM 은 **사람이 쓴 24개를 조합**했을 뿐이다.
`tail_waste` 를 **정의하는 것**이 물리를 이해하는 일이고, §29 로 가중치까지
떼어냈으니 남은 것은 "24개 중 8개 고르기" 다.

    F0  피처 없음     물리를 처음부터 코드로 옮길 수 있나
    F1  원시 값만     파생 물리량을 만들 수 있나        ★ 여기
    F2  기초 5개      그 위에 쌓을 수 있나
    F3  24개 전부     조합만 (= 지금까지)

`tail_waste` 를 만들려면 `ceil(M/tile_m)*ceil(N/tile_n)/sm_count` 를 유도해야
하고, 그것이 **wave quantization 을 이해했다는 증거**다.

## 평가 — 재발견

이름이 달라도 **수학적으로 같으면** 재발견으로 센다. 기준은 §8.4 와 같다:
스피어만 **과** 피어슨이 둘 다 0.95 를 넘으면 같은 것을 재는 것이다.
(둘 다 보는 이유는 단조 변환과 선형 관계를 구분하기 위해서다 — `sm_idle_cost`
와 `tail_waste` 는 스피어만 1.0 이지만 피어슨은 낮다.)

**부분 성공도 결과다.** 절반을 재발견하면 "LLM 이 물리량을 유도할 수 있다"
는 증거이고, 나머지는 사람이 보충하면 된다. **실패해도 결과다** —
"피처 정의는 사람이, 조합은 LLM 이" 가 정확한 역할 분담이라는 확인이다.

## ⚠️ 구조 홀드아웃을 보지 않는다 (§12.3d)

이 실험의 판정은 **재발견 수**와 **표본내 regret** 으로만 한다. 구조
홀드아웃은 F1 이 끝난 뒤 한 번만 본다.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.agents.openai_client import Budget, LLMConfig, OpenAILLM
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY, FeatureRegistry
from kernelrule.features.generated import (
    FeatureRejected,
    _reference_columns,
    register_generated,
)
from kernelrule.features.validate import _pearson, _spearman

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
DEFAULT_MODEL = "gpt-5.4"
OUT = Path("runs")

#: 재발견 판정 (§8.4 와 같은 기준). 스피어만 **과** 피어슨 둘 다.
RHO = 0.95


def _alt_hw(hw):
    """스케일 불변성 검사용 가짜 하드웨어.

    ★ **모든 수치 필드를 바꾼다.** 하나라도 그대로 두면 그것에 물린 피처가
    "hw 를 안 쓴다" 로 오판된다. 실제로 `max_threads_per_sm` 을 안 바꿔서
    occupancy 피처가 16회 연속 기각됐다 — `min(by_threads, ...)` 이 그
    항에 물려 값이 안 변했기 때문이다 (D-38).
    """
    return replace(hw, sm_count=hw.sm_count * 2 + 3,
                   smem_per_block=int(hw.smem_per_block * 0.7),
                   max_threads_per_sm=int(hw.max_threads_per_sm * 1.33),
                   peak_tflops_f16=hw.peak_tflops_f16 * 1.6,
                   bandwidth_gbps=hw.bandwidth_gbps * 0.8,
                   regs_per_sm=int(hw.regs_per_sm * 1.5),
                   l2_bytes=int(hw.l2_bytes * 2))


def _columns(reg: FeatureRegistry, table, matrix, shapes) -> dict:
    """피처별 값 벡터. 재발견 비교에 쓴다."""
    out: dict[str, list] = {n: [] for n in reg._items}
    for p in shapes:
        feats, info = matrix.for_shape(p)
        for n, acc in out.items():
            f = reg[n]
            v = getattr(info, n) if f.shape_level else getattr(feats, n)
            acc.append(np.full(1, float(v)) if f.shape_level
                       else np.asarray(v, float))
    return {n: np.concatenate(v) for n, v in out.items()}


def main(n_proposals: int = 20, condition: str = "F1",
         model: str = DEFAULT_MODEL) -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    hw_alt = _alt_hw(table.hw)

    gen = FeatureRegistry(f"generated-{condition}")
    llm = OpenAILLM(LLMConfig(model=model, temperature=1.0, concurrency=4),
                    feature_names=[], shape_values=[], registry=REGISTRY,
                    budget=Budget(max_calls=200), cache=False)

    d = OUT / f"featwriter-{condition}-{model}"
    d.mkdir(parents=True, exist_ok=True)
    log = d / "proposals.jsonl"
    log.write_text("")

    print("=" * 78)
    print(f"F1 — FeatureWriter 조건 {condition}, 제안 {n_proposals}회  [{model}]")
    print("=" * 78)
    print("  ★ 원시 값만 준다. 기존 피처 이름은 프롬프트에 하나도 없다"
          if condition in ("F0", "F1") else "  기존 피처를 보여준다")
    print()

    # ★ 사람이 쓴 24개의 기준 열은 바뀌지 않는다. 한 번만 만든다.
    ref_cols = _reference_columns(table, matrix, FeatureRegistry("empty"))

    t0 = time.perf_counter()
    failures: list[tuple[str, str]] = []
    for i in range(n_proposals):
        made = sorted(gen._items)
        # ★ 실패를 되먹인다. 안 주면 **같은 제안을 반복한다** — 첫 실행에서
        #   20회 중 16회가 같은 피처였다 (D-38). 132/240 재시도 소진과 같은
        #   병이다: 무엇이 틀렸는지 모르면 고칠 수 없다.
        recent = ""
        if failures:
            lines = "\n".join(f"- `{n}` — {e}" for n, e in failures[-5:])
            recent = ("\n\n## ★ 방금 거부된 것들 — 같은 것을 다시 내지 "
                      f"마세요\n\n{lines}\n\n거부 사유를 읽고 **다른 축**을 "
                      "찾거나, 같은 축이면 그 사유를 고쳐서 내세요.")
        task = ("## 이번에 만들 것\n\n피처 하나를 제안하세요."
                + (f"\n\n지금까지 만든 것: {made}\n**이것들과 다른 축**을 "
                   "찾으세요." if made else "") + recent)
        row: dict = {"i": i}
        try:
            out = llm.complete("feature", "", condition=condition, task=task,
                               registry=REGISTRY)
            row |= {k: out.get(k) for k in
                    ("name", "code", "rationale", "unit", "expected_range",
                     "direction")}
            f = register_generated(
                out["code"], registry=gen, meta=out, table=table,
                matrix=matrix, hw_alt=hw_alt,
                others=ref_cols | _reference_columns(
                    table, matrix, gen) if gen._items else ref_cols)
            row["accepted"] = True
            print(f"  #{i:02d}  ✓ {f.name:28s} {f.expected_range}", flush=True)
        except FeatureRejected as e:
            row |= {"accepted": False, "error": str(e)}
            failures.append((str(row.get("name") or "?"), str(e)[:180]))
            print(f"  #{i:02d}  ✗ {str(e)[:80]}", flush=True)
        except Exception as e:                              # noqa: BLE001
            row |= {"accepted": False, "error": f"{type(e).__name__}: {e}"}
            print(f"  #{i:02d}  ! {type(e).__name__}: {str(e)[:70]}")
        with log.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print(f"\n  채택 {len(gen._items)}/{n_proposals}   "
          f"{time.perf_counter() - t0:.0f}s   호출 {llm.budget.calls}"
          f"  입력 {llm.budget.input_tokens:,}  출력 {llm.budget.output_tokens:,}")
    if not gen._items:
        return

    # -- 재발견 판정 -------------------------------------------------------
    shapes = list(table.shapes())[:8]
    mine = _columns(gen, table, FeatureMatrix(table, gen), shapes)
    ref = _columns(REGISTRY, table, matrix, shapes)

    print(f"\n{'=' * 78}")
    print(f"재발견 — 사람이 쓴 {len(ref)}개 중 몇 개를 다시 만들었나")
    print(f"  기준: 스피어만 **과** 피어슨 둘 다 > {RHO} (§8.4)")
    print("=" * 78)
    found: dict[str, tuple[str, float, float]] = {}
    for gname, gv in mine.items():
        for rname, rv in ref.items():
            if len(gv) != len(rv):
                continue
            sp, pe = abs(_spearman(gv, rv)), abs(_pearson(gv, rv))
            if sp > RHO and pe > RHO:
                prev = found.get(rname)
                if prev is None or sp + pe > prev[1] + prev[2]:
                    found[rname] = (gname, sp, pe)
    for rname, (gname, sp, pe) in sorted(found.items()):
        print(f"  {rname:26s} <- {gname:26s} sp {sp:.3f}  pe {pe:.3f}")
    print(f"\n  ★ 재발견 {len(found)}/{len(ref)}   "
          f"새 축 {len(gen._items) - len(found)}개")

    novel = [n for n in sorted(gen._items)
             if n not in {g for g, _, _ in found.values()}]
    if novel:
        print(f"\n  기존 24개에 없던 축: {novel}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20,
         sys.argv[2] if len(sys.argv) > 2 else "F1",
         sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL)
