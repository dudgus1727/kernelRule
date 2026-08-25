"""★ 7. Architect A/B 조건 — 전이 주장의 관문.

    python3 experiments/architect_gate.py A 10
    python3 experiments/architect_gate.py B 10

## 무엇을 재는가

새 아키텍처로 규칙을 옮기려면 **표 없이 구조가 나와야 한다.** 표를 봐야
구조가 나오면 §29.5(c) 재생성이고, 전수를 잴 거면 표를 직접 쓰면 되므로
이 시스템을 쓸 이유가 없다.

    A 조건   하드웨어 사실 + 실행 모델 + 피처의 물리적 정의만
    B 조건   거기에 **학습 분할의** 집계를 더한다 (형상 식별 불가)

각 시도마다 가중치는 학습 분할에서 재적합한다 (§29 — 구조를 비교하려면
가중치 운을 제거해야 한다). 판정은 **검증 분할**로 한다.

    학습 regret 1.07 근처   ->  관문 통과. 표 없이 구조 생성 + 표본 재적합
    1.15+                   ->  구조 생성에 표가 필요하다

A 와 B 의 격차가 곧 "표의 값어치" 다. **B 를 포기하는 것이 아니다.**
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.agents.openai_client import DEFAULT_MODEL, Budget, LLMConfig, OpenAILLM
from kernelrule.agents.schemas import SchemaViolation, validate_rule_proposal
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate, evaluate_scores
from kernelrule.core.splits import Split, SplitSet, check_balance
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.report.table_facts import TableFacts
from kernelrule.rules.checks import RuleCheckError, check_rule

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
OUT = Path("runs")


def main(condition: str, n_tries: int,
         model: str = DEFAULT_MODEL) -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    # ★ 구조 분할 — 11008 레이어(MLP intermediate)를 통째로 홀드아웃.
    #   첫 실제 실행과 같은 분할이라 결과를 나란히 놓을 수 있다.
    #   무작위 k-fold 를 쓰지 않는 이유는 §10.1 — 같은 레이어 타입이
    #   양쪽에 섞이면 홀드아웃이 홀드아웃이 아니다.
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(train=Split("train", tuple(p for p in shapes
                                                 if p not in held)),
                      val=Split("val", tuple(held)), kind="nk11008")
    train, val = splits.train, splits.val
    check_balance(train, table.hw)
    facts = TableFacts.compute(table, train)

    llm = OpenAILLM(LLMConfig(model=model, concurrency=5),
                    feature_names=matrix.feature_names(),
                    shape_values=matrix.shape_value_names(),
                    registry=REGISTRY, budget=Budget(), cache=False)

    print("=" * 76)
    print(f"7. Architect 조건 {condition} — {n_tries}회  [{model}]")
    print("=" * 76)
    print(f"  학습 {len(train.shapes)}형상 / 검증 {len(val.shapes)}형상")
    print("  A 조건이면 프롬프트에 표에서 나온 문장이 **하나도** 없다\n")

    # ★ 시도마다 즉시 append 한다 (D-33). 끝에서 한 번 쓰면 중간에 죽을 때
    #   그때까지의 LLM 호출이 통째로 날아간다 — 실제로 RoundLoop 에서
    #   78분치를 잃었다. 오래 걸리는 실행일수록 죽을 확률이 높다.
    d = OUT / f"architect-{condition}-{model}"
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / "tries.jsonl"
    out_path.write_text("")

    def record(row: dict) -> None:
        rows.append(row)
        with out_path.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = []
    t0 = time.perf_counter()
    for i in range(n_tries):
        try:
            out = llm.complete("architect", "", condition=condition,
                               table_facts=facts)
            prop = validate_rule_proposal(out)
            # ★ `check_rule` 은 리포트를 돌려준다. `.raise_if_bad()` 를
            #   빠뜨리면 위반이 조용히 통과하고 채점기에서 AttributeError 로
            #   나온다 — 실제로 첫 호출이 그랬다.
            check_rule(prop.code, feature_names=matrix.feature_names(),
                       shape_value_names=matrix.shape_value_names(),
                       n_weights=len(prop.w0)).raise_if_bad()
            fit = fit_weights(compile_rule(prop.code), matrix, table, train,
                              prop.w0, max_evals=300)
            so = make_score_of(compile_rule(prop.code), matrix, fit.w)
            tr = evaluate_scores(so, table, train.shapes, ks=(1,)).at(1)
            va = evaluate_scores(so, table, val.shapes, ks=(1,)).at(1)
            n_terms = prop.code.count("s = s +") + 1
            record({"i": i, "train": tr, "val": va, "n_terms": n_terms,
                    "n_w": len(prop.w0), "code": prop.code,
                    "w": list(fit.w), "changes": prop.changes})
            print(f"  #{i:02d}  train {tr:.4f}  val {va:.4f}  "
                  f"{n_terms}항/{len(prop.w0)}w")
        except (SchemaViolation, RuleCheckError) as e:
            record({"i": i, "error": f"{type(e).__name__}: {e}"})
            print(f"  #{i:02d}  거부  {type(e).__name__}: {str(e)[:60]}")
        except Exception as e:                              # noqa: BLE001
            record({"i": i, "error": f"{type(e).__name__}: {e}"})
            print(f"  #{i:02d}  실패  {type(e).__name__}: {str(e)[:70]}")

    ok = [r for r in rows if "train" in r]
    v = evaluate(vendor_order_fn(table, load_vendor(VENDOR), mapping="nearest"),
                 table, list(train.shapes), ks=(1,))
    v_val = evaluate(vendor_order_fn(table, load_vendor(VENDOR),
                                     mapping="nearest"),
                     table, list(val.shapes), ks=(1,))
    print(f"\n  성공 {len(ok)}/{n_tries}   {time.perf_counter() - t0:.0f}s"
          f"   호출 {llm.budget.calls} (실패 {llm.budget.failed_calls})"
          f"  입력 {llm.budget.input_tokens:,}"
          f"  출력 {llm.budget.output_tokens:,}")
    # ★ 재시도 소진이 무엇 때문인지 모르면 프롬프트를 어디를 고칠지 모른다.
    vr = llm.violation_report()
    if vr.get("total"):
        print(f"  위반 {vr['total']}건: {vr['by_code']}")
        if vr.get("useless_retries"):
            print(f"  ★ 되먹임이 듣지 않는 사유: {vr['useless_retries']}")
    if ok:
        tr = np.array([r["train"] for r in ok])
        va = np.array([r["val"] for r in ok])
        best = min(ok, key=lambda r: r["train"])
        print(f"\n  {'':10s} {'최고':>8} {'중앙':>8} {'최악':>8}")
        print(f"  {'train':10s} {tr.min():8.4f} {np.median(tr):8.4f} "
              f"{tr.max():8.4f}")
        print(f"  {'val':10s} {va.min():8.4f} {np.median(va):8.4f} "
              f"{va.max():8.4f}")
        print(f"  {'벤더':10s} {v.at(1):8.4f} (train) / {v_val.at(1):.4f} (val)")
        print(f"\n  ★ 관문: train 최고 {tr.min():.4f}  "
              f"-> {'통과 (1.07 근처)' if tr.min() < 1.09 else '미달'}")
        print(f"\n  최고 규칙 (#{best['i']}, {best['n_terms']}항):")
        print("  " + best["code"].strip().replace("\n", "\n  "))
        print(f"  w = {[round(x, 3) for x in best['w']]}")
        print(f"  물리 설명: {best['changes'][:300]}")

    print(f"\n  -> {d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A",
         int(sys.argv[2]) if len(sys.argv) > 2 else 10,
         sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL)
