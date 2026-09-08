"""★ 7. The RuleWriter A/B condition — the pass condition of the transfer
claim.

    python3 experiments/rule_writer_gate.py A 10
    python3 experiments/rule_writer_gate.py B 10

## What is measured

Moving the rule to a new architecture requires **the structure to come out
without the table.** If the structure only comes out after seeing the table,
that is §29.5(c) regrow, and if an exhaustive measurement is going to be made
anyway, the table can be used directly and there is no reason for this
system.

    condition A   the hardware facts + the execution model + the physical
                  definitions of the features, and nothing else
    condition B   plus the aggregates **of the training split** (the shapes
                  cannot be identified)

For each attempt the weights are refitted on the training split (§29 — to
compare structures, the luck of the weights has to be removed). The judgement
is made on **the validation split**.

    training regret near 1.07   ->  the pass condition passes. The structure
                                    is generated without the table and
                                    refitted on a sample
    1.15+                       ->  generating the structure needs the table

The gap between A and B is exactly "what the table is worth". **It does not
mean giving B up.**
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
    # ★ The structural split — the whole 11008 layer (the MLP intermediate)
    #   is held out. It is the same split as the first real run, so the
    #   results can be put side by side. A random k-fold is not used for the
    #   reason in §10.1 — if the same layer type is mixed into both sides,
    #   the holdout is not a holdout.
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
    print(f"7. RuleWriter condition {condition} — {n_tries} attempts  "
          f"[{model}]")
    print("=" * 76)
    print(f"  training {len(train.shapes)} shapes / validation "
          f"{len(val.shapes)} shapes")
    print("  under condition A there is **not one** sentence in the prompt "
          "that came from the table\n")

    # ★ Every attempt is appended immediately (D-33). Writing once at the end
    #   loses every LLM call up to that point if it dies in the middle — 78
    #   minutes' worth really were lost in RoundLoop. The longer the run, the
    #   more likely it is to die.
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
            out = llm.complete("rule_writer", "", condition=condition,
                               table_facts=facts)
            prop = validate_rule_proposal(out)
            # ★ `check_rule` returns a report. Leaving out `.raise_if_bad()`
            #   lets a violation pass silently and it comes out as an
            #   AttributeError in the scorer — which is what the very first
            #   call did.
            check_rule(prop.code, feature_names=matrix.feature_names(),
                       shape_value_names=matrix.shape_value_names(),
                       n_weights=len(prop.w0)).raise_if_bad()
            fit = fit_weights(compile_rule(prop.code), matrix, table, train,
                              prop.w0, max_evals=300,
                          objective="regret")
            so = make_score_of(compile_rule(prop.code), matrix, fit.w)
            tr = evaluate_scores(so, table, train.shapes, ks=(1,)).at(1)
            va = evaluate_scores(so, table, val.shapes, ks=(1,)).at(1)
            n_terms = prop.code.count("s = s +") + 1
            record({"i": i, "train": tr, "val": va, "n_terms": n_terms,
                    "n_w": len(prop.w0), "code": prop.code,
                    "w": list(fit.w), "changes": prop.changes})
            print(f"  #{i:02d}  train {tr:.4f}  val {va:.4f}  "
                  f"{n_terms} terms/{len(prop.w0)}w")
        except (SchemaViolation, RuleCheckError) as e:
            record({"i": i, "error": f"{type(e).__name__}: {e}"})
            print(f"  #{i:02d}  refused  {type(e).__name__}: {str(e)[:60]}")
        except Exception as e:                              # noqa: BLE001
            record({"i": i, "error": f"{type(e).__name__}: {e}"})
            print(f"  #{i:02d}  failed   {type(e).__name__}: {str(e)[:70]}")

    # ★ An LLM call cannot be made again (D-33). It is kept before the scoring.
    llm.dump(d / "llm_calls")

    ok = [r for r in rows if "train" in r]
    v = evaluate(vendor_order_fn(table, load_vendor(VENDOR), mapping="nearest"),
                 table, list(train.shapes), ks=(1,))
    v_val = evaluate(vendor_order_fn(table, load_vendor(VENDOR),
                                     mapping="nearest"),
                     table, list(val.shapes), ks=(1,))
    print(f"\n  succeeded {len(ok)}/{n_tries}   "
          f"{time.perf_counter() - t0:.0f}s"
          f"   calls {llm.budget.calls} (failed {llm.budget.failed_calls})"
          f"  input {llm.budget.input_tokens:,}"
          f"  output {llm.budget.output_tokens:,}")
    # ★ Without knowing what exhausted the retries there is no knowing where
    #   to fix the prompt.
    vr = llm.violation_report()
    if vr.get("total"):
        print(f"  {vr['total']} violations: {vr['by_code']}")
        if vr.get("useless_retries"):
            print(f"  ★ reasons the feedback is not getting through: "
                  f"{vr['useless_retries']}")
    if ok:
        tr = np.array([r["train"] for r in ok])
        va = np.array([r["val"] for r in ok])
        best = min(ok, key=lambda r: r["train"])
        print(f"\n  {'':10s} {'best':>8} {'median':>8} {'worst':>8}")
        print(f"  {'train':10s} {tr.min():8.4f} {np.median(tr):8.4f} "
              f"{tr.max():8.4f}")
        print(f"  {'val':10s} {va.min():8.4f} {np.median(va):8.4f} "
              f"{va.max():8.4f}")
        print(f"  {'vendor':10s} {v.at(1):8.4f} (train) / "
              f"{v_val.at(1):.4f} (val)")
        print(f"\n  ★ the pass condition: the best train {tr.min():.4f}  "
              f"-> {'pass (near 1.07)' if tr.min() < 1.09 else 'short'}")
        print(f"\n  the best rule (#{best['i']}, {best['n_terms']} terms):")
        print("  " + best["code"].strip().replace("\n", "\n  "))
        print(f"  w = {[round(x, 3) for x in best['w']]}")
        print(f"  the physical explanation: {best['changes'][:300]}")

    print(f"\n  -> {d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A",
         int(sys.argv[2]) if len(sys.argv) > 2 else 10,
         sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL)
