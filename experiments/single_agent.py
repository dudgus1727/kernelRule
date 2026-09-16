"""★ The single-agent control — N=10, no loop (D-176 §2). **0 LLM calls**
(the 40 generations already ran; this scores them).

    python3 -m experiments.single_agent

```
⛔ The loop is not run. One RuleWriter call, ten times, pick by training.
⚠️ The budgets are NOT matched — single agent 10 LLM calls, the loop
   12 rounds x 7 = 103. That asymmetry is stated, not corrected (§2-5).
```

★ **All ten candidates are scored on the holdout**, not only the chosen
one. The chosen one is picked by **training** regret (⛔ the holdout is not
consulted for selection); scoring all ten afterwards is what makes
"where did the training winner land on the holdout" answerable — the size
of the selection overfit.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.f1_pipeline import _load_stage1, _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry

GPUS = ("a6000", "5090", "4090", "h100")
OUT = Path("docs/artifacts/single-agent.json")


def _shape(code: str, shape_names: set[str]) -> dict:
    return {
        "n_terms": len({int(m.group(1))
                        for m in re.finditer(r"w\[(\d+)\]", code)}),
        "n_paths": code.count("np.where")
        + len(re.findall(r"^\s+if ", code, re.M)),
        "branch_axes": sorted({m.group(1)
                               for m in re.finditer(r"\bp\.(\w+)", code)
                               if m.group(1) in shape_names})}


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    camp = json.loads(Path("docs/artifacts/campaign2.json").read_text())
    c2 = {}
    for r in camp["runs"]:
        if r.get("missing") or r["fold"] != 0:
            continue
        c2.setdefault(r["gpu"], []).append(r["holdout"])

    rows = []
    print("=" * 100)
    print("★ single agent N=10 vs the loop (D-176 §2). 0 LLM calls here")
    print("=" * 100)
    for gpu in GPUS:
        d = Path(f"runs/sa-{gpu}-f0")
        if not (d / "stage2-rule-writer" / "summary.json").exists():
            continue
        s = json.loads((d / "stage2-rule-writer" / "summary.json").read_text())
        T = TABLES[gpu]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        splits = _splits(table, fold=0, k=4, design="nkband")
        reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2",
                           table)
        m = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
        names = {n for n in reg._items if reg[n].shape_level}
        cands = []
        for t in s["tries"]:
            if not t.get("ok"):
                continue
            cs = canonical_score(t["code"], np.asarray(t["w0"], float),
                                 table=table, matrix=m, splits=splits)
            cands.append({"i": t["i"], "train": t["fit_regret"],
                          "holdout": round(cs.holdout, 6),
                          **_shape(t["code"], names)})
        by_train = sorted(cands, key=lambda c: c["train"])
        by_hold = sorted(cands, key=lambda c: c["holdout"])
        winner = by_train[0]
        rank = [c["i"] for c in by_hold].index(winner["i"]) + 1
        c2v = sorted(c2.get(gpu, []))
        rows.append({
            "gpu": gpu, "n_ok": s["n_ok"], "n_tries": s["n_tries"],
            "llm_calls": s["n_tries"],
            "chosen_i": winner["i"],
            "chosen_train": winner["train"],
            "chosen_holdout": winner["holdout"],
            "★ train_winner_holdout_rank": rank,
            "best_possible_holdout": by_hold[0]["holdout"],
            "selection_overfit": round(winner["holdout"]
                                       - by_hold[0]["holdout"], 6),
            "train_spread": [by_train[0]["train"], by_train[-1]["train"]],
            "holdout_spread": [by_hold[0]["holdout"], by_hold[-1]["holdout"]],
            "n_terms_median": st.median([c["n_terms"] for c in cands]),
            "n_paths_median": st.median([c["n_paths"] for c in cands]),
            "branch_axes_chosen": winner["branch_axes"],
            "c2_loop_holdout_seeds": c2v,
            "c2_loop_holdout_median": (round(st.median(c2v), 6)
                                       if c2v else None),
            "c2_loop_llm_calls": 103,
            "candidates": cands})
        r = rows[-1]
        print(f"  {gpu:6s} 통과 {s['n_ok']}/{s['n_tries']}  "
              f"고른것 학습 {winner['train']:.4f} 홀드아웃 {winner['holdout']:.4f}"
              f"  ★ 홀드아웃 순위 {rank}/10  "
              f"최선 {by_hold[0]['holdout']:.4f}  "
              f"| 루프(중앙) {r['c2_loop_holdout_median']:.4f}")
    agg = {
        "n": len(rows),
        "★ note": ("⚠️ budgets are NOT matched — 10 LLM calls against the "
                   "loop's 103 (12 rounds x 7). Stated, not corrected."),
        "single_beats_loop": sum(1 for r in rows
                                 if r["c2_loop_holdout_median"] is not None
                                 and r["chosen_holdout"]
                                 < r["c2_loop_holdout_median"]),
        "median_rank_of_train_winner": st.median(
            [r["★ train_winner_holdout_rank"] for r in rows]),
        "median_selection_overfit": round(st.median(
            [r["selection_overfit"] for r in rows]), 6),
    }
    Path(a.out).write_text(json.dumps({"aggregate": agg, "rows": rows},
                                      ensure_ascii=False, indent=1))
    print(f"\n  ★ 단일이 루프보다 나은 표 {agg['single_beats_loop']}/{agg['n']}"
          f"  · 학습 1위의 홀드아웃 순위 중앙 "
          f"{agg['median_rank_of_train_winner']:.1f}/10"
          f"  · 선택 과적합 중앙 {agg['median_selection_overfit']:+.4f}")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
