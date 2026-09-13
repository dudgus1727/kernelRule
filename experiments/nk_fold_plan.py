"""★ The `(N, K)` group split — **the layout, and the four checks that must
pass before anything runs** (D-171 §1). 0 LLM calls · 0 GPU.

    python3 -m experiments.nk_fold_plan

```
1 coverage  every shape is in ★ exactly one fold's val
2 twins     every val shape has ★ zero (N,K) siblings in train
            ⚠️ M siblings remain — unavoidable on this grid, and it is why
               this split measures "a layer shape never seen", not "a shape
               never seen"
3 balance   the regime composition per fold, and `check_balance` on train
            ⚠️ A fold's own regime counts are a **design check, not a
               result**. One fold's memory side is 4~8 shapes; regime
               readings come from the folds pooled (D-171 §1-4)
4 tables    the same grouping on the 5090 / 4090 / H100, whose shape counts
            differ — the fold sizes are reported per table rather than
            assumed equal
```
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

from experiments.transfer_29_5 import TABLES
from kernelrule.core.splits import (
    MIN_REGIME_FRAC,
    NK_LAYER,
    experiment_shapes,
    nk_group_folds,
    nk_groups,
    regime_of,
)
from kernelrule.core.table import PerfTable

OUT = "docs/artifacts/nk-fold-plan.json"
OUT_MD = "docs/artifacts/nk-fold-plan.md"
K = 4


def _check(table, folds, shapes) -> dict:
    keys = [p.key for p in shapes]
    seen: Counter = Counter()
    for f in folds:
        for p in f.val.shapes:
            seen[p.key] += 1
    coverage_ok = set(seen) == set(keys) and set(seen.values()) == {1}

    twins = []
    m_twins = 0
    for i, f in enumerate(folds):
        tr_nk = {(p.N, p.K) for p in f.train.shapes}
        tr_m = {p.M for p in f.train.shapes}
        for p in f.val.shapes:
            if (p.N, p.K) in tr_nk:
                twins.append([i, f"{p.M}x{p.N}x{p.K}"])
            if p.M in tr_m:
                m_twins += 1

    rows = []
    for i, f in enumerate(folds):
        cv = Counter(regime_of(p, table.hw, axis="roofline")
                     for p in f.val.shapes)
        ct = Counter(regime_of(p, table.hw, axis="roofline")
                     for p in f.train.shapes)
        n_tr = len(f.train.shapes)
        rows.append({
            "fold": i, "kind": f.kind,
            "n_val": len(f.val.shapes), "n_train": n_tr,
            "groups": sorted([list(g) for g in
                              {(p.N, p.K) for p in f.val.shapes}]),
            "val_regimes": dict(cv), "train_regimes": dict(ct),
            "train_minority_frac": round(min(ct.values()) / n_tr, 4),
            "val": [[p.M, p.N, p.K] for p in f.val.shapes]})
    return {"coverage_ok": coverage_ok, "nk_twins_in_train": twins,
            "m_twins_in_train": m_twins,
            "min_train_minority": min(r["train_minority_frac"] for r in rows),
            "folds": rows}


def main() -> None:
    warnings.simplefilter("ignore")
    res: dict = {"k": K, "nk_layer": NK_LAYER,
                 "min_regime_frac": MIN_REGIME_FRAC,
                 "design": "D-171 §1 — a whole (N,K) group on one side",
                 "tables": {}}
    print("=" * 92)
    print(f"★ the (N,K) group split, k={K}. 0 LLM calls")
    print("=" * 92)
    for name in ("a6000", "5090", "4090", "h100"):
        T = TABLES[name]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        shapes = experiment_shapes(table)
        groups = nk_groups(shapes)
        folds = nk_group_folds(shapes, k=K)
        r = _check(table, folds, shapes)
        r["n_shapes"] = len(shapes)
        r["n_groups"] = len(groups)
        r["group_sizes"] = {f"{g[0]}x{g[1]}": len(v)
                            for g, v in sorted(groups.items(),
                                               key=lambda kv: -len(kv[1]))}
        # ★ Is a failing balance check the split's doing or the table's?
        #   `check_balance` warns below 25% (D-144). On three of the four
        #   tables the **whole population** is under that, so no split of any
        #   kind can satisfy it — the ridge point differs per GPU and fewer
        #   shapes fall memory-bound. Recorded here so the warning is read as
        #   a property of the table (D-171 §1-4).
        whole = Counter(regime_of(q, table.hw, axis="roofline")
                        for q in shapes)
        r["table_regimes"] = dict(whole)
        r["table_minority_frac"] = round(min(whole.values()) / len(shapes), 4)
        r["balance_ceiling_is_the_table"] = bool(
            r["table_minority_frac"] < MIN_REGIME_FRAC)
        res["tables"][name] = r
        print(f"\n  ── {name}  {len(shapes)} shapes · {len(groups)} (N,K) "
              f"groups")
        for row in r["folds"]:
            g = ", ".join(f"({a},{b})" for a, b in row["groups"][:3])
            more = "" if len(row["groups"]) <= 3 \
                else f" +{len(row['groups']) - 3}"
            print(f"     fold{row['fold']}  val {row['n_val']:2d}  "
                  f"train {row['n_train']:2d} "
                  f"(minority {row['train_minority_frac']:.0%})  "
                  f"{len(row['groups'])} groups: {g}{more}")
        print(f"     ★ 1 coverage      {r['coverage_ok']}")
        print(f"     ★ 2 (N,K) twins   {len(r['nk_twins_in_train'])}  "
              f"(⚠️ M twins {r['m_twins_in_train']} — unavoidable)")
        print(f"     ★ 3 min minority  {r['min_train_minority']:.0%} "
              f"(threshold {MIN_REGIME_FRAC:.0%})"
              + ("   ⚠️ the whole table is "
                 f"{r['table_minority_frac']:.0%} — no split can reach it"
                 if r["balance_ceiling_is_the_table"] else ""))
    Path(OUT).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    Path(OUT_MD).write_text(_md(res))
    print(f"\n  -> {OUT}\n  -> {OUT_MD}")


def _md(res: dict) -> str:
    L = ["# The (N,K) group split (D-171 §1)", "",
         "> **reproduce** `python3 -m experiments.nk_fold_plan` · 0 LLM calls",
         (f"> fold 0 is the layer holdout (`{res['nk_layer']}`, both "
          f"orientations) — the old `nk11008` split is one fold of this "
          f"design"), "",
         "| table | shapes | (N,K) groups | fold0 | fold1 | fold2 | fold3 |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for n, r in res["tables"].items():
        v = [str(f["n_val"]) for f in r["folds"]]
        L.append(f"| {n} | {r['n_shapes']} | {r['n_groups']} | "
                 + " | ".join(v) + " |")
    L += ["", "## The four checks", "",
          ("| table | 1 coverage | 2 (N,K) twins in train | 3 min train "
           "minority | ⚠️ M twins |"),
          "|---|---|--:|--:|--:|"]
    for n, r in res["tables"].items():
        L.append(f"| {n} | {'ok' if r['coverage_ok'] else '**FAIL**'} | "
                 f"{len(r['nk_twins_in_train'])} | "
                 f"{r['min_train_minority']:.0%} | {r['m_twins_in_train']} |")
    L += ["", "## ⚠️ The balance check and what caps it", "",
          ("| table | whole population memory | best train minority | can "
           "any split reach 25%? |"),
          "|---|--:|--:|---|"]
    for n, r in res["tables"].items():
        L.append(f"| {n} | {r['table_minority_frac']:.0%} | "
                 f"{r['min_train_minority']:.0%} | "
                 + ("**no — the table itself is below it**"
                    if r["balance_ceiling_is_the_table"] else "yes") + " |")
    L += ["", ("★ On the 5090 / 4090 / H100 the memory-bound share of the "
               "whole 63~65 shapes is 14~21%, under the 25% `check_balance` "
               "threshold (D-144). **No split of those tables can satisfy "
               "it** — their ridge points are lower, so fewer shapes fall "
               "memory-bound. The warning is a property of the table, not of "
               "this design, and it applied equally to every earlier run on "
               "those tables."),
          "", ("⚠️ **M twins are not a defect.** M=1024 appears in nearly "
               "every group, so a validation shape almost always has an M "
               "sibling in training. This split measures **\"a layer shape "
               "never seen\"**, not \"a shape never seen\"."),
          "", ("⚠️ A fold's own regime counts are in the JSON as a design "
               "check. They are **not** a result — one fold's memory side is "
               "4~8 shapes. Regime readings come from the folds pooled."), ""]
    for n, r in res["tables"].items():
        L += [f"### {n}", "", "```"]
        for f in r["folds"]:
            L.append(f"fold{f['fold']}  val {f['n_val']:2d}  "
                     f"train {f['n_train']:2d}  "
                     + " ".join(f"({a},{b})" for a, b in f["groups"]))
        L += ["```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
