"""★ The 48-run (N,K)-fold campaign, aggregated (D-171 §5). **0 LLM calls.**

    python3 -m experiments.campaign_nk4            # per run + aggregates
    python3 -m experiments.campaign_nk4 --only a6000

It reads what the runs left behind — each run's `rounds.jsonl`,
`archive.jsonl`, `features.jsonl`, `hypotheses.jsonl`, `config.json`,
`trace.jsonl` — and computes §5-2 (per run), §5-3 (between runs) and §5-4
(what is lost if not recorded now).

⚠️ **Per group, never one geomean over everything** (principle 27). A single
number over four tables and four folds is not a quantity anyone can act on.

⛔ It does not compare against the 21-run campaign. The library, the shape
population (61 -> 65) and the split design all differ (D-171 §9).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.f1_pipeline import _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import run_registry

RUNS = Path("runs")
OUT = Path("docs/artifacts/campaign-nk4.json")
GPUS = ("a6000", "5090", "4090", "h100")
FOLDS = (0, 1, 2, 3)
SEEDS = (0, 1, 2)


def tag_of(gpu: str, fold: int, seed: int) -> str:
    return f"nk4-{gpu}-f{fold}-s{seed}"


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _best_by_train(arc: list[dict]) -> dict:
    """★ The best by **training** score. It does not look at the holdout
    (§10.2)."""
    return min(arc, key=lambda e: e["regret"])


def _branching(code: str, shape_names: set[str]) -> dict:
    """Which shape-level axes the rule branches on, and at what threshold.

    ★ D-171 §5-4(d). The library hands over 7 branchable axes (known7's 3 +
    k7-1's 4); how many a rule actually uses is what this campaign is for.
    """
    import re

    used = sorted({m.group(1) for m in re.finditer(r"\bp\.(\w+)", code)
                   if m.group(1) in shape_names})
    th = [[m.group(1), m.group(2), float(m.group(3))]
          for m in re.finditer(r"p\.(\w+)\s*([<>]=?)\s*([0-9.eE+-]+)", code)]
    return {"axes": used, "n_axes": len(used), "thresholds": th,
            "n_paths": code.count("np.where") + code.count("if ")}


def measure_run(gpu: str, fold: int, seed: int, table, shape_names) -> dict:
    tag = tag_of(gpu, fold, seed)
    d = RUNS / tag
    if not (d / "rounds.jsonl").exists():
        return {"run": tag, "missing": True}
    rounds = _rows(d / "rounds.jsonl")
    arc = _rows(d / "archive.jsonl")
    best = _best_by_train(arc)
    splits = _splits(table, fold=fold, k=4, design="nkgroup")
    # ★ The three seeds of one (table, fold) run **inside one process** and
    #   `stage3` hands them the same `matrix`, so an axis seed 0 built is
    #   still in the registry when seed 1 starts. Measured: seeds 1 and 2 of
    #   `nk4-a6000-f0` both use `pipeline_family_overhead`, which **seed 0**
    #   created in its round 0, and neither wrote it to its own
    #   `features.jsonl`.
    #   ⛔ So the registry of seed `s` is the library plus the axes of seeds
    #   0..s. Loading seed `s`'s own file alone raises `unregistered
    #   feature` — which is how this was found (D-165 §1's trap again).
    reg, origin = run_registry(f"nk4-{gpu}-f{fold}", table=table, seed=seed,
                               human=REGISTRY)
    from kernelrule.features.loader import load_generated

    for earlier in range(seed):
        fp = RUNS / tag_of(gpu, fold, earlier) / "features.jsonl"
        if fp.exists():
            for f in load_generated(fp, table=table):
                if f.name not in reg._items:
                    reg.add(f)
                    origin[f.name] = f"loop s{earlier} (inherited)"
    matrix = FeatureMatrix(table, reg)
    cs = canonical_score(best["code"], np.asarray(best["w"], float),
                         table=table, matrix=matrix, splits=splits)
    hyp = _rows(d / "hypotheses.jsonl") if (d / "hypotheses.jsonl").exists() \
        else []
    feats = _rows(d / "features.jsonl") if (d / "features.jsonl").exists() \
        else []
    tot = Counter()
    for r in rounds:
        for k in ("n_proposed", "n_scored", "n_accepted", "n_features_made",
                  "n_feature_requests", "n_rejected_schema",
                  "n_rejected_static", "n_rejected_sandbox",
                  "n_rejected_fit", "n_llm_error", "n_val_blowups"):
            tot[k] += r.get(k) or 0
    return {
        "run": tag, "gpu": gpu, "fold": fold, "seed": seed,
        "split_kind": splits.kind,
        # -- §5-2 -----------------------------------------------------------
        "holdout": round(cs.holdout, 6),
        "in_sample": round(cs.in_sample, 6),
        "by_regime": {k: round(v, 6) for k, v in cs.by_regime.items()},
        "n_holdout": cs.n_holdout,
        "train_best": best["regret"],
        "val_best": best.get("val_regret"),
        "len_w_best": len(best["w"]),
        "round_best": best.get("round"),
        # -- §5-4(a) rule size trajectory ------------------------------------
        "len_w_by_round": [len(e["w"]) for e in arc],
        "len_w_archive": sorted(len(e["w"]) for e in arc),
        "n_cells": len({tuple(e.get("cell") or []) for e in arc}),
        # -- §5-4(d) branching ------------------------------------------------
        "branching": _branching(best["code"], shape_names),
        # -- dead terms, split by reason (D-169 §2) ----------------------------
        "n_dead_by_weight": [r.get("n_dead_by_weight") for r in rounds],
        "n_dead_by_sens": [r.get("n_dead_by_sens") for r in rounds],
        "n_dead_terms": [r.get("n_dead_terms") for r in rounds],
        # -- §5-4(b) axes -----------------------------------------------------
        "axes_made_in_loop": [f.get("name") for f in feats],
        "axes_used_by_best": _branching(best["code"], shape_names)["axes"],
        # -- §5-4(c) lineage ---------------------------------------------------
        # ★ The archive records `parent_ids`, not a kind. The kind is in the
        #   round's `by_parent_kind` tally, so lineage is read there.
        "lineage_best_parents": best.get("parent_ids"),
        # ★ `by_parent_kind` is `{kind: {n, dup, scored}}` per round — the
        #   proposal counts, not a flat tally. `n` is what §5-4(c) wants.
        "by_parent_kind": dict(sum(
            (Counter({k: (v.get("n", 0) if isinstance(v, dict) else v)
                      for k, v in (r.get("by_parent_kind") or {}).items()})
             for r in rounds), Counter())),
        # -- §5-4(e) hypotheses ------------------------------------------------
        "n_hypotheses": len(hyp),
        "n_hyp_with_evidence": sum(1 for h in hyp if h.get("evidence_cases")),
        "n_hyp_new_feature": sum(1 for h in hyp
                                 if h.get("needs_new_feature")),
        # -- §5-5 ---------------------------------------------------------------
        "counts": dict(tot),
        "minutes": round(sum(r.get("seconds") or 0 for r in rounds) / 60, 1),
        # ★ `llm_calls` is `{role: n}` per round, not a number.
        "llm_calls": dict(sum(
            (Counter(r.get("llm_calls") or {}) for r in rounds), Counter())),
    }


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(GPUS))
    ap.add_argument("--out", default=str(OUT))
    # ★ `canonical_score` refits per regime at 300 evals on rules of up to 83
    #   weights — measured at ~2.5 min per run, so 48 runs in one process is
    #   two hours. The four tables are independent, so they are computed in
    #   four processes and merged here. **Same computation, same numbers** —
    #   the merge only concatenates rows and recomputes the aggregates.
    ap.add_argument("--merge", nargs="*", default=None,
                    help="merge per-table outputs into one artefact")
    a = ap.parse_args()

    if a.merge:
        rows = []
        for f in a.merge:
            rows.extend(json.loads(Path(f).read_text())["runs"])
        done = [r for r in rows if not r.get("missing")]
        Path(a.out).write_text(json.dumps(
            {"n_runs": len(done), "design": "D-171 — (N,K) group folds, k=4",
             "library": "k7-1 (known7 + 20 generated)",
             "note": ("⛔ not comparable with the 21-run campaign: library, "
                      "shape population (61 -> 65) and split design all "
                      "differ"),
             "runs": rows, "aggregates": _aggregate(done)},
            ensure_ascii=False, indent=1))
        print(f"merged {len(done)} runs -> {a.out}")
        return

    rows: list[dict] = []
    for gpu in a.only:
        T = TABLES[gpu]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        shape_names = {n for n in REGISTRY._items if REGISTRY[n].shape_level}
        # the run's own registry carries the generated shape axes too
        for fold in FOLDS:
            try:
                reg, _ = run_registry(f"nk4-{gpu}-f{fold}", table=table,
                                      seed=0, human=REGISTRY)
                shape_names |= {n for n in reg._items if reg[n].shape_level}
            except Exception:                            # noqa: BLE001, S112
                pass
            for seed in SEEDS:
                r = measure_run(gpu, fold, seed, table, shape_names)
                rows.append(r)
                if r.get("missing"):
                    print(f"  {r['run']:22s} ★ missing")
                    continue
                print(f"  {r['run']:22s} holdout {r['holdout']:.4f}  "
                      f"in-sample {r['in_sample']:.4f}  "
                      f"|w| {r['len_w_best']:3d}  cells {r['n_cells']:2d}  "
                      f"branch {r['branching']['n_axes']}  "
                      f"{r['minutes']:.0f}m")
    done = [r for r in rows if not r.get("missing")]
    out = {"n_runs": len(done), "design": "D-171 — (N,K) group folds, k=4",
           "library": "k7-1 (known7 + 20 generated)",
           "note": ("⛔ not comparable with the 21-run campaign: library, "
                    "shape population (61 -> 65) and split design all "
                    "differ"),
           "runs": rows, "aggregates": _aggregate(done)}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


def _aggregate(rows: list[dict]) -> dict:
    """§5-3 spreads and §5-4 statistics."""
    by_cell: dict = {}
    for r in rows:
        by_cell.setdefault((r["gpu"], r["fold"]), []).append(r["holdout"])
    seed_spread = {f"{g}-f{f}": round(max(v) - min(v), 4)
                   for (g, f), v in sorted(by_cell.items())}
    fold_spread = {}
    for g in GPUS:
        med = [st.median(by_cell[(g, f)]) for f in FOLDS
               if (g, f) in by_cell]
        if len(med) > 1:
            fold_spread[g] = round(max(med) - min(med), 4)
    seed_med = {g: round(st.median([v for (gg, _), vals in by_cell.items()
                                    if gg == g for v in [max(vals) - min(vals)]]), 4)
                for g in GPUS if any(gg == g for gg, _ in by_cell)}
    dead_w = [x for r in rows for x in (r["n_dead_by_weight"] or []) if x is not None]
    dead_s = [x for r in rows for x in (r["n_dead_by_sens"] or []) if x is not None]
    axis_names = Counter(n for r in rows for n in (r["axes_made_in_loop"] or []))
    branch_axes = Counter(n for r in rows for n in r["branching"]["axes"])
    sizes = [r["len_w_best"] for r in rows]
    hold = [r["holdout"] for r in rows]
    corr = (float(np.corrcoef(sizes, hold)[0, 1])
            if len(set(sizes)) > 1 else None)
    return {
        "seed_spread_per_cell": seed_spread,
        "seed_spread_median_per_gpu": seed_med,
        "fold_spread_per_gpu": fold_spread,
        "★ fold_over_seed": {
            g: (round(fold_spread[g] / seed_med[g], 2)
                if seed_med.get(g) else None)
            for g in fold_spread},
        "rule_size": {"min": min(sizes), "max": max(sizes),
                      "median": st.median(sizes),
                      "corr_size_vs_holdout": (round(corr, 4)
                                               if corr is not None else None),
                      "⚠️": "not a causal statement"},
        "dead_terms": {
            "by_weight_total": sum(dead_w), "by_sens_total": sum(dead_s),
            "by_weight_nonzero_rounds": sum(1 for x in dead_w if x),
            "n_rounds": len(dead_w),
            "★": ("if by_weight is 0 everywhere, no term died because the "
                  "fitter zeroed it — they died because they cannot change "
                  "the ranking (D-169 §2)")},
        "axes": {"made_total": sum(axis_names.values()),
                 "distinct": len(axis_names),
                 "repeated_across_runs": {k: v for k, v in
                                          axis_names.most_common()
                                          if v > 1}},
        "branching": {
            "axes_used": dict(branch_axes.most_common()),
            "n_distinct_used": len(branch_axes),
            "median_per_rule": st.median(
                [r["branching"]["n_axes"] for r in rows])},
        "lineage": dict(sum((Counter(r["by_parent_kind"]) for r in rows),
                            Counter())),
        "cost": {
            "llm_calls": dict(sum((Counter(r["llm_calls"]) for r in rows),
                                  Counter())),
            "minutes_summed": round(sum(r["minutes"] for r in rows), 1)},
    }


if __name__ == "__main__":
    main()
