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
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import run_registry

RUNS = Path("runs")
OUT = Path("docs/artifacts/campaign-nk4.json")
GPUS = ("a6000", "5090", "4090", "h100")
FOLDS = (0, 1, 2, 3)
#: ★ D-174 — campaign 2 is `c2-*` with **4** seeds and the `nkband` design.
#: The 48-run campaign was `nk4-*` with 3 seeds and `nkgroup`; both stay
#: readable so the older artefact can be regenerated.
CAMPAIGNS = {
    # ★ `leaked_seeds` — the 48-run campaign ran before D-172 §X, so a
    #   later seed's rules use axes an earlier seed built and the registry
    #   has to be rebuilt that way to re-score them at all.
    "nk4": {"prefix": "nk4", "seeds": (0, 1, 2), "design": "nkgroup",
            "leaked_seeds": True},
    "c2": {"prefix": "c2", "seeds": (0, 1, 2, 3), "design": "nkband",
           "leaked_seeds": False},
}
SEEDS = (0, 1, 2)
PREFIX = "nk4"
DESIGN = "nkgroup"
LEAKED_SEEDS = True


def tag_of(gpu: str, fold: int, seed: int) -> str:
    return f"{PREFIX}-{gpu}-f{fold}-s{seed}"


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _best_by_train(arc: list[dict], run: Path | None = None) -> dict:
    """★ The rule the run **ended with** — by training score, ⛔ never the
    holdout (§10.2).

    ⛔ 2026-09-18 (D-183): this was `min(arc, key=regret)`. On an **exact
    tie** that picks whichever line comes first in `archive.jsonl`, which is
    not the rule the archive kept: `Archive.best` only moves on
    `_key(e) < _key(best) - tol`, so the **first** rule to become best stays.

    ```
    실측  c2-h100-f2-s0 — 세 규칙이 학습 1.053852132000 으로 완전 동점
          archive.best ★ r0044 (val 1.064955)  ·  min() ★ r0046 (val 1.065637)
          -> 집계가 ★ 내보낸 규칙과 다른 규칙의 점수를 싣고 있었다
    ```

    ★ So the archive's own answer is read from `bests.jsonl` when it is
    there, and `min` is only the fallback for a run without it.
    """
    if run is not None:
        bp = run / "bests.jsonl"
        if bp.exists():
            b = _rows(bp)
            if b:
                return b[-1]
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
    # ★ what the loop itself wrote — `archive.best.val_regret` at the last
    #   round, i.e. the answer this run ended with, scored on this split.
    recorded_holdout = float(rounds[-1]["best_val_regret"])
    arc = _rows(d / "archive.jsonl")
    best = _best_by_train(arc, d)
    # ⛔ 2026-09-17 — 여기에 "nkgroup" 이 **상수로** 박혀 있었다. c2 는
    #   `nkband` 로 돌았는데 집계는 nkgroup 분할로 채점했고, fold0·fold1 은
    #   홀드아웃이 **100% 학습 형상**이었다 (pending_fixes 22).
    splits = _splits(table, fold=fold, k=4, design=DESIGN)
    # ★ The three seeds of one (table, fold) run **inside one process** and
    #   `stage3` hands them the same `matrix`, so an axis seed 0 built is
    #   still in the registry when seed 1 starts. Measured: seeds 1 and 2 of
    #   `nk4-a6000-f0` both use `pipeline_family_overhead`, which **seed 0**
    #   created in its round 0, and neither wrote it to its own
    #   `features.jsonl`.
    #   ⛔ So the registry of seed `s` is the library plus the axes of seeds
    #   0..s. Loading seed `s`'s own file alone raises `unregistered
    #   feature` — which is how this was found (D-165 §1's trap again).
    reg, origin = run_registry(f"{PREFIX}-{gpu}-f{fold}", table=table,
                               seed=seed, human=REGISTRY)
    from kernelrule.features.loader import load_generated

    # ⛔ Only the leaking campaign needs this. ★ Since D-172 §X each seed
    #   rebuilds the stage-1 registry, so a later seed's rules reference
    #   **only** its own axes — adding earlier seeds' axes here would put
    #   columns in the matrix that no rule reads.
    if LEAKED_SEEDS:
        for earlier in range(seed):
            fp = RUNS / tag_of(gpu, fold, earlier) / "features.jsonl"
            if fp.exists():
                for f in load_generated(fp, table=table):
                    if f.name not in reg._items:
                        reg.add(f)
                        origin[f.name] = f"loop s{earlier} (inherited)"
    # ★ D-171 §U — this script builds one matrix per run (48 of them),
    #   so the cache is what it is for. Key = bundle + registry lock.
    matrix = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
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
        # ★ 2026-09-18 (D-183): the holdout is **read** from the run's own
        #   `rounds.jsonl`, not recomputed. The loop already scored its
        #   answer on this split, and reading removes the whole class of bug
        #   this campaign was re-aggregated for — an aggregator picking the
        #   wrong split or registry (pending_fixes 22).
        #   ⚠️ `cs.holdout` is kept beside it as a **check**: measured across
        #   all 64 runs, they agree to 4.9e-07 (the artefact rounds to 6dp).
        "holdout": round(recorded_holdout, 6),
        "holdout_recomputed": round(cs.holdout, 6),
        "holdout_check_diff": round(abs(recorded_holdout - cs.holdout), 9),
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
    ap.add_argument("--campaign", choices=tuple(CAMPAIGNS), default="nk4")
    ap.add_argument("--only", nargs="*", default=list(GPUS))
    ap.add_argument("--out", default=str(OUT))
    # ★ When these numbers were made `canonical_score` refitted at 300
    #   evals (⛔ D-182 removed that) on rules of up to 83
    #   weights — measured at ~2.5 min per run, so 48 runs in one process is
    #   two hours. The four tables are independent, so they are computed in
    #   four processes and merged here. **Same computation, same numbers** —
    #   the merge only concatenates rows and recomputes the aggregates.
    ap.add_argument("--merge", nargs="*", default=None,
                    help="merge per-table outputs into one artefact")
    a = ap.parse_args()
    global SEEDS, PREFIX, DESIGN, LEAKED_SEEDS  # noqa: PLW0603
    c = CAMPAIGNS[a.campaign]
    SEEDS, PREFIX, DESIGN = c["seeds"], c["prefix"], c["design"]
    LEAKED_SEEDS = c["leaked_seeds"]

    if a.merge:
        rows = []
        for f in a.merge:
            rows.extend(json.loads(Path(f).read_text())["runs"])
        done = [r for r in rows if not r.get("missing")]
        Path(a.out).write_text(json.dumps(
            {"n_runs": len(done), "campaign": a.campaign, "design": DESIGN,
             "library": "k7-1 (known7 + 20 generated)",
             "seed_isolation": (not LEAKED_SEEDS),
             "note": ("⛔ conditions differ between campaigns — library, "
                      "shape population, split design, seed prompt and seed "
                      "isolation. Numbers are not put side by side."),
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
                reg, _ = run_registry(f"{PREFIX}-{gpu}-f{fold}", table=table,
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
