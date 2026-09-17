"""★ regret@k and hit@k per run, and the per-shape win/loss against the
baselines (D-171 §5-2). **0 LLM calls.**

    python3 -m experiments.nk4_atk

⚠️ **A different procedure from `canonical_score` when these numbers were
made, deliberately.** Canonical then refitted the weights on the training
split; this scores the rule **with the weights the loop left in the
archive**. ⛔ D-182 removed that refit, so the two have since converged on
this point — the numbers here were made before that. Two numbers from two
procedures are not put in one column — `campaign-nk4.json` carries the
canonical holdout and this file carries the @k family, and each says which
it is.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.f1_pipeline import _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.baselines.static_topk import StaticTopK
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate, evaluate_scores, geomean
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY
from kernelrule.features.loader import load_generated, run_registry

GPUS = ("a6000", "5090", "4090", "h100")
KS = (1, 3, 5, 10)
OUT = Path("docs/artifacts/nk4-at-k.json")
#: ★ D-174 — campaign 2 (`c2-*`, 4 seeds, `nkband`, seeds isolated).
CAMPAIGNS = {"nk4": ("nk4", (0, 1, 2), "nkgroup", True),
             "c2": ("c2", (0, 1, 2, 3), "nkband", False)}
PREFIX, SEEDS, DESIGN, LEAKED = CAMPAIGNS["nk4"]


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _registry(gpu, fold, seed, table):
    reg, _ = run_registry(f"{PREFIX}-{gpu}-f{fold}", table=table, seed=seed,
                          human=REGISTRY)
    # ⛔ Only the leaking campaign needs this (D-172 §X).
    for earlier in (range(seed) if LEAKED else ()):
        fp = Path(f"runs/{PREFIX}-{gpu}-f{fold}-s{earlier}/features.jsonl")
        if fp.exists():
            for f in load_generated(fp, table=table):
                if f.name not in reg._items:
                    reg.add(f)
    return reg


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", choices=tuple(CAMPAIGNS), default="nk4")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    global PREFIX, SEEDS, DESIGN, LEAKED  # noqa: PLW0603
    PREFIX, SEEDS, DESIGN, LEAKED = CAMPAIGNS[a.campaign]
    res: list[dict] = []
    per_shape: dict = {}
    print("=" * 96)
    print("★ regret@k · hit@k · per-shape win/loss (D-171 §5-2). 0 LLM calls")
    print("=" * 96)
    for gpu in GPUS:
        T = TABLES[gpu]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        vend = load_vendor(f"datasets/baselines/vendor-{gpu}-"
                           f"{T['env_hash'][:8]}.json")
        for fold in range(4):
            sp = _splits(table, fold=fold, k=4, design=DESIGN)
            hold = list(sp.val.shapes)
            ev_v = evaluate(vendor_order_fn(table, vend, mapping="nearest"),
                            table, hold, ks=KS, label="vendor")
            st1 = StaticTopK(table, hold, coverage="union").run(ks=KS)
            base_v = {p.key: float(ev_v.regret[i, 0])
                      for i, p in enumerate(ev_v.shapes)}
            for seed in SEEDS:
                d = Path(f"runs/{PREFIX}-{gpu}-f{fold}-s{seed}")
                if not (d / "archive.jsonl").exists():
                    continue
                best = min(_rows(d / "archive.jsonl"),
                           key=lambda e: e["regret"])
                reg = _registry(gpu, fold, seed, table)
                m = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
                fn = compile_rule(best["code"])
                ev = evaluate_scores(
                    make_score_of(fn, m, np.asarray(best["w"], float)),
                    table, hold, ks=KS, label=d.name)
                row = {"run": d.name, "gpu": gpu, "fold": fold, "seed": seed,
                       "n": len(hold),
                       "regret_at": {f"k{k}": round(ev.at(k), 6)
                                     for k in KS},
                       "hit_at": {f"k{k}": round(ev.hit_rate(k), 4)
                                  for k in (1, 3)},
                       "vendor_at1": round(float(geomean(ev_v.regret[:, 0])),
                                           6),
                       "static_top1_at1": round(float(st1.by_k[1]["all"]), 6)}
                wins_v = sum(1 for i, p in enumerate(ev.shapes)
                             if float(ev.regret[i, 0]) < base_v[p.key])
                row["shapes_better_than_vendor"] = f"{wins_v}/{len(hold)}"
                per_shape[d.name] = {
                    f"{p.M}x{p.N}x{p.K}": [round(float(ev.regret[i, 0]), 4),
                                           round(base_v[p.key], 4)]
                    for i, p in enumerate(ev.shapes)}
                res.append(row)
                print(f"  {d.name:22s} @1 {row['regret_at']['k1']:.4f}  "
                      f"@3 {row['regret_at']['k3']:.4f}  "
                      f"@10 {row['regret_at']['k10']:.4f}  "
                      f"hit@1 {row['hit_at']['k1']:.2f}  "
                      f"hit@3 {row['hit_at']['k3']:.2f}  "
                      f"vs vendor {row['shapes_better_than_vendor']}")
    agg = {"n_runs": len(res),
           "shapes_better_than_vendor_total": (
               f"{sum(int(r['shapes_better_than_vendor'].split('/')[0]) for r in res)}"
               f"/{sum(int(r['shapes_better_than_vendor'].split('/')[1]) for r in res)}"),
           "hit_at_1_median": float(np.median([r["hit_at"]["k1"]
                                               for r in res])),
           "hit_at_3_median": float(np.median([r["hit_at"]["k3"]
                                               for r in res])),
           "regret_at_median": {f"k{k}": float(np.median(
               [r["regret_at"][f"k{k}"] for r in res])) for k in KS}}
    Path(a.out).write_text(json.dumps(
        {"note": ("⚠️ scored with the archive's own weights — NOT the "
                  "refit `canonical_score` performed when these numbers "
                  "were made. Do not put the two in one column. ⛔ D-182 "
                  "removed that refit."),
         "ks": list(KS), "aggregate": agg, "runs": res,
         "per_shape_vs_vendor": per_shape}, ensure_ascii=False, indent=1))
    print(f"\n  {json.dumps(agg, ensure_ascii=False)}")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
