"""★ The rank-loss evolution result — the metrics of the pre-registration
`rank-evo-prereg.md`. 0 LLM calls.

    python3 experiments/rank_evo_report.py

The main metric is **tau**. `regret` is only recorded (pre-registration §4).
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
TOP_N = 100
TAU_SAMPLE, TAU_SEED = 4000, 12345
#: The ones that moved a lot inside the top 100 in the ceiling measurement
#: (ranking-ceiling.md §3)
WATCH = ("split_k_cost", "sm_idle_cost", "pipeline_warmup_frac",
         "tail_waste", "waves")


def _splits(t: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = t.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    sh = [p for p in t.shapes() if aligned(p)]
    held = [p for p in sh if 11008 in (p.N, p.K)]
    return SplitSet(train=Split("train", tuple(p for p in sh if p not in held)),
                    val=Split("val", tuple(held)), kind="nk11008")


def _taus(code, w, table, matrix, shapes):
    fn = compile_rule(code)
    w = np.asarray(w, dtype=np.float64)
    rng = np.random.default_rng(TAU_SEED)
    tt, ta = [], []
    for p in shapes:
        cand = table.candidates(p)
        s = np.asarray(make_score_of(fn, matrix, w)(p, cand), dtype=np.float64)
        t = np.asarray(table.times_of(p))
        top = np.argsort(t, kind="stable")[:TOP_N]
        if len(np.unique(t[top])) > 1:
            tt.append(kendalltau(s[top], t[top], variant="b").statistic)
        idx = rng.choice(len(t), size=min(TAU_SAMPLE, len(t)), replace=False)
        ta.append(kendalltau(s[idx], t[idx], variant="b").statistic)
    return float(np.median(tt)), float(np.median(ta))


def _feats(code: str) -> set:
    return {n.attr for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "f"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=[f"x-rank-rankevo-s{i}" for i in range(3)])
    ap.add_argument("--out", default="docs/artifacts/rank-evo.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    shapes = list(sp.train.shapes)

    print("=" * 78)
    print("rank-loss evolution — the pre-registration metrics   ★ the main "
          "metric is tau, regret is only recorded")
    print("=" * 78)

    seed = json.loads(Path("runs/F3rw-p8/stage2-rule-writer"
                           "/chosen.json").read_text())
    st, sa = _taus(seed["code"], seed["w0"], T, M, shapes)
    print(f"  the seed (shared)    top-100 tau {st:6.3f}   all {sa:6.3f}\n")

    print(f"  {'run':10s} {'rank':>7} {'regret':>8} {'top-100 tau':>13} "
          f"{'all':>8} {'cells':>6} {'trm':>4} {'config kinds':>13}")
    rows = []
    for run in a.runs:
        d = Path("runs") / run
        arc = sorted((json.loads(x) for x in
                      (d / "archive.jsonl").read_text().splitlines()
                      if x.strip()), key=lambda e: e.get("rank_loss", 1e9))
        best = arc[0]
        t100, tall = _taus(best["code"], best["w"], T, M, shapes)
        rd = [json.loads(x) for x in
              (d / "rounds.jsonl").read_text().splitlines() if x.strip()]
        # config diversity
        fn = compile_rule(best["code"])
        w = np.asarray(best["w"], dtype=np.float64)
        picks = []
        for p in shapes:
            cand = T.candidates(p)
            s = make_score_of(fn, M, w)(p, cand)
            j = int(cand.top_k(s, 1)[0])
            picks.append((str(cand.kernel_id[j]), int(cand.split_k[j]),
                          str(cand.split_k_mode[j])))
        row = {"run": run, "rank_loss": best.get("rank_loss"),
               "regret": best["regret"], "tau_top100": t100, "tau_all": tall,
               "n_cells": rd[-1]["n_cells"], "n_terms": len(best["w"]),
               "n_config_kinds": len(Counter(picks)),
               "feats": sorted(_feats(best["code"]))}
        rows.append(row)
        print(f"  {run.split('-')[-1]:10s} "
              f"{best.get('rank_loss', float('nan')):7.4f} "
              f"{best['regret']:8.4f} {t100:13.3f} {tall:8.3f} "
              f"{rd[-1]['n_cells']:6d} {len(best['w']):4d} "
              f"{row['n_config_kinds']:13d}")

    t1 = np.array([r["tau_top100"] for r in rows])
    ta = np.array([r["tau_all"] for r in rows])
    print(f"\n  {'median':10s} {'':7s} "
          f"{np.median([r['regret'] for r in rows]):8.4f} "
          f"{np.median(t1):13.3f} {np.median(ta):8.3f}")
    print(f"  {'range':10s} {'':7s} {'':8s} "
          f"{t1.min():.3f}~{t1.max():.3f}  {ta.min():.3f}~{ta.max():.3f}")

    print("\n" + "=" * 78)
    print("the verdict — the line nailed down in the pre-registration")
    print("=" * 78)
    m1, ma = float(np.median(t1)), float(np.median(ta))
    print(f"  top-100 tau median {m1:.3f}  -> " + (
        "★ success (>=0.30)" if m1 >= 0.30
        else "failure (<=0.15)" if m1 <= 0.15
        else "indistinguishable (0.15~0.30)"))
    print(f"  all-range tau median  {ma:.3f}  -> " + (
        "kept (>=0.30)" if ma >= 0.30
        else "★ a trade-off (<=0.15)" if ma <= 0.15 else "in between"))
    print("  ⚠️ 3 seeds cannot give significance — it is read by range "
          "separation")

    print("\n  ★ are the five axes the ceiling measurement named used "
          "(the ones that moved a lot inside the top 100)")
    for name in WATCH:
        n = sum(1 for r in rows if name in r["feats"])
        print(f"    {name:22s} {n}/{len(rows)} runs")
    allf = Counter(x for r in rows for x in r["feats"])
    print(f"\n  the union of the axes used, {len(allf)}: "
          f"{', '.join(sorted(allf))}")
    Path(a.out).write_text(json.dumps(
        {"seed_tau": [st, sa], "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
