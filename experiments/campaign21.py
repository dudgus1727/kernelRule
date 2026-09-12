"""★ The 21-run campaign, aggregated (D-168). **0 LLM calls.**

    python3 experiments/campaign21.py

It reads what the runs left behind — `runs/_campaign/<tag>.json` (per-run
scores, written while the campaign ran) plus each run's `rounds.jsonl` /
`features.jsonl` / `archive.jsonl` / `hypotheses.jsonl` — and computes the
comparisons the campaign exists for.

```
§3-3  seed spread vs split spread   ★ the reason 12 A6000 runs were made
§3-4a the trajectory of rule size   "was removing the cap right"
§3-4b axis generation statistics    do independent runs build similar axes
§3-4c lineage                       exploit / explore / cross
§3-4d hypothesis statistics
§3-5  time and tokens
```

⚠️ It reports **per group**, never one geomean over everything (principle
27 / the standing rule): a single number over four tables and four splits
is not a quantity anyone can act on.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

CAMP = Path("runs/_campaign")
OUT = Path("docs/artifacts/campaign-21.json")
#: (table, split) -> the three run tags
GROUPS = {
    ("a6000", "fold0"): [f"c21-a6000-f0-s{k}" for k in range(3)],
    ("a6000", "fold1"): [f"c21-a6000-f1-s{k}" for k in range(3)],
    ("a6000", "fold2"): [f"c21-a6000-f2-s{k}" for k in range(3)],
    ("a6000", "nk11008"): [f"c21-a6000-nk-s{k}" for k in range(3)],
    ("5090", "nk11008"): [f"c21-5090-nk-s{k}" for k in range(3)],
    ("4090", "nk11008"): [f"c21-4090-nk-s{k}" for k in range(3)],
    ("h100", "nk11008"): [f"c21-h100-nk-s{k}" for k in range(3)],
}


def _rows(tag: str) -> list[dict]:
    p = Path(f"runs/{tag}-s0/rounds.jsonl")
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    per = {}
    for tags in GROUPS.values():
        for t in tags:
            f = CAMP / f"{t}.json"
            if not f.exists():
                raise SystemExit(f"{f} is missing — the campaign is not "
                                 f"fully scored yet.")
            per[t] = json.loads(f.read_text())

    out: dict = {"n_runs": len(per), "groups": {}}

    # -- §3-3 ------------------------------------------------------------
    print("=" * 74)
    print("§3-3  seed spread vs split spread   (canonical holdout)")
    print("=" * 74)
    print(f"  {'table':6s} {'split':8s} {'s0':>8s} {'s1':>8s} {'s2':>8s}"
          f" {'median':>8s} {'spread':>8s}")
    for (gpu, split), tags in GROUPS.items():
        h = [per[t]["holdout"] for t in tags]
        g = {"holdout": h, "median": st.median(h),
             "spread": max(h) - min(h),
             "train": [per[t]["train_best"] for t in tags],
             "in_sample": [per[t]["in_sample"] for t in tags]}
        out["groups"][f"{gpu}/{split}"] = g
        print(f"  {gpu:6s} {split:8s} " + " ".join(f"{x:8.4f}" for x in h)
              + f" {g['median']:8.4f} {g['spread']:8.4f}")

    folds = [f"a6000/{s}" for s in ("fold0", "fold1", "fold2")]
    seed_sp = [out["groups"][k]["spread"] for k in folds]
    med = [out["groups"][k]["median"] for k in folds]
    out["a6000_kfold"] = {
        "seed_spread_per_fold": seed_sp,
        "seed_spread_median": st.median(seed_sp),
        "split_spread": max(med) - min(med), "fold_medians": med}
    print(f"\n  ★ A6000 k-fold: seed spread per fold "
          f"{[round(x, 4) for x in seed_sp]} (median "
          f"{st.median(seed_sp):.4f})")
    print(f"     split spread across fold medians "
          f"{max(med) - min(med):.4f}"
          f"  -> split / seed = "
          f"{(max(med) - min(med)) / st.median(seed_sp):.1f}x")

    # -- §3-4a -----------------------------------------------------------
    print("\n" + "=" * 74)
    print("§3-4a  rule size — the trajectory, and size vs performance")
    print("=" * 74)
    sizes, holds = [], []
    traj = {}
    for t, d in per.items():
        traj[t] = {"archive_len_w": d["len_w_archive"],
                   "best_len_w": d["len_w_best"]}
        sizes.append(d["len_w_best"])
        holds.append(d["holdout"])
    n = len(sizes)
    mx, my = sum(sizes) / n, sum(holds) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(sizes, holds, strict=True))
    vx = sum((a - mx) ** 2 for a in sizes) ** 0.5
    vy = sum((b - my) ** 2 for b in holds) ** 0.5
    r = cov / (vx * vy) if vx and vy else float("nan")
    out["rule_size"] = {"best_len_w": sizes, "holdout": holds,
                        "pearson_r_size_vs_holdout": round(r, 4),
                        "min": min(sizes), "max": max(sizes),
                        "median": st.median(sizes), "trajectory": traj}
    print(f"  best rule size {min(sizes)}~{max(sizes)} (median "
          f"{st.median(sizes)})")
    print(f"  ★ correlation size vs holdout  r = {r:+.4f}   "
          f"(n={n}; ⚠️ not a causal statement)")

    dead = {t: _rows(t) for t in per}
    dsum = {t: sum(x["n_dead_terms"] for x in rr) for t, rr in dead.items()}
    dscored = {t: sum(x["n_scored"] for x in rr) for t, rr in dead.items()}
    per_rule = [dsum[t] / max(1, dscored[t]) for t in per]
    out["dead_terms"] = {"per_run_total": dsum,
                         "per_scored_rule": {t: round(dsum[t] / max(1, dscored[t]), 2)
                                             for t in per},
                         "median_per_rule": round(st.median(per_rule), 2)}
    print(f"  ★ dead terms per scored rule: median "
          f"{st.median(per_rule):.2f}  range "
          f"{min(per_rule):.2f}~{max(per_rule):.2f}  (D-167 §N)")

    # -- §3-4b -----------------------------------------------------------
    print("\n" + "=" * 74)
    print("§3-4b  axis generation")
    print("=" * 74)
    made = {t: per[t]["axes_accepted"] for t in per}
    req = {t: per[t]["axes_requested"] for t in per}
    used = {t: per[t]["axes_used_by_best"] for t in per}
    names = Counter(n for v in made.values() for n in v)
    repeated = {k: v for k, v in names.items() if v > 1}
    out["axes"] = {"requested": req, "accepted": {t: len(v) for t, v in made.items()},
                   "used_by_best": {t: len(v) for t, v in used.items()},
                   "names": dict(made),
                   "repeated_across_runs": repeated}
    print(f"  requested {sum(req.values())}  accepted "
          f"{sum(len(v) for v in made.values())}  used by the best rule "
          f"{sum(len(v) for v in used.values())}")
    print(f"  ★ the same axis name in more than one run: {len(repeated)}"
          + (f"  {sorted(repeated.items(), key=lambda kv: -kv[1])[:5]}"
             if repeated else ""))

    # -- §3-4c -----------------------------------------------------------
    print("\n" + "=" * 74)
    print("§3-4c  lineage of the final rule")
    print("=" * 74)
    kinds = Counter()
    for t in per:
        rows = [json.loads(x) for x in
                Path(f"runs/{t}-s0/archive.jsonl").read_text().splitlines()
                if x.strip()]
        best = min(rows, key=lambda e: e["regret"])
        ch = (best.get("changes") or "").lower()
        k = ("cross" if "cross" in ch else
             "seed" if "seed" in ch else "mutation")
        kinds[k] += 1
    out["lineage"] = dict(kinds)
    print(f"  {dict(kinds)}")

    # -- §3-4d / §3-5 ----------------------------------------------------
    hyp = defaultdict(int)
    for t in per:
        p = Path(f"runs/{t}-s0/hypotheses.jsonl")
        for x in p.read_text().splitlines():
            if not x.strip():
                continue
            h = json.loads(x)
            hyp["n"] += 1
            hyp["with_cases"] += bool(h.get("evidence_cases"))
            hyp["wants_feature"] += bool(h.get("needs_new_feature")
                                         or h.get("physical_requirement"))
    out["hypotheses"] = dict(hyp)
    tin = sum(per[t]["tokens_in"] for t in per)
    tout = sum(per[t]["tokens_out"] for t in per)
    mins = sum(per[t]["minutes"] for t in per)
    out["cost"] = {"tokens_in": tin, "tokens_out": tout,
                   "llm_calls": sum(per[t]["llm_calls"] for t in per),
                   "round_minutes_total": round(mins, 1)}
    print("\n" + "=" * 74)
    print("§3-4d / §3-5  hypotheses · cost")
    print("=" * 74)
    print(f"  hypotheses {hyp['n']}  with evidence cases "
          f"{hyp['with_cases']} ({hyp['with_cases'] / max(1, hyp['n']):.0%})"
          f"  asking for a new axis {hyp['wants_feature']} "
          f"({hyp['wants_feature'] / max(1, hyp['n']):.0%})")
    print(f"  ★ {sum(per[t]['llm_calls'] for t in per)} LLM calls · "
          f"tokens in {tin:,} out {tout:,} · round wall clock "
          f"{mins / 60:.1f}h summed over runs")

    out["per_run"] = per
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"\nrecorded: {OUT}")


if __name__ == "__main__":
    main()
