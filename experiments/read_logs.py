"""★ Counting three things from the old logs — duplicates / refusals / dead
terms. 0 LLM calls (D-135 §2).

    python3 experiments/read_logs.py

Even for runs with no trace, it counts only what `llm_calls/` +
`rounds.jsonl` + `archive.jsonl` can count. **What cannot be counted is
written down as such** — that is where a trace is needed.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

from sigma_5090 import _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
_BIG = 1e9   # ★ the "unbound" side of `bounds`. `_proj` only clips, so it is
             # generous
RUNS = [f"F3rw-p8-s{i}" for i in range(6)]


def _rows(run: str, name: str) -> list[dict]:
    p = Path("runs") / run / name
    return ([json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            if p.exists() else [])


def _proposals(run: str) -> list[dict]:
    """The RuleEditor responses from `llm_calls`, in order."""
    out = []
    for f in sorted((Path("runs") / run / "llm_calls").glob(
            "*-rule_editor.json")):
        j = json.loads(f.read_text())
        r = j.get("response")
        if isinstance(r, dict) and r.get("code"):
            out.append({"seq": j.get("seq"), "code": r["code"],
                        "changes": r.get("changes", "")})
    return out


def dup_report() -> None:
    print("=" * 92)
    print("1. duplicates — it proposes code that already exists")
    print("=" * 92)
    print(f"  {'run':16s} {'prop':>5s} {'dup':>5s} {'rate':>7s}   "
          f"duplicates per parent kind")
    tot = Counter()
    for run in RUNS:
        rs = _rows(run, "rounds.jsonl")
        by: Counter = Counter()
        n = d = 0
        for x in rs:
            for k, v in (x.get("by_parent_kind") or {}).items():
                by[k] += v.get("dup", 0)
                n += v.get("n", 0)
                d += v.get("dup", 0)
        tot.update(by)
        print(f"  {run:16s} {n:5d} {d:5d} {d / max(1, n):7.1%}   {dict(by)}")
    print(f"\n  ★ total duplicates per parent kind: {dict(tot)}")
    # How many times the same code repeats — counted directly from the
    # responses
    print(f"\n  {'run':16s} "
          f"{'times the same code appeared 2+ times':>38s}  most repeated")
    for run in RUNS:
        c = Counter(p["code"].strip() for p in _proposals(run))
        rep = [v for v in c.values() if v > 1]
        print(f"  {run:16s} {sum(rep) - len(rep):38d}  "
              f"{max(c.values())} times")
    print("\n  ⚠️ **which hypothesis was assigned when a duplicate appeared "
          "cannot be counted from the old logs** —")
    print("     the per-proposal hypothesis assignment is kept only for the "
          "accepted rules. That is where the trace goes")


def reject_report() -> None:
    print("\n" + "=" * 92)
    print("2. refusals — the reasons and the distribution")
    print("=" * 92)
    keys = ("n_rejected_schema", "n_rejected_static", "n_rejected_sandbox",
            "n_rejected_fit", "n_llm_error")
    print(f"  {'run':16s} " + " ".join(f"{k[2:]:>10s}" for k in keys)
          + "   the reasons (rejections)")
    why: Counter = Counter()
    per_round: Counter = Counter()
    for run in RUNS:
        rs = _rows(run, "rounds.jsonl")
        tot = {k: sum(x.get(k, 0) for x in rs) for k in keys}
        for x in rs:
            for kind, _detail in (x.get("rejections") or []):
                why[kind] += 1
                per_round[x["round"]] += 1
        det = Counter(k for x in rs for k, _ in (x.get("rejections") or []))
        print(f"  {run:16s} " + " ".join(f"{tot[k]:10d}" for k in keys)
              + f"   {dict(det)}")
    print(f"\n  ★ totals per reason: {dict(why)}")
    print("  ★ refusals per round: "
          + " ".join(f"r{r}:{n}" for r, n in sorted(per_round.items())))
    if not per_round:
        print("     (there were no refusals)")


def dead_terms_report() -> None:
    print("\n" + "=" * 92)
    print("3. dead terms — the slots that can be removed without the training "
          "getting worse")
    print("=" * 92)
    warnings.simplefilter("ignore")
    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M, sp = FeatureMatrix(T, REGISTRY), _splits(T)
    train = list(sp.train.shapes)
    print(f"  {'run':16s} {'trm':>3s} {'dead':>5s}  the loss per slot (the "
          f"rise in training regret when it is pinned to 0)")
    for run in RUNS:
        arc = _rows(run, "archive.jsonl")
        if not arc:
            continue
        best = sorted(arc, key=lambda e: e["regret"])[0]
        fn = compile_rule(best["code"])
        n = len(best["w"])
        base = []
        for nm in ("short", "long"):
            g = [q for q in train if regime_of(q, T.hw) == nm]
            base.append(fit_weights(fn, M, T, Split("train", tuple(g)),
                                    best["w"], max_evals=200,
                                    objective="regret",
                                    warn_invariants=False))
        dead, cost = [], {}
        for i in range(n):
            w = list(best["w"])
            w[i] = 0.0
            # ★ That slot is **pinned to 0** (`bounds`). Without pinning, the
            #   fitter comes back out of 0, so what gets measured is not "a
            #   dead term" but "does it find the same point when the starting
            #   point is shaken" — the D-136 correction.
            bnd = [(-_BIG, _BIG)] * n
            bnd[i] = (0.0, 0.0)
            ok, worst = True, 0.0
            for j, nm in enumerate(("short", "long")):
                g = [q for q in train if regime_of(q, T.hw) == nm]
                fr = fit_weights(fn, M, T, Split("train", tuple(g)), w,
                                 max_evals=200, objective="regret",
                                 warn_invariants=False, bounds=bnd)
                d = fr.fit_regret - base[j].fit_regret
                worst = max(worst, d)
                if d > 1e-6:
                    ok = False
            cost[i] = worst
            if ok:
                dead.append(i)
        # ⚠️ "when did it come in" is not counted — the seed rule already uses
        #    all 8 slots, so in the code of `bests.jsonl` the first appearance
        #    is r0 for everything and it says nothing (D-136). Instead **the
        #    loss when each slot is pinned** is written down.
        print(f"  {run:16s} {n:3d} {len(dead):5d}  " + " ".join(
            f"{'★' if i in dead else ' '}w{i}:{cost[i]:+.4f}"
            for i in range(n)))
    print("\n  ★ marks a dead slot (pinning it to 0 makes neither regime "
          "worse). The number is how much the training regret worsens when "
          "it is pinned")
    print("  ⚠️ 'was it nearly deleted after it came in' cannot be counted "
          "from the old logs — the")
    print("     proposal's changes and its parent are not kept per proposal. "
          "That is where the trace goes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dead", action="store_true")
    a = ap.parse_args()
    dup_report()
    reject_report()
    if not a.skip_dead:
        dead_terms_report()


if __name__ == "__main__":
    main()
