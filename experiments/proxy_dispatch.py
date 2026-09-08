"""8-1 the proxy dispatch — can the regime be known at deployment time? 0 LLM
calls.

    python3 experiments/proxy_dispatch.py

## Why it is measured

Whether it is the per-regime refit (§29.5 b) or a stratified report, it is
only useful if **the regime can be known at deployment time**. `t_best` is a
value that needs an exhaustive measurement, so a number cut by it is an
oracle, not an artefact.

    the proxy:  SOL = max(2MNK/effective peak, bytes/effective bandwidth)
                the shape + the hardware alone
    the truth:  t_best                    ★ needs an exhaustive measurement

The agreement rate of the two definitions is measured, and so is how robust
that agreement is (the boundary margin). **Even 100% agreement is a
coincidence of this table if the margin is thin**, so both have to be looked
at.

## The result (2026-08-21)

61/61 agreement. But the closest boundary margin is only 1.13x. See
`docs/glossary.md`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401  — it fills REGISTRY
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate, evaluate_scores, geomean
from kernelrule.core.splits import _DUMMY_CFG, Split
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.features.physical import log_sol_ms
from kernelrule.rules.human_guided import CODE as PS
from kernelrule.rules.human_guided import W0 as PS_W0

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
RUN_REAL = Path("runs/real-gpt-5.4-mini-2026-03-17/archive.jsonl")

#: The boundary between the fast and slow regimes (§30). Below it the tick
#: resolution dominates the ranking.
BOUNDARY_MS = 0.5


def _setup():
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    sol = {p: 2 ** log_sol_ms(p, table.hw, _DUMMY_CFG) for p in shapes}
    best = {p: float(table.times_of(p).min()) for p in shapes}
    return table, shapes, sol, best


def main() -> None:
    table, shapes, sol, best = _setup()
    matrix = FeatureMatrix(table, REGISTRY)
    proxy_fast = {p: sol[p] < BOUNDARY_MS for p in shapes}
    true_fast = {p: best[p] < BOUNDARY_MS for p in shapes}

    print("=" * 74)
    print("8-1. the regime judgement — the proxy vs the truth")
    print("=" * 74)
    print("  the proxy:  SOL < 0.5ms   the shape + hardware constants alone. "
          "Known at deployment time")
    print("  the truth:  t_best < 0.5ms            ★ needs an exhaustive "
          "measurement\n")

    tp = sum(1 for p in shapes if proxy_fast[p] and true_fast[p])
    fp = sum(1 for p in shapes if proxy_fast[p] and not true_fast[p])
    fn = sum(1 for p in shapes if not proxy_fast[p] and true_fast[p])
    tn = sum(1 for p in shapes if not proxy_fast[p] and not true_fast[p])
    n = len(shapes)
    print("                     true fast   true slow")
    print(f"  proxy fast         {tp:8d}   {fp:9d}")
    print(f"  proxy slow         {fn:8d}   {tn:9d}")
    print(f"\n  agreement {tp + tn}/{n} = {(tp + tn) / n:.1%}   "
          f"recall(fast) {tp / max(tp + fn, 1):.1%}   "
          f"disagreements {fp + fn}")
    for label, cond in (("the proxy says fast but it is really slow",
                         lambda p: proxy_fast[p] and not true_fast[p]),
                        ("the proxy says slow but it is really fast",
                         lambda p: not proxy_fast[p] and true_fast[p])):
        bad = [p for p in shapes if cond(p)]
        if bad:
            print(f"  ★ {label}:")
            for p in bad:
                print(f"      {p.M}x{p.N}x{p.K}  SOL {sol[p] * 1000:7.1f}us"
                      f"  t_best {best[p] * 1000:8.1f}us")

    # -- what actually needs the dispatch -----------------------------------
    with RUN_REAL.open() as fh:
        archive = [json.loads(ln) for ln in fh if ln.strip()]
    evolved = min(archive, key=lambda e: e["regret"])
    vendor = load_vendor(VENDOR)

    def refit(code, w0, sh):
        return fit_weights(compile_rule(code), matrix, table,
                           Split("train", tuple(sh)), w0, max_evals=300,
                          objective="regret")

    def score(code, w, sh):
        return evaluate_scores(make_score_of(compile_rule(code), matrix, w),
                               table, sh, ks=(1,))

    print(f"\n{'=' * 74}")
    print("which artefact needs the regime judgement")
    print("=" * 74)
    ev = score(evolved["code"], evolved["w"], shapes)
    print(f"  evolved (a single rule, no dispatch)     all61 {ev.at(1):.4f}")
    print("    -> it uses the same rule on every shape. The regime judgement "
          "is **not needed**.")
    print("       It branches on is_memory_bound, but that too is a roofline "
          "proxy.")

    for name, part in (("the proxy (SOL)", proxy_fast),
                       ("the truth (t_best) ★oracle", true_fast)):
        fast = [p for p in shapes if part[p]]
        slow = [p for p in shapes if not part[p]]
        e_f = score(PS, refit(PS, PS_W0, fast).w, fast)
        e_s = score(PS, refit(PS, PS_W0, slow).w, slow)
        i_f = {p: i for i, p in enumerate(e_f.shapes)}
        i_s = {p: i for i, p in enumerate(e_s.shapes)}
        per = np.array([e_f.regret[i_f[p], 0] if p in i_f
                        else e_s.regret[i_s[p], 0] for p in shapes])
        print(f"\n  human_guided refitted per regime — the boundary is "
              f"{name}")
        print(f"    fast {len(fast):2d} shapes {e_f.at(1):.4f} | "
              f"slow {len(slow):2d} shapes {e_s.at(1):.4f} | "
              f"all61 {geomean(per):.4f}")

    v = evaluate(vendor_order_fn(table, vendor, mapping="nearest"),
                 table, shapes, ks=(1,), label="vendor")
    print(f"\n  vendor all61 {v.at(1):.4f}   ★ the pass condition 1.080")


def margins() -> None:
    """★ Is the 100% agreement robust — it looks at the boundary margin.

    SOL is a lower bound, so `t_best` is always above it. The two judgements
    only differ in the narrow band where `SOL < 0.5 <= t_best`. **Whether
    there simply happened to be no shape in that band, or whether it is safe
    in principle**, has to be told apart.
    """
    _, shapes, sol, best = _setup()
    rows = sorted(((abs(math.log2(sol[p] / BOUNDARY_MS)), p)
                   for p in shapes), key=lambda r: r[0])
    print("the 8 shapes closest to the boundary (0.5ms) — a misjudgement "
          "happens here")
    print(f"  {'shape':22s} {'SOL(us)':>10} {'t_best(us)':>11} "
          f"{'margin(x)':>10}  judgement")
    for d, p in rows[:8]:
        agree = (sol[p] < BOUNDARY_MS) == (best[p] < BOUNDARY_MS)
        print(f"  {p.M}x{p.N}x{p.K:<10} {sol[p] * 1000:10.1f} "
              f"{best[p] * 1000:11.1f} {2 ** d:10.2f}  "
              f"{'agree' if agree else '★disagree'}")
    ratios = sorted(best[p] / sol[p] for p in shapes)
    close = sum(1 for d, _ in rows if 2 ** d < 2.0)
    print(f"\n  within 2x of the boundary: {close}/{len(rows)} shapes")
    print(f"  t_best / SOL median: {ratios[len(ratios) // 2]:.3f}")
    lo = BOUNDARY_MS / ratios[len(ratios) // 2]
    print(f"  ★ the danger band: SOL ∈ [{lo * 1000:.0f}, "
          f"{BOUNDARY_MS * 1000:.0f}] us"
          "  — a shape in that band is effectively a coin flip")


if __name__ == "__main__":
    main()
    print()
    margins()
