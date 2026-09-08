"""8-2 how many regimes to split into. 0 LLM calls.

    python3 experiments/regime_count.py

## What is measured

Adding regimes shrinks the shapes per regime and runs into the §10.1 lower
bound (15~20). What is measured is **whether the gain from adding a boundary
is worth that price**.

★ **Measuring in-sample gives no answer.** Adding regimes multiplies the free
weights by k, so the regret on the fitted shapes **must** improve. That is
not evidence that the regime is real, only that there are more parameters.

So it is measured on a holdout:

    sorted by SOL, 1 in every 3 goes to the holdout (it spreads evenly across
    the regimes)
    both the boundaries and the weights are set **on the training shapes only**
    a holdout shape is scored with the weights of the regime its own SOL falls
    into

The structure is fixed, so **only the worth of the boundary** remains. If the
gain is near the noise floor, "adding more does not help" is the result, and
that itself is evidence about the number of regimes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
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

#: §10.1 — below this many per regime, that regime's fit cannot be trusted.
MIN_PER_REGIME = 15


def main() -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    sol = {p: log_sol_ms(p, table.hw, _DUMMY_CFG) for p in shapes}
    ordered = sorted(shapes, key=lambda p: sol[p])

    # ★ 1 in every 3 in SOL order goes to the holdout. It spreads evenly
    #   across the regimes, so every regime has holdout shapes at any k.
    holdout = [p for i, p in enumerate(ordered) if i % 3 == 2]
    train = [p for i, p in enumerate(ordered) if i % 3 != 2]

    def run(k: int, code: str, w0) -> tuple[float, float, list[int]]:
        """k equal parts. **Both the boundaries and the weights** are set on
        the training shapes only."""
        m = len(train)
        bounds = [round(i * m / k) for i in range(k + 1)]
        groups = [train[a:b]
                  for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        # The regime boundary = an SOL cut based on the training shapes
        cuts = [sol[groups[i][0]] for i in range(1, k)]

        def regime_index(p) -> int:
            return sum(1 for c in cuts if sol[p] >= c)

        per_tr: dict = {}
        per_ho: dict = {}
        for gi, g in enumerate(groups):
            fit = fit_weights(compile_rule(code), matrix, table,
                              Split("train", tuple(g)), w0, max_evals=300,
                          objective="regret")
            so = make_score_of(compile_rule(code), matrix, fit.w)
            ev = evaluate_scores(so, table, g, ks=(1,))
            for i, p in enumerate(ev.shapes):
                per_tr[p] = ev.regret[i, 0]
            mine = [p for p in holdout if regime_index(p) == gi]
            if mine:
                eh = evaluate_scores(so, table, mine, ks=(1,))
                for i, p in enumerate(eh.shapes):
                    per_ho[p] = eh.regret[i, 0]
        return (geomean(np.array([per_tr[p] for p in train])),
                geomean(np.array([per_ho[p] for p in holdout])),
                [len(g) for g in groups])

    print("=" * 72)
    print("8-2. the number of regimes — is there a gain from adding a boundary")
    print("=" * 72)
    print("  the structure is fixed, only the weights are refitted per regime "
          "— only the worth of the boundary remains")
    with RUN_REAL.open() as fh:
        archive = [json.loads(ln) for ln in fh if ln.strip()]
    evolved = min(archive, key=lambda e: e["regret"])

    for label, code, w0 in (("human_guided", PS, PS_W0),
                            ("evolved", evolved["code"], evolved["w"])):
        print(f"\n  [{label}]")
        print(f"  {'split':>5} {'train41':>9} {'★holdout20':>12}  "
              f"{'training per regime':<22}")
        for k in (1, 2, 3, 4, 5):
            tr, ho, sizes = run(k, code, w0)
            warn = ("  ★ below the §10.1 lower bound"
                    if min(sizes) < MIN_PER_REGIME else "")
            print(f"  {k:5d} {tr:9.4f} {ho:12.4f}  {str(sizes):<22}{warn}")

    v = evaluate(vendor_order_fn(table, load_vendor(VENDOR), mapping="nearest"),
                 table, shapes, ks=(1,), label="vendor")
    print(f"\n  vendor {v.at(1):.4f}   ★ the pass condition 1.080")
    print("\n  ※ the training score must improve as k grows (the free weights "
          "scale with k).\n     ★ the judgement is made on the holdout column "
          "alone.")
    print("  ※ the equal-part boundaries differ from 0.5ms — what is measured "
          "is 'is there a gain\n     from adding a boundary', not whether "
          "0.5ms is optimal.")


if __name__ == "__main__":
    main()
