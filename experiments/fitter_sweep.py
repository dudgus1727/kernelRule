"""★ Why the fitter does not move half the time — sweeping the simplex step.
0 LLM calls.

    python3 experiments/fitter_sweep.py

The decision criteria were nailed down in `docs/artifacts/fitter-sweep.md`
**before the experiment**. Setting the criteria after seeing the results is
contamination (D-50).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core import weights as W
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
RULES = Path("docs/artifacts/rules")
#: (name, SIMPLEX_SCALE, SIMPLEX_ABS)
SWEEP = [("relative 0.6 (now)", 0.6, 0.0), ("relative 1.5", 1.5, 0.0),
         ("relative 4.0", 4.0, 0.0), ("absolute 1.0", 0.0, 1.0),
         ("absolute 5.0", 0.0, 5.0)]


def main() -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    index = json.loads((RULES / "index.json").read_text())

    print("=" * 78)
    print("sweeping the fitter's simplex step — the decision criteria are in "
          "fitter-sweep.md")
    print("=" * 78)
    print(f"  {len(index)} rules x 2 regimes = {len(index) * 2} runs\n")
    print(f"  {'setting':18s} {'moved':>10} {'struct HO median':>18} "
          f"{'gain':>8} {'worsened':>10} {'evals median':>14}")

    base_ho = None
    for label, scale, absst in SWEEP:
        W.SIMPLEX_SCALE, W.SIMPLEX_ABS = scale, absst
        moved = tot = 0
        evals = []
        per_run = []
        worse = 0
        for row in index:
            run = row["run"]
            d = Path("runs") / run
            with (d / "archive.jsonl").open() as fh:
                arc = [json.loads(ln) for ln in fh if ln.strip()]
            best = min(arc, key=lambda e: e["regret"])
            fn = compile_rule(best["code"])
            reg = {}
            for name in ("short", "long"):
                g_tr = [p for p in train if regime_of(p, table.hw) == name]
                g_ho = [p for p in held if regime_of(p, table.hw) == name]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fr = fit_weights(fn, matrix, table,
                                     Split("train", tuple(g_tr)), best["w"],
                                     max_evals=300, warn_invariants=False,
                          objective="regret")
                tot += 1
                moved += int(fr.moved)
                evals.append(fr.n_evals)
                e = evaluate_scores(make_score_of(fn, matrix, fr.w), table,
                                    g_ho, ks=(1,))
                for i, p in enumerate(e.shapes):
                    reg[p] = e.regret[i, 0]
            ho = geomean(np.array([reg[p] for p in held if p in reg]))
            per_run.append(ho)
            if base_ho is not None and ho > base_ho[len(per_run) - 1] + 1e-6:
                worse += 1
        med = float(np.median(per_run))
        if base_ho is None:
            base_ho, imp = per_run, 0.0
        else:
            imp = float(np.median(base_ho)) - med
        print(f"  {label:16s} {moved:4d}/{tot:<4d} ({moved / tot:3.0%}) "
              f"{med:12.4f} {imp:+8.4f} {worse:9d} {int(np.median(evals)):10d}")
    W.SIMPLEX_SCALE, W.SIMPLEX_ABS = 0.6, 0.0


if __name__ == "__main__":
    main()
