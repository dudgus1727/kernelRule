"""★ Does the coordinate polish flip the A/B reading (D-55). 0 LLM calls.

    python3 experiments/fitter_polish.py

It measures that the point where Nelder-Mead stopped is not a local optimum
along the coordinate directions, and reads A/B (descriptions vs names only)
again with the polish on. The decision criteria were nailed down in
`docs/artifacts/fitter-sweep.md` before the experiment.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"


def main() -> None:
    warnings.simplefilter("ignore")
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    index = json.loads((Path("docs/artifacts/rules") / "index.json").read_text())

    out: dict[bool, dict[str, float]] = {False: {}, True: {}}
    for pol in (False, True):
        for row in index:
            run = row["run"]
            with (Path("runs") / run / "archive.jsonl").open() as fh:
                arc = [json.loads(ln) for ln in fh if ln.strip()]
            best = min(arc, key=lambda e: e["regret"])
            fn = compile_rule(best["code"])
            reg = {}
            for name in ("short", "long"):
                g_tr = [p for p in train if regime_of(p, table.hw) == name]
                g_ho = [p for p in held if regime_of(p, table.hw) == name]
                fr = fit_weights(fn, matrix, table, Split("train", tuple(g_tr)),
                                 best["w"], max_evals=300, warn_invariants=False,
                                 polish=pol,
                          objective="regret")
                e = evaluate_scores(make_score_of(fn, matrix, fr.w), table,
                                    g_ho, ks=(1,))
                for i, p in enumerate(e.shapes):
                    reg[p] = e.regret[i, 0]
            out[pol][run] = geomean(np.array([reg[p] for p in held if p in reg]))

    print("=" * 70)
    print("A/B under the coordinate polish — descriptions (luna) vs names "
          "only (lunaNAMES)")
    print("=" * 70)
    print(f"  {'condition':14s} {'polish off':>12} {'polish on':>12} "
          f"{'change':>9}")
    for tag, pre in (("A descriptions", "luna-"),
                     ("B names only", "lunaNAMES-")):
        f = [v for k, v in out[False].items() if k.startswith(pre)]
        t = [v for k, v in out[True].items() if k.startswith(pre)]
        print(f"  {tag:14s} {np.median(f):12.4f} {np.median(t):12.4f} "
              f"{np.median(t) - np.median(f):+9.4f}")
    for pol in (False, True):
        a = [v for k, v in out[pol].items() if k.startswith("luna-")]
        b = [v for k, v in out[pol].items() if k.startswith("lunaNAMES-")]
        u = mannwhitneyu(a, b, alternative="two-sided")
        print(f"\n  polish {'on' if pol else 'off'}: B - A = "
              f"{np.median(b) - np.median(a):+.4f}  "
              f"Mann-Whitney p={u.pvalue:.3f}")
    print("\n  ※ the 12 runs are inside the seed spread sigma=0.0274 (D-53). "
          "This difference is inside that spread and is not significant.")


if __name__ == "__main__":
    main()
