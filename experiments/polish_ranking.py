"""★ Does the polish change the **ranking** of the rules — the ground for
whether to delete and re-run. 0 LLM calls.

    python3 experiments/polish_ranking.py

The criterion is the ranking, not the absolute value. If the scores move but
the relative order holds, the archive updates and the parent selection would
have been the same, and then the evolutionary trajectory is the same too.

```
the ranking holds     -> no re-run needed
the ranking flips     -> the trajectory differed. Delete + re-run
```

**Whether each run's "best rule" changes** is the more direct judgement — the
evolution picks parents by the best of an archive cell, not by the whole
ranking.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, experiment_shapes, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
#: This many archive candidates per run are looked at. It is what the
#: evolution actually ranked.
N_CAND = 12


def _fit_regret(fn, matrix, table, train, w0, *, polish: bool) -> float:
    """It fits on both regimes and combines the training regret (the same
    procedure as the scoring)."""
    tot, n = 0.0, 0
    for name in ("short", "long"):
        g = [p for p in train if regime_of(p, table.hw) == name]
        fr = fit_weights(fn, matrix, table, Split("train", tuple(g)), w0,
                         max_evals=300, warn_invariants=False, polish=polish,
                          objective="regret")
        tot += np.log(fr.fit_regret) * len(g)
        n += len(g)
    return float(np.exp(tot / n))


def main() -> None:
    warnings.simplefilter("ignore")
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    shapes = experiment_shapes(table)
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    index = json.loads(Path("docs/artifacts/rules/index.json").read_text())

    print("=" * 76)
    print("the ranking before and after the polish — would the evolution have "
          "made the same choice")
    print("=" * 76)
    print(f"  the top {N_CAND} of each run's archive are ranked\n")
    print(f"  {'run':16s} {'cand':>5} {'Kendall tau':>12} {'p':>7} "
          f"{'best rule':>11} {'top-3 set':>11}")

    taus, same_best, same_top3, n_run = [], 0, 0, 0
    for row in index:
        run = row["run"]
        with (Path("runs") / run / "archive.jsonl").open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        cand = sorted(arc, key=lambda e: e["regret"])[:N_CAND]
        if len(cand) < 3:
            print(f"  {run:16s} {len(cand):5d}  too few candidates — skipped")
            continue
        off, on = [], []
        for e in cand:
            fn = compile_rule(e["code"])
            off.append(_fit_regret(fn, matrix, table, train, e["w"],
                                   polish=False))
            on.append(_fit_regret(fn, matrix, table, train, e["w"],
                                  polish=True))
        t = kendalltau(off, on)
        b = int(np.argmin(off)) == int(np.argmin(on))
        t3 = (set(np.argsort(off)[:3].tolist())
              == set(np.argsort(on)[:3].tolist()))
        taus.append(t.statistic)
        same_best += b
        same_top3 += t3
        n_run += 1
        print(f"  {run:16s} {len(cand):5d} {t.statistic:12.3f} "
              f"{t.pvalue:7.3f} {'kept' if b else '★changed':>11} "
              f"{'kept' if t3 else '★changed':>11}")

    print()
    print(f"  Kendall tau median {np.median(taus):.3f}  "
          f"min {min(taus):.3f}  max {max(taus):.3f}")
    print(f"  the best rule kept  {same_best}/{n_run}")
    print(f"  the top-3 set kept  {same_top3}/{n_run}")
    print()
    if same_best == n_run and np.median(taus) > 0.9:
        print("  the verdict: the ranking holds — the evolution would have "
              "made the same choice")
    else:
        print("  ★ the verdict: the ranking changes — the trajectory "
              "differed. Delete + re-run")


if __name__ == "__main__":
    main()
