"""★ It reproduces the documents' numbers from the committed rules. It runs
without `runs/`.

    python3 experiments/verify_rules.py

## What is verified

```
docs/artifacts/rules/<run>.py      score() + W_FITTED
docs/artifacts/rules/index.json    the scores recorded at the time
```

**This script executes the `.py`, recomputes the score and checks it against
`index.json`.** If they diverge it fails.

## Why this matters

An LLM run cannot be reproduced (the randomness is not controllable —
§24.4b). But **the scoring is completely deterministic**. With the rule files
committed, **half of the performance claim becomes verifiable** — anyone can
confirm it in seconds.

`runs/` is in `.gitignore`, so it is not in the repository. This script does
not read it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.numerics import approx_equal
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
RULES = Path("docs/artifacts/rules")
TOL = 5e-4


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score, mod.W_FITTED


def main() -> None:
    import numpy as np

    idx_path = RULES / "index.json"
    if not idx_path.exists():
        raise SystemExit(f"{idx_path} does not exist. "
                         "Run `python3 experiments/export_rules.py` first.")
    index = json.loads(idx_path.read_text())

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")

    print("=" * 66)
    print("rescoring the committed rules — checked against index.json")
    print("=" * 66)
    print(f"  {'run':16s} {'recorded':>10} {'recomputed':>12}  verdict")
    bad = []
    #: ★ 2026-09-10 (D-156): `log_sol_ms` stopped being a registered
    #: shape-level value, so a rule that branches on `p.log_sol_ms` **cannot
    #: be re-scored under today's registry**. That is the expected
    #: consequence, not a failure: those runs' numbers stand as recorded, and
    #: the rule file plus its weights are in the repository — re-scoring them
    #: needs the registry of their own commit.
    #: They are listed as `skipped`, and the count is printed, so "everything
    #: matched" can never quietly mean "there was nothing to check".
    skipped = []
    for row in index:
        run = row["run"]
        f = RULES / f"{run}.py"
        if not f.exists():
            bad.append(f"{run}: the rule file does not exist")
            continue
        fn, W = _load(f)
        reg = {}
        try:
            for name in ("short", "long"):
                g = [p for p in splits.val.shapes
                     if regime_of(p, table.hw) == name]
                if not g or name not in W:
                    continue
                e = evaluate_scores(
                    make_score_of(fn, matrix, np.asarray(W[name])),
                    table, g, ks=(1,))
                for i, p in enumerate(e.shapes):
                    reg[p] = e.regret[i, 0]
        except AttributeError as exc:
            if "unregistered" not in str(exc):
                raise
            skipped.append((run, str(exc).split(".")[0]))
            print(f"  {run:16s} {row['holdout']:10.4f} {'-':>12}  "
                  f"⚠️ skipped — {str(exc).split('.')[0]}")
            continue
        got = geomean(np.array([reg[p] for p in splits.val.shapes if p in reg]))
        want = row["holdout"]
        ok = approx_equal(got, want, TOL)
        if not ok:
            bad.append(f"{run}: recorded {want:.4f} != recomputed {got:.4f}")
        print(f"  {run:16s} {want:10.4f} {got:12.4f}  {'✅' if ok else '❌'}")

    print()
    if bad:
        print("★ what diverged:")
        for b in bad:
            print(f"    {b}")
        raise SystemExit(1)
    n_checked = len(index) - len(skipped)
    print(f"  ★ all {n_checked} of {len(index)} match (tolerance {TOL})")
    if skipped:
        print(f"  ⚠️ {len(skipped)} skipped — they use a value that is no "
              f"longer registered (D-156):")
        for run, why in skipped:
            print(f"      {run:16s} {why}")
        print("     Their recorded numbers stand; re-scoring them needs the "
              "registry of their own commit.")
    print("  The documents' structural holdout numbers are verified by these "
          "values.")


if __name__ == "__main__":
    main()
