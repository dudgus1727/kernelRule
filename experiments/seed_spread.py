"""★ It decomposes the seed spread by regime (D-40). 0 LLM calls.

    python3 experiments/seed_spread.py

## Why

Six seeds of the same condition give 1.0518~1.1496 (a spread of 0.098) on the
structural holdout. **Most of the differences compared so far fall inside
that spread** — A/B 0.016, the RuleWriter seed 0.022, and "on a par with the
vendor" was one seed.

Without knowing where the spread comes from, there is no knowing where to
run the experiment either.

    the slow regime, 20 shapes   static top-1 1.015   headroom 1.5%
                                 ← is this what makes the spread?
    the fast regime, 41 shapes   static top-1 1.163   headroom 16.3%

In a band with 1.5% of headroom nobody can win anyway, but the ranking of
those shapes can wobble per seed and drag the overall geomean around.

## The verdict

```
if the slow side's spread is large  -> the experiment moves to the fast
                                       regime
if they are similar                 -> another cause. Suspect the number of
                                       rounds or the temperature
```
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.scoring import evaluate
from kernelrule.core.splits import Split, SplitSet, experiment_shapes, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"

#: Six runs of the same condition (no seed rule + feature descriptions, 24
#: features).
#: ★ The `gpt-5.4` runs this script used to read **were deleted** (D-52 — the
#: artefacts of a model introduced without instruction). To use it again,
#: first make runs with the instructed model, as `experiments/seed_selection.py`
#: does, and replace the list below with those.
#: Silently skipping a run that does not exist **draws a conclusion without
#: knowing the sample shrank.**
def _require(runs: list[str]) -> list[str]:
    from pathlib import Path as _P
    if not runs:
        raise SystemExit(
            "the list of runs to compare is empty. The gpt-5.4 artefacts this "
            "script used to read were deleted (D-52).\n"
            "Make runs with the instructed model and fill in the list — "
            "running with an empty list draws a conclusion from a sample of "
            "0.")
    missing = [r for r in runs if not (_P("runs") / r / "archive.jsonl").exists()]
    if missing:
        raise SystemExit(
            "the runs this script used to read are gone (the gpt-5.4 "
            "artefacts were deleted — D-52):\n  " + "\n  ".join(missing)
            + "\nMake runs with the instructed model and change the list.")
    return runs


#: The list of runs to compare. ★ Replace it with runs of the instructed model.
SAME_CONDITION: list[str] = []
def main() -> None:
    _require(SAME_CONDITION)
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    shapes = experiment_shapes(table)
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")
    n_fast = sum(1 for p in held if regime_of(p, table.hw) == "short")

    print("=" * 74)
    print("the seed spread decomposed by regime — 6 runs of the same "
          "condition (no seed rule + descriptions)")
    print("=" * 74)
    print(f"  structural holdout {len(held)} shapes = fast {n_fast} / slow "
          f"{len(held) - n_fast}\n")
    print(f"  {'run':30s} {'all':>8} {'fast':>8} {'slow':>8}")

    rows = []
    for run in SAME_CONDITION:
        f = Path("runs") / run / "archive.jsonl"
        if not f.exists():
            print(f"  {run:30s} (missing)")
            continue
        with f.open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        best = min(arc, key=lambda e: e["regret"])
        r = canonical_score(best["code"], best["w"], table=table,
                            matrix=matrix, splits=splits)
        fast = r.by_regime.get("short", float("nan"))
        slow = r.by_regime.get("long", float("nan"))
        rows.append((r.holdout, fast, slow))
        print(f"  {run:30s} {r.holdout:8.4f} {fast:8.4f} {slow:8.4f}")

    if len(rows) < 2:
        return
    a = np.array(rows)
    print(f"\n  {'':30s} {'all':>8} {'fast':>8} {'slow':>8}")
    for label, fn in (("median", np.median), ("min", np.min),
                      ("max", np.max)):
        print(f"  {label:30s} " + " ".join(f"{fn(a[:, i]):8.4f}"
                                           for i in range(3)))
    spread = a.max(axis=0) - a.min(axis=0)
    print(f"  {'★ spread (max-min)':30s} "
          + " ".join(f"{x:8.4f}" for x in spread))
    print(f"  {'coefficient of variation (sd/mean)':34s} "
          + " ".join(f"{a[:, i].std() / a[:, i].mean():8.4f}"
                     for i in range(3)))

    v = load_vendor(VENDOR)
    for name, group in (("fast", [p for p in held
                                  if regime_of(p, table.hw) == "short"]),
                        ("slow", [p for p in held
                                  if regime_of(p, table.hw) == "long"])):
        e = evaluate(vendor_order_fn(table, v, mapping="nearest"),
                     table, group, ks=(1,))
        print(f"  {'vendor ' + name:30s} {e.at(1):8.4f}")

    print()
    if spread[2] > spread[1] * 1.5:
        print("  ★ the slow regime's spread is more than 1.5x the fast "
              "regime's — the hypothesis is confirmed.")
        print("     The experiment moves to the fast regime.")
    elif spread[1] > spread[2] * 1.5:
        print("  ★ the fast regime's spread is larger — the hypothesis is "
              "rejected. It is another cause.")
    else:
        print("  ★ the two regimes' spreads are similar — it is not a "
              "problem of the slow regime alone.")
        print("     The number of rounds or the temperature has to be "
              "suspected.")

    # -- is split_k_io_amplification new information or a better form? ------
    print(f"\n{'=' * 74}")
    print("is split_k_io_amplification new information or a better form")
    print("=" * 74)
    from kernelrule.features.loader import extended_registry, load_generated
    from kernelrule.features.validate import _pearson, _spearman

    # ★ A placeholder. The gpt-5.4 artefacts were deleted (D-52)
    gen = load_generated("runs/featwriter-F1-<model>/proposals.jsonl",
                         table=table, only={"split_k_io_amplification"})
    if not gen:
        print("  (the feature was not found)")
        return
    ext = extended_registry(REGISTRY, gen)
    mat = FeatureMatrix(table, ext)
    cols: dict[str, list] = {}
    for p in list(table.shapes())[:12]:
        fe, _info = mat.for_shape(p)
        for n in ("split_k_io_amplification", "split_k_cost",
                  "log_workspace_bytes"):
            cols.setdefault(n, []).append(np.asarray(getattr(fe, n), float))
    c = {n: np.concatenate(v) for n, v in cols.items()}
    base = c["split_k_io_amplification"]
    for n in ("split_k_cost", "log_workspace_bytes"):
        sp, pe = abs(_spearman(base, c[n])), abs(_pearson(base, c[n]))
        verdict = ("a better form (the same information)" if sp > 0.95
                   else "partial overlap" if sp > 0.7 else "new information")
        print(f"  vs {n:24s} sp {sp:.3f}  pe {pe:.3f}   {verdict}")


if __name__ == "__main__":
    main()
