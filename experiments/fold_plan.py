"""★ It **writes the 3-fold layout to a file** (D-144). 0 LLM calls · 0 GPU.

    python3 experiments/fold_plan.py

It records the train/val list per fold and the memory/compute counts. A human
has to be able to look at the layout before the run and see that it is what
was intended.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

from kernelrule.core.splits import regime_of, stratified_kfold
from kernelrule.core.table import PerfTable

BUNDLE = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
#: ★ The randomness that builds the folds. **It is kept separate from the
#: evolution seed** (D-144).
SPLIT_SEED = 12345


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--out", default="docs/artifacts/fold-plan.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(BUNDLE[0], env_hash=BUNDLE[1], ok_only=False)

    def aligned(p) -> bool:
        d = T.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in T.shapes() if aligned(p)]
    tot = Counter(regime_of(p, T.hw, axis="roofline") for p in shapes)
    print("=" * 84)
    print(f"stratified {a.k}-fold — split seed {a.split_seed} (separate from "
          f"the evolution seed)")
    print("=" * 84)
    print(f"  aligned-8 shapes {len(shapes)}   {dict(tot)}")
    folds = stratified_kfold(shapes, T.hw, k=a.k, seed=a.split_seed)
    out: dict = {"bundle": BUNDLE[0], "k": a.k, "split_seed": a.split_seed,
                 "n_shapes": len(shapes), "totals": dict(tot), "folds": []}
    print(f"\n  {'fold':6s} {'train':>6s} {'mem/comp':>10s}   "
          f"{'val':>4s} {'mem/comp':>10s}")
    for i, sp in enumerate(folds):
        ct = Counter(regime_of(p, T.hw, axis="roofline")
                     for p in sp.train.shapes)
        cv = Counter(regime_of(p, T.hw, axis="roofline")
                     for p in sp.val.shapes)
        print(f"  {i:<6d} {len(sp.train.shapes):6d} "
              f"{f'{ct[chr(109)+chr(101)+chr(109)]}/{ct[chr(99)+chr(111)+chr(109)+chr(112)]}':>10s}   "
              f"{len(sp.val.shapes):4d} "
              f"{f'{cv[chr(109)+chr(101)+chr(109)]}/{cv[chr(99)+chr(111)+chr(109)+chr(112)]}':>10s}")
        out["folds"].append({
            "fold": i, "kind": sp.kind,
            "n_train": len(sp.train.shapes), "n_val": len(sp.val.shapes),
            "train_regimes": dict(ct), "val_regimes": dict(cv),
            "val": [[p.M, p.N, p.K] for p in sp.val.shapes],
            "train": [[p.M, p.N, p.K] for p in sp.train.shapes]})

    # ★ Are the folds' vals disjoint / is their union the whole set
    seen: Counter = Counter()
    for f in out["folds"]:
        for v in f["val"]:
            seen[tuple(v)] += 1
    print(f"\n  ★ does each val appear exactly once: "
          f"{set(seen.values()) == {1} and len(seen) == len(shapes)}")
    print("  ⚠️ a limit — each of the three folds' val is in another fold's "
          "**train**.")
    print("     And we have looked at all 61 shapes. They are not 'entirely "
          "new shapes',")
    print("     so the claim reaches only as far as **'it does not depend on "
          "one particular split'**.")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
