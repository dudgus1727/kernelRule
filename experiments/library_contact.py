"""★ How much of each split's holdout was read while the **shared feature
library** was being built (D-168 §2-1). **0 LLM calls.**

    python3 experiments/library_contact.py

## Why this number exists

The 21-run campaign builds the stage-1 feature library **once** and every
run shares it. It has to: a rule can only be moved to another GPU if the
source and the target know the same axes (§1-1 of the campaign order).

But stage 1's AUC check **reads the answer** — that is why D-166 §C-1
narrowed it to the training shapes. So the shapes that were in the
library's training split had their answers read while axes were being
accepted or rejected, and for any *other* split those shapes may be
holdout.

```
this script answers   "of split S's holdout, how many shapes were in the
                       library's training set?"
```

## What it found (A6000, 2026-09-11)

★ It is **not** a small contact, and which split builds the library
decides where the contact lands.

```
library built on         fold0     fold1     fold2   nk11008   total
fold0                     0/21   ★ 21/21  ★ 19/19     13/20      53
fold1                    21/21      0/21     19/19     13/20      53
fold2                    21/21     21/21      0/19     14/20      56
★ nk11008                14/21     14/21     13/19   ★  0/20      41
```

The three folds partition the 61 shapes, so `Σ_fold |val ∩ T| = |T|` — the
**total cannot be reduced**, only redistributed. `nk11008` is the choice
that makes the fold contamination a constant (14 · 14 · 13) so the
split-spread comparison is not confounded by it, leaves the transfer
source clean (0/20), and happens to be the smallest total.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

OUT = Path("docs/artifacts/library-contact.json")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    warnings.simplefilter("ignore")

    from experiments.f1_pipeline import _splits
    from experiments.transfer_29_5 import TABLES
    from kernelrule.core.table import PerfTable

    def keys(shapes):
        return {(p.M, p.N, p.K) for p in shapes}

    a6000 = PerfTable.from_bundle(TABLES["a6000"]["bundle"],
                                  env_hash=TABLES["a6000"]["env_hash"],
                                  ok_only=False)
    spec = {"fold0": {"fold": 0}, "fold1": {"fold": 1}, "fold2": {"fold": 2},
            "nk11008": {}}
    sp = {k: _splits(a6000, split_seed=12345, k=3, **v)
          for k, v in spec.items()}

    rows = {}
    print(f"{'library built on':22s} "
          + "  ".join(f"{k:>9s}" for k in spec) + "   total")
    for lib in spec:
        train = keys(sp[lib].train.shapes)
        cells, total = {}, 0
        for s in spec:
            val = keys(sp[s].val.shapes)
            n = len(val & train)
            total += n
            cells[s] = {"contact": n, "n_val": len(val)}
        rows[lib] = {"n_train": len(train), "splits": cells, "total": total}
        print(f"{lib:22s} " + "  ".join(
            f"{cells[s]['contact']:2d}/{cells[s]['n_val']:2d}".rjust(9)
            for s in spec) + f"   {total}  (|T|={len(train)})")

    # ★ The target GPUs. Their shape sets differ, so the intersection is
    #   taken on (M,N,K).
    chosen = "nk11008"
    lib_train = keys(sp[chosen].train.shapes)
    targets = {}
    print(f"\nthe target tables against the chosen library ({chosen}):")
    for g in ("5090", "4090", "h100"):
        tb = PerfTable.from_bundle(TABLES[g]["bundle"],
                                   env_hash=TABLES[g]["env_hash"],
                                   ok_only=False)
        s = _splits(tb, split_seed=12345, k=3)
        val, train = keys(s.val.shapes), keys(s.train.shapes)
        targets[g] = {"n_val": len(val), "contact": len(val & lib_train),
                      "n_shapes": len(val | train),
                      "shared_with_a6000": len((val | train) & lib_train)}
        print(f"  {g:6s} nk11008 holdout {len(val):2d}  contact "
              f"{len(val & lib_train):2d}   (table {len(val | train)} shapes, "
              f"{len((val | train) & lib_train)} of them in the library "
              f"training set)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"note": ("of each split's holdout, how many shapes were in the "
                  "training set the shared feature library was built on "
                  "(stage 1's AUC check reads the answer — D-166 §C-1)"),
         "split_seed": 12345, "k": 3, "chosen_library_split": chosen,
         "a6000": rows, "targets": targets},
        ensure_ascii=False, indent=1) + "\n")
    print(f"\nrecorded: {OUT}")


if __name__ == "__main__":
    main()
