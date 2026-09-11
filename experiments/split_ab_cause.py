"""★ Why did 6 axis verdicts flip when validation was narrowed to the
training split? (D-167 §O). **0 LLM calls.**

    python3 experiments/split_ab_cause.py

`experiments/validate_split_ab.py` (D-166 §C) measured **that** 6 of 60
recorded axes pass on the whole table and fail on the training split alone.
It did not measure **why**, and the obvious reading — "the holdout is what
made them pass" — is wrong.

The validator runs on the table's 66 shapes. The experiments run on 61
(`kernelrule.core.splits.experiment_shapes`). This script asks, for each
flipped axis, whether it is **constant on the 61** and varies only on the
5 shapes no experiment ever sees.

```
if so   the axis cannot change a ranking anywhere in the experiment —
        not in training, not in scoring. It is a wasted registry slot and a
        wasted fitter dimension, ★ not a leak and not a number that moves
result
otherwise  the flip has another cause and this script says so
```

⚠️ The spread is measured over `matrix` columns, i.e. per (shape, config).
"Constant" here means **every candidate of every one of the 61 shapes has
the same value.**
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
ENV = "c63710df"
OUT = Path("docs/artifacts/split-ab-cause.json")
#: run -> the axes that flipped (from `validate-split-ab.json`)
SRC = Path("docs/artifacts/validate-split-ab.json")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    warnings.simplefilter("ignore")

    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import experiment_shapes
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY
    from kernelrule.features.loader import run_registry

    if not SRC.exists():
        raise SystemExit(f"{SRC} does not exist. Run "
                         f"`python3 experiments/validate_split_ab.py` first.")
    flipped = {run: [a["name"] for a in v["axes"] if not a["same"]]
               for run, v in json.loads(SRC.read_text())["runs"].items()}

    table = PerfTable.from_bundle(BUNDLE, env_hash=ENV, ok_only=False)
    used = experiment_shapes(table)
    keep = {p.key for p in used}
    dropped = [p for p in table.shapes() if p.key not in keep]
    print(f"table {len(table.shapes())}  used {len(used)}  "
          f"excluded {len(dropped)}")
    for p in dropped:
        print(f"  excluded  {p.M}x{p.N}x{p.K}")

    rows = []
    for run, names in flipped.items():
        if not names:
            continue
        reg, _origin = run_registry(run, table=table, seed=0, human=REGISTRY)
        matrix = FeatureMatrix(table, reg)
        for name in names:
            # ★ A shape-level axis is one number per shape and lives on
            #   `info`; a config-level axis is a column on `f`. Reading the
            #   wrong one raises rather than silently returning nothing.
            is_shape = reg[name].shape_level
            inside, outside = [], []
            for p in table.shapes():
                f, info = matrix.for_shape(p)
                col = np.atleast_1d(np.asarray(
                    getattr(info, name) if is_shape else getattr(f, name),
                    dtype=np.float64))
                (inside if p.key in keep else outside).append(col)
            a = np.concatenate(inside)
            b = np.concatenate(outside)
            row = {
                "run": run, "axis": name, "shape_level": bool(is_shape),
                "used_std": float(a.std()), "used_min": float(a.min()),
                "used_max": float(a.max()),
                "excluded_std": float(b.std()), "excluded_min": float(b.min()),
                "excluded_max": float(b.max()),
                "n_distinct_used": int(len(np.unique(np.round(a, 9)))),
                "n_distinct_excluded": int(len(np.unique(np.round(b, 9)))),
            }
            row["constant_on_used"] = row["n_distinct_used"] == 1
            rows.append(row)
            mark = "★ constant on the 61" if row["constant_on_used"] else "—"
            print(f"  {run:8s} {name:38s} used [{a.min():.4f},{a.max():.4f}] "
                  f"distinct {row['n_distinct_used']:2d} | excluded "
                  f"[{b.min():.4f},{b.max():.4f}] "
                  f"distinct {row['n_distinct_excluded']:2d}  {mark}")

    n_const = sum(1 for r in rows if r["constant_on_used"])
    print(f"\n★ {n_const} of {len(rows)} flipped axes are constant on the "
          f"{len(used)} shapes the experiments use")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"bundle": BUNDLE, "n_table": len(table.shapes()),
         "n_used": len(used),
         "excluded_shapes": [f"{p.M}x{p.N}x{p.K}" for p in dropped],
         "n_constant_on_used": n_const, "n_axes": len(rows), "axes": rows},
        ensure_ascii=False, indent=1) + "\n")
    print(f"recorded: {OUT}")


if __name__ == "__main__":
    main()
