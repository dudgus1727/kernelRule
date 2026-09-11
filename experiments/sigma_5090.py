"""★ The 5090 seed spread σ — the ground of the §29.5 decision line. 0 LLM
calls.

    python3 experiments/sigma_5090.py

## Why it is measured again

The A6000's σ must not be used on the 5090 as it is (the pre-registration
`transfer-prereg.md`). The tick is 1/64 and the ridge is 0.74x — the tie
structure differs, so the spread can differ too.

## What is measured

**Exactly the final scoring procedure**: pick one from the archive by the
training score, refit the weights per regime, and score on the structural
holdout. The standard deviation between seeds is σ.

## ★ The σ of n=3 is very wide

The chi-squared confidence interval is reported **alongside**. Using the point
estimate alone narrows the decision line below what it really is — and that is
exactly the road to "reporting a difference we cannot measure" (principle 7).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.runset import assert_same_condition
from kernelrule.core.splits import Split, SplitSet, experiment_shapes
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY


def _splits(table: PerfTable) -> SplitSet:
    shapes = experiment_shapes(table)
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def _sigma_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """The sample standard deviation and its chi-squared confidence interval.
    With a small `n` it is very wide."""
    from scipy.stats import chi2

    n = len(x)
    s = float(np.std(x, ddof=1))
    lo = s * float(np.sqrt((n - 1) / chi2.ppf(1 - (1 - conf) / 2, n - 1)))
    hi = s * float(np.sqrt((n - 1) / chi2.ppf((1 - conf) / 2, n - 1)))
    return s, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="datasets/rtx-5090-sm_120-5bb6f403")
    ap.add_argument("--env-hash", default="5bb6f403")
    ap.add_argument("--runs", nargs="+",
                    default=[f"x-hwold-5090sigma-s{i}" for i in range(3)])
    ap.add_argument("--out", default="docs/artifacts/sigma-5090.json")
    a = ap.parse_args()

    warnings.simplefilter("ignore")
    table = PerfTable.from_bundle(a.bundle, env_hash=a.env_hash, ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    sp = _splits(table)

    print("=" * 74)
    print(f"the seed spread σ   table={Path(a.bundle).name}")
    print("=" * 74)
    print(f"  training {len(sp.train.shapes)} / structural holdout "
          f"{len(sp.val.shapes)}  ({sp.kind})")
    print("  the procedure: 1 from the archive by the training score -> "
          "per-regime refit -> the holdout\n")

    # ★ Is the set's condition single (D-120)? σ is the seed spread **of one
    #   condition** — mixing two conditions makes it a between-condition
    #   difference, not a seed spread.
    assert_same_condition(a.runs, label="the set σ is measured on")
    hold, train = [], []
    for r in a.runs:
        f = Path("runs") / r / "archive.jsonl"
        arc = sorted((json.loads(x) for x in f.read_text().splitlines()
                      if x.strip()), key=lambda e: e["regret"])
        e = arc[0]                   # ★ it is chosen by the training score
        res = canonical_score(e["code"], e["w"], table=table, matrix=matrix,
                              splits=sp)
        hold.append(res.holdout)
        train.append(float(e["regret"]))
        print(f"  {r:28s} train {e['regret']:.4f}   "
              f"holdout {res.holdout:.4f}",
              flush=True)

    h = np.array(hold)
    s, lo, hi = _sigma_ci(h)
    print(f"\n  median {np.median(h):.4f}   "
          f"range {h.min():.4f}~{h.max():.4f}   "
          f"width {h.max() - h.min():.4f}")
    print(f"  ★ σ = {s:.4f}   95% confidence interval [{lo:.4f}, {hi:.4f}]   "
          f"(n={len(h)})")
    print(f"\n  ⚠️ it is an interval for n={len(h)}. **The upper bound is "
          f"used** — setting")
    print("     the decision line from the point estimate means reporting a "
          "difference we cannot measure (principle 7).")

    def need(delta: float, sig: float) -> float:
        """The 2-sample t approximation of the number of seeds, two-sided
        0.05, power 0.8."""
        return 2.0 * (2.8 * sig / delta) ** 2

    print(f"\n  {'difference delta':>18}  {'seeds (σ point est.)':>22}  "
          f"{'seeds (σ upper bound)':>22}")
    for d in (0.02, 0.03, 0.05, 0.10):
        print(f"  {d:18.2f}  {need(d, s):22.0f}  {need(d, hi):22.0f}")

    Path(a.out).write_text(json.dumps({
        "bundle": a.bundle, "env_hash": a.env_hash, "runs": a.runs,
        "split_kind": sp.kind, "n_train": len(sp.train.shapes),
        "n_holdout": len(sp.val.shapes),
        "train_regret": train, "holdout_regret": hold,
        "sigma": s, "sigma_ci95": [lo, hi], "n_seeds": len(h),
        "procedure": ("1 from the archive by the training score -> "
                      "per-regime refit -> the structural holdout "
                      "(nk11008)"),
    }, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
