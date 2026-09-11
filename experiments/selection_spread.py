"""★ Is the spread made by the evolution or by the selection? 0 LLM calls.

    python3 experiments/selection_spread.py

## The hypothesis

Six seeds of the same condition give a spread of 0.098 on the structural
holdout. But there are several rules inside an archive, and we pick only the
one with **the minimum training score**.

    the first run: the training-best rule's holdout 1.2035 / the
                   validation-best rule's 1.1305
                   -> which one is picked changed it by 0.073

On the 41 training shapes the top rules differ by 0.001, and the holdout is
a different set of shapes. It is natural for the ranking to flip.

## What is measured

For each run the top k by training score are taken and the holdout
distribution is looked at.

    the seed-to-seed spread at k=1        0.098 (already known)
    the spread of the **best** holdout    narrow means "there is a good rule
    among the top k                       and we cannot pick it"
    the spread of the **median** holdout  narrow means "whichever is picked
    of the top k                          it is similar and only the best
                                          jumps"

The response differs. And **the ensemble** (the rank average of the top k) is
measured alongside — it removes the selection itself, and variance reduction
is a well-known property.

⚠️ The holdout is **not used for selection** (§10.2 / D-36). Looking at the
holdout best here is a **diagnostic** of "is there such a rule in the
archive", not a selection criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate, geomean
from kernelrule.core.splits import Split, SplitSet, experiment_shapes, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
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
RUNS: list[str] = []
KS = (1, 3, 5, 10)


def main() -> None:
    _require(RUNS)
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    shapes = experiment_shapes(table)
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")
    train = list(splits.train.shapes)

    def fit_per_regime(code, w0):
        """The per-regime weights. The ensemble has to use the same procedure
        for the comparison to hold."""
        out = {}
        fn = compile_rule(code)
        for name in ("short", "long"):
            g = [p for p in train if regime_of(p, table.hw) == name]
            out[name] = fit_weights(fn, matrix, table,
                                    Split("train", tuple(g)), w0,
                                    max_evals=300,
                          objective="regret").w
        return fn, out

    def ensemble_regret(fitted: list) -> float:
        """It picks by the **rank average** of the top k. It removes the
        selection itself."""
        regs = []
        for p in held:
            reg = regime_of(p, table.hw)
            cand = table.candidates(p)
            ranks = np.zeros(len(cand.tiebreak), dtype=float)
            for fn, ws in fitted:
                sc = make_score_of(fn, matrix, ws[reg])(p, cand)
                # ★ The ranks have to **preserve ties**.
                #   `argsort(argsort(x))` gives different ranks to equal
                #   scores, and that order is decided by the array index —
                #   which disables the final scoring tie-break in `top_k`
                #   (§30.7: 29/66 shapes are tied at the optimum).
                #   The k=1 ensemble really did differ from the single rule
                #   by 0.009.
                ranks += np.unique(sc, return_inverse=True)[1].astype(float)
            pick = cand.top_k(ranks, 1)[0]
            t = table.times_of(p)
            regs.append(float(t[pick] / t.min()))
        return geomean(np.array(regs))

    print("=" * 76)
    print("is the spread made by the evolution or by the selection")
    print("=" * 76)
    print(f"  {len(RUNS)} runs of the same condition / structural holdout "
          f"{len(held)} shapes\n")
    hdr = "  ".join(f"k={k}" for k in KS)
    print(f"  {'run':30s} {'k=1':>8}   "
          f"{'best holdout among the top k':>30}")
    print(f"  {'':30s} {'':8s}   {hdr:>30}")

    best_of: dict[int, list] = {k: [] for k in KS}
    med_of: dict[int, list] = {k: [] for k in KS}
    ens_of: dict[int, list] = {k: [] for k in KS}
    for run in RUNS:
        f = Path("runs") / run / "archive.jsonl"
        if not f.exists():
            continue
        with f.open() as fh:
            arc = sorted((json.loads(ln) for ln in fh if ln.strip()),
                         key=lambda e: e["regret"])
        # The holdout scores of the top 10 are computed once each
        hos, fitted = [], []
        for e in arc[:max(KS)]:
            r = canonical_score(e["code"], e["w"], table=table, matrix=matrix,
                                splits=splits)
            hos.append(r.holdout)
            fitted.append(fit_per_regime(e["code"], e["w"]))
        line = []
        for k in KS:
            sub = hos[:k]
            best_of[k].append(min(sub))
            med_of[k].append(float(np.median(sub)))
            ens_of[k].append(ensemble_regret(fitted[:k]))
            line.append(f"{min(sub):.4f}")
        print(f"  {run:30s} {hos[0]:8.4f}   " + "  ".join(line), flush=True)

    v = evaluate(vendor_order_fn(table, load_vendor(VENDOR),
                                 mapping="nearest"),
                 table, held, ks=(1,))
    print(f"\n{'=' * 76}")
    print("the seed-to-seed spread (max - min)")
    print("=" * 76)
    print(f"  {'':22s} " + "  ".join(f"k={k:<6d}" for k in KS))
    for label, d in (("best of the top k", best_of),
                     ("median of the top k", med_of),
                     ("★ ensemble (rank average)", ens_of)):
        sp = [max(d[k]) - min(d[k]) for k in KS]
        print(f"  {label:22s} " + "  ".join(f"{x:8.4f}" for x in sp))
    print()
    print(f"  {'':22s} " + "  ".join(f"k={k:<6d}" for k in KS))
    for label, d in (("best of the top k", best_of),
                     ("median of the top k", med_of),
                     ("★ ensemble (rank average)", ens_of)):
        md = [float(np.median(d[k])) for k in KS]
        print(f"  {label:22s} " + "  ".join(f"{x:8.4f}" for x in md)
              + "   (the median)")
    print(f"\n  vendor {v.at(1):.4f}   ★ the pass condition")


if __name__ == "__main__":
    main()
