"""★ Does the fitter **create overfitting** — CMA vs Nelder-Mead. 0 LLM calls.

    python3 experiments/fitter_regress.py

The pre-registration is `docs/artifacts/fitter-regress-prereg.md`.

Re-deriving the baseline in §3 turned the direction around (1.0762 ->
1.0987). Three things about the procedure differed (the fitter, the budget,
the number of seeds), so it could not be used for a judgement. **Here the
variable is reduced to the fitter alone** — the same rules are refitted with
both fitters.

```
NM   nelder-mead · 4 restarts · fit 300 · polish 600  ← the procedure used so
                                                        far
CMA  cma         · 1 restart  · fit 300 · polish 600  ← what §3 used
```

⚠️ The training regret is **not used for the judgement** — CMA fitting the
training better is what it is designed to do, and that is the mechanism of
overfitting, not the judgement (pre-registration §3).

⚠️ 2026-09-08 (D-146): **the two group labels stay in Korean.** They are the
top-level keys of `fitter-regress.json` and the row names of the document,
and `docs/` is not translated.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from regret_at_k import _measure as _measure_k
from scipy.stats import wilcoxon
from two_stage import A6000, _fit, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

#: The main target — the 6 regret-evolved structures. **The same six** D-77
#: and D-123 used.
MAIN = ("arch24 6구조", [f"F3rw-p8-s{i}" for i in range(6)])
#: Secondary — the budget-8 arm of §3. Structures evolved with CMA. They are
#: looked at **separately**.
SIDE = ("rb08 3구조 (CMA 진화)", [f"F3rw-p8-cma-s{i}" for i in range(3)])

ARMS = [("NM", "nelder-mead", 4), ("CMA", "cma", 1)]
REPORT_KS = (1, 10, 50, 100)
#: The A6000 seed spread. It is **a ruler for reading the size**, not a
#: decision line.
SIGMA = 0.0124
#: The unpaired reference line (§29.5). It is not set anew here (principle 7).
DELTA = 0.0516


def _best(run: str) -> dict:
    f = Path("runs") / run / "archive.jsonl"
    arc = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return sorted(arc, key=lambda e: e["regret"])[0]


def _one(group: tuple, T, M, train, hold, out: dict) -> None:
    label, runs = group
    print("\n" + "=" * 88)
    print(f"{label} — {len(runs)} structures, the variable is **the fitter "
          f"alone**")
    print("=" * 88)
    rows: dict = {}
    for name, method, nres in ARMS:
        ho, tr, ks = [], [], []
        for r in runs:
            e = _best(r)
            fn, ws = _fit(e["code"], e["w"], T, M, train, "regret",
                          method=method, n_restarts=nres)
            h = _measure_k(fn, ws, T, M, hold)
            t = _measure_k(fn, ws, T, M, train)
            ho.append(h["regret_at_k"][1])
            tr.append(t["regret_at_k"][1])
            ks.append([h["regret_at_k"][k] for k in REPORT_KS])
        rows[name] = {"hold": ho, "train": tr,
                      "ks": np.array(ks).tolist()}
    print(f"  {'':6s} {'train median':>13} {'holdout median':>16} {'gap':>9} "
          f"{'holdout range':>20}")
    for name, _, _ in ARMS:
        d = rows[name]
        g = [h - t for h, t in zip(d["hold"], d["train"], strict=True)]
        print(f"  {name:6s} {np.median(d['train']):13.4f} "
              f"{np.median(d['hold']):16.4f} {np.median(g):+9.4f} "
              f"{min(d['hold']):10.4f}~{max(d['hold']):.4f}")
        d["gap"] = g

    dif = [c - n for n, c in zip(rows["NM"]["hold"], rows["CMA"]["hold"],
                                 strict=True)]
    worse = sum(1 for x in dif if x > 0)
    print("\n  ★ the paired difference (CMA - NM, positive = CMA is "
          "**worse**)")
    print("     " + "  ".join(f"{x:+.4f}" for x in dif))
    print(f"     structures where CMA is worse {worse}/{len(dif)}   "
          f"median {np.median(dif):+.4f}   "
          f"(σ={SIGMA}, reference line {DELTA})")
    p = float("nan")
    if len(dif) >= 5 and any(abs(x) > 0 for x in dif):
        p = float(wilcoxon(dif, alternative="greater").pvalue)
        print(f"     paired one-sided Wilcoxon p = {p:.4f}  "
              f"{'★ significant (<=0.05)' if p <= 0.05 else 'not significant'}")
    else:
        print("     ★ n<5, so Wilcoxon is not used — only the signs are read")

    dtr = [c - n for n, c in zip(rows["NM"]["train"], rows["CMA"]["train"],
                                 strict=True)]
    print("\n  checking the mechanism — what about the training? "
          "(CMA - NM, negative = CMA fits better)")
    print("     " + "  ".join(f"{x:+.4f}" for x in dtr)
          + f"   structures where CMA is better "
            f"{sum(1 for x in dtr if x < 0)}/{len(dtr)}")
    dg = [c - n for n, c in zip(rows["NM"]["gap"], rows["CMA"]["gap"],
                                strict=True)]
    print(f"     the median difference of the gap (holdout-train) "
          f"{np.median(dg):+.4f} — positive means CMA widens it more")

    print("\n  the regret@k medians")
    print(f"  {'':6s} " + " ".join(f"{f'k={k}':>8s}" for k in REPORT_KS))
    for name, _, _ in ARMS:
        a = np.array(rows[name]["ks"])
        print(f"  {name:6s} " + " ".join(f"{m:8.3f}"
                                         for m in np.median(a, axis=0)))
    out[label] = {"rows": rows, "paired_hold": dif, "paired_train": dtr,
                  "paired_gap": dg, "p_hold": p, "n_worse": worse}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/fitter-regress.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    out: dict = {"sigma": SIGMA, "delta": DELTA, "ks": list(REPORT_KS),
                 "arms": [list(x) for x in ARMS]}
    for g in (MAIN, SIDE):
        _one(g, T, M, train, hold, out)

    m = out[MAIN[0]]
    print("\n" + "=" * 88)
    print("★ the verdict — exactly as in pre-registration §3")
    print("=" * 88)
    if m["p_hold"] <= 0.05 and np.median(m["paired_hold"]) > 0:
        print("  ★ the fitter creates overfitting — NM at budget 8, and CMA")
        print("     only when 16 dimensions are needed. The (b) refit of the "
              "4090 transfer goes to NM too")
    elif m["p_hold"] <= 0.05:
        print("  ★ unexpected — CMA is significantly better. That is a branch "
              "the pre-registration does not have")
    else:
        print("  indistinguishable — 1.0762 -> 1.0987 is not the fitter's "
              "fault")
        print("     the remaining candidates: the fit budget (200 vs 300) · "
              "the number of seeds (6 vs 3) ·")
        print("     the polish fix (D-122) · the evolutionary path itself")
    print("  ⚠️ this result does not overturn D-123 (the §2 pass condition) — "
          "that is about the **16-dimensional")
    print("     reach rate** and this is about **8-dimensional "
          "generalisation** (pre-registration §5)")

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                      default=float))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
