"""★ Picking the regime split axis again — SOL 0.5 vs roofline vs no split.
0 LLM calls.

    python3 experiments/regime_axis.py

The pre-registration is `docs/artifacts/regime-axis-prereg.md`.

## Why it is reimplemented

`canonical_score` hardcodes the regime names `("short","long")` and
`regime_of(axis="size")`. Swapping the axis means rebuilding that procedure
here. **So arm ① is first checked against the known representative value** —
if it differs, it stops.

⚠️ 2026-09-08 (D-146): the arm labels were translated together with the keys
of `arms` in `regime-axis.json` and the row names of the document.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import warnings
from pathlib import Path

import numpy as np
from sigma_5090 import _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.numerics import approx_equal
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import _DUMMY_CFG, Split
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.features.physical import is_memory_bound

BUNDLE = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
RUNS = [f"F3rw-p8-nan-s{i}" for i in range(6)]
ROUND = 11
#: The value arm ① has to reproduce (D-140). A difference means the
#: reimplementation is wrong.
KNOWN_ARM1 = 1.0886
#: The decision line. **It is not set anew here** (principle 7).
DELTA, BUFFER = 0.0516, 0.0589
#: If a regime has fewer training shapes than this, its weights are hard to
#: trust.
MIN_PER_REGIME = 8


def _rule(run: str) -> dict:
    rows = [json.loads(x) for x
            in (Path("runs") / run / "bests.jsonl").read_text().splitlines()
            if x.strip()]
    return max([e for e in rows if e["round"] <= ROUND],
               key=lambda e: e["round"])


#: ⛔ 2026-09-17 (D-179) — the SOL axis this script compared against was
#: **removed from the code**. The 0.5 ms boundary was ours (chosen on the
#: a6000 table), and the scoring path used it to fit two weight vectors while
#: the loop evolved one.
#:
#: ⚠️ The script is kept: its recorded numbers are on the record and this is
#: how they were produced. ⛔ It cannot be re-run, and it says so instead of
#: quietly swapping in another cut.
def _sol(thr: float):
    def f(p, hw):
        raise SystemExit(
            "regime_axis._sol: the SOL axis was removed at D-179. This "
            "script cannot be re-run; its numbers stay in docs/decisions.md. "
            "⛔ Do not substitute another boundary.")
    return f


def _roof(p, hw):
    return "mem" if is_memory_bound(p, hw, _DUMMY_CFG) else "comp"


def _one(p, hw):
    return "all"


ARMS = [
    ("① SOL 0.5",  _sol(0.5),  ("short", "long")),
    ("② roofline", _roof,      ("mem", "comp")),
    ("③ no split",   _one,       ("all",)),
    ("①' SOL 0.25", _sol(0.25), ("short", "long")),
    ("①'' SOL 1.0", _sol(1.0),  ("short", "long")),
]


def score(code, w0, T, M, sp, reg_fn, names, max_evals=300):
    """**The same procedure** as `canonical_score`, with only the regime
    function swapped."""
    fn = compile_rule(code)
    train, val = list(sp.train.shapes), list(sp.val.shapes)
    per_shape, warns = {}, []
    for nm in names:
        g_tr = [p for p in train if reg_fn(p, T.hw) == nm]
        g_ho = [p for p in val if reg_fn(p, T.hw) == nm]
        if not g_tr:
            if g_ho:
                warns.append(f"{nm}: 0 training but {len(g_ho)} holdout")
            continue
        if len(g_tr) < MIN_PER_REGIME:
            warns.append(f"{nm}: training {len(g_tr)} < {MIN_PER_REGIME}")
        fit = fit_weights(fn, M, T, Split("train", tuple(g_tr)), w0,
                          max_evals=max_evals, objective="regret",
                          warn_invariants=False)
        if not g_ho:
            continue
        e = evaluate_scores(make_score_of(fn, M, fit.w), T, g_ho, ks=(1,))
        for i, p in enumerate(e.shapes):
            per_shape[p] = float(e.regret[i, 0])
    if len(per_shape) < len(val):
        warns.append(f"only {len(per_shape)} of the {len(val)} holdout scored")
    return (geomean(np.array([per_shape[p] for p in val if p in per_shape])),
            per_shape, warns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/regime-axis.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(BUNDLE[0], env_hash=BUNDLE[1], ok_only=False)
    M, sp = FeatureMatrix(T, REGISTRY), _splits(T)
    rules = [(r, _rule(r)) for r in RUNS]

    out: dict = {"delta": DELTA, "buffer": BUFFER, "round": ROUND,
                 "runs": RUNS, "arms": {}}
    print("=" * 96)
    print("the regime split axis — the same six rules, only the weights redone "
          "(0 LLM calls)")
    print("=" * 96)
    print(f"  {'arm':14s} {'median':>8s} {'range':>19s} "
          f"{'holdout per regime':>20s}")

    shape_reg = {}
    for lab, fnr, names in ARMS:
        vals, per_run, allw = [], [], set()
        for run, r in rules:
            v, ps, w = score(r["code"], r["w"], T, M, sp, fnr, names)
            vals.append(v)
            per_run.append({str(k): x for k, x in ps.items()})
            allw.update(w)
        cnt = {nm: sum(1 for p in sp.val.shapes if fnr(p, T.hw) == nm)
               for nm in names}
        shape_reg[lab] = {str(p): fnr(p, T.hw) for p in sp.val.shapes}
        out["arms"][lab] = {"vals": vals, "median": st.median(vals),
                            "holdout_counts": cnt, "warnings": sorted(allw),
                            "per_run_shape": per_run}
        print(f"  {lab:14s} {st.median(vals):8.4f} "
              f"{min(vals):9.4f}~{max(vals):<9.4f} {str(cnt):>20s}"
              + ("   ⚠️ " + "; ".join(sorted(allw)) if allw else ""))

    m1 = out["arms"]["① SOL 0.5"]["median"]
    print(f"\n  ★ the reproduction check: arm ① median {m1:.4f}  vs  the known "
          f"representative value {KNOWN_ARM1:.4f}   "
          f"difference {m1 - KNOWN_ARM1:+.4f}")
    if not approx_equal(m1, KNOWN_ARM1, 5e-4):
        print("  ⛔ it does not reproduce — the reimplementation is wrong. The "
              "other arms are not reported.")
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        raise SystemExit(1)
    print("  ✅ reproduced — the other arms can be read")

    print("\n" + "-" * 96)
    print("  the verdict (the decision line 0.0516, the buffer band ~0.0589)")
    base = out["arms"]["① SOL 0.5"]["median"]
    for lab in ("② roofline", "③ no split", "①' SOL 0.25", "①'' SOL 1.0"):
        d = out["arms"][lab]["median"] - base
        v = ("indistinguishable" if abs(d) < DELTA else
             "★ near the decision line (no verdict)" if abs(d) < BUFFER else
             ("★ better than ①" if d < 0 else "★ worse than ①"))
        print(f"  {lab:14s} vs ① {d:+.4f}   {v}")

    # Split per regime — ② and ③ are looked at through ②'s split
    print("\n" + "-" * 96)
    print("  ★ the per-regime holdout regret seen through the roofline split "
          "(the median of 6 runs)")
    reg2 = shape_reg["② roofline"]
    print(f"  {'arm':14s} {'mem':>9s} {'comp':>9s}")
    for lab in ("① SOL 0.5", "② roofline", "③ no split"):
        cells = {}
        for nm in ("mem", "comp"):
            per = []
            for d in out["arms"][lab]["per_run_shape"]:
                v = [x for k, x in d.items() if reg2.get(k) == nm]
                if v:
                    per.append(geomean(np.array(v)))
            cells[nm] = st.median(per) if per else float("nan")
        out["arms"][lab]["by_roofline"] = cells
        print(f"  {lab:14s} {cells['mem']:9.4f} {cells['comp']:9.4f}")

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
