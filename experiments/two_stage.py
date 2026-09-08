"""★ The two-stage objective (A) + the transfer of a rank rule (B). 0 LLM
calls.

    python3 experiments/two_stage.py

The **pre-registration** is `docs/artifacts/two-stage-prereg.md` — the
decision line was nailed down first.

## A — the structure by rank, the weights by regret

It takes **the final structure as it is** from the 3 rank-evolution runs and
refits only the weights.

## B — does a rank-evolved rule move over to the 5090

§29.5 was done with `regret`-evolved rules only. A rank rule can be
different.

⚠️ **Everything is measured on the holdout.** D-101's tau (0.389) is a value
on the 41 training shapes and cannot be put alongside (principle 4).

⚠️ 2026-09-08 (D-146): **the row labels stay in Korean.** They are the row
names of `docs/artifacts/two-stage.md` (and `regret-at-k.md`) and the keys of
`two-stage.json`, and `docs/` is not translated.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
RANK_RUNS = [f"x-rank-rankevo-s{i}" for i in range(3)]
REG_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
TOP_N, TAU_SAMPLE, TAU_SEED, N_DRAWS = 100, 4000, 12345, 20


def _splits(t: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = t.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    sh = [p for p in t.shapes() if aligned(p)]
    held = [p for p in sh if 11008 in (p.N, p.K)]
    return SplitSet(train=Split("train", tuple(p for p in sh if p not in held)),
                    val=Split("val", tuple(held)), kind="nk11008")


def _best(run: str, by: str) -> dict:
    f = Path("runs") / run / "archive.jsonl"
    arc = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    key = (lambda e: e.get("rank_loss", 1e9)) if by == "rank" \
        else (lambda e: e["regret"])
    return sorted(arc, key=key)[0]


def _fit(code, w0, table, matrix, train, objective, *,
         rank_top_k: int = TOP_N, rank_lambda: float = 0.0,
         method: str = "nelder-mead", n_restarts: int = 4):
    """It fits per regime — the final scoring procedure (§10).

    ★ `rank_top_k` / `rank_lambda` are **conditions of that run**. Leaving
    them at the default measures the whole k sweep and λ sweep at k=100 and
    λ=0 (principle 37).

    ★ `method` / `n_restarts` are conditions too (D-123). The defaults are
    the values **every report so far** used, so the existing artefacts do not
    change by one character (principle 36). A report measuring a 16-term rule
    has to pass `method="cma", n_restarts=1` — Nelder-Mead reaches only 92%
    in 16 dimensions (D-77·D-123). Without it, what gets measured is **the
    failure of the measuring side's fitter**, not "budget 16 is bad".
    """
    fn = compile_rule(code)
    ws = {}
    for nm in ("short", "long"):
        g = [q for q in train if regime_of(q, table.hw) == nm]
        ws[nm] = fit_weights(fn, matrix, table, Split("train", tuple(g)), w0,
                             max_evals=300, objective=objective,
                             method=method, n_restarts=n_restarts,
                             rank_top_k=rank_top_k,
                             rank_lambda=rank_lambda if objective == "rank"
                             else 0.0).w
    return fn, ws


def _measure(fn, ws, table, matrix, shapes, top_n: int = TOP_N) -> tuple:
    """(regret, the top-`top_n` tau, the all-range tau, **the number of
    shapes where it is undefined**).

    ★ Do look at the fourth value. If the rule gives a **constant score**
    inside the top `top_n`, tau is undefined. Not counting that and taking a
    median either spreads a `nan` (numpy) or silently drops the shape. It
    really happened at k=10 — one shape had exactly 1 distinct score value.
    """
    rng = np.random.default_rng(TAU_SEED)
    regs, tt, ta, undef = [], [], [], []
    for p in shapes:
        cand = table.candidates(p)
        w = ws[regime_of(p, table.hw)] if isinstance(ws, dict) else ws
        s = np.asarray(make_score_of(fn, matrix, w)(p, cand), dtype=np.float64)
        t = np.asarray(table.times_of(p))
        regs.append(float(t[cand.top_k(s, 1)[0]] / t.min()))
        top = np.argsort(t, kind="stable")[:top_n]
        if len(np.unique(t[top])) > 1:
            v = kendalltau(s[top], t[top], variant="b").statistic
            (tt if np.isfinite(v) else undef).append(v)
        idx = rng.choice(len(t), size=min(TAU_SAMPLE, len(t)), replace=False)
        ta.append(kendalltau(s[idx], t[idx], variant="b").statistic)
    return (float(np.exp(np.mean(np.log(regs)))),
            float(np.median(tt)) if tt else float("nan"),
            float(np.median(ta)),
            len(undef))


def _floor(table, shapes, top_n: int = TOP_N) -> tuple:
    """★ The random floor (the mean of 20 draws). The floor is a sample too
    (principle 7)."""
    rng = np.random.default_rng(0)
    R, T1, TA = [], [], []
    for _ in range(N_DRAWS):
        regs, tt, ta = [], [], []
        for p in shapes:
            t = np.asarray(table.times_of(p))
            s = rng.random(len(t))
            regs.append(float(t[int(np.argmin(s))] / t.min()))
            top = np.argsort(t, kind="stable")[:top_n]
            if len(np.unique(t[top])) > 1:
                tt.append(kendalltau(s[top], t[top], variant="b").statistic)
            idx = rng.choice(len(t), size=min(TAU_SAMPLE, len(t)),
                             replace=False)
            ta.append(kendalltau(s[idx], t[idx], variant="b").statistic)
        R.append(float(np.exp(np.mean(np.log(regs)))))
        T1.append(float(np.median(tt)))
        TA.append(float(np.median(ta)))
    return float(np.mean(R)), float(np.mean(T1)), float(np.mean(TA))


def _row(label, vals) -> None:
    v = np.array(vals)
    print(f"  {label:34s} {np.median(v[:, 0]):8.4f} {np.median(v[:, 1]):12.3f} "
          f"{np.median(v[:, 2]):10.3f}   "
          f"({v[:, 1].min():.3f}~{v[:, 1].max():.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/two-stage.json")
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    out: dict = {}

    A = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    mA = FeatureMatrix(A, REGISTRY)
    spA = _splits(A)
    hold = list(spA.val.shapes)
    train = list(spA.train.shapes)

    print("=" * 82)
    print("A. the objective 2x2 — which of the structure and the weights "
          "holds the ranking ability")
    print("=" * 82)
    print(f"  A6000 holdout {len(hold)} shapes   ★ D-101's tau is a value on "
          f"the 41 training shapes and cannot be put beside this\n")
    print(f"  {'':34s} {'regret':>8} {'top-100 tau':>12} {'all':>10}")

    res: dict = {}
    for name, runs, by, obj in (
            ("rank structure + rank weights", RANK_RUNS, "rank", "rank"),
            ("★ rank structure + regret weights", RANK_RUNS, "rank", "regret"),
            # ★ The empty cells of the 2x2 (D-103). Saying "the weights hold
            #   it" from three cells was **a statement made from the diagonal
            #   alone**.
            ("★ regret structure + rank weights", REG_RUNS, "regret", "rank"),
            ("regret structure + regret weights", REG_RUNS, "regret", "regret")):
        vals = []
        for run in runs:
            e = _best(run, by)
            fn, ws = _fit(e["code"], e["w"], A, mA, train, obj)
            vals.append(_measure(fn, ws, A, mA, hold))
        res[name] = vals
        _row(name, vals)
    fl = _floor(A, hold)
    print(f"  {'★ random floor (20 draws)':34s} {fl[0]:8.4f} {fl[1]:12.3f} "
          f"{fl[2]:10.3f}")
    out["A"] = {"rows": dict(res), "floor": fl,
                "n_holdout": len(hold)}

    mid = np.array(res["★ rank structure + regret weights"])
    r, t1 = float(np.median(mid[:, 0])), float(np.median(mid[:, 1]))
    print("\n  the verdict — the line nailed down in the pre-registration")
    print(f"    regret {r:.4f} / top-100 tau {t1:.3f}  ->  " + (
        "★ success (regret<=1.10 and tau>=0.30)"
        if r <= 1.10 and t1 >= 0.30
        else "the weights erase the tau (regret<=1.10, tau<=0.15)"
        if r <= 1.10 and t1 <= 0.15
        else "★ the structure does not suit regret (regret>=1.30) -> (2) is "
             "needed"
        if r >= 1.30 else "indistinguishable"))

    if a.skip_b:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return

    # ---------------------------------------------------------------- B
    B = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    mB = FeatureMatrix(B, REGISTRY)
    spB = _splits(B)
    holdB, trainB = list(spB.val.shapes), list(spB.train.shapes)
    print("\n" + "=" * 82)
    print("B. the transfer of a rank-evolved rule — A6000 -> 5090")
    print("=" * 82)
    print(f"  5090 holdout {len(holdB)} shapes")
    print("  ⚠️ the 5090's top 100 spans 1.2% with 5 distinct values — there "
          "is little ranking to learn\n")
    print(f"  {'':34s} {'regret':>8} {'top-100 tau':>12} {'all':>10}")

    resB: dict = {}
    for name, obj in (("(a) full transplant (A6000 weights)", None),
                      ("★ (b) refit (5090 rank loss)", "rank")):
        vals = []
        for run in RANK_RUNS:
            e = _best(run, "rank")
            if obj is None:
                fn, ws = _fit(e["code"], e["w"], A, mA, train, "rank")
            else:
                fn, ws = _fit(e["code"], e["w"], B, mB, trainB, obj)
            vals.append(_measure(fn, ws, B, mB, holdB))
        resB[name] = vals
        _row(name, vals)
    flB = _floor(B, holdB)
    print(f"  {'★ random floor (20 draws)':34s} {flB[0]:8.4f} {flB[1]:12.3f} "
          f"{flB[2]:10.3f}")
    out["B"] = {"rows": resB, "floor": flB, "n_holdout": len(holdB)}

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3 seeds cannot give significance — it is read by range "
          "separation")


if __name__ == "__main__":
    main()
