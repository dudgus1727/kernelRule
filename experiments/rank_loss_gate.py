"""★ The rank-loss pass condition — does the surrogate loss diverge from the
objective? 0 LLM calls.

    python3 experiments/rank_loss_gate.py

## Why a rank loss

If only some of the configs are measured, `regret` cannot be used —
**because the optimum is not known.**

```
regret = time / ★ the optimal time     undefined if the optimum is unknown
rank loss = is the ordering among the ones measured right
                                       ★ the optimum need not be known
```

**But a surrogate loss can diverge from the objective.** That is measured
before any config sampling is done. **Here the configs are used
exhaustively** — mixing in the sampling variable makes it impossible to tell
what caused what.

```
A) fit with the rank loss -> ★ scored with regret
B) fit with regret        -> scored with regret    (the current way)
```

## The three things the design keeps

```
1 ★ not every pair is counted the same
    Among 19,635, the ordering of the 15,000th and the 16,000th means
    nothing. The positives are **the true top K** and the negatives are
    drawn evenly across the rank bands

2 ★ pairs within the noise floor are dropped
    `NoiseModel.resolvable(t_i, t_j)` makes that judgement.
    Without dropping them it fits the noise (the 5090's answer set is 9 at
    the median and up to 724)

3 ★ the relaxation temperature is not tuned — T is **fixed** at 1
    The score is linear in w (s = Φw), so scaling w by c scales s by c.
    That is, T **is not identified separately** from the magnitude of w.
    Having a T does not add a degree of freedom, it duplicates one.
    ★ So T=1 is a normalisation, not a choice.
```

## Linearity

`s = Φ w` is used, so **the score has to be linear in w.** It is actually
checked per structure (does 2w give 2s), and if not, that structure is
dropped and the fact is written down.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, SplitSet, experiment_shapes, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
SRC_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
N_POS = 32          # the true top K
N_NEG = 256         # evenly across the rank bands
TEMP = 1.0          # ★ fixed. See the docstring above


def _splits(table: PerfTable) -> SplitSet:
    shapes = experiment_shapes(table)
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def _phi(fn, matrix, table, p, n_w: int) -> np.ndarray:
    """The per-term value matrix Φ (candidates x terms). Drawn with unit basis
    weights."""
    cand = table.candidates(p)
    cols = []
    for j in range(n_w):
        w = np.zeros(n_w)
        w[j] = 1.0
        cols.append(np.asarray(make_score_of(fn, matrix, w)(p, cand),
                               dtype=np.float64))
    return np.stack(cols, axis=1)


def _is_linear(phi: np.ndarray, fn, matrix, table, p, w: np.ndarray) -> bool:
    """Is s = Φw? If not, this method cannot be used — it is not passed over
    silently."""
    cand = table.candidates(p)
    s = np.asarray(make_score_of(fn, matrix, w)(p, cand), dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(phi @ w)
    if not ok.any():
        return False
    d = np.abs(s[ok] - (phi @ w)[ok])
    scale = np.maximum(np.abs(s[ok]), 1.0)
    return bool(np.max(d / scale) < 1e-9)


def _pairs(table, p, rng) -> tuple[np.ndarray, np.ndarray]:
    """(positive, negative) indices. **Pairs the noise cannot separate are
    dropped.**"""
    t = table.times_of(p)
    order = np.argsort(t, kind="stable")
    pos = order[:N_POS]
    rest = order[N_POS:]
    if len(rest) == 0:
        return pos, rest
    # Evenly across the rank bands — drawing only from the tail teaches only
    # the easy pairs
    idx = np.unique(np.linspace(0, len(rest) - 1, N_NEG).astype(int))
    neg = rest[idx]
    return pos, neg


def _pair_data(fn, matrix, table, shapes, n_w, rng):
    """It pre-builds the Φ submatrix and the valid-pair mask per shape."""
    out = []
    for p in shapes:
        phi = _phi(fn, matrix, table, p, n_w)
        pos, neg = _pairs(table, p, rng)
        if len(neg) == 0:
            continue
        t = table.times_of(p)
        # ★ Only the pairs the noise floor can separate
        res = table.noise.resolvable(t[pos][:, None], t[neg][None, :])
        worse = t[neg][None, :] > t[pos][:, None]
        mask = res & worse
        if not mask.any():
            continue
        out.append((phi[pos], phi[neg], mask))
    return out


def _rank_loss_and_grad(w, data):
    """The mean logistic pair loss and its gradient. Since s = Φw it comes out
    analytically."""
    tot, n = 0.0, 0
    g = np.zeros_like(w)
    for phi_p, phi_n, mask in data:
        sp = phi_p @ w
        sn = phi_n @ w
        d = (sn[None, :] - sp[:, None]) / TEMP     # it should be positive
        # log(1 + exp(-d)) — stably
        loss = np.logaddexp(0.0, -d)
        sig = -1.0 / (1.0 + np.exp(d))             # d(loss)/d(d)
        m = mask.astype(np.float64)
        tot += float((loss * m).sum())
        n += int(mask.sum())
        gm = sig * m / TEMP
        g += (gm.sum(axis=0) @ phi_n) - (gm.sum(axis=1) @ phi_p)
    if n == 0:
        return 0.0, g
    return tot / n, g / n


def _regret_on(fn, matrix, table, shapes, ws) -> float:
    regs = []
    for p in shapes:
        cand = table.candidates(p)
        sc = make_score_of(fn, matrix, ws[regime_of(p, table.hw)])(p, cand)
        t = table.times_of(p)
        regs.append(float(t[cand.top_k(sc, 1)[0]] / t.min()))
    return geomean(np.array(regs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/rank-loss-gate.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    B = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    mB = FeatureMatrix(B, REGISTRY)
    sp = _splits(B)

    print("=" * 78)
    print("the rank-loss pass condition — does the surrogate loss diverge "
          "from the objective")
    print("=" * 78)
    print(f"  5090 training {len(sp.train.shapes)} / holdout "
          f"{len(sp.val.shapes)}   ★ the configs are exhaustive")
    print(f"  pairs: {N_POS} positives from the top, {N_NEG} negatives across "
          f"the rank bands, pairs the noise cannot separate excluded, "
          f"T = {TEMP} fixed\n")
    print(f"  {'structure':26s} {'A) rank loss':>13} {'B) regret':>11} "
          f"{'A-B':>9}  linear")

    rows = []
    for run in SRC_RUNS:
        f = Path("runs") / run / "archive.jsonl"
        e = sorted((json.loads(x) for x in f.read_text().splitlines()
                    if x.strip()), key=lambda z: z["regret"])[0]
        fn = compile_rule(e["code"])
        w0 = np.asarray(e["w"], dtype=np.float64)
        n_w = len(w0)

        # The linearity check — one shape is enough (the structure does not
        # depend on the shape)
        p0 = sp.train.shapes[0]
        phi0 = _phi(fn, mB, B, p0, n_w)
        lin = _is_linear(phi0, fn, mB, B, p0, w0 * 1.7)
        if not lin:
            print(f"  {run:26s} {'—':>13} {'—':>11} {'—':>9}  ★ non-linear")
            rows.append({"run": run, "linear": False})
            continue

        ws_a, ws_b = {}, {}
        for nm in ("short", "long"):
            g = [q for q in sp.train.shapes if regime_of(q, B.hw) == nm]
            rng = np.random.default_rng(0)
            data = _pair_data(fn, mB, B, g, n_w, rng)
            r = minimize(_rank_loss_and_grad, w0, args=(data,), jac=True,
                         method="L-BFGS-B",
                         options={"maxiter": 500, "maxfun": 2000})
            ws_a[nm] = r.x
            # ★ The B arm is regret by definition (D-99). It is stated —
            #   the default changed to rank, so without saying it the two
            #   arms become the same.
            ws_b[nm] = fit_weights(fn, mB, B, Split("train", tuple(g)), w0,
                                   max_evals=300, objective="regret").w
        va = _regret_on(fn, mB, B, list(sp.val.shapes), ws_a)
        vb = _regret_on(fn, mB, B, list(sp.val.shapes), ws_b)
        print(f"  {run:26s} {va:13.4f} {vb:11.4f} {va - vb:+9.4f}  ✓")
        rows.append({"run": run, "linear": True, "rank_loss": va,
                     "regret": vb, "diff": va - vb})

    ok = [r for r in rows if r.get("linear")]
    if ok:
        A_ = np.array([r["rank_loss"] for r in ok])
        Bv = np.array([r["regret"] for r in ok])
        from scipy.stats import wilcoxon
        try:
            _, pv = wilcoxon(A_, Bv)
        except Exception:                                   # noqa: BLE001
            pv = float("nan")
        print(f"\n  median  A) {np.median(A_):.4f}   B) {np.median(Bv):.4f}   "
              f"difference {np.median(A_) - np.median(Bv):+.4f}")
        print(f"  ★ paired Wilcoxon two-sided p = {pv:.4f}  "
              f"(structures where A is worse: "
              f"{int(np.sum(A_ - Bv > 0))}/{len(ok)})")
        print("  ★ for reference, the decision line (σ upper bound, n=6 "
              "paired): 0.0516 — sigma-5090.json")
    Path(a.out).write_text(json.dumps(
        {"bundle": G5090[0], "n_pos": N_POS, "n_neg": N_NEG, "temp": TEMP,
         "note": ("the configs are exhaustive. The sampling variable was not "
                  "mixed in. T is not identified separately from the "
                  "magnitude of w, so it is fixed at 1"),
         "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
