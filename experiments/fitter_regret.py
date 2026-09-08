"""★ The §2 pass condition — does the fitter hold up in 16 dimensions **on
the regret path**? 0 LLM calls.

    OMP_NUM_THREADS=1 python3 -m experiments.fitter_regret --jobs 20
    python3 -m experiments.fitter_regret --reduce docs/artifacts/fitter-regret.json

★ **It is called with `-m`** — it imports the procedure constants from
`experiments.fitter_dim`, so the repository root has to be on `sys.path`.

The pre-registration is `docs/artifacts/fitter-regret-prereg.md`. The
procedure is **the same** as D-77 (`fitter_dim.py`) — no new metric is made
(principle 2). The only difference is the arms: the total of 900 evals is
given equally to five arms.

```
A8          8 terms   Nelder-Mead 300/4 + polish 600   <- the instrument
                                                          check (D-77 100%)
NM16       16 terms   Nelder-Mead 300/4 + polish 600   <- the current one
                                                          (D-77 B16-lo)
CMA16      16 terms   CMA-ES      300/1 + polish 600
CMA16-full 16 terms   CMA-ES      900/1 + no polish
SURR16     16 terms   surrogate 150 -> regret NM 150/2 + polish 600   <- (b)
```

⚠️ `SURR16` has a rank-loss piece (`rank_loss_top1`) in its first stage. **It
is used only to make the starting point and the acceptance is regret**
(pre-registration §2).

The absolute value of regret is not reported in the documents (D-56 §2). Only
differences and ratios are used.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.fitter_dim import (
    BUNDLE,
    N_PROBE,
    N_RESTART_FITS,
    PROBE_HI,
    PROBE_LO,
    RUNS,
    TARGET_REACH,
    extend_code,
)

#: **The same total evals** per arm. Exactly the table in pre-registration §2.
TOTAL_EVALS = 900

ARMS = {
    #  name          trm  new init  optimiser      fit  restart  polish  surr
    "A8":         (8,  None, "nelder-mead", 300, 4, 600, 0),
    "NM16":       (16, 0.01, "nelder-mead", 300, 4, 600, 0),
    "CMA16":      (16, 0.01, "cma",         300, 1, 600, 0),
    "CMA16-full": (16, 0.01, "cma",         900, 1, 0,   0),
    "SURR16":     (16, 0.01, "nelder-mead", 150, 2, 600, 150),
}
#: The 16-term arms — the target of the decision line and the "same point"
#: comparison. `A8` is the instrument check.
ARMS16 = [a for a, v in ARMS.items() if v[0] == 16]
#: The description criterion of "the same point" in pre-registration §4. It is
#: not a decision line.
SAME_VALUE = 1e-9
SAME_DIR = 0.999

_G: dict = {}


def _init() -> None:
    warnings.simplefilter("ignore")
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import regime_of
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    _G["table"] = table
    _G["matrix"] = FeatureMatrix(table, REGISTRY)
    _G["f_names"] = sorted(REGISTRY.names(shape_level=False))
    _G["p_names"] = sorted(REGISTRY.names(shape_level=True))
    _G["groups"] = {n: [p for p in train if regime_of(p, table.hw) == n]
                    for n in ("short", "long")}


def work(task: tuple[str, str, str]) -> dict:
    arm, run, regime = task
    from kernelrule.core import weights as W
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.splits import Split
    from kernelrule.core.weights import fit_weights

    n_terms, init, method, max_evals, n_restarts, pol, init_evals = ARMS[arm]
    arc = [json.loads(ln) for ln in
           (Path("runs") / run / "archive.jsonl").read_text().splitlines()
           if ln.strip()]
    best = min(arc, key=lambda e: e["regret"])
    code, w0 = best["code"], list(best["w"])
    pick: list[str] = []
    if n_terms > len(w0):
        code, pick = extend_code(code, _G["f_names"], _G["p_names"],
                                 n_terms - len(w0))
        w0 = w0 + [init] * (n_terms - len(w0))
    fn = compile_rule(code)

    g = _G["groups"][regime]
    sp = Split("train", tuple(g))
    # ★ `objective="regret"` is passed **as a literal** — a test checks in the
    #   source that an experiment script states its objective
    #   (`test_history_experiments_pin_their_objective`).
    kw = dict(max_evals=max_evals, n_restarts=n_restarts,
              warn_invariants=False, polish=pol > 0, polish_budget=pol,
              method=method)
    if init_evals:
        kw["init_objective"] = "rank_top1"
        kw["init_evals"] = init_evals
    fr = fit_weights(fn, _G["matrix"], _G["table"], sp, w0,
                     objective="regret", **kw)

    # ★ The probe is measured with **the same objective** as the fit (D-103).
    #   Here everything is regret.
    prob = W._Problem(_G["matrix"], _G["table"], tuple(g), 1)

    def _value(w):
        return prob.regret(fn, w)

    rng = np.random.default_rng(7)
    bv = np.inf
    for _ in range(N_PROBE):
        c = np.exp(rng.uniform(np.log(PROBE_LO), np.log(PROBE_HI), size=len(w0)))
        v = _value(c)
        if np.isfinite(v) and v < bv:
            bv = v

    rb = np.inf
    rs = np.random.default_rng(11)
    for _ in range(N_RESTART_FITS):
        stw = np.exp(rs.uniform(np.log(PROBE_LO), np.log(PROBE_HI),
                                size=len(w0)))
        try:
            r = fit_weights(fn, _G["matrix"], _G["table"], sp, stw,
                            objective="regret", **kw)
        except Exception:
            continue
        rb = min(rb, _value(r.w))

    return dict(arm=arm, run=run, regime=regime, fit=_value(fr.w),
                fit_regret=fr.fit_regret, n_evals=fr.n_evals,
                n_fit_evals=fr.n_fit_evals, n_init_evals=fr.n_init_evals,
                moved=bool(fr.moved), seconds=fr.seconds,
                probe_best=float(bv), restart_best=float(rb),
                w=[float(x) for x in fr.w], added=pick)


def _same_point(by: dict) -> dict:
    """★ Pre-registration §4 — did the arms reach **the same point**? It is a
    description, not a judgement."""
    out: dict = {}
    for a, b in combinations(ARMS16, 2):
        ca, cb = by.get(a, {}), by.get(b, {})
        keys = sorted(set(ca) & set(cb))
        if not keys:
            continue
        dv, cs = [], []
        for kk in keys:
            dv.append(abs(ca[kk]["fit"] - cb[kk]["fit"]))
            wa = np.asarray(ca[kk]["w"], dtype=np.float64)
            wb = np.asarray(cb[kk]["w"], dtype=np.float64)
            na, nb = np.linalg.norm(wa), np.linalg.norm(wb)
            cs.append(float(wa @ wb / (na * nb)) if na and nb else float("nan"))
        fin = [c for c in cs if np.isfinite(c)]
        out[f"{a} vs {b}"] = {
            "n": len(keys),
            "same_value": sum(1 for d in dv if d < SAME_VALUE),
            "dfit_median": st.median(dv), "dfit_max": max(dv),
            "cos_median": (st.median(fin) if fin else float("nan")),
            "cos_min": (min(fin) if fin else float("nan")),
            "same_dir": sum(1 for c in fin if c >= SAME_DIR),
        }
    return out


def summarize(rows: list[dict], arms: list[str]) -> dict:
    by: dict = {}
    for r in rows:
        by.setdefault(r["arm"], {})[(r["run"], r["regime"])] = r

    print()
    print(f"  {'arm':11s} {'reach':>9} {'refit reach':>12} {'moved':>8} "
          f"{'loss vs A8':>11} {'actual evals':>13}")
    summary: dict = {}
    for arm in arms:
        cells = by.get(arm, {})
        if not cells:
            continue
        n = len(cells)
        reach = sum(1 for r in cells.values()
                    if not (r["probe_best"] < r["fit"] - 1e-9))
        rreach = sum(1 for r in cells.values()
                     if not (r["restart_best"] < r["fit"] - 1e-9))
        mv = sum(1 for r in cells.values() if r["moved"])
        rg = sorted((r["fit"] - r["restart_best"] for r in cells.values()
                     if r["restart_best"] < r["fit"] - 1e-9), reverse=True)
        gaps = ([cells[k]["fit"] - by["A8"][k]["fit"]
                 for k in cells if k in by["A8"]]
                if arm != "A8" and "A8" in by else [])
        losses = [g for g in gaps if g > 1e-9]
        ev = [r["n_evals"] for r in cells.values()]
        summary[arm] = {
            "n": n, "reach": reach, "restart_reach": rreach, "moved": mv,
            "dim_loss": (f"{len(losses)}/{len(gaps)}" if gaps else None),
            "dim_loss_max": (max(losses) if losses else 0.0),
            "restart_lost": len(rg),
            "restart_gap_max": (max(rg) if rg else 0.0),
            "restart_gap_median": (st.median(rg) if rg else 0.0),
            "evals_median": st.median(ev), "evals_max": max(ev),
            "_gaps_vs_A8": sorted(gaps, reverse=True),
            "_restart_gaps": rg,
        }
        s = summary[arm]
        print(f"  {arm:11s} {reach:2d}/{n} {reach / n:5.0%} "
              f"{rreach:2d}/{n} {rreach / n:5.0%} {mv:2d}/{n:<5d} "
              f"{(s['dim_loss'] or '—'):>11} "
              f"{s['evals_median']:6.0f}/{s['evals_max']:<6.0f}")

    print()
    for arm in arms:
        g = (summary.get(arm) or {}).get("_gaps_vs_A8")
        if g:
            print(f"  {arm} vs A8 gap (positive = 16 dimensions is worse): "
                  + ", ".join(f"{x:+.4f}" for x in g))

    same = _same_point(by)
    if same:
        print()
        print(f"  ★ did they reach the same point (same value <{SAME_VALUE:g}, "
              f"same direction cos>={SAME_DIR})")
        print(f"  {'pair':26s} {'same val':>9} {'|dfit| median':>14} "
              f"{'max':>8} {'cos median':>11} {'min':>8} {'same dir':>9}")
        for kk, v in same.items():
            print(f"  {kk:26s} {v['same_value']:2d}/{v['n']:<6d} "
                  f"{v['dfit_median']:14.4f} {v['dfit_max']:8.4f} "
                  f"{v['cos_median']:11.3f} {v['cos_min']:8.3f} "
                  f"{v['same_dir']:2d}/{v['n']}")
    summary["_same_point"] = same

    print()
    a8 = summary.get("A8")
    if a8:
        r = a8["reach"] / a8["n"]
        print(f"  ★ the instrument check — A8 reach rate {r:.0%} "
              f"{'pass' if r >= TARGET_REACH else '★ short (D-77 was 100%)'}")
    passed = [a for a in ARMS16
              if a in summary and summary[a]["reach"] / summary[a]["n"]
              >= TARGET_REACH]
    summary["_pass"] = passed
    if passed:
        print(f"  ★ the judgement passes — the 16-term arms with a reach rate "
              f"of {TARGET_REACH:.0%} or more: {passed}")
        print("     On a tie, the refit reach rate, and if that ties too, "
              "NM16 (pre-registration §3)")
    else:
        print(f"  ★ the judgement — every 16-term arm is below a reach rate "
              f"of {TARGET_REACH:.0%}. D-77 is a property of the regret path")
        print("     -> §3-1 (budget 8 vs 16) is dropped and 3-2·3-3 are done "
              "at budget 8 (the instruction §5)")
    print("  ★ do not compare the reach rate between 8 and 16 dimensions — "
          "4000 random points are far sparser in 16 dimensions (D-77)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", choices=list(ARMS))
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", default="docs/artifacts/fitter-regret.json")
    ap.add_argument("--reduce", metavar="JSON",
                    help="rebuild only the summary from the saved cells (no "
                         "fitting)")
    a = ap.parse_args()
    arms = a.arm or list(ARMS)

    if a.reduce:
        old = json.loads(Path(a.reduce).read_text())
        rows = old["_cells"]
        out = dict(old)
        out["summary"] = summarize(rows, arms)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\n  -> {a.out}")
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tasks = [(arm, run, rg) for arm in arms for run in RUNS
             for rg in ("short", "long")]
    print("=" * 78)
    print(f"§2 pass condition — the 16-dimensional fitter on the regret path, "
          f"{len(tasks)} cells, arms {arms}")
    print(f"★ the total of {TOTAL_EVALS} evals is given equally to each arm — "
          f"the actual value is reported")
    print("=" * 78)

    import multiprocessing as mp
    with mp.Pool(a.jobs, initializer=_init) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(work, tasks), 1):
            rows.append(r)
            print(f"  [{i:2d}/{len(tasks)}] {r['arm']:11s} {r['run'][-2:]:3s} "
                  f"{r['regime']:5s} fit={r['n_fit_evals']:4d} "
                  f"surr={r['n_init_evals']:4d} total={r['n_evals']:5d} "
                  f"{r['seconds']:6.1f}s", flush=True)

    summary = summarize(rows, arms)
    Path(a.out).write_text(json.dumps({
        "_procedure": dict(
            bundle=BUNDLE, runs=RUNS, objective="regret", n_probe=N_PROBE,
            probe_range=[PROBE_LO, PROBE_HI],
            n_restart_fits=N_RESTART_FITS, arms=ARMS,
            total_evals=TOTAL_EVALS, prereg="fitter-regret-prereg.md",
            note="`fit` is raw data for reproduction. The absolute value is "
                 "not a reporting target (D-56 §2) — the documents use only "
                 "the differences."),
        "_cells": rows, "summary": summary},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
