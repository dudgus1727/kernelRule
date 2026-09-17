"""★ The **preliminary check** for the budget-16 experiment — does the fitter
hold up in 16 dimensions? 0 LLM calls.

    python3 experiments/fitter_dim.py            # everything (about 15 min,
                                                 # 12 processes)
    python3 experiments/fitter_dim.py --arm B16-one

**Why this comes first.** If the budget is raised from 8 to 16 and the result
gets worse, two readings diverge: "budget 16 is bad" and "the fitter cannot
find it in 16 dimensions". The latter has to be ruled out before the former
can be measured (principle 1 — infrastructure -> checker -> subject).

## What is measured

```
reach rate      the pre-registration criterion (D-56). The fraction where
                4000 random points fail to beat the fit result
                ★ As the dimension goes up, 4000 points get relatively
                  sparser, so this **rises by itself**. This metric must not
                  be compared between 8 and 16 dimensions
dimension       ★ the fraction that **ends up worse** when only the terms are
loss rate         added from the same starting point
                If the new terms' initial weights are small, the extended
                rule is almost the same function as the original. So if the
                16-dimensional fit ends up worse than the 8-dimensional one,
                that loss is **because of the dimension**, not the structure.
                It is fair to the dimension
refit reach     the fraction where refitting from a random starting point
rate            beats starting from w0
                It compares against a random 'fit' instead of a random
                'point' — it is fair to the dimension
```

## The arms

```
A8        the original 8 terms
B16-lo    16 terms, the new terms' initial weight 0.01, the budget unchanged
          (300 / 600)
B16-one   16 terms, the new terms' initial weight 1.0,  the budget unchanged
C16-one   16 terms, the new terms' initial weight 1.0,  ★ the budget scaled
          (600 / 4000)
```

The two `B16-*` arms are measured separately because **the initial weight is
a confound** — the simplex step is proportional to `max(|start|, 1.0)` so
0.01 and 1.0 get the same step, but the polish `d * max(|t[i]|, 1.0)` is the
same too, so the difference comes only from the function itself.
`C16-one` separates **whether the dimension loss is the budget's fault or the
algorithm's**.

The absolute value of regret is not reported (D-56 §2). Only differences and
ratios are used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
#: The human-24 library arm. The same runs as the baseline arm of the budget
#: experiment.
RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
#: ★ The 3 rank-loss evolution runs (D-101). Used when `--objective rank` —
#: to measure that path it has to be measured on the structure that path
#: produced.
RANK_RUNS = [f"x-rank-rankevo-s{i}" for i in range(3)]
N_PROBE = 4000
PROBE_LO, PROBE_HI = 0.05, 50.0
#: The number of random starts for the refit reach rate. One of them is one
#: fit, so it is expensive.
N_RESTART_FITS = 3
TARGET_REACH = 0.90
BIG = 16

ARMS = {
    #  name        terms  new-term init  max_evals  polish_budget
    "A8":      (8,  None, 300, 600),
    "B16-lo":  (16, 0.01, 300, 600),
    "B16-one": (16, 1.0,  300, 600),
    "C16-one": (16, 1.0,  600, 4000),
}


def extend_code(code: str, f_names: list[str], p_names: list[str],
                n_add: int) -> tuple[str, list[str]]:
    """Extends the rule by `n_add` **linear terms**.

    It attaches unused features one at a time — if a new term repeats the
    same physical quantity as an existing one, it becomes 'a duplicate term
    appeared' instead of 'only the dimension went up', and the measurement
    gets muddy.

    ★ `f` and `p` are **different name spaces**. `f` is the (shape, config)
    matrix and `p` is a shape-level value, so writing `p.roofline_ratio` as
    `f.` raises `AttributeError`. 5 of the human 24 (`arith_intensity`,
    `can_use_cp_async`, `is_memory_bound`, the SOL bound, `roofline_ratio`)
    are shape level, so `f` has only 19, and there are rules that cannot fill
    8 terms from those alone. When they run short it fills up with **the
    square of an already used feature** — it is not a new physical quantity,
    but it is a linearly independent term, so the dimension goes up honestly.
    What was attached is recorded.
    """
    used_f = set(re.findall(r"\bf\.(\w+)", code))
    used_p = set(re.findall(r"\bp\.(\w+)", code))
    pool = [f"f.{n}" for n in f_names if n not in used_f]
    pool += [f"p.{n}" for n in p_names if n not in used_p]
    # ★ The square of a binary feature is **itself** (0^2=0, 1^2=1).
    #   Attaching it as it is creates a fully duplicated term and makes "the
    #   dimension went up" false. It is excluded.
    cont = [n for n in sorted(used_f)
            if not (n.startswith(("is_", "has_", "can_")))]
    pool += [f"np.square(f.{n})" for n in cont]
    pool += [f"(f.{x} * f.{y})" for i, x in enumerate(cont)
             for y in cont[i + 1:]]
    if len(pool) < n_add:
        raise SystemExit(f"there are only {len(pool)} terms to attach — "
                         f"{n_add} are needed")
    pick = pool[:n_add]
    k = max(int(i) for i in re.findall(r"\bw\[(\d+)\]", code)) + 1
    lines = [f"    s = s + {e} * w[{k + i}]" for i, e in enumerate(pick)]
    out = re.sub(r"\n(\s*)return s\s*$",
                 "\n" + "\n".join(lines) + r"\n\1return s\n",
                 code.rstrip() + "\n")
    return out, pick


_G: dict = {}


#: The workers inherit it by fork. `main` fills it in.
_OBJECTIVE = {"v": "regret"}


def _init() -> None:
    warnings.simplefilter("ignore")
    _G["objective"] = _OBJECTIVE["v"]
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import experiment_shapes, regime_of
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    shapes = experiment_shapes(table)
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

    n_terms, init, max_evals, pol = ARMS[arm]
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
    fr = fit_weights(fn, _G["matrix"], _G["table"], sp, w0, max_evals=max_evals,
                     warn_invariants=False, polish=True, polish_budget=pol,
                          objective=_G.get("objective", "regret"))

    # ★ The probe has to be measured with **the same objective as the fit**
    #   (D-103). At first `prob.regret` was used fixed, and comparing a
    #   result fitted with `objective="rank"` against a regret probe
    #   **measures a different thing** — the reach rate came out 0%, and that
    #   was the metric being off, not something about the fitter.
    obj = _G.get("objective", "regret")
    prob = W._Problem(_G["matrix"], _G["table"], tuple(g), 1)
    if obj == "rank":
        prob.build_pairs(_G["table"], 100)

    def _value(w):
        return prob.regret(fn, w) if obj == "regret" else prob.rank_loss(fn, w)

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
        st = np.exp(rs.uniform(np.log(PROBE_LO), np.log(PROBE_HI), size=len(w0)))
        try:
            r = fit_weights(fn, _G["matrix"], _G["table"], sp, st,
                            max_evals=max_evals, warn_invariants=False,
                            polish=True, polish_budget=pol,
                          objective=_G.get("objective", "regret"))
        except Exception:
            continue
        rb = min(rb, _value(r.w))

    return dict(arm=arm, run=run, regime=regime, fit=_value(fr.w),
                fit_regret=fr.fit_regret,
                n_evals=fr.n_evals, n_fit_evals=fr.n_fit_evals,
                # ★ It compares on the value with the polish evaluations
                #   taken out — comparing on the summed value makes the
                #   polish budget always exceed the cap and come out 100%.
                hit_cap=fr.n_fit_evals >= max_evals,
                moved=bool(fr.moved), seconds=fr.seconds,
                probe_best=float(bv), restart_best=float(rb),
                added=pick)


def summarize(rows: list[dict], arms: list[str]) -> dict:
    """Builds the per-arm summary from the cells. **It does not fit again** —
    it can be re-derived from the saved json with `--reduce`."""
    import statistics as _st

    by: dict = {}
    for r in rows:
        by.setdefault(r["arm"], {})[(r["run"], r["regime"])] = r

    print()
    print(f"  {'arm':9s} {'reach':>8} {'refit reach':>11} {'dim loss':>10} "
          f"{'budget up':>9} {'moved':>6}")
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
        cap = sum(1 for r in cells.values() if r["hit_cap"])
        mv = sum(1 for r in cells.values() if r["moved"])
        rg = sorted((r["fit"] - r["restart_best"] for r in cells.values()
                     if r["restart_best"] < r["fit"] - 1e-9), reverse=True)
        gaps = ([cells[k]["fit"] - by["A8"][k]["fit"]
                 for k in cells if k in by["A8"]]
                if arm != "A8" and "A8" in by else [])
        losses = [g for g in gaps if g > 1e-9]
        loss = f"{len(losses)}/{len(gaps)}" if gaps else None
        summary[arm] = {
            "n": n, "reach": reach, "restart_reach": rreach,
            "budget_used_up": cap, "moved": mv,
            "dim_loss": loss,
            "dim_loss_max": (max(losses) if losses else 0.0),
            "restart_lost": len(rg),
            "restart_gap_max": (max(rg) if rg else 0.0),
            "restart_gap_median": (_st.median(rg) if rg else 0.0),
            # ★ A key starting with `_` is skipped by the md/json agreement
            #   check — the per-cell raw data is for reproduction, not a
            #   reporting target.
            "_gaps_vs_A8": sorted(gaps, reverse=True),
            "_restart_gaps": rg,
        }
        print(f"  {arm:9s} {reach:2d}/{n} {reach / n:5.0%} "
              f"{rreach:2d}/{n} {rreach / n:5.0%} "
              f"{(loss or '—'):>10} {cap:2d}/{n:<5d} {mv:2d}/{n}")

    print()
    for arm in arms:
        g = (summary.get(arm) or {}).get("_gaps_vs_A8")
        if g:
            print(f"  {arm} vs A8 gap (positive = 16 dimensions is worse): "
                  + ", ".join(f"{x:+.4f}" for x in g))

    print()
    a8 = summary.get("A8")
    if a8:
        r = a8["reach"] / a8["n"]
        print(f"  ★ the 8-dimensional reach rate {r:.0%} "
              f"{'passes' if r >= TARGET_REACH else 'falls short'} — the "
              f"pre-registration criterion")
    print("  ★ **do not compare** the 16-dimensional reach rate with the "
          "8-dimensional one — 4000 random points are far sparser in 16 "
          "dimensions")
    print("  ★ 'budget up' is a state, not an anomaly — the restart schedule "
          "uses the whole budget by design (D-76)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", choices=list(ARMS))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default="docs/artifacts/fitter-dim16.json")
    ap.add_argument("--objective", choices=("regret", "rank"),
                    default="regret",
                    help="★ the default is regret — that is the condition "
                         "D-77 measured. rank is differentiable and uses "
                         "L-BFGS, so the result can differ")
    ap.add_argument("--reduce", metavar="JSON",
                    help="rebuild only the summary from the saved cells (no "
                         "fitting)")
    a = ap.parse_args()
    arms = a.arm or list(ARMS)

    if a.reduce:
        old = json.loads(Path(a.reduce).read_text())
        rows = old.get("_cells") or old["cells"]
        summary = summarize(rows, arms)
        out = dict(old)
        out["summary"] = summary
        out["_cells"] = rows
        out.pop("cells", None)
        if "procedure" in out:
            out["_procedure"] = out.pop("procedure")
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\n  -> {a.out}")
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _OBJECTIVE["v"] = a.objective
    # ★ To measure the rank-loss path it has to be measured on **the
    #   structure that path produced**.
    runs = RANK_RUNS if a.objective == "rank" else RUNS
    tasks = [(arm, run, rg) for arm in arms for run in runs
             for rg in ("short", "long")]
    print("=" * 78)
    print(f"the 16-dimensional fitter check — {len(tasks)} cells, "
          f"arms {arms}, ★ objective {a.objective}, {len(runs)} runs")
    print("=" * 78)

    import multiprocessing as mp
    with mp.Pool(a.jobs, initializer=_init) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(work, tasks), 1):
            rows.append(r)
            print(f"  [{i:2d}/{len(tasks)}] {r['arm']:8s} {r['run'][-2:]:3s} "
                  f"{r['regime']:5s} fit={r['n_fit_evals']:4d} "
                  f"total={r['n_evals']:5d} "
                  f"{'cap' if r['hit_cap'] else '   ':4s} "
                  f"{r['seconds']:5.1f}s", flush=True)

    summary = summarize(rows, arms)

    Path(a.out).write_text(json.dumps({
        "_procedure": dict(bundle=BUNDLE, runs=runs,
                           objective=a.objective, n_probe=N_PROBE,
                           probe_range=[PROBE_LO, PROBE_HI],
                           n_restart_fits=N_RESTART_FITS, arms=ARMS,
                           note="`fit` is raw data for reproduction. The "
                                "absolute value is not a reporting target "
                                "(D-56 §2) — the documents use only the "
                                "differences."),
        "_cells": rows, "summary": summary},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
