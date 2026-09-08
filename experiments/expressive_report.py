"""★ The §3 expressiveness set — budget, product, exponent **on the regret
path**. Reported in one go.

    python3 experiments/expressive_report.py       # -> docs/artifacts/expressive-regret.json

The pre-registration is `docs/artifacts/expressive-regret-prereg.md`. The
decision line is nailed down there — it is not set here.

```
rb08   budget 8            ★ the internal baseline of the four arms
rb16   budget 16
rprod  budget 8 + the product hint
rpow   budget 8 + the exponent hint
```

All four arms run with **the same seeds and the same fitter** (CMA-ES,
1 restart, fit 300 / polish 600) — it is the arm the §2 pass condition
picked (D-123).

⚠️ **The fitter on the measuring side is CMA too.** Refitting with
Nelder-Mead reaches only 92% on a 16-term rule (D-77), so it would measure
the failure of the measuring side rather than "budget 16 is bad". That is
why `_fit(method="cma", n_restarts=1)` is given.

⚠️ The baseline `1.0762` is a **Nelder-Mead 200/600** number. It is written
down for reference only and `rb08` is what the judgement uses (principle 4).

The helper functions come from `two_stage.py` / `regret_at_k.py` /
`wall_report.py` (principle 2).
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from power_report import _proposals
from regret_at_k import _measure as _measure_k
from two_stage import A6000, _fit, _splits
from wall_report import _n_terms, _prod_pairs

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.runset import assert_same_condition, run_condition
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.rules.checks import _numeric_literals

#: (tag, label, **what this arm has different** from the baseline)
ARMS = [("rb08", "budget 8 (baseline)", None),
        ("rb16", "budget 16", "parameters"),
        ("rprod", "the product hint", "product_hint"),
        ("rpow", "the exponent hint", "power_hint")]
SEEDS = 3
#: Pre-registration §3. **It is not set here** (principle 7).
DELTA = 0.0516
#: The same condition under the old fitter (Nelder-Mead 200/600). **It is a
#: reference value** (principle 4).
OLD_BASELINE = 1.0762
#: The k values to report. Pre-registration §1 — 1 is the main metric, and
#: 10/50/100 look at "does it point at the region more precisely".
REPORT_KS = (1, 10, 50, 100)


def _runs(tag: str) -> list[str]:
    return [f"f1pipe-F3-{tag}-s{i}" for i in range(SEEDS)]


def _rows(run: str, name: str) -> list[dict]:
    f = Path("runs") / run / name
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()]


def _best(run: str) -> dict:
    """★ The best regret. All four arms have regret as the objective."""
    return sorted(_rows(run, "archive.jsonl"), key=lambda e: e["regret"])[0]


def _spent(code: str, n_w: int) -> int:
    """The budget unit = numeric literals + weights. **Not the number of
    terms** (D-108)."""
    counted, _ = _numeric_literals(ast.parse(code))
    return len(counted) + n_w


def _n_power(code: str) -> int:
    """The number of terms that have **a weight in the exponent slot**, as in
    `np.power(f.*, w[i])`."""
    n = 0
    for x in ast.walk(ast.parse(code)):
        if (isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                and x.func.attr == "power" and len(x.args) == 2
                and any(isinstance(m, ast.Name) and m.id == "w"
                        for m in ast.walk(x.args[1]))):
            n += 1
        if (isinstance(x, ast.BinOp) and isinstance(x.op, ast.Pow)
                and any(isinstance(m, ast.Name) and m.id == "w"
                        for m in ast.walk(x.right))):
            n += 1
    return n


def _squares_and_crosses(code: str) -> tuple[int, int]:
    """★ It counts **squares and cross products separately** (the D-110
    correction).

    Folding them into a union makes `f.a * f.a` and `f.a * f.b` the same
    "pair", and the most frequent pair was in fact `reg_pressure^3`.
    """
    sq = cr = 0
    for a, b in _prod_pairs(code):
        if a == b:
            sq += 1
        else:
            cr += 1
    for x in ast.walk(ast.parse(code)):
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) \
                and x.func.attr == "square":
            sq += 1
    return sq, cr


def _condition_table(out: dict) -> None:
    """★ §0 — is the condition single per arm, and is **exactly one** thing
    different between arms?

    This is the place that made principle 39 executable (D-120). If it
    catches something here, every number below is meaningless — it stops
    first (§26.4).
    """
    print("=" * 92)
    print("§0  conditions — single per arm, one difference between arms "
          "(principle 39)")
    print("=" * 92)
    base = None
    for tag, label, diff in ARMS:
        cond = assert_same_condition(_runs(tag), label=f"{label} ({tag})")
        c = cond or run_condition(_runs(tag)[0])
        print(f"  {label:16s} budget {str(c['parameters']):>3s}  "
              f"objective {c['objective']:6s}  fitter {c['fit_method']}/"
              f"{c['fit_restarts']}  product {str(c['product_hint']):5s}  "
              f"exponent {str(c['power_hint']):5s}  "
              f"seed {str(c['seed_sha'])[:8]}")
        out.setdefault("condition", {})[tag] = c
        if base is None:
            base = c
            continue
        got = sorted(k for k in base
                     if str(base[k]) != str(c[k]))
        exp = [diff] if diff else []
        if got != exp:
            raise SystemExit(
                f"★ the keys where {label} differs from the baseline are "
                f"{got} — they must be {exp}. If two or more conditions "
                f"differ, we do not know what was measured.")
        print(f"  {'':16s} -> keys differing from the baseline: {got} ✅")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/expressive-regret.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    out: dict = {"delta": DELTA, "old_baseline": OLD_BASELINE,
                 "ks": list(REPORT_KS)}
    _condition_table(out)

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    out["n_holdout"] = len(hold)
    best = {tag: [_best(r) for r in _runs(tag)] for tag, _, _ in ARMS}

    # ------------------------------------------------ §1 was it binding
    print("\n" + "=" * 92)
    print("§1  did the budget **bind** — this is where it was void three "
          "times (D-105·106·107)")
    print("=" * 92)
    print(f"  {'':16s} {'budget spent':>22} {'terms (all)':>18} "
          f"{'★ round-1 terms':>18}")
    for tag, label, _ in ARMS:
        sp_, tm, r0 = [], [], []
        cap = out["condition"][tag]["parameters"]
        for r in _runs(tag):
            for e in _rows(r, "archive.jsonl"):
                sp_.append(_spent(e["code"], len(e["w"])))
                tm.append(_n_terms(e["code"]))
            # ★ Round 1 is counted from **the proposals** (the D-107 check).
            #   It must not be counted from the archive — MAP-Elites replaces
            #   them with later rounds, so no `round==0` entry survives (in
            #   rb16 there really were 0 of them).
            for q in _proposals(Path("runs") / r)[:12]:
                try:
                    r0.append(_n_terms(q["code"]))
                except SyntaxError:
                    continue
        n_at = sum(1 for x in sp_ if x >= cap)
        print(f"  {label:16s} median {np.median(sp_):4.1f} / cap {cap:2d} "
              f"({n_at:2d}/{len(sp_):2d} at cap)   "
              f"median {np.median(tm):4.1f} ({min(tm)}~{max(tm)})   "
              f"median {(np.median(r0) if r0 else float('nan')):4.1f} "
              f"({(min(r0) if r0 else 0)}~{(max(r0) if r0 else 0)})")
        out.setdefault("bind", {})[tag] = {
            "spent": sp_, "terms": tm, "round0_terms": r0, "cap": cap,
            "n_at_cap": n_at}
    b16 = out["bind"]["rb16"]
    if b16["terms"] and max(b16["terms"]) <= 8:
        print("\n  ★ void — not one rule in the budget-16 arm goes over "
              "8 terms (that is what D-107 was)")

    # ------------------------------------------------- §2 the main metric
    print("\n" + "=" * 92)
    print("§2  the main metric — regret@1, the A6000 holdout of 20 shapes, "
          "3 seeds")
    print("=" * 92)
    meas: dict = {}
    for tag, _label, _ in ARMS:
        vals = []
        for e in best[tag]:
            fn, ws = _fit(e["code"], e["w"], T, M, train, "regret",
                          method="cma", n_restarts=1)
            vals.append(_measure_k(fn, ws, T, M, hold))
        meas[tag] = vals
        out.setdefault("measure", {})[tag] = [
            {"regret_at_k": {str(k): v["regret_at_k"][k] for k in REPORT_KS},
             "tau_raw": v["tau_raw"], "tau_noise": v["tau_noise"]}
            for v in vals]
    r1 = {t: [v["regret_at_k"][1] for v in meas[t]] for t, _, _ in ARMS}
    print(f"  {'':16s} {'median':>8} {'seed range':>19} "
          f"{'vs baseline':>11}  verdict")
    b = float(np.median(r1["rb08"]))
    for tag, label, _ in ARMS:
        v = r1[tag]
        m = float(np.median(v))
        d = b - m                     # positive = this arm is **better**
        if tag == "rb08":
            verdict = "— (the baseline)"
        elif d >= DELTA:
            verdict = f"★ expressiveness was the wall (+{d:.4f} >= {DELTA})"
        else:
            verdict = f"indistinguishable ({d:+.4f} < {DELTA})"
        print(f"  {label:16s} {m:8.4f}   {min(v):.4f}~{max(v):.4f}   "
              f"{d:+11.4f}  {verdict}")
        print(f"  {'':16s} per seed " + "  ".join(f"{x:.4f}" for x in v))
    print(f"\n  {'ref: old fitter':16s} {OLD_BASELINE:8.4f}   "
          "← budget 8 under Nelder-Mead 200/600. ★ Not used for the "
          "judgement (principle 4)")
    print("  ★ the judgement is read from **the seed range** — 3 seeds do "
          "not give significance")
    print("     (the decision line 0.0516 is a power calculation for n=6)")
    out["r1"] = r1

    # ---------------------------------------------------------- §3 regret@k
    print("\n" + "=" * 92)
    print("§3  regret@k — does it point at the region **more precisely**, or "
          "does only the top-1 get better")
    print("=" * 92)
    print(f"  {'':16s} " + " ".join(f"{f'k={k}':>16s}"
                                    for k in REPORT_KS))
    for tag, label, _ in ARMS:
        arr = np.array([[v["regret_at_k"][k] for k in REPORT_KS]
                        for v in meas[tag]])
        med = np.median(arr, axis=0)
        print(f"  {label:16s} " + " ".join(
            f"{med[i]:6.3f} {arr[:, i].min():.3f}-{arr[:, i].max():.3f}"
            for i in range(len(REPORT_KS))))
    print()
    print(f"  {'':16s} {'top-100 tau':>12} {'noise-aware':>12}")
    for tag, label, _ in ARMS:
        tr = [v["tau_raw"] for v in meas[tag]]
        tn = [v["tau_noise"] for v in meas[tag]]
        print(f"  {label:16s} {np.median(tr):12.3f} {np.median(tn):12.3f}")

    # --------------------------------------------- §4 the gap and the cost
    print("\n" + "=" * 92)
    print("§4  the train-holdout gap / the fitter / the rejection rate / "
          "the cost")
    print("=" * 92)
    print(f"  {'':16s} {'train':>8} {'holdout':>9} {'gap':>9} "
          f"{'fit moved':>9} {'rej rate':>8} {'min':>7}")
    for tag, label, _ in ARMS:
        tr, ho, mv, sc, prop, rej = [], [], 0, 0, 0, 0
        secs = 0.0
        for r in _runs(tag):
            rs = _rows(r, "rounds.jsonl")
            tr.append(rs[-1]["best_regret"])
            ho.append(rs[-1]["best_val_regret"])
            for x in rs:
                prop += x["n_proposed"]
                rej += (x["n_rejected_schema"] + x["n_rejected_static"]
                        + x["n_rejected_sandbox"] + x["n_rejected_fit"])
                mv += x["n_fit_moved"]
                sc += x["n_scored"]
                secs += x["seconds"]
        g = [h - t for h, t in zip(ho, tr, strict=True)]
        print(f"  {label:16s} {np.median(tr):8.4f} {np.median(ho):9.4f} "
              f"{np.median(g):+9.4f} {mv / max(1, sc):9.1%} "
              f"{rej / max(1, prop):7.1%} {secs / 60:7.1f}")
        out.setdefault("cost", {})[tag] = {
            "train": tr, "hold": ho, "gap": g, "moved": mv / max(1, sc),
            "rej": rej / max(1, prop), "minutes": secs / 60,
            "n_proposed": prop}

    # ------------------------------------------------------- §5 the forms
    print("\n" + "=" * 92)
    print("§5  was the form actually used — ★ squares and cross products are "
          "counted **separately** (D-110)")
    print("=" * 92)
    print(f"  {'':16s} {'proposals':>22} {'archive':>22}")
    print(f"  {'':16s} {'ruleswithprod  sq  cross':>22} "
          f"{'ruleswithprod  sq  cross':>22} "
          f"{'exponent prop/archive':>20}")
    for tag, label, _ in ARMS:
        agg = {}
        # ★ The proposals are read from `llm_calls/*-rule_editor.json` —
        #   there is no `proposals.jsonl` (`power_report._proposals`,
        #   principle 2).
        for name in ("prop", "arc"):
            n = nsq = ncr = npw = used = 0
            for r in _runs(tag):
                d = Path("runs") / r
                src = (_proposals(d) if name == "prop"
                       else _rows(r, "archive.jsonl"))
                for e in src:
                    code = e.get("code")
                    if not code:
                        continue
                    try:
                        sq, cr = _squares_and_crosses(code)
                        pw = _n_power(code)
                    except SyntaxError:
                        continue
                    n += 1
                    nsq += sq
                    ncr += cr
                    npw += pw
                    used += 1 if (sq or cr) else 0
            agg[name] = dict(n=n, used=used, sq=nsq, cr=ncr, pw=npw)
        p_, ar = agg["prop"], agg["arc"]
        print(f"  {label:16s} "
              f"{p_['used']:3d}/{p_['n']:<4d} {p_['sq']:4d} {p_['cr']:4d}   "
              f"{ar['used']:3d}/{ar['n']:<4d} {ar['sq']:4d} {ar['cr']:4d}   "
              f"{p_['pw']:4d} / {ar['pw']:<4d}")
        out.setdefault("form", {})[tag] = agg

    # ------------------------------------------------------ §6 the verdict
    print("\n" + "=" * 92)
    print("§6  ★ the verdict — the line from pre-registration §3, not set "
          "after seeing the results")
    print("=" * 92)
    hits = [label for tag, label, _ in ARMS
            if tag != "rb08" and b - float(np.median(r1[tag])) >= DELTA]
    if hits:
        print(f"  ★ expressiveness was the wall — {hits}")
        print("     -> what did not work under the rank loss was a property "
              "of that objective")
        print("     -> the 'wall' section of conclusion.md gets rewritten "
              "(pre-registration §4)")
    else:
        print("  ★ all three indistinguishable — the wall is a property of "
              "the feature space")
        print("     It is not linear/non-linear, and it is not the budget. "
              "The candidate that remains is **a tree's")
        print("     conditional branching** — our `np.where` varies only by "
              "the shape level (`p.*`)")
        print("     while GBDT also varies by thresholds on config-level "
              "features")
        print("     (it is registered as a candidate in pre-registration §4)")
    out["hits"] = hits

    n_cfg = {}
    for tag, _label, _ in ARMS:
        picks = []
        for e in best[tag]:
            fn, ws = _fit(e["code"], e["w"], T, M, train, "regret",
                          method="cma", n_restarts=1)
            for p in hold:
                cand = T.candidates(p)
                from kernelrule.core.splits import regime_of
                from kernelrule.core.weights import make_score_of
                s = make_score_of(fn, M, ws[regime_of(p, T.hw)])(p, cand)
                j = int(cand.top_k(s, 1)[0])
                picks.append((str(cand.kernel_id[j]), int(cand.split_k[j])))
        n_cfg[tag] = len(Counter(picks))
    print("\n  the kinds of config picked (3 seeds x 20 holdout shapes): "
          + ", ".join(f"{lab}={n_cfg[t]}" for t, lab, _ in ARMS))
    out["n_config"] = n_cfg

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
