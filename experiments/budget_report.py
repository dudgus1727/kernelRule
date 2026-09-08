"""★ Budget 8 vs 16 — does the budget **break through the wall**? 0 LLM calls.

    python3 experiments/budget_report.py

The pre-registration is `docs/artifacts/budget-prereg.md`. The wall is D-104
— a rule with low regret has a top-100 tau near 0, and a rule with a high tau
has a regret around 1.6.

The helper functions come from `two_stage.py` (principle 2).

⚠️ `runs/x-rank-b16` was **discarded** (D-107 — the output schema was frozen
at 8, so all 29 rules had 8 terms). The 16 arm here is `b16b`.
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from two_stage import A6000, _fit, _floor, _measure, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY
from kernelrule.rules.checks import _numeric_literals

ARMS = [("b08", "budget 8"), ("b16b", "budget 16")]
SEEDS = 3
WATCH = ("split_k_cost", "sm_idle_cost", "pipeline_warmup_frac",
         "tail_waste", "waves")


def _rows(d: Path, name: str) -> list[dict]:
    return [json.loads(x) for x in (d / name).read_text().splitlines()
            if x.strip()]


def _pick(d: Path) -> dict:
    """The best rank loss — both arms have rank as the objective."""
    return sorted(_rows(d, "archive.jsonl"),
                  key=lambda e: e.get("rank_loss", 1e9))[0]


def _feats(code: str) -> set:
    return {n.attr for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "f"}


def _n_terms(code: str) -> int:
    return max([n.slice.value for n in ast.walk(ast.parse(code))
                if isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id == "w"
                and isinstance(n.slice, ast.Constant)] + [-1]) + 1


def _spent(code: str, n_w: int) -> int:
    """★ The budget unit = numeric literals + weights. **Not the number of
    terms.**"""
    counted, _ = _numeric_literals(ast.parse(code))
    return len(counted) + n_w


def _configs(code, w, table, matrix, shapes) -> int:
    fn, w = compile_rule(code), np.asarray(w, dtype=np.float64)
    picks = []
    for p in shapes:
        cand = table.candidates(p)
        j = int(cand.top_k(make_score_of(fn, matrix, w)(p, cand), 1)[0])
        picks.append((str(cand.kernel_id[j]), int(cand.split_k[j]),
                      str(cand.split_k_mode[j])))
    return len(Counter(picks))


def _blk(label: str, vals: list[tuple]) -> dict:
    v = np.array(vals)
    print(f"  {label:16s} {np.median(v[:, 0]):8.4f} {np.median(v[:, 1]):12.3f} "
          f"{np.median(v[:, 2]):10.3f}   "
          f"({v[:, 1].min():+.3f}~{v[:, 1].max():+.3f})")
    return {"vals": [list(x) for x in vals],
            "med": [float(np.median(v[:, i])) for i in range(3)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/budget.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    dirs = {t: [Path("runs") / f"f1pipe-F3-{t}-s{i}" for i in range(SEEDS)]
            for t, _ in ARMS}
    best = {t: [_pick(d) for d in dirs[t]] for t, _ in ARMS}
    out: dict = {"n_holdout": len(hold)}

    # ------------------------------------------------- §1 was it binding
    print("=" * 80)
    print("§1  did the budget **bind** — this comes first (it was wrong three "
          "times in D-107)")
    print("=" * 80)
    print(f"  {'':16s} {'budget spent (literals+weights)':>32} "
          f"{'terms':>16}")
    for tag, label in ARMS:
        sp_, tm = [], []
        lim = json.loads((dirs[tag][0] / "config.json").read_text())
        for d in dirs[tag]:
            for e in _rows(d, "archive.jsonl"):
                sp_.append(_spent(e["code"], len(e["w"])))
                tm.append(_n_terms(e["code"]))
        # ★ The D-128 rename. The old artefacts were converted, but both are
        #   read.
        _rc = lim["rule_constraints"]
        cap = _rc.get("parameters", _rc.get("budget"))
        n_at = sum(1 for x in sp_ if x >= cap)
        print(f"  {label:16s} median {np.median(sp_):4.1f} / cap {cap:2d}   "
              f"rules that used the cap {n_at:2d}/{len(sp_):2d}   median "
              f"{np.median(tm):4.1f} ({min(tm)}~{max(tm)})")
        out.setdefault("bind", {})[tag] = {
            "spent": sp_, "terms": tm, "cap": cap, "n_at_cap": n_at}
    print("\n  ⚠️ the pre-registration said 'void if no rule uses 16 terms'.")
    print("     **That used the wrong unit** — the budget counts literals +")
    print("     weights, not terms. By term count the max is 15, but by "
          "budget spent it reaches the cap.")

    # ------------------------------------------------- §2 the main metric
    print("\n" + "=" * 80)
    print("§2  the main metric — the A6000 holdout of 20 shapes, the median "
          "of 3 seeds")
    print("=" * 80)
    fl = _floor(T, hold)
    for obj, head in (
            ("regret", ("★ regret-refitted weights — the cell the "
                        "decision line is on")),
            ("rank", "rank-fitted weights")):
        print(f"\n  --- {head} ---")
        print(f"  {'':16s} {'regret':>8} {'top-100 tau':>12} {'all':>10}"
              f"   (tau100 range)")
        for tag, label in ARMS:
            vals = [_measure(*_fit(e["code"], e["w"], T, M, train, obj),
                             T, M, hold) for e in best[tag]]
            out.setdefault(obj, {})[tag] = _blk(label, vals)
        print(f"  {'★ random floor':16s} {fl[0]:8.4f} {fl[1]:12.3f} "
              f"{fl[2]:10.3f}")
    out["floor"] = fl

    print("\n  the verdict — the line nailed down in the pre-registration")
    r16 = out["regret"]["b16b"]["med"]
    r08 = out["regret"]["b08"]["med"]
    t16, g16 = r16[1], r16[0]
    verdict = ("★ the budget broke through the wall"
               if t16 >= 0.20 and g16 <= 1.20
               else "tau rose but regret goes over 1.20 — it moved to the "
                    "rank arm" if t16 >= 0.20
               else "★ the wall is not the budget's fault" if t16 <= 0.10
               else "indistinguishable (0.10~0.20)")
    print(f"    budget 16, the regret refit: tau {t16:+.3f} / "
          f"regret {g16:.4f}")
    print(f"    budget 8, the same cell:     tau {r08[1]:+.3f} / regret "
          f"{r08[0]:.4f}")
    print(f"    -> {verdict}")

    # ------------------------------------------------- §3 overfitting
    print("\n" + "=" * 80)
    print("§3  the train-holdout gap / the fitter / the cost")
    print("=" * 80)
    print(f"  {'':16s} {'train':>8} {'holdout':>9} {'gap':>9} "
          f"{'fitter reach':>13} {'rej':>7} {'config':>7} {'min':>7}")
    for tag, label in ARMS:
        tr, ho, reach = [], [], []
        prop = rej = mv = sc = 0
        secs = 0.0
        for d in dirs[tag]:
            rs = _rows(d, "rounds.jsonl")
            tr.append(rs[-1]["best_regret"])
            ho.append(rs[-1]["best_val_regret"])
            reach.append(sum(x["n_fit_moved"] for x in rs)
                         / max(1, sum(x["n_scored"] for x in rs)))
            for r in rs:
                prop += r["n_proposed"]
                rej += (r["n_rejected_schema"] + r["n_rejected_static"]
                        + r["n_rejected_sandbox"] + r["n_rejected_fit"])
                mv += r["n_fit_moved"]
                sc += r["n_scored"]
                secs += r["seconds"]
        g = [h - t for h, t in zip(ho, tr, strict=True)]
        nc = [_configs(e["code"], e["w"], T, M, hold) for e in best[tag]]
        print(f"  {label:16s} {np.median(tr):8.4f} {np.median(ho):9.4f} "
              f"{np.median(g):+9.4f} {np.mean(reach):12.1%} "
              f"{rej / prop:7.1%} {np.median(nc):7.1f} {secs / 60:7.1f}")
        print(f"  {'':16s} {'':8s} {'':9s} per seed "
              f"{' '.join(f'{x:+.4f}' for x in g)}")
        out.setdefault("cost", {})[tag] = {
            "train": tr, "hold": ho, "gap": g, "reach": reach,
            "rej": rej / prop, "n_config": nc, "minutes": secs / 60}

    print("\n  ★ the ceiling measurement's five axes (/3 seeds)")
    print(f"    {'':16s} " + " ".join(f"{n[:11]:>12s}" for n in WATCH))
    for tag, label in ARMS:
        fs = [_feats(e["code"]) for e in best[tag]]
        print(f"    {label:16s} "
              + " ".join(f"{sum(n in f for f in fs):>12d}" for n in WATCH))
        out.setdefault("cost", {})[tag]["feats"] = [sorted(f) for f in fs]
    for tag, label in ARMS:
        fs = [_feats(e["code"]) for e in best[tag]]
        print(f"    {label} axis counts: {[len(f) for f in fs]}  "
              f"union {len(set().union(*fs))}")

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3 seeds cannot give significance — it is read by range "
          "separation (principle 27)")


if __name__ == "__main__":
    main()
