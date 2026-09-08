"""★ The weight in the exponent slot — the result of form (b). 0 LLM calls.

    python3 experiments/power_report.py

The pre-registration is `docs/artifacts/power-prereg.md`. The baseline is
`rankevo` (the same condition, only without the hint).

## Why the uptake rate alone is not enough

"18% appeared in the proposals but only 1 in the archive" reads two ways —
**it was tried and lost** and **it was a bad proposal to begin with**
(unrelated to the form). So the proposals themselves are **fitted directly**
and put up against each other (§2).
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from pathlib import Path

import numpy as np
from two_stage import A6000, _fit, _floor, _measure, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import _Problem
from kernelrule.features import REGISTRY
from kernelrule.rules.checks import exponent_indices, weight_bounds

ARMS = [("rankevo", "baseline (no hint)"), ("pow", "exponent slot stated")]
SEEDS = 3
N_HEAD = 20          # ★ the head-to-head sample (the same count each side)


def _rows(d: Path, name: str) -> list[dict]:
    return [json.loads(x) for x in (d / name).read_text().splitlines()
            if x.strip()]


def _best(d: Path) -> dict:
    return sorted(_rows(d, "archive.jsonl"),
                  key=lambda e: e.get("rank_loss", 1e9))[0]


def _proposals(d: Path) -> list[dict]:
    out = []
    for f in sorted((d / "llm_calls").glob("*-rule_editor.json")):
        r = json.loads(f.read_text()).get("response")
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:                           # noqa: BLE001
                continue
        if isinstance(r, dict) and r.get("code") and r.get("w0"):
            out.append(r)
    return out


def _n_terms(code: str) -> int:
    return max([n.slice.value for n in ast.walk(ast.parse(code))
                if isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id == "w"
                and isinstance(n.slice, ast.Constant)] + [-1]) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/power.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    dirs = {t: [Path("runs") / f"f1pipe-F3-{t}-s{i}" for i in range(SEEDS)]
            for t, _ in ARMS}
    out: dict = {"n_holdout": len(hold)}

    # -------------------------------------------- §1 was the form used
    print("=" * 84)
    print("§1  was the form actually used — proposals / archive / the fitted "
          "exponents")
    print("=" * 84)
    for tag, lab in ARMS:
        pr = ex = arc_n = arc_e = 0
        expo_w = []
        for d in dirs[tag]:
            for p in _proposals(d):
                pr += 1
                if exponent_indices(p["code"]):
                    ex += 1
            for e in _rows(d, "archive.jsonl"):
                arc_n += 1
                idx = exponent_indices(e["code"])
                if idx:
                    arc_e += 1
                    expo_w += [float(e["w"][i]) for i in idx if i < len(e["w"])]
        print(f"  {lab:22s} proposals {ex:3d}/{pr:3d} "
              f"({ex / max(1, pr):5.1%})   "
              f"archive {arc_e:2d}/{arc_n:2d} ({arc_e / max(1, arc_n):5.1%})")
        if expo_w:
            print(f"  {'':22s} ★ fitted exponents "
                  f"{[round(x, 3) for x in expo_w]}  "
                  f"median |w-1| {np.median(np.abs(np.array(expo_w) - 1)):.3f}")
        out.setdefault("uptake", {})[tag] = {
            "prop": pr, "prop_expo": ex, "arc": arc_n, "arc_expo": arc_e,
            "expo_w": expo_w}

    # ------------------------------- §2 the proposals head to head
    print("\n" + "=" * 84)
    print("§2  ★ was it tried and lost — the proposals are **fitted "
          "directly** and compared")
    print("=" * 84)
    props = [p for d in dirs["pow"] for p in _proposals(d)]
    with_e = [p for p in props if exponent_indices(p["code"])]
    without = [p for p in props if not exponent_indices(p["code"])]
    rng = np.random.default_rng(0)
    # ⚠️ 2026-09-08 (D-146): **these two labels stay in Korean.** They are
    #    the row names of `docs/artifacts/power.md` and the keys of `head` in
    #    `power.json`, and `docs/` is not translated.
    pick = {"지수 있음": [with_e[i] for i in
                       rng.choice(len(with_e), N_HEAD, replace=False)],
            "지수 없음": [without[i] for i in
                       rng.choice(len(without), N_HEAD, replace=False)]}
    prob = _Problem(M, T, sp.train.shapes, 1)
    prob.build_pairs(T, 100)
    print(f"  {N_HEAD} are drawn at random from each side of the same run's "
          f"proposals and fitted **by the same procedure**\n")
    print(f"  {'':16s} {'train rank loss':>16} {'holdout regret':>15} "
          f"{'trm':>4}   (rank-loss range)")
    for lab, ps in pick.items():
        rl, rg, tm = [], [], []
        for p in ps:
            try:
                w0 = list(p["w0"])
                fn, ws = _fit(p["code"], w0, T, M, train, "rank")
            except Exception:                           # noqa: BLE001
                continue
            w = ws["long"]
            rl.append(float(prob.rank_loss(compile_rule(p["code"]),
                                           np.asarray(w, float))))
            rg.append(_measure(fn, ws, T, M, hold)[0])
            tm.append(_n_terms(p["code"]))
        rl, rg = np.array(rl), np.array(rg)
        print(f"  {lab:16s} {np.median(rl):16.4f} {np.median(rg):15.4f} "
              f"{np.median(tm):4.1f}   ({rl.min():.3f}~{rl.max():.3f})")
        out.setdefault("head", {})[lab] = {
            "rank_loss": rl.tolist(), "regret": rg.tolist(),
            "terms": tm}

    # -------------------------------- §3 the cell the decision line is on
    print("\n" + "=" * 84)
    print("§3  the final metric — the holdout of 20 shapes, the median of "
          "3 seeds")
    print("=" * 84)
    fl = _floor(T, hold)
    for obj, head in (("regret", ("★ the regret refit — the cell the "
                                  "decision line is on")),
                      ("rank", "the rank fit")):
        print(f"\n  --- {head} ---")
        print(f"  {'':22s} {'regret':>8} {'top-100 tau':>12} {'all':>9}"
              f"   (tau range)")
        for tag, lab in ARMS:
            vals = []
            for d in dirs[tag]:
                e = _best(d)
                fn, ws = _fit(e["code"], e["w"], T, M, train, obj)
                vals.append(_measure(fn, ws, T, M, hold))
            v = np.array(vals)
            und = int(v[:, 3].sum())
            print(f"  {lab:22s} {np.median(v[:, 0]):8.4f} "
                  f"{np.median(v[:, 1]):12.3f} {np.median(v[:, 2]):9.3f}"
                  f"   ({v[:, 1].min():+.3f}~{v[:, 1].max():+.3f})"
                  + (f"  ⚠️ tau undefined in {und} shapes" if und else ""))
            out.setdefault(obj, {})[tag] = [list(x) for x in v]
        print(f"  {'★ random floor':22s} {fl[0]:8.4f} {fl[1]:12.3f} "
              f"{fl[2]:9.3f}")
    out["floor"] = fl

    r = out["regret"]["pow"]
    med_r, med_t = float(np.median([x[0] for x in r])), \
        float(np.median([x[1] for x in r]))
    print("\n  the verdict — the line nailed down in the pre-registration")
    print(f"    regret {med_r:.4f} / top-100 tau {med_t:+.3f}  ->  " + (
        "★ the wall got lower" if med_t >= 0.20 and med_r <= 1.15
        else "the wall is not a problem of the form"))

    # ------------------------------------------------------------ in common
    print("\n" + "=" * 84)
    print("in common")
    print("=" * 84)
    print(f"  {'':22s} {'trm':>4} {'rej':>7} {'fitter':>8} {'min':>7}")
    for tag, lab in ARMS:
        arc = [e for d in dirs[tag] for e in _rows(d, "archive.jsonl")]
        prop = rej = mv = sc = 0
        secs = 0.0
        for d in dirs[tag]:
            for x in _rows(d, "rounds.jsonl"):
                prop += x["n_proposed"]
                rej += (x["n_rejected_schema"] + x["n_rejected_static"]
                        + x["n_rejected_sandbox"] + x["n_rejected_fit"])
                mv += x["n_fit_moved"]
                sc += x["n_scored"]
                secs += x["seconds"]
        print(f"  {lab:22s} "
              f"{np.median([_n_terms(e['code']) for e in arc]):4.0f} "
              f"{rej / prop:7.1%} {mv / max(1, sc):8.1%} {secs / 60:7.1f}")
        out.setdefault("common", {})[tag] = {
            "rej": rej / prop, "reach": mv / max(1, sc), "minutes": secs / 60}

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3 seeds cannot give significance — it is read by range "
          "separation (principle 27)")
    _ = weight_bounds


if __name__ == "__main__":
    main()
