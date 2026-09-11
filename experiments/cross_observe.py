"""★ The `cross` observation — the pre-registration is
`docs/artifacts/cross-prereg.md`. 0 LLM calls.

    python3 experiments/cross_observe.py --runs F3rw-p8-cross-s{0,1,2} \\
                                         --control F3rw-p8-abl-analyst-s{0,1,2}

## The four things

    1 ★ does the child actually mix the terms of two parents  cross.jsonl
    2   the archive cell occupancy                             rounds.jsonl
    3 ★ is the ensemble starting to work (the D-42 retest)     archive.jsonl
                                                              + the table
    4   the duplicate rate of cross                            rounds.jsonl
                                                              by_parent_kind

## ⚠️ Number 3 cannot be put beside the old D-42 numbers

D-42's width 0.098 -> 0.100 / median 1.0817 -> 1.1137 came from **deleted
gpt-5.4 artefacts** (D-52, which is why `selection_spread.py:RUNS` is empty).
A different model cannot be put side by side. So **a control of the same
model (`abl-B`) is computed here alongside.**

⚠️ Even so, `abl-B` uses round-robin assignment (before D-94) — a confound the
pre-registration wrote down. Number 3 is read **for direction only** and is
not used for a performance judgement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, SplitSet, experiment_shapes, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
KS = (1, 3, 5)


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# -- 1. mixing --------------------------------------------------------------
def obs1(runs: list[Path]) -> None:
    print("=" * 76)
    print("1 ★ does a cross child actually mix the terms of two parents")
    print("=" * 76)
    print("  the criterion: among the proposals that had something to mix, "
          "the fraction that used **the unique terms of both sides**")
    print("        >= 40% it mixes / <= 10% it copies one side\n")
    tot_m = tot_mix = tot_cp = tot_n = 0
    for d in runs:
        rows = _jsonl(d / "cross.jsonl")
        m = [r for r in rows if r["mixable"]]
        mix = [r for r in m if r["mixed"]]
        cp = [r for r in rows if r["copied"]]
        tot_n += len(rows)
        tot_m += len(m)
        tot_mix += len(mix)
        tot_cp += len(cp)
        rate = f"{len(mix) / len(m):.1%}" if m else "—"
        print(f"  {d.name:26s} proposals {len(rows):3d} | could have mixed "
              f"{len(m):3d} | mixed {len(mix):3d} = {rate:>6s} | "
              f"copied whole {len(cp):3d}")
    r = tot_mix / tot_m if tot_m else float("nan")
    print(f"\n  {'total':26s} proposals {tot_n:3d} | could have mixed "
          f"{tot_m:3d} "
          f"| mixed {tot_mix:3d} = {r:.1%} | copied {tot_cp:3d}")
    if not tot_m:
        print("  verdict: ★ sample 0 — there was no proposal with anything "
              "to mix. No judgement is made (principle 27)")
        return
    verdict = ("★ the crossover really runs" if r >= 0.40
               else "★ the prompt is not getting through — it stops before "
                    "performance is looked at"
               if r <= 0.10 else "in between — a band the pre-registration "
                                 "does not cover")
    print(f"  verdict: {verdict}")
    # Which terms were dropped — there is a budget, so merging must drop some
    dropped = [len(set(x["a"]) | set(x["b"])) - len(x["child"])
               for x in _all(runs) if x["mixable"]]
    if dropped:
        print(f"  the number of terms dropped when merging: median "
              f"{np.median(dropped):.1f}  "
              f"range {min(dropped)}~{max(dropped)}")


def _all(runs: list[Path]) -> list[dict]:
    return [r for d in runs for r in _jsonl(d / "cross.jsonl")]


def _cell(agg: dict, k: str) -> str:
    v = agg.get(k)
    if not v or not v["n"]:
        return f"{'—':>16}"
    return (f"{v['dup']}/{v['n']}={v['dup'] / v['n']:.0%} "
            f"scored{v['scored']}").rjust(16)


# -- 2. cell occupancy / 4. duplicate rate -----------------------------------
def obs2_4(runs: list[Path], control: list[Path]) -> None:
    print("\n" + "=" * 76)
    print("2  archive cell occupancy   ·   4  duplicate rate per parent kind")
    print("=" * 76)
    print(f"  {'run':26s} {'cells(end)':>11} {'cells(max)':>11}   "
          f"{'exploit':>16} {'explore':>16} {'cross':>16}")
    for label, group in (("cross1", runs), ("control abl-B", control)):
        if not group:
            continue
        print(f"  -- {label} " + "-" * 40)
        for d in group:
            rs = _jsonl(d / "rounds.jsonl")
            if not rs:
                continue
            cells = [x.get("n_cells", 0) for x in rs]
            agg: dict = {}
            for x in rs:
                for k, v in (x.get("by_parent_kind") or {}).items():
                    a = agg.setdefault(k, {"n": 0, "dup": 0, "scored": 0})
                    for kk in a:
                        a[kk] += v.get(kk, 0)
            print(f"  {d.name:26s} {cells[-1]:11d} {max(cells):11d}   "
                  + " ".join(_cell(agg, k) for k in
                             ("exploit", "explore", "cross")))
    print("\n  ⚠️ number 4 is compared only among runs that have "
          "`by_parent_kind` (after D-94). A '—' in the control means it was "
          "not recorded then.")


# -- 3. the ensemble --------------------------------------------------------
def obs3(runs: list[Path], control: list[Path]) -> None:
    print("\n" + "=" * 76)
    print("3 ★ is the ensemble starting to work — the D-42 retest")
    print("=" * 76)
    print("  ⚠️ the old D-42 numbers (width 0.098 / median 1.0817->1.1137) "
          "are from **deleted")
    print("     gpt-5.4 artefacts**. The model differs, so they are not put "
          "side by side.")
    print("  ⚠️ the control uses round-robin assignment (before D-94) — only "
          "the direction is read.\n")
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    shapes = experiment_shapes(table)
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")
    train = list(splits.train.shapes)

    def fit_per_regime(code, w0):
        fn = compile_rule(code)
        out = {}
        for name in ("short", "long"):
            g = [p for p in train if regime_of(p, table.hw) == name]
            out[name] = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                                    w0, max_evals=300,
                          objective="regret").w
        return fn, out

    def ensemble(fitted: list) -> float:
        regs = []
        for p in held:
            reg = regime_of(p, table.hw)
            cand = table.candidates(p)
            ranks = np.zeros(len(cand.tiebreak), dtype=float)
            for fn, ws in fitted:
                sc = make_score_of(fn, matrix, ws[reg])(p, cand)
                # ★ Tie-preserving ranks (D-41). argsort(argsort(.)) will not
                #   do
                ranks += np.unique(sc, return_inverse=True)[1].astype(float)
            pick = cand.top_k(ranks, 1)[0]
            t = table.times_of(p)
            regs.append(float(t[pick] / t.min()))
        return geomean(np.array(regs))

    for label, group in (("cross1 (two parents)", runs),
                         ("control abl-B (one parent)", control)):
        if not group:
            continue
        print(f"  -- {label} " + "-" * 34)
        ens: dict = {k: [] for k in KS}
        single = []
        for d in group:
            arc = sorted(_jsonl(d / "archive.jsonl"),
                         key=lambda e: e["regret"])[:max(KS)]
            if not arc:
                continue
            fitted = [fit_per_regime(e["code"], e["w"]) for e in arc]
            h = canonical_score(arc[0]["code"], arc[0]["w"], table=table,
                                matrix=matrix, splits=splits).holdout
            single.append(h)
            row = []
            for k in KS:
                v = ensemble(fitted[:k])
                ens[k].append(v)
                row.append(f"{v:.4f}")
            print(f"  {d.name:26s} k=1 single {h:.4f} | ensemble "
                  + "  ".join(f"k={k} {x}" for k, x in zip(KS, row,
                                                           strict=True)),
                  flush=True)
        if single:
            print(f"  {'width (max-min)':26s} single "
                  f"{max(single) - min(single):.4f} | "
                  + "  ".join(f"k={k} {max(ens[k]) - min(ens[k]):.4f}"
                              for k in KS))
            print(f"  {'median':26s} single {np.median(single):.4f} | "
                  + "  ".join(f"k={k} {np.median(ens[k]):.4f}" for k in KS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--control", nargs="*", default=[])
    ap.add_argument("--skip-ensemble", action="store_true",
                    help="number 3 runs a lot of fitting — for looking at "
                         "1·2·4 only")
    a = ap.parse_args()
    runs = [Path("runs") / x for x in a.runs]
    ctl = [Path("runs") / x for x in a.control]
    missing = [d for d in runs + ctl if not d.exists()]
    if missing:
        raise SystemExit("runs that do not exist: "
                         + ", ".join(str(x) for x in missing))
    obs1(runs)
    obs2_4(runs, ctl)
    if not a.skip_ensemble:
        obs3(runs, ctl)


if __name__ == "__main__":
    main()
