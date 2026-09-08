"""★ Stage 1 — does the fix work? It refits the 12 stored rules. 0 LLM calls.

    python3 experiments/fitter_movement.py

**The criterion: a reach rate of 90% or more** (approved 2026-08-26, D-56).
It used to be "a fit-movement rate of 90%", but that metric **counts a
failure even when the starting point is already optimal** — it was the
measuring instrument that was wrong, not the thing measured. The movement
rate is still recorded as a diagnostic.

```
movement rate  did it leave the starting point         the process
reach rate     did it end up worse than somewhere      the result
               within arm's reach                      <- "did it do its job"
```

**Do not read the reach rate as "the fitter finds the optimum".** 4000 random
points in 8 dimensions are sparse, and there is no guarantee that log-uniform
0.05~50 covers the real optimum. The accurate statement is "the fitter ends
up somewhere worse than a 4000-point random search in N% of cases".

The **absolute value** of regret **is not reported.** It is a number marked
for retirement (D-56 §2).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core import weights as W
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
#: The polish budget. The cost cap written in the pre-registration is twice
#: the fit's 305 (610), and the default is 600. It was pulled out so it can be
#: raised by an argument — **a value measured with it raised is not a pass**
#: (D-59). It is reported together with the fact that the cap was exceeded.
POLISH_BUDGET = int(__import__("os").environ.get("KERNELRULE_POLISH_BUDGET",
                                                 "600"))
TARGET_MOVED = 0.90
TARGET_REACH = 0.90
#: The random probe. The sign is positive (every feature is 'bigger is worse')
#: and the magnitude is log-uniform over a factor of 1000.
N_PROBE = 4000
PROBE_LO, PROBE_HI = 0.05, 50.0


def _condition_of(run: str) -> str:
    """Which condition this run is — were the feature descriptions given?

    If `runs/*/config.json` exists it is read from there. **The 12
    luna/lunaNAMES runs ran before `dump()` wrote config.json, so they have
    none** — for those it falls back to the name prefix. That is a guess, so
    the fact is said on screen (§26.4).
    """
    cfg = Path("runs") / run / "config.json"
    if cfg.exists():
        d = json.loads(cfg.read_text())
        det = (d.get("llm") or {}).get("feature_detail") or d.get("feature_detail")
        if det:
            return "A descriptions" if det == "full" else "B names only"
    return ("B names only" if run.startswith("lunaNAMES")
            else "A descriptions")


def main() -> None:
    warnings.simplefilter("ignore")
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    # ★ The default is the 12 committed rules, but it also takes arbitrary
    #   runs — putting a pass condition on a verification run means looking at
    #   those runs (D-60).
    import sys
    args = [x for x in sys.argv[1:] if not x.startswith("-")]
    if args:
        index = [{"run": r} for r in args]
        print(f"  target: the {len(index)} runs given as arguments {args}\n")
    else:
        index = json.loads(
            Path("docs/artifacts/rules/index.json").read_text())

    print("=" * 76)
    print(f"stage 1 — the fitter's quality. The pass line: ★ a reach rate of "
          f"{TARGET_REACH:.0%} (approved in D-56)")
    if POLISH_BUDGET != 600:
        print(f"  ★ polish budget {POLISH_BUDGET} — if it went over the "
              "pre-registration cap of 610, this result **is not a pass** "
              "(D-59)")
    print("=" * 76)
    print(f"  {'rule':16s} {'regime':6s} {'off':>6} {'on':>6} "
          f"{'random wins':>13} {'reach':>6}")

    n = n_off = n_on = n_probe_beats = n_reach = 0
    gaps: list[float] = []
    #: condition -> [n reached, total]. ★ Whether the failures cluster in one
    #: condition changes the re-run design (D-60) — if they cluster, that
    #: condition can simply be avoided.
    by_cond: dict[str, list[int]] = {}
    named = any((Path("runs") / r["run"] / "config.json").exists()
                for r in index)
    for row in index:
        run = row["run"]
        with (Path("runs") / run / "archive.jsonl").open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        best = min(arc, key=lambda e: e["regret"])
        fn = compile_rule(best["code"])
        for name in ("short", "long"):
            g = [p for p in train if regime_of(p, table.hw) == name]
            sp = Split("train", tuple(g))
            # ★ `polish` is **stated explicitly.** After the default flipped
            #   to True this line was left out, so the "polish off" column was
            #   really on, and the two columns both came out at 62.5%
            #   (principle 1 — judging silently to one side).
            off = fit_weights(fn, matrix, table, sp, best["w"], max_evals=300,
                              warn_invariants=False, polish=False,
                          objective="regret")
            on = fit_weights(fn, matrix, table, sp, best["w"], max_evals=300,
                             warn_invariants=False, polish=True,
                             polish_budget=POLISH_BUDGET,
                          objective="regret")
            prob = W._Problem(matrix, table, tuple(g), 1)
            rng = np.random.default_rng(7)
            bv = np.inf
            w0 = np.asarray(best["w"], dtype=np.float64)
            for _ in range(N_PROBE):
                c = np.exp(rng.uniform(np.log(PROBE_LO), np.log(PROBE_HI),
                                       size=w0.size))
                v = prob.regret(fn, c)
                if np.isfinite(v) and v < bv:
                    bv = v
            beat = bv < on.fit_regret - 1e-9
            n += 1
            n_off += off.moved
            n_on += on.moved
            n_probe_beats += beat
            if beat:
                gaps.append(on.fit_regret - bv)
            n_reach += not beat
            c = _condition_of(run)
            by_cond.setdefault(c, [0, 0])
            by_cond[c][0] += not beat
            by_cond[c][1] += 1
            print(f"  {run:16s} {name:6s} "
                  f"{'moved' if off.moved else 'still':>6} "
                  f"{'moved' if on.moved else 'still':>6} "
                  f"{('★ ' + format(bv - on.fit_regret, '+.4f')) if beat else '—':>13} "
                  f"{'—' if beat else 'OK':>6}")

    print()
    print(f"  movement rate  polish off {n_off:2d}/{n} = {n_off / n:5.1%}")
    print(f"  movement rate  polish on  {n_on:2d}/{n} = {n_on / n:5.1%}   "
          f"(diagnostic)")
    r = n_reach / n
    print(f"  ★ reach rate {n_reach:2d}/{n} = {r:5.1%}   "
          f"{'pass' if r >= TARGET_REACH else '★ short (the pass line is 90%)'}")
    print(f"          = the fraction where {N_PROBE} random points fail to "
          f"beat the fit result")
    print()
    print("  the reach rate per condition — ★ do the failures cluster (D-60)")
    for c in sorted(by_cond):
        ok, tot = by_cond[c]
        print(f"    {c:12s} {ok:2d}/{tot} = {ok / tot:6.1%}")
    if not named:
        print("    ※ the condition was guessed from the **run name prefix** — "
              "these runs have no config.json")
    if gaps:
        print(f"\n  the gaps of the {len(gaps)} reach failures: "
              f"{', '.join(f'{g:.4f}' for g in sorted(gaps, reverse=True))}")
        print(f"  the largest {max(gaps):.4f} — "
              + ("negligible (<0.005)" if max(gaps) < 0.005 else
                 "★ the fitter really is failing to find it. Consider an "
                 "alternative"))


if __name__ == "__main__":
    main()
