"""★ The per-round **final scoring** curve — "how many rounds are needed"
from the data. 0 LLM calls.

    python3 experiments/round_curve.py                # new runs + old runs
    python3 experiments/round_curve.py --group F3rw-p8

## Why

Both `12` and `patience 3` were values set without a ground. And stopping on
the loop's internal score (`best_val_regret`) stops **while the final scoring
is still rising** (D-131 — that is how +0.0187 was lost).

**So the curve of the final scoring itself is produced.**

```
from the archive up to each round r, take **the training-score best**
-> refit per regime on the training split -> score on the 20 holdout shapes
= the value that would have been reported if it had stopped at that round
```

## The verdict — nailed down in advance in the pre-registration (D-132 §2)

```
flat within σ(0.0113) from rN on   ★ N is the number of rounds needed
it keeps rising to the cap         ★ the cap is not enough either — that
                                     fact is written down
```

★ 0 LLM calls. It refits once per round per seed, so it takes tens of
minutes.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import warnings
from pathlib import Path

from sigma_5090 import _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
#: (group name, run prefix, number of seeds, note)
GROUPS = [
    ("F3rw-p8", "F3rw-p8", 6,
     "new — early stopping off, 24 rounds (D-132)"),
    ("F3rw-p8-old", "F3rw-p8-old", 6, "old — 12 rounds, patience 10"),
    # ★ The campaign D-131 measured the prompt effect on (it stopped at
    #   r4~r6 with patience 3)
    ("F3rw-p8-p3", "F3rw-p8-p3", 6,
     "new — patience 3, ended at r4~r6"),
    # ★ The representative re-measurement after D-136 — only the __import__
    #   fix differs from F3rw-p8
    ("F3rw-p8-nan", "F3rw-p8-nan", 6,
     "new — the __import__ fix, 24 rounds"),
]
#: The seed spread used to judge flatness. **It is not set anew here**
#: (principle 7).
SIGMA = 0.0113


def _rows(run: str, name: str) -> list[dict]:
    p = Path("runs") / run / name
    return ([json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            if p.exists() else [])


def _series(run: str) -> tuple[list[dict], str]:
    """The 'best at the time' per round. **It returns the source alongside.**

    ★ `bests.jsonl` is the right answer — one line per round, recording that
    round's best down to the code and the weights.

    ⚠️ `archive.jsonl` is **the final archive state** (`loop.py:1351`,
    `archive.dump`). Picking "the best among those with round <= r" there
    shows **only what survived to the end** — an elite pushed out along the
    way is missing, so an earlier round comes out worse than it really was
    (or empty, as a nan). **It is accurate only at the last round.** It is
    used only for old runs that have no `bests.jsonl` (D-139).
    """
    bs = _rows(run, "bests.jsonl")
    if bs:
        return sorted(bs, key=lambda e: e["round"]), "bests"
    arc = _rows(run, "archive.jsonl")
    n_rounds = max(e.get("round", 0) for e in arc) + 1
    out = []
    for r in range(n_rounds):
        sub = [e for e in arc if e.get("round", 99) <= r]
        out.append(sorted(sub, key=lambda e: e["regret"])[0] if sub
                   else {"code": None, "w": None, "round": r})
    return out, "archive-snapshot"


def _curve(run: str, T, M, sp) -> tuple[list[float], str]:
    """The 'value that would have been reported if it had stopped there' per
    round."""
    series, src = _series(run)
    out, prev_code = [], None
    for b in series:
        if b["code"] is None:
            out.append(float("nan"))
            continue
        key = (b["code"], tuple(b["w"]))
        if key == prev_code:      # ★ if the best did not change, no remeasure
            out.append(out[-1])
            continue
        prev_code = key
        out.append(canonical_score(b["code"], b["w"], table=T, matrix=M,
                                   splits=sp).holdout)
    return out, src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append")
    ap.add_argument("--out", default="docs/artifacts/round-curve.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M, sp = FeatureMatrix(T, REGISTRY), None
    sp = _splits(T)
    out: dict = {"sigma": SIGMA, "groups": {}}

    for name, prefix, n, note in GROUPS:
        if a.group and name not in a.group:
            continue
        print("=" * 92)
        print(f"{name}  — {note}")
        print("=" * 92)
        curves, srcs = {}, set()
        for i in range(n):
            run = f"{prefix}-s{i}"
            if not (Path("runs") / run / "archive.jsonl").exists():
                print(f"  ⚠️ missing run: {run}")
                continue
            c, src = _curve(run, T, M, sp)
            srcs.add(src)
            curves[run] = c
            print(f"  {run:18s} " + " ".join(f"{x:.4f}" for x in c),
                  flush=True)
        if not curves:
            continue
        L = max(len(c) for c in curves.values())
        med = [st.median([c[min(r, len(c) - 1)] for c in curves.values()])
               for r in range(L)]
        print(f"\n  {'median':18s} " + " ".join(f"{x:.4f}" for x in med))
        # ★ Where it flattens — from here to the end it is within σ
        final = med[-1]
        flat = next((r for r in range(L)
                     if all(abs(med[k] - final) < SIGMA for k in range(r, L))),
                    None)
        print(f"  ★ the round from which it stays within σ({SIGMA}) of the "
              f"final value: "
              + (f"r{flat}" if flat is not None else "none"))
        print(f"  ★ the improvement up to the last round (r0 -> end): "
              f"{med[0] - final:+.4f}   "
              f"the last 4 rounds' improvement: "
              f"{med[max(0, L - 5)] - final:+.4f}")
        print(f"  ★ source: {sorted(srcs)}"
              + ("   ⚠️ archive-snapshot is **accurate only at the last "
                 "round**"
                 if "archive-snapshot" in srcs else ""))
        out["groups"][name] = {"curves": curves, "median": med,
                               "source": sorted(srcs),
                               "flat_from": flat, "note": note}

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                      default=float))
    print(f"\n  -> {a.out}")
    print("  ★ the verdict is made only by the two branches of the "
          "pre-registration (D-132 §2)")


if __name__ == "__main__":
    main()
