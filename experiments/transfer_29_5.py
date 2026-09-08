"""★ §29.5 structural transfer — (a) full transplant / (b) refit /
(c) regrow. 0 LLM calls.

    python3 experiments/transfer_29_5.py --pair a6000 5090
    python3 experiments/transfer_29_5.py --pair a6000 4090   # when the table
                                                             # comes

The **pre-registration** is `docs/artifacts/transfer-prereg.md` + the
addendum.

## The three arms

```
(a) full transplant   the A6000 structure + the A6000 weights as they are
                      -> scored on the 5090
(b) refit             the structure fixed, only the weights done again on
                      the 5090 training split
(c) regrow            from scratch on the 5090 table   <- already run
                      (5090sigma, 3 seeds)
```

## ★ One rule is not picked (principle 5)

Picking one "rule evolved on the A6000" makes that choice an optimism bias.
The training-best rule of **all 6 seeds of the A6000 F3 campaign** is moved
over and the distribution is reported.

## ★ The baseline — the pre-registration's fallback is used

`nvMatmulHeuristics` is not in this environment (the import fails). As the
pre-registration wrote down in advance, **`human_guided` refitted on the
5090** is used. It was not picked after seeing the results.

## ★ The flipped shapes both included and excluded

2 of the 20 holdout shapes are bound flips. Reporting only one side would be
a choice.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.crosstable import bound_flipped, common_shapes
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.runset import assert_same_condition
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

#: ★ The table registry. **A new GPU only needs one line here**
#: (principle 2).
#:   `runs` is the campaign run **from scratch** on that table = (c) regrow,
#:   and the same list is also used as the **source** of (a)(b) in another
#:   pair.
#: ★ (c) is once per GPU and (a)(b) are free however many combinations there
#: are.
TABLES: dict[str, dict] = {
    "a6000": {
        "bundle": "datasets/rtx-a6000-sm_86-c63710df",
        "env_hash": "c63710df",
        # ★ 2026-09-06 (D-141): the representative value changed to
        #   `F3rw-p8-nan` and the rounds are 12, so **the rule at r11** is
        #   used (D-140).
        #   ⚠️ `F3rw-p8-s{i}` used to be written here, and the rename made
        #      that name point at a different campaign — **the artefacts do
        #      not record which campaign** the old transfer numbers came
        #      from. That is why `src_runs` below is recorded.
        "runs": [f"F3rw-p8-nan-s{i}" for i in range(6)],
        "round": 11},
    "5090": {
        "bundle": "datasets/rtx-5090-sm_120-5bb6f403",
        "env_hash": "5bb6f403",
        # ★ 6 (c) regrow runs. The first 3 and the last 3 use **the same
        #   stage-2 seed** (`--seed-from`) — otherwise the seed rule differs
        #   and so does the condition (D-84). The sample unit is the run
        #   (principle 28).
        # ⚠️ 2026-09-03 correction (D-119): the last three (`-b-`) are the
        #    `human_guided` **hand seed**, so they are not "from scratch on
        #    the 5090". They are taken out of (c). The old value 1.0485 is
        #    the median of those six and it is kept in `c-ladder.md` §0.
        # ★ 2026-09-04 correction (D-126): it was changed to **the ladder's
        #    new (c)**. `5090sigma-s*` had both the hw numbers and the
        #    warnings from the A6000 (D-113·116). `5090sigma-hw2-s*` is the
        #    one that got the 5090 hw generated from the bundle, and it is
        #    **the same version** as the 4090 (c). The old values are not
        #    deleted — they are still in `transfer-29-5.md` and
        #    `c-ladder.md`.
        "runs": [f"F3rw-p8-5090-s{i}" for i in range(3)]},
    "4090": {
        "bundle": "datasets/rtx-4090-sm_89-ad95d455",
        "env_hash": "ad95d455",
        # ★ 3 (c) regrow seeds. **The command is the same** as the 5090's new
        #   (c) (only the table differs).
        "runs": [f"F3rw-p8-4090-s{i}" for i in range(3)]},
    "h100": {
        "bundle": "datasets/h100-nvl-sm_90-63684546",
        "env_hash": "63684546",
        # ★ There is **no (c)**. It can be used only as a destination — as a
        #   source it is refused.
        "runs": []},
}


def _splits(table: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def _best(run: str, rnd: int | None = None) -> dict:
    """One by the training score. **It does not look at the holdout**
    (§10.2).

    If `rnd` is given, **the best of that round** is read from
    `bests.jsonl`. `archive.jsonl` is the final state, so it cannot give an
    intermediate round (D-139).
    """
    if rnd is not None:
        f = Path("runs") / run / "bests.jsonl"
        if not f.exists():
            raise SystemExit(
                f"{f} does not exist — the round cannot be specified (D-139)")
        at = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        at = [e for e in at if e["round"] <= rnd]
        if not at:
            raise SystemExit(f"{run} has no best at or below r{rnd}")
        return max(at, key=lambda e: e["round"])
    f = Path("runs") / run / "archive.jsonl"
    arc = sorted((json.loads(x) for x in f.read_text().splitlines()
                  if x.strip()), key=lambda e: e["regret"])
    return arc[0]


def _fit_per_regime(code, w0, table, matrix, train):
    fn = compile_rule(code)
    out = {}
    for name in ("short", "long"):
        g = [p for p in train if regime_of(p, table.hw) == name]
        out[name] = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                                w0, max_evals=300,
                          objective="regret").w
    return fn, out


def _score_on(fn, ws, table, matrix, shapes) -> float:
    """Scores the shapes with the per-regime weights. **The regime is judged
    with this table's hardware.**"""
    regs = []
    for p in shapes:
        reg = regime_of(p, table.hw)
        cand = table.candidates(p)
        sc = make_score_of(fn, matrix, ws[reg])(p, cand)
        pick = cand.top_k(sc, 1)[0]
        t = table.times_of(p)
        regs.append(float(t[pick] / t.min()))
    return geomean(np.array(regs))


def _row(label: str, xs: list[float]) -> str:
    a = np.array(xs)
    return (f"  {label:34s} median {np.median(a):.4f}  "
            f"range {a.min():.4f}~{a.max():.4f}  "
            f"width {a.max() - a.min():.4f}"
            f"  (n={len(a)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, default=["a6000", "5090"],
                    metavar=("SRC", "DST"),
                    help=f"two table names. Registered: {sorted(TABLES)}")
    ap.add_argument("--out", default=None,
                    help="the default is "
                         "docs/artifacts/transfer-<src>-<dst>.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    src, dst = a.pair
    for n in (src, dst):
        if n not in TABLES:
            raise SystemExit(
                f"the table {n!r} is not registered. Put one line in TABLES. "
                f"What is there now: {sorted(TABLES)}")
    if src == dst:
        raise SystemExit("the same table to itself is not a transfer.")
    S, D = TABLES[src], TABLES[dst]
    SRC_RUNS, DST_RUNS = S["runs"], D["runs"]
    S_RND, D_RND = S.get("round"), D.get("round")
    if not SRC_RUNS:
        raise SystemExit(
            f"{src} has no (c) regrow run — **it cannot be used as a "
            f"source**. The structure has to come from there.")
    if not DST_RUNS:
        print(f"  ⚠️ {dst} has **no** (c) regrow — only (a)(b) are reported")
    # ★ Whether the set has a single condition is looked at **here** (D-120).
    #   Looking later turns it into "it was there and we did not look"
    #   (principle 39).
    assert_same_condition(SRC_RUNS, label=f"{a.pair[0]} (c) regrow")
    assert_same_condition(DST_RUNS, label=f"{a.pair[1]} (c) regrow")
    missing = [r for r in SRC_RUNS + DST_RUNS
               if not (Path("runs") / r / "archive.jsonl").exists()]
    if missing:
        raise SystemExit(
            "runs that do not exist: " + ", ".join(missing)
            + "\n  ★ skipping silently draws a conclusion without knowing "
              "the sample shrank.")
    out = a.out or f"docs/artifacts/transfer-{src}-{dst}.json"

    A = PerfTable.from_bundle(S["bundle"], env_hash=S["env_hash"],
                              ok_only=False)
    B = PerfTable.from_bundle(D["bundle"], env_hash=D["env_hash"],
                              ok_only=False)
    mA, mB = FeatureMatrix(A, REGISTRY), FeatureMatrix(B, REGISTRY)
    spA, spB = _splits(A), _splits(B)

    flip = {(p.M, p.N, p.K) for p, _, _ in bound_flipped(A, B)}
    common = {(p.M, p.N, p.K) for p in common_shapes(A, B)}
    hold = [p for p in spB.val.shapes if (p.M, p.N, p.K) in common]
    hold_nf = [p for p in hold if (p.M, p.N, p.K) not in flip]

    print("=" * 78)
    print(f"§29.5 structural transfer  {src} -> {dst}")
    print("=" * 78)
    print(f"  ridge {A.hw.ridge_point:.1f} -> {B.hw.ridge_point:.1f}  "
          f"({B.hw.ridge_point / A.hw.ridge_point:.2f}x)   "
          f"SM {A.hw.sm_count} -> {B.hw.sm_count}")
    # ★ It differs per pair. Hardcoding it makes it **lie** on another pair.
    print(f"  holdout {len(hold)} shapes (within the {len(common)} common)   "
          f"flips excluded {len(hold_nf)} shapes")
    print(f"  {dst} training {len(spB.train.shapes)} shapes — used for the "
          f"(b) refit")
    print("  ★ the baseline: no nvMatmulHeuristics -> the pre-registration's"
          f" fallback ({dst}-refitted human_guided)\n")

    res: dict = {"a": [], "b": [], "c": [], "a_nf": [], "b_nf": [],
                 "c_nf": [], "src": []}

    for run in SRC_RUNS:
        e = _best(run, S_RND)
        fn = compile_rule(e["code"])
        # (a) the per-regime weights fitted on the A6000, **as they are**
        _, wsA = _fit_per_regime(e["code"], e["w"], A, mA,
                                 list(spA.train.shapes))
        va = _score_on(fn, wsA, B, mB, hold)
        va_nf = _score_on(fn, wsA, B, mB, hold_nf)
        # (b) only the weights done again on the 5090 training split
        _, wsB = _fit_per_regime(e["code"], e["w"], B, mB,
                                 list(spB.train.shapes))
        vb = _score_on(fn, wsB, B, mB, hold)
        vb_nf = _score_on(fn, wsB, B, mB, hold_nf)
        res["src"].append(run)
        res["a"].append(va)
        res["a_nf"].append(va_nf)
        res["b"].append(vb)
        res["b_nf"].append(vb_nf)
        print(f"  {run:26s} (a) {va:.4f}   (b) {vb:.4f}   "
              f"[flips excluded {va_nf:.4f} / {vb_nf:.4f}]", flush=True)

    print()
    for run in DST_RUNS:
        e = _best(run, D_RND)
        fn, ws = _fit_per_regime(e["code"], e["w"], B, mB,
                                 list(spB.train.shapes))
        vc = _score_on(fn, ws, B, mB, hold)
        vc_nf = _score_on(fn, ws, B, mB, hold_nf)
        res["c"].append(vc)
        res["c_nf"].append(vc_nf)
        print(f"  {run:26s} (c) {vc:.4f}"
              f"                 [flips excluded {vc_nf:.4f}]", flush=True)

    from kernelrule.rules.human_guided import CODE as PS_CODE
    from kernelrule.rules.human_guided import W0 as PS_W0
    fn, ws = _fit_per_regime(PS_CODE, list(PS_W0), B, mB,
                             list(spB.train.shapes))
    base = _score_on(fn, ws, B, mB, hold)
    base_nf = _score_on(fn, ws, B, mB, hold_nf)
    res["baseline"] = base
    res["baseline_nf"] = base_nf

    print("\n" + "=" * 78)
    print(f"holdout {len(hold)} shapes (flips included)")
    print("=" * 78)
    print(_row("(a) full transplant", res["a"]))
    print(_row("(b) refit", res["b"]))
    print(_row("(c) regrow", res["c"]) if res["c"]
          else "  (c) regrow                        ★ none — this table's "
               "native is unmeasured")
    # ★ This differs per pair too (the same mistake as the two places above
    #   is not made a third time)
    print(f"  {f'★ baseline human_guided({dst}-refitted)':34s} {base:.4f}")
    print(f"\nholdout {len(hold_nf)} shapes (flips excluded)")
    print("-" * 78)
    print(_row("(a) full transplant", res["a_nf"]))
    print(_row("(b) refit", res["b_nf"]))
    print(_row("(c) regrow", res["c_nf"]) if res["c_nf"]
          else "  (c) regrow                        ★ none")
    print(f"  {'★ baseline human_guided':34s} {base_nf:.4f}")

    res["pair"] = [src, dst]
    # ★ Which runs were used — the rename made it impossible to tell which
    #   campaign an old artefact came from. That does not happen again
    #   (principle 2, D-141).
    res["src_runs"] = list(SRC_RUNS)
    res["dst_runs"] = list(DST_RUNS)
    res["src_round"] = S_RND
    res["dst_round"] = D_RND
    res["ridge"] = [A.hw.ridge_point, B.hw.ridge_point]
    res["n_holdout"] = [len(hold), len(hold_nf)]
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {out}")
    print("  ⚠️ no significance is attached — the σ confidence interval is "
          "wide (sigma-5090.json)")


if __name__ == "__main__":
    main()
