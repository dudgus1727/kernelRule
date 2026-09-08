"""★ Is the wall the budget's fault — do the two objectives use overlapping
axes? 0 LLM calls.

    python3 experiments/feature_overlap.py

## What is asked

Do the `regret`-evolved rule and the rank-loss-evolved rule **use the same
axes**?

    union <= 8    8 can hold both -> the wall is not the budget's fault
    union >  8    the budget is short -> 16 might break through

## ⚠️ The number of axes is **not** the budget unit

The budget counts `numeric literals + the number of weights`
(`rules/checks.py`). One term can use two axes —

    np.where(p.is_memory_bound, f.log_dram_traffic, f.log_inst_total) * w[4]

holds 3 axes (2 features + 1 predicate) in 1 weight. So **even if the union
of axes goes over 8 it can fit in 8 terms.** The decision line is read only
as a proxy for "does it fit comfortably", and the number of terms is counted
alongside.

## ⚠️ A Jaccard cannot be read without a floor (principle 7)

"the two families overlap by 0.5" is on its own neither large nor small.
Three things are reported together: the Jaccard **within** a family (how
alike the same objective is to itself), the Jaccard **between** families, and
**the random floor** (sets of the same size drawn at random from the
registry).

⚠️ 2026-09-08 (D-146): **the three pair labels stay in Korean.** They are the
row names of `docs/artifacts/feature-overlap.md` and the keys of
`feature-overlap.json`, and `docs/` is not translated.
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from itertools import combinations, product
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.features import REGISTRY

REG_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
RANK_RUNS = [f"x-rank-rankevo-s{i}" for i in range(3)]
#: The ones that moved a lot inside the top 100 in the ceiling measurement
#: (ranking-ceiling.md §3).
WATCH = ("split_k_cost", "sm_idle_cost", "pipeline_warmup_frac",
         "tail_waste", "waves")
N_DRAWS = 2000


def _best(run: str, by: str) -> dict:
    arc = [json.loads(x) for x in
           (Path("runs") / run / "archive.jsonl").read_text().splitlines()
           if x.strip()]
    key = (lambda e: e.get("rank_loss", 1e9)) if by == "rank" \
        else (lambda e: e["regret"])
    return sorted(arc, key=key)[0]


def _feats(code: str) -> set[str]:
    """It counts `f.<name>` only — `p.<name>` (the predicates) separately."""
    return {n.attr for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "f"}


def _preds(code: str) -> set[str]:
    return {n.attr for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "p"}


def _n_terms(code: str) -> int:
    return max([n.slice.value for n in ast.walk(ast.parse(code))
                if isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id == "w"
                and isinstance(n.slice, ast.Constant)] + [-1]) + 1


def _jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def _floor(sizes_a, sizes_b, pool: int, rng) -> tuple[float, float]:
    """★ The random floor — sets of the same size drawn at random from the
    registry."""
    js, us = [], []
    for _ in range(N_DRAWS):
        na = int(rng.choice(sizes_a))
        nb = int(rng.choice(sizes_b))
        a = set(rng.choice(pool, size=na, replace=False))
        b = set(rng.choice(pool, size=nb, replace=False))
        js.append(_jac(a, b))
        us.append(len(a | b))
    return float(np.mean(js)), float(np.mean(us))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/feature-overlap.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)

    fam = {"regret": [(r, _best(r, "regret")) for r in REG_RUNS],
           "rank": [(r, _best(r, "rank")) for r in RANK_RUNS]}
    F = {k: [_feats(e["code"]) for _, e in v] for k, v in fam.items()}
    P = {k: [_preds(e["code"]) for _, e in v] for k, v in fam.items()}
    TERMS = {k: [_n_terms(e["code"]) for _, e in v] for k, v in fam.items()}
    pool = len(REGISTRY.names()) if hasattr(REGISTRY, "names") \
        else len(list(REGISTRY))

    print("=" * 80)
    print("the axes the two objectives use — is the wall the budget's fault")
    print("=" * 80)
    print(f"  the registry has {pool} axes (F3)")
    print(f"  regret evolution {len(REG_RUNS)} runs / rank evolution "
          f"{len(RANK_RUNS)} runs — **one final best rule** from each run\n")

    print(f"  {'run':22s} {'trm':>3} {'ax':>3} {'pred':>4}  axis names")
    for k in ("regret", "rank"):
        for (r, _), f, p, t in zip(fam[k], F[k], P[k], TERMS[k], strict=True):
            print(f"  {r.replace('f1pipe-F3-', ''):22s} {t:3d} {len(f):3d} "
                  f"{len(p):4d}  {', '.join(sorted(f))}")
        print()

    uni = {k: set().union(*F[k]) for k in F}
    both = uni["regret"] | uni["rank"]
    print("  the family union")
    print(f"    regret {len(uni['regret']):2d}  "
          f"{', '.join(sorted(uni['regret']))}")
    print(f"    rank   {len(uni['rank']):2d}  {', '.join(sorted(uni['rank']))}")
    print(f"    ★ both {len(both):2d}   "
          f"jaccard {_jac(uni['regret'], uni['rank']):.3f}")
    print(f"    rank only {sorted(uni['rank'] - uni['regret'])}")
    print(f"    regret only {sorted(uni['regret'] - uni['rank'])}")
    print("    ⚠️ the run counts are 6 against 3, so the union sizes cannot "
          "be put side by side —")
    print("       it is read from the **rule pairs** below.")

    print("\n" + "=" * 80)
    print("rule-pair jaccard — ★ the floor and the within-family value are "
          "put beside it (principle 7)")
    print("=" * 80)
    rows = {}
    for lab, pairs in (
            ("within regret", list(combinations(range(len(F["regret"])), 2))),
            ("within rank", list(combinations(range(len(F["rank"])), 2))),
            ("★ between families", None)):
        if pairs is None:
            ps = [(F["regret"][i], F["rank"][j]) for i, j
                  in product(range(len(F["regret"])), range(len(F["rank"])))]
        else:
            key = "regret" if "regret" in lab else "rank"
            ps = [(F[key][i], F[key][j]) for i, j in pairs]
        js = [_jac(x, y) for x, y in ps]
        us = [len(x | y) for x, y in ps]
        rows[lab] = {"jaccard": js, "union": us}
        print(f"  {lab:14s} n={len(js):3d}  jaccard median "
              f"{np.median(js):.3f} "
              f"({min(js):.3f}~{max(js):.3f})   union median "
              f"{np.median(us):.1f} ({min(us)}~{max(us)})")
    fj, fu = _floor(([len(x) for x in F["regret"]]),
                    ([len(x) for x in F["rank"]]), pool, rng)
    print(f"  {'★ random floor':14s} n={N_DRAWS}  jaccard mean {fj:.3f}"
          f"                     union mean {fu:.1f}")

    print("\n" + "=" * 80)
    print("the verdict — the line nailed down in the instruction (union <= 8 "
          "means it is not the budget's fault)")
    print("=" * 80)
    u = rows["★ between families"]["union"]
    med, n_le = float(np.median(u)), sum(1 for x in u if x <= 8)
    print(f"  the between-family axis union, median {med:.1f}  "
          f"({n_le}/{len(u)} pairs are 8 or under)")
    print("  -> " + ("★ 8 can hold both — the wall is not the budget's fault"
                     if med <= 8 else
                     "★ the axes go over 8 — there is room for the budget to "
                     "break through"))
    # ★ The decision line is moved onto the budget unit. A real rule holds
    #   several axes in one term — dividing by that density is what gives
    #   "how many terms are needed".
    allf = F["regret"] + F["rank"]
    allp = P["regret"] + P["rank"]
    allt = TERMS["regret"] + TERMS["rank"]
    dens = [(len(f) + len(q)) / t for f, q, t
            in zip(allf, allp, allt, strict=True)]
    ub = [len(x | y) + len(p | q) for (x, p), (y, q) in product(
        zip(F["regret"], P["regret"], strict=True),
        zip(F["rank"], P["rank"], strict=True))]
    need = [u / d for u, d in product(ub, [float(np.median(dens))])]
    print("\n  ⚠️ the number of axes != the budget unit. The term counts are "
          "all 8 and one term holds")
    print(f"     {min(dens):.2f}~{max(dens):.2f} axes (median "
          f"{np.median(dens):.2f}) — that counts the predicates too.")
    print(f"     At that density, holding the between-family union "
          f"(axes+predicates {int(min(ub))}~{int(max(ub))}) needs "
          f"**{np.median(need):.1f} terms**.")
    print(f"     -> 8 is {'short' if np.median(need) > 8 else 'comfortable'}, "
          f"16 is {'comfortable' if np.median(need) <= 16 else 'short'}. "
          f"The shortfall is about {max(0.0, np.median(need) - 8):.1f} terms.")

    print("\n  ★ are the ceiling measurement's five axes **in the regret "
          "rules too**")
    print(f"    {'':24s} {'regret':>8} {'rank':>6}")
    watch = {}
    for n in WATCH:
        cr = sum(n in f for f in F["regret"])
        ck = sum(n in f for f in F["rank"])
        watch[n] = [cr, ck]
        print(f"    {n:24s} {cr:5d}/{len(F['regret'])} {ck:4d}/{len(F['rank'])}")

    Path(a.out).write_text(json.dumps(
        {"features": {k: [sorted(x) for x in v] for k, v in F.items()},
         "preds": {k: [sorted(x) for x in v] for k, v in P.items()},
         "terms": TERMS, "union": {k: sorted(v) for k, v in uni.items()},
         "pairs": rows, "floor": {"jaccard": fj, "union": fu},
         "watch": watch, "pool": pool,
         "density": dens, "union_both": ub, "terms_needed": need}, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
