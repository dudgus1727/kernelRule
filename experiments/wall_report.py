"""★ The three that try to lower the wall — A (the k sweep) · B (the product
term) · C (the λ sweep). 0 LLM calls.

    python3 experiments/wall_report.py

The pre-registration is `docs/artifacts/wall-prereg.md`.

⚠️ **Everything is on the 20 holdout shapes.** The comparison between
families is done with tau at **k fixed to 100** — measuring each run at its
own k changes "what was measured" instead of "what got better" (principle 4).
The value measured at that run's own k is put separately in §A2.

The helper functions come from `two_stage.py` (principle 2).
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
from kernelrule.core.numerics import approx_equal
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import _Problem, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.rules.checks import _numeric_literals

#: (tag, label, k, λ, product hint). `rankevo` is the existing run at
#: k=100 λ=0.
ARMS = [
    ("rankevo", "k=100 (existing)", 100, 0.0, False),
    ("k010", "k=10", 10, 0.0, False),
    ("k020", "k=20", 20, 0.0, False),
    ("k050", "k=50", 50, 0.0, False),
    ("prod", "product stated", 100, 0.0, True),
    ("lam03", "λ=0.3", 100, 0.3, False),
    ("lam10", "λ=1", 100, 1.0, False),
    ("lam30", "λ=3", 100, 3.0, False),
]
A_ARMS = ["k010", "k020", "k050", "rankevo"]
B_ARMS = ["rankevo", "prod"]
C_ARMS = ["rankevo", "lam03", "lam10", "lam30"]
SEEDS = 3
WATCH = ("split_k_cost", "sm_idle_cost", "pipeline_warmup_frac",
         "tail_waste", "waves")


def _rows(d: Path, name: str) -> list[dict]:
    return [json.loads(x) for x in (d / name).read_text().splitlines()
            if x.strip()]


def _best(d: Path) -> dict:
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


def _prod_pairs(code: str) -> list[tuple]:
    """The `f.* x f.*` products. **Products with a weight in them are left
    out** — only feature-to-feature ones are counted."""
    def has_w(x):
        return any(isinstance(m, ast.Name) and m.id == "w"
                   for m in ast.walk(x))

    def fs(x):
        return {m.attr for m in ast.walk(x) if isinstance(m, ast.Attribute)
                and isinstance(m.value, ast.Name) and m.value.id == "f"}

    out = []
    for n in ast.walk(ast.parse(code)):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            a, b = fs(n.left), fs(n.right)
            if a and b and not has_w(n.left) and not has_w(n.right):
                # ★ The two sides are kept **separately**. Folding them into
                #   a union makes a square (`f.a * f.a`) indistinguishable
                #   from a product.
                out.append((tuple(sorted(a)), tuple(sorted(b))))
    return out


def _prompt_k(text: str) -> str:
    """The `k` the system prompt states, read from the saved prompt.

    ⚠️ 2026-09-08 (D-146): the prompts became English, but **the Korean form
    is still matched** — the prompts saved with the old runs are in Korean
    and this reads those artefacts.
    """
    for w in text.split("\n"):
        if "config " in w and "개" in w:               # the old Korean prompt
            return "k=" + w.split("config ")[1].split("개")[0]
        if "fastest configs" in w:                    # the English prompt
            head = w.split("fastest configs")[0].split()
            if head:
                return "k=" + head[-2 if head[-1] == "genuinely" else -1]
    return "?"


def _configs(code, w, table, matrix, shapes) -> int:
    fn, w = compile_rule(code), np.asarray(w, dtype=np.float64)
    picks = []
    for p in shapes:
        cand = table.candidates(p)
        j = int(cand.top_k(make_score_of(fn, matrix, w)(p, cand), 1)[0])
        picks.append((str(cand.kernel_id[j]), int(cand.split_k[j]),
                      str(cand.split_k_mode[j])))
    return len(Counter(picks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/wall.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    A = {t: (lab, k, lam, ph) for t, lab, k, lam, ph in ARMS}
    dirs = {t: [Path("runs") / f"f1pipe-F3-{t}-s{i}" for i in range(SEEDS)]
            for t, *_ in ARMS}
    best = {t: [_best(d) for d in dirs[t]] for t, *_ in ARMS}
    out: dict = {"n_holdout": len(hold)}

    # -------------------------------------------- §0 checking the conditions
    print("=" * 86)
    print("§0  did the condition actually take — read from the artefacts "
          "(D-105/D-107)")
    print("=" * 86)
    print(f"  {'':14s} {'k':>5} {'λ':>5} {'prod':>6}   "
          f"what the system prompt says k is")
    for t, lab, k, lam, ph in ARMS:
        c = json.loads((dirs[t][0] / "config.json").read_text())
        got_k = c["loop"].get("rank_top_k", 100)
        got_l = c["loop"].get("rank_lambda", 0.0)
        f = dirs[t][0] / "llm_calls" / "_system-rule_editor.md"
        says = "(not recorded)" if not f.exists() else _prompt_k(f.read_text())
        ok = "✅" if (got_k == k and approx_equal(got_l, lam)) else "⛔"
        print(f"  {lab:14s} {got_k:5d} {got_l:5.1f} {str(ph):>7}   {says}  {ok}")
        out.setdefault("cond", {})[t] = {"k": got_k, "lambda": got_l,
                                         "prompt": says}

    prob100 = _Problem(M, T, sp.train.shapes, 1)
    prob100.build_pairs(T, 100)

    # ★ The fit is done **once only** per (arm, objective). `rankevo` appears
    #   in all three of A·B·C, and without a cache the same fit would run
    #   several times.
    fits: dict = {}

    def measure(tag: str, obj: str, top_n: int = 100) -> list[tuple]:
        key = (tag, obj)
        if key not in fits:
            _lab, k, lam, _ph = A[tag]
            fits[key] = [_fit(e["code"], e["w"], T, M, train, obj,
                              rank_top_k=k, rank_lambda=lam)
                         for e in best[tag]]
        return [_measure(fn, ws, T, M, hold, top_n) for fn, ws in fits[key]]

    floors: dict = {}

    def floor(top_n: int):
        if top_n not in floors:
            floors[top_n] = _floor(T, hold, top_n)
        return floors[top_n]

    def block(tags, title, top_n=100, objs=("regret", "rank")):
        print(f"\n{'=' * 86}\n{title}\n{'=' * 86}")
        fl = floor(top_n)
        for obj, head in (("regret", ("★ regret refit — the cell the "
                                      "decision line is on")),
                          ("rank", "rank fit (fitted at that arm's k·λ)")):
            if obj not in objs:
                continue
            print(f"\n  --- {head} ---")
            print(f"  {'':14s} {'regret':>8} {f'top-{top_n} tau':>13} "
                  f"{'all':>9}   (tau range)")
            for t in tags:
                v = np.array(measure(t, obj, top_n))
                und = int(v[:, 3].sum())
                print(f"  {A[t][0]:14s} {np.median(v[:, 0]):8.4f} "
                      f"{np.median(v[:, 1]):12.3f} {np.median(v[:, 2]):9.3f}"
                      f"   ({v[:, 1].min():+.3f}~{v[:, 1].max():+.3f})"
                      + (f"  ⚠️ tau undefined in {und} shapes" if und else ""))
                out.setdefault(f"{obj}@{top_n}", {})[t] = [list(x) for x in v]
            print(f"  {'★ random floor':14s} {fl[0]:8.4f} {fl[1]:12.3f} "
                  f"{fl[2]:9.3f}")
        out.setdefault("floor", {})[str(top_n)] = fl

    block(A_ARMS, "A.  the k sweep — the between-family comparison is "
                  "measured at k fixed to 100")

    print("\n" + "=" * 86)
    print("A2.  tau measured at that run's own k — ★ the floor differs per k")
    print("=" * 86)
    print(f"  {'':14s} {'k':>4} {'top-k tau':>11} {'random floor':>13} "
          f"{'above floor':>12}")
    for t in A_ARMS:
        k = A[t][1]
        v = np.array(measure(t, "rank", k))
        f = floor(k)
        und = int(v[:, 3].sum())
        print(f"  {A[t][0]:14s} {k:4d} {np.median(v[:, 1]):10.3f} "
              f"{f[1]:11.3f} {np.median(v[:, 1]) - f[1]:+8.3f}"
              + (f"   ⚠️ {und} shapes undefined (the score is constant)"
                 if und else ""))
        out.setdefault("own_k", {})[t] = {
            "k": k, "tau": [x[1] for x in v], "floor": f[1],
            "undef": und}

    block(B_ARMS, "B.  the product term — does stating it change anything")

    print("\n" + "=" * 86)
    print("B2.  was the product term actually used — the whole archive")
    print("=" * 86)
    print(f"  {'':14s} {'rules w/ prod':>14} {'prods per rule':>15}   "
          f"the frequent pairs")
    for t in B_ARMS:
        arc = [e for d in dirs[t] for e in _rows(d, "archive.jsonl")]
        ps = [_prod_pairs(e["code"]) for e in arc]
        n = sum(1 for x in ps if x)
        cnt = Counter(x for xs in ps for x in xs)
        top = ", ".join(f"{'+'.join(x)} x {'+'.join(y)} ({v})"
                        for (x, y), v in cnt.most_common(3))
        print(f"  {A[t][0]:14s} {n:5d}/{len(arc):<4d} "
              f"{np.mean([len(x) for x in ps]):10.2f}   {top[:70]}")
        out.setdefault("prod", {})[t] = {
            "n_with": n, "n_rules": len(arc),
            "per_rule": [len(x) for x in ps],
            "pairs": {f"{'+'.join(x)} x {'+'.join(y)}": v
                      for (x, y), v in cnt.most_common(10)}}

    block(C_ARMS, "C.  the λ sweep — the Pareto curve")

    print("\n" + "=" * 86)
    print("C2.  ★ the pure rank_loss(k=100) — recomputed for the "
          "between-family comparison")
    print("=" * 86)
    print("  the archive's `rank_loss` has λ mixed into it and cannot be put "
          "side by side.")
    print(f"\n  {'':14s} {'pure rank_loss':>15} {'train regret':>13}")
    for t in C_ARMS:
        rl, rg = [], []
        for e in best[t]:
            rl.append(float(prob100.rank_loss(compile_rule(e["code"]),
                                              np.asarray(e["w"], float))))
            rg.append(e["regret"])
        print(f"  {A[t][0]:14s} {np.median(rl):14.4f} {np.median(rg):11.4f}")
        out.setdefault("pure", {})[t] = {"rank_loss": rl, "regret": rg}

    # ------------------------------------------------------ in common
    print("\n" + "=" * 86)
    print("in common — terms/nodes/rejection rate/cost/axes")
    print("=" * 86)
    print(f"  {'':14s} {'trm':>4} {'nodes':>6} {'budget':>8} {'rej':>7} "
          f"{'fitter':>7} {'config':>7} {'min':>7}")
    for t, *_ in ARMS:
        arc = [e for d in dirs[t] for e in _rows(d, "archive.jsonl")]
        tm = [_n_terms(e["code"]) for e in arc]
        nd = [sum(1 for _ in ast.walk(ast.parse(e["code"]))) for e in arc]
        sp_ = [len(_numeric_literals(ast.parse(e["code"]))[0]) + len(e["w"])
               for e in arc]
        prop = rej = mv = sc = 0
        secs = 0.0
        for d in dirs[t]:
            for r in _rows(d, "rounds.jsonl"):
                prop += r["n_proposed"]
                rej += (r["n_rejected_schema"] + r["n_rejected_static"]
                        + r["n_rejected_sandbox"] + r["n_rejected_fit"])
                mv += r["n_fit_moved"]
                sc += r["n_scored"]
                secs += r["seconds"]
        nc = [_configs(e["code"], e["w"], T, M, hold) for e in best[t]]
        print(f"  {A[t][0]:14s} {np.median(tm):4.0f} {np.median(nd):6.0f} "
              f"{np.median(sp_):8.1f} {rej / prop:7.1%} {mv / max(1, sc):7.1%} "
              f"{np.median(nc):7.1f} {secs / 60:7.1f}")
        out.setdefault("common", {})[t] = {
            "terms": tm, "nodes": nd, "spent": sp_, "rej": rej / prop,
            "reach": mv / max(1, sc), "n_config": nc, "minutes": secs / 60,
            "feats": [sorted(_feats(e["code"])) for e in best[t]]}

    print(f"\n  ★ the ceiling measurement's five axes (/3 seeds)\n    {'':14s} "
          + " ".join(f"{n[:11]:>12s}" for n in WATCH))
    for t, *_ in ARMS:
        fs = [_feats(e["code"]) for e in best[t]]
        print(f"    {A[t][0]:14s} "
              + " ".join(f"{sum(n in f for f in fs):>12d}" for n in WATCH))

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3 seeds cannot give significance — it is read by range "
          "separation (principle 27)")


if __name__ == "__main__":
    main()
