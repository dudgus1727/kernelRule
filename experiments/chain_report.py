"""★ The four-way relay result — two orderings + budget 8/16. 0 LLM calls.

    python3 experiments/chain_report.py

## What is measured

    ord-r2g   rank -> regret   budget 8
    ord-g2r   regret -> rank   budget 8
    bud08     rank fixed       budget 8      ← the baseline of the budget
                                               contrast
    bud16     rank fixed       budget 16

⚠️ **Everything is measured on the 20 holdout shapes.** The tau in D-101 is a
value on the 41 training shapes and cannot be put alongside (principle 4).

## Why it is measured twice

The objective when the artefact is chosen (selection) and when the weights
are fitted (fitting) **can differ.** The four arms have different final
objectives, so measuring with one fit alone would compare *the way of
evaluating* rather than *the ordering*. So the same rule is measured **both**
with a regret fit and with a rank fit, and the two blocks are put side by
side.

The helper functions are taken from `two_stage.py` as they are — the
procedure has to be a single one for things to be put side by side
(principle 2).
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
from kernelrule.core.weights import _Problem, make_score_of
from kernelrule.features import REGISTRY

#: (tag, label, the final objective) — the final objective is **the criterion
#: for choosing the artefact**.
ARMS = [
    ("ord-r2g", "4-1 rank->regret", "regret"),
    ("ord-g2r", "4-2 regret->rank", "rank"),
    ("bud08", "budget 8  (rank fixed)", "rank"),
    ("bud16", "budget 16 (rank fixed)", "rank"),
]
SEEDS = 3
#: The five axes the ceiling measurement named (ranking-ceiling.md §3).
WATCH = ("split_k_cost", "sm_idle_cost", "pipeline_warmup_frac",
         "tail_waste", "waves")


def _rows(d: Path, name: str) -> list[dict]:
    return [json.loads(x) for x in (d / name).read_text().splitlines()
            if x.strip()]


def _pick(d: Path, by: str) -> dict:
    """The archive best — **by that arm's final objective**."""
    key = (lambda e: e.get("rank_loss", 1e9)) if by == "rank" \
        else (lambda e: e["regret"])
    return sorted(_rows(d, "archive.jsonl"), key=key)[0]


def _feats(code: str) -> set:
    return {n.attr for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "f"}


def _switch_round(d: Path) -> int | None:
    """The switch round — read from the run artefact `config.json`.

    ★ It is read from the artefact, not from the log (stdout). The log has
    three seeds mixed into one file, so the attribution to a seed is only by
    eye, and once it is deleted it cannot be traced back.
    """
    c = json.loads((d / "config.json").read_text())
    r = int(c.get("switch_round", -1))
    return None if r < 0 else r


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
    print(f"  {label:22s} {np.median(v[:, 0]):8.4f} {np.median(v[:, 1]):12.3f} "
          f"{np.median(v[:, 2]):10.3f}   "
          f"({v[:, 1].min():+.3f}~{v[:, 1].max():+.3f})"
          f" ({v[:, 2].min():+.3f}~{v[:, 2].max():+.3f})")
    return {"vals": vals, "med": [float(np.median(v[:, i])) for i in range(3)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/chain.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    dirs = {tag: [Path("runs") / f"f1pipe-F3-{tag}-s{i}" for i in range(SEEDS)]
            for tag, _, _ in ARMS}
    best = {tag: [_pick(d, by) for d in dirs[tag]] for tag, _, by in ARMS}
    out: dict = {"n_holdout": len(hold), "n_train": len(train)}
    # -------------------------------------------------------- §1 the metrics
    print("=" * 84)
    print("§1  the final metrics — the A6000 holdout of 20 shapes")
    print("=" * 84)
    print("  selection: the archive best by each arm's **final objective** "
          "(the median of 3 seeds)")
    fl = _floor(T, hold)
    for obj, head in (("regret", ("regret-fitted weights (the final "
                                  "scoring §10 path)")),
                      ("rank", "rank-fitted weights")):
        print(f"\n  --- {head} ---")
        print(f"  {'':22s} {'regret':>8} {'top-100 tau':>12} {'all':>10}"
              f"   (tau100 range)")
        for tag, label, _ in ARMS:
            vals = [_measure(*_fit(e["code"], e["w"], T, M, train, obj),
                             T, M, hold) for e in best[tag]]
            out.setdefault(obj, {})[tag] = _blk(label, vals)
        print(f"  {'★ random floor (draw 20)':22s} {fl[0]:8.4f} "
              f"{fl[1]:12.3f} {fl[2]:10.3f}")
    out["floor"] = fl

    # -------------------------------------------------------- §2 the switch
    print("\n" + "=" * 84)
    print("§2  the switch — when it changed, and what jumps in the round "
          "right after")
    print("=" * 84)
    print("  ⚠️ the `best` in the log is **by the objective of the moment**, "
          "so it jumps")
    print("     by itself at the switch. Here both metrics are written down "
          "every round and compared.")
    prob = _Problem(M, T, sp.train.shapes, 1)
    prob.build_pairs(T, 100)
    print(f"  the rank loss was **recomputed here** — during a run it is "
          f"recorded only when objective=='rank', so\n"
          f"     the other side of the switch is NaN and they cannot be put "
          f"side by side ({prob.n_pairs:,} pairs, {prob.n_dropped:,} "
          f"undecidable pairs excluded).")
    for tag, label, _ in ARMS[:2]:
        print(f"\n  {label}")
        arm = []
        for i, d in enumerate(dirs[tag]):
            rd = {r["round"]: r for r in _rows(d, "rounds.jsonl")}
            bs = _rows(d, "bests.jsonl")
            sw = _switch_round(d)
            arm.append(sw)
            print(f"    s{i}  switch r{sw}")
            print(f"      {'round':>7} {'regret':>8} {'rank loss':>10} "
                  f"{'cells':>6}")
            seq = []
            for b in bs:
                if not (sw - 2 <= b["round"] <= sw + 2):
                    continue
                rl = float(prob.rank_loss(compile_rule(b["code"]),
                                          np.asarray(b["w"], dtype=float)))
                seq.append((b["round"], b["regret"], rl,
                            rd[b["round"]]["n_cells"]))
                print(f"      {'r' + str(b['round']):>7} {b['regret']:8.4f} "
                      f"{rl:10.4f} {rd[b['round']]['n_cells']:6d}"
                      + ("  ★switch" if b["round"] == sw else ""))
            pre = [x for x in seq if x[0] == sw - 1][0]
            post = [x for x in seq if x[0] == sw][0]
            print(f"      -> regret {pre[1]:.4f}->{post[1]:.4f} "
                  f"({post[1] - pre[1]:+.4f})   rank {pre[2]:.4f}->"
                  f"{post[2]:.4f} "
                  f"({post[2] - pre[2]:+.4f})   cells {pre[3]}->{post[3]}")
            out.setdefault("switch", {}).setdefault(tag, []).append(
                {"seed": i, "round": sw, "seq": seq})

    # -------------------------------------------------------- §3 the budget
    print("\n" + "=" * 84)
    print("§3  the budget — the train vs holdout gap / the number of terms / "
          "the fitter reach rate")
    print("=" * 84)
    print("  ⚠️ the budget-16 arm is **void** (D-105). The prompt rendered "
          "as 8, so")
    print("     the model heard the same cap in both arms — it is not a "
          "budget contrast but")
    print("     a repetition of the same condition. The two lines below have "
          "to be read that way.\n")
    print(f"  {'':22s} {'train regret':>12} {'holdout':>9} {'gap':>8} "
          f"{'terms':>6} {'fitter reach':>13}")
    for tag, label, _ in ARMS:
        g, tr, ho, reach, terms = [], [], [], [], []
        for d, e in zip(dirs[tag], best[tag], strict=True):
            rd = _rows(d, "rounds.jsonl")[-1]
            tr.append(rd["best_regret"])
            ho.append(rd["best_val_regret"])
            g.append(rd["best_val_regret"] - rd["best_regret"])
            allr = _rows(d, "rounds.jsonl")
            reach.append(sum(x["n_fit_moved"] for x in allr)
                         / max(1, sum(x["n_scored"] for x in allr)))
            terms.append(len(e["w"]))
        print(f"  {label:22s} {np.median(tr):12.4f} {np.median(ho):9.4f} "
              f"{np.median(g):+8.4f} {np.median(terms):6.1f} "
              f"{np.mean(reach):10.1%}")
        print(f"  {'':22s} {'':12s} {'':9s}   gap per seed "
              f"{' '.join(f'{x:+.4f}' for x in g)}   terms {terms}")
        out.setdefault("budget", {})[tag] = {
            "train": tr, "hold": ho, "gap": g, "terms": terms,
            "reach": reach}

    # -------------------------------------------------------- §4 in common
    print("\n" + "=" * 84)
    print("§4  in common — the axes that survived / config diversity / the "
          "rejection rate / the cost")
    print("=" * 84)
    print(f"  {'':22s} {'config kinds':>12} {'rej rate':>9} {'acc rate':>9} "
          f"{'LLM calls':>10} {'min':>7}")
    for tag, label, _ in ARMS:
        nc = [_configs(e["code"], e["w"], T, M, hold) for e in best[tag]]
        prop = rej = acc = sc = calls = secs = 0
        for d in dirs[tag]:
            for r in _rows(d, "rounds.jsonl"):
                prop += r["n_proposed"]
                rej += (r["n_rejected_schema"] + r["n_rejected_static"]
                        + r["n_rejected_sandbox"] + r["n_rejected_fit"])
                acc += r["n_accepted"]
                sc += r["n_scored"]
                calls += sum(r["llm_calls"].values())
                secs += r["seconds"]
        print(f"  {label:22s} {np.median(nc):12.1f} {rej / prop:9.1%} "
              f"{acc / max(1, sc):9.1%} {calls:10d} {secs / 60:7.1f}")
        out.setdefault("common", {})[tag] = {
            "n_config": nc, "rej": rej / prop, "acc": acc / max(1, sc),
            "llm_calls": calls, "minutes": secs / 60,
            "feats": [sorted(_feats(e["code"])) for e in best[tag]]}

    print("\n  ★ the five axes the ceiling measurement named — does the "
          "final rule use them (/3 seeds)")
    print(f"    {'':22s} " + " ".join(f"{n[:11]:>12s}" for n in WATCH))
    for tag, label, _ in ARMS:
        fs = [_feats(e["code"]) for e in best[tag]]
        print(f"    {label:22s} "
              + " ".join(f"{sum(n in f for f in fs):>12d}" for n in WATCH))
    allf = Counter(x for tag, _, _ in ARMS for e in best[tag]
                   for x in _feats(e["code"]))
    print(f"\n  the union of the axes used, {len(allf)} of them (the "
          f"parenthesis is how many of the 12 rules):")
    print("    " + ", ".join(f"{k}({v})" for k, v in allf.most_common()))

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3 seeds cannot give significance — it is read by range "
          "separation (principle 27)")


if __name__ == "__main__":
    main()
