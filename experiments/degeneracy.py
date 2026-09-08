"""★ Does the rule actually look at the shape — the degeneracy check. 0 LLM
calls.

    python3 experiments/degeneracy.py

The **pre-registration** is `docs/artifacts/degeneracy-prereg.md` — the
decision line was nailed down first.

## What is asked

```
memorisation   if p.M == 4096                    ★ the static check blocks it
degeneracy     it picks the same config regardless of the shape
                                                 ★ nothing blocks it
```

## ★ It is measured **before** the transfer (on the A6000)

The ranking collapsing after a transfer is fixed by the `(b)` refit. **If it
cannot rank even before the transfer, the rule is a different thing
entirely.**
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
SRC_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
PCTS = (1.0, 5.0, 10.0)
#: The all-range tau would be 200 million pairs with 20,000 candidates, so it
#: is approximated by a sample. **A fixed seed.**
TAU_SAMPLE = 4000
TAU_SEED = 12345
TOP_N = 100          # ★ the top-rank tau is exhaustive


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/degeneracy.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    A = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    mA = FeatureMatrix(A, REGISTRY)
    sp = _splits(A)
    shapes = list(sp.train.shapes)      # ★ the 41 training shapes

    print("=" * 78)
    print("the degeneracy check — does the rule actually look at the shape   "
          "★ before the transfer (A6000)")
    print("=" * 78)
    n_c = [len(A.times_of(p)) for p in shapes]
    print(f"  training {len(shapes)} shapes   candidates, median "
          f"{int(np.median(n_c))}")
    ans_sz = sorted(int(A.answer_mask(p).sum()) for p in shapes)
    print(f"  answer set size: median {int(np.median(ans_sz))}  "
          f"range {ans_sz[0]}~{ans_sz[-1]}   shapes with exactly 1: "
          f"{sum(1 for x in ans_sz if x == 1)}")

    # -- the static top-1 — ★ the representative implementation is used. A
    #    third definition is not made (principle 2. `_arith_intensity`
    #    already stepped on this today)
    #    `StaticTopK`'s key is (kernel_id, split_k, split_k_mode) — not the
    #    axis coordinates. So the rule's picks are counted by the same key.
    from kernelrule.baselines.static_topk import StaticTopK

    st = StaticTopK(A, shapes, coverage="union").run(ks=(1,))
    static_key = st.chosen[0]
    print(f"  static top-1 (representative: all statuses + the union cover)  "
          f"regret {st.by_k[1]['all']:.4f}  cover {st.coverage[1]:.0%}")
    print("  ★ the key = (kernel_id, split_k, split_k_mode) — the rule's "
          "picks are counted by the same key\n")

    # ★ Does a ranking exist at the top at all — this comes before reading tau
    uniq = np.array([len(np.unique(np.asarray(A.times_of(p))[
        np.argsort(np.asarray(A.times_of(p)), kind="stable")[:TOP_N]]))
        for p in shapes])
    print(f"  ★ distinct time values among the true top {TOP_N}: median "
          f"{int(np.median(uniq))}  range {uniq.min()}~{uniq.max()}   "
          f"all tied in {int((uniq == 1).sum())} shapes / "
          f"10 or fewer in {int((uniq <= 10).sum())} shapes")
    print(f"     the tick is {A.noise.tick_ms} ms — at the top the resolution "
          f"dominates\n")

    rng0 = np.random.default_rng(TAU_SEED)
    res: dict = {"runs": {}, "n_shapes": len(shapes),
                 "answer_set_sizes": ans_sz}

    print(f"  {'structure':22s} {'kinds':>6} {'mode':>5} {'=static':>8} "
          f"{'hit@1':>6} {'tau all':>9} {'★tau top100':>12}")
    for run in SRC_RUNS:
        f = Path("runs") / run / "archive.jsonl"
        e = sorted((json.loads(x) for x in f.read_text().splitlines()
                    if x.strip()), key=lambda z: z["regret"])[0]
        fn = compile_rule(e["code"])
        ws = {}
        for nm in ("short", "long"):
            g = [q for q in shapes if regime_of(q, A.hw) == nm]
            ws[nm] = fit_weights(fn, mA, A, Split("train", tuple(g)),
                                 e["w"], max_evals=300,
                          objective="regret").w

        picks, hit1, taus, taus_top = [], 0, [], []
        n_flat_top = 0
        hits = {q: [0, 0] for q in PCTS}
        for p in shapes:
            cand = A.candidates(p)
            sc = np.asarray(make_score_of(fn, mA, ws[regime_of(p, A.hw)])(
                p, cand), dtype=float)
            t = A.times_of(p)
            pick = int(cand.top_k(sc, 1)[0])
            picks.append((str(cand.kernel_id[pick]),
                          int(cand.split_k[pick]),
                          str(cand.split_k_mode[pick])))
            hit1 += int(t[pick] == t.min())
            # tau — all-range by sample, the top exhaustively
            n = len(t)
            idx = rng0.choice(n, size=min(TAU_SAMPLE, n), replace=False)
            taus.append(kendalltau(sc[idx], t[idx], variant="b").statistic)
            top = np.argsort(t, kind="stable")[:TOP_N]
            # ★ If the times of the top TOP_N are all tied, tau is undefined.
            #   One `nan` pollutes the whole median — they are counted and
            #   dropped.
            if len(np.unique(t[top])) > 1:
                taus_top.append(kendalltau(sc[top], t[top],
                                           variant="b").statistic)
            else:
                n_flat_top += 1
            order = np.argsort(sc, kind="stable")
            am = A.answer_mask(p)
            best = int(np.argmin(t))
            for q in PCTS:
                k = max(1, int(np.ceil(n * q / 100.0)))
                s_ = set(order[:k].tolist())
                hits[q][0] += int(best in s_)
                hits[q][1] += int(am[order[:k]].any())

        c = Counter(picks)
        n_static = sum(1 for k in picks if k == static_key)
        row = {"n_kinds": len(c), "top_count": c.most_common(1)[0][1],
               "same_as_static": n_static, "hit1": hit1,
               "tau_all": float(np.median(taus)),
               "tau_top100": (float(np.median(taus_top)) if taus_top
                              else float("nan")),
               "n_flat_top": n_flat_top,
               "hits": {str(q): {"best": hits[q][0], "answer": hits[q][1]}
                        for q in PCTS}}
        res["runs"][run] = row
        print(f"  {run:22s} {len(c):6d} {row['top_count']:5d} "
              f"{n_static:8d} {hit1:6d} {row['tau_all']:9.3f} "
              f"{row['tau_top100']:12.3f}"
              + (f"  (tied {n_flat_top})" if n_flat_top else ""))

    # -- the random floor (20 draws — ★ the floor is a sample too,
    #    principle 7) ------------------------------------------------------
    rng = np.random.default_rng(0)
    b_hit1, b_hits = [], {q: [[], []] for q in PCTS}
    b_tau, b_tau_top, b_kinds = [], [], []
    for _ in range(20):
        h1 = 0
        picks = []
        hh = {q: [0, 0] for q in PCTS}
        tt, ttt = [], []
        for p in shapes:
            t = A.times_of(p)
            n = len(t)
            sc = rng.random(n)
            order = np.argsort(sc, kind="stable")
            cd = A.candidates(p)
            j = int(order[0])
            picks.append((str(cd.kernel_id[j]), int(cd.split_k[j]),
                          str(cd.split_k_mode[j])))
            h1 += int(t[int(order[0])] == t.min())
            am = A.answer_mask(p)
            best = int(np.argmin(t))
            for q in PCTS:
                k = max(1, int(np.ceil(n * q / 100.0)))
                s_ = set(order[:k].tolist())
                hh[q][0] += int(best in s_)
                hh[q][1] += int(am[order[:k]].any())
            idx = rng.choice(n, size=min(TAU_SAMPLE, n), replace=False)
            tt.append(kendalltau(sc[idx], t[idx], variant="b").statistic)
            top = np.argsort(t, kind="stable")[:TOP_N]
            if len(np.unique(t[top])) > 1:
                ttt.append(kendalltau(sc[top], t[top], variant="b").statistic)
        b_hit1.append(h1)
        b_kinds.append(len(Counter(picks)))
        b_tau.append(float(np.median(tt)))
        b_tau_top.append(float(np.median(ttt)) if ttt else float("nan"))
        for q in PCTS:
            b_hits[q][0].append(hh[q][0])
            b_hits[q][1].append(hh[q][1])
    print(f"  {'★ random (mean of 20)':22s} {np.mean(b_kinds):6.1f} "
          f"{'—':>5} {'—':>8} {np.mean(b_hit1):6.1f} "
          f"{np.mean(b_tau):9.3f} {np.mean(b_tau_top):12.3f}")

    print(f"\n  {'structure':22s} " + "  ".join(
        f"{q}% answer/best" for q in PCTS))
    for run in SRC_RUNS:
        h = res["runs"][run]["hits"]
        print(f"  {run:22s} " + "  ".join(
            f"{h[str(q)]['answer']:2d}/{h[str(q)]['best']:2d}"
            f" ({h[str(q)]['answer'] / len(shapes):.0%})" for q in PCTS))
    print(f"  {'★ random mean':22s} " + "  ".join(
        f"{np.mean(b_hits[q][1]):4.1f}/{np.mean(b_hits[q][0]):4.1f}"
        f" ({np.mean(b_hits[q][1]) / len(shapes):.0%})" for q in PCTS))

    res["random"] = {
        "n_kinds": float(np.mean(b_kinds)), "hit1": float(np.mean(b_hit1)),
        "tau_all": float(np.mean(b_tau)),
        "tau_top100": float(np.mean(b_tau_top)),
        "hits": {str(q): {"best": float(np.mean(b_hits[q][0])),
                          "answer": float(np.mean(b_hits[q][1]))}
                 for q in PCTS}}

    # -- the verdict (the line nailed down in the pre-registration) ---------
    kinds = [res["runs"][r]["n_kinds"] for r in SRC_RUNS]
    tt = [res["runs"][r]["tau_top100"] for r in SRC_RUNS]
    n_flat = sum(res["runs"][r]["n_flat_top"] for r in SRC_RUNS) // len(SRC_RUNS)
    print("\n" + "=" * 78)
    print("the verdict — the line nailed down in the pre-registration")
    print("=" * 78)
    mk = float(np.median(kinds))
    mt = float(np.nanmedian(tt))
    print(f"  diversity median {mk:.1f} kinds  -> " + (
        "★ it really looks at the shape (>=20)" if mk >= 20
        else "★ effectively a constant with variations (<=5)" if mk <= 5
        else "in between (6~19)"))
    print(f"  top-100 tau median {mt:.3f}  "
          f"(★ undefined because of ties in {n_flat}/{len(shapes)} shapes, "
          f"excluded)  -> " + (
        "★ it ranks in the band that matters (>=0.30)" if mt >= 0.30
        else "★ it cannot rank (<=0.10)" if mt <= 0.10 else "in between"))
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
