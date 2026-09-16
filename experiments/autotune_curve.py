"""★ The autotuning curve — how many kernel runs at deployment (D-176 §1).
**0 LLM calls · 0 GPU.**

    python3 -m experiments.autotune_curve --gpu a6000

The axis is **"how many kernels must be run at deployment time"**:

```
★ our rule   ★ 0 — it computes
★ vendor     ★ 0
★ autotuning ★ k, ★ per shape, at deployment
```

The table is exhaustive, so the simulation is a lookup. ⛔ No kernel is run.

## ★ The search space — only over candidates that exist

```
free axes 11   tile_m 4 · tile_n 3 · tile_k 2 · ext_warp_m 4 · ext_warp_n 2
               ext_warp_k 2 · ext_stages 7 · ext_swizzle_type 2
               ext_swizzle_n 4 · split_k 8 · split_k_mode 2
⛔ align_a/b/c are ★ not free — the shape fixes them (measured: one value
   per shape)
cartesian product          ★ 344,064
valid combinations / shape ★ 15,015 (median, 3,465~17,325)
★ valid fraction           ★ 4.36%
```

**Drawing the axes independently would spend 96% of the budget on
combinations that do not exist.** So the search runs **over that shape's own
candidate list**, using the axis structure conditionally: at each step only
the values that still exist given what has been chosen are offered. That is
what a CUTLASS profiler does.

## How the conditional draw is implemented

Axes are chosen in a fixed order; before each one the candidate frame is
filtered by the choices already made, so only values that still exist are
offered. A trial therefore **always lands on a real candidate** and never
wastes `k`.

⚠️ Optuna refuses a *changing* domain under one parameter name
(`CategoricalDistribution does not support dynamic value space`). The
documented way to express a conditional space is to **put the condition in
the name**: the parameter for `tile_n` after `tile_m=128` is
`tile_n@tile_m=128`, and that name's domain is fixed. TPE then models each
branch from the trials that reached it — which is what a conditional
sampler should do, and also why it has fewer observations per branch than a
flat space would.

★ Whether TPE beats random here is a **result**, not an assumption — an
11-axis discrete space with strong interactions and a branching name space
may not suit it.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from experiments.f1_pipeline import _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.core.table import PerfTable

#: ★ The order axes are chosen in. Fixed so a run is reproducible; the
#: conditional filter makes the order matter only for how the tree is cut.
AXES = ("tile_m", "tile_n", "tile_k", "ext_warp_m", "ext_warp_n",
        "ext_warp_k", "ext_stages", "ext_swizzle_type", "ext_swizzle_n",
        "split_k", "split_k_mode")
#: ⚠️ 2026-09-15 (D-176 §2): the top is **4096**, not 8192. Measured on the
#: a6000's 65 shapes, random reaches 1.0041 at k=4096 (27% of the candidates)
#: and our rule is passed at k ≈ 150~200 — the interesting range is inside
#: 1~1024. ⛔ This is not cutting where we win: the crossing is drawn.
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
REPEATS = 50
#: ★ The four arms, in the order they are reported.
ARMS = ("random", "random_pruned", "random_cond", "tpe")
#: ⚠️ ★ **TPE stops at 1024; the three random arms go to 4096.**
#:
#: Measured on a6000 `1x11008x4096` (15,015 candidates), one seed, this
#: implementation:
#:
#: ```
#: k=  512   ★   22.5s   dup 62.7%
#: k= 1024   ★  163.0s   dup 65.0%   trials 2,924   ★ regret 1.0000
#: k= 2048   ★ 1065.0s   dup 75.1%   trials 8,212   ★ regret 1.0000
#: k= 4096   ★ ~2 hours per shape per seed at that growth
#: ```
#:
#: Two things grow together: the duplicate rate rises, so more draws are
#: needed per new point, and Optuna's per-trial cost grows with the number
#: of completed trials — the path-in-name encoding adds 11 fresh parameter
#: names every trial.
#:
#: ⛔ This is **not** cutting k where we win. TPE reaches regret **1.0000**
#: — the shape's global optimum — by k=1024, and the crossing against our
#: rule is at k ≈ 50~60, twenty times lower. The 2048 and 4096 cells of the
#: TPE curve are left **empty** rather than extrapolated.
TPE_KMAX = 1024
OUT = Path("docs/artifacts/autotune-curve.json")


def _prune_mask(df) -> np.ndarray:
    """★ The 'pruned random' arm — no spill, and the tile not larger than the
    problem. ⚠️ Measured, not assumed: it removes far less than it sounds."""
    ok = (df["spill_bytes"].to_numpy() == 0)
    ok &= (df["tile_m"].to_numpy() <= df["M"].to_numpy())
    ok &= (df["tile_n"].to_numpy() <= df["N"].to_numpy())
    return ok


def _cond_random_curve(df, times: np.ndarray, ks, rng,
                       repeats: int = 5) -> tuple[dict, dict]:
    """★ The control that separates **the parameterisation** from **the
    model** (D-176 §1-1, last line).

    It walks the same conditional axis order as `_tpe_curve` and picks
    **uniformly among the values still alive** at each axis — no TPE, no
    model, same duplicate rule (a repeat does not count toward k).

    ⚠️ This is not the same thing as drawing uniformly from the candidate
    list. Uniform-over-axis-values is heavily biased relative to
    uniform-over-candidates, because a value that survives into few
    candidates is offered just as often as one that survives into many.
    ★ Whatever the TPE arm gains over **this** arm is what the sampler's
    model contributed; what this arm gains over plain `random` is what the
    axis structure contributed on its own.
    """
    cols = {a: df[a].to_numpy() for a in AXES}
    best = times.min()
    kmax = min(max(ks), len(times))
    runs = []
    dups = []
    for _ in range(repeats):
        seen: set = set()
        vals: list[float] = []
        n_dup = 0
        cap = kmax * DUP_CAP
        n_draw = 0
        while len(vals) < kmax and n_draw < cap:
            n_draw += 1
            alive = np.ones(len(times), dtype=bool)
            key = []
            for a in AXES:
                dom = sorted({x.item() if hasattr(x, "item") else x
                              for x in cols[a][alive]})
                v = dom[int(rng.integers(len(dom)))]
                key.append(v)
                alive &= (cols[a] == v)
            i = int(np.nonzero(alive)[0][0])
            k = tuple(key)
            if k in seen:
                n_dup += 1
                continue
            seen.add(k)
            vals.append(float(times[i]))
        runs.append(np.minimum.accumulate(np.asarray(vals)))
        dups.append(n_dup / max(1, n_draw))
    curve = {str(k): float(np.mean([r[min(k, len(r)) - 1] / best
                                    for r in runs])) for k in ks}
    return curve, {"duplicate_rate": round(float(np.mean(dups)), 4),
                   "repeats": repeats}


def _random_curve(times: np.ndarray, ks, rng, mask=None) -> dict:
    """k draws without replacement -> the best time seen."""
    idx = np.nonzero(mask)[0] if mask is not None else np.arange(len(times))
    if idx.size == 0:
        return {str(k): float("nan") for k in ks}
    best = times.min()
    out = {}
    for k in ks:
        kk = min(k, idx.size)
        vals = []
        for _ in range(REPEATS):
            pick = rng.choice(idx, size=kk, replace=False)
            vals.append(float(times[pick].min() / best))
        out[str(k)] = float(np.mean(vals))
    return out


#: ★ How many draws a study may make in total while collecting `kmax`
#: **novel** ones. A duplicate costs sampler time but not `k`, so the cap
#: only stops a pathological loop.
DUP_CAP = 6
#: ★ D-176 §1-3 — the sampler settings, chosen and recorded rather than left
#: to the default. ⚠️ In **Optuna 5.0.0** `multivariate=None` (the default)
#: resolves to **True** for a single-objective study, so "we left it at the
#: default" would not have been a statement about what ran.
#:
#: ★ Measured on a6000 `1x11008x4096` (15,015 candidates), one seed:
#:
#: ```
#: mv=False g=False   n=512  ★  22.5s   dup 62.7%   regret 1.0072
#: mv=True  g=False   n=512  ★  23.1s   dup 62.7%   regret 1.0072  ← identical
#: mv=True  g=True    n=256  ★ 210.5s   dup 62.2%   regret 1.0072  (not better)
#: ```
#:
#: ⚠️ `mv=True, g=False` came out **bit-identical** to `mv=False`. That is
#: the path-in-name encoding: Optuna's multivariate TPE models only the
#: parameters common to every completed trial, and once two trials take
#: different branches the only shared name is the first axis. So there is
#: nothing for it to model jointly. `group=True` does decompose the space,
#: but it was **≥10x slower and no better** (210s at n=256 against 8.8s).
#:
#: ★ `experiments/tpe_settings.py` then measured all three against random
#: (a6000 fold0, 2 shapes, 2 seeds, k<=256):
#:
#: ```
#: k=64   random 1.0664  uni 1.0544  mv 1.0544  ★ mv+group 1.0289
#: k=256  random 1.0310  uni 1.0506  mv 1.0506  ★ mv+group 1.0258
#: 초/형상/시드     ★ 11.2        13.4          ★ 213.9
#: ★ 교집합 탐색 공간 ★ 1 개 이름 — group=False 가 함께 모형화할 것이 없다
#: ```
#:
#: ⚠️ **The best arm is not the one this curve runs.** `group=True` was
#: better at both ks, and it is **19x** the cost at k=256 alone, growing
#: super-quadratically; at k=4096 x 3 seeds x 256 shapes it is out of reach.
#: ⛔ That is recorded rather than hidden — the curve below is TPE with
#: `mv=False, group=False`, and it is the **weaker** of the two TPEs.
#:
#: ★ So: `False/False` — chosen for cost, with the better arm's numbers on
#: the record, not because it is a default.
TPE_MULTIVARIATE = False
TPE_GROUP = False


def _tpe_curve(df, times: np.ndarray, ks, seed: int,
               multivariate: bool = TPE_MULTIVARIATE,
               group: bool = TPE_GROUP) -> tuple[dict, dict]:
    """★ TPE over the **valid** candidates, choosing axes conditionally, and
    ⛔ **not counting a repeat against k** (D-176 §1).

    ## ⚠️ Not Optuna's default — and that favours TPE

    Optuna's TPE **may propose the same point again**, and by default that
    costs a trial. That default is right for a continuous, noisy objective:
    re-evaluating tells you something. ⛔ Ours is neither.

    ```
    ★ discrete · finite      median 15,015 candidates per shape
    ★ deterministic          a table lookup — a repeat carries 0 information
    ★ the cost axis is "how many kernels were run" -> a repeat is pure waste
    ★ real autotuners do not re-run a config they measured
      (CUTLASS profiler runs each valid kernel once; AutoTVM records what it
       measured)
    ```

    So a repeat is told to the sampler (its model stays consistent) but is
    **not counted toward k**. ★ This is a change **in TPE's favour**, made
    deliberately, and the duplicate rate is reported so the size of the
    favour is visible. ⛔ It must be stated wherever this curve is used:
    the comparison was adjusted to help TPE, not to hurt it.

    ## ★ multivariate / group — set deliberately, not left to the default

    ⚠️ In **Optuna 5.0.0** `multivariate=None` is the default and resolves
    to **`True`** for a single-objective study, so "the default" is not a
    description of what ran. Both are therefore passed explicitly:

    ```
    ★ multivariate = TPE_MULTIVARIATE
    ★ group        = TPE_GROUP
    ★ n_startup_trials = Optuna's default (10) — recorded per run
    ```

    Why `group` matters here: with `multivariate=True, group=False` Optuna
    models jointly only the parameters **common to every completed trial**
    (the intersection search space). Our names carry the path, so after two
    trials that took different branches the intersection is just the first
    axis — the rest fall back to the independent (univariate) TPE.
    `group=True` decomposes the space into subspaces instead, so a branch
    can be modelled jointly with the trials that actually reached it.

    ★ Which of the three settings is used is a **measurement**, not a
    preference: `experiments/tpe_settings.py` runs them against random on
    real shapes and its numbers pick the constants above. ⛔ And whether the
    winner beats random at all stays a result — it is reported either way.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cols = {a: df[a].to_numpy() for a in AXES}
    best = times.min()
    ks = tuple(k for k in ks if k <= TPE_KMAX)
    kmax = min(max(ks), len(times))
    sampler = optuna.samplers.TPESampler(seed=seed,
                                         multivariate=multivariate,
                                         group=group)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    novel: list[float] = []
    seen_keys: dict[tuple, float] = {}
    n_dup = 0
    domain_sizes: list[int] = []
    cap = kmax * DUP_CAP
    n_trials = 0
    while len(novel) < kmax and n_trials < cap:
        trial = study.ask()
        n_trials += 1
        alive = np.ones(len(times), dtype=bool)
        path = ""
        key = []
        for a in AXES:
            dom = sorted({x.item() if hasattr(x, "item") else x
                          for x in cols[a][alive]})
            domain_sizes.append(len(dom))
            # ★ The condition goes in the **name** so each domain is fixed
            #   (Optuna refuses a dynamic value space under one name).
            v = trial.suggest_categorical(f"{a}{path}", dom)
            path += f"|{a}={v}"
            key.append(v)
            alive &= (cols[a] == v)
        # ★ Always a real candidate — that is the point of the conditional
        #   draw.
        i = int(np.nonzero(alive)[0][0])
        t = float(times[i])
        k = tuple(key)
        study.tell(trial, t)          # the sampler still learns from it
        if k in seen_keys:
            n_dup += 1                # ⛔ costs no kernel run
            continue
        seen_keys[k] = t
        novel.append(t)
    run = np.minimum.accumulate(np.asarray(novel))
    curve = {str(k): float(run[min(k, len(run)) - 1] / best) for k in ks}
    info = {"n_novel": len(novel), "n_trials": n_trials,
            "n_duplicate": n_dup,
            "duplicate_rate": round(n_dup / max(1, n_trials), 4),
            "hit_cap": n_trials >= cap,
            "mean_domain_size": float(np.mean(domain_sizes)),
            "n_startup_trials": sampler._n_startup_trials,
            "multivariate": multivariate, "group": group}
    return curve, info


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="a6000")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--tpe-seeds", type=int, default=3)
    # ★ Shapes are independent, so a fold can be split across processes.
    #   ⛔ This is the **only** parallelism taken: inside one TPE study the
    #   k-th trial must see k-1 observations, and `n_jobs>1` would let
    #   workers sample from a stale model — a weaker sampler measured under
    #   the same name (D-176 §1-1).
    ap.add_argument("--shape-slice", default=None, metavar="I/N",
                    help="score shapes I, I+N, I+2N... of this fold only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"docs/artifacts/autotune-{a.gpu}.json")

    T = TABLES[a.gpu]
    table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                  ok_only=False)
    folds = range(4) if a.fold is None else (a.fold,)
    # ★ Per **shape**, not per fold. A fold can be split across processes,
    #   so the aggregation has to happen after the shards are merged
    #   (`experiments/autotune_report.py`) or a slice would be averaged as
    #   if it were a whole fold.
    res: dict = {"gpu": a.gpu, "ks": list(KS), "repeats": REPEATS,
                 "axes": list(AXES), "arms": list(ARMS),
                 "tpe_seeds": a.tpe_seeds, "tpe_kmax": TPE_KMAX,
                 "tpe_kmax_note": ("TPE was not run above this k — see "
                                   "runs/x-autotune-aborted-k4096/. The "
                                   "other arms go to max(KS)."),
                 "tpe_multivariate": TPE_MULTIVARIATE,
                 "tpe_group": TPE_GROUP,
                 "shape_slice": a.shape_slice, "folds": {}}
    print("=" * 96)
    print(f"★ the autotuning curve — {a.gpu} (D-176 §1). 0 LLM · 0 GPU")
    print("=" * 96)
    for f in folds:
        sp = _splits(table, fold=f, k=4, design="nkband")
        hold = list(sp.val.shapes)
        if a.shape_slice:
            i, n = (int(x) for x in a.shape_slice.split("/"))
            hold = hold[i::n]
            if not hold:
                continue
        rows = []
        for p in hold:
            rng = np.random.default_rng(hash((p.M, p.N, p.K)) % (2 ** 31))
            df = table.frame_for(p)
            times = table.times_of(p)
            m = _prune_mask(df)
            curves = {"random": _random_curve(times, KS, rng),
                      "random_pruned": _random_curve(times, KS, rng, mask=m)}
            curves["random_cond"], crinfo = _cond_random_curve(df, times, KS,
                                                               rng)
            cs, infos = [], []
            for s_ in range(a.tpe_seeds):
                c, info = _tpe_curve(df, times, KS, seed=s_)
                cs.append(c)
                infos.append(info)
            # ⚠️ only the ks TPE was actually run at (<= TPE_KMAX). The
            #    rest are absent, ⛔ not filled in.
            curves["tpe"] = {k: float(np.mean([c[k] for c in cs]))
                             for k in cs[0]}
            rows.append({
                "shape": [p.M, p.N, p.K],
                "n_valid": int(len(times)), "n_pruned": int(m.sum()),
                "curves": curves,
                "cond_random_duplicate_rate": crinfo["duplicate_rate"],
                "tpe_duplicate_rate": round(float(np.mean(
                    [i["duplicate_rate"] for i in infos])), 4),
                "tpe_hit_cap": sum(1 for i in infos if i["hit_cap"]),
                "tpe_mean_domain_size": round(float(np.mean(
                    [i["mean_domain_size"] for i in infos])), 2),
                "tpe_n_startup_trials": infos[0]["n_startup_trials"]})
            print(f"  {p.M}x{p.N}x{p.K}  후보 {len(times):,}  "
                  f"TPE 중복 {rows[-1]['tpe_duplicate_rate']:.0%}  "
                  f"k=64 무작위 {curves['random']['64']:.4f} "
                  f"조건부 {curves['random_cond']['64']:.4f} "
                  f"TPE {curves['tpe']['64']:.4f}", flush=True)
        res["folds"][str(f)] = {"n_shapes": len(rows), "shapes": rows}
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
