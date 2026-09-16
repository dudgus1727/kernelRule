"""★ Which TPE settings — and does TPE use the axis structure at all?
(D-176 §1-3). **0 LLM calls · 0 GPU.**

    python3 -m experiments.tpe_settings --gpu a6000

⚠️ In **Optuna 5.0.0** `multivariate=None` is the default and resolves to
**`True`** for a single-objective study. So "we used the default" describes
nothing; the setting has to be chosen, and the choice has to be measured.

```
★ arm 0  random          the same draw-without-replacement arm as the curve
★ arm 1  mv=False        independent (univariate) TPE
★ arm 2  mv=True  g=False   joint over the ★ intersection ★ search space
★ arm 3  mv=True  g=True    joint over ★ decomposed subspaces
```

★ Arms 2 and 3 differ **because the parameter names carry the path**
(`tile_n|tile_m=128`). With `group=False` Optuna models jointly only the
parameters common to every completed trial, which after two different
branches is just the first axis — everything else falls back to univariate.
`group=True` decomposes instead, so a branch is modelled with the trials
that reached it. This script reports the intersection size so that is a
measured number and not a claim.

⛔ If no TPE arm beats random, that is the result and it is reported. The
arm chosen for the curve is the best of 1~3 **on these shapes**, and the
choice is recorded in `TPE_MULTIVARIATE` / `TPE_GROUP`.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from experiments.autotune_curve import _random_curve, _tpe_curve
from experiments.f1_pipeline import _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.core.scoring import geomean
from kernelrule.core.table import PerfTable

#: ★ Short ks — this is a settings probe, not the curve. The curve's own ks
#: go to 4096 (`autotune_curve.KS`).
#:
#: ⚠️ It stops at **256** because of arm 3. `multivariate=True, group=True`
#: re-decomposes the search space from every completed trial, and with the
#: path-in-name encoding each trial adds 11 fresh parameter names, so its
#: cost explodes: measured 3.5s at n=64, 31s at n=128, **210s at n=256** on
#: one shape and one seed. A first attempt at k<=512 was killed at 50
#: minutes without finishing 2 shapes. ⛔ That cost is itself a finding and
#: is recorded rather than hidden by quietly dropping the arm.
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
ARMS = {"tpe_uni": (False, False),
        "tpe_mv": (True, False),
        "tpe_mv_group": (True, True)}
OUT = Path("docs/artifacts/tpe-settings.json")


def _intersection_size(seed: int, df, times, n: int = 24) -> int:
    """★ How many parameter names survive Optuna's intersection search space
    after `n` conditional trials — the number `group=False` would model
    jointly."""
    import optuna

    from experiments.autotune_curve import AXES

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cols = {a: df[a].to_numpy() for a in AXES}
    st = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))

    def obj(trial):
        alive = np.ones(len(times), dtype=bool)
        path = ""
        for a in AXES:
            dom = sorted({x.item() if hasattr(x, "item") else x
                          for x in cols[a][alive]})
            v = trial.suggest_categorical(f"{a}{path}", dom)
            path += f"|{a}={v}"
            alive &= (cols[a] == v)
        return float(times[int(np.nonzero(alive)[0][0])])

    st.optimize(obj, n_trials=n, show_progress_bar=False)
    return len(optuna.search_space.IntersectionSearchSpace().calculate(st))


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="a6000")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-shapes", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    T = TABLES[a.gpu]
    table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                  ok_only=False)
    hold = list(_splits(table, fold=a.fold, k=4, design="nkband").val.shapes)
    # ★ A spread, not the first few: every n-th shape of the fold.
    step = max(1, len(hold) // a.n_shapes)
    hold = hold[::step][:a.n_shapes]

    print("=" * 92)
    print(f"★ TPE 설정 측정 — {a.gpu} fold{a.fold} · {len(hold)} 형상 "
          f"· seeds {a.seeds} · k<={max(KS)} (D-176 §1-3)")
    print("=" * 92)
    print("  ⚠️ Optuna 5.0.0 에서 multivariate 기본값(None)은 단일 목적일 때 "
          "True 로 풀린다 — 기본값은 설명이 되지 않는다")

    rows: dict = {m: [] for m in ("random", *ARMS)}
    dup: dict = {m: [] for m in ARMS}
    secs: dict = dict.fromkeys(ARMS, 0.0)
    inter: list[int] = []
    rng = np.random.default_rng(0)
    for p in hold:
        df = table.frame_for(p)
        times = table.times_of(p)
        rows["random"].append(_random_curve(times, KS, rng))
        inter.append(_intersection_size(0, df, times))
        for m, (mv, g) in ARMS.items():
            cs, ds = [], []
            t0 = time.time()
            for s in range(a.seeds):
                c, info = _tpe_curve(df, times, KS, seed=s,
                                     multivariate=mv, group=g)
                cs.append(c)
                ds.append(info["duplicate_rate"])
            secs[m] += time.time() - t0
            rows[m].append({k: float(np.mean([c[k] for c in cs]))
                            for k in map(str, KS)})
            dup[m].append(float(np.mean(ds)))

    agg = {m: {k: float(geomean(np.array([r[k] for r in v])))
               for k in map(str, KS)} for m, v in rows.items()}
    # ★ The pick: best area under the curve over k, among the TPE arms.
    score = {m: float(np.mean([agg[m][str(k)] for k in KS])) for m in ARMS}
    best = min(score, key=score.__getitem__)
    kend = str(max(KS))
    beats_random = agg[best][kend] < agg["random"][kend]
    # ⚠️ Written **before** anything is printed — a formatting slip must not
    #    destroy a run that took half an hour (it did once).
    Path(a.out).write_text(json.dumps(
        {"gpu": a.gpu, "fold": a.fold, "ks": list(KS), "seeds": a.seeds,
         "n_shapes": len(hold),
         "shapes": [[p.M, p.N, p.K] for p in hold],
         "optuna_version": __import__("optuna").__version__,
         "default_multivariate_note": ("Optuna 5.0.0: multivariate=None -> "
                                       "True for single-objective"),
         "curves": agg, "duplicate_rate": {m: float(np.mean(dup[m]))
                                           for m in ARMS},
         "sec_per_shape_per_seed": {m: round(secs[m] / (len(hold) * a.seeds),
                                             2) for m in ARMS},
         "intersection_names_median": int(np.median(inter)),
         "curve_mean": score, "chosen_arm": best,
         "chosen": {"multivariate": ARMS[best][0], "group": ARMS[best][1]},
         # ⚠️ The best arm here is **not** the one the k<=4096 curve runs.
         #    `group=True` re-decomposes the search space from every
         #    completed trial and the path-in-name encoding adds 11 fresh
         #    names per trial, so its cost is ~19x at k=256 and grows
         #    super-quadratically. ⛔ Recorded, not hidden: the curve is run
         #    with the affordable arm and this file says what that cost.
         "used_by_the_curve": {"arm": "tpe_uni", "multivariate": False,
                               "group": False},
         "why_not_the_best_arm": ("group=True was measured better but costs "
                                  "213.9 s per shape per seed at k<=256 "
                                  "against 11.2 s; at k=4096 x 3 seeds x "
                                  "256 shapes it is out of reach"),
         f"beats_random_at_{kend}": bool(beats_random)},
        ensure_ascii=False, indent=1))
    print(f"\n  {'k':>5} " + " ".join(f"{m:>13}" for m in agg))
    for k in (x for x in (1, 8, 64, 256, 512) if str(x) in agg["random"]):
        print(f"  {k:>5} " + " ".join(f"{agg[m][str(k)]:13.4f}" for m in agg))
    print(f"\n  {'중복률':>14} " + " ".join(
        f"{np.mean(dup[m]):13.1%}" for m in ARMS))
    print(f"  {'초/형상/시드':>14} " + " ".join(
        f"{secs[m] / (len(hold) * a.seeds):13.1f}" for m in ARMS))
    print(f"\n  ★ 교집합 탐색 공간 {int(np.median(inter))} 개 이름 "
          f"(24 시행 뒤 중앙) — group=False 가 함께 모형화하는 축 수")
    print(f"  ★ 고른 팔 {best}  (곡선 평균 {score[best]:.4f})"
          f"  · 무작위보다 나은가(k={kend}) {beats_random}")
    if not beats_random:
        print("  ⛔ TPE 가 무작위를 못 이긴다 — 그것이 결과다. 그대로 싣는다")
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
