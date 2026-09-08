"""★ Cascade — the regret rule picks the region, the rank rule the order.
0 LLM calls.

    python3 experiments/cascade.py

The pre-registration is `docs/artifacts/cascade-prereg.md`.

```
stage 1   the regret rule narrows to the top k      (region selection)
stage 2   the rank rule picks first place inside it (the order)
```

The wall's two statements are complementary, so they are joined (D-118). **It
does not remove the wall; it goes around it at deployment time.**
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from two_stage import A6000, _fit, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.runset import assert_same_condition
from kernelrule.core.splits import regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY

KS = (10, 20, 50)
N_DRAWS = 20
REG_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
RANK_RUNS = [f"x-rank-rankevo-s{i}" for i in range(3)]


def _best(run: str, by: str) -> dict:
    arc = [json.loads(x) for x in
           (Path("runs") / run / "archive.jsonl").read_text().splitlines()
           if x.strip()]
    key = (lambda e: e.get("rank_loss", 1e9)) if by == "rank" \
        else (lambda e: e["regret"])
    return sorted(arc, key=key)[0]


def _scores(fn, ws, table, matrix, shapes) -> dict:
    """The score array per shape. It is pulled out in advance so the fit runs
    only once."""
    out = {}
    for p in shapes:
        cand = table.candidates(p)
        w = ws[regime_of(p, table.hw)]
        out[p.key] = np.asarray(make_score_of(fn, matrix, w)(p, cand),
                                dtype=np.float64)
    return out


def _geo(v) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=np.float64)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/cascade.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    assert_same_condition(REG_RUNS, label="the regret arm")
    assert_same_condition(RANK_RUNS, label="the rank arm")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    times = {p.key: np.asarray(T.times_of(p), dtype=np.float64) for p in hold}
    cands = {p.key: T.candidates(p) for p in hold}
    out: dict = {"ks": list(KS), "n_holdout": len(hold)}

    print("=" * 86)
    print("Cascade — stage 1 the regret rule (the region) -> stage 2 the rank "
          "rule (the order)")
    print("=" * 86)
    print(f"  A6000 holdout {len(hold)} shapes · the regret arm "
          f"{len(REG_RUNS)} structures x the rank arm {len(RANK_RUNS)} "
          f"structures = {len(REG_RUNS) * len(RANK_RUNS)} combinations\n")

    # -- the fit runs once per structure ---------------------------------
    S1 = {}
    for r in REG_RUNS:
        e = _best(r, "regret")
        S1[r] = _scores(*_fit(e["code"], e["w"], T, M, train, "regret"),
                        T, M, hold)
    S2 = {}
    for r in RANK_RUNS:
        e = _best(r, "rank")
        S2[r] = _scores(*_fit(e["code"], e["w"], T, M, train, "rank"),
                        T, M, hold)

    # -- the baseline: regret alone --------------------------------------
    solo = []
    for r in REG_RUNS:
        solo.append(_geo([times[p.key][int(cands[p.key].top_k(S1[r][p.key],
                                                              1)[0])]
                          / times[p.key].min() for p in hold]))
    print(f"  baseline  regret alone, regret@1   median {np.median(solo):.4f}"
          f"   range {min(solo):.4f}~{max(solo):.4f}")
    out["solo"] = solo

    rng = np.random.default_rng(0)
    print(f"\n  {'k':>4} {'★ cascade median':>18} {'range':>17} "
          f"{'ceiling oracle@k':>18} {'floor random-in-k':>19} {'hit@k':>7}")
    for k in KS:
        # The ceiling / the floor / hit@k are per stage-1 structure
        ceil_, floor_, hit_ = [], [], []
        casc = []
        for r1 in REG_RUNS:
            picks = {p.key: np.asarray(cands[p.key].top_k(S1[r1][p.key], k))
                     for p in hold}
            ceil_.append(_geo([times[p.key][picks[p.key]].min()
                               / times[p.key].min() for p in hold]))
            hit_.append(float(np.mean(
                [bool(times[p.key][picks[p.key]].min()
                      <= times[p.key].min() + 0.0) for p in hold])))
            draws = []
            for _ in range(N_DRAWS):
                draws.append(_geo(
                    [times[p.key][rng.choice(picks[p.key])]
                     / times[p.key].min() for p in hold]))
            floor_.append(float(np.mean(draws)))
            for r2 in RANK_RUNS:
                v = []
                for p in hold:
                    idx = picks[p.key]
                    j = idx[int(np.argmin(S2[r2][p.key][idx]))]
                    v.append(times[p.key][j] / times[p.key].min())
                casc.append(_geo(v))
        print(f"  {k:>4} {np.median(casc):18.4f} "
              f"{min(casc):8.4f}~{max(casc):<8.4f} {np.median(ceil_):18.4f} "
              f"{np.median(floor_):19.4f} {np.median(hit_):7.1%}")
        out.setdefault("cascade", {})[str(k)] = casc
        out.setdefault("ceiling", {})[str(k)] = ceil_
        out.setdefault("floor_in_k", {})[str(k)] = floor_
        out.setdefault("hit_at_k", {})[str(k)] = hit_

    # -- the verdict ------------------------------------------------------
    print("\n" + "=" * 86)
    print("the verdict — the line nailed down in the pre-registration")
    print("=" * 86)
    lo, hi = min(solo), max(solo)
    print(f"  the seed range of regret alone  {lo:.4f}~{hi:.4f}")
    for k in KS:
        m = float(np.median(out["cascade"][str(k)]))
        v = ("★ it goes around" if m < lo else
             "★ it does not (inside the range)" if m <= hi
             else "★ stage 2 does harm")
        f = float(np.median(out["floor_in_k"][str(k)]))
        c = float(np.median(out["ceiling"][str(k)]))
        pos = (m - c) / (f - c) if f > c else float("nan")
        print(f"  k={k:<3d} cascade {m:.4f}   {v}")
        print(f"        ceiling {c:.4f} · floor {f:.4f} → "
              f"stage 2 closes {1 - pos:.0%} of the gap between them")

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ the 18 combinations are not independent samples (they come "
          "from 9 structures). No significance is given (principle 27)")


if __name__ == "__main__":
    main()
