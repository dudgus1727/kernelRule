"""★ `regret@k` — is the wall made by **the metric**? 0 LLM calls.

    python3 experiments/regret_at_k.py

The pre-registration is `docs/artifacts/regret-at-k-prereg.md`.

## Why

The rank loss **drops the pairs the noise cannot separate.** `tau` does not —
`kendalltau(variant="b")` treats them as tied **only when the times are
exactly equal**. In the holdout's top 100, 47.2% of pairs are unresolvable
while tau treats only 14.8% as tied. **32.4% get scored on an order that does
not exist.**

```
regret@k = (the mean true time of the top k the rule picked)
           / (the mean of the true top k)
```

It is a mean, so it is robust to noise, and predicting 38th place as 36th is
barely punished.

⚠️ 2026-09-08 (D-146): **the arm labels stay in Korean.** They are the row
names of `docs/artifacts/regret-at-k.md` and the keys of
`regret-at-k.json`, and `docs/` is not translated.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from two_stage import A6000, _fit, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY

KS = (1, 3, 5, 10, 20, 50, 100)
TOP_N = 100
N_DRAWS = 20
CATASTROPHE = 1.15

#: (label, the runs, the archive selection criterion, the weight-fitting
#: objective, k, λ)
ARMS: list[tuple] = [
    ("regret structure+regret w", [f"F3rw-p8-s{i}" for i in range(6)],
     "regret", "regret", 100, 0.0),
    ("★ regret structure+rank w", [f"F3rw-p8-s{i}" for i in range(6)],
     "regret", "rank", 100, 0.0),
    ("★ rank structure+regret w", [f"x-rank-rankevo-s{i}" for i in range(3)],
     "rank", "regret", 100, 0.0),
    ("rank structure+rank w", [f"x-rank-rankevo-s{i}" for i in range(3)],
     "rank", "rank", 100, 0.0),
    ("product term (prod)", [f"x-rank-prod-s{i}" for i in range(3)],
     "rank", "rank", 100, 0.0),
    ("k=10", [f"x-rank-k010-s{i}" for i in range(3)], "rank", "rank",
     10, 0.0),
    ("k=20", [f"x-rank-k020-s{i}" for i in range(3)], "rank", "rank",
     20, 0.0),
    ("k=50", [f"x-rank-k050-s{i}" for i in range(3)], "rank", "rank",
     50, 0.0),
    ("λ=1", [f"x-rank-lam10-s{i}" for i in range(3)], "rank", "rank",
     100, 1.0),
    ("budget 16", [f"x-rank-b16b-s{i}" for i in range(3)], "rank", "rank",
     100, 0.0),
]


def _best(run: str, by: str) -> dict:
    f = Path("runs") / run / "archive.jsonl"
    arc = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    key = (lambda e: e.get("rank_loss", 1e9)) if by == "rank" \
        else (lambda e: e["regret"])
    return sorted(arc, key=key)[0]


def _noise_ranks(t: np.ndarray, noise) -> np.ndarray:
    """★ The true ranks with **everything inside the noise tied together**.

    It sweeps in ascending time and opens a new group when
    `resolvable(the group's first element, t)` becomes True (single linkage).
    **It does not redefine the floor** — it uses `NoiseModel.resolvable` as
    it is (principle 2).
    """
    order = np.argsort(t, kind="stable")
    out = np.empty(len(t), dtype=np.float64)
    g, head = 0, t[order[0]]
    for i in order:
        if bool(noise.resolvable(np.array([head]), np.array([t[i]]))[0]):
            g += 1
            head = t[i]
        out[i] = g
    return out


def _measure(fn, ws, table, matrix, shapes) -> dict:
    """The per-shape regret@k / both taus / the catastrophe list."""
    rk = {k: [] for k in KS}
    tau_raw, tau_noise, per_shape1 = [], [], {}
    for p in shapes:
        cand = table.candidates(p)
        w = ws[regime_of(p, table.hw)] if isinstance(ws, dict) else ws
        s = np.asarray(make_score_of(fn, matrix, w)(p, cand), dtype=np.float64)
        t = np.asarray(table.times_of(p), dtype=np.float64)
        ts = np.sort(t)
        for k in KS:
            pick = np.asarray(cand.top_k(s, k))
            rk[k].append(float(t[pick].mean() / ts[:k].mean()))
        per_shape1[str(p.key)] = rk[1][-1]
        top = np.argsort(t, kind="stable")[:TOP_N]
        if len(np.unique(t[top])) > 1:
            v = kendalltau(s[top], t[top], variant="b").statistic
            if np.isfinite(v):
                tau_raw.append(float(v))
        nr = _noise_ranks(t, table.noise)[top]
        if len(np.unique(nr)) > 1:
            v = kendalltau(s[top], nr, variant="b").statistic
            if np.isfinite(v):
                tau_noise.append(float(v))
    return {"regret_at_k": {k: float(np.exp(np.mean(np.log(v))))
                            for k, v in rk.items()},
            "tau_raw": float(np.median(tau_raw)) if tau_raw else float("nan"),
            "tau_noise": (float(np.median(tau_noise)) if tau_noise
                          else float("nan")),
            "per_shape_r1": per_shape1}


def _floor(table, matrix, shapes, rng) -> dict:
    """★ The random floor (the mean of 20 draws). The floor is a sample too
    (principle 7)."""
    acc = {k: [] for k in KS}
    for _ in range(N_DRAWS):
        one = {k: [] for k in KS}
        for p in shapes:
            cand = table.candidates(p)
            t = np.asarray(table.times_of(p), dtype=np.float64)
            s = rng.random(len(t))
            ts = np.sort(t)
            for k in KS:
                pick = np.asarray(cand.top_k(s, k))
                one[k].append(float(t[pick].mean() / ts[:k].mean()))
        for k in KS:
            acc[k].append(float(np.exp(np.mean(np.log(one[k])))))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _row(label: str, vals: list[dict]) -> None:
    a = np.array([[v["regret_at_k"][k] for k in KS] for v in vals])
    print(f"  {label:20s} " + " ".join(
        f"{m:6.3f}" for m in np.median(a, axis=0)))
    # ★ The judgement is made by **the seed range** (pre-registration §5).
    #   Trimming it makes it unreadable.
    print(f"  {'':20s} " + " ".join(
        f"{a[:, i].min():.2f}-{a[:, i].max():.2f}" for i in range(len(KS))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/regret-at-k.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    out: dict = {"ks": list(KS), "n_holdout": len(hold)}

    print("=" * 92)
    print("§1  how tau treats the noise as it stands — the 20 holdout shapes")
    print("=" * 92)
    tot = res = eq = 0
    for p in hold:
        t = np.sort(np.asarray(T.times_of(p)))[:TOP_N]
        iu, ju = np.triu_indices(len(t), k=1)
        tot += iu.size
        res += int(T.noise.resolvable(t[iu], t[ju]).sum())
        eq += int((t[iu] == t[ju]).sum())
    print(f"  pairs within the top 100: {tot:,}")
    print(f"    pairs the noise cannot separate {tot - res:,} "
          f"({1 - res / tot:.1%})")
    print(f"    pairs tau-b treats as tied      {eq:,} ({eq / tot:.1%})  "
          "← only those with exactly equal times")
    print(f"  ★ the difference, {(tot - res - eq) / tot:.1%}, **gets scored "
          f"on an order that does not exist**")
    out["pairs"] = {"total": tot, "unresolvable": tot - res, "tied": eq}

    print("\n" + "=" * 92)
    print("§2  regret@k — all on the 20 holdout shapes. Top=median, "
          "bottom=seed range")
    print("=" * 92)
    print(f"  {'':20s} " + " ".join(f"{'k=' + str(k):>6}" for k in KS)
          + "\n" + f"  {'':20s} " + " ".join(f"{'(range)':>9}" for _ in KS))
    rows: dict[str, list[dict]] = {}
    for label, runs, by, obj, k, lam in ARMS:
        if not all((Path("runs") / r / "archive.jsonl").exists() for r in runs):
            print(f"  {label:20s} (no such run — skipped)")
            continue
        vals = []
        for r in runs:
            e = _best(r, by)
            fn, ws = _fit(e["code"], e["w"], T, M, train, obj,
                          rank_top_k=k, rank_lambda=lam if obj == "rank" else 0.0)
            vals.append(_measure(fn, ws, T, M, hold))
        rows[label] = vals
        _row(label, vals)
    fl = _floor(T, M, hold, np.random.default_rng(0))
    print(f"  {'★ random floor':20s} " + " ".join(f"{fl[k]:6.3f}" for k in KS))
    out["floor"] = {str(k): v for k, v in fl.items()}
    out["arms"] = {lab: [v["regret_at_k"] for v in vs]
                   for lab, vs in rows.items()}

    print("\n" + "=" * 92)
    print("§3  (a) per-shape regret@1 — where the catastrophes (>1.15) come "
          "from")
    print("=" * 92)
    print(f"  {'':22s} {'catastrophic shapes (median/20)':>32}  seed range"
          f"   union")
    cat: dict[str, set] = {}
    for label, vs in rows.items():
        sets = [{s for s, v in x["per_shape_r1"].items() if v > CATASTROPHE}
                for x in vs]
        allc = set().union(*sets)
        cat[label] = allc
        ns = [len(s) for s in sets]
        print(f"  {label:22s} {np.median(ns):32.1f}  "
              f"{min(ns):2d}~{max(ns):-2d}"
              f"      {len(allc):2d}")
    if cat:
        common = set.intersection(*[c for c in cat.values() if c]) \
            if all(cat.values()) else set()
        allu = set().union(*cat.values())
        print(f"\n  ★ shapes catastrophic in every arm: {len(common)} / "
              f"shapes catastrophic in any arm: {len(allu)}")
        print("  -> " + ("closer to a property of the shape"
                         if len(common) >= 0.5 * len(allu)
                         else "closer to a property of the rule"))
        print("  ⚠️ the rank arms have 13~17 of the 20 shapes catastrophic — "
              "**it is not concentrated in a few shapes.**")
        out["catastrophe"] = {"per_arm": {k: sorted(v) for k, v in cat.items()},
                              "common": sorted(common), "any": sorted(allu)}

    print("\n" + "=" * 92)
    print("§4  (b) the noise-aware tau — beside the old tau")
    print("=" * 92)
    print(f"  {'':22s} {'old tau':>9} {'noise-aware':>12} {'diff':>8}")
    for label, vs in rows.items():
        r = float(np.median([v["tau_raw"] for v in vs]))
        n = float(np.median([v["tau_noise"] for v in vs]))
        print(f"  {label:22s} {r:9.3f} {n:12.3f} {n - r:+8.3f}")
        out.setdefault("tau", {})[label] = {"raw": r, "noise": n}

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3 and 6 seeds cannot give significance — it is read by the "
          "seed range (principle 27)")


if __name__ == "__main__":
    main()
