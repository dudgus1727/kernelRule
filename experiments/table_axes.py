"""★ How wide the answer set is **per table** — the third axis of the
transfer loss. 0 LLM calls.

    python3 experiments/table_axes.py

What D-126 observed: the transfer loss is set by **the target, not the
source**, and that order is the same as the order of that table's native
score. Neither the ridge nor the bound flips can produce that order. **The
candidate that remains is "how narrow a target first place is on that
table".**

```
answer-set size   the number of configs **not separable by noise** from best
                  (NoiseModel)
top-rank spread   (t_k - t_1) / t_1  — how many % slower kth place is than
                  first
```

★ No new threshold is made — it is `table.noise.resolvable` as it is
(principle 2).
⚠️ This is **an observation**. It attaches to D-126's post-hoc observation and
is not a cause.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from kernelrule.core.crosstable import common_shapes
from kernelrule.core.table import PerfTable

TABLES = {
    "a6000": ("datasets/rtx-a6000-sm_86-c63710df", "c63710df"),
    "4090": ("datasets/rtx-4090-sm_89-ad95d455", "ad95d455"),
    "5090": ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403"),
    # ★ 2026-09-06 (D-141). ⚠️ Only this table's answer set has a dispute over
    #   the noise coefficients — `--sigma-rel` can swap the coefficient in and
    #   report the range of the effect in parentheses.
    "h100": ("datasets/h100-nvl-sm_90-63684546", "63684546"),
}
KS = (10, 100)


def _axes(T: PerfTable, shapes) -> dict:
    ans, spread = [], {k: [] for k in KS}
    for p in shapes:
        t = np.sort(np.asarray(T.times_of(p), dtype=np.float64))
        best = t[0]
        # ★ The number of configs not separable by noise from best = "the
        #   answer set"
        ok = T.noise.resolvable(np.full(t.shape, best), t)
        ans.append(int((~ok).sum()))
        for k in KS:
            if len(t) > k:
                spread[k].append(float((t[k - 1] - best) / best))
    return {"n_shapes": len(list(shapes)),
            "answer_set": ans,
            "answer_median": float(np.median(ans)),
            "spread": {k: float(np.median(v)) for k, v in spread.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/table-axes.json")
    ap.add_argument("--sigma-rel", type=float, default=None, metavar="X",
                    help="recount with these tables' sigma_rel replaced by X. "
                         "★ It is for reporting in parentheses how far the "
                         "coefficient reaches into the answer set")
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = {n: PerfTable.from_bundle(b, env_hash=h, ok_only=False)
         for n, (b, h) in TABLES.items()}
    # ★ It is measured only on the shapes **present in every registered
    #   table** — a different shape sweep (the 4090 is on the large-M side)
    #   would measure the shape composition rather than the table's property.
    common = common_shapes(T["a6000"], T["4090"])
    keys = {(p.M, p.N, p.K) for p in common} & {
        (p.M, p.N, p.K) for p in common_shapes(T["a6000"], T["5090"])}
    if "h100" in T:
        keys &= {(p.M, p.N, p.K)
                 for p in common_shapes(T["a6000"], T["h100"])}
    if a.only:
        T = {n: t for n, t in T.items() if n in a.only}
    if a.sigma_rel is not None:
        # ★ Only the coefficient is swapped. **Everything else stays** — it is
        #   for reporting the range of the effect in parentheses, not for
        #   fixing the coefficient (D-141).
        import dataclasses
        for n, t in T.items():
            # `noise` is a read-only property — the backing field is changed.
            t._noise = dataclasses.replace(
                t.noise, sigma_rel_coef=a.sigma_rel,
                source=t.noise.source + f" [★ sigma_rel swapped to {a.sigma_rel}]")
        print(f"  ★ measuring with sigma_rel changed to {a.sigma_rel} — this "
              f"is not the original")
    print("=" * 88)
    print(f"the table's axes — measured only on the {len(keys)} shapes present "
          f"in all {len(T)} tables (0 LLM calls)")
    print("=" * 88)
    print(f"  {'table':8s} {'answer set (median)':>20s} {'distribution':>24s} "
          f"{'10th vs 1st':>13s} {'100th':>10s}")
    out: dict = {"n_common": len(keys), "ks": list(KS)}
    for n, t in T.items():
        sh = [p for p in t.shapes() if (p.M, p.N, p.K) in keys]
        r = _axes(t, sh)
        q = np.percentile(r["answer_set"], [25, 75])
        dist = f"[{q[0]:.0f}, {q[1]:.0f}]  max {max(r['answer_set'])}"
        print(f"  {n:8s} {r['answer_median']:20.1f} {dist:>24s} "
              f"{r['spread'][10]:12.1%} {r['spread'][100]:10.1%}")
        out[n] = r
    print()
    print("  ★ the wider the answer set, the more it is a table where "
          "'anything will do' —")
    print("     moving someone else's weights over is punished less "
          "(D-126's hypothesis)")
    print("  ⚠️ it is an observation. There are three tables and it is a cut "
          "the pre-registration did not have (principle 27)")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
