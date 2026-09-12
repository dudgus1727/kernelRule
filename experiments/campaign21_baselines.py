"""★ The 21-run campaign against the baselines, on each split's own holdout.
**0 LLM calls · the committed vendor files are read, never regenerated.**

    python3 experiments/campaign21_baselines.py

```
vendor        nvMatmulHeuristics' per-shape recommendation, from
              datasets/baselines/vendor-<gpu>-<hash>.json (D-158)
              "nearest" and "strict" mapping are both reported (D-158 §3)
static top-1  ★ the **ceiling** of "one config for everything": the single
              config that best covers those very holdout shapes
              (facility-location greedy, the same call D-157 and
              `transfer_generated.py` make). It is fitted on the shapes it
              is scored on, so it is not a trained baseline — it is the
              best a fixed config could possibly do there
our rule      the median of the three seeds' canonical holdout
```

⚠️ **Same shapes, same denominator, same aggregation** (documentation rule
3): every number here is `regret@1`, geometric mean over *that split's*
holdout shapes. ⛔ But the static column is **not** under the same rule as
the other two — see above. Read it as a ceiling, never as a competitor.

⚠️ This is the geomean comparison. The per-shape sign test the re-run
pre-registration calls the main metric needs each run's per-shape regret,
which means refitting all 21 rules — `--per-shape` does that and costs
hours.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import warnings
from pathlib import Path

OUT = Path("docs/artifacts/campaign-21-baselines.json")
GROUPS = [
    ("a6000", "fold0", {"fold": 0}), ("a6000", "fold1", {"fold": 1}),
    ("a6000", "fold2", {"fold": 2}), ("a6000", "nk11008", {}),
    ("5090", "nk11008", {}), ("4090", "nk11008", {}), ("h100", "nk11008", {}),
]
TAGS = {("a6000", "fold0"): "c21-a6000-f0", ("a6000", "fold1"): "c21-a6000-f1",
        ("a6000", "fold2"): "c21-a6000-f2", ("a6000", "nk11008"): "c21-a6000-nk",
        ("5090", "nk11008"): "c21-5090-nk", ("4090", "nk11008"): "c21-4090-nk",
        ("h100", "nk11008"): "c21-h100-nk"}


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.parse_args()

    from experiments.f1_pipeline import _splits
    from experiments.transfer_29_5 import TABLES
    from kernelrule.baselines.static_topk import StaticTopK
    from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
    from kernelrule.core.scoring import evaluate, geomean
    from kernelrule.core.table import PerfTable

    per_run = {f.stem: json.loads(f.read_text())
               for f in Path("runs/_campaign").glob("c21-*.json")}
    tables: dict[str, PerfTable] = {}
    out: dict = {"note": ("regret@1, geometric mean over that split's "
                          "holdout shapes. The vendor files are the "
                          "committed ones (D-158) and were not regenerated."),
                 "groups": {}}

    print(f"  {'table':6s} {'split':8s} {'n':>3s} {'vendor':>8s} "
          f"{'strict':>8s} {'stat1★':>8s} {'ours(med)':>10s} "
          f"{'ours best':>10s}  verdict")
    for gpu, split, kw in GROUPS:
        if gpu not in tables:
            tables[gpu] = PerfTable.from_bundle(
                TABLES[gpu]["bundle"], env_hash=TABLES[gpu]["env_hash"],
                ok_only=False)
        t = tables[gpu]
        sp = _splits(t, split_seed=12345, k=3, **kw)
        hold = list(sp.val.shapes)

        vpath = Path(f"datasets/baselines/"
                     f"vendor-{gpu}-{TABLES[gpu]['env_hash'][:8]}.json")
        vend = load_vendor(vpath)
        v = {}
        for mapping in ("nearest", "strict"):
            ev = evaluate(vendor_order_fn(t, vend, mapping=mapping), t, hold,
                          ks=(1,), label=f"vendor-{mapping}")
            v[mapping] = {"geomean": float(geomean(ev.regret[:, 0])),
                          "per_shape": [float(x) for x in ev.regret[:, 0]]}

        # ★ On the holdout shapes themselves — a ceiling, not a baseline.
        s1 = StaticTopK(t, hold, coverage="union").run(ks=(1,))
        static_hold = float(s1.by_k[1]["all"])

        tags = [f"{TAGS[(gpu, split)]}-s{k}" for k in range(3)]
        ours = [per_run[x]["holdout"] for x in tags if x in per_run]
        med = st.median(ours) if ours else float("nan")
        best = min(ours) if ours else float("nan")
        vn = v["nearest"]["geomean"]
        verdict = ("★ 우리가 낫다" if med < vn else
                   "벤더가 낫다" if med > vn else "같다")
        out["groups"][f"{gpu}/{split}"] = {
            "n_holdout": len(hold), "vendor_nearest": vn,
            "vendor_strict": v["strict"]["geomean"],
            "vendor_per_shape_nearest": v["nearest"]["per_shape"],
            "shapes": [f"{p.M}x{p.N}x{p.K}" for p in hold],
            "static_top1_ceiling_on_holdout": static_hold, "ours": ours,
            "ours_median": med, "ours_best": best,
            "delta_median_minus_vendor": med - vn}
        print(f"  {gpu:6s} {split:8s} {len(hold):3d} {vn:8.4f} "
              f"{v['strict']['geomean']:8.4f} {static_hold:8.4f} "
              f"{med:10.4f} {best:10.4f}  {verdict}")

    wins = sum(1 for g in out["groups"].values()
               if g["ours_median"] < g["vendor_nearest"])
    out["groups_where_ours_is_better"] = wins
    out["n_groups"] = len(out["groups"])
    print(f"\n  ★ 일곱 그룹 중 우리 중앙값이 벤더보다 나은 곳: {wins}/"
          f"{len(out['groups'])}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"  recorded: {OUT}")


if __name__ == "__main__":
    main()
