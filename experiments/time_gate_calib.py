"""★ The time-axis check (D-192) on axes whose behaviour is already known —
before it judges anything new. **LLM 0회 · GPU 0. train 형상만.**

    python3 -m experiments.time_gate_calib --out docs/artifacts/time-gate/calib.json

```
hand-built (D-189)   the time-model pool that took a 3-weight rule past the
                     vendor — these should pass
d190 stage 1         the axes the FeatureWriter labelled "time ratio" in the
                     D-190 campaign, found monotone by the analysis — these
                     should be caught
```

The unit gate is not applied here (the hand-built pool says "log2 ratio"):
every listed axis is diagnosed. Fold 0's training shapes only.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

HAND = ("docs/artifacts/a6000-rules/features/pool_current.jsonl",
        ("tm_log_est", "splitk_roofline_log_time", "log_roofline_time_mraster",
         "splitk_excess_log", "tm_crit_ratio"))
D190 = [(f"runs/d190-a6000-f{f}/stage1-features/proposals.jsonl", None)
        for f in range(4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    # ★ 한 캠페인의 stage-1 시간 축을, fold 마다 그 fold 의 train 으로
    ap.add_argument("--campaign", default=None,
                    help="e.g. tgate — runs/<c>-a6000-f{f}/stage1-features")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    import kernelrule.features.physical  # noqa: F401
    from experiments import a6000_probe as P
    from experiments.f1_pipeline import _splits
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import FeatureRegistry
    from kernelrule.features.loader import load_generated
    from kernelrule.features.time_gate import diagnose, verdict

    P._G["gpu"] = "a6000"
    t = P._table()
    trains = {f: list(_splits(t, fold=f, k=4, design="nkband").train.shapes)
              for f in range(4)}
    train = trains[a.fold]
    out = []
    if a.campaign:
        groups = [(f"{a.campaign} f{f} stage 1",
                   Path(f"runs/{a.campaign}-a6000-f{f}/stage1-features/"
                        "proposals.jsonl"), None, f) for f in range(4)]
    else:
        groups = [("hand-built (D-189)", Path(HAND[0]), set(HAND[1]),
                   a.fold)]
        groups += [(f"d190 f{i} stage 1", Path(p), None, a.fold)
                   for i, (p, _x) in enumerate(D190)]
    for label, path, only, fold in groups:
        train = trains[fold]
        for f in load_generated(path, table=t):
            if f.shape_level:
                continue
            if only is not None and f.name not in only:
                continue
            if only is None and "time" not in str(f.unit).lower():
                continue
            reg = FeatureRegistry(f"calib-{f.name}")
            reg.add(f)
            probe = FeatureMatrix(t, reg)
            d = diagnose(f.name, f.direction, t, probe, train)
            v = verdict(d)
            out.append({"group": label, "name": f.name, "unit": f.unit,
                        "diag": d, "verdict": v})
            sl = " ".join(f"{k}:{s['end_miss']}/{s['interior']}/{s['n']}"
                          for k, s in d["slices"].items())
            print(f"  {label:20s} {f.name:34s} rank {d['rank']:+.2f}  pick "
                  f"{d['pick_gm']:.3f}  end-miss/interior/n {sl}  "
                  f"{'PASS' if v is None else 'CAUGHT'}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"fold": a.fold, "campaign": a.campaign,
                                 "n_train": len(train),
                                 "axes": out}, ensure_ascii=False, indent=1))
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
