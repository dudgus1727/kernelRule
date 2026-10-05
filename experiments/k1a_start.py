"""★ 개선안 1번 진단의 **출발점** — K1a 를 루프의 적합으로 fold 마다 맞춘 값.
**LLM 0회 · GPU 0.**

    python3 -m experiments.k1a_start --out docs/artifacts/k1a-loop/start.json

루프의 `seed()` 와 같은 길이다: `loop._fit_rule` (u · dim), 시작은 K1a 의 w0, 행렬은
그 fold 의 stage 1 라이브러리 (`runs/k1a-a6000-f{f}`, 루프 축이 생기기 전). 그래서
trace 의 첫 `round_start.archive_best` 와 train regret 이 같아야 한다 — 그것을 대조한다.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

FOLDS = (0, 1, 2, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="k1a")
    ap.add_argument("--gpu", default="a6000",
                    choices=("a6000", "5090", "4090", "h100"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    import kernelrule.features.physical  # noqa: F401
    from experiments import a6000_probe as P
    from experiments.f1_pipeline import _load_stage1, _splits
    from kernelrule.core.loop import _fit_rule
    from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.features import REGISTRY
    from kernelrule.features.loader import base_registry

    P._G["gpu"] = a.gpu
    t = P._table()
    out = {}
    for f in FOLDS:
        d = Path(f"runs/{a.prefix}-{a.gpu}-f{f}")
        seed = json.loads((d / "stage2-rule-writer" / "chosen.json")
                          .read_text())
        sp = _splits(t, fold=f, k=4, design="nkband")
        # ★ D-193 — the campaign's own condition (F2 before it)
        cond = P._condition_of(d)
        reg = _load_stage1(d, base_registry(cond, human=REGISTRY), cond, t)
        m = FeatureMatrix(t, reg, cache_dir=CACHE_DIR)
        fr = _fit_rule(compile_rule(seed["code"]), seed["code"], seed["w0"],
                       None, matrix=m, table=t, train=sp.train, val=sp.val,
                       objective="regret", rank_top_k=100, rank_lambda=0.0,
                       fit_space="u", fit_budget="dim")
        # ★ D-193 — a campaign that stopped after stage 2 has no loop trace;
        #   the stage-2 score of the same fit (`chosen.json`) is the check
        tr = Path(f"runs/{a.prefix}-{a.gpu}-f{f}-s0/trace.jsonl")
        ev = ([json.loads(x) for x in tr.read_text().splitlines() if x]
              if tr.exists() else [])
        rs = next((e for e in ev if e.get("ev") == "round_start"),
                  {"archive_best": seed.get("fit_regret")})
        out[f] = {"n_val": len(sp.val.shapes), "train": fr.fit_regret,
                  "holdout": fr.val_regret, "w": [float(x) for x in fr.w],
                  "trace_seed_train": rs.get("archive_best")}
        print(f"  f{f}  train {fr.fit_regret:.4f} (trace "
              f"{rs.get('archive_best')})  holdout {fr.val_regret:.4f}",
              flush=True)
    n = sum(v["n_val"] for v in out.values())
    pooled = float(np.exp(sum(v["n_val"] * np.log(v["holdout"])
                              for v in out.values()) / n))
    a.out.write_text(json.dumps({"what": "K1a fitted by the loop's own fit, "
                                 "per fold — the loop's starting point",
                                 "pooled65_holdout": pooled, "folds": out},
                                ensure_ascii=False, indent=1))
    print(f"  pooled 65 holdout {pooled:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
