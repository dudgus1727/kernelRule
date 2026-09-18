"""★ 캠페인 2 전이 48칸을 다시 낸다 (D-183 §3). **0 LLM.**

    python3 -m experiments.c2_transfer_rescore --fold <f>   # 갈래
    python3 -m experiments.c2_transfer_rescore --merge <f...>

`transfer_nk4.py` 는 ⛔ **못 돈다** — `_fit_per_regime` / `_score_on` 이
`regime_of` 를 축 없이 부른다 (D-179). 그 축은 없어졌고 그 파일은 옛 수치가
어떻게 나왔는지의 기록으로 남는다.

## ⚠️ 팔의 뜻이 바뀐다 — 적어 둔다

```
             옛 (D-182 이전)                    ★ 새
(a) 그대로   소스의 ★ 두 벌 가중치를 대상에      소스의 ★ 한 벌을 대상 홀드아웃에
             체제별로 적용                       (canonical_score · ★ 적합 0회)
(b) 재적합   대상 학습에서 ★ 체제별 두 벌 적합    대상 학습에서 ★ 한 벌 적합
             -> 그 두 벌로 채점                   -> 그 한 벌로 채점
원주민       대상의 규칙을 ★ 다시 적합해 채점     ★ 대상 루프가 끝낸 가중치 그대로
                                                 = campaign2 의 홀드아웃과 ★ 같은 자
```

★ **원주민이 캠페인 표와 같은 자가 된 것이 이번의 이득이다.** 예전에는
전이표의 원주민과 `campaign2.json` 의 홀드아웃이 서로 다른 절차였다.

⛔ (b) 만 적합을 쓴다 — "가중치만 재적합" 이 그 팔의 정의이기 때문이다.
그것은 `canonical_score` 가 아니라 이 파일이 하는 일이다 (D-182 §⑤ 와 같은
갈래: 적합은 실험의 것이지 채점기의 것이 아니다).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.f1_pipeline import _load_stage1, _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.baselines.static_topk import StaticTopK
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.scoring import evaluate, evaluate_scores, geomean
from kernelrule.core.splits import Split
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry, load_generated

GPUS = ("a6000", "5090", "4090", "h100")
SEEDS = (0, 1, 2, 3)
DESIGN = "nkband"
OUT = Path("docs/artifacts/campaign2-transfer.json")
VENDOR = {g: f"datasets/baselines/vendor-{g}-{TABLES[g]['env_hash'][:8]}.json"
          for g in GPUS}


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _best_of(gpu: str, fold: int):
    """★ 그 (표, fold) 가 끝낸 규칙 — 4시드 중 ★ 학습 점수 최소.
    ⛔ 홀드아웃을 보지 않는다. 각 시드의 답은 `bests.jsonl` 의 마지막 줄이다
    (D-183 §2-1: 동점에서 `min(archive)` 와 갈린다)."""
    out = None
    for s in SEEDS:
        bp = Path(f"runs/c2-{gpu}-f{fold}-s{s}/bests.jsonl")
        if not bp.exists():
            continue
        e = _rows(bp)[-1]
        if out is None or e["regret"] < out[0]["regret"]:
            out = (e, s)
    return out if out else (None, None)


def _registry(gpu: str, fold: int, seed: int, table):
    reg = _load_stage1(Path(f"runs/c2-{gpu}-f{fold}"),
                       base_registry("F2", human=REGISTRY), "F2", table)
    fp = Path(f"runs/c2-{gpu}-f{fold}-s{seed}/features.jsonl")
    if fp.exists():
        for f in load_generated(fp, table=table):
            if f.name not in reg._items:
                reg.add(f)
    return reg


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.merge:
        cells = []
        for f in a.merge:
            cells += json.loads(Path(f).read_text())["cells"]
        cells.sort(key=lambda c: (c["fold"], c["src"], c["dst"]))
        Path(a.out).write_text(json.dumps(
            {"n_cells": len(cells), "campaign": "c2", "design": DESIGN,
             "note": ("★ D-183 재집계. (a)·원주민은 canonical_score (★ 적합 "
                      "0회), (b) 만 대상 학습에서 ★ 한 벌 적합. 옛 표는 "
                      "체제별 두 벌이었고 원주민도 다시 적합했다."),
             "cells": cells, "summary": _summary(cells)},
            ensure_ascii=False, indent=1))
        print(f"merged -> {a.out}  ({len(cells)} cells)")
        _print(_summary(cells))
        return

    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import fit_weights, make_score_of

    folds = range(4) if a.fold is None else (a.fold,)
    tables = {g: PerfTable.from_bundle(TABLES[g]["bundle"],
                                       env_hash=TABLES[g]["env_hash"],
                                       ok_only=False) for g in GPUS}
    cells = []
    print("=" * 104)
    print("★ 캠페인 2 전이 48칸 — 재집계 (D-183 §3). 0 LLM")
    print("=" * 104)
    for f in folds:
        sp = {g: _splits(tables[g], fold=f, k=4, design=DESIGN) for g in GPUS}
        native_cache: dict = {}
        for src in GPUS:
            e, seed = _best_of(src, f)
            if e is None:
                continue
            for dst in GPUS:
                if dst == src:
                    continue
                tB, sB = tables[dst], sp[dst]
                hold = list(sB.val.shapes)
                regB = _registry(src, f, seed, tB)
                mB = FeatureMatrix(tB, regB, cache_dir=CACHE_DIR)
                w = np.asarray(e["w"], float)
                # (a) ★ the source's weights as they are — 0 fits
                ga = canonical_score(e["code"], w, table=tB, matrix=mB,
                                     splits=sB).holdout
                # (b) ★ one refit on the target's training split
                fn = compile_rule(e["code"])
                fr = fit_weights(fn, mB, tB, Split("train",
                                                   tuple(sB.train.shapes)),
                                 w, max_evals=300, objective="regret")
                gb = float(geomean(evaluate_scores(
                    make_score_of(fn, mB, fr.w), tB, hold,
                    ks=(1,)).regret[:, 0]))
                # native ★ = the target's own loop answer, scored as it is
                if (dst, f) not in native_cache:
                    ne, ns = _best_of(dst, f)
                    if ne is None:
                        native_cache[(dst, f)] = None
                    else:
                        mN = FeatureMatrix(tB, _registry(dst, f, ns, tB),
                                           cache_dir=CACHE_DIR)
                        native_cache[(dst, f)] = canonical_score(
                            ne["code"], np.asarray(ne["w"], float),
                            table=tB, matrix=mN, splits=sB).holdout
                gn = native_cache[(dst, f)]
                vend = load_vendor(VENDOR[dst])
                gv = float(geomean(evaluate(
                    vendor_order_fn(tB, vend, mapping="nearest"), tB, hold,
                    ks=(1,), label="vendor").regret[:, 0]))
                gs = float(StaticTopK(tB, hold, coverage="union"
                                      ).run(ks=(1,)).by_k[1]["all"])
                cells.append({
                    "fold": f, "src": src, "dst": dst, "src_seed": seed,
                    "n_holdout": len(hold),
                    "a_as_is": round(float(ga), 6),
                    "b_refit": round(float(gb), 6),
                    "native": (round(float(gn), 6) if gn is not None
                               else None),
                    "vendor": round(gv, 6), "static_top1": round(gs, 6),
                    "b_moved": bool(fr.moved)})
                c = cells[-1]
                print(f"  f{f} {src:5s}->{dst:5s}  (a) {c['a_as_is']:.4f}  "
                      f"(b) {c['b_refit']:.4f}  원주민 "
                      f"{c['native']:.4f}  벤더 {c['vendor']:.4f}",
                      flush=True)
    out = Path(a.out)
    out.write_text(json.dumps({"n_cells": len(cells), "cells": cells},
                              ensure_ascii=False, indent=1))
    print(f"\n  -> {out}")


def _summary(cells: list[dict]) -> dict:
    ok = [c for c in cells if c["native"] is not None]
    ga = [c["a_as_is"] - c["native"] for c in ok]
    gb = [c["b_refit"] - c["native"] for c in ok]
    return {"n": len(ok),
            "a_minus_native_median": round(st.median(ga), 6),
            "b_minus_native_median": round(st.median(gb), 6),
            "a_beats_native": sum(1 for x in ga if x < 0),
            "b_beats_native": sum(1 for x in gb if x < 0),
            "b_beats_vendor": sum(1 for c in ok
                                  if c["b_refit"] < c["vendor"]),
            "b_moved": sum(1 for c in ok if c["b_moved"])}


def _print(s: dict) -> None:
    print(f"\n  ★ {s['n']}칸  (a)−원주민 중앙 {s['a_minus_native_median']:+.4f}"
          f"  (b)−원주민 중앙 {s['b_minus_native_median']:+.4f}")
    print(f"     원주민 넘음  (a) {s['a_beats_native']}/{s['n']}  "
          f"(b) {s['b_beats_native']}/{s['n']}  ·  벤더 넘음 (b) "
          f"{s['b_beats_vendor']}/{s['n']}  ·  적합 이동 {s['b_moved']}/{s['n']}")


if __name__ == "__main__":
    main()
