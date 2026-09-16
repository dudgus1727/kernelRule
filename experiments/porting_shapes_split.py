"""★ 형상 16개만 만들었다면 — ★ 일부는 적합에, 일부는 **고르는 데** 쓴다
(D-177 §3). **0 LLM calls · 0 GPU.**

    python3 -m experiments.porting_shapes_split --dst <gpu>
    python3 -m experiments.porting_shapes_split --merge a.json ...

두 측정 사이가 비어 있었다.

```
porting_shapes_pick     16개를 만들고 ★ 그 16개에서 고른다
                        -> 자기가 맞춘 형상에서 고르는 셈이라 잘 못 고른다
porting_shapes_select   ★ 처음 보는 형상에서 고르면 잘 고른다
                        -> 그러나 후보를 만드는 데 형상 47개를 썼다
```

그 사이가 이것이다.

```
예산   대상 형상 ★ 16개 — 통째로 빌드된 것은 이게 전부
적합   ★ 그중 절반(8) 의 부분집합들로 가중치를 맞춘다
고르기 ⛔ 나머지 절반(8) 에서 — ★ 적합 때 안 본 형상
       (후보가 고른 config 만 재면 되므로 형상을 새로 만들지 않는다)
평가   fold0 홀드아웃 전체
반복   ★ 가르기 5회 · 양방향 -> 10점, 중앙
```

⚠️ 고르는 쪽 8개도 **통째로 빌드되어 있어야** 하는 것은 아니다. 후보 K개를
줄 세우는 데는 그 K개가 **고른 config 만** 재면 된다 (§select 의 논거).
여기서는 표를 조회해 그 값을 낸다.
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
from experiments.porting_shapes import FOLD, GPUS, _refit, _sample, _seed_of
from experiments.porting_shapes_pick import BUDGET, _score_on
from experiments.transfer_29_5 import TABLES
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry

N_HALVINGS = 5
SUB = 4          # subsets of the fit half
N_SUB = 5
OUT = Path("docs/artifacts/porting-shapes-split.json")


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.merge:
        rows: list[dict] = []
        for f in a.merge:
            rows += json.loads(Path(f).read_text())["rows"]
        Path(a.out).write_text(json.dumps(
            {"budget": BUDGET, "n_halvings": N_HALVINGS, "rows": rows},
            ensure_ascii=False, indent=1))
        print(f"merged -> {a.out}  ({len(rows)} rows)")
        _summary(rows)
        return

    rows = []
    print("=" * 108)
    print(f"★ 형상 {BUDGET}개를 적합/고르기로 쪼갠다 (D-177 §3). 0 LLM · 0 GPU")
    print("=" * 108)
    for dst in (GPUS if a.dst is None else (a.dst,)):
        T = TABLES[dst]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        splits = _splits(table, fold=FOLD, k=4, design="nkband")
        train = list(splits.train.shapes)
        val = list(splits.val.shapes)
        for src in GPUS:
            if src == dst:
                continue
            d = Path(f"runs/pc-{src}2{dst}-f{FOLD}")
            ch = d / "stage2-rule-writer" / "chosen.json"
            if not ch.exists():
                continue
            e = json.loads(ch.read_text())
            reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2",
                               table)
            m = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
            built = _sample(train, BUDGET, _seed_of(src, dst, BUDGET, 0))
            rng = np.random.default_rng(_seed_of(src, dst, "split"))
            got, picked_n = [], []
            for hv in range(N_HALVINGS):
                idx = rng.permutation(BUDGET)
                h = BUDGET // 2
                halves = ([built[int(i)] for i in idx[:h]],
                          [built[int(i)] for i in idx[h:]])
                for fit_half, sel_half in (halves, halves[::-1]):
                    cands = []
                    for n in (SUB, h):
                        for rep in range(N_SUB if n < h else 1):
                            s = (fit_half if n == h else
                                 _sample(fit_half, n,
                                         _seed_of(src, dst, "split", hv, n,
                                                  rep)))
                            r = _refit(e["code"], e["w0"], table=table,
                                       matrix=m, sample=s, val=val)
                            cands.append({
                                "n": n, "rep": rep,
                                "holdout": r["holdout"],
                                # ⛔ the selection signal — the half this
                                #    candidate never saw
                                "on_sel": _score_on(e["code"], r["weights"],
                                                    table=table, matrix=m,
                                                    shapes=sel_half)})
                    best = min(cands, key=lambda c: c["on_sel"])
                    oracle = min(cands, key=lambda c: c["holdout"])
                    got.append({"halving": hv, "picked": best["holdout"],
                                "picked_n": best["n"],
                                "oracle": oracle["holdout"],
                                "n_candidates": len(cands)})
                    picked_n.append(best["n"])
            p = sorted(x["picked"] for x in got)
            rows.append({
                "src": src, "dst": dst, "budget": BUDGET,
                "n_points": len(got),
                "picked_median": round(st.median(p), 6),
                "picked_min": round(p[0], 6), "picked_max": round(p[-1], 6),
                "oracle_median": round(st.median(
                    [x["oracle"] for x in got]), 6),
                "selection_loss": round(st.median(p) - st.median(
                    [x["oracle"] for x in got]), 6),
                "picked_n_counts": {str(n): picked_n.count(n)
                                    for n in sorted(set(picked_n))},
                "b_refit": e["b_refit"], "a_as_is": e["a_as_is"],
                "native": e["native"], "points": got})
            r = rows[-1]
            print(f"  {src:5s}->{dst:5s}  고른것 {r['picked_median']:.4f} "
                  f"[{r['picked_min']:.4f}~{r['picked_max']:.4f}]  "
                  f"오라클 {r['oracle_median']:.4f}  "
                  f"손해 {r['selection_loss']:+.4f}  "
                  f"| N=48 {r['b_refit']:.4f}  원주민 {r['native']:.4f}",
                  flush=True)
    Path(a.out).write_text(json.dumps(
        {"budget": BUDGET, "n_halvings": N_HALVINGS, "rows": rows},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


def _summary(rows: list[dict]) -> None:
    p = [r["picked_median"] - r["native"] for r in rows]
    o = [r["oracle_median"] - r["native"] for r in rows]
    b = [r["b_refit"] - r["native"] for r in rows]
    print(f"\n  원주민대비 중앙  ★ 고른것 {st.median(p):+.4f}  "
          f"오라클 {st.median(o):+.4f}  N=48 {st.median(b):+.4f}")
    print(f"  원주민 넘음      ★ 고른것 {sum(1 for x in p if x < 0)}/{len(p)}"
          f"  오라클 {sum(1 for x in o if x < 0)}/{len(o)}"
          f"  N=48 {sum(1 for x in b if x < 0)}/{len(b)}")


if __name__ == "__main__":
    main()
