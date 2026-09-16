"""★ "16개를 만들었다면, 그 안에서 고른 최선은 어디까지 가나" (D-177 §3).
**0 LLM calls · 0 GPU.**

    python3 -m experiments.porting_shapes_pick --dst <gpu>
    python3 -m experiments.porting_shapes_pick --merge a.json b.json ...

§1-1 의 곡선은 N 마다 **따로** 뽑은 표본이다. 이 파일은 예산을 형상 수로
묶어서 묻는다.

```
예산   ★ 대상 형상 ★ 16개를 만들었다 — 그 16개가 가진 전부다
후보   그 16개의 ★ 부분집합으로 맞춘 가중치들
       4개짜리 10개 · 8개짜리 10개 · ★ 16개 전부 1개  = 21후보
선택   ⛔ 홀드아웃을 보지 않는다 — ★ 그 16개에서의 점수로 고른다
평가   ★ fold0 홀드아웃 전체 (canonical 과 같은 절차)
```

## ⚠️ 이 선택 규칙의 편향

16개 전부로 맞춘 후보는 **자기가 본 16개에서** 채점되고, 4개짜리 후보는 그중
12개를 처음 본다. 그러므로 이 규칙은 ★ **큰 N 쪽으로 기운다**. 그래도 이것이
홀드아웃을 안 보는 규칙 중 가장 단순한 것이라 그대로 쓰고, 기운다는 사실을
적는다.

★ 오라클(홀드아웃을 보고 고른 최선)도 같이 낸다. ⛔ 쓸 수 있는 값이 아니라
**상한**이다 — 둘을 나란히 두어야 선택이 얼마를 잃는지 읽힌다.
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
from experiments.porting_shapes import (
    FOLD,
    GPUS,
    _refit,
    _sample,
    _seed_of,
)
from experiments.transfer_29_5 import TABLES
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import regime_of
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry

BUDGET = 16
SUBSETS = (4, 8)
N_SUB = 10
OUT = Path("docs/artifacts/porting-shapes-pick.json")


def _score_on(code: str, weights: dict, *, table, matrix, shapes) -> float:
    """★ The candidate's regret on a given shape set, using the same
    per-regime weights it was fitted with."""
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import make_score_of

    fn = compile_rule(code)
    got = []
    for name in ("short", "long"):
        g = [p for p in shapes if regime_of(p, table.hw) == name]
        if not g or name not in weights:
            continue
        e = evaluate_scores(make_score_of(fn, matrix,
                                          np.asarray(weights[name], float)),
                            table, g, ks=(1,))
        got += [e.regret[i, 0] for i in range(len(e.shapes))]
    return float(geomean(np.array(got))) if got else float("nan")


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.merge is not None:
        rows: list[dict] = []
        for p in a.merge:
            rows += json.loads(Path(p).read_text())["rows"]
        Path(a.out or OUT).write_text(json.dumps(
            {"budget": BUDGET, "subsets": list(SUBSETS), "n_sub": N_SUB,
             "rows": rows}, ensure_ascii=False, indent=1))
        print(f"merged {len(a.merge)} -> {a.out or OUT}  ({len(rows)} rows)")
        return

    rows = []
    print("=" * 100)
    print(f"★ 형상 {BUDGET}개를 만들었다면 — 부분집합 후보 중 고른 최선 "
          f"(D-177 §3). 0 LLM · 0 GPU")
    print("=" * 100)
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
            # ★ the 16 shapes this budget bought — the same pick (b) used
            built = _sample(train, BUDGET, _seed_of(src, dst, BUDGET, 0))
            cands = []
            for n in (*SUBSETS, BUDGET):
                for rep in range(N_SUB if n < BUDGET else 1):
                    s = (built if n == BUDGET
                         else _sample(built, n, _seed_of(src, dst, "pick", n,
                                                         rep)))
                    r = _refit(e["code"], e["w0"], table=table, matrix=m,
                               sample=s, val=val)
                    cands.append({
                        "n": n, "rep": rep,
                        "holdout": r["holdout"],
                        # ⛔ the selection score — the 16 shapes only
                        "on_built": _score_on(e["code"], r["weights"],
                                              table=table, matrix=m,
                                              shapes=built),
                        "pooled_regimes": r["pooled_regimes"],
                        "moved": r["moved"],
                        "shapes": [[p.M, p.N, p.K] for p in s]})
            pick = min(cands, key=lambda c: c["on_built"])
            oracle = min(cands, key=lambda c: c["holdout"])
            rows.append({
                "src": src, "dst": dst, "budget": BUDGET,
                "n_candidates": len(cands),
                "built_shapes": [[p.M, p.N, p.K] for p in built],
                "picked": {"n": pick["n"], "rep": pick["rep"],
                           "holdout": round(pick["holdout"], 6),
                           "on_built": round(pick["on_built"], 6)},
                "oracle": {"n": oracle["n"], "rep": oracle["rep"],
                           "holdout": round(oracle["holdout"], 6)},
                "selection_loss": round(pick["holdout"] - oracle["holdout"],
                                        6),
                "b_refit": e["b_refit"], "a_as_is": e["a_as_is"],
                "native": e["native"], "candidates": cands})
            r = rows[-1]
            print(f"  {src:5s}->{dst:5s}  고른것 n={r['picked']['n']:2d} "
                  f"홀드아웃 {r['picked']['holdout']:.4f}  "
                  f"오라클 {r['oracle']['holdout']:.4f} "
                  f"(n={r['oracle']['n']})  "
                  f"선택손해 {r['selection_loss']:+.4f}  "
                  f"| N=48 {r['b_refit']:.4f}  원주민 {r['native']:.4f}",
                  flush=True)
    out = Path(a.out or OUT)
    out.write_text(json.dumps({"budget": BUDGET, "subsets": list(SUBSETS),
                               "n_sub": N_SUB, "rows": rows},
                              ensure_ascii=False, indent=1))
    if rows:
        g = [r["picked"]["holdout"] - r["native"] for r in rows]
        print(f"\n  ★ 고른것 원주민대비 중앙 {st.median(g):+.4f}  "
              f"· 원주민 넘음 {sum(1 for x in g if x < 0)}/{len(rows)}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
