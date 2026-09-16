"""★ 홀드아웃에서 고르는 것은 ★ 할 수 있다 — 그 비용과 ★ 정직한 값
(D-177 §3). **0 LLM calls · 0 GPU · ⛔ 적합을 다시 하지 않는다.**

    python3 -m experiments.porting_shapes_select

## 왜 할 수 있나

후보 K개를 줄 세우는 데 **표를 새로 만들 필요가 없다.**

```
regret_i(형상) = t_i(형상) / t_best(형상)
t_best 는 ★ 후보와 무관한 상수  -> geomean regret 의 대소
                               == ★ 고른 config 실측 시간의 대소
```

그러므로 형상마다 유효 config 15,015 개를 다 만들 것 없이 ★ **후보 K개가
고른 config 만** 빌드·측정하면 같은 순서가 나온다.

```
형상 M개에서 후보 K개를 고르는 비용   ★ K x M 번 빌드·측정
형상 M개를 통째로 만드는 비용         M x 15,015 번
K=30 · M=17 이면  510 대 255,255  = ★ 0.2%
```

## ⛔ 고르는 것과 보고하는 것은 다르다

고른 그 형상에서 낸 값은 **낙관적으로 치우친다** — 후보 30개 중 최선을
집었으니 그 숫자는 절차의 성능이 아니라 상한이다. 그래서 홀드아웃을 둘로
갈라

```
★ 고르기   절반 A 에서 (후보 K개의 실측만으로 가능)
★ 보고     ⛔ 나머지 절반 B 에서 — ★ 고를 때 안 본 형상
★ 양쪽을 바꿔 한 번 더 하고 평균
```

한다. `선택손해` = 오라클(전체에서 고르고 전체에서 보고) − 정직한 값.
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
from experiments.porting_shapes import FOLD, GPUS, _seed_of
from experiments.transfer_29_5 import TABLES
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import regime_of
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry

CURVE = Path("docs/artifacts/porting-shapes.json")
OUT = Path("docs/artifacts/porting-shapes-select.json")
#: ★ how the holdout is cut in two. 20 random halvings, averaged — one
#: halving would make the answer a property of that cut.
N_HALVINGS = 20


def _per_shape(code: str, weights: dict, *, table, matrix, val) -> dict:
    """★ Every holdout shape's regret for one candidate. ⛔ No fitting —
    the weights are the ones `porting_shapes` already fitted."""
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import make_score_of

    fn = compile_rule(code)
    out: dict = {}
    for name in ("short", "long"):
        g = [p for p in val if regime_of(p, table.hw) == name]
        if not g or name not in weights:
            continue
        e = evaluate_scores(make_score_of(fn, matrix,
                                          np.asarray(weights[name], float)),
                            table, g, ks=(1,))
        for i, p in enumerate(e.shapes):
            out[(p.M, p.N, p.K)] = float(e.regret[i, 0])
    return out


def _gm(d: dict, keys) -> float:
    v = [d[k] for k in keys if k in d]
    return float(geomean(np.array(v))) if v else float("nan")


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default=str(CURVE))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    curve = json.loads(Path(a.curve).read_text())
    by_dir: dict = {}
    for r in curve["rows"]:
        by_dir.setdefault((r["src"], r["dst"]), []).append(r)

    rows = []
    print("=" * 108)
    print("★ 홀드아웃에서 고르기 — 비용과 ★ 정직한 값 (D-177 §3). "
          "0 LLM · ⛔ 적합 다시 안 함")
    print("=" * 108)
    for dst in GPUS:
        T = TABLES[dst]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        val = list(_splits(table, fold=FOLD, k=4, design="nkband").val.shapes)
        keys = [(p.M, p.N, p.K) for p in val]
        for src in GPUS:
            if src == dst:
                continue
            rs = by_dir.get((src, dst))
            if not rs:
                continue
            d = Path(f"runs/pc-{src}2{dst}-f{FOLD}")
            e = json.loads(
                (d / "stage2-rule-writer" / "chosen.json").read_text())
            reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2",
                               table)
            m = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
            cands = []
            for row in rs:
                for rep in row["reps"]:
                    cands.append({
                        "N": row["N"], "rep": rep["rep"],
                        "per_shape": _per_shape(e["code"], rep["weights"],
                                                table=table, matrix=m,
                                                val=val)})
            for c in cands:
                c["all"] = _gm(c["per_shape"], keys)
            oracle = min(cands, key=lambda c: c["all"])
            # ★ the honest number — choose on half, read on the other half
            rng = np.random.default_rng(_seed_of(src, dst, "halve"))
            honest, chosen_n = [], []
            for _ in range(N_HALVINGS):
                idx = rng.permutation(len(keys))
                h = len(keys) // 2
                A = [keys[int(i)] for i in idx[:h]]
                B = [keys[int(i)] for i in idx[h:]]
                for pick_on, read_on in ((A, B), (B, A)):
                    best = min(cands, key=lambda c: _gm(c["per_shape"],
                                                        pick_on))
                    honest.append(_gm(best["per_shape"], read_on))
                    chosen_n.append(best["N"])
            rows.append({
                "src": src, "dst": dst, "n_candidates": len(cands),
                "n_holdout": len(keys),
                "oracle_holdout": round(oracle["all"], 6),
                "oracle_N": oracle["N"], "oracle_rep": oracle["rep"],
                "honest_median": round(st.median(honest), 6),
                "honest_min": round(min(honest), 6),
                "honest_max": round(max(honest), 6),
                "selection_loss": round(st.median(honest) - oracle["all"], 6),
                "chosen_N_counts": {str(n): chosen_n.count(n)
                                    for n in sorted(set(chosen_n))},
                "n_halvings": N_HALVINGS,
                # ★ the price of choosing: K candidates x M shapes, one
                #   build+measure each
                "select_builds": len(cands) * len(keys),
                "b_refit": e["b_refit"], "a_as_is": e["a_as_is"],
                "native": e["native"]})
            r = rows[-1]
            print(f"  {src:5s}->{dst:5s}  오라클 {r['oracle_holdout']:.4f} "
                  f"(N={r['oracle_N']})  ★ 정직 {r['honest_median']:.4f} "
                  f"[{r['honest_min']:.4f}~{r['honest_max']:.4f}]  "
                  f"선택손해 {r['selection_loss']:+.4f}  "
                  f"| N=48 {r['b_refit']:.4f}  원주민 {r['native']:.4f}",
                  flush=True)
    Path(a.out).write_text(json.dumps(
        {"n_halvings": N_HALVINGS, "rows": rows,
         "why": ("ranking candidates by geomean regret on a shape set is the "
                 "same ranking as by their picked configs' measured time, so "
                 "choosing costs K x M builds, not M x 15,015"),
         }, ensure_ascii=False, indent=1))
    if rows:
        o = [r["oracle_holdout"] - r["native"] for r in rows]
        h = [r["honest_median"] - r["native"] for r in rows]
        b = [r["b_refit"] - r["native"] for r in rows]
        print(f"\n  원주민대비 중앙   오라클 {st.median(o):+.4f}  "
              f"★ 정직 {st.median(h):+.4f}  N=48 재적합 {st.median(b):+.4f}")
        print(f"  원주민 넘음       오라클 {sum(1 for x in o if x < 0)}/12  "
              f"★ 정직 {sum(1 for x in h if x < 0)}/12  "
              f"N=48 {sum(1 for x in b if x < 0)}/12")
        print(f"  ★ 고르는 비용     후보 {rows[0]['n_candidates']} x 형상 "
              f"{rows[0]['n_holdout']} = {rows[0]['select_builds']} 빌드")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
