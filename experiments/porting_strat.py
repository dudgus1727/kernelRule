"""★ 전이 형상 뽑기 — ★ 무작위 대신 층화 (D-178). **0 LLM · 0 GPU.**

    python3 -m experiments.porting_strat --plan          # 층별 구성만
    python3 -m experiments.porting_strat --dst <g> --src <s>
    python3 -m experiments.porting_strat --merge <f...>

D-177 은 대상 형상을 무작위로 뽑았다. 실무를 반영하지 않는다 — N 이 정해지면
대표 형상을 고르게 고른다. 그리고 D-177 의 **뽑기 폭 0.22~0.27 대 N 의 효과
0.02~0.04** 가, 어느 형상이 뽑히느냐가 N 보다 열 배 크게 결과를 정한다는
증거다. "N 을 늘려도 잘 안 메워진다" 가 ★ N 이 작아서가 아니라 ★ 뽑기가
나빠서일 수 있다.

## 층

```
① roofline   memory-bound / compute-bound      2
② M 크기     ★ 그 fold 학습 분할의 3분위        3
★ 층 = 6
```

⚠️ 3분위 경계는 ★ 표마다 fold 마다 다시 계산한다. a6000 fold0 에서는
**256 · 1024** 가 나온다.

## 배분 — 층이 빌 때의 규칙

```
★ 각 층에 N // 6 씩
★ 나머지는 ★ 그 fold 학습 분할에서 형상이 많은 층부터 한 개씩
★ 어느 층이 가진 형상보다 많이 배정되면 ★ 다음 큰 층으로 넘긴다
★ N=4 면 층 6개 중 ★ 두 층은 못 채운다 — 어느 층인지 보고한다
```

## 층 안에서는 무작위

층을 정해도 층 안에는 여러 형상이 있다. ★ 반복 10회로 층 안 뽑기의 폭을
재고, 무작위(D-177)와 같은 자리에서 견준다.

⛔ 평가는 D-177 과 같다 — 대상 fold0 홀드아웃 **전체**, `porting_shapes._refit`.
⛔ D-177 의 무작위 수치는 고치지 않는다. 대조군으로 남긴다.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.c2_ref import label, native, transfer
from experiments.f1_pipeline import _load_stage1, _splits
from experiments.porting_shapes import FOLD, GPUS, NS, _refit, _seed_of
from experiments.transfer_29_5 import TABLES
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.splits import regime_of
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry

REPEATS = 10
OUT = Path("docs/artifacts/porting-strat.json")
M_BANDS = ("M_low", "M_mid", "M_high")


def m_tertiles(shapes) -> tuple[float, float]:
    """★ The M tertile cuts of **this** train split. ⛔ Not a constant —
    every table and fold recomputes them."""
    m = np.array([p.M for p in shapes], float)
    return float(np.quantile(m, 1 / 3)), float(np.quantile(m, 2 / 3))


def stratum_of(p, table, cuts) -> tuple[str, str]:
    q1, q2 = cuts
    band = M_BANDS[0] if q1 >= p.M else (M_BANDS[1] if q2 >= p.M
                                         else M_BANDS[2])
    return (regime_of(p, table.hw, axis="roofline"), band)


def strata_of(shapes, table, cuts) -> dict:
    out: dict = {}
    for p in shapes:
        out.setdefault(stratum_of(p, table, cuts), []).append(p)
    return out


def allocate(strata: dict, n: int) -> dict:
    """★ N // 6 each, the remainder to the biggest strata first, spilling on
    to the next biggest when a stratum runs out."""
    order = sorted(strata, key=lambda k: (-len(strata[k]), str(k)))
    alloc = dict.fromkeys(order, 0)
    base = n // 6
    left = n
    for k in order:                      # the even part
        take = min(base, len(strata[k]))
        alloc[k] = take
        left -= take
    while left > 0:                      # the remainder, biggest first
        moved = False
        for k in order:
            if left == 0:
                break
            if alloc[k] < len(strata[k]):
                alloc[k] += 1
                left -= 1
                moved = True
        if not moved:                    # every stratum exhausted
            break
    return alloc


def sample_stratified(strata: dict, n: int, seed: int) -> tuple[list, dict]:
    """★ Even across strata, ★ random within one."""
    alloc = allocate(strata, n)
    rng = np.random.default_rng(seed)
    got = []
    for k in sorted(alloc, key=str):
        v = strata[k]
        take = alloc[k]
        if take <= 0:
            continue
        idx = rng.choice(len(v), size=take, replace=False)
        got += [v[int(i)] for i in sorted(idx)]
    return got, {f"{a}|{b}": c for (a, b), c in alloc.items()}


def _plan() -> None:
    print("=" * 96)
    print("★ 층별 구성 (D-178 §1). 0 LLM · 0 GPU")
    print("=" * 96)
    res: dict = {"ns": list(NS), "tables": {}}
    for gpu in GPUS:
        T = TABLES[gpu]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        train = list(_splits(table, fold=FOLD, k=4,
                             design="nkband").train.shapes)
        cuts = m_tertiles(train)
        strata = strata_of(train, table, cuts)
        sizes = {f"{a}|{b}": len(v) for (a, b), v in
                 sorted(strata.items(), key=lambda kv: -len(kv[1]))}
        print(f"\n  ── {gpu}  학습 {len(train)} 형상 · ★ M 3분위 경계 "
              f"{cuts[0]:.0f} · {cuts[1]:.0f}")
        print(f"     M 분포 {dict(sorted(Counter(p.M for p in train).items()))}")
        print(f"     층 {sizes}")
        empt = {}
        for n in NS:
            al = allocate(strata, n)
            miss = [f"{a}|{b}" for (a, b), c in al.items() if c == 0]
            empt[str(n)] = miss
            print(f"     N={n:2d}  배분 "
                  + " ".join(f"{a[:4]}/{b[2:]}:{c}"
                             for (a, b), c in sorted(al.items(), key=str))
                  + (f"   ⚠️ 못 채운 층 {len(miss)}" if miss else ""))
        res["tables"][gpu] = {
            "n_train": len(train), "m_cuts": list(cuts),
            "strata_sizes": sizes,
            "alloc": {str(n): {f"{a}|{b}": c
                               for (a, b), c in allocate(strata, n).items()}
                      for n in NS},
            "unfilled": empt}
    Path("docs/artifacts/porting-strat-plan.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print("\n  -> docs/artifacts/porting-strat-plan.json")


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.plan:
        _plan()
        return
    if a.merge is not None:
        rows: list[dict] = []
        for p in a.merge:
            rows += json.loads(Path(p).read_text())["rows"]
        Path(a.out or OUT).write_text(json.dumps(
            {"ns": list(NS), "repeats": REPEATS, "fold": FOLD,
             "sampling": "stratified: roofline x M-tertile, 6 strata",
             "rows": rows}, ensure_ascii=False, indent=1))
        print(f"merged {len(a.merge)} -> {a.out or OUT}  ({len(rows)} rows)")
        return

    rows = []
    print("=" * 108)
    print("★ 층화 뽑기 — (a) 재적합 곡선 (D-178 §2). 0 LLM · 0 GPU")
    print("=" * 108)
    for dst in (GPUS if a.dst is None else (a.dst,)):
        T = TABLES[dst]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        splits = _splits(table, fold=FOLD, k=4, design="nkband")
        train = list(splits.train.shapes)
        val = list(splits.val.shapes)
        cuts = m_tertiles(train)
        strata = strata_of(train, table, cuts)
        for src in (GPUS if a.src is None else (a.src,)):
            if src == dst:
                continue
            d = Path(f"runs/pc-{src}2{dst}-f{FOLD}")
            ch = d / "stage2-rule-writer" / "chosen.json"
            if not ch.exists():
                print(f"  ⚠️ {src}->{dst} 씨앗 없음 — 건너뜀")
                continue
            e = json.loads(ch.read_text())
            # ★ D-186 — 재집계된 c2 값
            tr = transfer(src, dst, FOLD)
            e["a_as_is"], e["b_refit"] = tr["a_as_is"], tr["b_refit"]
            e["native"] = native(dst, FOLD)
            reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2",
                               table)
            m = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
            for n in NS:
                got = []
                for rep in range(REPEATS):
                    seed = _seed_of("strat", src, dst, n, rep)
                    s, alloc = sample_stratified(strata, n, seed)
                    r = _refit(e["code"], e["w0"], table=table, matrix=m,
                               sample=s, val=val)
                    r.update({"rep": rep, "sample_seed": seed,
                              "alloc": alloc,
                              "shapes": [[p.M, p.N, p.K] for p in s]})
                    got.append(r)
                h = sorted(x["holdout"] for x in got)
                rows.append({
                    "src": src, "dst": dst, "fold": FOLD, "N": n,
                    "sampling": "stratified",
                    "m_cuts": list(cuts),
                    "n_strata": len(strata),
                    "n_unfilled": sum(1 for v in
                                      allocate(strata, n).values() if v == 0),
                    "n_train_full": len(train), "n_holdout": len(val),
                    "a_as_is": e["a_as_is"], "b_refit": e["b_refit"],
                    "native": e["native"],
                    "holdout_median": round(st.median(h), 6),
                    "holdout_min": round(h[0], 6),
                    "holdout_max": round(h[-1], 6),
                    "spread": round(h[-1] - h[0], 6),
                    "gap_median": round(st.median(h) - e["native"], 6),
                    "n_pooled": sum(1 for x in got if x["pooled_regimes"]),
                    "n_not_moved": sum(1 for x in got if not x["moved"]),
                    "reps": got})
                r = rows[-1]
                print(f"  {src:5s}->{dst:5s} N={n:2d}  홀드아웃 중앙 "
                      f"{r['holdout_median']:.4f} "
                      f"[{r['holdout_min']:.4f}~{r['holdout_max']:.4f}] "
                      f"폭 {r['spread']:.4f}  "
                      f"원주민대비 {r['gap_median']:+.4f}  "
                      f"못채운층 {r['n_unfilled']}  "
                      f"체제보충 {r['n_pooled']}/10", flush=True)
    out = Path(a.out or OUT)
    out.write_text(json.dumps({"ns": list(NS), "repeats": REPEATS,
                               "fold": FOLD, "sampling": "stratified",
                               "note": label(), "rows": rows},
                              ensure_ascii=False, indent=1))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
