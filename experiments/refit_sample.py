"""★ §29.5(b) 의 **비용** 주장 — 표본 몇 %면 가중치가 맞춰지나. LLM 0회.

    python3 experiments/refit_sample.py --workers 6

## 무엇이 주장인가

> §29.5(b) 가 성립하면 **"표 15시간" 이 "표본 5% + 수 초"** 가 된다.

`transfer_29_5.py` 는 학습 분할 **41형상 전부**로 재적합했다. 그것은
**"구조가 전이되나"** 를 잰 것이고, 여기서 재는 것은 **"그것이 싼가"**
다. 둘은 다른 질문이다.

## 재는 법

```
5090 학습 41형상에서 k개를 뽑아 **그것만으로** 체제별 가중치를 맞춘다
-> 5090 구조 홀드아웃 20형상에서 채점    (홀드아웃은 언제나 전부)
k = 2(5%) 4 8 12 20 41(100%)
```

★ **어느 k 개를 뽑느냐도 결과다.** 뽑기를 여러 번 해서 분포를 낸다 —
"5% 면 된다" 가 운 좋은 한 번이면 안 된다.

## ★ 파국의 정체를 함께 남긴다 (2026-09-01)

최악이 2.35 였다 — "조금 나쁜" 것이 아니라 뭔가 깨진 것이다. 대응이
갈리므로 원인을 기록한다.

```
적합 집합에서는 좋은데 홀드아웃이 나쁘다  -> ★ 과적합. 표본 문제다
                                          적합 예산을 늘려도 소용없다
둘 다 나쁘다                              -> 적합기가 못 찾았다
```

★ 그리고 **형상 / 파라미터 비율**을 함께 본다. `k=2` 는 체제당 1형상에
가중치 8개다 — 어떻게 골라도 과소결정이다. "몇 %" 가 맞는 단위가
아닐 수 있다.

★ **체제별로 층화해서 뽑는다.** 체제마다 가중치를 따로 맞추므로 한
체제가 0개면 적합 자체가 안 된다. k=2 면 체제당 1개다. **이 층화는
조건이고, 실무에서도 그렇게 해야 한다** — "아무 2개나" 가 아니다.

## 하지 않는 것

`(c)` 와 겨루지 않는다. 여기서 겨루는 것은 **같은 구조의 41형상 재적합**
이다 — "표본을 줄이면 얼마나 잃나" 가 질문이다.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
SRC_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
#: ★ 8 과 12 사이가 꼭짓점이라 10/14/16 을 채운다 (2026-09-01)
KS = (2, 4, 8, 10, 12, 14, 16, 20, 41)

#: 워커가 fork 로 물려받는다 (4.2 GB 행렬을 복사하지 않는다).
_W: dict = {}


def _splits(table: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def _stratified(rng, by_regime: dict, k: int) -> list:
    """체제별로 고르게 k개. **각 체제에서 최소 1개**를 보장한다."""
    names = sorted(by_regime)
    take = dict.fromkeys(names, 1)
    left = k - len(names)
    if left < 0:
        raise ValueError(f"k={k} 가 체제 수 {len(names)} 보다 작다")
    # 남은 자리는 체제 크기에 비례해서
    sizes = np.array([len(by_regime[n]) for n in names], dtype=float)
    extra = np.floor(sizes / sizes.sum() * left).astype(int)
    for i in range(left - int(extra.sum())):
        extra[i % len(names)] += 1
    out = []
    for n, e in zip(names, extra, strict=True):
        pool = by_regime[n]
        idx = rng.choice(len(pool), size=min(len(pool), take[n] + int(e)),
                         replace=False)
        out.extend(pool[i] for i in idx)
    return out


def _job(arg):
    """(규칙 index, k, 뽑기 seed) -> 홀드아웃 regret."""
    ri, k, seed = arg
    e = _W["rules"][ri]
    table, matrix = _W["table"], _W["matrix"]
    by_regime, hold = _W["by_regime"], _W["hold"]
    rng = np.random.default_rng(seed)
    sample = (list(_W["train"]) if k >= len(_W["train"])
              else _stratified(rng, by_regime, k))
    fn = compile_rule(e["code"])
    ws, fits, moved = {}, [], []
    for name in ("short", "long"):
        g = [p for p in sample if regime_of(p, table.hw) == name]
        if not g:                       # ★ 조용히 넘기지 않는다
            return {"ri": ri, "k": k, "seed": seed, "holdout": float("nan"),
                    "empty_regime": name}
        fr = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                         e["w"], max_evals=300,
                          objective="regret")
        ws[name] = fr.w
        fits.append(fr.fit_regret)
        moved.append(bool(fr.moved))
    regs = []
    for p in hold:
        cand = table.candidates(p)
        sc = make_score_of(fn, matrix, ws[regime_of(p, table.hw)])(p, cand)
        t = table.times_of(p)
        regs.append(float(t[cand.top_k(sc, 1)[0]] / t.min()))
    return {"ri": ri, "run": _W["runs"][ri], "k": k, "seed": seed,
            "n_sample": len(sample),
            # ★ 적합 집합에서의 값. 홀드아웃과 갈리면 과적합이다
            "fit_regret": float(np.exp(np.mean(np.log(fits)))),
            "moved": all(moved),
            "n_terms": _W["n_terms"][ri],
            "n_weights": len(e["w"]),
            "holdout": float(geomean(np.array(regs)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10,
                    help="k 마다 몇 번 뽑나. k=41 은 결정적이라 1번")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="docs/artifacts/refit-sample.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    table = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    sp = _splits(table)
    train = list(sp.train.shapes)
    by_regime: dict = {}
    for p in train:
        by_regime.setdefault(regime_of(p, table.hw), []).append(p)

    rules = []
    for r in SRC_RUNS:
        f = Path("runs") / r / "archive.jsonl"
        arc = sorted((json.loads(x) for x in f.read_text().splitlines()
                      if x.strip()), key=lambda e: e["regret"])
        rules.append(arc[0])

    # ★ 파국이 항 수와 관계있나 — 예산 실험(8 vs 16)과 이어지는 질문이다
    from kernelrule.rules.checks import check_rule
    n_terms = []
    for e in rules:
        try:
            n_terms.append(check_rule(
                e["code"], feature_names=REGISTRY.names(shape_level=False),
                shape_value_names=REGISTRY.names(shape_level=True),
                n_weights=len(e["w"])).n_terms)
        except Exception:                                   # noqa: BLE001
            n_terms.append(-1)
    _W.update(table=table, matrix=matrix, train=train, by_regime=by_regime,
              hold=list(sp.val.shapes), rules=rules, runs=SRC_RUNS,
              n_terms=n_terms)

    print("=" * 76)
    print("§29.5(b) 비용 — 표본 몇 %면 가중치가 맞춰지나")
    print("=" * 76)
    print(f"  5090 학습 {len(train)}형상  "
          + "  ".join(f"{n} {len(v)}" for n, v in sorted(by_regime.items())))
    print(f"  홀드아웃 {len(sp.val.shapes)}형상 — **언제나 전부**")
    print(f"  구조 {len(rules)}개 (A6000 F3 6시드의 학습 최고)")
    print(f"  뽑기 k 마다 {a.draws}번, 체제별 층화\n")

    jobs = [(ri, k, 1000 * ri + 7 * k + d)
            for ri in range(len(rules)) for k in KS
            for d in range(1 if k >= len(train) else a.draws)]
    if a.workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=a.workers,
                                 mp_context=get_context("fork")) as ex:
            out = list(ex.map(_job, jobs, chunksize=4))
    else:
        out = [_job(j) for j in jobs]

    by_k: dict = {k: [] for k in KS}
    for r in out:
        if np.isfinite(r["holdout"]):
            by_k[r["k"]].append(r["holdout"])

    print(f"  {'k':>4} {'표본%':>7}  {'중앙':>8} {'범위':>19} {'폭':>8}  n")
    rows = []
    for k in KS:
        v = np.array(by_k[k])
        pct = 100.0 * k / len(train)
        print(f"  {k:4d} {pct:6.1f}%  {np.median(v):8.4f}  "
              f"{v.min():.4f}~{v.max():.4f}  {v.max() - v.min():8.4f}  "
              f"{len(v)}")
        rows.append({"k": k, "pct": pct, "median": float(np.median(v)),
                     "min": float(v.min()), "max": float(v.max()),
                     "n": len(v), "values": [float(x) for x in v]})

    full = np.median(by_k[len(train)]) if by_k.get(len(train)) else float("nan")
    print(f"\n  ★ 41형상(100%) 중앙 {full:.4f} 대비 손실")
    for k in KS:
        if k >= len(train):
            continue
        v = np.array(by_k[k])
        print(f"     k={k:2d} ({100.0 * k / len(train):4.1f}%)  "
              f"중앙 +{np.median(v) - full:.4f}   "
              f"최악 +{v.max() - full:.4f}")
    # ------------------------------------------------------------------
    # ★ 파국의 정체
    # ------------------------------------------------------------------
    ok = [r for r in out if np.isfinite(r["holdout"])]
    cat = [r for r in ok if r["holdout"] > 1.15]
    print("\n" + "=" * 76)
    print(f"★ 파국(홀드아웃 > 1.15) {len(cat)}건 / {len(ok)}건")
    print("=" * 76)
    if cat:
        fr = np.array([r["fit_regret"] for r in cat])
        fr_ok = np.array([r["fit_regret"] for r in ok if r not in cat])
        print(f"  적합 집합에서의 regret   파국 중앙 {np.median(fr):.4f}   "
              f"나머지 중앙 {np.median(fr_ok):.4f}")
        print("  ★ 파국 쪽 적합 regret 이 **낮으면** 과적합이다 "
              "(표본 문제) / 높으면 적합 실패다")
        print(f"  적합기가 움직인 비율     파국 {np.mean([r['moved'] for r in cat]):.0%}"
              f"   나머지 {np.mean([r['moved'] for r in ok if r not in cat]):.0%}")
        print("\n  구조별 파국 (몰려 있나 흩어져 있나)")
        for i, run in enumerate(SRC_RUNS):
            n_c = sum(1 for r in cat if r["ri"] == i)
            n_o = sum(1 for r in ok if r["ri"] == i)
            print(f"    {run:26s} 항 {n_terms[i]:2d}  파국 {n_c:3d}/{n_o:3d}"
                  f" = {n_c / max(n_o, 1):5.1%}")
        print("\n  ★ 형상/파라미터 비 — 'k' 가 아니라 이것이 단위일 수 있다")
        print(f"  {'k':>4} {'체제당 형상(최소)':>16} {'가중치':>7} "
              f"{'비율':>7} {'파국':>7}")
        for k in KS:
            rows_k = [r for r in ok if r["k"] == k]
            if not rows_k:
                continue
            per = k // len(by_regime)
            w = int(np.median([r["n_weights"] for r in rows_k]))
            c = np.mean([r["holdout"] > 1.15 for r in rows_k])
            print(f"  {k:4d} {per:16d} {w:7d} {per / max(w, 1):7.2f} "
                  f"{c:7.0%}")
    print("\n  ⚠️ 유의성은 붙이지 않는다 — 5090 σ 신뢰구간이 넓다")
    Path(a.out).write_text(json.dumps(
        {"bundle": G5090[0], "n_train": len(train),
         "n_holdout": len(sp.val.shapes), "src_runs": SRC_RUNS,
         "draws": a.draws, "stratified_by_regime": True, "rows": rows,
         "n_terms": n_terms, "raw": out},
        ensure_ascii=False, indent=1))
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
