"""★ 표본 크기가 문제인가 표본 **선택**이 문제인가. LLM 0회.

    python3 experiments/subset_design.py --workers 6

**사전 등록** `docs/artifacts/subset-design-prereg.md`

## 무작위 뽑기는 최악의 경우를 잰다

`refit_sample.py` 는 체제별 층화 뒤 **무작위**로 뽑는다. 실무에서는
새 GPU 에서 "어느 형상을 잴까" 를 **우리가 정한다** — 형상 그리드
설계는 kernelTab 이 매 캠페인 하는 일이다.

## ★ 표를 재기 전에 계산되는 것만 쓴다

```
쓴다     SOL 하한 / arith_intensity / ridge 대비 위치 / 층 라벨
★ 안 쓴다  난이도 / best_ms / distinct_time_frac — 표를 재야 안다
```

표 유래 값으로 고르면 **그 자체가 누출**이고 "표 없이 고른다" 는
시나리오가 깨진다. 아래 `_features_without_table` 이 그 경계다.

## 전략

```
sol      각 체제 안에서 log(SOL) 분위수를 고르게 덮는다
ridge    ★ ridge 경계에 가까운 것부터 — 거기서 분류가 뒤집힌다
sol+ridge  절반씩
layer    kernelTab 의 층에서 고르게 (★ d_alignment 는 학습 41에 0개다)
```
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
SRC_RUNS = [f"f1pipe-F3-arch24-s{i}" for i in range(6)]
KS = (2, 4, 8, 10, 12)
STRATEGIES = ("sol", "ridge", "sol+ridge", "layer")
#: 파국의 문턱. 사전 등록에 박았다.
CATASTROPHE = 1.15

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


def _features_without_table(p, hw) -> dict:
    """★ **표를 재기 전에** 계산되는 값만. 이것이 이 실험의 경계다.

    `difficulty` / `best_ms` / `distinct_time_frac` 은 여기 없다 —
    표를 재야 알 수 있고, 쓰면 누출이다 (사전 등록 §3-2).
    """
    from kernelrule.core.splits import _DUMMY_CFG
    from kernelrule.features.physical import arith_intensity, log_sol_ms

    ai = arith_intensity(p, hw, _DUMMY_CFG)
    return {"log_sol": log_sol_ms(p, hw, _DUMMY_CFG),
            "arith_intensity": ai,
            # ridge 대비 위치. 1 에 가까울수록 체제 경계다
            "roofline_ratio": ai / hw.ridge_point}


def _spread_pick(vals: np.ndarray, n: int) -> list[int]:
    """`vals` 분포를 고르게 덮는 n개. **분위수 중앙에 가장 가까운 것.**

    결정론이다 — 같은 입력이면 같은 답이다. 동률이면 작은 인덱스.
    """
    if n >= len(vals):
        return list(range(len(vals)))
    qs = [(i + 0.5) / n for i in range(n)]
    targets = np.quantile(vals, qs)
    out: list[int] = []
    for t in targets:
        order = np.argsort(np.abs(vals - t), kind="stable")
        for j in order:
            if int(j) not in out:
                out.append(int(j))
                break
    return out


def _select(strategy: str, k: int) -> list:
    """전략대로 k개. **체제별 층화는 두 팔이 공유하는 조건이다.**"""
    by_regime, feats, layers = _W["by_regime"], _W["feats"], _W["layers"]
    names = sorted(by_regime)
    sizes = np.array([len(by_regime[n]) for n in names], dtype=float)
    take = np.ones(len(names), dtype=int)
    left = k - len(names)
    extra = np.floor(sizes / sizes.sum() * left).astype(int)
    for i in range(left - int(extra.sum())):
        extra[i % len(names)] += 1
    take = take + extra

    out = []
    for name, want in zip(names, take, strict=True):
        pool = by_regime[name]
        n = int(min(want, len(pool)))
        if strategy == "layer":
            # 층별로 돌아가며 하나씩. 층 안에서는 SOL 중앙에 가까운 것
            buckets: dict = {}
            for p in pool:
                buckets.setdefault(layers.get((p.M, p.N, p.K), "?"), []).append(p)
            keys = sorted(buckets)
            picked: list = []
            r = 0
            while len(picked) < n:
                b = buckets[keys[r % len(keys)]]
                rest = [q for q in b if q not in picked]
                if rest:
                    v = np.array([feats[q.key]["log_sol"] for q in rest])
                    picked.append(rest[_spread_pick(v, 1)[0]])
                r += 1
                if r > 100 * n:
                    break
            out.extend(picked)
            continue
        if strategy == "ridge":
            # 경계에 가까운 것부터 (|roofline_ratio - 1| 오름차순)
            d = np.array([abs(feats[q.key]["roofline_ratio"] - 1.0)
                          for q in pool])
            idx = list(np.argsort(d, kind="stable")[:n])
        elif strategy == "sol":
            v = np.array([feats[q.key]["log_sol"] for q in pool])
            idx = _spread_pick(v, n)
        elif strategy == "sol+ridge":
            n_r = n // 2
            d = np.array([abs(feats[q.key]["roofline_ratio"] - 1.0)
                          for q in pool])
            idx = list(np.argsort(d, kind="stable")[:n_r])
            v = np.array([feats[q.key]["log_sol"] for q in pool])
            for j in _spread_pick(v, n):
                if len(idx) >= n:
                    break
                if j not in idx:
                    idx.append(j)
        else:
            raise ValueError(strategy)
        out.extend(pool[int(i)] for i in idx)
    return out


def _job(arg):
    ri, k, strategy = arg
    e = _W["rules"][ri]
    table, matrix, hold = _W["table"], _W["matrix"], _W["hold"]
    sample = _select(strategy, k)
    fn = compile_rule(e["code"])
    ws, fits = {}, []
    for name in ("short", "long"):
        g = [p for p in sample if regime_of(p, table.hw) == name]
        if not g:
            return {"ri": ri, "k": k, "strategy": strategy,
                    "holdout": float("nan"), "empty_regime": name}
        fr = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                         e["w"], max_evals=300,
                          objective="regret")
        ws[name] = fr.w
        fits.append(fr.fit_regret)
    regs = []
    for p in hold:
        cand = table.candidates(p)
        sc = make_score_of(fn, matrix, ws[regime_of(p, table.hw)])(p, cand)
        t = table.times_of(p)
        regs.append(float(t[cand.top_k(sc, 1)[0]] / t.min()))
    return {"ri": ri, "run": _W["runs"][ri], "k": k, "strategy": strategy,
            "n_sample": len(sample),
            "shapes": [[p.M, p.N, p.K] for p in sample],
            "fit_regret": float(np.exp(np.mean(np.log(fits)))),
            "holdout": float(geomean(np.array(regs)))}


def _wilson_hi(k: int, n: int, z: float = 1.96) -> float:
    """0/n 을 '파국 없음' 으로 쓰지 않기 위한 95% 상한 (원칙 27)."""
    if n == 0:
        return 1.0
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    r = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return float(min(1.0, (c + r) / d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="docs/artifacts/subset-design.json")
    ap.add_argument("--random-json", default="docs/artifacts/refit-sample.json",
                    help="무작위 팔. 같은 k 끼리만 견준다")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    table = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    sp = _splits(table)
    train = list(sp.train.shapes)

    # ★ 조인 키를 맞춘다. `p.key` 는 (M,N,K,dtype) 이고 `shape_layers`
    #   는 [M,N,K] 다 — 그대로 넣으면 **하나도 안 맞는다.**
    layers: dict = {}
    for name, shs in (table.meta.get("shape_layers") or {}).items():
        for sh in shs:
            layers[(sh[0], sh[1], sh[2])] = name
    feats = {p.key: _features_without_table(p, table.hw) for p in train}
    by_regime: dict = {}
    for p in train:
        by_regime.setdefault(regime_of(p, table.hw), []).append(p)

    rules = []
    for r in SRC_RUNS:
        f = Path("runs") / r / "archive.jsonl"
        arc = sorted((json.loads(x) for x in f.read_text().splitlines()
                      if x.strip()), key=lambda e: e["regret"])
        rules.append(arc[0])

    _W.update(table=table, matrix=matrix, train=train, by_regime=by_regime,
              hold=list(sp.val.shapes), rules=rules, runs=SRC_RUNS,
              feats=feats, layers=layers)

    print("=" * 78)
    print("표본 크기인가 표본 선택인가 — 설계된 부분집합")
    print("=" * 78)
    print(f"  5090 학습 {len(train)}형상  "
          + "  ".join(f"{n} {len(v)}" for n, v in sorted(by_regime.items())))
    lay_n: dict = {}
    for p in train:
        key = layers.get((p.M, p.N, p.K), "?")
        lay_n[key] = lay_n.get(key, 0) + 1
    print(f"  층 분포 {dict(sorted(lay_n.items()))}")
    # ★ 라벨이 안 붙으면 `layer` 전략은 **한 통에서 뽑는 것**이 된다.
    #   숫자는 나오는데 전략이 안 돈 것이다 (원칙 1). 멈춘다.
    if lay_n.get("?", 0):
        raise SystemExit(
            f"★ 층 라벨이 안 붙은 형상 {lay_n['?']}개. `shape_layers` 조인 "
            "키를 확인하라 — `p.key` 는 (M,N,K,dtype) 이고 번들은 "
            "[M,N,K] 다.\n  조용히 두면 `layer` 전략이 한 통에서 뽑으면서 "
            "결과는 그럴듯하게 나온다.")
    absent = sorted(set(table.meta.get("shape_layers") or {}) - set(lay_n))
    if absent:
        print(f"  ★ 학습 41형상에 **없는 층**: {absent}  "
              "— '층 균등' 은 나머지 층 균등이다")
    print(f"  구조 {len(rules)}개   전략 {list(STRATEGIES)}")
    print("  ★ 설계 팔은 결정론이다 — 뽑기 운이 없다\n")

    jobs = [(ri, k, st) for ri in range(len(rules)) for k in KS
            for st in STRATEGIES]
    if a.workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=a.workers,
                                 mp_context=get_context("fork")) as ex:
            out = list(ex.map(_job, jobs, chunksize=2))
    else:
        out = [_job(j) for j in jobs]

    rnd: dict = {}
    rp = Path(a.random_json)
    if rp.exists():
        for row in json.loads(rp.read_text())["rows"]:
            rnd[row["k"]] = row["values"]

    print(f"  {'k':>4} {'전략':>10} {'중앙':>8} {'최악':>8} "
          f"{'파국':>7} {'95%상한':>8}   {'무작위 파국':>12}")
    rows = []
    for k in KS:
        for st in STRATEGIES:
            v = np.array([r["holdout"] for r in out
                          if r["k"] == k and r["strategy"] == st
                          and np.isfinite(r["holdout"])])
            if not len(v):
                continue
            nc = int((v > CATASTROPHE).sum())
            hi = _wilson_hi(nc, len(v))
            rv = np.array(rnd.get(k, []))
            rtxt = (f"{(rv > CATASTROPHE).mean():11.0%}" if len(rv)
                    else f"{'—':>11}")
            print(f"  {k:4d} {st:>10} {np.median(v):8.4f} {v.max():8.4f} "
                  f"{nc:3d}/{len(v):<3d} {hi:8.0%}   {rtxt}")
            rows.append({"k": k, "strategy": st, "median": float(np.median(v)),
                         "max": float(v.max()), "n_catastrophe": nc,
                         "n": len(v), "wilson_hi": hi,
                         "values": [float(x) for x in v]})
        print()

    print("  ⚠️ 설계 팔은 k마다 6건뿐이다 (구조 6 x 뽑기 1). "
          "0/6 은 '파국 없음' 이 아니라 **95% 상한 39%** 다 (원칙 27).")
    print("  ⚠️ 유의성은 붙이지 않는다.")
    Path(a.out).write_text(json.dumps(
        {"bundle": G5090[0], "n_train": len(train), "src_runs": SRC_RUNS,
         "strategies": list(STRATEGIES), "catastrophe": CATASTROPHE,
         "layer_counts": lay_n, "rows": rows, "raw": out},
        ensure_ascii=False, indent=1))
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
