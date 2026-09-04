"""★ 벤더 비교 — 형상별이 주 지표다. LLM 0회.

    python3 experiments/vendor_compare.py

## 왜 형상별인가

```
보조   실행 6개의 부호검정.  p 하한 0.031 — 5/6 이면 p=0.22 로 아무 말도 못 한다
★ 주   각 형상에서 6실행의 **중앙값** vs 벤더, 형상 20개 부호검정
       p 하한 ~1e-6.  분산이 한 번만 든다 (실행 간 분산이 중앙값으로 흡수)
```

형상별이 §30.4 의 "geomean 은 소수 형상이 끈다" 문제도 피한다.
**둘 다 보고하되 형상별이 주 지표다** (재실행 실험 계획서 §3).
"""

from __future__ import annotations

import json
import warnings
from math import comb
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate, evaluate_scores, geomean
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY, FeatureRegistry
from kernelrule.features.loader import extended_registry, load_generated

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"

#: 두 팔. `(실행 접두사, 라이브러리 만드는 법)`
ARMS = {
    "F1 21개": ("F1rw-p8-s",
                "runs/F1rw-p8/stage1-features/proposals.jsonl"),
    "사람 24개": ("F3rw-p8-s", None),
}


def _sign_test(wins: int, losses: int) -> float:
    """양측 부호검정. 동점은 뺀다."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    warnings.simplefilter("ignore")
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]

    # -- 벤더: 형상별 regret ---------------------------------------------
    vend = load_vendor(VENDOR)
    ev = evaluate(vendor_order_fn(table, vend, mapping="nearest"),
                  table, held, ks=(1,), label="벤더")
    v_by_shape = {p: float(ev.regret[i, 0]) for i, p in enumerate(ev.shapes)}
    print("=" * 78)
    print("벤더 비교 — ★ 형상별이 주 지표")
    print("=" * 78)
    print(f"  구조 홀드아웃 {len(held)}형상   "
          f"벤더 geomean {geomean(np.array(list(v_by_shape.values()))):.4f}\n")

    for tag, (pre, lib) in ARMS.items():
        if lib:
            reg = extended_registry(FeatureRegistry("F1-empty"),
                                    load_generated(lib, table=table,
                                                   exclude=set()),
                                    name="F1-lib")
        else:
            reg = FeatureRegistry("human24")
            for n in sorted(REGISTRY._items):
                reg.add(REGISTRY[n])
        matrix = FeatureMatrix(table, reg)

        # 실행마다 형상별 regret. 최종 채점 절차대로 **체제별로 재적합**한다.
        per_run: list[dict] = []
        for s in range(6):
            arc = Path("runs") / f"{pre}{s}" / "archive.jsonl"
            if not arc.exists():
                continue
            rows = [json.loads(ln) for ln in arc.open() if ln.strip()]
            if not rows:
                continue
            best = min(rows, key=lambda e: e["regret"])
            fn = compile_rule(best["code"])
            reg_by_shape: dict = {}
            for name in ("short", "long"):
                g_tr = [p for p in train if regime_of(p, table.hw) == name]
                g_ho = [p for p in held if regime_of(p, table.hw) == name]
                fr = fit_weights(fn, matrix, table, Split("train", tuple(g_tr)),
                                 best["w"], max_evals=300,
                                 warn_invariants=False,
                          objective="regret")
                e = evaluate_scores(make_score_of(fn, matrix, fr.w), table,
                                    g_ho, ks=(1,))
                for i, p in enumerate(e.shapes):
                    reg_by_shape[p] = float(e.regret[i, 0])
            per_run.append(reg_by_shape)

        _report(tag, per_run, v_by_shape, held, table)


def _report(tag, per_run, v_by_shape, held, table) -> None:
    print("=" * 78)
    print(f"{tag}   실행 {len(per_run)}개")
    print("=" * 78)

    # ★ 주 지표 — 형상별 중앙값 vs 벤더
    med = {p: float(np.median([r[p] for r in per_run if p in r]))
           for p in held if any(p in r for r in per_run)}
    rows = [(p, med[p], v_by_shape[p]) for p in med if p in v_by_shape]
    w = sum(1 for _, m, v in rows if m < v - 1e-9)
    lo = sum(1 for _, m, v in rows if m > v + 1e-9)
    print(f"  ★ 형상별  이김 {w} / 짐 {lo} / 동점 {len(rows)-w-lo}"
          f"   부호검정 p = {_sign_test(w, lo):.2e}")
    print(f"     geomean  우리 {geomean(np.array([m for _, m, _ in rows])):.4f}"
          f"   벤더 {geomean(np.array([v for _, _, v in rows])):.4f}")

    # 보조 — 실행 6개
    vg = geomean(np.array([v_by_shape[p] for p in held if p in v_by_shape]))
    runs = [geomean(np.array([r[p] for p in held if p in r])) for r in per_run]
    rw = sum(1 for x in runs if x < vg)
    print(f"  보조 실행별  이김 {rw}/{len(runs)}"
          f"   부호검정 p = {_sign_test(rw, len(runs)-rw):.3f}")

    # ★ 체제별 분해
    print(f"\n  {'체제':22s} {'형상':>4} {'우리':>8} {'벤더':>8} "
          f"{'이김/짐':>9} {'p':>9}")
    for name, label in (("short", "빠른 (SOL<0.5ms)"),
                        ("long", "느린 (SOL>=0.5ms)")):
        g = [(p, m, v) for p, m, v in rows
             if regime_of(p, table.hw) == name]
        if not g:
            continue
        gw = sum(1 for _, m, v in g if m < v - 1e-9)
        gl = sum(1 for _, m, v in g if m > v + 1e-9)
        print(f"  {label:22s} {len(g):4d} "
              f"{geomean(np.array([m for _, m, _ in g])):8.4f} "
              f"{geomean(np.array([v for _, _, v in g])):8.4f} "
              f"{gw:4d}/{gl:<4d} {_sign_test(gw, gl):9.3f}")
    print()


if __name__ == "__main__":
    main()
