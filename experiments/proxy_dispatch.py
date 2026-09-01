"""8-1 대리 지표 디스패치 — 배포 시점에 체제를 알 수 있는가. LLM 호출 0회.

    python3 experiments/proxy_dispatch.py

## 왜 재는가

체제별 재적합(§29.5 b)이든 층화 보고든, **배포 시점에 체제를 알 수 있어야**
쓸모가 있다. `t_best` 는 전수 측정을 해야 아는 값이므로 그것으로 자른
숫자는 오라클이지 산출물이 아니다.

    대리:  SOL = max(2MNK/실효피크, 바이트/실효대역폭)   형상 + 하드웨어만
    정답:  t_best                                      ★ 전수 측정 필요

두 정의의 일치율, 그리고 그 일치가 얼마나 견고한지(경계 여유)를 잰다.
**100% 일치도 여유가 얇으면 이 표의 우연**이므로 둘 다 봐야 한다.

## 결과 (2026-08-21)

61/61 일치. 다만 경계 최근접 여유가 1.13배뿐이다. `docs/glossary.md` 참조.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401  — REGISTRY 를 채운다
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate, evaluate_scores, geomean
from kernelrule.core.splits import _DUMMY_CFG, Split
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.features.physical import log_sol_ms
from kernelrule.rules.physics_seeded import CODE as PS
from kernelrule.rules.physics_seeded import W0 as PS_W0

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
RUN_REAL = Path("runs/real-gpt-5.4-mini-2026-03-17/archive.jsonl")

#: 빠른/느린 체제의 경계 (§30). 아래에서 눈금 해상도가 순위를 지배한다.
BOUNDARY_MS = 0.5


def _setup():
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    sol = {p: 2 ** log_sol_ms(p, table.hw, _DUMMY_CFG) for p in shapes}
    best = {p: float(table.times_of(p).min()) for p in shapes}
    return table, shapes, sol, best


def main() -> None:
    table, shapes, sol, best = _setup()
    matrix = FeatureMatrix(table, REGISTRY)
    proxy_fast = {p: sol[p] < BOUNDARY_MS for p in shapes}
    true_fast = {p: best[p] < BOUNDARY_MS for p in shapes}

    print("=" * 74)
    print("8-1. 체제 판정 — 대리 지표 vs 정답")
    print("=" * 74)
    print("  대리:  SOL < 0.5ms   형상 + 하드웨어 상수만. 배포 시점에 안다")
    print("  정답:  t_best < 0.5ms                    ★ 전수 측정 필요\n")

    tp = sum(1 for p in shapes if proxy_fast[p] and true_fast[p])
    fp = sum(1 for p in shapes if proxy_fast[p] and not true_fast[p])
    fn = sum(1 for p in shapes if not proxy_fast[p] and true_fast[p])
    tn = sum(1 for p in shapes if not proxy_fast[p] and not true_fast[p])
    n = len(shapes)
    print("                     정답 빠름   정답 느림")
    print(f"  대리 빠름          {tp:8d}   {fp:9d}")
    print(f"  대리 느림          {fn:8d}   {tn:9d}")
    print(f"\n  일치 {tp + tn}/{n} = {(tp + tn) / n:.1%}   "
          f"재현율(빠름) {tp / max(tp + fn, 1):.1%}   불일치 {fp + fn}개")
    for label, cond in (("대리는 빠르다는데 실제로 느림",
                         lambda p: proxy_fast[p] and not true_fast[p]),
                        ("대리는 느리다는데 실제로 빠름",
                         lambda p: not proxy_fast[p] and true_fast[p])):
        bad = [p for p in shapes if cond(p)]
        if bad:
            print(f"  ★ {label}:")
            for p in bad:
                print(f"      {p.M}x{p.N}x{p.K}  SOL {sol[p] * 1000:7.1f}us"
                      f"  t_best {best[p] * 1000:8.1f}us")

    # -- 무엇이 디스패치를 필요로 하는가 ------------------------------------
    with RUN_REAL.open() as fh:
        archive = [json.loads(ln) for ln in fh if ln.strip()]
    evolved = min(archive, key=lambda e: e["regret"])
    vendor = load_vendor(VENDOR)

    def refit(code, w0, sh):
        return fit_weights(compile_rule(code), matrix, table,
                           Split("train", tuple(sh)), w0, max_evals=300,
                          objective="regret")

    def score(code, w, sh):
        return evaluate_scores(make_score_of(compile_rule(code), matrix, w),
                               table, sh, ks=(1,))

    print(f"\n{'=' * 74}")
    print("어떤 산출물이 체제 판정을 필요로 하는가")
    print("=" * 74)
    ev = score(evolved["code"], evolved["w"], shapes)
    print(f"  evolved (단일 규칙, 디스패치 없음)     전체61 {ev.at(1):.4f}")
    print("    -> 모든 형상에 같은 규칙을 쓴다. 체제 판정이 **필요 없다**.")
    print("       is_memory_bound 로 분기하지만 그것도 roofline 대리 지표다.")

    for name, part in (("대리(SOL)", proxy_fast),
                       ("정답(t_best) ★오라클", true_fast)):
        fast = [p for p in shapes if part[p]]
        slow = [p for p in shapes if not part[p]]
        e_f = score(PS, refit(PS, PS_W0, fast).w, fast)
        e_s = score(PS, refit(PS, PS_W0, slow).w, slow)
        i_f = {p: i for i, p in enumerate(e_f.shapes)}
        i_s = {p: i for i, p in enumerate(e_s.shapes)}
        per = np.array([e_f.regret[i_f[p], 0] if p in i_f
                        else e_s.regret[i_s[p], 0] for p in shapes])
        print(f"\n  physics_seeded 체제별 재적합 — 경계 {name}")
        print(f"    빠른 {len(fast):2d}형상 {e_f.at(1):.4f} | "
              f"느린 {len(slow):2d}형상 {e_s.at(1):.4f} | "
              f"전체61 {geomean(per):.4f}")

    v = evaluate(vendor_order_fn(table, vendor, mapping="nearest"),
                 table, shapes, ks=(1,), label="벤더")
    print(f"\n  벤더 전체61 {v.at(1):.4f}   ★ 관문 1.080")


def margins() -> None:
    """★ 100% 일치가 견고한가 — 경계 여유를 본다.

    SOL 은 하한이므로 `t_best` 는 항상 그 위다. 두 판정이 갈리는 것은
    `SOL < 0.5 <= t_best` 인 좁은 띠에서만이다. **그 띠에 형상이 없었을
    뿐인지, 원리적으로 안전한지**를 구분해야 한다.
    """
    _, shapes, sol, best = _setup()
    rows = sorted(((abs(math.log2(sol[p] / BOUNDARY_MS)), p)
                   for p in shapes), key=lambda r: r[0])
    print("경계(0.5ms)에 가장 가까운 형상 8개 — 오판은 여기서 난다")
    print(f"  {'형상':22s} {'SOL(us)':>10} {'t_best(us)':>11} "
          f"{'여유(배)':>9}  판정")
    for d, p in rows[:8]:
        agree = (sol[p] < BOUNDARY_MS) == (best[p] < BOUNDARY_MS)
        print(f"  {p.M}x{p.N}x{p.K:<10} {sol[p] * 1000:10.1f} "
              f"{best[p] * 1000:11.1f} {2 ** d:9.2f}  "
              f"{'일치' if agree else '★불일치'}")
    ratios = sorted(best[p] / sol[p] for p in shapes)
    close = sum(1 for d, _ in rows if 2 ** d < 2.0)
    print(f"\n  경계에서 2배 이내: {close}/{len(rows)}형상")
    print(f"  t_best / SOL 중앙값: {ratios[len(ratios) // 2]:.3f}")
    lo = BOUNDARY_MS / ratios[len(ratios) // 2]
    print(f"  ★ 위험 띠: SOL ∈ [{lo * 1000:.0f}, {BOUNDARY_MS * 1000:.0f}] us"
          "  — 이 구간의 형상은 사실상 동전 던지기다")


if __name__ == "__main__":
    main()
    print()
    margins()
