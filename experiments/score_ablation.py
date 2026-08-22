"""씨앗 절제 채점 — 정준 절차 + 시드별 최악값 병기.

    python3 experiments/score_ablation.py

## 무엇을 가르는가

```
(나) - 기존(1.0652)  =  리포트 정화 효과   ★ 음수일 수 있다
(가) - (나)          =  씨앗의 질
(다)                 =  씨앗이 없을 때의 바닥
```

## 왜 시드 3개인가

20라운드는 확률적이라 궤적 하나로는 못 가린다 (F-10). **중앙과 최악을
함께** 낸다 — 최고만 보면 시드를 늘릴수록 좋아진다.

## 절차 (§30.8b — 절차가 붙지 않은 숫자는 비교하지 않는다)

SOL 2분할 -> 체제마다 가중치 따로 적합 -> 체제별 평가 후 61형상 결합.
홀드아웃은 각 체제 안에서 SOL 순 3개마다 1개(19형상). 유의성은 홀드아웃.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import compare, evaluate, evaluate_scores, geomean
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.rules.physics_seeded import CODE as PS
from kernelrule.rules.physics_seeded import W0 as PS_W0

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
CONDS = [("가-architect", "Architect 씨앗"), ("나-physics", "physics 씨앗"),
         ("다-noseed", "씨앗 없음"),
         # (나') — 프롬프트 상수를 뺀 뒤 같은 조건 (§12.3b). 차이가 곧
         # "그 누출이 얼마나 도움이 됐는가" 다.
         ("clean-나-physics", "physics 씨앗(정화)"),
         # 피처의 물리적 정의를 Optimizer 에게도 준 뒤 (D-34). (다)와의
         # 차이가 "설명이 실제로 기여했나" 의 측정이다.
         ("desc-다-noseed", "씨앗 없음(설명)")]


def main() -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    fast = [p for p in shapes if regime_of(p, table.hw) == "short"]
    slow = [p for p in shapes if regime_of(p, table.hw) == "long"]

    # ★ 진짜 구조 홀드아웃 = 루프가 **한 번도 못 본** 형상 (nk11008 val).
    #   아래 `thirds` 홀드아웃은 61형상 전체에서 뽑으므로 19 중 11 이 루프의
    #   학습 형상이다 — 구조는 그것들을 보고 진화했다. **가중치만**
    #   홀드아웃인 셈이다. 두 숫자를 함께 낸다.
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    loop_train = [p for p in shapes if p not in held]
    st_fit = ([p for p in loop_train if p in fast],
              [p for p in loop_train if p in slow])
    st_ev = ([p for p in held if p in fast], [p for p in held if p in slow])

    def thirds(g):
        g = sorted(g, key=lambda p: table.frame_for(p).index[0])
        return ([p for i, p in enumerate(g) if i % 3 != 2],
                [p for i, p in enumerate(g) if i % 3 == 2])

    f_fit, f_ho = thirds(fast)
    s_fit, s_ho = thirds(slow)
    holdout = f_ho + s_ho

    def canonical(code, w0):
        """체제별 적합 -> (표본내61, 홀드아웃19, 홀드아웃 regret/tol)."""
        out = {}
        for tag, fits, evals in (("in", [fast, slow], [fast, slow]),
                                 ("ho", [f_fit, s_fit], [f_ho, s_ho]),
                                 ("st", list(st_fit), list(st_ev))):
            reg, tol = {}, {}
            for grp, ev_grp in zip(fits, evals, strict=True):
                fit = fit_weights(compile_rule(code), matrix, table,
                                  Split("train", tuple(grp)), w0, max_evals=300)
                e = evaluate_scores(
                    make_score_of(compile_rule(code), matrix, fit.w),
                    table, ev_grp, ks=(1,))
                for i, p in enumerate(e.shapes):
                    reg[p], tol[p] = e.regret[i, 0], e.tol[i]
            out[tag] = (reg, tol)
        g_in = geomean(np.array([out["in"][0][p] for p in shapes]))
        g_ho = geomean(np.array([out["ho"][0][p] for p in holdout]))
        g_st = geomean(np.array([out["st"][0][p] for p in held]))
        return g_in, g_ho, g_st, out["st"]

    vendor = load_vendor(VENDOR)
    v_all = evaluate(vendor_order_fn(table, vendor, mapping="nearest"),
                     table, shapes, ks=(1,))
    v_ho = evaluate(vendor_order_fn(table, vendor, mapping="nearest"),
                    table, held, ks=(1,))

    print("=" * 78)
    print("씨앗 절제 — 정준 절차 채점 (체제별 적합, 61형상 결합)")
    print("=" * 78)
    print(f"  {'조건':16s} {'시드':>4} {'표본내61':>9} {'가중치HO':>10} "
          f"{'★구조HO(20)':>12} {'라운드':>6}")

    rows: dict[str, list] = {}
    trajectories: dict[str, list] = {}
    for key, label in CONDS:
        for s in range(3):
            d = Path("runs") / f"seedabl-{key}-s{s}"
            f = d / "archive.jsonl"
            if not f.exists():
                print(f"  {label:16s} {s:4d}  (없음)")
                continue
            with f.open() as fh:
                arc = [json.loads(ln) for ln in fh if ln.strip()]
            best = min(arc, key=lambda e: e["regret"])
            # ★ 아카이브는 **엘리트 집합**이지 라운드별 시계열이 아니다.
            #   라운드 수와 궤적은 rounds.jsonl 에서 읽어야 한다.
            with (d / "rounds.jsonl").open() as fh:
                rds = [json.loads(ln) for ln in fh if ln.strip()]
            n_r = len(rds)
            try:
                g_in, g_ho, g_st, ho = canonical(best["code"], best["w"])
            except Exception as exc:                        # noqa: BLE001
                print(f"  {label:16s} {s:4d}  실패 {type(exc).__name__}")
                continue
            rows.setdefault(label, []).append((g_in, g_ho, g_st, ho))
            trajectories.setdefault(label, []).append(
                [r["best_regret"] for r in sorted(rds, key=lambda r: r["round"])])
            print(f"  {label:16s} {s:4d} {g_in:9.4f} {g_ho:10.4f} "
                  f"{g_st:12.4f} {n_r:6d}")

    print(f"\n  {'조건':16s} {'표본내61':>9} {'최악':>8}  "
          f"{'★구조HO(20)':>11} {'최악':>8}")
    for label, rs in rows.items():
        ins = sorted(r[0] for r in rs)
        sts = sorted(r[2] for r in rs)
        print(f"  {label:16s} {ins[len(ins) // 2]:9.4f} {ins[-1]:8.4f}  "
              f"{sts[len(sts) // 2]:11.4f} {sts[-1]:8.4f}")
    print(f"  {'벤더 nearest':16s} {v_all.at(1):9.4f} {'':8s}  "
          f"{v_ho.at(1):11.4f}")
    g_ps, _, h_ps, _ = canonical(PS, PS_W0)
    print(f"  {'physics_seeded':16s} {g_ps:9.4f} {'':8s}  {h_ps:11.4f}")
    print("  ※ 기존 evolved(오염,mini)는 분할이 달라 여기 넣지 않는다 (D-31)")

    # -- 유의성 — 조건별 중앙 시드 -----------------------------------------
    print(f"\n{'=' * 78}")
    print("유의성 — ★ 구조 홀드아웃 20형상(nk11008), 조건별 **중앙** 시드")
    print("=" * 78)
    base = evaluate_scores(make_score_of(compile_rule(PS), matrix,
                                         np.ones(len(PS_W0))),
                           table, held, ks=(1,))
    for label, rs in rows.items():
        mid = sorted(rs, key=lambda r: r[2])[len(rs) // 2]
        reg, tol = mid[3]
        ev = replace(base,
                     regret=np.array([reg[p] for p in held]).reshape(-1, 1),
                     tol=np.array([tol[p] for p in held]), label=label)
        c = compare(ev, v_ho, table, name_a="A", name_b="벤더")
        print(f"  {label:16s} {c.geo_a:.4f} vs {c.geo_b:.4f}   "
              f"이김 {int(c.a_wins.sum()):2d} / 짐 {int(c.a_loses.sum()):2d}"
              f" / 구분불가 {int(c.tied.sum()):2d}")

    # -- 궤적 --------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("라운드별 학습 최고 (시드 중앙값) — 씨앗이 출발점만 올리는가")
    print("=" * 78)
    print(f"  {'':16s} " + " ".join(f"r{i:<5d}" for i in range(12)))
    for label, trs in trajectories.items():
        n = min(len(t) for t in trs)
        med = [float(np.median([t[i] for t in trs])) for i in range(n)]
        wor = [float(np.max([t[i] for t in trs])) for i in range(n)]
        print(f"  {label:16s} " + " ".join(f"{x:6.3f}" for x in med))
        print(f"  {'  (최악)':16s} " + " ".join(f"{x:6.3f}" for x in wor))


if __name__ == "__main__":
    main()
