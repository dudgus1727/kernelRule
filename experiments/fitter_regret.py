"""★ §2 관문 — **regret 경로**에서 적합기가 16차원을 버티는가. LLM 0회.

    OMP_NUM_THREADS=1 python3 -m experiments.fitter_regret --jobs 20
    python3 -m experiments.fitter_regret --reduce docs/artifacts/fitter-regret.json

★ **`-m` 으로 부른다** — `experiments.fitter_dim` 에서 절차 상수를
가져오므로 저장소 뿌리가 `sys.path` 에 있어야 한다.

사전 등록 `docs/artifacts/fitter-regret-prereg.md`. 절차는 D-77
(`fitter_dim.py`) 과 **같다** — 새 지표를 만들지 않는다 (원칙 2).
다른 것은 팔뿐이다: 총 evals 900 을 다섯 팔에 똑같이 준다.

```
A8          8항  Nelder-Mead 300/4 + 다듬기 600     <- 장치 점검 (D-77 100%)
NM16       16항  Nelder-Mead 300/4 + 다듬기 600     <- 지금 (D-77 B16-lo)
CMA16      16항  CMA-ES      300/1 + 다듬기 600
CMA16-full 16항  CMA-ES      900/1 + 다듬기 없음
SURR16     16항  대리 150 -> regret NM 150/2 + 다듬기 600   <- (나)
```

⚠️ `SURR16` 은 1단계에 순위 손실 조각(`rank_loss_top1`)이 들어간다.
**초기점 생성에만 쓰고 채택은 regret 이다** (사전 등록 §2).

regret 의 절대값은 문서에 보고하지 않는다 (D-56 §2). 차이와 비율만 쓴다.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.fitter_dim import (
    BUNDLE,
    N_PROBE,
    N_RESTART_FITS,
    PROBE_HI,
    PROBE_LO,
    RUNS,
    TARGET_REACH,
    extend_code,
)

#: 팔마다 **같은 총 evals**. 사전 등록 §2 의 표 그대로.
TOTAL_EVALS = 900

ARMS = {
    #  이름          항  새항초기값  최적화기        적합  재시작  다듬기  대리
    "A8":         (8,  None, "nelder-mead", 300, 4, 600, 0),
    "NM16":       (16, 0.01, "nelder-mead", 300, 4, 600, 0),
    "CMA16":      (16, 0.01, "cma",         300, 1, 600, 0),
    "CMA16-full": (16, 0.01, "cma",         900, 1, 0,   0),
    "SURR16":     (16, 0.01, "nelder-mead", 150, 2, 600, 150),
}
#: 16항 팔 — 판정선과 "같은 점" 비교의 대상이다. `A8` 은 장치 점검이다.
ARMS16 = [a for a, v in ARMS.items() if v[0] == 16]
#: 사전 등록 §4 의 "같은 점" 기술 기준. 판정선이 아니다.
SAME_VALUE = 1e-9
SAME_DIR = 0.999

_G: dict = {}


def _init() -> None:
    warnings.simplefilter("ignore")
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import regime_of
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    _G["table"] = table
    _G["matrix"] = FeatureMatrix(table, REGISTRY)
    _G["f_names"] = sorted(REGISTRY.names(shape_level=False))
    _G["p_names"] = sorted(REGISTRY.names(shape_level=True))
    _G["groups"] = {n: [p for p in train if regime_of(p, table.hw) == n]
                    for n in ("short", "long")}


def work(task: tuple[str, str, str]) -> dict:
    arm, run, regime = task
    from kernelrule.core import weights as W
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.splits import Split
    from kernelrule.core.weights import fit_weights

    n_terms, init, method, max_evals, n_restarts, pol, init_evals = ARMS[arm]
    arc = [json.loads(ln) for ln in
           (Path("runs") / run / "archive.jsonl").read_text().splitlines()
           if ln.strip()]
    best = min(arc, key=lambda e: e["regret"])
    code, w0 = best["code"], list(best["w"])
    pick: list[str] = []
    if n_terms > len(w0):
        code, pick = extend_code(code, _G["f_names"], _G["p_names"],
                                 n_terms - len(w0))
        w0 = w0 + [init] * (n_terms - len(w0))
    fn = compile_rule(code)

    g = _G["groups"][regime]
    sp = Split("train", tuple(g))
    # ★ `objective="regret"` 를 **리터럴로** 넘긴다 — 실험 스크립트가
    #   목적함수를 밝히는지 시험이 소스에서 검사한다
    #   (`test_history_experiments_pin_their_objective`).
    kw = dict(max_evals=max_evals, n_restarts=n_restarts,
              warn_invariants=False, polish=pol > 0, polish_budget=pol,
              method=method)
    if init_evals:
        kw["init_objective"] = "rank_top1"
        kw["init_evals"] = init_evals
    fr = fit_weights(fn, _G["matrix"], _G["table"], sp, w0,
                     objective="regret", **kw)

    # ★ 탐침은 적합과 **같은 목적함수**로 잰다 (D-103). 여기는 전부 regret 이다.
    prob = W._Problem(_G["matrix"], _G["table"], tuple(g), 1)

    def _value(w):
        return prob.regret(fn, w)

    rng = np.random.default_rng(7)
    bv = np.inf
    for _ in range(N_PROBE):
        c = np.exp(rng.uniform(np.log(PROBE_LO), np.log(PROBE_HI), size=len(w0)))
        v = _value(c)
        if np.isfinite(v) and v < bv:
            bv = v

    rb = np.inf
    rs = np.random.default_rng(11)
    for _ in range(N_RESTART_FITS):
        stw = np.exp(rs.uniform(np.log(PROBE_LO), np.log(PROBE_HI),
                                size=len(w0)))
        try:
            r = fit_weights(fn, _G["matrix"], _G["table"], sp, stw,
                            objective="regret", **kw)
        except Exception:
            continue
        rb = min(rb, _value(r.w))

    return dict(arm=arm, run=run, regime=regime, fit=_value(fr.w),
                fit_regret=fr.fit_regret, n_evals=fr.n_evals,
                n_fit_evals=fr.n_fit_evals, n_init_evals=fr.n_init_evals,
                moved=bool(fr.moved), seconds=fr.seconds,
                probe_best=float(bv), restart_best=float(rb),
                w=[float(x) for x in fr.w], added=pick)


def _same_point(by: dict) -> dict:
    """★ 사전 등록 §4 — 팔들이 **같은 점**에 도달했는가. 기술이지 판정이 아니다."""
    out: dict = {}
    for a, b in combinations(ARMS16, 2):
        ca, cb = by.get(a, {}), by.get(b, {})
        keys = sorted(set(ca) & set(cb))
        if not keys:
            continue
        dv, cs = [], []
        for kk in keys:
            dv.append(abs(ca[kk]["fit"] - cb[kk]["fit"]))
            wa = np.asarray(ca[kk]["w"], dtype=np.float64)
            wb = np.asarray(cb[kk]["w"], dtype=np.float64)
            na, nb = np.linalg.norm(wa), np.linalg.norm(wb)
            cs.append(float(wa @ wb / (na * nb)) if na and nb else float("nan"))
        fin = [c for c in cs if np.isfinite(c)]
        out[f"{a} vs {b}"] = {
            "n": len(keys),
            "same_value": sum(1 for d in dv if d < SAME_VALUE),
            "dfit_median": st.median(dv), "dfit_max": max(dv),
            "cos_median": (st.median(fin) if fin else float("nan")),
            "cos_min": (min(fin) if fin else float("nan")),
            "same_dir": sum(1 for c in fin if c >= SAME_DIR),
        }
    return out


def summarize(rows: list[dict], arms: list[str]) -> dict:
    by: dict = {}
    for r in rows:
        by.setdefault(r["arm"], {})[(r["run"], r["regime"])] = r

    print()
    print(f"  {'팔':11s} {'도달률':>9} {'재적합 도달':>11} {'이동':>7} "
          f"{'A8대비 손실':>11} {'실제 evals':>11}")
    summary: dict = {}
    for arm in arms:
        cells = by.get(arm, {})
        if not cells:
            continue
        n = len(cells)
        reach = sum(1 for r in cells.values()
                    if not (r["probe_best"] < r["fit"] - 1e-9))
        rreach = sum(1 for r in cells.values()
                     if not (r["restart_best"] < r["fit"] - 1e-9))
        mv = sum(1 for r in cells.values() if r["moved"])
        rg = sorted((r["fit"] - r["restart_best"] for r in cells.values()
                     if r["restart_best"] < r["fit"] - 1e-9), reverse=True)
        gaps = ([cells[k]["fit"] - by["A8"][k]["fit"]
                 for k in cells if k in by["A8"]]
                if arm != "A8" and "A8" in by else [])
        losses = [g for g in gaps if g > 1e-9]
        ev = [r["n_evals"] for r in cells.values()]
        summary[arm] = {
            "n": n, "reach": reach, "restart_reach": rreach, "moved": mv,
            "dim_loss": (f"{len(losses)}/{len(gaps)}" if gaps else None),
            "dim_loss_max": (max(losses) if losses else 0.0),
            "restart_lost": len(rg),
            "restart_gap_max": (max(rg) if rg else 0.0),
            "restart_gap_median": (st.median(rg) if rg else 0.0),
            "evals_median": st.median(ev), "evals_max": max(ev),
            "_gaps_vs_A8": sorted(gaps, reverse=True),
            "_restart_gaps": rg,
        }
        s = summary[arm]
        print(f"  {arm:11s} {reach:2d}/{n} {reach / n:5.0%} "
              f"{rreach:2d}/{n} {rreach / n:5.0%} {mv:2d}/{n:<4d} "
              f"{(s['dim_loss'] or '—'):>11} "
              f"{s['evals_median']:5.0f}/{s['evals_max']:<5.0f}")

    print()
    for arm in arms:
        g = (summary.get(arm) or {}).get("_gaps_vs_A8")
        if g:
            print(f"  {arm} 대 A8 격차 (양수 = 16차원이 더 나쁘다): "
                  + ", ".join(f"{x:+.4f}" for x in g))

    same = _same_point(by)
    if same:
        print()
        print(f"  ★ 도달한 점이 같은가 (같은 값 <{SAME_VALUE:g}, "
              f"같은 방향 cos>={SAME_DIR})")
        print(f"  {'짝':26s} {'같은 값':>8} {'|dfit| 중앙':>11} "
              f"{'최대':>8} {'cos 중앙':>9} {'최소':>8} {'같은 방향':>9}")
        for kk, v in same.items():
            print(f"  {kk:26s} {v['same_value']:2d}/{v['n']:<5d} "
                  f"{v['dfit_median']:11.4f} {v['dfit_max']:8.4f} "
                  f"{v['cos_median']:9.3f} {v['cos_min']:8.3f} "
                  f"{v['same_dir']:2d}/{v['n']}")
    summary["_same_point"] = same

    print()
    a8 = summary.get("A8")
    if a8:
        r = a8["reach"] / a8["n"]
        print(f"  ★ 장치 점검 — A8 도달률 {r:.0%} "
              f"{'통과' if r >= TARGET_REACH else '★ 미달 (D-77 은 100%)'}")
    passed = [a for a in ARMS16
              if a in summary and summary[a]["reach"] / summary[a]["n"]
              >= TARGET_REACH]
    summary["_pass"] = passed
    if passed:
        print(f"  ★ 판정 통과 — 도달률 {TARGET_REACH:.0%} 이상인 16항 팔: "
              f"{passed}")
        print("     동점이면 재적합 도달률로, 그것도 같으면 NM16 (사전 등록 §3)")
    else:
        print(f"  ★ 판정 — 16항 팔 전부 도달률 {TARGET_REACH:.0%} 미만. "
              "D-77 은 regret 경로의 성질이다")
        print("     -> §3-1(예산 8 vs 16)을 접고 3-2·3-3 만 예산 8 로 (지시문 §5)")
    print("  ★ 도달률을 8차원과 16차원 사이에서 비교하지 마라 — 무작위 "
          "4000점이 16차원에서 훨씬 성기다 (D-77)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", choices=list(ARMS))
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", default="docs/artifacts/fitter-regret.json")
    ap.add_argument("--reduce", metavar="JSON",
                    help="저장된 칸에서 요약만 다시 만든다 (적합 없음)")
    a = ap.parse_args()
    arms = a.arm or list(ARMS)

    if a.reduce:
        old = json.loads(Path(a.reduce).read_text())
        rows = old["_cells"]
        out = dict(old)
        out["summary"] = summarize(rows, arms)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\n  -> {a.out}")
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tasks = [(arm, run, rg) for arm in arms for run in RUNS
             for rg in ("short", "long")]
    print("=" * 78)
    print(f"§2 관문 — regret 경로 16차원 적합기, {len(tasks)}칸, 팔 {arms}")
    print(f"★ 총 evals {TOTAL_EVALS} 을 팔마다 같게 준다 — 실제 값을 보고한다")
    print("=" * 78)

    import multiprocessing as mp
    with mp.Pool(a.jobs, initializer=_init) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(work, tasks), 1):
            rows.append(r)
            print(f"  [{i:2d}/{len(tasks)}] {r['arm']:11s} {r['run'][-2:]:3s} "
                  f"{r['regime']:5s} 적합={r['n_fit_evals']:4d} "
                  f"대리={r['n_init_evals']:4d} 총={r['n_evals']:5d} "
                  f"{r['seconds']:6.1f}s", flush=True)

    summary = summarize(rows, arms)
    Path(a.out).write_text(json.dumps({
        "_procedure": dict(
            bundle=BUNDLE, runs=RUNS, objective="regret", n_probe=N_PROBE,
            probe_range=[PROBE_LO, PROBE_HI],
            n_restart_fits=N_RESTART_FITS, arms=ARMS,
            total_evals=TOTAL_EVALS, prereg="fitter-regret-prereg.md",
            note="`fit` 은 재현용 원자료다. 절대값은 보고 대상이 아니다 "
                 "(D-56 §2) — 문서는 차이만 쓴다."),
        "_cells": rows, "summary": summary},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
