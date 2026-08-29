"""★ 예산 16 실험의 **선행 검사** — 적합기가 16차원에서 버티는가. LLM 0회.

    python3 experiments/fitter_dim.py            # 전체 (약 15분, 12프로세스)
    python3 experiments/fitter_dim.py --arm B16-one

**왜 먼저 하는가.** 예산을 8 -> 16 으로 늘리고 결과가 나빠지면 두 해석이
갈린다: "예산 16 이 나쁘다" 와 "적합기가 16차원에서 못 찾는다". 뒤엣것을
먼저 배제해야 앞엣것을 잴 수 있다 (원칙 1 — 인프라 -> 검사기 -> 피험자).

## 잰다

```
도달률       사전 등록 기준 (D-56). 무작위 4000점이 적합 결과를 못 이긴 비율
             ★ 차원이 오르면 4000점이 상대적으로 성겨져 **저절로 올라간다**.
                이 지표는 8차원과 16차원 사이에서 비교하면 안 된다
차원 손실률  ★ 같은 출발점에서 항만 늘렸을 때 **더 나쁘게 끝나는 비율**
             새 항의 초기 가중치가 작으면 확장 규칙은 원본과 거의 같은 함수다.
             그러므로 16차원 적합이 8차원 적합보다 나쁘게 끝나면, 그 손실은
             **차원 때문**이지 구조 때문이 아니다. 차원에 공평하다
재적합 도달률 무작위 출발점에서 다시 적합한 것이 w0 출발을 이기는 비율
             무작위 '점' 대신 무작위 '적합' 과 견준다 — 차원에 공평하다
```

## 팔

```
A8        원본 8항
B16-lo    16항, 새 항 초기 가중치 0.01,  예산 그대로 (300 / 600)
B16-one   16항, 새 항 초기 가중치 1.0,   예산 그대로
C16-one   16항, 새 항 초기 가중치 1.0,   ★ 예산 비례 (600 / 4000)
```

`B16-*` 두 팔은 **초기 가중치가 교락**이라 나눠 잰다 — 심플렉스 스텝이
`max(|start|, 1.0)` 비례라 0.01 과 1.0 이 같은 스텝을 받지만, 다듬기의
`d * max(|t[i]|, 1.0)` 도 같으므로 차이는 함수 자체에서만 온다.
`C16-one` 은 **차원 손실이 예산 탓인지 알고리즘 탓인지**를 가른다.

regret 의 절대값은 보고하지 않는다 (D-56 §2). 차이와 비율만 쓴다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
#: 사람 24개 라이브러리 팔. 예산 실험의 기준선 팔과 같은 실행들이다.
RUNS = [f"f1pipe-F3-arch24-s{i}" for i in range(6)]
N_PROBE = 4000
PROBE_LO, PROBE_HI = 0.05, 50.0
#: 재적합 도달률의 무작위 출발 횟수. 한 번이 적합 한 번이라 비싸다.
N_RESTART_FITS = 3
TARGET_REACH = 0.90
BIG = 16

ARMS = {
    #  이름        항  새항초기값  max_evals  polish_budget
    "A8":      (8,  None, 300, 600),
    "B16-lo":  (16, 0.01, 300, 600),
    "B16-one": (16, 1.0,  300, 600),
    "C16-one": (16, 1.0,  600, 4000),
}


def extend_code(code: str, f_names: list[str], p_names: list[str],
                n_add: int) -> tuple[str, list[str]]:
    """규칙을 `n_add` 개의 **선형 항**으로 늘린다.

    안 쓰인 피처를 하나씩 붙인다 — 새 항이 기존 항과 같은 물리량을 반복하면
    '차원만 늘었다' 가 아니라 '중복 항이 생겼다' 가 되어 측정이 흐려진다.

    ★ `f` 와 `p` 는 **다른 이름 공간**이다. `f` 는 (형상, config) 행렬이고
    `p` 는 형상 수준 값이라 `p.roofline_ratio` 를 `f.` 로 쓰면 `AttributeError`
    가 난다. 사람 24개 중 5개(`arith_intensity`, `can_use_cp_async`,
    `is_memory_bound`, `log_sol_ms`, `roofline_ratio`)가 형상 수준이라
    `f` 에는 19개뿐이고, 그것만으로는 8항을 못 채우는 규칙이 있다.
    모자라면 **이미 쓴 피처의 제곱**으로 채운다 — 새 물리량은 아니지만
    선형 독립인 항이라 차원은 정직하게 늘어난다. 무엇을 붙였는지 기록한다.
    """
    used_f = set(re.findall(r"\bf\.(\w+)", code))
    used_p = set(re.findall(r"\bp\.(\w+)", code))
    pool = [f"f.{n}" for n in f_names if n not in used_f]
    pool += [f"p.{n}" for n in p_names if n not in used_p]
    # ★ 이진 피처의 제곱은 **자기 자신**이다 (0^2=0, 1^2=1). 그대로 붙이면
    #   완전 중복 항이 생겨 "차원이 늘었다" 가 거짓이 된다. 제외한다.
    cont = [n for n in sorted(used_f)
            if not (n.startswith(("is_", "has_", "can_")))]
    pool += [f"np.square(f.{n})" for n in cont]
    pool += [f"(f.{x} * f.{y})" for i, x in enumerate(cont)
             for y in cont[i + 1:]]
    if len(pool) < n_add:
        raise SystemExit(f"붙일 항이 {len(pool)}개뿐이다 — {n_add}개 필요")
    pick = pool[:n_add]
    k = max(int(i) for i in re.findall(r"\bw\[(\d+)\]", code)) + 1
    lines = [f"    s = s + {e} * w[{k + i}]" for i, e in enumerate(pick)]
    out = re.sub(r"\n(\s*)return s\s*$",
                 "\n" + "\n".join(lines) + r"\n\1return s\n",
                 code.rstrip() + "\n")
    return out, pick


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

    n_terms, init, max_evals, pol = ARMS[arm]
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
    fr = fit_weights(fn, _G["matrix"], _G["table"], sp, w0, max_evals=max_evals,
                     warn_invariants=False, polish=True, polish_budget=pol)

    prob = W._Problem(_G["matrix"], _G["table"], tuple(g), 1)
    rng = np.random.default_rng(7)
    bv = np.inf
    for _ in range(N_PROBE):
        c = np.exp(rng.uniform(np.log(PROBE_LO), np.log(PROBE_HI), size=len(w0)))
        v = prob.regret(fn, c)
        if np.isfinite(v) and v < bv:
            bv = v

    rb = np.inf
    rs = np.random.default_rng(11)
    for _ in range(N_RESTART_FITS):
        st = np.exp(rs.uniform(np.log(PROBE_LO), np.log(PROBE_HI), size=len(w0)))
        try:
            r = fit_weights(fn, _G["matrix"], _G["table"], sp, st,
                            max_evals=max_evals, warn_invariants=False,
                            polish=True, polish_budget=pol)
        except Exception:
            continue
        rb = min(rb, r.fit_regret)

    return dict(arm=arm, run=run, regime=regime, fit=fr.fit_regret,
                n_evals=fr.n_evals, n_fit_evals=fr.n_fit_evals,
                # ★ 다듬기 평가를 뺀 값으로 견준다 — 합산값으로 견주면
                #   다듬기 예산이 상한을 언제나 넘어 100% 로 나온다.
                hit_cap=fr.n_fit_evals >= max_evals,
                moved=bool(fr.moved), seconds=fr.seconds,
                probe_best=float(bv), restart_best=float(rb),
                added=pick)


def summarize(rows: list[dict], arms: list[str]) -> dict:
    """칸에서 팔별 요약을 만든다. **적합을 다시 하지 않는다** — `--reduce`
    로 저장된 json 에서 그대로 다시 뽑을 수 있다."""
    import statistics as _st

    by: dict = {}
    for r in rows:
        by.setdefault(r["arm"], {})[(r["run"], r["regime"])] = r

    print()
    print(f"  {'팔':9s} {'도달률':>8} {'재적합 도달':>11} {'차원 손실':>10} "
          f"{'예산소진':>8} {'이동':>6}")
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
        cap = sum(1 for r in cells.values() if r["hit_cap"])
        mv = sum(1 for r in cells.values() if r["moved"])
        rg = sorted((r["fit"] - r["restart_best"] for r in cells.values()
                     if r["restart_best"] < r["fit"] - 1e-9), reverse=True)
        gaps = ([cells[k]["fit"] - by["A8"][k]["fit"]
                 for k in cells if k in by["A8"]]
                if arm != "A8" and "A8" in by else [])
        losses = [g for g in gaps if g > 1e-9]
        loss = f"{len(losses)}/{len(gaps)}" if gaps else None
        summary[arm] = {
            "n": n, "reach": reach, "restart_reach": rreach,
            "budget_used_up": cap, "moved": mv,
            "dim_loss": loss,
            "dim_loss_max": (max(losses) if losses else 0.0),
            "restart_lost": len(rg),
            "restart_gap_max": (max(rg) if rg else 0.0),
            "restart_gap_median": (_st.median(rg) if rg else 0.0),
            # ★ `_` 로 시작하면 md/json 일치 검사가 건너뛴다 — 칸별 원자료는
            #   재현용이지 보고 대상이 아니다.
            "_gaps_vs_A8": sorted(gaps, reverse=True),
            "_restart_gaps": rg,
        }
        print(f"  {arm:9s} {reach:2d}/{n} {reach / n:5.0%} "
              f"{rreach:2d}/{n} {rreach / n:5.0%} "
              f"{(loss or '—'):>10} {cap:2d}/{n:<5d} {mv:2d}/{n}")

    print()
    for arm in arms:
        g = (summary.get(arm) or {}).get("_gaps_vs_A8")
        if g:
            print(f"  {arm} 대 A8 격차 (양수 = 16차원이 더 나쁘다): "
                  + ", ".join(f"{x:+.4f}" for x in g))

    print()
    a8 = summary.get("A8")
    if a8:
        r = a8["reach"] / a8["n"]
        print(f"  ★ 8차원 도달률 {r:.0%} "
              f"{'통과' if r >= TARGET_REACH else '미달'} — 사전 등록 기준")
    print("  ★ 16차원 도달률은 8차원과 **비교하지 마라** — 무작위 4000점이 "
          "16차원에서 훨씬 성기다")
    print("  ★ '예산소진' 은 이상이 아니라 상태다 — 재시작 일정이 예산을 "
          "설계상 전부 쓴다 (D-76)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", choices=list(ARMS))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default="docs/artifacts/fitter-dim16.json")
    ap.add_argument("--reduce", metavar="JSON",
                    help="저장된 칸에서 요약만 다시 만든다 (적합 없음)")
    a = ap.parse_args()
    arms = a.arm or list(ARMS)

    if a.reduce:
        old = json.loads(Path(a.reduce).read_text())
        rows = old.get("_cells") or old["cells"]
        summary = summarize(rows, arms)
        out = dict(old)
        out["summary"] = summary
        out["_cells"] = rows
        out.pop("cells", None)
        if "procedure" in out:
            out["_procedure"] = out.pop("procedure")
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\n  -> {a.out}")
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tasks = [(arm, run, rg) for arm in arms for run in RUNS
             for rg in ("short", "long")]
    print("=" * 78)
    print(f"16차원 적합기 검사 — {len(tasks)}칸, 팔 {arms}")
    print("=" * 78)

    import multiprocessing as mp
    with mp.Pool(a.jobs, initializer=_init) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(work, tasks), 1):
            rows.append(r)
            print(f"  [{i:2d}/{len(tasks)}] {r['arm']:8s} {r['run'][-2:]:3s} "
                  f"{r['regime']:5s} 적합={r['n_fit_evals']:4d} "
                  f"총={r['n_evals']:5d} "
                  f"{'상한' if r['hit_cap'] else '  ':4s} "
                  f"{r['seconds']:5.1f}s", flush=True)

    summary = summarize(rows, arms)

    Path(a.out).write_text(json.dumps({
        "_procedure": dict(bundle=BUNDLE, runs=RUNS, n_probe=N_PROBE,
                           probe_range=[PROBE_LO, PROBE_HI],
                           n_restart_fits=N_RESTART_FITS, arms=ARMS,
                           note="`fit` 은 재현용 원자료다. 절대값은 보고 "
                                "대상이 아니다 (D-56 §2) — 문서는 차이만 쓴다."),
        "_cells": rows, "summary": summary},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
