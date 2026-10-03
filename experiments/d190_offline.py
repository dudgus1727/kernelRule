"""★ D-190 — 루프를 고치기 전에 c2 a6000 규칙으로 먼저 잰다. **LLM 0회 · GPU 0.**

    python3 -m experiments.d190_offline prune  --out docs/artifacts/d190/prune.json
    python3 -m experiments.d190_offline select --out docs/artifacts/d190/select.json

## prune — 죽은 가중치 정리 (개선 7번)

c2 a6000 16실행의 Archive 엘리트 전부(약 136개)에 대해:

```
죽은 가중치   dead_by_weight (|w| < 1e-3) ∪ dead_by_sensitivity (w[i] 를 ±50%
              바꿔도 train regret 변화 < 1e-6) — fit_weights 와 같은 정의
순서          점수 기여(_contributions, 시간을 안 읽음)가 작은 것부터
한 번에 하나  항을 지우고 재적합 없이 train 을 채점 -> 나빠지지 않으면 둔다
보고          holdout (fold 의 val) — 정리 전 · 후
```

## select — train 에서 떼어 낸 val 로 고르기 (개선 3번)

```
val     fold 의 train 을 (N,K) 그룹 단위 nkband k=4 로 다시 나눈 것 중 하나
        (실행 s 는 안쪽 fold s) — holdout 이 (N,K) 그룹 통째라서 같은 성격
적합    train - val 로, 루프와 같은 적합기(fitter_for) · ★ LLM 이 준 w0 에서
        시작 (llm_calls 에서 코드로 찾는다; 없으면 엘리트 w 에서 — 표시)
선택    실행마다 val regret 최소인 엘리트
보고    그 엘리트의 holdout — train 전체로 적합한 값(루프가 기록한 val_regret)
        과, train - val 로 적합한 w 그대로의 값 둘 다
```

⚠️ 둘 다 **최종 Archive 안에서 고르는 효과만** 잰다. 루프 안에서는 선택
기준이 다음 라운드의 부모까지 바꾸는데, 그건 오프라인으로 잴 수 없다.

## fit — 표준화 좌표 · 부모 가중치 상속 · 차원 비례 예산 (개선 6번)

c2 a6000 16실행의 exploit 자식(부모 = 그 라운드 시작의 최선, bests.jsonl)을
실행마다 2개(라운드 3 · 8) 골라 같은 자식을 세 방식으로 적합한다:

```
old   space=w · LLM 의 w0 · 예산 300 고정 · 2좌표 polish 전부   (c2 와 같음)
mid   space=u · 부모 적합값 상속 + 새 항 크기 · 예산 300 고정
new   space=u · 상속 · 예산 가중치당 37.5 · 2좌표 polish 는 8개 이하만
```

보고: 자식 train regret 과 부모 train regret 의 차(부모보다 0.01 넘게 나쁨 /
동률 / 나음), train gm, holdout gm.

## prune_cv — 개선 버전(d190)의 규칙으로 7번을 다시 잰다 (2026-10-03)

`prune` 은 c2 규칙으로 쟀다. 그런데 d190 은 피처(시간 축 · 래스터 축) · 적합 방식
· 최선 교체 규칙이 다르고, 죽은 가중치의 비율부터 다르다 (seed 0: c2 65% · d190 42%).
c2 에서 나빴다는 것이 d190 에서도 나쁘다는 보장은 없다. 그리고 `prune` 은 holdout
으로 판정했다 — 캠페인이 보고하는 바로 그 holdout 이다.

```
대상    d190 16실행 각각이 보고한 최종 최선 규칙 (bests.jsonl 의 마지막)
정리    prune 과 같다 — 죽은 가중치를 하나씩 지우고 train 이 안 나빠지면 둔다
비교    원래 구조 vs 정리한 구조, 각각 train 4조각 inner-CV
        (3조각으로 다시 적합 -> 1조각 채점, a6000_probe._inner_parts)
        적합은 그 캠페인의 루프와 같은 방식 (config.json 의 fit_space · fit_budget)
        시작점은 각 구조의 train 적합값 — 두 쪽에 똑같이
holdout 적기만 한다 (재적합 없이, prune 과 같은 방식). 판정에 쓰지 않는다
```

    python3 -m experiments.d190_offline prune_cv --prefix runs/d190-a6000 \
        --out docs/artifacts/d190/prune_cv.json

## 미리 정한 판정 (측정 전에 적는다)

```
prune   정리 후 holdout gm 이 정리 전보다 0.001 넘게 나빠지지 않으면 적용
select  val 로 고른 holdout gm 이 train 으로 고른 것보다 낮고,
        나아진 실행 수 > 나빠진 실행 수 이면 적용
fit     new 가 old 보다 "부모보다 0.01 넘게 나쁨" 비율이 낮고, train gm 이
        나쁘지 않으면 유지 (아니면 LoopConfig 기본값을 old 로 되돌린다)
        new 와 mid 의 train gm 차가 0.002 안이면 예산은 flat
prune_cv  16실행의 최종 최선 규칙에서, 정리한 구조의 inner-CV gm (모든 train
          형상을 모은 기하평균) 이 원래 구조보다 0.001 넘게 나빠지지 않으면
          "넣어도 된다". holdout 은 판정에 쓰지 않는다
```

⚠️ 2026-10-02 정정 (fit 결과를 보기 **전에**): 처음 적은 fit 판정에는
"holdout gm 이 0.005 넘게 나쁘지 않으면" 과 "holdout 차도 0.002 안" 이
있었다. 변경 검토가 이것을 짚었다 — 여기서 쓰는 holdout 은 캠페인이 보고할
**같은 a6000 nkband fold 의 holdout** 이라, 그것으로 조건을 고르면 그 숫자는
더는 깨끗한 holdout 이 아니다 (§10.2). 그래서 fit 판정은 train 쪽만 본다.
holdout 은 계산해 적되 판정에 쓰지 않는다.
⚠️ prune · select 의 판정과 `FIT_NOISE_TOL` 의 근거(c2 최선 교체의 holdout
변화)는 이미 같은 fold 의 holdout 을 읽었다 — D-190 기록과 캠페인 결과에
그 사실을 적는다.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

FOLDS = range(4)
SEEDS = range(4)
PREFIX = "runs/c2-a6000"
N_PROC = 4          # ★ cgroup cpu.max = 400000/100000 — CPU 4개

_G: dict = {}


def _gm(v) -> float:
    v = np.asarray(v, float)
    return float(np.exp(np.mean(np.log(v))))


def select_split(train_shapes, j: int, k: int = 4):
    """fold 의 train 을 (N,K) 그룹 단위로 k 개로 나눈 것 중 j 번째."""
    from kernelrule.core.splits import nk_band_folds

    s = nk_band_folds(list(train_shapes), k=k, name="sel")[j]
    return list(s.train.shapes), list(s.val.shapes)


def _load() -> None:
    import kernelrule.features.physical  # noqa: F401
    from experiments.f1_pipeline import _load_stage1, _splits
    from experiments.transfer_29_5 import TABLES
    from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY
    from kernelrule.features.loader import base_registry, load_generated

    T = TABLES["a6000"]
    tab = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                ok_only=False)
    _G["t"], _G["m"], _G["sp"], _G["arc"] = tab, {}, {}, {}
    for f in FOLDS:
        _G["sp"][f] = _splits(tab, fold=f, k=4, design="nkband")
        for s in SEEDS:
            reg = _load_stage1(Path(f"{PREFIX}-f{f}"),
                               base_registry("F2", human=REGISTRY), "F2", tab)
            fp = Path(f"{PREFIX}-f{f}-s{s}/features.jsonl")
            if fp.exists():
                for x in load_generated(fp, table=tab):
                    if x.name not in reg._items:
                        reg.add(x)
            _G["m"][(f, s)] = FeatureMatrix(tab, reg, cache_dir=CACHE_DIR)
            arc = Path(f"{PREFIX}-f{f}-s{s}/archive.jsonl")
            _G["arc"][(f, s)] = [json.loads(line) for line in
                                 arc.read_text().splitlines() if line]


def _prob(run, shapes):
    from kernelrule.core.weights import _Problem

    return _Problem(_G["m"][run], _G["t"], list(shapes), 1)


# -- prune -----------------------------------------------------------------
def _prune_task(t):
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import _contributions, _sensitivity
    from kernelrule.rules.prune import prune_dead, removable_indices

    run, i = t
    e = _G["arc"][run][i]
    sp = _G["sp"][run[0]]
    tr, ho = _prob(run, sp.train.shapes), _prob(run, sp.val.shapes)
    fn = compile_rule(e["code"])
    w = np.asarray(e["w"], float)
    base = tr.regret(fn, w)
    sens = _sensitivity(tr, fn, w, base, 0.5)
    dead = [k for k in range(len(w)) if abs(w[k]) < 1e-3 or sens[k] < 1e-6]
    con = _contributions(tr, fn, w)
    order = sorted(dead, key=lambda k: (float(con[k]) if con is not None
                                        else 0.0, k))

    def regret_of(code, ww):
        try:
            return tr.regret(compile_rule(code), np.asarray(ww, float))
        except Exception:                                   # noqa: BLE001
            return float("inf")

    code2, w2, log = prune_dead(e["code"], list(w), order, regret_of, base)
    fn2 = compile_rule(code2)
    # ★ 대조: 지운 항 = w[i] = 0 이어야 한다. 같은 regret 이 나오는지 본다.
    gone = sorted({x["i"] for x in log if x.get("kept")}
                  | {a for x in log if x.get("kept") for a in x["also"]})
    wz = w.copy()
    wz[gone] = 0.0
    return {"run": f"{run[0]}-{run[1]}", "rule": e["rule_id"],
            "n_w": len(w), "n_dead": len(dead),
            "n_removable": len(removable_indices(e["code"])),
            "n_kept": sum(1 for x in log if x.get("kept")),
            "n_w_after": len(w2),
            "why": {k: sum(1 for x in log if x.get("why") == k)
                    for k in ("not_removable", "worse", "already_gone")},
            "train_rec": e["regret"], "train": base,
            "train_after": tr.regret(fn2, np.asarray(w2, float)),
            "train_zeroed": tr.regret(fn, wz),
            "holdout_rec": e["val_regret"], "holdout": ho.regret(fn, w),
            "holdout_after": ho.regret(fn2, np.asarray(w2, float)),
            "holdout_zeroed": ho.regret(fn, wz)}


def _report_prune(rows) -> dict:
    by_run: dict = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    best = [min(v, key=lambda r: r["train"]) for v in by_run.values()]
    d_all = [np.log(r["holdout_after"]) - np.log(r["holdout"]) for r in rows]
    out = {
        "n_elites": len(rows),
        "weights_before": int(sum(r["n_w"] for r in rows)),
        "weights_dead": int(sum(r["n_dead"] for r in rows)),
        "weights_removed": int(sum(r["n_w"] - r["n_w_after"] for r in rows)),
        "zeroed_mismatch": sum(1 for r in rows
                               if abs(r["holdout_after"] - r["holdout_zeroed"])
                               > 1e-9),
        "all": {"holdout_gm_before": _gm([r["holdout"] for r in rows]),
                "holdout_gm_after": _gm([r["holdout_after"] for r in rows]),
                "n_better": sum(1 for x in d_all if x < -1e-9),
                "n_worse": sum(1 for x in d_all if x > 1e-9)},
        "train_best": {
            "holdout_gm_before": _gm([r["holdout"] for r in best]),
            "holdout_gm_after": _gm([r["holdout_after"] for r in best]),
            "n_w_before": [r["n_w"] for r in best],
            "n_w_after": [r["n_w_after"] for r in best]},
    }
    out["verdict_apply"] = bool(out["all"]["holdout_gm_after"]
                                <= out["all"]["holdout_gm_before"] + 0.001
                                and out["train_best"]["holdout_gm_after"]
                                <= out["train_best"]["holdout_gm_before"]
                                + 0.001)
    return out


# -- select ----------------------------------------------------------------
def _w0_index() -> dict:
    """코드 -> LLM 이 준 w0. RuleEditor 와 RuleWriter 응답에서."""
    out: dict = {}
    dirs = [Path(f"{PREFIX}-f{f}-s{s}/llm_calls") for f in FOLDS for s in SEEDS]
    dirs += [Path(f"{PREFIX}-f{f}/stage2-rule-writer/llm_calls") for f in FOLDS]
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                r = json.loads(p.read_text()).get("response") or {}
            except Exception:                               # noqa: BLE001
                continue
            if isinstance(r, dict) and r.get("code") and r.get("w0"):
                out.setdefault(r["code"].strip(), list(r["w0"]))
    return out


def _select_task(t):
    import warnings as _w

    _w.simplefilter("ignore")
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.splits import Split
    from kernelrule.core.weights import fit_weights
    from kernelrule.rules.checks import fitter_for, weight_bounds

    run, i = t
    e = _G["arc"][run][i]
    sp = _G["sp"][run[0]]
    fit_on, sel = select_split(sp.train.shapes, run[1])
    w0 = _G["w0"].get(e["code"].strip())
    src = "llm_w0"
    if w0 is None or len(w0) != len(e["w"]):
        w0, src = list(e["w"]), "elite_w"
    fn = compile_rule(e["code"])
    ft = fitter_for(len(w0))
    try:
        fr = fit_weights(fn, _G["m"][run], _G["t"], Split("train", tuple(fit_on)),
                         np.asarray(w0, float), objective="regret",
                         warn_invariants=False, method=ft["fit_method"],
                         n_restarts=ft["fit_restarts"],
                         max_evals=ft["max_evals"],
                         bounds=weight_bounds(e["code"], len(w0)))
    except Exception as ex:                                 # noqa: BLE001
        return {"run": f"{run[0]}-{run[1]}", "rule": e["rule_id"],
                "err": f"{type(ex).__name__}: {ex}"[:200]}
    return {"run": f"{run[0]}-{run[1]}", "rule": e["rule_id"],
            "n_w": len(e["w"]), "w0_from": src,
            "n_fit": len(fit_on), "n_sel": len(sel),
            "train_rec": e["regret"],
            "fit_sub": fr.fit_regret,
            "sel": _prob(run, sel).regret(fn, fr.w),
            "holdout_full_train_w": e["val_regret"],
            "holdout_sub_w": _prob(run, sp.val.shapes).regret(fn, fr.w)}


def _report_select(rows) -> dict:
    by_run: dict = {}
    for r in rows:
        if "err" not in r:
            by_run.setdefault(r["run"], []).append(r)
    pick_tr, pick_sel, pick_sel_sub, pick_tie, oracle = [], [], [], [], []
    per_run = {}
    for run, v in sorted(by_run.items()):
        a = min(v, key=lambda r: r["train_rec"])
        b = min(v, key=lambda r: r["sel"])
        # 잡음(0.01) 안이면 가중치가 적은 쪽
        lo = b["sel"]
        c = min((r for r in v if np.log(r["sel"]) - np.log(lo) < 0.01),
                key=lambda r: (r["n_w"], r["sel"]))
        pick_tr.append(a["holdout_full_train_w"])
        pick_sel.append(b["holdout_full_train_w"])
        pick_sel_sub.append(b["holdout_sub_w"])
        pick_tie.append(c["holdout_full_train_w"])
        oracle.append(min(r["holdout_full_train_w"] for r in v))
        per_run[run] = {"train_pick": a["rule"], "sel_pick": b["rule"],
                        "tie_pick": c["rule"],
                        "holdout_train_pick": a["holdout_full_train_w"],
                        "holdout_sel_pick": b["holdout_full_train_w"],
                        "holdout_sel_pick_sub_w": b["holdout_sub_w"],
                        "n_sel": b["n_sel"]}
    d = np.log(pick_sel) - np.log(pick_tr)
    allr = [r for v in by_run.values() for r in v]
    from scipy.stats import spearmanr

    out = {"n_runs": len(by_run), "n_rules": len(allr),
           "n_err": sum(1 for r in rows if "err" in r),
           "w0_from": {k: sum(1 for r in allr if r["w0_from"] == k)
                       for k in ("llm_w0", "elite_w")},
           "holdout_gm": {"train_pick": _gm(pick_tr),
                          "sel_pick_refit_full_train": _gm(pick_sel),
                          "sel_pick_keep_sub_w": _gm(pick_sel_sub),
                          "sel_pick_tie_fewer_w": _gm(pick_tie),
                          "oracle_in_archive": _gm(oracle)},
           "n_runs_better": int((d < -1e-9).sum()),
           "n_runs_worse": int((d > 1e-9).sum()),
           "n_runs_same_pick": sum(1 for v in per_run.values()
                                   if v["train_pick"] == v["sel_pick"]),
           "spearman_vs_holdout": {
               "train": float(spearmanr([r["train_rec"] for r in allr],
                                        [r["holdout_full_train_w"]
                                         for r in allr])[0]),
               "sel": float(spearmanr([r["sel"] for r in allr],
                                      [r["holdout_full_train_w"]
                                       for r in allr])[0])},
           "per_run": per_run}
    out["verdict_apply"] = bool(
        out["holdout_gm"]["sel_pick_refit_full_train"]
        < out["holdout_gm"]["train_pick"]
        and out["n_runs_better"] > out["n_runs_worse"])
    return out


# -- fit -------------------------------------------------------------------
FIT_ROUNDS = (3, 8)


def _fit_children() -> list:
    """(run, round, child code, LLM w0, parent code, parent w, parent
    train regret) — the first exploit child of each chosen round."""
    out = []
    for run in sorted(_G["arc"]):
        d = Path(f"{PREFIX}-f{run[0]}-s{run[1]}")
        bests = [json.loads(x) for x in
                 (d / "bests.jsonl").read_text().splitlines() if x]
        ev = [json.loads(x) for x in
              (d / "trace.jsonl").read_text().splitlines() if x]
        for r in FIT_ROUNDS:
            cand = [e for e in ev if e.get("ev") == "proposal"
                    and e.get("kind") == "exploit" and e["round"] == r]
            for e in cand:
                w0 = _G["w0"].get(e["code"].strip())
                if w0 is None or len(w0) != e["n_weights"]:
                    continue
                par = bests[r - 1]
                out.append((run, r, e["code"], list(w0), par["code"],
                            list(par["w"]), float(par["regret"])))
                break
    return out


def _fit_task(t):
    import warnings as _w

    _w.simplefilter("ignore")
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import fit_weights
    from kernelrule.rules.checks import (
        FITTER_SWITCH_DIM,
        fitter_for,
        weight_bounds,
    )
    from kernelrule.rules.inherit import inherit_weights

    (run, r, code, w0, pcode, pw, preg), arm = t
    sp = _G["sp"][run[0]]
    fn = compile_rule(code)
    if arm == "old":
        start, new, space, budget = w0, None, "w", "flat"
    else:
        start, new, _src = inherit_weights(code, w0, [(pcode, pw)])
        space, budget = "u", ("dim" if arm == "new" else "flat")
    ft = fitter_for(len(w0), budget=budget)
    t0 = time.time()
    try:
        fr = fit_weights(
            fn, _G["m"][run], _G["t"], sp.train, np.asarray(start, float),
            objective="regret", warn_invariants=False,
            method=ft["fit_method"], n_restarts=ft["fit_restarts"],
            max_evals=ft["max_evals"], val_split=sp.val,
            bounds=weight_bounds(code, len(w0)), space=space,
            new_terms=new if space == "u" else None,
            polish_pairs_max_dim=(FITTER_SWITCH_DIM if budget == "dim"
                                  else None))
    except Exception as ex:                                 # noqa: BLE001
        return {"run": f"{run[0]}-{run[1]}", "round": r, "arm": arm,
                "err": f"{type(ex).__name__}: {ex}"[:200]}
    return {"run": f"{run[0]}-{run[1]}", "round": r, "arm": arm,
            "n_w": len(w0), "n_new": int(sum(new)) if new else None,
            "parent_train": preg, "train": fr.fit_regret,
            "holdout": fr.val_regret, "evals": int(fr.n_evals),
            "seconds": round(time.time() - t0, 1)}


def _report_fit(rows) -> dict:
    out = {}
    for arm in ("old", "mid", "new"):
        v = [r for r in rows if r["arm"] == arm and "err" not in r]
        d = [r["train"] - r["parent_train"] for r in v]
        out[arm] = {
            "n": len(v), "n_err": sum(1 for r in rows
                                      if r["arm"] == arm and "err" in r),
            "worse_than_parent": sum(1 for x in d if x >= 0.01),
            "tie": sum(1 for x in d if abs(x) < 0.01),
            "better_than_parent": sum(1 for x in d if x <= -0.01),
            "train_gm": _gm([r["train"] for r in v]) if v else None,
            "holdout_gm": _gm([r["holdout"] for r in v]) if v else None,
            "seconds_median": float(np.median([r["seconds"] for r in v]))
            if v else None}
    o, n, m = out["old"], out["new"], out["mid"]
    # ★ train 쪽만 본다 — holdout 은 서술용 (위 docstring 의 정정)
    if o["n"] and n["n"]:
        out["verdict_keep_new"] = bool(
            n["worse_than_parent"] / n["n"] < o["worse_than_parent"] / o["n"]
            and n["train_gm"] <= o["train_gm"])
    if n["n"] and m["n"]:
        out["verdict_budget_flat"] = bool(
            abs(n["train_gm"] - m["train_gm"]) < 0.002)
    return out


# -- prune_cv ----------------------------------------------------------------
def _fit_setting() -> tuple[str, str]:
    """그 캠페인의 루프가 쓴 적합 (config.json). 없으면 옛 방식."""
    c = json.loads(Path(f"{PREFIX}-f0-s0/config.json").read_text())
    lp = c.get("loop", {})
    return lp.get("fit_space", "w"), lp.get("fit_budget", "flat")


def _best_of(run) -> dict:
    p = Path(f"{PREFIX}-f{run[0]}-s{run[1]}/bests.jsonl")
    return [json.loads(x) for x in p.read_text().splitlines() if x][-1]


def _prune_best_task(run):
    """1단계 — 최종 최선 규칙을 정리한다 (prune 과 같은 절차)."""
    import warnings as _w

    _w.simplefilter("ignore")
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import _contributions, _sensitivity
    from kernelrule.rules.prune import prune_dead

    best = _best_of(run)
    sp = _G["sp"][run[0]]
    tr, ho = _prob(run, sp.train.shapes), _prob(run, sp.val.shapes)
    fn = compile_rule(best["code"])
    w = np.asarray(best["w"], float)
    base = tr.regret(fn, w)
    sens = _sensitivity(tr, fn, w, base, 0.5)
    dead = [k for k in range(len(w)) if abs(w[k]) < 1e-3 or sens[k] < 1e-6]
    con = _contributions(tr, fn, w)
    order = sorted(dead, key=lambda k: (float(con[k]) if con is not None
                                        else 0.0, k))

    def regret_of(code, ww):
        try:
            return tr.regret(compile_rule(code), np.asarray(ww, float))
        except Exception:                                   # noqa: BLE001
            return float("inf")

    code2, w2, _log = prune_dead(best["code"], list(w), order, regret_of,
                                 base)
    fn2 = compile_rule(code2)
    return {"run": f"{run[0]}-{run[1]}", "rule": best["rule_id"],
            "code": best["code"], "w": [float(x) for x in w],
            "code_pruned": code2, "w_pruned": [float(x) for x in w2],
            "n_w": len(w), "n_dead": len(dead), "n_w_after": len(w2),
            "train": base,
            "train_after": tr.regret(fn2, np.asarray(w2, float)),
            "holdout": ho.regret(fn, w),
            "holdout_after": ho.regret(fn2, np.asarray(w2, float))}


def _cv_task(t):
    """2단계 — 한 구조를 train 3조각으로 다시 적합하고 남은 1조각을 채점."""
    import warnings as _w

    _w.simplefilter("ignore")
    from experiments.a6000_probe import _inner_parts
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.scoring import evaluate_scores
    from kernelrule.core.splits import Split
    from kernelrule.core.weights import fit_weights, make_score_of
    from kernelrule.rules.checks import (
        FITTER_SWITCH_DIM,
        fitter_for,
        weight_bounds,
    )

    run, arm, part, code, w = t
    space, budget = _G["fit"]
    parts = _inner_parts(list(_G["sp"][run[0]].train.shapes))
    fit_on = [p for i, q in enumerate(parts) if i != part for p in q]
    fn = compile_rule(code)
    ft = fitter_for(len(w), budget=budget)
    key = {"run": f"{run[0]}-{run[1]}", "arm": arm, "part": part}
    try:
        fr = fit_weights(
            fn, _G["m"][run], _G["t"], Split("train", tuple(fit_on)),
            np.asarray(w, float), objective="regret", warn_invariants=False,
            method=ft["fit_method"], n_restarts=ft["fit_restarts"],
            max_evals=ft["max_evals"], bounds=weight_bounds(code, len(w)),
            space=space,
            polish_pairs_max_dim=(FITTER_SWITCH_DIM if budget == "dim"
                                  else None))
    except Exception as ex:                                 # noqa: BLE001
        return {**key, "err": f"{type(ex).__name__}: {ex}"[:200]}
    ev = evaluate_scores(make_score_of(fn, _G["m"][run], fr.w), _G["t"],
                         parts[part], ks=(1,))
    return {**key, "regrets": [float(x) for x in ev.regret[:, 0]]}


def _report_prune_cv(pr_rows: list, cv_rows: list) -> dict:
    got: dict = {}
    bad = set()
    for r in cv_rows:
        if "err" in r:
            bad.add(r["run"])
            continue
        got.setdefault((r["run"], r["arm"]), {})[r["part"]] = r["regrets"]
    per_run, a_all, b_all = {}, [], []
    for pr in pr_rows:
        k = pr["run"]
        oa = got.get((k, "orig"), {})
        ob = got.get((k, "pruned"), oa if pr["n_w_after"] == pr["n_w"]
                     else {})
        if k in bad or len(oa) != 4 or len(ob) != 4:
            per_run[k] = {"skipped": True}
            continue
        a = [x for i in range(4) for x in oa[i]]
        b = [x for i in range(4) for x in ob[i]]
        a_all += a
        b_all += b
        per_run[k] = {"n_w": pr["n_w"], "n_w_after": pr["n_w_after"],
                      "inner_orig": _gm(a), "inner_pruned": _gm(b),
                      "holdout_orig": pr["holdout"],
                      "holdout_pruned": pr["holdout_after"]}
    ok = [v for v in per_run.values() if not v.get("skipped")]
    d = [np.log(v["inner_pruned"]) - np.log(v["inner_orig"]) for v in ok]
    out = {"fit": list(_G["fit"]), "n_runs": len(ok),
           "n_skipped": len(per_run) - len(ok),
           "weights_before": int(sum(v["n_w"] for v in ok)),
           "weights_after": int(sum(v["n_w_after"] for v in ok)),
           "inner_cv_gm_orig": _gm(a_all) if a_all else None,
           "inner_cv_gm_pruned": _gm(b_all) if b_all else None,
           "n_runs_inner_better": int(sum(1 for x in d if x < -1e-9)),
           "n_runs_inner_worse": int(sum(1 for x in d if x > 1e-9)),
           "holdout_gm_orig_not_used": _gm([v["holdout_orig"] for v in ok])
           if ok else None,
           "holdout_gm_pruned_not_used": _gm([v["holdout_pruned"]
                                              for v in ok]) if ok else None,
           "per_run": per_run}
    if a_all:
        out["verdict_apply"] = bool(out["inner_cv_gm_pruned"]
                                    <= out["inner_cv_gm_orig"] + 0.001)
    return out


def _main_prune_cv(a, t0) -> None:
    _G["fit"] = _fit_setting()
    runs = sorted(_G["arc"])
    with mp.get_context("fork").Pool(N_PROC) as pool:
        pr_rows = pool.map(_prune_best_task, runs)
    print(f"  정리 끝 {time.time() - t0:.0f}s — 가중치 "
          f"{sum(r['n_w'] for r in pr_rows)} -> "
          f"{sum(r['n_w_after'] for r in pr_rows)}  (적합 {_G['fit']})",
          flush=True)
    tasks = []
    for run, pr in zip(runs, pr_rows, strict=True):
        tasks += [(run, "orig", k, pr["code"], pr["w"]) for k in range(4)]
        if pr["n_w_after"] < pr["n_w"]:   # 아무것도 안 지웠으면 같은 구조
            tasks += [(run, "pruned", k, pr["code_pruned"], pr["w_pruned"])
                      for k in range(4)]
    cv_rows = []
    with mp.get_context("fork").Pool(N_PROC) as pool:
        for k, r in enumerate(pool.imap_unordered(_cv_task, tasks,
                                                  chunksize=1), 1):
            cv_rows.append(r)
            if k % 8 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)}  {time.time() - t0:.0f}s",
                      flush=True)
    rep = _report_prune_cv(pr_rows, cv_rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"mode": a.mode, "source": PREFIX,
                                 "report": rep, "rows": pr_rows,
                                 "cv_rows": cv_rows},
                                ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_run"},
                     ensure_ascii=False, indent=1, default=float))
    print(f"끝 {time.time() - t0:.0f}s -> {a.out}")


def main() -> None:
    global PREFIX  # noqa: PLW0603 — 측정 대상 캠페인을 바꾸는 한 곳
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prune", "select", "fit", "prune_cv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default=PREFIX,
                    help="runs/<campaign>-a6000 (기본 c2)")
    a = ap.parse_args()
    PREFIX = a.prefix
    warnings.simplefilter("ignore")
    t0 = time.time()
    _load()
    if a.mode == "prune_cv":
        print(f"적재 {time.time() - t0:.0f}s", flush=True)
        _main_prune_cv(a, t0)
        return
    if a.mode in ("select", "fit"):
        _G["w0"] = _w0_index()
    if a.mode == "select":
        for f in FOLDS:
            for j in SEEDS:
                fit_on, sel = select_split(_G["sp"][f].train.shapes, j)
                print(f"  fold {f} 안쪽 {j}: 적합 {len(fit_on)} · val {len(sel)}")
    print(f"적재 {time.time() - t0:.0f}s", flush=True)
    if a.mode == "fit":
        kids = _fit_children()
        print(f"  exploit 자식 {len(kids)}개 x 3", flush=True)
        tasks = [(k, arm) for k in kids for arm in ("old", "mid", "new")]
    else:
        tasks = [(run, i) for run in _G["arc"]
                 for i in range(len(_G["arc"][run]))]
    fn = {"prune": _prune_task, "select": _select_task,
          "fit": _fit_task}[a.mode]
    rows = []
    with mp.get_context("fork").Pool(N_PROC) as pool:
        for k, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=1), 1):
            rows.append(r)
            if k % 8 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)}  {time.time() - t0:.0f}s",
                      flush=True)
    rep = {"prune": _report_prune, "select": _report_select,
           "fit": _report_fit}[a.mode](rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"mode": a.mode, "source": PREFIX,
                                 "report": rep, "rows": rows},
                                ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_run"},
                     ensure_ascii=False, indent=1, default=float))
    print(f"끝 {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
