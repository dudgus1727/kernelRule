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

## 미리 정한 판정 (측정 전에 적는다)

```
prune   정리 후 holdout gm 이 정리 전보다 0.001 넘게 나빠지지 않으면 적용
select  val 로 고른 holdout gm 이 train 으로 고른 것보다 낮고,
        나아진 실행 수 > 나빠진 실행 수 이면 적용
```
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prune", "select"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    t0 = time.time()
    _load()
    if a.mode == "select":
        _G["w0"] = _w0_index()
        for f in FOLDS:
            for j in SEEDS:
                fit_on, sel = select_split(_G["sp"][f].train.shapes, j)
                print(f"  fold {f} 안쪽 {j}: 적합 {len(fit_on)} · val {len(sel)}")
    print(f"적재 {time.time() - t0:.0f}s", flush=True)
    tasks = [(run, i) for run in _G["arc"] for i in range(len(_G["arc"][run]))]
    fn = _prune_task if a.mode == "prune" else _select_task
    rows = []
    with mp.get_context("fork").Pool(N_PROC) as pool:
        for k, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=1), 1):
            rows.append(r)
            if k % 8 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)}  {time.time() - t0:.0f}s",
                      flush=True)
    rep = (_report_prune if a.mode == "prune" else _report_select)(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"mode": a.mode, "source": PREFIX,
                                 "report": rep, "rows": rows},
                                ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_run"},
                     ensure_ascii=False, indent=1, default=float))
    print(f"끝 {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
