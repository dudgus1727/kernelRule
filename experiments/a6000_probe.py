"""★ a6000 규칙 손편집 — 한 규칙을 ★ 같은 절차로 채점한다 (D-188). **LLM 0회 · GPU 0.**

    python3 -m experiments.a6000_probe --rule R.json --out OUT.json
    python3 -m experiments.a6000_probe --make-baselines DIR     # 대조군 규칙 파일

## 왜 필요한가

"a6000 에서만 진다" 의 원인을 찾아 규칙을 고치고 다시 재는 일을 **열 번 넘게**
한다. 반복마다 절차가 조금씩 다르면 숫자끼리 비교가 안 된다. 그래서 채점을
이 파일 하나로 못 박는다.

## 절차 — 모든 반복이 똑같이

```
표        a6000 (rtx-a6000-sm_86-c63710df) · 65형상 · nkband k=4 · fold 0~3
어휘      k7-1 라이브러리 27축 (네 fold · 네 GPU 가 ★ 한 벌로 같다)
          + 규칙 파일이 부르는 loop 축(`extra_features`) — 코드라 어느 표에서나 계산된다
적합      학습 분할에서만 · objective=regret · CMA · 재시작 2 · 평가 2000 (고정)
★ 고르는 값  inner-CV — 학습 48형상을 4조각으로 나눠 3조각에 맞추고 1조각을 잰다
           (홀드아웃을 ⛔ 안 본다. 반복마다 무엇을 남길지는 이 값으로 정한다)
★ 보고 값    fold 홀드아웃 — 네 fold 의 홀드아웃이 65형상을 ★ 정확히 한 번씩 덮는다
           -> 65형상 합동 기하평균 + fold 별
참고       벤더 · static top-1 · c2 원주민(루프가 끝낸 w 그대로) 을 같은 형상에서
```

⚠️ **홀드아웃을 반복마다 읽는다** — 사람이 그 숫자를 보고 다음 가설을 고르므로
뒤로 갈수록 낙관적이 된다. 그래서 반복 기록에 inner-CV 를 같이 싣고, 최종
판정은 두 값이 같은 쪽을 가리킬 때만 한다.

⛔ `kernelrule/` 은 건드리지 않는다. 이것은 측정 도구다.
"""

from __future__ import annotations

import os

# ★ 병렬 적합이 BLAS 스레드와 싸우지 않게 — numpy 를 부르기 전에
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

GPU = "a6000"
FOLDS = (0, 1, 2, 3)
DESIGN = "nkband"
FIT = {"method": "cma", "n_restarts": 2, "max_evals": 2000}
N_INNER = 4

_G: dict = {}          # fork 로 자식에게 넘기는 것 (표 · 행렬 · 분할)


def _gm(v) -> float:
    v = np.asarray(v, float)
    return float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")


def _table():
    from experiments.transfer_29_5 import TABLES
    from kernelrule.core.table import PerfTable

    T = TABLES[GPU]
    return PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                 ok_only=False)


def _registry(table, extra: list[str]):
    """k7-1 라이브러리 + 규칙이 부르는 loop 축."""
    from experiments.f1_pipeline import _load_stage1
    from kernelrule.features import REGISTRY
    from kernelrule.features.loader import base_registry, load_generated

    reg = _load_stage1(Path(f"runs/c2-{GPU}-f0"),
                       base_registry("F2", human=REGISTRY), "F2", table)
    for fp in extra or []:
        for f in load_generated(Path(fp), table=table):
            if f.name not in reg._items:
                reg.add(f)
    return reg


def _inner_parts(shapes) -> list[list]:
    """학습 형상을 N_INNER 조각으로 — 크기 순으로 줄 세워 돌려 담는다(결정적)."""
    s = sorted(shapes, key=lambda p: (p.M * p.N * p.K, p.M, p.N, p.K))
    return [s[i::N_INNER] for i in range(N_INNER)]


def _fit_task(task):
    """(fold, part) — part=None 이면 학습 전부에 맞추고 홀드아웃까지 읽는다."""
    import warnings as _w
    _w.simplefilter("ignore")
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.scoring import evaluate_scores
    from kernelrule.core.splits import Split
    from kernelrule.core.weights import fit_weights, make_score_of

    fold, part = task
    t, m = _G["table"], _G["matrix"]
    spec = _G["spec"][fold]
    sp = _G["splits"][fold]
    train = list(sp.train.shapes)
    if part is None:
        fit_on, read_on = train, list(sp.val.shapes)
    else:
        parts = _inner_parts(train)
        read_on = parts[part]
        fit_on = [p for i, q in enumerate(parts) if i != part for p in q]
    fn = compile_rule(spec["code"])
    fr = fit_weights(fn, m, t, Split("train", tuple(fit_on)),
                     np.asarray(spec["w0"], float), objective="regret",
                     warn_invariants=False, **FIT)
    so = make_score_of(fn, m, fr.w)
    ev = evaluate_scores(so, t, read_on, ks=(1,))
    reg = {f"{p.M}x{p.N}x{p.K}": float(r)
           for p, r in zip(ev.shapes, ev.regret[:, 0], strict=True)}
    out = {"fold": fold, "part": part, "regret": reg,
           "w": [float(x) for x in fr.w], "moved": bool(fr.moved)}
    if part is None:
        tr = evaluate_scores(so, t, train, ks=(1,))
        out["train_gm"] = _gm(tr.regret[:, 0])
        # ★ 무엇을 골랐나 — 홀드아웃 형상에서 split_k
        picks = {}
        for p in read_on:
            df = t.frame_for(p).reset_index(drop=True)
            tt = np.asarray(t.times_of(p))
            i = int(t.candidates(p).top_k(so(p, t.candidates(p)), 1)[0])
            j = int(np.argmin(tt))
            picks[f"{p.M}x{p.N}x{p.K}"] = {
                "sk_pick": int(df.iloc[i]["split_k"]),
                "sk_best": int(df.iloc[j]["split_k"]),
                "tile_pick": f"{df.iloc[i]['tile_m']}x{df.iloc[i]['tile_n']}"
                             f"x{df.iloc[i]['tile_k']}",
                "tile_best": f"{df.iloc[j]['tile_m']}x{df.iloc[j]['tile_n']}"
                             f"x{df.iloc[j]['tile_k']}"}
        out["picks"] = picks
    return out


def _refs(table, splits) -> dict:
    """벤더 · static top-1 · c2 원주민 — 같은 홀드아웃 형상에서."""
    from experiments.c2_ref import native
    from experiments.transfer_29_5 import TABLES
    from kernelrule.baselines.static_topk import StaticTopK
    from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
    from kernelrule.core.scoring import evaluate

    vend = load_vendor(f"datasets/baselines/vendor-{GPU}-"
                       f"{TABLES[GPU]['env_hash'][:8]}.json")
    vfn = vendor_order_fn(table, vend, mapping="nearest")
    out = {}
    for f in FOLDS:
        hold = list(splits[f].val.shapes)
        ev = evaluate(vfn, table, hold, ks=(1,), label="vendor")
        out[f] = {
            "vendor": {f"{p.M}x{p.N}x{p.K}": float(r)
                       for p, r in zip(ev.shapes, ev.regret[:, 0], strict=True)},
            "static_top1": float(StaticTopK(table, hold, coverage="union")
                                 .run(ks=(1,)).by_k[1]["all"]),
            "c2_native_median": native(GPU, f)}
    return out


def evaluate_rule(spec_by_fold: dict, workers: int = 20) -> dict:
    """★ 규칙 하나(fold 마다 같거나 다른 코드)를 이 파일의 절차로 채점."""
    from experiments.f1_pipeline import _splits
    from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
    from kernelrule.rules.checks import check_rule, limits_for

    warnings.simplefilter("ignore")
    t = _G.get("table") or _table()
    _G["table"] = t
    splits = _G.get("splits") or {f: _splits(t, fold=f, k=4, design=DESIGN)
                                  for f in FOLDS}
    _G["splits"] = splits
    extra = sorted({x for s in spec_by_fold.values()
                    for x in s.get("extra_features", [])})
    reg = _registry(t, extra)
    m = FeatureMatrix(t, reg, cache_dir=CACHE_DIR)
    _G["matrix"], _G["spec"] = m, spec_by_fold

    checks = {}
    for f, s in spec_by_fold.items():
        rep = check_rule(s["code"], feature_names=m.feature_names(),
                         shape_value_names=m.shape_value_names(),
                         n_weights=len(s["w0"]), limits=limits_for())
        checks[f] = {"ok": rep.ok, "violations": list(rep.violations),
                     "n_paths": rep.n_paths, "depth": rep.branch_depth,
                     "n_weights": len(s["w0"]),
                     "features": sorted(rep.features_used)}
    good = [f for f in FOLDS if checks[f]["ok"]]
    tasks = [(f, None) for f in good] + [(f, i) for f in good
                                         for i in range(N_INNER)]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(tasks) or 1)) as pool:
        res = pool.map(_fit_task, tasks)
    refs = _G.get("refs") or _refs(t, splits)
    _G["refs"] = refs

    per_fold = {}
    pooled_hold, pooled_inner, pooled_vend = [], [], []
    for f in FOLDS:
        if f not in good:
            per_fold[f] = {"check": checks[f]}
            continue
        full = next(r for r in res if r["fold"] == f and r["part"] is None)
        inner = [r for r in res if r["fold"] == f and r["part"] is not None]
        ir = [v for r in inner for v in r["regret"].values()]
        hv = list(full["regret"].values())
        vv = [refs[f]["vendor"][k] for k in full["regret"]]
        hard = [k for k, x in full["picks"].items() if x["sk_best"] >= 3]
        per_fold[f] = {
            "check": checks[f],
            "train_gm": round(full["train_gm"], 6),
            "inner_cv_gm": round(_gm(ir), 6),
            "holdout_gm": round(_gm(hv), 6),
            "vendor_gm": round(_gm(vv), 6),
            "static_top1": round(refs[f]["static_top1"], 6),
            "c2_native_median": round(refs[f]["c2_native_median"], 6),
            "beats_vendor": _gm(hv) < _gm(vv),
            "n_hold": len(hv),
            "hold_sk_ge3": {"n": len(hard),
                            "picked_ge3": sum(1 for k in hard
                                              if full["picks"][k]["sk_pick"] >= 3),
                            "gm": round(_gm([full["regret"][k] for k in hard]), 6)
                            if hard else None,
                            "vendor_gm": round(_gm([refs[f]["vendor"][k]
                                                    for k in hard]), 6)
                            if hard else None},
            "hold_sk_lt3_gm": round(_gm([full["regret"][k] for k in full["regret"]
                                         if k not in hard]), 6),
            "sk_picked_hold": dict(Counter(x["sk_pick"]
                                           for x in full["picks"].values())),
            "w": full["w"], "moved": full["moved"],
            "regret": full["regret"], "picks": full["picks"]}
        pooled_hold += hv
        pooled_inner += ir
        pooled_vend += vv
    ok = [f for f in FOLDS if f in good]
    agg = {
        "n_folds_ok": len(ok),
        "holdout_gm_65": round(_gm(pooled_hold), 6) if ok else None,
        "inner_cv_gm": round(_gm(pooled_inner), 6) if ok else None,
        "vendor_gm_65": round(_gm(pooled_vend), 6) if ok else None,
        "folds_beating_vendor": sum(1 for f in ok if per_fold[f]["beats_vendor"]),
        "holdout_by_fold": {f: per_fold[f]["holdout_gm"] for f in ok},
        "inner_by_fold": {f: per_fold[f]["inner_cv_gm"] for f in ok},
        "sk_ge3_picked_on_hard": (
            f"{sum(per_fold[f]['hold_sk_ge3']['picked_ge3'] for f in ok)}/"
            f"{sum(per_fold[f]['hold_sk_ge3']['n'] for f in ok)}"),
        "hard_gm": round(_gm([per_fold[f]["regret"][k] for f in ok
                              for k, x in per_fold[f]["picks"].items()
                              if x["sk_best"] >= 3]), 6) if ok else None,
        "easy_gm": round(_gm([per_fold[f]["regret"][k] for f in ok
                              for k, x in per_fold[f]["picks"].items()
                              if x["sk_best"] < 3]), 6) if ok else None,
    }
    return {"protocol": {"fit": FIT, "n_inner": N_INNER, "design": DESIGN,
                         "gpu": GPU, "library": "k7-1 (27축)",
                         "extra_features": extra},
            "aggregate": agg, "per_fold": per_fold}


def _load_spec(path: Path) -> dict:
    d = json.loads(path.read_text())
    if "per_fold" in d and isinstance(d["per_fold"], dict):
        return {int(k): v for k, v in d["per_fold"].items()}
    return {f: {"code": d["code"], "w0": d["w0"],
                "extra_features": d.get("extra_features", [])} for f in FOLDS}


def make_baselines(out_dir: Path) -> None:
    """★ 대조군 규칙 파일 — c2 a6000 의 진화된 규칙(fold 마다 그 fold 의 것).

    ⛔ 원주민 공식값(루프가 끝낸 w)과는 다른 자다: 여기서는 ★ 같은 절차로
    다시 맞춘다. 손편집한 규칙과 같은 자로 견주려면 이것이 대조군이다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in (0, 1, 2, 3):
        pf = {}
        for f in FOLDS:
            run = Path(f"runs/c2-{GPU}-f{f}-s{seed}")
            e = [json.loads(x) for x in (run / "bests.jsonl").read_text()
                 .splitlines() if x.strip()][-1]
            fp = run / "features.jsonl"
            pf[str(f)] = {"code": e["code"], "w0": e["w"],
                          "extra_features": [str(fp)] if fp.exists() else []}
        (out_dir / f"c2_evolved_s{seed}.json").write_text(json.dumps(
            {"name": f"c2 evolved s{seed}", "per_fold": pf},
            ensure_ascii=False, indent=1))
    print(f"  -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--make-baselines", type=Path, default=None)
    # ★ 5분짜리 채점 전에 규칙 검사만 — 떨어질 규칙에 적합을 돌리지 않는다
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.make_baselines:
        make_baselines(a.make_baselines)
        return
    spec = _load_spec(a.rule)
    if a.check_only:
        from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
        from kernelrule.rules.checks import check_rule, limits_for

        warnings.simplefilter("ignore")
        t = _table()
        extra = sorted({x for s in spec.values()
                        for x in s.get("extra_features", [])})
        m = FeatureMatrix(t, _registry(t, extra), cache_dir=CACHE_DIR)
        bad = 0
        for f, s in spec.items():
            rep = check_rule(s["code"], feature_names=m.feature_names(),
                             shape_value_names=m.shape_value_names(),
                             n_weights=len(s["w0"]), limits=limits_for())
            bad += not rep.ok
            print(f"  f{f} {'★ 통과' if rep.ok else '⛔ ' + '; '.join(rep.violations)}"
                  f"  경로 {rep.n_paths}  w {len(s['w0'])}")
        raise SystemExit(1 if bad else 0)
    r = evaluate_rule(spec, workers=a.workers)
    r["rule_file"] = str(a.rule)
    r["rule"] = json.loads(a.rule.read_text())
    a.out.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    g = r["aggregate"]
    print(f"★ {a.rule.name}  inner-CV {g['inner_cv_gm']}  ★ 홀드아웃(65) "
          f"{g['holdout_gm_65']}  벤더(65) {g['vendor_gm_65']}  벤더 넘은 fold "
          f"{g['folds_beating_vendor']}/{g['n_folds_ok']}  sk>=3 {g['sk_ge3_picked_on_hard']}"
          f"  어려운 {g['hard_gm']} 쉬운 {g['easy_gm']}")
    for f, v in r["per_fold"].items():
        if "holdout_gm" not in v:
            print(f"   f{f} ⛔ {v['check']['violations'][:1]}")
            continue
        print(f"   f{f}  학습 {v['train_gm']:.4f}  inner {v['inner_cv_gm']:.4f}  "
              f"홀드아웃 {v['holdout_gm']:.4f}  벤더 {v['vendor_gm']:.4f}  "
              f"static {v['static_top1']:.4f}  c2 {v['c2_native_median']:.4f}  "
              f"경로 {v['check']['n_paths']}  w {v['check']['n_weights']}")


if __name__ == "__main__":
    main()
