"""★ D-190 캠페인의 집계 — c2 · splitk(D-187) 와 같은 (fold, seed) 로 짝짓는다.
**LLM 0회 · GPU 0.**

    python3 -m experiments.d190_campaign                  # 끝난 실행 전부
    python3 -m experiments.d190_campaign --arms c2 d190    # 갈래 고르기

⛔ **아무것도 새로 적합하지 않는다.** 실행이 남긴 `rounds.jsonl` ·
`bests.jsonl` · `failures.jsonl` · `trace.jsonl` 과 표를 읽어 센다.

## 무엇을 보나

```
★ holdout       루프가 적은 마지막 라운드의 `best_val_regret` (D-182 · D-186)
                실행별 · fold 별 중앙값 · seed 별 65형상 pooled
                (exp(Σ n_f·ln h_f / 65) — 벤더 1.081488 이 이 척도)
★ 벤더          fold 별 1.092486 / 1.089068 / 1.080987 / 1.062970
★ 픽            최종 최선 규칙의 top-1 이 65형상에서 고른 것:
                hungry (최적 split_k >= 3) 에서 sk>=3 을 고른 수,
                big (최적 config 의 output_work_per_sm > 2e5) 에서 고른
                raster, 점수가 완전 동점인 top-1 의 비율
★ 규칙          가중치 수 · cfg.raster_* 를 읽는 피처 · 시간 단위 피처를 쓰는가
★ 루프          실패 판정 분포 (made_worse / tie / better_not_kept),
                상속된 가중치 비율, 라운드 시간
```

⚠️ 픽은 65형상 전부에서 잰다 — holdout 포함. **서술일 뿐** 어떤 선택에도
쓰지 않는다 (§10.2 는 선택을 막는다).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import statistics as st  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

GPU = "a6000"
FOLDS = (0, 1, 2, 3)
SEEDS = (0, 1, 2, 3)
ARMS = ("c2", "splitk", "d190")
VENDOR = {0: 1.092486, 1: 1.089068, 2: 1.080987, 3: 1.062970}
VENDOR_POOLED = 1.081488
OUT = Path("docs/artifacts/d190-campaign.json")


def _rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _gm(v) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(v, float)))))


_MATRIX: dict = {}
_LIB: dict = {}


def _matrix_for(arm: str, fold: int, seed: int, table):
    import kernelrule.features.physical  # noqa: F401
    from experiments.f1_pipeline import _load_stage1
    from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
    from kernelrule.features import REGISTRY
    from kernelrule.features.loader import base_registry, load_generated

    d = Path(f"runs/{arm}-{GPU}-f{fold}")
    reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2", table)
    fp = Path(f"{d}-s{seed}") / "features.jsonl"
    if fp.exists():
        for f in load_generated(fp, table=table):
            if f.name not in reg._items:
                reg.add(f)
    _LIB[(arm, fold, seed)] = reg
    return FeatureMatrix(table, reg, cache_dir=CACHE_DIR)


def _feature_flags(code: str, reg) -> dict:
    """Which of the rule's features read `cfg.raster_*`, and which declare
    a time unit."""
    used = sorted(set(re.findall(r"\bf\.([a-z][a-z0-9_]*)", code)))
    raster, timed = [], []
    for n in used:
        if n not in reg._items:
            continue
        f = reg[n]
        src = str(getattr(f, "source", "") or "")
        if "raster_order" in src or "raster_width" in src:
            raster.append(n)
        if "time" in str(getattr(f, "unit", "")).lower():
            timed.append(n)
    return {"n_features": len(used), "raster": raster, "time": timed}


def _picks(code, w, table, matrix, shapes, cls) -> dict:
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import make_score_of

    so = make_score_of(compile_rule(code), matrix, np.asarray(w, float))
    hungry = big = big_raster = 0
    hungry_hit = 0
    raster_big: Counter = Counter()
    ties = 0
    for p in shapes:
        cand = table.candidates(p)
        df = table.frame_for(p).reset_index(drop=True)
        t = np.asarray(table.times_of(p))
        s = np.asarray(so(p, cand), float)
        i = int(cand.top_k(s, 1)[0])
        ties += int(np.sum(s == s[i]) > 1)
        c = cls[f"{p.M}x{p.N}x{p.K}"]
        if c == "hungry":
            hungry += 1
            hungry_hit += int(int(df.iloc[i]["split_k"]) >= 3)
        if c == "big":
            big += 1
            rr = (f"{df.iloc[i].get('ext_swizzle_type')}"
                  f"{df.iloc[i].get('ext_swizzle_n')}")
            raster_big[rr] += 1
            j = int(np.argmin(t))
            big_raster += int(
                (df.iloc[i].get("ext_swizzle_type"),
                 df.iloc[i].get("ext_swizzle_n"))
                == (df.iloc[j].get("ext_swizzle_type"),
                    df.iloc[j].get("ext_swizzle_n")))
    return {"hungry": hungry, "hungry_sk_ge3": hungry_hit,
            "big": big, "big_raster_as_optimal": big_raster,
            "big_raster_picked": dict(raster_big),
            "top1_tied_frac": round(ties / len(shapes), 4)}


def _run(arm: str, fold: int, seed: int, table, shapes, cls) -> dict:
    run = Path(f"runs/{arm}-{GPU}-f{fold}-s{seed}")
    rr = _rows(run / "rounds.jsonl")
    if not rr:
        return {"arm": arm, "fold": fold, "seed": seed, "missing": True}
    best = _rows(run / "bests.jsonl")[-1]
    fails = _rows(run / "failures.jsonl")
    ev = _rows(run / "trace.jsonl")
    props = [e for e in ev if e.get("ev") == "proposal"]
    inh = [e for e in props if e.get("n_inherited") is not None]
    m = _MATRIX.setdefault((arm, fold, seed),
                           _matrix_for(arm, fold, seed, table))
    return {
        "arm": arm, "fold": fold, "seed": seed, "run": run.name,
        "n_rounds": len(rr),
        "holdout": float(rr[-1]["best_val_regret"]),
        "train": float(rr[-1]["best_regret"]),
        "n_weights": len(best["w"]),
        "minutes": round(sum(x.get("seconds") or 0 for x in rr) / 60, 1),
        "failures": dict(Counter(f.get("verdict") for f in fails)),
        "inherited_frac": (round(sum(e["n_inherited"] for e in inh)
                                 / max(1, sum(e["n_weights"] for e in inh)),
                                 4) if inh else None),
        "rule": _feature_flags(best["code"], _LIB[(arm, fold, seed)]),
        "picks": _picks(best["code"], best["w"], table, m, shapes, cls)}


def _pooled(rows: list[dict], n_val: dict) -> float:
    tot = sum(n_val[r["fold"]] for r in rows)
    return float(np.exp(sum(n_val[r["fold"]] * np.log(r["holdout"])
                            for r in rows) / tot))


def summarize(rows: list[dict], n_val: dict) -> dict:
    out: dict = {"vendor_pooled": VENDOR_POOLED, "arms": {}}
    have = {(r["arm"], r["fold"], r["seed"]): r for r in rows
            if not r.get("missing")}
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm and not r.get("missing")]
        if not mine:
            continue
        per_seed = {}
        for s in SEEDS:
            rs = [r for r in mine if r["seed"] == s]
            if len(rs) == len(FOLDS):
                per_seed[s] = round(_pooled(rs, n_val), 6)
        per_fold = {f: round(st.median([r["holdout"] for r in mine
                                        if r["fold"] == f]), 6)
                    for f in FOLDS if any(r["fold"] == f for r in mine)}
        out["arms"][arm] = {
            "n_runs": len(mine),
            "holdout_gm_all_runs": round(_gm([r["holdout"] for r in mine]),
                                         6),
            "pooled65_per_seed": per_seed,
            "median_per_fold": per_fold,
            "folds_beating_vendor_median": sum(
                1 for f, v in per_fold.items() if v < VENDOR[f]),
            "hungry_sk_ge3": sum(r["picks"]["hungry_sk_ge3"] for r in mine),
            "hungry_total": sum(r["picks"]["hungry"] for r in mine),
            "big_raster_as_optimal": sum(r["picks"]["big_raster_as_optimal"]
                                         for r in mine),
            "big_total": sum(r["picks"]["big"] for r in mine),
            "top1_tied_frac_mean": round(float(np.mean(
                [r["picks"]["top1_tied_frac"] for r in mine])), 4),
            "rules_reading_raster": sum(1 for r in mine
                                        if r["rule"]["raster"]),
            "rules_using_time_axes": sum(1 for r in mine
                                         if r["rule"]["time"]),
            "n_weights_median": st.median([r["n_weights"] for r in mine]),
            "minutes_median": st.median([r["minutes"] for r in mine]),
        }
    pairs = {}
    for a, b in (("d190", "c2"), ("d190", "splitk")):
        d = [np.log(have[(a, f, s)]["holdout"])
             - np.log(have[(b, f, s)]["holdout"])
             for f in FOLDS for s in SEEDS
             if (a, f, s) in have and (b, f, s) in have]
        if d:
            pairs[f"{a}_vs_{b}"] = {
                "n": len(d), "log_diff_median": round(float(np.median(d)), 4),
                "n_better": int(sum(1 for x in d if x < 0)),
                "n_worse": int(sum(1 for x in d if x > 0))}
    out["pairs"] = pairs
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    from experiments import a6000_probe as P
    from experiments.f1_pipeline import _splits

    P._G["gpu"] = GPU
    table = P._table()
    sp = {f: _splits(table, fold=f, k=4, design="nkband") for f in FOLDS}
    shapes = [p for f in FOLDS for p in sp[f].val.shapes]
    n_val = {f: len(sp[f].val.shapes) for f in FOLDS}
    rows = []
    cls = None
    for arm in a.arms:
        for f in FOLDS:
            for s in SEEDS:
                if not Path(f"runs/{arm}-{GPU}-f{f}-s{s}/rounds.jsonl"
                            ).exists():
                    continue
                if cls is None:
                    m0 = _matrix_for(arm, f, s, table)
                    _MATRIX[(arm, f, s)] = m0
                    cls = P._classes(table, m0, shapes)
                rows.append(_run(arm, f, s, table, shapes, cls))
                r = rows[-1]
                print(f"  {arm:6s} f{f} s{s}  holdout {r['holdout']:.4f}  "
                      f"hungry {r['picks']['hungry_sk_ge3']}/"
                      f"{r['picks']['hungry']}  big-raster "
                      f"{r['picks']['big_raster_as_optimal']}/"
                      f"{r['picks']['big']}  w {r['n_weights']}", flush=True)
    s = summarize(rows, n_val)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": s, "rows": rows},
                                ensure_ascii=False, indent=1, default=str))
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
