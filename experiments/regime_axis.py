"""★ 체제 분할 축을 다시 고른다 — SOL 0.5 vs roofline vs 안 나눔. LLM 0회.

    python3 experiments/regime_axis.py

실험 계획서 `docs/artifacts/regime-axis-prereg.md`.

## 왜 다시 구현하나

`canonical_score` 는 체제 이름 `("short","long")` 과 `regime_of(axis="size")`
를 하드코딩한다. 축을 갈아 끼우려면 그 절차를 여기서 다시 만들어야 한다.
**그래서 팔 ① 이 알려진 대표값과 같은지 먼저 확인한다** — 다르면 멈춘다.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import warnings
from pathlib import Path

import numpy as np

from sigma_5090 import _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.numerics import approx_equal
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import _DUMMY_CFG, Split
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.features.physical import is_memory_bound, log_sol_ms

BUNDLE = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
RUNS = [f"F3rw-p8-nan-s{i}" for i in range(6)]
ROUND = 11
#: 팔 ① 이 재현해야 하는 값 (D-140). 다르면 재구현이 틀린 것이다.
KNOWN_ARM1 = 1.0886
#: 판정선. **여기서 새로 정하지 않는다** (원칙 7).
DELTA, BUFFER = 0.0516, 0.0589
#: 체제당 학습 형상이 이보다 적으면 그 가중치를 믿기 어렵다.
MIN_PER_REGIME = 8


def _rule(run: str) -> dict:
    rows = [json.loads(x) for x
            in (Path("runs") / run / "bests.jsonl").read_text().splitlines()
            if x.strip()]
    return max([e for e in rows if e["round"] <= ROUND],
               key=lambda e: e["round"])


def _sol(thr: float):
    def f(p, hw):
        return ("short" if log_sol_ms(p, hw, _DUMMY_CFG) < math.log2(thr)
                else "long")
    return f


def _roof(p, hw):
    return "mem" if is_memory_bound(p, hw, _DUMMY_CFG) else "comp"


def _one(p, hw):
    return "all"


ARMS = [
    ("① SOL 0.5",  _sol(0.5),  ("short", "long")),
    ("② roofline", _roof,      ("mem", "comp")),
    ("③ 안 나눔",   _one,       ("all",)),
    ("①' SOL 0.25", _sol(0.25), ("short", "long")),
    ("①'' SOL 1.0", _sol(1.0),  ("short", "long")),
]


def score(code, w0, T, M, sp, reg_fn, names, max_evals=300):
    """`canonical_score` 와 **같은 절차**, 체제 함수만 갈아 끼운다."""
    fn = compile_rule(code)
    train, val = list(sp.train.shapes), list(sp.val.shapes)
    per_shape, warns = {}, []
    for nm in names:
        g_tr = [p for p in train if reg_fn(p, T.hw) == nm]
        g_ho = [p for p in val if reg_fn(p, T.hw) == nm]
        if not g_tr:
            if g_ho:
                warns.append(f"{nm}: 학습 0개인데 홀드아웃 {len(g_ho)}개")
            continue
        if len(g_tr) < MIN_PER_REGIME:
            warns.append(f"{nm}: 학습 {len(g_tr)}개 < {MIN_PER_REGIME}")
        fit = fit_weights(fn, M, T, Split("train", tuple(g_tr)), w0,
                          max_evals=max_evals, objective="regret",
                          warn_invariants=False)
        if not g_ho:
            continue
        e = evaluate_scores(make_score_of(fn, M, fit.w), T, g_ho, ks=(1,))
        for i, p in enumerate(e.shapes):
            per_shape[p] = float(e.regret[i, 0])
    if len(per_shape) < len(val):
        warns.append(f"홀드아웃 {len(val)} 중 {len(per_shape)}개만 채점")
    return (geomean(np.array([per_shape[p] for p in val if p in per_shape])),
            per_shape, warns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/regime-axis.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(BUNDLE[0], env_hash=BUNDLE[1], ok_only=False)
    M, sp = FeatureMatrix(T, REGISTRY), _splits(T)
    rules = [(r, _rule(r)) for r in RUNS]

    out: dict = {"delta": DELTA, "buffer": BUFFER, "round": ROUND,
                 "runs": RUNS, "arms": {}}
    print("=" * 96)
    print("체제 분할 축 — 같은 규칙 여섯, 가중치만 다시 (LLM 0회)")
    print("=" * 96)
    print(f"  {'팔':14s} {'중앙':>8s} {'범위':>19s} {'체제별 홀드아웃 수':>18s}")

    shape_reg = {}
    for lab, fnr, names in ARMS:
        vals, per_run, allw = [], [], set()
        for run, r in rules:
            v, ps, w = score(r["code"], r["w"], T, M, sp, fnr, names)
            vals.append(v)
            per_run.append({str(k): x for k, x in ps.items()})
            allw.update(w)
        cnt = {nm: sum(1 for p in sp.val.shapes if fnr(p, T.hw) == nm)
               for nm in names}
        shape_reg[lab] = {str(p): fnr(p, T.hw) for p in sp.val.shapes}
        out["arms"][lab] = {"vals": vals, "median": st.median(vals),
                            "holdout_counts": cnt, "warnings": sorted(allw),
                            "per_run_shape": per_run}
        print(f"  {lab:14s} {st.median(vals):8.4f} "
              f"{min(vals):9.4f}~{max(vals):<9.4f} {str(cnt):>18s}"
              + ("   ⚠️ " + "; ".join(sorted(allw)) if allw else ""))

    m1 = out["arms"]["① SOL 0.5"]["median"]
    print(f"\n  ★ 재현 검증: 팔 ① 중앙 {m1:.4f}  vs  알려진 대표값 "
          f"{KNOWN_ARM1:.4f}   차 {m1 - KNOWN_ARM1:+.4f}")
    if not approx_equal(m1, KNOWN_ARM1, 5e-4):
        print("  ⛔ 재현이 안 된다 — 재구현이 틀렸다. 다른 팔을 보고하지 않는다.")
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        raise SystemExit(1)
    print("  ✅ 재현됨 — 다른 팔을 읽어도 된다")

    print("\n" + "-" * 96)
    print("  판정 (판정선 0.0516, 완충대 ~0.0589)")
    base = out["arms"]["① SOL 0.5"]["median"]
    for lab in ("② roofline", "③ 안 나눔", "①' SOL 0.25", "①'' SOL 1.0"):
        d = out["arms"][lab]["median"] - base
        v = ("구분 불가" if abs(d) < DELTA else
             "★ 판정선 근처 (판정 안 함)" if abs(d) < BUFFER else
             ("★ ① 보다 좋다" if d < 0 else "★ ① 보다 나쁘다"))
        print(f"  {lab:14s} ① 대비 {d:+.4f}   {v}")

    # 체제별로 나눠서 — ② 의 구분으로 ②③ 을 본다
    print("\n" + "-" * 96)
    print("  ★ roofline 구분으로 본 체제별 홀드아웃 regret (6실행 중앙)")
    reg2 = shape_reg["② roofline"]
    print(f"  {'팔':14s} {'mem':>9s} {'comp':>9s}")
    for lab in ("① SOL 0.5", "② roofline", "③ 안 나눔"):
        cells = {}
        for nm in ("mem", "comp"):
            per = []
            for d in out["arms"][lab]["per_run_shape"]:
                v = [x for k, x in d.items() if reg2.get(k) == nm]
                if v:
                    per.append(geomean(np.array(v)))
            cells[nm] = st.median(per) if per else float("nan")
        out["arms"][lab]["by_roofline"] = cells
        print(f"  {lab:14s} {cells['mem']:9.4f} {cells['comp']:9.4f}")

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
