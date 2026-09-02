"""★ 두 단계 목적함수(A) + 순위 규칙의 전이(B). LLM 0회.

    python3 experiments/two_stage.py

**사전 등록** `docs/artifacts/two-stage-prereg.md` — 판정선을 먼저 박았다.

## A — 구조는 순위로, 가중치는 regret 으로

순위 진화 3실행의 **최종 구조를 그대로** 쓰고 가중치만 다시 맞춘다.

## B — 순위 진화 규칙이 5090 으로 옮겨가나

§29.5 는 `regret` 진화 규칙으로만 했다. 순위 규칙은 다를 수 있다.

⚠️ **전부 홀드아웃에서 잰다.** D-101 의 tau(0.389)는 학습 41형상
값이라 나란히 못 놓는다 (원칙 4).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
RANK_RUNS = [f"f1pipe-F3-rankevo-s{i}" for i in range(3)]
REG_RUNS = [f"f1pipe-F3-arch24-s{i}" for i in range(6)]
TOP_N, TAU_SAMPLE, TAU_SEED, N_DRAWS = 100, 4000, 12345, 20


def _splits(t: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = t.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    sh = [p for p in t.shapes() if aligned(p)]
    held = [p for p in sh if 11008 in (p.N, p.K)]
    return SplitSet(train=Split("train", tuple(p for p in sh if p not in held)),
                    val=Split("val", tuple(held)), kind="nk11008")


def _best(run: str, by: str) -> dict:
    f = Path("runs") / run / "archive.jsonl"
    arc = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    key = (lambda e: e.get("rank_loss", 1e9)) if by == "rank" \
        else (lambda e: e["regret"])
    return sorted(arc, key=key)[0]


def _fit(code, w0, table, matrix, train, objective, *,
         rank_top_k: int = TOP_N, rank_lambda: float = 0.0):
    """체제별로 맞춘다 — 정준 절차 (§10).

    ★ `rank_top_k` / `rank_lambda` 는 **그 실행의 조건**이다. 기본값으로
    두면 k 스윕·λ 스윕을 전부 k=100·λ=0 으로 재게 된다 (원칙 37).
    """
    fn = compile_rule(code)
    ws = {}
    for nm in ("short", "long"):
        g = [q for q in train if regime_of(q, table.hw) == nm]
        ws[nm] = fit_weights(fn, matrix, table, Split("train", tuple(g)), w0,
                             max_evals=300, objective=objective,
                             rank_top_k=rank_top_k,
                             rank_lambda=rank_lambda if objective == "rank"
                             else 0.0).w
    return fn, ws


def _measure(fn, ws, table, matrix, shapes, top_n: int = TOP_N) -> tuple:
    """(regret, 상위 `top_n` tau, 전구간 tau, **정의 안 되는 형상 수**).

    ★ 네 번째 값을 꼭 보라. 규칙이 상위 `top_n` 안에서 **상수 점수**를
    내면 tau 가 정의되지 않는다. 그것을 안 세고 중앙값을 내면 `nan` 이
    번지거나(numpy) 조용히 형상이 빠진다. k=10 에서 실제로 났다 —
    한 형상에서 점수 고유값이 1개였다.
    """
    rng = np.random.default_rng(TAU_SEED)
    regs, tt, ta, undef = [], [], [], []
    for p in shapes:
        cand = table.candidates(p)
        w = ws[regime_of(p, table.hw)] if isinstance(ws, dict) else ws
        s = np.asarray(make_score_of(fn, matrix, w)(p, cand), dtype=np.float64)
        t = np.asarray(table.times_of(p))
        regs.append(float(t[cand.top_k(s, 1)[0]] / t.min()))
        top = np.argsort(t, kind="stable")[:top_n]
        if len(np.unique(t[top])) > 1:
            v = kendalltau(s[top], t[top], variant="b").statistic
            (tt if np.isfinite(v) else undef).append(v)
        idx = rng.choice(len(t), size=min(TAU_SAMPLE, len(t)), replace=False)
        ta.append(kendalltau(s[idx], t[idx], variant="b").statistic)
    return (float(np.exp(np.mean(np.log(regs)))),
            float(np.median(tt)) if tt else float("nan"),
            float(np.median(ta)),
            len(undef))


def _floor(table, shapes, top_n: int = TOP_N) -> tuple:
    """★ 무작위 바닥 (20뽑기 평균). 바닥도 표본이다 (원칙 7)."""
    rng = np.random.default_rng(0)
    R, T1, TA = [], [], []
    for _ in range(N_DRAWS):
        regs, tt, ta = [], [], []
        for p in shapes:
            t = np.asarray(table.times_of(p))
            s = rng.random(len(t))
            regs.append(float(t[int(np.argmin(s))] / t.min()))
            top = np.argsort(t, kind="stable")[:top_n]
            if len(np.unique(t[top])) > 1:
                tt.append(kendalltau(s[top], t[top], variant="b").statistic)
            idx = rng.choice(len(t), size=min(TAU_SAMPLE, len(t)),
                             replace=False)
            ta.append(kendalltau(s[idx], t[idx], variant="b").statistic)
        R.append(float(np.exp(np.mean(np.log(regs)))))
        T1.append(float(np.median(tt)))
        TA.append(float(np.median(ta)))
    return float(np.mean(R)), float(np.mean(T1)), float(np.mean(TA))


def _row(label, vals) -> None:
    v = np.array(vals)
    print(f"  {label:34s} {np.median(v[:, 0]):8.4f} {np.median(v[:, 1]):12.3f} "
          f"{np.median(v[:, 2]):10.3f}   "
          f"({v[:, 1].min():.3f}~{v[:, 1].max():.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/two-stage.json")
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    out: dict = {}

    A = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    mA = FeatureMatrix(A, REGISTRY)
    spA = _splits(A)
    hold = list(spA.val.shapes)
    train = list(spA.train.shapes)

    print("=" * 82)
    print("A. 목적함수 2x2 — 구조와 가중치 중 무엇이 순위 능력을 담나")
    print("=" * 82)
    print(f"  A6000 홀드아웃 {len(hold)}형상   ★ D-101 의 tau 는 학습 "
          f"41형상 값이라 여기와 나란히 못 놓는다\n")
    print(f"  {'':34s} {'regret':>8} {'상위100 tau':>12} {'전구간':>10}")

    res: dict = {}
    for name, runs, by, obj in (
            ("순위 구조 + 순위 가중치", RANK_RUNS, "rank", "rank"),
            ("★ 순위 구조 + regret 가중치", RANK_RUNS, "rank", "regret"),
            # ★ 2x2 의 빈 칸 (D-103). 세 칸으로 "가중치가 들고 있다" 고
            #   말한 것은 **대각선만 보고 한 말**이었다.
            ("★ regret 구조 + 순위 가중치", REG_RUNS, "regret", "rank"),
            ("regret 구조 + regret 가중치", REG_RUNS, "regret", "regret")):
        vals = []
        for run in runs:
            e = _best(run, by)
            fn, ws = _fit(e["code"], e["w"], A, mA, train, obj)
            vals.append(_measure(fn, ws, A, mA, hold))
        res[name] = vals
        _row(name, vals)
    fl = _floor(A, hold)
    print(f"  {'★ 무작위 바닥 (20뽑기)':34s} {fl[0]:8.4f} {fl[1]:12.3f} "
          f"{fl[2]:10.3f}")
    out["A"] = {"rows": dict(res), "floor": fl,
                "n_holdout": len(hold)}

    mid = np.array(res["★ 순위 구조 + regret 가중치"])
    r, t1 = float(np.median(mid[:, 0])), float(np.median(mid[:, 1]))
    print("\n  판정 — 사전 등록에 박은 선")
    print(f"    regret {r:.4f} / 상위100 tau {t1:.3f}  ->  " + (
        "★ 성공 (regret<=1.10 이면서 tau>=0.30)"
        if r <= 1.10 and t1 >= 0.30
        else "가중치가 tau 를 지운다 (regret<=1.10, tau<=0.15)"
        if r <= 1.10 and t1 <= 0.15
        else "★ 구조가 regret 에 안 맞는다 (regret>=1.30) -> (2) 필요"
        if r >= 1.30 else "구분 불가"))

    if a.skip_b:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return

    # ---------------------------------------------------------------- B
    B = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    mB = FeatureMatrix(B, REGISTRY)
    spB = _splits(B)
    holdB, trainB = list(spB.val.shapes), list(spB.train.shapes)
    print("\n" + "=" * 82)
    print("B. 순위 진화 규칙의 전이 — A6000 -> 5090")
    print("=" * 82)
    print(f"  5090 홀드아웃 {len(holdB)}형상")
    print("  ⚠️ 5090 상위 100 은 폭 1.2% / 고유값 5개다 — 배울 순위가 적다\n")
    print(f"  {'':34s} {'regret':>8} {'상위100 tau':>12} {'전구간':>10}")

    resB: dict = {}
    for name, obj in (("(a) 완전 이식 (A6000 가중치)", None),
                      ("★ (b) 재적합 (5090 순위 손실)", "rank")):
        vals = []
        for run in RANK_RUNS:
            e = _best(run, "rank")
            if obj is None:
                fn, ws = _fit(e["code"], e["w"], A, mA, train, "rank")
            else:
                fn, ws = _fit(e["code"], e["w"], B, mB, trainB, obj)
            vals.append(_measure(fn, ws, B, mB, holdB))
        resB[name] = vals
        _row(name, vals)
    flB = _floor(B, holdB)
    print(f"  {'★ 무작위 바닥 (20뽑기)':34s} {flB[0]:8.4f} {flB[1]:12.3f} "
          f"{flB[2]:10.3f}")
    out["B"] = {"rows": resB, "floor": flB, "n_holdout": len(holdB)}

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3시드는 유의성을 못 낸다 — 범위 분리로 읽는다")


if __name__ == "__main__":
    main()
