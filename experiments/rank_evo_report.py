"""★ 순위 손실 진화 결과 — 사전 등록 `rank-evo-prereg.md` 의 지표. LLM 0회.

    python3 experiments/rank_evo_report.py

주 지표는 **tau** 다. `regret` 은 기록만 한다 (사전 등록 §4).
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
TOP_N = 100
TAU_SAMPLE, TAU_SEED = 4000, 12345
#: 상한 측정에서 상위 100 안에서 크게 변하던 것들 (ranking-ceiling.md §3)
WATCH = ("split_k_cost", "sm_idle_cost", "pipeline_warmup_frac",
         "tail_waste", "waves")


def _splits(t: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = t.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    sh = [p for p in t.shapes() if aligned(p)]
    held = [p for p in sh if 11008 in (p.N, p.K)]
    return SplitSet(train=Split("train", tuple(p for p in sh if p not in held)),
                    val=Split("val", tuple(held)), kind="nk11008")


def _taus(code, w, table, matrix, shapes):
    fn = compile_rule(code)
    w = np.asarray(w, dtype=np.float64)
    rng = np.random.default_rng(TAU_SEED)
    tt, ta = [], []
    for p in shapes:
        cand = table.candidates(p)
        s = np.asarray(make_score_of(fn, matrix, w)(p, cand), dtype=np.float64)
        t = np.asarray(table.times_of(p))
        top = np.argsort(t, kind="stable")[:TOP_N]
        if len(np.unique(t[top])) > 1:
            tt.append(kendalltau(s[top], t[top], variant="b").statistic)
        idx = rng.choice(len(t), size=min(TAU_SAMPLE, len(t)), replace=False)
        ta.append(kendalltau(s[idx], t[idx], variant="b").statistic)
    return float(np.median(tt)), float(np.median(ta))


def _feats(code: str) -> set:
    return {n.attr for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "f"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=[f"x-rank-rankevo-s{i}" for i in range(3)])
    ap.add_argument("--out", default="docs/artifacts/rank-evo.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    shapes = list(sp.train.shapes)

    print("=" * 78)
    print("순위 손실 진화 — 사전 등록 지표   ★ 주 지표는 tau, regret 은 기록만")
    print("=" * 78)

    seed = json.loads(Path("runs/F3rw-p8/stage2-rule-writer"
                           "/chosen.json").read_text())
    st, sa = _taus(seed["code"], seed["w0"], T, M, shapes)
    print(f"  씨앗 (공통)          상위100 tau {st:6.3f}   전구간 {sa:6.3f}\n")

    print(f"  {'실행':10s} {'rank':>7} {'regret':>8} {'상위100 tau':>12} "
          f"{'전구간':>8} {'셀':>4} {'항':>3} {'config종류':>9}")
    rows = []
    for run in a.runs:
        d = Path("runs") / run
        arc = sorted((json.loads(x) for x in
                      (d / "archive.jsonl").read_text().splitlines()
                      if x.strip()), key=lambda e: e.get("rank_loss", 1e9))
        best = arc[0]
        t100, tall = _taus(best["code"], best["w"], T, M, shapes)
        rd = [json.loads(x) for x in
              (d / "rounds.jsonl").read_text().splitlines() if x.strip()]
        # config 다양성
        fn = compile_rule(best["code"])
        w = np.asarray(best["w"], dtype=np.float64)
        picks = []
        for p in shapes:
            cand = T.candidates(p)
            s = make_score_of(fn, M, w)(p, cand)
            j = int(cand.top_k(s, 1)[0])
            picks.append((str(cand.kernel_id[j]), int(cand.split_k[j]),
                          str(cand.split_k_mode[j])))
        row = {"run": run, "rank_loss": best.get("rank_loss"),
               "regret": best["regret"], "tau_top100": t100, "tau_all": tall,
               "n_cells": rd[-1]["n_cells"], "n_terms": len(best["w"]),
               "n_config_kinds": len(Counter(picks)),
               "feats": sorted(_feats(best["code"]))}
        rows.append(row)
        print(f"  {run.split('-')[-1]:10s} {best.get('rank_loss', float('nan')):7.4f} "
              f"{best['regret']:8.4f} {t100:12.3f} {tall:8.3f} "
              f"{rd[-1]['n_cells']:4d} {len(best['w']):3d} "
              f"{row['n_config_kinds']:9d}")

    t1 = np.array([r["tau_top100"] for r in rows])
    ta = np.array([r["tau_all"] for r in rows])
    print(f"\n  {'중앙':10s} {'':7s} {np.median([r['regret'] for r in rows]):8.4f} "
          f"{np.median(t1):12.3f} {np.median(ta):8.3f}")
    print(f"  {'범위':10s} {'':7s} {'':8s} "
          f"{t1.min():.3f}~{t1.max():.3f}  {ta.min():.3f}~{ta.max():.3f}")

    print("\n" + "=" * 78)
    print("판정 — 사전 등록에 박은 선")
    print("=" * 78)
    m1, ma = float(np.median(t1)), float(np.median(ta))
    print(f"  상위100 tau 중앙 {m1:.3f}  -> " + (
        "★ 성공 (>=0.30)" if m1 >= 0.30
        else "실패 (<=0.15)" if m1 <= 0.15 else "구분 불가 (0.15~0.30)"))
    print(f"  전구간 tau 중앙  {ma:.3f}  -> " + (
        "유지 (>=0.30)" if ma >= 0.30
        else "★ trade-off (<=0.15)" if ma <= 0.15 else "가운데"))
    print("  ⚠️ 3시드는 유의성을 못 낸다 — 범위 분리로 읽는다")

    print("\n  ★ 상한 측정이 지목한 다섯 축이 쓰이는가 "
          "(상위 100 안에서 크게 변하던 것)")
    for name in WATCH:
        n = sum(1 for r in rows if name in r["feats"])
        print(f"    {name:22s} {n}/{len(rows)} 실행")
    allf = Counter(x for r in rows for x in r["feats"])
    print(f"\n  쓴 축 합집합 {len(allf)}개: "
          f"{', '.join(sorted(allf))}")
    Path(a.out).write_text(json.dumps(
        {"seed_tau": [st, sa], "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
