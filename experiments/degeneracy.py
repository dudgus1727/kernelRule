"""★ 규칙이 형상을 실제로 보는가 — 축퇴 확인. LLM 0회.

    python3 experiments/degeneracy.py

**실험 계획서** `docs/artifacts/degeneracy-prereg.md` — 판정선을 먼저 박았다.

## 무엇을 묻는가

```
암기   if p.M == 4096       ★ 정적 검사가 막는다
축퇴   형상과 무관하게 같은 config 를 고른다   ★ 아무것도 안 막는다
```

## ★ 전이 **전**(A6000)에서 잰다

전이 후 순위가 무너지는 것은 `(b)` 재적합이 고친다. **전이 전에도 못
매기면 규칙의 정체가 다르다.**
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
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
SRC_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
PCTS = (1.0, 5.0, 10.0)
#: 전 구간 tau 는 후보 2만개면 쌍이 2억이라 표본으로 근사한다. **고정 시드.**
TAU_SAMPLE = 4000
TAU_SEED = 12345
TOP_N = 100          # ★ 상위권 tau 는 전수


def _splits(table: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/degeneracy.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    A = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    mA = FeatureMatrix(A, REGISTRY)
    sp = _splits(A)
    shapes = list(sp.train.shapes)      # ★ 학습 41형상

    print("=" * 78)
    print("축퇴 확인 — 규칙이 형상을 실제로 보는가   ★ 전이 전 (A6000)")
    print("=" * 78)
    n_c = [len(A.times_of(p)) for p in shapes]
    print(f"  학습 {len(shapes)}형상   후보 중앙 {int(np.median(n_c))}개")
    ans_sz = sorted(int(A.answer_mask(p).sum()) for p in shapes)
    print(f"  정답 집합 크기: 중앙 {int(np.median(ans_sz))}  "
          f"범위 {ans_sz[0]}~{ans_sz[-1]}   1개인 형상 "
          f"{sum(1 for x in ans_sz if x == 1)}개")

    # -- 정적 top-1 — ★ 대표값 구현을 쓴다. 세 번째 정의를 만들지 않는다
    #    (원칙 2. 오늘 `_arith_intensity` 에서 이미 밟았다)
    #    `StaticTopK` 의 키는 (kernel_id, split_k, split_k_mode) 다 —
    #    축 좌표가 아니다. 그래서 규칙의 선택도 같은 키로 센다.
    from kernelrule.baselines.static_topk import StaticTopK

    st = StaticTopK(A, shapes, coverage="union").run(ks=(1,))
    static_key = st.chosen[0]
    print(f"  정적 top-1 (대표값: 전체 status + 합집합 덮개)  "
          f"regret {st.by_k[1]['all']:.4f}  덮개 {st.coverage[1]:.0%}")
    print("  ★ 키 = (kernel_id, split_k, split_k_mode) — 규칙의 선택도 "
          "같은 키로 센다\n")

    # ★ 상위권에 순위가 존재하는가 — tau 를 읽기 전에 이것부터 본다
    uniq = np.array([len(np.unique(np.asarray(A.times_of(p))[
        np.argsort(np.asarray(A.times_of(p)), kind="stable")[:TOP_N]]))
        for p in shapes])
    print(f"  ★ 참 상위 {TOP_N}개의 서로 다른 시간값: 중앙 "
          f"{int(np.median(uniq))}  범위 {uniq.min()}~{uniq.max()}   "
          f"전부 동률 {int((uniq == 1).sum())}형상 / "
          f"10개 이하 {int((uniq <= 10).sum())}형상")
    print(f"     눈금 {A.noise.tick_ms} ms — 상위권은 분해능이 지배한다\n")

    rng0 = np.random.default_rng(TAU_SEED)
    res: dict = {"runs": {}, "n_shapes": len(shapes),
                 "answer_set_sizes": ans_sz}

    print(f"  {'구조':22s} {'종류':>5} {'최빈':>5} {'정적과':>6} "
          f"{'hit@1':>6} {'tau 전구간':>10} {'★tau 상위100':>12}")
    for run in SRC_RUNS:
        f = Path("runs") / run / "archive.jsonl"
        e = sorted((json.loads(x) for x in f.read_text().splitlines()
                    if x.strip()), key=lambda z: z["regret"])[0]
        fn = compile_rule(e["code"])
        ws = {}
        for nm in ("short", "long"):
            g = [q for q in shapes if regime_of(q, A.hw) == nm]
            ws[nm] = fit_weights(fn, mA, A, Split("train", tuple(g)),
                                 e["w"], max_evals=300,
                          objective="regret").w

        picks, hit1, taus, taus_top = [], 0, [], []
        n_flat_top = 0
        hits = {q: [0, 0] for q in PCTS}
        for p in shapes:
            cand = A.candidates(p)
            sc = np.asarray(make_score_of(fn, mA, ws[regime_of(p, A.hw)])(
                p, cand), dtype=float)
            t = A.times_of(p)
            pick = int(cand.top_k(sc, 1)[0])
            picks.append((str(cand.kernel_id[pick]),
                          int(cand.split_k[pick]),
                          str(cand.split_k_mode[pick])))
            hit1 += int(t[pick] == t.min())
            # tau — 전 구간은 표본, 상위권은 전수
            n = len(t)
            idx = rng0.choice(n, size=min(TAU_SAMPLE, n), replace=False)
            taus.append(kendalltau(sc[idx], t[idx], variant="b").statistic)
            top = np.argsort(t, kind="stable")[:TOP_N]
            # ★ 상위 TOP_N 의 시간이 전부 동률이면 tau 가 정의되지 않는다.
            #   `nan` 하나가 median 을 통째로 오염시킨다 — 세고 뺀다.
            if len(np.unique(t[top])) > 1:
                taus_top.append(kendalltau(sc[top], t[top],
                                           variant="b").statistic)
            else:
                n_flat_top += 1
            order = np.argsort(sc, kind="stable")
            am = A.answer_mask(p)
            best = int(np.argmin(t))
            for q in PCTS:
                k = max(1, int(np.ceil(n * q / 100.0)))
                s_ = set(order[:k].tolist())
                hits[q][0] += int(best in s_)
                hits[q][1] += int(am[order[:k]].any())

        c = Counter(picks)
        n_static = sum(1 for k in picks if k == static_key)
        row = {"n_kinds": len(c), "top_count": c.most_common(1)[0][1],
               "same_as_static": n_static, "hit1": hit1,
               "tau_all": float(np.median(taus)),
               "tau_top100": (float(np.median(taus_top)) if taus_top
                              else float("nan")),
               "n_flat_top": n_flat_top,
               "hits": {str(q): {"best": hits[q][0], "answer": hits[q][1]}
                        for q in PCTS}}
        res["runs"][run] = row
        print(f"  {run:22s} {len(c):5d} {row['top_count']:5d} "
              f"{n_static:6d} {hit1:6d} {row['tau_all']:10.3f} "
              f"{row['tau_top100']:12.3f}"
              + (f"  (동률 {n_flat_top})" if n_flat_top else ""))

    # -- 무작위 바닥 (20번 뽑기 — ★ 바닥도 표본이다, 원칙 7) ---------------
    rng = np.random.default_rng(0)
    b_hit1, b_hits = [], {q: [[], []] for q in PCTS}
    b_tau, b_tau_top, b_kinds = [], [], []
    for _ in range(20):
        h1 = 0
        picks = []
        hh = {q: [0, 0] for q in PCTS}
        tt, ttt = [], []
        for p in shapes:
            t = A.times_of(p)
            n = len(t)
            sc = rng.random(n)
            order = np.argsort(sc, kind="stable")
            cd = A.candidates(p)
            j = int(order[0])
            picks.append((str(cd.kernel_id[j]), int(cd.split_k[j]),
                          str(cd.split_k_mode[j])))
            h1 += int(t[int(order[0])] == t.min())
            am = A.answer_mask(p)
            best = int(np.argmin(t))
            for q in PCTS:
                k = max(1, int(np.ceil(n * q / 100.0)))
                s_ = set(order[:k].tolist())
                hh[q][0] += int(best in s_)
                hh[q][1] += int(am[order[:k]].any())
            idx = rng.choice(n, size=min(TAU_SAMPLE, n), replace=False)
            tt.append(kendalltau(sc[idx], t[idx], variant="b").statistic)
            top = np.argsort(t, kind="stable")[:TOP_N]
            if len(np.unique(t[top])) > 1:
                ttt.append(kendalltau(sc[top], t[top], variant="b").statistic)
        b_hit1.append(h1)
        b_kinds.append(len(Counter(picks)))
        b_tau.append(float(np.median(tt)))
        b_tau_top.append(float(np.median(ttt)) if ttt else float("nan"))
        for q in PCTS:
            b_hits[q][0].append(hh[q][0])
            b_hits[q][1].append(hh[q][1])
    print(f"  {'★ 무작위 (20뽑기 평균)':22s} {np.mean(b_kinds):5.1f} "
          f"{'—':>5} {'—':>6} {np.mean(b_hit1):6.1f} "
          f"{np.mean(b_tau):10.3f} {np.mean(b_tau_top):12.3f}")

    print(f"\n  {'구조':22s} " + "  ".join(
        f"{q}% 정답/최적" for q in PCTS))
    for run in SRC_RUNS:
        h = res["runs"][run]["hits"]
        print(f"  {run:22s} " + "  ".join(
            f"{h[str(q)]['answer']:2d}/{h[str(q)]['best']:2d}"
            f" ({h[str(q)]['answer'] / len(shapes):.0%})" for q in PCTS))
    print(f"  {'★ 무작위 평균':22s} " + "  ".join(
        f"{np.mean(b_hits[q][1]):4.1f}/{np.mean(b_hits[q][0]):4.1f}"
        f" ({np.mean(b_hits[q][1]) / len(shapes):.0%})" for q in PCTS))

    res["random"] = {
        "n_kinds": float(np.mean(b_kinds)), "hit1": float(np.mean(b_hit1)),
        "tau_all": float(np.mean(b_tau)),
        "tau_top100": float(np.mean(b_tau_top)),
        "hits": {str(q): {"best": float(np.mean(b_hits[q][0])),
                          "answer": float(np.mean(b_hits[q][1]))}
                 for q in PCTS}}

    # -- 판정 (실험 계획서에 박은 선) ----------------------------------------
    kinds = [res["runs"][r]["n_kinds"] for r in SRC_RUNS]
    tt = [res["runs"][r]["tau_top100"] for r in SRC_RUNS]
    n_flat = sum(res["runs"][r]["n_flat_top"] for r in SRC_RUNS) // len(SRC_RUNS)
    print("\n" + "=" * 78)
    print("판정 — 실험 계획서에 박은 선")
    print("=" * 78)
    mk = float(np.median(kinds))
    mt = float(np.nanmedian(tt))
    print(f"  다양성 중앙 {mk:.1f}종  -> " + (
        "★ 형상을 실제로 본다 (>=20)" if mk >= 20
        else "★ 사실상 상수 + 변주 (<=5)" if mk <= 5 else "가운데 (6~19)"))
    print(f"  상위100 tau 중앙 {mt:.3f}  "
          f"(★ 동률로 정의 불가 {n_flat}/{len(shapes)}형상 제외)  -> " + (
        "★ 의미 있는 구간에서 순위를 매긴다 (>=0.30)" if mt >= 0.30
        else "★ 매기지 못한다 (<=0.10)" if mt <= 0.10 else "가운데"))
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
