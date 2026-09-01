"""★ §29.5 구조 전이 — (a) 완전 이식 / (b) 재적합 / (c) 재생성. LLM 0회.

    python3 experiments/transfer_29_5.py --pair a6000 5090
    python3 experiments/transfer_29_5.py --pair a6000 4090   # 표가 오면

**사전 등록** `docs/artifacts/transfer-prereg.md` + 부칙.

## 세 팔

```
(a) 완전 이식   A6000 구조 + A6000 가중치 그대로 -> 5090 에서 채점
(b) 재적합      구조 고정, 5090 학습 분할로 가중치만 다시
(c) 재생성      5090 표에서 처음부터   <- 이미 돌았다 (5090sigma 3시드)
```

## ★ 규칙을 하나 고르지 않는다 (원칙 5)

"A6000 에서 진화한 규칙" 을 하나 고르면 그 선택이 낙관 편향이 된다.
**A6000 F3 캠페인 6시드 전부**의 학습 최고 규칙을 옮기고 분포를 낸다.

## ★ 기준선 — 사전 등록의 대체안을 쓴다

`nvMatmulHeuristics` 가 이 환경에 없다 (import 실패). 사전 등록이
미리 적어 둔 대로 **5090 에서 재적합한 `physics_seeded`** 를 쓴다.
결과를 보고 고른 것이 아니다.

## ★ 뒤집힌 형상을 포함/제외 양쪽으로

홀드아웃 20형상 중 2개가 바운드 뒤집힘이다. 한쪽만 내면 고른 것이 된다.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.crosstable import bound_flipped, common_shapes
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

#: ★ 표 등록부. **새 GPU 는 여기 한 줄이면 된다** (원칙 2).
#:   `runs` 는 그 표에서 **처음부터** 돌린 캠페인 = (c) 재생성이고,
#:   같은 목록이 다른 쌍에서 (a)(b) 의 **출처**로도 쓰인다.
#: ★ (c) 는 GPU 당 한 번이고 (a)(b) 는 조합이 몇 개든 공짜다.
TABLES: dict[str, dict] = {
    "a6000": {
        "bundle": "datasets/rtx-a6000-sm_86-c63710df",
        "env_hash": "c63710df",
        "runs": [f"f1pipe-F3-arch24-s{i}" for i in range(6)]},
    "5090": {
        "bundle": "datasets/rtx-5090-sm_120-5bb6f403",
        "env_hash": "5bb6f403",
        "runs": [f"f1pipe-F3-5090sigma-s{i}" for i in range(3)]},
    # ★ 4090 표가 오면 여기 세 줄. transfer-prereg 부칙의 세 쌍이 열린다.
    # "4090": {"bundle": "datasets/rtx-4090-...", "env_hash": "...",
    #          "runs": [f"f1pipe-F3-4090sigma-s{i}" for i in range(6)]},
}


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


def _best(run: str) -> dict:
    """학습 점수로 아카이브에서 하나. **홀드아웃을 안 본다** (§10.2)."""
    f = Path("runs") / run / "archive.jsonl"
    arc = sorted((json.loads(x) for x in f.read_text().splitlines()
                  if x.strip()), key=lambda e: e["regret"])
    return arc[0]


def _fit_per_regime(code, w0, table, matrix, train):
    fn = compile_rule(code)
    out = {}
    for name in ("short", "long"):
        g = [p for p in train if regime_of(p, table.hw) == name]
        out[name] = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                                w0, max_evals=300).w
    return fn, out


def _score_on(fn, ws, table, matrix, shapes) -> float:
    """체제별 가중치로 형상들을 채점. **체제 판정은 이 표의 하드웨어로.**"""
    regs = []
    for p in shapes:
        reg = regime_of(p, table.hw)
        cand = table.candidates(p)
        sc = make_score_of(fn, matrix, ws[reg])(p, cand)
        pick = cand.top_k(sc, 1)[0]
        t = table.times_of(p)
        regs.append(float(t[pick] / t.min()))
    return geomean(np.array(regs))


def _row(label: str, xs: list[float]) -> str:
    a = np.array(xs)
    return (f"  {label:34s} 중앙 {np.median(a):.4f}  "
            f"범위 {a.min():.4f}~{a.max():.4f}  폭 {a.max() - a.min():.4f}"
            f"  (n={len(a)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, default=["a6000", "5090"],
                    metavar=("SRC", "DST"),
                    help=f"표 이름 둘. 등록된 것: {sorted(TABLES)}")
    ap.add_argument("--out", default=None,
                    help="기본은 docs/artifacts/transfer-<src>-<dst>.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    src, dst = a.pair
    for n in (src, dst):
        if n not in TABLES:
            raise SystemExit(
                f"등록되지 않은 표 {n!r}. TABLES 에 한 줄 넣어라. "
                f"지금 있는 것: {sorted(TABLES)}")
    if src == dst:
        raise SystemExit("같은 표끼리는 전이가 아니다.")
    S, D = TABLES[src], TABLES[dst]
    SRC_RUNS, DST_RUNS = S["runs"], D["runs"]
    missing = [r for r in SRC_RUNS + DST_RUNS
               if not (Path("runs") / r / "archive.jsonl").exists()]
    if missing:
        raise SystemExit(
            "없는 실행: " + ", ".join(missing)
            + "\n  ★ 조용히 건너뛰면 표본이 줄어든 줄 모르고 결론을 낸다.")
    out = a.out or f"docs/artifacts/transfer-{src}-{dst}.json"

    A = PerfTable.from_bundle(S["bundle"], env_hash=S["env_hash"],
                              ok_only=False)
    B = PerfTable.from_bundle(D["bundle"], env_hash=D["env_hash"],
                              ok_only=False)
    mA, mB = FeatureMatrix(A, REGISTRY), FeatureMatrix(B, REGISTRY)
    spA, spB = _splits(A), _splits(B)

    flip = {(p.M, p.N, p.K) for p, _, _ in bound_flipped(A, B)}
    common = {(p.M, p.N, p.K) for p in common_shapes(A, B)}
    hold = [p for p in spB.val.shapes if (p.M, p.N, p.K) in common]
    hold_nf = [p for p in hold if (p.M, p.N, p.K) not in flip]

    print("=" * 78)
    print(f"§29.5 구조 전이  {src} -> {dst}")
    print("=" * 78)
    print(f"  ridge {A.hw.ridge_point:.1f} -> {B.hw.ridge_point:.1f}  "
          f"({B.hw.ridge_point / A.hw.ridge_point:.2f}배)   "
          f"SM {A.hw.sm_count} -> {B.hw.sm_count}")
    print(f"  홀드아웃 {len(hold)}형상 (공통 53 안)   "
          f"뒤집힘 제외 {len(hold_nf)}형상")
    print(f"  5090 학습 {len(spB.train.shapes)}형상 — (b) 재적합에 쓴다")
    print("  ★ 기준선: nvMatmulHeuristics 없음 -> 사전 등록의 대체안"
          " (5090 재적합 physics_seeded)\n")

    res: dict = {"a": [], "b": [], "c": [], "a_nf": [], "b_nf": [],
                 "c_nf": [], "src": []}

    for run in SRC_RUNS:
        e = _best(run)
        fn = compile_rule(e["code"])
        # (a) A6000 에서 맞춘 체제별 가중치를 **그대로**
        _, wsA = _fit_per_regime(e["code"], e["w"], A, mA,
                                 list(spA.train.shapes))
        va = _score_on(fn, wsA, B, mB, hold)
        va_nf = _score_on(fn, wsA, B, mB, hold_nf)
        # (b) 5090 학습 분할로 가중치만 다시
        _, wsB = _fit_per_regime(e["code"], e["w"], B, mB,
                                 list(spB.train.shapes))
        vb = _score_on(fn, wsB, B, mB, hold)
        vb_nf = _score_on(fn, wsB, B, mB, hold_nf)
        res["src"].append(run)
        res["a"].append(va)
        res["a_nf"].append(va_nf)
        res["b"].append(vb)
        res["b_nf"].append(vb_nf)
        print(f"  {run:26s} (a) {va:.4f}   (b) {vb:.4f}   "
              f"[뒤집힘 제외 {va_nf:.4f} / {vb_nf:.4f}]", flush=True)

    print()
    for run in DST_RUNS:
        e = _best(run)
        fn, ws = _fit_per_regime(e["code"], e["w"], B, mB,
                                 list(spB.train.shapes))
        vc = _score_on(fn, ws, B, mB, hold)
        vc_nf = _score_on(fn, ws, B, mB, hold_nf)
        res["c"].append(vc)
        res["c_nf"].append(vc_nf)
        print(f"  {run:26s} (c) {vc:.4f}"
              f"                 [뒤집힘 제외 {vc_nf:.4f}]", flush=True)

    from kernelrule.rules.physics_seeded import CODE as PS_CODE
    from kernelrule.rules.physics_seeded import W0 as PS_W0
    fn, ws = _fit_per_regime(PS_CODE, list(PS_W0), B, mB,
                             list(spB.train.shapes))
    base = _score_on(fn, ws, B, mB, hold)
    base_nf = _score_on(fn, ws, B, mB, hold_nf)
    res["baseline"] = base
    res["baseline_nf"] = base_nf

    print("\n" + "=" * 78)
    print(f"홀드아웃 {len(hold)}형상 (뒤집힘 포함)")
    print("=" * 78)
    print(_row("(a) 완전 이식", res["a"]))
    print(_row("(b) 재적합", res["b"]))
    print(_row("(c) 재생성", res["c"]))
    print(f"  {'★ 기준선 physics_seeded(5090 재적합)':34s} {base:.4f}")
    print(f"\n홀드아웃 {len(hold_nf)}형상 (뒤집힘 제외)")
    print("-" * 78)
    print(_row("(a) 완전 이식", res["a_nf"]))
    print(_row("(b) 재적합", res["b_nf"]))
    print(_row("(c) 재생성", res["c_nf"]))
    print(f"  {'★ 기준선 physics_seeded':34s} {base_nf:.4f}")

    res["pair"] = [src, dst]
    res["ridge"] = [A.hw.ridge_point, B.hw.ridge_point]
    res["n_holdout"] = [len(hold), len(hold_nf)]
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {out}")
    print("  ⚠️ 유의성은 붙이지 않는다 — σ 신뢰구간이 넓다 "
          "(sigma-5090.json)")


if __name__ == "__main__":
    main()
