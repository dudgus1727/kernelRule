"""★ `cross` 관찰 — 실험 계획서 `docs/artifacts/cross-prereg.md`. LLM 0회.

    python3 experiments/cross_observe.py --runs F3rw-p8-cross-s{0,1,2} \\
                                         --control F3rw-p8-abl-analyst-s{0,1,2}

## 네 가지

    1 ★ 자식이 두 부모의 항을 실제로 섞는가       cross.jsonl
    2   아카이브 셀 점유                          rounds.jsonl
    3 ★ 앙상블이 통하기 시작하는가 (D-42 재시험)  archive.jsonl + 표
    4   cross 의 중복률                           rounds.jsonl by_parent_kind

## ⚠️ 3번은 옛 D-42 수치와 나란히 못 놓는다

D-42 의 폭 0.098 -> 0.100 / 중앙 1.0817 -> 1.1137 은 **삭제된 gpt-5.4
산출물**에서 나왔다 (D-52, `selection_spread.py:RUNS` 가 비어 있는 이유).
모델이 다르면 나란히 못 놓는다. 그래서 **같은 모델의 대조군(`abl-B`)을
여기서 함께 계산한다.**

⚠️ 그래도 `abl-B` 는 라운드로빈 배정(D-94 이전)이다 — 실험 계획서가 적어
둔 교락이다. 3번은 **방향만** 읽고 성능 판정에 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
KS = (1, 3, 5)


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# -- 1. 섞임 -----------------------------------------------------------------
def obs1(runs: list[Path]) -> None:
    print("=" * 76)
    print("1 ★ cross 자식이 두 부모의 항을 실제로 섞는가")
    print("=" * 76)
    print("  기준: 섞을 것이 있었던 제안 중 **양쪽 고유 항을 둘 다** 쓴 비율")
    print("        >= 40% 섞는다 / <= 10% 한쪽만 베낀다\n")
    tot_m = tot_mix = tot_cp = tot_n = 0
    for d in runs:
        rows = _jsonl(d / "cross.jsonl")
        m = [r for r in rows if r["mixable"]]
        mix = [r for r in m if r["mixed"]]
        cp = [r for r in rows if r["copied"]]
        tot_n += len(rows)
        tot_m += len(m)
        tot_mix += len(mix)
        tot_cp += len(cp)
        rate = f"{len(mix) / len(m):.1%}" if m else "—"
        print(f"  {d.name:26s} 제안 {len(rows):3d} | 섞을 수 있었던 것 "
              f"{len(m):3d} | 섞음 {len(mix):3d} = {rate:>6s} | "
              f"통째로 베낌 {len(cp):3d}")
    r = tot_mix / tot_m if tot_m else float("nan")
    print(f"\n  {'합계':26s} 제안 {tot_n:3d} | 섞을 수 있었던 것 {tot_m:3d} "
          f"| 섞음 {tot_mix:3d} = {r:.1%} | 베낌 {tot_cp:3d}")
    if not tot_m:
        print("  판정: ★ 표본 0 — 섞을 것이 있었던 제안이 없다. "
              "판정하지 않는다 (원칙 27)")
        return
    verdict = ("★ 교차가 실제로 돈다" if r >= 0.40
               else "★ 프롬프트가 안 듣는다 — 성능을 보기 전에 멈춘다"
               if r <= 0.10 else "가운데 — 실험 계획서에 없는 구간이다")
    print(f"  판정: {verdict}")
    # 어떤 항을 버렸나 — 예산이 있으므로 합치면 반드시 버려야 한다
    dropped = [len(set(x["a"]) | set(x["b"])) - len(x["child"])
               for x in _all(runs) if x["mixable"]]
    if dropped:
        print(f"  합쳤을 때 버린 항 수: 중앙 {np.median(dropped):.1f}  "
              f"범위 {min(dropped)}~{max(dropped)}")


def _all(runs: list[Path]) -> list[dict]:
    return [r for d in runs for r in _jsonl(d / "cross.jsonl")]


def _cell(agg: dict, k: str) -> str:
    v = agg.get(k)
    if not v or not v["n"]:
        return f"{'—':>16}"
    return (f"{v['dup']}/{v['n']}={v['dup'] / v['n']:.0%} "
            f"채점{v['scored']}").rjust(16)


# -- 2. 셀 점유 / 4. 중복률 ---------------------------------------------------
def obs2_4(runs: list[Path], control: list[Path]) -> None:
    print("\n" + "=" * 76)
    print("2  아카이브 셀 점유   ·   4  부모 종류별 중복률")
    print("=" * 76)
    print(f"  {'실행':26s} {'셀(최종)':>9} {'셀(최대)':>9}   "
          f"{'exploit':>16} {'explore':>16} {'cross':>16}")
    for label, group in (("cross1", runs), ("대조 abl-B", control)):
        if not group:
            continue
        print(f"  -- {label} " + "-" * 40)
        for d in group:
            rs = _jsonl(d / "rounds.jsonl")
            if not rs:
                continue
            cells = [x.get("n_cells", 0) for x in rs]
            agg: dict = {}
            for x in rs:
                for k, v in (x.get("by_parent_kind") or {}).items():
                    a = agg.setdefault(k, {"n": 0, "dup": 0, "scored": 0})
                    for kk in a:
                        a[kk] += v.get(kk, 0)
            print(f"  {d.name:26s} {cells[-1]:9d} {max(cells):9d}   "
                  + " ".join(_cell(agg, k) for k in
                             ("exploit", "explore", "cross")))
    print("\n  ⚠️ 4번은 `by_parent_kind` 가 있는 실행끼리만 비교한다 "
          "(D-94 이후). 대조군에 '—' 가 있으면 그때는 안 남겼다는 뜻이다.")


# -- 3. 앙상블 ---------------------------------------------------------------
def obs3(runs: list[Path], control: list[Path]) -> None:
    print("\n" + "=" * 76)
    print("3 ★ 앙상블이 통하기 시작하는가 — D-42 재시험")
    print("=" * 76)
    print("  ⚠️ 옛 D-42 수치(폭 0.098 / 중앙 1.0817->1.1137)는 **삭제된")
    print("     gpt-5.4 산출물**의 것이다. 모델이 다르므로 나란히 안 놓는다.")
    print("  ⚠️ 대조군은 라운드로빈 배정(D-94 이전)이다 — 방향만 읽는다.\n")
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")
    train = list(splits.train.shapes)

    def fit_per_regime(code, w0):
        fn = compile_rule(code)
        out = {}
        for name in ("short", "long"):
            g = [p for p in train if regime_of(p, table.hw) == name]
            out[name] = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                                    w0, max_evals=300,
                          objective="regret").w
        return fn, out

    def ensemble(fitted: list) -> float:
        regs = []
        for p in held:
            reg = regime_of(p, table.hw)
            cand = table.candidates(p)
            ranks = np.zeros(len(cand.tiebreak), dtype=float)
            for fn, ws in fitted:
                sc = make_score_of(fn, matrix, ws[reg])(p, cand)
                # ★ 동률 보존 순위 (D-41). argsort(argsort(.)) 는 안 된다
                ranks += np.unique(sc, return_inverse=True)[1].astype(float)
            pick = cand.top_k(ranks, 1)[0]
            t = table.times_of(p)
            regs.append(float(t[pick] / t.min()))
        return geomean(np.array(regs))

    for label, group in (("cross1 (두 부모)", runs),
                         ("대조 abl-B (한 부모)", control)):
        if not group:
            continue
        print(f"  -- {label} " + "-" * 34)
        ens: dict = {k: [] for k in KS}
        single = []
        for d in group:
            arc = sorted(_jsonl(d / "archive.jsonl"),
                         key=lambda e: e["regret"])[:max(KS)]
            if not arc:
                continue
            fitted = [fit_per_regime(e["code"], e["w"]) for e in arc]
            h = canonical_score(arc[0]["code"], arc[0]["w"], table=table,
                                matrix=matrix, splits=splits).holdout
            single.append(h)
            row = []
            for k in KS:
                v = ensemble(fitted[:k])
                ens[k].append(v)
                row.append(f"{v:.4f}")
            print(f"  {d.name:26s} k=1단일 {h:.4f} | 앙상블 "
                  + "  ".join(f"k={k} {x}" for k, x in zip(KS, row,
                                                           strict=True)),
                  flush=True)
        if single:
            print(f"  {'폭 (최대-최소)':26s} 단일 "
                  f"{max(single) - min(single):.4f} | "
                  + "  ".join(f"k={k} {max(ens[k]) - min(ens[k]):.4f}"
                              for k in KS))
            print(f"  {'중앙':26s} 단일 {np.median(single):.4f} | "
                  + "  ".join(f"k={k} {np.median(ens[k]):.4f}" for k in KS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--control", nargs="*", default=[])
    ap.add_argument("--skip-ensemble", action="store_true",
                    help="3번은 적합을 많이 돌린다 — 1·2·4 만 볼 때")
    a = ap.parse_args()
    runs = [Path("runs") / x for x in a.runs]
    ctl = [Path("runs") / x for x in a.control]
    missing = [d for d in runs + ctl if not d.exists()]
    if missing:
        raise SystemExit("없는 실행: " + ", ".join(str(x) for x in missing))
    obs1(runs)
    obs2_4(runs, ctl)
    if not a.skip_ensemble:
        obs3(runs, ctl)


if __name__ == "__main__":
    main()
