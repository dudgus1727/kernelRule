"""★ 벽이 예산 탓인가 — 두 목적함수가 쓰는 축이 겹치나. LLM 0회.

    python3 experiments/feature_overlap.py

## 묻는 것

`regret` 진화 규칙과 순위 손실 진화 규칙이 **같은 축을 쓰는가.**

    합집합 <= 8    8 로 둘 다 담을 수 있다 -> 벽이 예산 탓이 아니다
    합집합 >  8    예산이 부족하다 -> 16 이 뚫을 수 있다

## ⚠️ 축 개수는 예산 단위가 **아니다**

예산은 `숫자 리터럴 + 가중치 개수` 를 센다 (`rules/checks.py`). 한 항이
축 두 개를 쓸 수 있다 —

    np.where(p.is_memory_bound, f.log_dram_traffic, f.log_inst_total) * w[4]

축 3개(피처 2 + 술어 1)를 가중치 1개로 담는다. 그래서 **축 합집합이 8을
넘어도 8항에 들어갈 수 있다.** 판정선은 "넉넉히 담기나" 의 대리 지표로만
읽고, 항 수도 같이 센다.

## ⚠️ 자카드는 바닥이 있어야 읽힌다 (원칙 7)

"두 계열이 0.5 만큼 겹친다" 는 그 자체로 크지도 작지도 않다. 세 가지를
같이 낸다: 계열 **안**의 자카드(같은 목적함수끼리 얼마나 같은가),
계열 **사이**의 자카드, 그리고 **무작위 바닥**(같은 크기의 집합을
레지스트리에서 아무렇게나 뽑았을 때).
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from itertools import combinations, product
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.features import REGISTRY

REG_RUNS = [f"f1pipe-F3-arch24-s{i}" for i in range(6)]
RANK_RUNS = [f"f1pipe-F3-rankevo-s{i}" for i in range(3)]
#: 상한 측정에서 상위 100 안에서 크게 변하던 것들 (ranking-ceiling.md §3).
WATCH = ("split_k_cost", "sm_idle_cost", "pipeline_warmup_frac",
         "tail_waste", "waves")
N_DRAWS = 2000


def _best(run: str, by: str) -> dict:
    arc = [json.loads(x) for x in
           (Path("runs") / run / "archive.jsonl").read_text().splitlines()
           if x.strip()]
    key = (lambda e: e.get("rank_loss", 1e9)) if by == "rank" \
        else (lambda e: e["regret"])
    return sorted(arc, key=key)[0]


def _feats(code: str) -> set[str]:
    """`f.<이름>` 만 센다 — `p.<이름>` (술어)은 따로."""
    return {n.attr for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "f"}


def _preds(code: str) -> set[str]:
    return {n.attr for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "p"}


def _n_terms(code: str) -> int:
    return max([n.slice.value for n in ast.walk(ast.parse(code))
                if isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id == "w"
                and isinstance(n.slice, ast.Constant)] + [-1]) + 1


def _jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def _floor(sizes_a, sizes_b, pool: int, rng) -> tuple[float, float]:
    """★ 무작위 바닥 — 같은 크기를 레지스트리에서 아무렇게나 뽑는다."""
    js, us = [], []
    for _ in range(N_DRAWS):
        na = int(rng.choice(sizes_a))
        nb = int(rng.choice(sizes_b))
        a = set(rng.choice(pool, size=na, replace=False))
        b = set(rng.choice(pool, size=nb, replace=False))
        js.append(_jac(a, b))
        us.append(len(a | b))
    return float(np.mean(js)), float(np.mean(us))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/feature-overlap.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)

    fam = {"regret": [(r, _best(r, "regret")) for r in REG_RUNS],
           "rank": [(r, _best(r, "rank")) for r in RANK_RUNS]}
    F = {k: [_feats(e["code"]) for _, e in v] for k, v in fam.items()}
    P = {k: [_preds(e["code"]) for _, e in v] for k, v in fam.items()}
    TERMS = {k: [_n_terms(e["code"]) for _, e in v] for k, v in fam.items()}
    pool = len(REGISTRY.names()) if hasattr(REGISTRY, "names") \
        else len(list(REGISTRY))

    print("=" * 80)
    print("두 목적함수가 쓰는 축 — 벽이 예산 탓인가")
    print("=" * 80)
    print(f"  레지스트리 {pool}개 축 (F3)")
    print(f"  regret 진화 {len(REG_RUNS)}실행 / 순위 진화 "
          f"{len(RANK_RUNS)}실행 — 각 실행의 **최종 최고 규칙 하나**\n")

    print(f"  {'실행':22s} {'항':>3} {'축':>3} {'술어':>4}  축 이름")
    for k in ("regret", "rank"):
        for (r, _), f, p, t in zip(fam[k], F[k], P[k], TERMS[k], strict=True):
            print(f"  {r.replace('f1pipe-F3-', ''):22s} {t:3d} {len(f):3d} "
                  f"{len(p):4d}  {', '.join(sorted(f))}")
        print()

    uni = {k: set().union(*F[k]) for k in F}
    both = uni["regret"] | uni["rank"]
    print("  계열 합집합")
    print(f"    regret {len(uni['regret']):2d}개  {', '.join(sorted(uni['regret']))}")
    print(f"    순위   {len(uni['rank']):2d}개  {', '.join(sorted(uni['rank']))}")
    print(f"    ★ 둘 다 {len(both):2d}개   자카드 {_jac(uni['regret'], uni['rank']):.3f}")
    print(f"    순위에만 {sorted(uni['rank'] - uni['regret'])}")
    print(f"    regret 에만 {sorted(uni['regret'] - uni['rank'])}")
    print("    ⚠️ 실행 수가 6 대 3 이라 합집합 크기는 나란히 못 놓는다 —")
    print("       아래 **규칙 쌍**으로 본다.")

    print("\n" + "=" * 80)
    print("규칙 쌍 자카드 — ★ 바닥과 계열 안 값을 같이 놓는다 (원칙 7)")
    print("=" * 80)
    rows = {}
    for lab, pairs in (
            ("regret 안", list(combinations(range(len(F["regret"])), 2))),
            ("순위 안", list(combinations(range(len(F["rank"])), 2))),
            ("★ 계열 사이", None)):
        if pairs is None:
            ps = [(F["regret"][i], F["rank"][j]) for i, j
                  in product(range(len(F["regret"])), range(len(F["rank"])))]
        else:
            key = "regret" if "regret" in lab else "rank"
            ps = [(F[key][i], F[key][j]) for i, j in pairs]
        js = [_jac(x, y) for x, y in ps]
        us = [len(x | y) for x, y in ps]
        rows[lab] = {"jaccard": js, "union": us}
        print(f"  {lab:14s} n={len(js):3d}  자카드 중앙 {np.median(js):.3f} "
              f"({min(js):.3f}~{max(js):.3f})   합집합 중앙 "
              f"{np.median(us):.1f} ({min(us)}~{max(us)})")
    fj, fu = _floor(([len(x) for x in F["regret"]]),
                    ([len(x) for x in F["rank"]]), pool, rng)
    print(f"  {'★ 무작위 바닥':14s} n={N_DRAWS}  자카드 평균 {fj:.3f}"
          f"                    합집합 평균 {fu:.1f}")

    print("\n" + "=" * 80)
    print("판정 — 지시에 박은 선 (합집합 <= 8 이면 예산 탓이 아니다)")
    print("=" * 80)
    u = rows["★ 계열 사이"]["union"]
    med, n_le = float(np.median(u)), sum(1 for x in u if x <= 8)
    print(f"  계열 사이 축 합집합 중앙 {med:.1f}  ({n_le}/{len(u)} 쌍이 8 이하)")
    print("  -> " + ("★ 8 로 둘 다 담을 수 있다 — 벽이 예산 탓이 아니다"
                     if med <= 8 else
                     "★ 축이 8을 넘는다 — 예산이 뚫을 여지가 있다"))
    # ★ 판정선을 예산 단위로 옮긴다. 실제 규칙은 한 항에 축을 여럿
    #   담는다 — 그 밀도로 나눠야 "몇 항이 필요한가" 가 나온다.
    allf = F["regret"] + F["rank"]
    allp = P["regret"] + P["rank"]
    allt = TERMS["regret"] + TERMS["rank"]
    dens = [(len(f) + len(q)) / t for f, q, t
            in zip(allf, allp, allt, strict=True)]
    ub = [len(x | y) + len(p | q) for (x, p), (y, q) in product(
        zip(F["regret"], P["regret"], strict=True),
        zip(F["rank"], P["rank"], strict=True))]
    need = [u / d for u, d in product(ub, [float(np.median(dens))])]
    print("\n  ⚠️ 축 개수 != 예산 단위. 항 수는 전부 8 이고 한 항이 축을")
    print(f"     {min(dens):.2f}~{max(dens):.2f}개 (중앙 {np.median(dens):.2f}) "
          f"담는다 — 술어까지 센 값이다.")
    print(f"     그 밀도로 계열 사이 합집합(축+술어 {int(min(ub))}~{int(max(ub))})을 "
          f"담으려면 **{np.median(need):.1f}항** 이 필요하다.")
    print(f"     -> 8 은 {'모자란다' if np.median(need) > 8 else '넉넉하다'}, "
          f"16 은 {'넉넉하다' if np.median(need) <= 16 else '모자란다'}. "
          f"부족분은 약 {max(0.0, np.median(need) - 8):.1f}항이다.")

    print("\n  ★ 상한 측정의 다섯 축이 **regret 규칙에도** 있는가")
    print(f"    {'':24s} {'regret':>8} {'순위':>6}")
    watch = {}
    for n in WATCH:
        cr = sum(n in f for f in F["regret"])
        ck = sum(n in f for f in F["rank"])
        watch[n] = [cr, ck]
        print(f"    {n:24s} {cr:5d}/{len(F['regret'])} {ck:4d}/{len(F['rank'])}")

    Path(a.out).write_text(json.dumps(
        {"features": {k: [sorted(x) for x in v] for k, v in F.items()},
         "preds": {k: [sorted(x) for x in v] for k, v in P.items()},
         "terms": TERMS, "union": {k: sorted(v) for k, v in uni.items()},
         "pairs": rows, "floor": {"jaccard": fj, "union": fu},
         "watch": watch, "pool": pool,
         "density": dens, "union_both": ub, "terms_needed": need}, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
