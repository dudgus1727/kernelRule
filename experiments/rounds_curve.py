"""★ 12라운드가 맞나 — **기록된 곡선만** 본다. LLM 0회.

    python3 experiments/rounds_curve.py

사전 등록 `docs/artifacts/rounds-prereg.md`.

"조기 종료가 안 걸렸다" 는 "수렴했다" 가 아니다 — `patience=10` 에
12라운드면 판정 창이 둘뿐이고, 새 셀이 하나만 생겨도 안 멈춘다.

★ 유의 문턱은 루프가 쓰는 것 그대로다 (`is_significant`). 새 기준을
만들지 않는다 (원칙 2).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from two_stage import A6000, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate_scores
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY

#: (새 태그, 옛 디렉토리 접두, 시드 수)
#: (태그, 실행 디렉토리 접두, 시드 수). D-128 개명 뒤 둘이 같다.
GROUPS = [("F3rw-p8", "F3rw-p8", 6),
          ("F1rw-p8", "F1rw-p8", 6),
          ("F2rw-p8", "F2rw-p8", 6)]
#: 검토할 patience. **바꾸지 않는다** — "그랬다면 언제 멈췄을까" 만 본다.
PATIENCES = (3, 5, 7, 10)
#: 판정에 쓰는 "마지막 3라운드" (0부터 세는 파일의 round 필드)
LAST3 = (9, 10, 11)


def _r(x) -> str:
    """`None` 이면 '없음'. 라운드 번호는 0부터다."""
    return "없음" if x is None else f"r{x}"


def _rows(run: str) -> list[dict]:
    f = Path("runs") / run / "rounds.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()]


def _tol(run: str, T, M, hold) -> float | None:
    """그 실행의 최종 최고 규칙으로 잰 **노이즈 문턱** (루프와 같은 경로).

    ★ `None` 이면 **계산할 수 없었다** — F1/F2 실행의 규칙은 루프 안에서
    만든 피처를 참조하는데 기본 레지스트리에 그 축이 없다. 다른 값으로
    메우지 않는다 (원칙 2). 부르는 쪽이 근사임을 표시하고 쓴다.
    """
    f = Path("runs") / run / "archive.jsonl"
    arc = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    e = sorted(arc, key=lambda z: z["regret"])[0]
    try:
        fn = compile_rule(e["code"])
        ev = evaluate_scores(make_score_of(fn, M, np.asarray(e["w"], float)),
                             T, list(hold), ks=(1,))
    except AttributeError:
        return None
    from kernelrule.core.scoring import geomean
    return float(geomean(ev.tol))


def _stop_round(vals: list[float], n: int, thr: float,
                cells: list[int]) -> int | None:
    """`patience=n` 이면 **몇 번째 라운드 끝에** 멈췄을까 (루프의 공식).

    ★ 새 셀 조건도 그대로 쓴다 — 그것 때문에 안 멈추는 경우가 요점이다.
    """
    for end in range(n, len(vals)):
        improved = vals[end - n] - vals[end]
        # 새 셀이 최근 n 라운드 안에 생겼나
        new_cell = any(cells[i] > cells[i - 1]
                       for i in range(max(1, end - n + 1), end + 1))
        if abs(improved) > thr or new_cell:
            continue
        return end
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/rounds-curve.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    hold = list(_splits(T).val.shapes)
    out: dict = {"last3": list(LAST3), "patiences": list(PATIENCES),
                 "groups": {}}

    for tag, prefix, n_seeds in GROUPS:
        print("=" * 96)
        print(f"{tag}  (구 {prefix})  — 시드 {n_seeds}")
        print("=" * 96)
        g: dict = {}
        for i in range(n_seeds):
            run = f"{prefix}-s{i}"
            rr = _rows(run)
            vals = [x["best_val_regret"] for x in rr]
            cells = [x["n_cells"] for x in rr]
            thr = _tol(run, T, M, hold)
            approx = thr is None
            if approx:
                # ★ F3 6시드가 **전부 같은 문턱**(0.0047)을 냈다 — 문턱은
                #   규칙이 아니라 홀드아웃 형상이 정한다. 그 값을 빌려 쓰되
                #   **근사라고 표시한다**. 판정은 F3 로만 한다 (사전 등록 §2).
                thr = out.get("_f3_thr", float("nan"))
            # 라운드마다의 개선 (양수 = 좋아졌다)
            d = [vals[k - 1] - vals[k] for k in range(1, len(vals))]
            any_imp = [k for k, x in enumerate(d, start=1) if x > 0]
            sig_imp = [k for k, x in enumerate(d, start=1) if x > thr]
            new_cell = [k for k in range(1, len(cells))
                        if cells[k] > cells[k - 1]]
            stops = {n: _stop_round(vals, n, thr, cells) for n in PATIENCES}
            if tag == "F3rw-p8":
                out["_f3_thr"] = thr
            g[run] = {"vals": vals, "cells": cells, "thr": thr,
                      "thr_is_approx": approx,
                      "last_any": (max(any_imp) if any_imp else None),
                      "last_sig": (max(sig_imp) if sig_imp else None),
                      "last_new_cell": (max(new_cell) if new_cell else None),
                      "sig_in_last3": [k for k in sig_imp if k in LAST3],
                      "stops": stops}
            print(f"  {run:28s} 문턱 {thr:.4f}{'(근사)' if approx else '     '}  "
                  f"마지막 개선 r{g[run]['last_any']}  "
                  f"★ 마지막 **유의** 개선 "
                  f"{_r(g[run]['last_sig'])}  "
                  f"마지막 새 셀 r{g[run]['last_new_cell']}  "
                  f"멈춤(p3/p4/p10) "
                  + "/".join(str(stops[n]) if stops[n] is not None else "-"
                             for n in PATIENCES))
            print(f"  {'':28s} 곡선 "
                  + " ".join(f"{v:.4f}" for v in vals))
        out["groups"][tag] = g

        sig = [v["last_sig"] for v in g.values()]
        in3 = [r for r, v in g.items() if v["sig_in_last3"]]
        print("\n  마지막 유의 개선 라운드: "
              + ", ".join("없음" if s is None else f"r{s}" for s in sig))
        print(f"  ★ 마지막 3라운드(r9·r10·r11)에 유의 개선이 있는 시드: "
              f"{len(in3)}/{len(g)}  {in3}")
        cells_end = [v["last_new_cell"] for v in g.values()]
        print("  마지막 새 셀 라운드: "
              + ", ".join("없음" if c is None else f"r{c}" for c in cells_end))
        print()

    # ------------------------------------------------------------ 판정
    main_g = out["groups"]["F3rw-p8"]
    in3 = [r for r, v in main_g.items() if v["sig_in_last3"]]
    late_cell = [r for r, v in main_g.items()
                 if v["last_new_cell"] is not None and v["last_new_cell"] >= 9]
    # ------------------------------------------------ patience 고르기 (D-129)
    import statistics as _st
    print("=" * 96)
    print("★ patience 별 — 가정 종료 라운드와 **놓칠 개선** "
          "(사전 등록 patience-prereg.md)")
    print("=" * 96)
    pat: dict = {}
    for tag, g in out["groups"].items():
        print(f"\n  {tag}")
        for n in PATIENCES:
            miss, stops = [], []
            for v in g.values():
                e = v["stops"][n]
                stops.append(e)
                # ★ 그때 멈췄으면 잃었을 양. 끝까지 간 시드는 0 이다
                miss.append(0.0 if e is None else v["vals"][e] - v["vals"][-1])
            early = sum(1 for e in stops if e is not None)
            med, mx = _st.median(miss), max(miss)
            pat.setdefault(tag, {})[n] = {
                "stops": stops, "miss": miss, "median": med, "max": mx,
                "n_early": early}
            print(f"    patience {n:2d}  종료 "
                  + " ".join("-" if e is None else f"r{e:<2d}" for e in stops)
                  + f"   12 전 종료 {early}/{len(stops)}"
                  + f"   ★ 놓칠 개선 중앙 {med:+.4f}  최대 {mx:+.4f}")
        cum = [v["vals"][6] - v["vals"][-1] for v in g.values()]
        print(f"    ★ r6 -> r11 누적 개선  중앙 {_st.median(cum):+.4f}  "
              + " ".join(f"{c:+.4f}" for c in cum))
        pat[tag]["cum_r6_r11"] = cum
    out["patience"] = pat

    SIGMA = 0.0124
    main = pat["F3rw-p8"]
    ok = [n for n in PATIENCES if main[n]["median"] < SIGMA]
    pick = min(ok) if ok else max(PATIENCES)
    print(f"\n  ★ 고른 patience = {pick}  "
          + (f"(놓칠 개선 중앙 {main[pick]['median']:+.4f} < σ {SIGMA})"
             if ok else
             f"— ⚠️ 어느 값도 σ {SIGMA} 아래가 아니다. 큰 쪽으로 간다"))
    out["patience_pick"] = pick

    print()
    print("=" * 96)
    print("★ 판정 — 사전 등록 §2 의 셋 중에서")
    print("=" * 96)
    if in3:
        verdict = "(나) 12 가 부족하다"
        print(f"  ★ (나) — 마지막 3라운드에 유의 개선이 있는 시드 {in3}")
        print("     -> 라운드 24 를 n=6 으로 재야 한다 (약 6,000호출 / 15시간)")
    elif late_cell:
        verdict = "(다) 애매하다 — 유의 개선은 끝났는데 새 셀이 늦게까지 생긴다"
        print("  ★ (다) — 유의 개선은 r8 이하에서 끝났는데 새 셀이 "
              f"r9 이후에도 생긴다: {late_cell}")
        print("     -> patience 조정을 **검토**한다. ⚠️ 여기서 바꾸지 않는다 "
              "(조건 변경이라 별도 사전 등록)")
    else:
        verdict = "(가) 12 로 충분하다"
        print("  ★ (가) — 6시드 전부 마지막 유의 개선이 r8 이하이고 "
              "새 셀도 r9 이후 없다")
        print("     -> 라운드는 축이 아니다. 각주에 곡선과 함께 적는다")
    out["verdict"] = verdict
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
