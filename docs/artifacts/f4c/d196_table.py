"""D-196 결과표 — 1단계 (GPU 마다 새로 키움) · 2단계 (전이 12 쌍). LLM 0 · GPU 0.
미리 정한 기준 (docs/decisions.md D-196) 그대로 읽기만 한다. 없는 파일은 '-'.

    python3 docs/artifacts/f4c/d196_table.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

D = Path(__file__).resolve().parent
GPUS = ("a6000", "5090", "4090", "h100")
#: 1단계 판정 — K1a_G + 0.02 (a6000 은 D-195 단계 2 의 기준)
K1A = {"a6000": (1.0437, 1.0479), "5090": (1.0429, 1.0354),
       "4090": (1.0384, 1.0380), "h100": (1.0615, 1.0635)}
BAR1 = {g: round(K1A[g][0] + 0.02, 4) for g in GPUS}


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def native(g: str, cap: bool) -> dict | None:
    name = "refit_cap1.json" if cap else "refit_nocap.json"
    return _load(D / name if g == "a6000" else D / g / name)


def transfer(s: str, t: str, kind: str) -> dict | None:
    return _load(D / "transfer" / f"{s}_to_{t}.{kind}.json")


def n_bad(r: dict, bar: float = 1.2) -> int:
    return sum(v > bar for f in r["per_fold"].values()
               for v in f.get("regret", {}).values())


def n_hold(r: dict) -> int:
    return sum(len(f.get("regret", {})) for f in r["per_fold"].values())


def weights(r: dict) -> str:
    return "/".join(str(f["check"]["n_weights"]) for f in r["per_fold"].values())


def worst(r: dict, k: int = 3) -> str:
    v = sorted(((x, f, s) for f, fr in r["per_fold"].items()
                for s, x in fr.get("regret", {}).items()), reverse=True)[:k]
    return ", ".join(f"f{f} {s} {x:.2f}" for x, f, s in v)


def fmt(x: float | None) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) \
        else f"{x:.4f}"


def stage1() -> None:
    print("1단계 — 새로 키움 (같은 절차 재적합). 판정: 상한 1 의 inner-CV <= K1a_G + 0.02")
    print(f"{'GPU':6s} {'inner(상한1)':>11s} {'기준':>7s} {'판정':4s} {'holdout':>8s} "
          f"{'>1.2':>5s} {'inner(없음)':>11s} {'holdout':>8s} {'>1.2':>5s} "
          f"{'벤더':>7s} {'K1a in/ho':>15s}  w (fold)")
    for g in GPUS:
        c, n = native(g, True), native(g, False)
        if c is None:
            print(f"{g:6s} -")
            continue
        a = c["aggregate"]
        ok = "통과" if a["inner_cv_gm"] <= BAR1[g] else "실패"
        na = n["aggregate"] if n else {}
        print(f"{g:6s} {a['inner_cv_gm']:11.4f} {BAR1[g]:7.4f} {ok:4s} "
              f"{a['holdout_gm_65']:8.4f} {n_bad(c):5d} "
              f"{fmt(na.get('inner_cv_gm')):>11s} {fmt(na.get('holdout_gm_65')):>8s} "
              f"{(n_bad(n) if n else '-'):>5} {a['vendor_gm_65']:7.4f} "
              f"{K1A[g][0]:.4f}/{K1A[g][1]:.4f}  {weights(c)}")
    print()
    for g in GPUS:
        c = native(g, True)
        if c:
            a = c["aggregate"]
            print(f"  {g:6s} holdout fold {a['holdout_by_fold']}  분류 {a['holdout_by_class']}")
            print(f"  {'':6s} 가장 나쁜 holdout 형상: {worst(c)}")


def stage2() -> None:
    print("\n2단계 — 전이. 판정 (b): inner-CV_b(S->T) <= inner-CV_c(T) + 0.02 · 관찰 (a): holdout < 벤더_T")
    print(f"{'S -> T':14s} {'(a) holdout':>11s} {'>1.2':>5s} {'벤더_T':>7s} {'(a)<벤더':>8s} "
          f"{'(b) inner':>9s} {'기준':>7s} {'판정':4s} {'(b) holdout':>11s} {'>1.2':>5s} "
          f"{'(c) holdout':>11s}")
    n_pass = n_done = n_a = n_a_done = 0
    for s in GPUS:
        for t in GPUS:
            if s == t:
                continue
            a, b, c = (transfer(s, t, "nofit"), transfer(s, t, "refit_cap1"),
                       native(t, True))
            aa = a["aggregate"] if a else {}
            ba = b["aggregate"] if b else {}
            ca = c["aggregate"] if c else {}
            bar = ca["inner_cv_gm"] + 0.02 if c else None
            if b and c:
                ok = ba["inner_cv_gm"] <= bar
                n_pass += ok
                n_done += 1
                verdict = "통과" if ok else "실패"
            else:
                verdict = "-"
            if a:
                below = aa["holdout_gm_65"] < aa["vendor_gm_65"]
                n_a += below
                n_a_done += 1
            print(f"{s + ' -> ' + t:14s} {fmt(aa.get('holdout_gm_65')):>11s} "
                  f"{(n_bad(a) if a else '-'):>5} {fmt(aa.get('vendor_gm_65')):>7s} "
                  f"{('예' if a and below else '아니오' if a else '-'):>8s} "
                  f"{fmt(ba.get('inner_cv_gm')):>9s} {fmt(bar):>7s} {verdict:4s} "
                  f"{fmt(ba.get('holdout_gm_65')):>11s} {(n_bad(b) if b else '-'):>5} "
                  f"{fmt(ca.get('holdout_gm_65')):>11s}")
    print(f"\n  (b) 전이된다 {n_pass}/{n_done} 쌍 (끝난 것 중) · (a) 벤더보다 낮다 {n_a}/{n_a_done} 쌍")


if __name__ == "__main__":
    stage1()
    stage2()
