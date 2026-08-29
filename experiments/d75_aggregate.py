"""★ 실행 단위 집계 — D-75 요구 빈도와 D-78 분기 상수. **LLM 호출 0회**.

    python3 experiments/d75_aggregate.py 'f1pipe-F3-d75run3-s*'

**표본 하나 = 실행 하나다** (원칙 28). 한 실행의 가설/제안은 같은 씨앗과
계보에서 나오므로 독립이 아니다 — 제안 단위로 세면 n 이 부풀고 한 실행의
습관이 모집단 비율로 읽힌다 (D-79 철회, D-80).

기준선은 `f1pipe-F3-arch24-s*` 6실행이다 — 같은 조건(F3 사람24), 같은
모델(`gpt-5.6-luna` medium), 같은 라운드 구간.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

BASELINE = "f1pipe-F3-arch24-s*"
#: 옛 실행은 12라운드다. 새 실행이 4라운드면 **같은 구간만** 본다 (원칙 4).
MAX_ROUND = 3
#: 실행당 optimize 호출 상한 = 라운드 x 12.
MAX_OPT = (MAX_ROUND + 1) * 12
_DODGE = ("np.isfinite(", "np.sign(")


def _branch_consts(code: str) -> list:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [s.value for n in ast.walk(tree) if isinstance(n, ast.Compare)
            for s in [n.left, *n.comparators]
            if isinstance(s, ast.Constant)
            and isinstance(s.value, (int, float))
            and not isinstance(s.value, bool)]


def _requirement(h: dict) -> str:
    v = h.get("needs_new_feature") or h.get("physical_requirement")
    return str(v).strip() if v else ""


def per_run(pattern: str) -> list[dict]:
    out = []
    for d in sorted(Path("runs").glob(pattern)):
        hp = d / "hypotheses.jsonl"
        if not hp.exists():
            continue
        k = n = 0
        early = [0, 0]
        for ln in hp.read_text().splitlines():
            if not ln.strip():
                continue
            h = json.loads(ln)
            # ★ 1차 Analyst 만. 옛 실행은 라운드당 Analyst 가 한 번이었다.
            if h.get("analyst_pass", 1) != 1 or h.get("round", 0) > MAX_ROUND:
                continue
            n += 1
            hit = bool(_requirement(h))
            k += hit
            if h.get("round", 0) <= 1:
                early[1] += 1
                early[0] += hit
        lit = dodge = m = 0
        for g in sorted((d / "llm_calls").glob("*optimize.json"))[:MAX_OPT]:
            r = json.loads(g.read_text())["response"]
            code = (r or {}).get("code") if isinstance(r, dict) else None
            if not code:
                continue
            m += 1
            lit += bool(_branch_consts(code))
            dodge += any(x in code for x in _DODGE)
        out.append(dict(run=d.name, k=k, n=n, rate=(k / n if n else 0.0),
                        early=early, lit=lit, dodge=dodge, m=m))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--baseline", default=BASELINE)
    a = ap.parse_args()

    from scipy.stats import fisher_exact, mannwhitneyu

    old, new = per_run(a.baseline), per_run(a.pattern)
    if not new:
        raise SystemExit(f"{a.pattern} 에 맞는 실행이 없다")

    print("=" * 76)
    print(f"실행 단위 집계 (r0~r{MAX_ROUND}) — 표본 하나 = 실행 하나 (원칙 28)")
    print("=" * 76)
    for lbl, rows in (("기준선", old), ("새 조건", new)):
        print(f"\n{lbl} {len(rows)}실행")
        print(f"  {'실행':32s} {'요구':>10} {'r0~r1':>9} "
              f"{'리터럴비교':>10} {'우회':>8}")
        for r in rows:
            e = r["early"]
            print(f"  {r['run']:32s} {r['k']:2d}/{r['n']:2d}={r['rate']:5.1%} "
                  f"{e[0]:2d}/{e[1]:2d}={(e[0]/e[1] if e[1] else 0):5.1%} "
                  f"{r['lit']:3d}/{r['m']:3d}={r['lit']/max(r['m'],1):5.1%} "
                  f"{r['dodge']:3d}={r['dodge']/max(r['m'],1):5.1%}")

    ro = [r["rate"] for r in old]
    rn = [r["rate"] for r in new]
    print("\n★ 요구 빈도 — Mann-Whitney U (단측: 새 < 옛)")
    print(f"   옛 {[f'{x:.1%}' for x in ro]}")
    print(f"   새 {[f'{x:.1%}' for x in rn]}")
    pv = mannwhitneyu(ro, rn, alternative="greater").pvalue
    floor = mannwhitneyu([1] * len(ro), [0] * len(rn),
                         alternative="greater").pvalue
    print(f"   p = {pv:.4f}   (완전 분리 시 최소 가능 p = {floor:.5f})")
    print("   " + ("★ 여전히 눌린다" if pv < 0.05 else
                   "★ 기준선과 구분 불가 — '같다' 가 아니다 (원칙 27)"))

    print("\n★ D-78 분기 상수 — 실행 단위 (한 번이라도 쓴 실행)")
    for key, lbl in (("lit", "리터럴 비교"), ("dodge", "우회 isfinite/sign")):
        ao = sum(1 for r in old if r[key] > 0)
        an = sum(1 for r in new if r[key] > 0)
        p = fisher_exact([[ao, len(old) - ao], [an, len(new) - an]])[1]
        print(f"   {lbl:20s} 옛 {ao}/{len(old)}  새 {an}/{len(new)}  "
              f"Fisher p = {p:.4f}")


if __name__ == "__main__":
    main()
