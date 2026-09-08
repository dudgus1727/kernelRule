"""★ The per-run aggregate — the D-75 requirement frequency and the D-78
branch constants. **0 LLM calls**.

    python3 experiments/d75_aggregate.py 'F3hg-p8-d75-b-s*'

**One sample = one run** (principle 28). The hypotheses and proposals of one
run come from the same seed and lineage, so they are not independent —
counting per proposal inflates n and reads one run's habit as a population
rate (D-79 retracted, D-80).

The baseline is the 6 `F3rw-p8-s*` runs — the same condition (F3, the human
24), the same model (`gpt-5.6-luna` medium), the same round band.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

BASELINE = "F3rw-p8-s*"
#: The old runs are 12 rounds. If a new run is 4 rounds, **only the same band**
#: is looked at (principle 4).
MAX_ROUND = 3
#: The per-run cap on optimize calls = rounds x 12.
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
            # ★ The 1st Analyst only. In the old runs there was one Analyst
            #   per round.
            if h.get("analyst_pass", 1) != 1 or h.get("round", 0) > MAX_ROUND:
                continue
            n += 1
            hit = bool(_requirement(h))
            k += hit
            if h.get("round", 0) <= 1:
                early[1] += 1
                early[0] += hit
        lit = dodge = m = 0
        for g in sorted((d / "llm_calls").glob("*rule_editor.json"))[:MAX_OPT]:
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
        raise SystemExit(f"there is no run matching {a.pattern}")

    print("=" * 76)
    print(f"the per-run aggregate (r0~r{MAX_ROUND}) — one sample = one run "
          f"(principle 28)")
    print("=" * 76)
    for lbl, rows in (("the baseline", old), ("the new condition", new)):
        print(f"\n{lbl}, {len(rows)} runs")
        print(f"  {'run':32s} {'req':>10} {'r0~r1':>9} "
              f"{'literal cmp':>12} {'dodge':>8}")
        for r in rows:
            e = r["early"]
            print(f"  {r['run']:32s} {r['k']:2d}/{r['n']:2d}={r['rate']:5.1%} "
                  f"{e[0]:2d}/{e[1]:2d}={(e[0]/e[1] if e[1] else 0):5.1%} "
                  f"{r['lit']:3d}/{r['m']:3d}={r['lit']/max(r['m'],1):5.1%} "
                  f"{r['dodge']:3d}={r['dodge']/max(r['m'],1):5.1%}")

    ro = [r["rate"] for r in old]
    rn = [r["rate"] for r in new]
    print("\n★ the requirement frequency — Mann-Whitney U (one-sided: new < "
          "old)")
    print(f"   old {[f'{x:.1%}' for x in ro]}")
    print(f"   new {[f'{x:.1%}' for x in rn]}")
    pv = mannwhitneyu(ro, rn, alternative="greater").pvalue
    floor = mannwhitneyu([1] * len(ro), [0] * len(rn),
                         alternative="greater").pvalue
    print(f"   p = {pv:.4f}   (the smallest possible p under complete "
          f"separation = {floor:.5f})")
    print("   " + ("★ it is still suppressed" if pv < 0.05 else
                   "★ indistinguishable from the baseline — that is not "
                   "'the same' (principle 27)"))

    print("\n★ the D-78 branch constants — per run (runs that used one at "
          "least once)")
    for key, lbl in (("lit", "literal comparison"),
                     ("dodge", "dodge isfinite/sign")):
        ao = sum(1 for r in old if r[key] > 0)
        an = sum(1 for r in new if r[key] > 0)
        p = fisher_exact([[ao, len(old) - ao], [an, len(new) - an]])[1]
        print(f"   {lbl:22s} old {ao}/{len(old)}  new {an}/{len(new)}  "
              f"Fisher p = {p:.4f}")


if __name__ == "__main__":
    main()
