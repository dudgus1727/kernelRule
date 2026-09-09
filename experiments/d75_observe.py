"""★ Observing a verification run — the D-75 path and the D-78 branch
constants. **0 LLM calls**.

    python3 experiments/d75_observe.py runs/x-probe-d78d75v1-s0

The pre-registration is the six items in
`docs/artifacts/d75-verification-prereg.md`.
**Performance is not looked at** — 2 rounds and 1 seed are inside the seed
spread.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

#: If it is in a requirement sentence, **table information has leaked**
#: (pre-registration item 5).
#: ⚠️ The Korean patterns stay. Checked 2026-09-09 (D-147): **136 of the 136**
#: d75 Analyst logs this script reads are Korean, so dropping them would make
#: the leak check pass by looking at nothing.
_LEAK = (
    (re.compile(r"사례\s*#?\d"), "a case number"),
    (re.compile(r"case\s*#?\d"), "a case number"),
    (re.compile(r"regret|리그렛"), "regret"),
    (re.compile(r"\b(1024|2048|4096|8192|11008|5120|13824)\b"), "a shape size"),
    (re.compile(r"홀드아웃|검증 분할|학습 분할"), "a split name"),
    (re.compile(r"holdout|validation split|training split"), "a split name"),
    (re.compile(r"\d+\s*번째 형상|형상 목록"), "a shape list"),
    (re.compile(r"the \d+(st|nd|rd|th) shape|shape list"), "a shape list"),
)

#: The forms that stand in for the constant 1 (what D-78 tried to remove).
_DODGE = ("np.sign(", "np.isfinite(", "np.sqrt(", "np.square(")


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _calls(d: Path) -> list[dict]:
    out = []
    for f in sorted((d / "llm_calls").glob("*.json")):
        out.append(json.loads(f.read_text()))
    return out


def _branch_constants(code: str) -> tuple[int, list]:
    """It counts the numeric literals that are direct operands of an
    `ast.Compare`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0, []
    vals = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            for side in [n.left, *n.comparators]:
                if (isinstance(side, ast.Constant)
                        and isinstance(side.value, (int, float))
                        and not isinstance(side.value, bool)):
                    vals.append(side.value)
    return len(vals), vals


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    d = Path(sys.argv[1])
    hyps = _jsonl(d / "hypotheses.jsonl")
    feats = _jsonl(d / "features.jsonl")
    arc = _jsonl(d / "archive.jsonl")
    calls = _calls(d)

    print("=" * 74)
    print(f"D-75 / D-78 verification observation — {d}")
    print("  ★ performance is not looked at (2 rounds, 1 seed is inside the "
          "seed spread)")
    print("=" * 74)

    # 1. requirement frequency ---------------------------------------------
    #  ★ It counts **the first Analyst response only**. In the old runs there
    #    was one Analyst per round, so mixing in a second, re-entered response
    #    changes the denominator (principle 4).
    p1 = [h for h in hyps if h.get("analyst_pass", 1) == 1]
    p2 = [h for h in hyps if h.get("analyst_pass") == 2]
    filled = [h for h in p1
              if (h.get("physical_requirement") or h.get("needs_new_feature"))]
    print(f"\n1. requirement frequency (the 1st Analyst only)  "
          f"{len(filled)}/{len(p1)} hypotheses"
          + (f" = {len(filled)/len(p1):.1%}" if p1 else ""))
    print("   the baseline 10/56 = 17.9%  (the same condition F3 · the same "
          "model · r0~r1, D-79)")
    if p1:
        from scipy.stats import fisher_exact
        pv = fisher_exact([[10, 46], [len(filled), len(p1) - len(filled)]])[1]
        print(f"   Fisher p = {pv:.4f}  "
              + ("★ indistinguishable from the baseline" if pv >= 0.05
                 else "★ significantly different"))
    print(f"   (the {len(p2)} hypotheses from the 2nd Analyst were excluded "
          f"here)")

    # Per round band — it can change as it goes deeper
    for lo, hi, lbl in ((0, 1, "r0~r1"), (2, 99, "r2 onward")):
        g = [h for h in p1 if lo <= h.get("round", 0) <= hi]
        k = sum(1 for h in g
                if (h.get("physical_requirement") or h.get("needs_new_feature")))
        if g:
            print(f"   {lbl:8s} {k}/{len(g)} = {k/len(g):5.1%}")

    # 3. the requirement contents + 5. is it physical language ------
    print(f"\n3+5. the requirement contents and the **leak check** "
          f"({len(filled)} of them)")
    print("   ⚠️ the old 303 had 0/303 leaks (D-79) — anything here is a new "
          "problem")
    n_leak = 0
    for h in filled:
        t = (h.get("physical_requirement") or h.get("needs_new_feature") or "")
        hits = [why for rx, why in _LEAK if rx.search(t)]
        n_leak += bool(hits)
        mark = f"★leak {hits}" if hits else "OK"
        print(f"   [{h.get('round','?')}] {mark}\n       {t}")
    print(f"\n   ★ leaks {n_leak}/{len(filled)}"
          + ("  — condition 1 becomes meaningless. It has to be fixed before "
             "the real run"
             if n_leak else "  — it came in physical language only"))

    # 2. the axes made and how much they are used --------------------------
    made = [f for f in feats if f.get("accepted")]
    print(f"\n2. axes made  {len(made)}/{len(feats)} attempts")
    best = min(arc, key=lambda e: e["regret"]) if arc else None
    for f in feats:
        if f.get("accepted"):
            used_best = bool(best) and f"f.{f['name']}" in best["code"]
            used_any = sum(1 for e in arc if f"f.{f['name']}" in e["code"])
            print(f"   ✓ r{f['round']} {f['name']}"
                  f"   in the best rule {'yes' if used_best else 'no'}"
                  f" / in the archive {used_any}/{len(arc)}")
        else:
            print(f"   ✗ r{f['round']} {str(f.get('error'))[:80]}")

    # 4. does the re-entered Analyst use the new axis ----------------------
    print("\n4. ★ does the re-entered Analyst mention the new axis")
    seq = [(c["seq"], c["role"], c["response"]) for c in calls]
    seq.sort()
    names_made = {f["round"]: f["name"] for f in feats if f.get("accepted")}
    seen_feature = 0
    for i, (s, role, _resp) in enumerate(seq):
        if role != "feature":
            continue
        rnd = sorted(names_made)[seen_feature] if seen_feature < len(names_made) \
            else None
        seen_feature += 1
        nxt = next((r for r in seq[i + 1:] if r[1] == "analyze"), None)
        if nxt is None:
            print(f"   seq{s}: there is no analyze after the feature — it did "
                  f"not go back")
            continue
        nm = names_made.get(rnd) if rnd is not None else None
        blob = json.dumps(nxt[2], ensure_ascii=False)
        hit = bool(nm) and nm in blob
        print(f"   seq{s} feature -> seq{nxt[0]} analyze : "
              f"{'★ mentioned' if hit else 'not mentioned'}  ({nm})")

    # over the cap ---------------------------------------------------------
    over = [f for f in feats if f.get("over_cap")]
    print(f"\n2b. requirements not made because of the per-round cap  "
          f"{len(over)}")
    for f in over:
        print(f"   r{f['round']}  {f['requirement'][:70]}")

    # 6. branch constants --------------------------------------------------
    print(f"\n6. branch constants (the {len(arc)} archive rules)")
    n_bc = sum(1 for e in arc if _branch_constants(e["code"])[0])
    vals: list = []
    for e in arc:
        vals += _branch_constants(e["code"])[1]
    print(f"   rules using a number in a comparison  {n_bc}/{len(arc)}   "
          f"values {sorted(set(vals))}")
    for pat in _DODGE:
        k = sum(1 for e in arc if pat in e["code"])
        print(f"   {pat:16s} {k}/{len(arc)}")
    print("   ⚠️ the old 330 rules are a different condition (a summed "
          "budget), so both N and the condition differ — it is a first "
          "impression, not a conclusion")

    # the call count -------------------------------------------------------
    roles: dict[str, int] = {}
    for c in calls:
        roles[c["role"]] = roles.get(c["role"], 0) + 1
    print(f"\n{len(calls)} calls  {roles}")


if __name__ == "__main__":
    main()
