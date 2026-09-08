"""★ Measuring diversity and assignment — it reads existing artefacts only.
**0 LLM calls**.

    python3 experiments/diversity_audit.py

It measures five things (§3-1, §4-3, §4-5).

```
1  do the Analyst hypotheses really name **different axes**  hypotheses.jsonl
2  how different are the 10 RuleWriter outputs               stage2-*/summary.json
3  changes x parent kind — does the duplicate rate follow     archive/rounds
   the parent kind
4  does cross really use two parents                         ★ it reads the code
5  is the hypothesis assignment skewed to the front          ★ it falls out of
                                                             the arithmetic
```

**It measures before any prescription is made** — if the overlap is small, a
diversity device is unnecessary.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

RUNS = Path("runs")


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


# ---------------------------------------------------------------- 1. Analyst
def analyst_overlap() -> None:
    """Do the hypotheses of the same round name the same axes?"""
    print("=" * 74)
    print("1. Analyst — how much the hypotheses of the same round overlap")
    print("=" * 74)
    feats, regimes, words, n_pairs = [], [], [], 0
    per_round: dict[tuple, list[dict]] = {}
    for f in sorted(RUNS.glob("*/hypotheses.jsonl")):
        for ln in f.read_text().splitlines():
            if not ln.strip():
                continue
            h = json.loads(ln)
            if h.get("analyst_pass", 1) != 1:
                continue
            per_round.setdefault((f.parent.name, h.get("round")), []).append(h)
    for hs in per_round.values():
        for a, b in combinations(hs, 2):
            n_pairs += 1
            feats.append(_jaccard(set(a.get("measurable_with") or []),
                                  set(b.get("measurable_with") or [])))
            regimes.append(
                (a.get("affected_regime") or "").strip()
                == (b.get("affected_regime") or "").strip() != "")
            # ★ The `가-힣` class stays: old `claim` texts are in Korean and
            #   must still be tokenised (D-146).
            wa = set(re.findall(r"[가-힣A-Za-z_]{2,}", a.get("claim", "")))
            wb = set(re.findall(r"[가-힣A-Za-z_]{2,}", b.get("claim", "")))
            words.append(_jaccard(wa, wb))
    import statistics as st
    print(f"  {len(per_round)} rounds, {n_pairs} hypothesis pairs")
    print(f"  measurable_with jaccard   median {st.median(feats):.2f}  "
          f"mean {st.mean(feats):.2f}  "
          f"exact match {sum(1 for x in feats if x == 1.0) / len(feats):.0%}")
    print(f"  affected_regime identical {sum(regimes) / n_pairs:.0%}")
    print(f"  claim vocabulary jaccard  median {st.median(words):.2f}")
    print("  ★ a large overlap means a diversity device is needed / a small "
          "one means what we have is enough")


# ------------------------------------------------------------ 2. RuleWriter
def rule_writer_spread() -> None:
    print("\n" + "=" * 74)
    print("2. RuleWriter — how different the 10 of one campaign are")
    print("=" * 74)
    import statistics as st
    for summ in sorted(RUNS.glob("*/stage2-rule-writer/summary.json")):
        d = json.loads(summ.read_text())
        ok = [t for t in d.get("tries", []) if t.get("ok") and t.get("code")]
        if len(ok) < 2:
            continue
        sets = [set(re.findall(r"\bf\.(\w+)", t["code"])) for t in ok]
        js = [_jaccard(a, b) for a, b in combinations(sets, 2)]
        nterm = [len(t.get("w0") or []) for t in ok]
        fits = sorted(t["fit_regret"] for t in ok)
        print(f"  {summ.parent.parent.name:26s} n={len(ok):2d}  "
              f"feature jaccard median {st.median(js):.2f}  "
              f"terms {min(nterm)}~{max(nterm)}  "
              f"score min {fits[0]:.4f} median {st.median(fits):.4f} "
              f"max {fits[-1]:.4f}")
    print("  ★ a high jaccard means 'the same thing 10 times' — the spread is "
          "only the width of luck")


# ------------------------------------------------- 3. changes x parent kind
def parent_kind_effect() -> None:
    """The duplicate rate per parent kind. `rounds.jsonl` does not record the
    kind, so it is read from the prompts in `llm_calls`
    (`round=r parent=kind`)."""
    print("\n" + "=" * 74)
    print("3. per parent kind — did the proposal actually reach the scoring")
    print("=" * 74)
    kinds: Counter = Counter()
    seen_by_kind: dict[str, Counter] = {}
    for d in sorted(RUNS.glob("*/llm_calls")):
        codes: dict[str, list[str]] = {}
        for g in sorted(d.glob("*rule_editor.json")):
            j = json.loads(g.read_text())
            m = re.search(r"parent=(\w+)", str(j.get("prompt", "")))
            k = m.group(1) if m else "?"
            kinds[k] += 1
            resp = j.get("response") or {}
            code = resp.get("code") if isinstance(resp, dict) else None
            if code:
                codes.setdefault(k, []).append(code.strip())
        for k, cs in codes.items():
            c = seen_by_kind.setdefault(k, Counter())
            c["n"] += len(cs)
            c["uniq"] += len(set(cs))
    if not kinds:
        print("  ★ the prompts are not kept in `llm_calls` — this axis cannot "
              "be measured.")
        print("     ⚠️ `dump()` does keep `prompt`, but it is empty in the "
              "old runs.")
        print("     Instead the **duplicate rate** per round is looked at "
              "(the parent kind cannot be separated).")
        for d in sorted(RUNS.glob("*/rounds.jsonl"))[:6]:
            R = [json.loads(x) for x in d.read_text().splitlines() if x.strip()]
            prop = sum(r["n_proposed"] for r in R)
            sc = sum(r["n_scored"] for r in R)
            print(f"     {d.parent.name:28s} scored {sc}/{prop} = "
                  f"{sc / prop:.0%}")
        return
    print(f"  {'parent kind':12s} {'props':>6} {'uniq':>6} {'uniq rate':>10}")
    for k, c in sorted(seen_by_kind.items()):
        print(f"  {k:12s} {c['n']:6d} {c['uniq']:6d} "
              f"{c['uniq'] / c['n']:10.0%}")
    print("  ★ a low unique rate for explore/cross means "
          "'hypothesis-parent mismatch'")


# ------------------------------------------------------------- 4. cross
def cross_uses_two_parents() -> None:
    print("\n" + "=" * 74)
    print("4. ★ does cross really use two parents — it reads the code")
    print("=" * 74)
    from kernelrule.core.archive import Archive
    from kernelrule.core.loop import RoundLoop

    src = inspect.getsource(Archive.parents)
    gives_two = "size=2" in src
    import textwrap
    body = textwrap.dedent(inspect.getsource(RoundLoop.run_round))
    tree = ast.parse(body)
    uses = sorted({ast.unparse(n) for n in ast.walk(tree)
                   if isinstance(n, ast.Subscript)
                   and ast.unparse(n).startswith("ps[")})
    print(f"  archive.parents gives cross two : {gives_two}")
    print(f"  what run_round actually uses    : {uses}")
    if gives_two and uses == ["ps[0]"]:
        print("  ★ the second parent is **thrown away**. cross is the same as "
              "explore.")
        print("     The prompt has only one parent slot too "
              "(`{parent_code}`).")
    print("  ⚠️ the archive's diversity has never been combined — it may be "
          "related to D-42 (the ensemble failure)")


# --------------------------------------------------------- 5. assignment bias
def assignment_bias() -> None:
    print("\n" + "=" * 74)
    print("5. hypothesis assignment — the arithmetic bias of the round robin")
    print("=" * 74)
    print("  hyps[i % len(hyps)],  12 parents")
    print(f"  {'n hyps':>7} {'assignment counts':>28} {'max/min':>9}")
    for n in range(2, 9):
        cnt = Counter(i % n for i in range(12))
        v = [cnt[k] for k in range(n)]
        print(f"  {n:7d} {str(v):>28} {max(v) / min(v):9.2f}")
    print("  ★ the earlier hypotheses get used more often. The **order** the "
          "Analyst produced becomes the assignment frequency — that is not "
          "the intent")
    print("  ⚠️ it is correlated with the parent kind too: i=0~5 exploit / "
          "6~8 explore / 9~11 cross")


def main() -> None:
    analyst_overlap()
    rule_writer_spread()
    parent_kind_effect()
    cross_uses_two_parents()
    assignment_bias()


if __name__ == "__main__":
    main()
