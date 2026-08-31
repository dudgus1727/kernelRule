"""★ 다양성·배정 실측 — 기존 산출물만 읽는다. **LLM 호출 0회**.

    python3 experiments/diversity_audit.py

다섯 가지를 잰다 (§3-1, §4-3, §4-5).

```
1  Analyst 가설이 실제로 **다른 축**을 말하나       hypotheses.jsonl
2  RuleWriter 10개가 얼마나 다른가                stage2-*/summary.json
3  changes x 부모 종류 — 중복률이 부모 종류를 따르나  archive/rounds
4  cross 가 실제로 두 부모를 쓰나                  ★ 코드를 읽는다
5  가설 배정이 앞쪽에 치우치나                     ★ 산술로 나온다
```

**처방을 만들기 전에 잰다** — 겹침이 작으면 다양성 장치가 불필요하다.
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
    """같은 라운드의 가설들이 같은 축을 말하는가."""
    print("=" * 74)
    print("1. Analyst — 같은 라운드 가설끼리 얼마나 겹치나")
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
            wa = set(re.findall(r"[가-힣A-Za-z_]{2,}", a.get("claim", "")))
            wb = set(re.findall(r"[가-힣A-Za-z_]{2,}", b.get("claim", "")))
            words.append(_jaccard(wa, wb))
    import statistics as st
    print(f"  라운드 {len(per_round)}개, 가설 쌍 {n_pairs}개")
    print(f"  measurable_with 자카드   중앙 {st.median(feats):.2f}  "
          f"평균 {st.mean(feats):.2f}  "
          f"완전 일치 {sum(1 for x in feats if x == 1.0) / len(feats):.0%}")
    print(f"  affected_regime 동일     {sum(regimes) / n_pairs:.0%}")
    print(f"  claim 어휘 자카드        중앙 {st.median(words):.2f}")
    print("  ★ 겹침이 크면 다양성 장치가 필요하다 / 작으면 지금으로 충분")


# ------------------------------------------------------------ 2. RuleWriter
def rule_writer_spread() -> None:
    print("\n" + "=" * 74)
    print("2. RuleWriter — 한 캠페인의 10개가 얼마나 다른가")
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
              f"피처 자카드 중앙 {st.median(js):.2f}  "
              f"항 {min(nterm)}~{max(nterm)}  "
              f"점수 최소 {fits[0]:.4f} 중앙 {st.median(fits):.4f} "
              f"최대 {fits[-1]:.4f}")
    print("  ★ 자카드가 높으면 '같은 것을 10번' 이다 — 격차는 운의 폭일 뿐")


# ------------------------------------------------- 3. changes x 부모 종류
def parent_kind_effect() -> None:
    """부모 종류별 중복률. `rounds.jsonl` 은 종류를 안 남기므로
    `llm_calls` 의 프롬프트에서 읽는다 (`round=r parent=kind`)."""
    print("\n" + "=" * 74)
    print("3. 부모 종류별 — 제안이 실제로 채점까지 갔나")
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
        print("  ★ `llm_calls` 에 프롬프트가 안 남아 있다 — 이 축으로는 못 잰다.")
        print("     ⚠️ `dump()` 가 `prompt` 를 남기지만 옛 실행에는 비어 있다.")
        print("     대신 라운드별 **중복률**을 본다 (부모 종류는 못 가른다).")
        for d in sorted(RUNS.glob("*/rounds.jsonl"))[:6]:
            R = [json.loads(x) for x in d.read_text().splitlines() if x.strip()]
            prop = sum(r["n_proposed"] for r in R)
            sc = sum(r["n_scored"] for r in R)
            print(f"     {d.parent.name:28s} 채점 {sc}/{prop} = {sc / prop:.0%}")
        return
    print(f"  {'부모 종류':10s} {'제안':>6} {'고유':>6} {'고유율':>7}")
    for k, c in sorted(seen_by_kind.items()):
        print(f"  {k:10s} {c['n']:6d} {c['uniq']:6d} {c['uniq'] / c['n']:7.0%}")
    print("  ★ explore/cross 의 고유율이 낮으면 '가설-부모 불일치' 다")


# ------------------------------------------------------------- 4. cross
def cross_uses_two_parents() -> None:
    print("\n" + "=" * 74)
    print("4. ★ cross 가 실제로 두 부모를 쓰나 — 코드를 읽는다")
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
    print(f"  archive.parents 가 cross 에 둘을 준다 : {gives_two}")
    print(f"  run_round 가 실제로 쓰는 것           : {uses}")
    if gives_two and uses == ["ps[0]"]:
        print("  ★ 두 번째 부모가 **버려진다**. cross 는 explore 와 같다.")
        print("     프롬프트에도 부모 자리가 하나뿐이다 (`{parent_code}`).")
    print("  ⚠️ 아카이브의 다양성이 결합된 적이 없다 — D-42(앙상블 실패)와 "
          "관련 있을 수 있다")


# --------------------------------------------------------- 5. 배정 편중
def assignment_bias() -> None:
    print("\n" + "=" * 74)
    print("5. 가설 배정 — 라운드로빈의 산술적 편중")
    print("=" * 74)
    print("  hyps[i % len(hyps)],  부모 12개")
    print(f"  {'가설 수':>6} {'배정 횟수':>28} {'최대/최소':>9}")
    for n in range(2, 9):
        cnt = Counter(i % n for i in range(12))
        v = [cnt[k] for k in range(n)]
        print(f"  {n:6d} {str(v):>28} {max(v) / min(v):9.2f}")
    print("  ★ 앞쪽 가설이 더 자주 쓰인다. Analyst 가 낸 **순서**가 "
          "배정 빈도가 된다 — 의도가 아니다")
    print("  ⚠️ 부모 종류와도 상관된다: i=0~5 exploit / 6~8 explore / 9~11 cross")


def main() -> None:
    analyst_overlap()
    rule_writer_spread()
    parent_kind_effect()
    cross_uses_two_parents()
    assignment_bias()


if __name__ == "__main__":
    main()
