"""★ 검증 실행 관찰 — D-75 경로와 D-78 분기 상수. **LLM 호출 0회**.

    python3 experiments/d75_observe.py runs/x-probe-d78d75v1-s0

실험 계획서: `docs/artifacts/d75-verification-prereg.md` 의 여섯 항목.
**성능은 안 본다** — 2라운드 1시드는 시드 폭 안이다.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

#: 요구 문장에 있으면 **표 정보가 샌 것**이다 (실험 계획서 5번).
_LEAK = (
    (re.compile(r"사례\s*#?\d"), "사례 번호"),
    (re.compile(r"regret|리그렛"), "regret"),
    (re.compile(r"\b(1024|2048|4096|8192|11008|5120|13824)\b"), "형상 크기"),
    (re.compile(r"홀드아웃|검증 분할|학습 분할"), "분할 이름"),
    (re.compile(r"\d+\s*번째 형상|형상 목록"), "형상 목록"),
)

#: 상수 1 을 대신 만드는 형태들 (D-78 이 없애려던 것).
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
    """`ast.Compare` 의 직접 피연산자인 숫자 리터럴을 센다."""
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
    print(f"D-75 / D-78 검증 관찰 — {d}")
    print("  ★ 성능은 보지 않는다 (2라운드 1시드는 시드 폭 안)")
    print("=" * 74)

    # 1. 요구 빈도 -------------------------------------------------------
    #  ★ **첫 Analyst 응답만** 센다. 옛 실행은 라운드당 Analyst 가 한 번
    #    이었으므로, 되돌아간 두 번째 응답을 섞으면 분모가 달라진다 (원칙 4).
    p1 = [h for h in hyps if h.get("analyst_pass", 1) == 1]
    p2 = [h for h in hyps if h.get("analyst_pass") == 2]
    filled = [h for h in p1
              if (h.get("physical_requirement") or h.get("needs_new_feature"))]
    print(f"\n1. 요구 빈도 (1차 Analyst 만)  {len(filled)}/{len(p1)} 가설"
          + (f" = {len(filled)/len(p1):.1%}" if p1 else ""))
    print("   기준선 10/56 = 17.9%  (같은 조건 F3 · 같은 모델 · r0~r1, D-79)")
    if p1:
        from scipy.stats import fisher_exact
        pv = fisher_exact([[10, 46], [len(filled), len(p1) - len(filled)]])[1]
        print(f"   Fisher p = {pv:.4f}  "
              + ("★ 기준선과 구분 불가" if pv >= 0.05 else "★ 유의하게 다르다"))
    print(f"   (2차 Analyst 가설 {len(p2)}건은 여기서 제외했다)")

    # 라운드 구간별 — 깊어지면 달라질 수 있다
    for lo, hi, lbl in ((0, 1, "r0~r1"), (2, 99, "r2 이후")):
        g = [h for h in p1 if lo <= h.get("round", 0) <= hi]
        k = sum(1 for h in g
                if (h.get("physical_requirement") or h.get("needs_new_feature")))
        if g:
            print(f"   {lbl:8s} {k}/{len(g)} = {k/len(g):5.1%}")

    # 3. 요구 내용 + 5. 물리 언어인가 ------------------------------------
    print(f"\n3+5. 요구 내용과 **누출 검사** ({len(filled)}건)")
    print("   ⚠️ 옛 303건은 누출 0/303 이었다 (D-79) — 여기서 나오면 새 문제다")
    n_leak = 0
    for h in filled:
        t = (h.get("physical_requirement") or h.get("needs_new_feature") or "")
        hits = [why for rx, why in _LEAK if rx.search(t)]
        n_leak += bool(hits)
        mark = f"★누출 {hits}" if hits else "OK"
        print(f"   [{h.get('round','?')}] {mark}\n       {t}")
    print(f"\n   ★ 누출 {n_leak}/{len(filled)}"
          + ("  — 조건 1 이 무의미해진다. 본 실행 전에 고쳐야 한다"
             if n_leak else "  — 물리 언어로만 왔다"))

    # 2. 만든 축과 사용률 ------------------------------------------------
    made = [f for f in feats if f.get("accepted")]
    print(f"\n2. 만든 축  {len(made)}/{len(feats)} 시도")
    best = min(arc, key=lambda e: e["regret"]) if arc else None
    for f in feats:
        if f.get("accepted"):
            used_best = bool(best) and f"f.{f['name']}" in best["code"]
            used_any = sum(1 for e in arc if f"f.{f['name']}" in e["code"])
            print(f"   ✓ r{f['round']} {f['name']}"
                  f"   최선규칙 {'예' if used_best else '아니오'}"
                  f" / 아카이브 {used_any}/{len(arc)}")
        else:
            print(f"   ✗ r{f['round']} {str(f.get('error'))[:80]}")

    # 4. 되돌아간 Analyst 가 새 축을 쓰는가 -------------------------------
    print("\n4. ★ 되돌아간 Analyst 가 새 축을 언급하는가")
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
            print(f"   seq{s}: 피처 뒤에 analyze 가 없다 — 되돌아가지 않았다")
            continue
        nm = names_made.get(rnd) if rnd is not None else None
        blob = json.dumps(nxt[2], ensure_ascii=False)
        hit = bool(nm) and nm in blob
        print(f"   seq{s} 피처 -> seq{nxt[0]} analyze : "
              f"{'★ 언급함' if hit else '언급 없음'}  ({nm})")

    # 상한 초과 -----------------------------------------------------------
    over = [f for f in feats if f.get("over_cap")]
    print(f"\n2b. 라운드당 상한에 걸려 안 만든 요구  {len(over)}건")
    for f in over:
        print(f"   r{f['round']}  {f['requirement'][:70]}")

    # 6. 분기 상수 -------------------------------------------------------
    print(f"\n6. 분기 상수 (아카이브 {len(arc)}규칙)")
    n_bc = sum(1 for e in arc if _branch_constants(e["code"])[0])
    vals: list = []
    for e in arc:
        vals += _branch_constants(e["code"])[1]
    print(f"   비교에 숫자를 쓴 규칙  {n_bc}/{len(arc)}   값 {sorted(set(vals))}")
    for pat in _DODGE:
        k = sum(1 for e in arc if pat in e["code"])
        print(f"   {pat:16s} {k}/{len(arc)}")
    print("   ⚠️ 옛 330규칙은 다른 조건(합산 예산)이라 N 도 조건도 다르다 — "
          "첫 인상이지 결론이 아니다")

    # 호출 수 -----------------------------------------------------------
    roles: dict[str, int] = {}
    for c in calls:
        roles[c["role"]] = roles.get(c["role"], 0) + 1
    print(f"\n호출 {len(calls)}회  {roles}")


if __name__ == "__main__":
    main()
