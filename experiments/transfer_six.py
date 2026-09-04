"""★ 세 쌍 **여섯 방향** — 무엇이 전이를 어렵게 하나. LLM 0회.

    python3 experiments/transfer_six.py

`transfer_29_5.py` 가 방향마다 낸 json 을 모아 한 장으로 만든다.
**여기서 새로 적합하지 않는다** — 절차가 갈리면 안 된다 (원칙 2).

사전 등록 `docs/artifacts/transfer-4090-prereg.md` §5 의 세 질문:

```
1  A6000<->4090 (바운드 뒤집힘 0) 에서 (a) 가 되나
2  그 쌍에서 (b) ≈ (c) 가 유지되나
3  세 쌍에서 (a) 의 손해가 ridge 차이와 상관이 있나
   ★ 쌍이 셋뿐이라 기술 통계다 — 상관계수를 내지 않는다 (원칙 27)
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: 여섯 방향. (출처, 대상)
DIRS = [("a6000", "4090"), ("4090", "a6000"),
        ("a6000", "5090"), ("5090", "a6000"),
        ("4090", "5090"), ("5090", "4090")]
#: 쌍 이름 -> 바운드 뒤집힘 수 (`transfer_check` 실측, 사전 등록 §1).
#: ★ 키는 **정렬한 순서**다 — `_pair` 가 그렇게 만든다 (방향이 둘이라
#: 한쪽만 넣으면 반대 방향에서 KeyError 가 난다. 실제로 났다).
FLIP = {tuple(sorted(k)): v for k, v in
        {("a6000", "4090"): 0, ("a6000", "5090"): 4,
         ("4090", "5090"): 3}.items()}
#: 판정선. 사전 등록 §4 — 여기서 새로 정하지 않는다 (원칙 7).
DELTA = 0.0516


def _pair(src: str, dst: str) -> tuple:
    return tuple(sorted((src, dst)))


def _load(src: str, dst: str) -> dict | None:
    f = Path(f"docs/artifacts/transfer-{src}-{dst}.json")
    return json.loads(f.read_text()) if f.exists() else None


def _med(x) -> float:
    return float(np.median(np.asarray(x, dtype=np.float64)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/transfer-six.json")
    a = ap.parse_args()

    got = {(s, d): _load(s, d) for s, d in DIRS}
    missing = [f"{s}->{d}" for (s, d), v in got.items() if v is None]
    if missing:
        print(f"  ⚠️ 아직 없는 방향: {missing}")
        print("     ★ 없는 것을 빼고 결론 내지 않는다 — 다 돌고 다시 부른다")

    print("=" * 96)
    print("세 쌍 여섯 방향 — 홀드아웃 20형상 (뒤집힘 포함) / 괄호는 뒤집힘 제외")
    print("=" * 96)
    print(f"  {'방향':16s} {'ridge비':>8} {'뒤집힘':>6} "
          f"{'(a) 완전이식':>22} {'(b) 재적합':>22} {'(c) 재생성':>22}")
    rows: dict = {}
    for s, d in DIRS:
        v = got[(s, d)]
        if v is None:
            print(f"  {s + ' -> ' + d:16s} {'—':>8} {'—':>6} "
                  f"{'(아직 안 돌았다)':>22}")
            continue
        r = v["ridge"][1] / v["ridge"][0]
        f = FLIP[_pair(s, d)]
        cell = []
        for k in ("a", "b", "c"):
            cell.append(f"{_med(v[k]):.4f} ({_med(v[k + '_nf']):.4f})")
        print(f"  {s + ' -> ' + d:16s} {r:8.2f} {f:6d} "
              + " ".join(f"{c:>22s}" for c in cell))
        rows[f"{s}->{d}"] = {
            "ridge_ratio": r, "flipped": f,
            "a": _med(v["a"]), "b": _med(v["b"]), "c": _med(v["c"]),
            "a_nf": _med(v["a_nf"]), "b_nf": _med(v["b_nf"]),
            "c_nf": _med(v["c_nf"]),
            "a_range": [min(v["a"]), max(v["a"])],
            "b_range": [min(v["b"]), max(v["b"])],
            "c_range": [min(v["c"]), max(v["c"])],
            "baseline": v["baseline"], "n": [len(v["a"]), len(v["c"])]}

    # ------------------------------------------------------------ 질문 1·2
    print("\n" + "=" * 96)
    print("질문 1·2 — 방향마다 (a) 와 (b) 가 (c) 에서 얼마나 떨어져 있나")
    print("=" * 96)
    print(f"  {'방향':16s} {'뒤집힘':>6} {'(a)-(c)':>10} {'(b)-(c)':>10}"
          f"   {'뒤집힘 제외 (a)-(c)':>20} {'(b)-(c)':>10}")
    for s, d in DIRS:
        k = f"{s}->{d}"
        if k not in rows:
            continue
        w = rows[k]
        da, db = w["a"] - w["c"], w["b"] - w["c"]
        dan, dbn = w["a_nf"] - w["c_nf"], w["b_nf"] - w["c_nf"]
        w["a_minus_c"], w["b_minus_c"] = da, db
        w["a_minus_c_nf"], w["b_minus_c_nf"] = dan, dbn
        print(f"  {k:16s} {w['flipped']:6d} {da:+10.4f} {db:+10.4f}   "
              f"{dan:+20.4f} {dbn:+10.4f}")
    print(f"\n  판정선 delta = {DELTA} (사전 등록 §4).  "
          "양수 = (c) 보다 나쁘다")
    print("  ★ (b)-(c) 가 delta 안이면 '구조는 옮겨지고 가중치만 다시 맞추면"
          " 된다' 가 그 방향에서 선다")

    # -------------------------------------------------------------- 질문 3
    print("\n" + "=" * 96)
    print("질문 3 — (a) 의 손해가 ridge 차이와 같이 가나  ★ 쌍 3개, 기술 통계")
    print("=" * 96)
    print(f"  {'쌍':16s} {'ridge비':>8} {'뒤집힘':>6} "
          f"{'(a)-(c) 두 방향 평균':>20} {'뒤집힘 제외':>12}")
    per_pair: dict = {}
    for p in sorted({_pair(s, d) for s, d in DIRS}):
        ks = [f"{s}->{d}" for s, d in DIRS if _pair(s, d) == p and
              f"{s}->{d}" in rows]
        if not ks:
            continue
        m = float(np.mean([rows[k]["a_minus_c"] for k in ks]))
        mn = float(np.mean([rows[k]["a_minus_c_nf"] for k in ks]))
        rr = float(np.mean([rows[k]["ridge_ratio"] for k in ks]))
        # ★ 비율은 방향마다 역수라 평균이 1 근처로 쏠린다. **차이의 절대값**
        #   으로 읽는다 — 같은 쌍의 두 방향이 같은 값을 갖는다.
        far = abs(np.log(rows[ks[0]]["ridge_ratio"]))
        per_pair["+".join(p)] = {"ridge_log_dist": far, "flipped": FLIP[p],
                                 "a_minus_c_mean": m, "a_minus_c_nf_mean": mn,
                                 "dirs": ks, "ridge_ratio_mean": rr}
        print(f"  {'+'.join(p):16s} {far:8.3f} {FLIP[p]:6d} "
              f"{m:20.4f} {mn:12.4f}")
    print("\n  ridge비 열은 **|log(비율)|** 이다 — 방향마다 역수라 그냥 "
          "평균하면 1 근처로 뭉갠다")
    print("  ★ 상관계수를 내지 않는다. 쌍이 셋이다 (원칙 27)")

    Path(a.out).write_text(json.dumps(
        {"delta": DELTA, "dirs": rows, "pairs": per_pair,
         "missing": missing}, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
