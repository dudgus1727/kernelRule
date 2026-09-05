"""★ 라운드별 **최종 채점** 곡선 — "몇 라운드가 필요한가" 를 데이터로. LLM 0회.

    python3 experiments/round_curve.py                      # 새 실행 + 옛 실행
    python3 experiments/round_curve.py --group F3rw-p8

## 왜

`12` 도 `patience 3` 도 근거 없이 정한 값이었다. 그리고 루프 내부
점수(`best_val_regret`)로 멈추면 **최종 채점이 아직 오르고 있는데**
멈춘다 (D-131 — 그렇게 +0.0187 을 잃었다).

**그래서 최종 채점 자체의 곡선을 낸다.**

```
각 라운드 r 까지의 아카이브에서 **학습 점수 최고**를 고르고
-> 체제별로 학습 분할에 재적합 -> 홀드아웃 20형상에서 채점
= 그 라운드에서 멈췄다면 보고했을 값
```

## 판정 — 실험 계획서에 미리 박았다 (D-132 §2)

```
rN 부터 σ(0.0113) 안에서 평평   ★ N 이 필요한 라운드다
상한까지 계속 오른다            ★ 상한도 부족하다 — 그 사실을 적는다
```

★ LLM 0회. 시드마다 라운드 수만큼 재적합하므로 몇십 분 걸린다.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import warnings
from pathlib import Path

from sigma_5090 import _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
#: (묶음 이름, 실행 접두, 시드 수, 설명)
GROUPS = [
    ("F3rw-p8", "F3rw-p8", 6, "새 — 조기 종료 끔, 라운드 24 (D-132)"),
    ("F3rw-p8-old", "F3rw-p8-old", 6, "옛 — 라운드 12, patience 10"),
    # ★ D-131 이 프롬프트 효과를 잰 캠페인 (patience 3 으로 r4~r6 에서 멈췄다)
    ("F3rw-p8-p3", "F3rw-p8-p3", 6, "새 — patience 3, r4~r6 에서 종료"),
    # ★ D-136 이후의 대표값 재측정 — __import__ 수정만 F3rw-p8 과 다르다
    ("F3rw-p8-nan", "F3rw-p8-nan", 6, "새 — __import__ 수정, 라운드 24"),
]
#: 평평함 판정에 쓰는 시드 폭. **여기서 새로 정하지 않는다** (원칙 7).
SIGMA = 0.0113


def _curve(run: str, T, M, sp) -> list[float]:
    """라운드마다 '그때 멈췄다면 보고했을 값'."""
    arc = [json.loads(x) for x in
           (Path("runs") / run / "archive.jsonl").read_text().splitlines()
           if x.strip()]
    n_rounds = max(e.get("round", 0) for e in arc) + 1
    out, prev_code = [], None
    for r in range(n_rounds):
        sub = [e for e in arc if e.get("round", 99) <= r]
        if not sub:
            out.append(float("nan"))
            continue
        b = sorted(sub, key=lambda e: e["regret"])[0]
        key = (b["code"], tuple(b["w"]))
        if key == prev_code:            # ★ 최고가 안 바뀌면 다시 안 잰다
            out.append(out[-1])
            continue
        prev_code = key
        out.append(canonical_score(b["code"], b["w"], table=T, matrix=M,
                                   splits=sp).holdout)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append")
    ap.add_argument("--out", default="docs/artifacts/round-curve.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M, sp = FeatureMatrix(T, REGISTRY), None
    sp = _splits(T)
    out: dict = {"sigma": SIGMA, "groups": {}}

    for name, prefix, n, note in GROUPS:
        if a.group and name not in a.group:
            continue
        print("=" * 92)
        print(f"{name}  — {note}")
        print("=" * 92)
        curves = {}
        for i in range(n):
            run = f"{prefix}-s{i}"
            if not (Path("runs") / run / "archive.jsonl").exists():
                print(f"  ⚠️ 없는 실행: {run}")
                continue
            c = _curve(run, T, M, sp)
            curves[run] = c
            print(f"  {run:18s} " + " ".join(f"{x:.4f}" for x in c),
                  flush=True)
        if not curves:
            continue
        L = max(len(c) for c in curves.values())
        med = [st.median([c[min(r, len(c) - 1)] for c in curves.values()])
               for r in range(L)]
        print(f"\n  {'중앙':18s} " + " ".join(f"{x:.4f}" for x in med))
        # ★ 평평해지는 지점 — 여기부터 끝까지 σ 안이다
        final = med[-1]
        flat = next((r for r in range(L)
                     if all(abs(med[k] - final) < SIGMA for k in range(r, L))),
                    None)
        print(f"  ★ 끝값에서 σ({SIGMA}) 안으로 들어오는 라운드: "
              + (f"r{flat}" if flat is not None else "없음"))
        print(f"  ★ 마지막 라운드까지의 개선 (r0 -> 끝): "
              f"{med[0] - final:+.4f}   "
              f"마지막 4라운드 개선: {med[max(0, L - 5)] - final:+.4f}")
        out["groups"][name] = {"curves": curves, "median": med,
                               "flat_from": flat, "note": note}

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                      default=float))
    print(f"\n  -> {a.out}")
    print("  ★ 판정은 실험 계획서(D-132 §2)의 두 갈래로만 한다")


if __name__ == "__main__":
    main()
