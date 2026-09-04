"""★ F2 3단계 관찰 — 실험 계획서에 적은 항목 셋. LLM 0회.

    python3 experiments/f2_observe.py

```
1. 7항 씨앗이 8항이 되는가 (언제?)
   D-54 조사에서 F1 의 #9 가 7항으로 시작해 r1 에 스스로 8항을 채웠다.
   여유가 있어도 채운다면 "7항 시작이 이점이 아니다" 가 된다

2. np.where 선택 구조가 살아남는가
   씨앗에 없던 형태다. 유지되면 F2 라이브러리의 고유 기여다
   버려지면 changes 필드에서 이유를 읽는다

3. 새 축 7개가 진화 후에도 쓰이는가
```
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np

#: 1단계에서 "새 축" 으로 판정된 것 (사람 24개와 sp <= 0.95).
NEW_AXES = {"compute_k_underfill_fraction", "instruction_overhead_fraction",
            "l2_residency_pressure", "pipeline_fill_drain_fraction",
            "reduction_work_fraction", "resident_block_scarcity",
            "roofline_memory_gap", "sm_resource_pressure",
            "tiled_global_traffic_ratio"}

KNOWN5 = {"tail_waste", "occupancy_deficit", "roofline_ratio",
          "edge_waste", "has_spill"}


def n_terms(code: str) -> int:
    return len(set(re.findall(r"w\[(\d+)\]", code)))


def structure(code: str) -> dict:
    """`if` 재가중 / `np.where` 선택 / 쓰인 피처."""
    t = ast.parse(code)
    where_sel = 0
    for n in ast.walk(t):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "where" and "p." in ast.unparse(n.args[0])):
            # ★ 두 가지를 가른다 — 선택(둘 다 피처)과 게이팅(한쪽이 상수)
            a, b = ast.unparse(n.args[1]), ast.unparse(n.args[2])
            if "f." in a and "f." in b:
                where_sel += 1
    return {
        "n_terms": n_terms(code),
        "if": sum(1 for n in ast.walk(t) if isinstance(n, ast.If)),
        "where_select": where_sel,
        "feats": {n.attr for n in ast.walk(t) if isinstance(n, ast.Attribute)
                  and isinstance(n.value, ast.Name) and n.value.id == "f"},
        "shape_vals": {n.attr for n in ast.walk(t)
                       if isinstance(n, ast.Attribute)
                       and isinstance(n.value, ast.Name) and n.value.id == "p"},
    }


def main(tag: str = "F2rw-p8") -> None:
    seed = json.loads(Path(f"runs/{tag}/stage2-rule-writer/chosen.json").read_text())
    s0 = structure(seed["code"])
    print("=" * 78)
    print("F2 3단계 관찰 — 실험 계획서 항목")
    print("=" * 78)
    print(f"  씨앗 {seed['source']}  {s0['n_terms']}항  "
          f"if {s0['if']}  where-선택 {s0['where_select']}\n")

    print("1. ★ 7항 씨앗이 8항이 되는가")
    print(f"   {'실행':6s} {'r0':>3} {'r1':>3} {'r2':>3} {'r3':>3} {'r4':>3} "
          f"{'r5':>3} {'r6':>3} {'r7':>3} {'r8':>3} {'r9':>3} {'r10':>4} "
          f"{'r11':>4}   최종 최고")
    hit8 = []
    for s in range(6):
        f = Path("runs") / f"{tag}-s{s}" / "archive.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(ln) for ln in f.open() if ln.strip()]
        by_r: dict[int, int] = {}
        for e in rows:
            r = int(e.get("round", -1))
            by_r[r] = max(by_r.get(r, 0), n_terms(e["code"]))
        cells = [f"{by_r.get(r, 0) or '.':>3}" for r in range(12)]
        best = min(rows, key=lambda e: e["regret"])
        first8 = next((r for r in range(12) if by_r.get(r, 0) >= 8), None)
        hit8.append(first8)
        print(f"   s{s:<5d} " + " ".join(cells)
              + f"   {n_terms(best['code'])}항"
              + (f"  (8항 첫 등장 r{first8})" if first8 is not None else
                 "  (8항 안 나옴)"))
    n_hit = sum(1 for x in hit8 if x is not None)
    print(f"\n   -> 8항이 나온 실행 {n_hit}/{len(hit8)}"
          + (f", 첫 등장 라운드 중앙 {int(np.median([x for x in hit8 if x is not None]))}"
             if n_hit else ""))

    print("\n2. ★ np.where 선택 구조가 살아남는가")
    print(f"   {'실행':6s} {'최고규칙':>8} {'if':>3} {'where-선택':>10}  changes")
    for s in range(6):
        f = Path("runs") / f"{tag}-s{s}" / "archive.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(ln) for ln in f.open() if ln.strip()]
        best = min(rows, key=lambda e: e["regret"])
        st = structure(best["code"])
        print(f"   s{s:<5d} {st['n_terms']:8d} {st['if']:3d} "
              f"{st['where_select']:10d}  {str(best.get('changes'))[:44]}")

    print("\n3. ★ 새 축이 진화 후에도 쓰이는가")
    used: set[str] = set()
    for s in range(6):
        f = Path("runs") / f"{tag}-s{s}" / "archive.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(ln) for ln in f.open() if ln.strip()]
        best = min(rows, key=lambda e: e["regret"])
        st = structure(best["code"])
        print(f"   s{s}  새축 {len(st['feats'] & NEW_AXES)}  "
              f"known5 {len((st['feats'] | st['shape_vals']) & KNOWN5)}  "
              f"{sorted(st['feats'] & NEW_AXES)}")
        used |= st["feats"] & NEW_AXES
    print(f"\n   -> 6실행의 최고 규칙에서 쓰인 새 축 {len(used)}/{len(NEW_AXES)}")
    print(f"      안 쓰인 것: {sorted(NEW_AXES - used)}")


if __name__ == "__main__":
    main()
