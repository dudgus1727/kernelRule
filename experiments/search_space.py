"""★ The search space the autotuning curve runs on (D-176 §1-1 · §4).
**0 LLM calls · 0 GPU.**

    python3 -m experiments.search_space

```
★ 자유 축 11개        표에서 값 집합을 읽어 센다
⛔ align_a/b/c        자유 축이 아니다 — ★ 형상이 정한다 (여기서 확인한다)
데카르트 곱           축 크기의 곱
형상당 유효 조합       그 형상의 후보 행 수
★ 유효 비율           유효 / 데카르트
가지치기 후           스필 없음 · 타일 <= 문제 (§1-2)
```

⚠️ Every number here is counted from the bundle, not quoted from the order.
If a table disagrees with another, both are printed.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import warnings
from pathlib import Path

import numpy as np

from experiments.autotune_curve import AXES, _prune_mask
from experiments.transfer_29_5 import TABLES
from kernelrule.core.splits import experiment_shapes
from kernelrule.core.table import PerfTable

ALIGN = ("align_a", "align_b", "align_c")
OUT = Path("docs/artifacts/search-space.json")


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    res: dict = {"axes": list(AXES), "align_axes": list(ALIGN),
                 "implementation": (
                     "conditional draw over the shape's own candidate list: "
                     "axes in a fixed order, each domain filtered by the "
                     "choices already made, so a draw always lands on a real "
                     "candidate (D-176 §1-1)"),
                 "tables": {}}
    print("=" * 96)
    print("★ 탐색 공간 (D-176 §1-1). 0 LLM · 0 GPU — 표에서 센다")
    print("=" * 96)
    for gpu, T in TABLES.items():
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        shapes = experiment_shapes(table)
        dom: dict = {x: set() for x in AXES}
        n_valid, n_pruned, align_free = [], [], 0
        for p in shapes:
            d = table.frame_for(p)
            for x in AXES:
                dom[x] |= {v.item() if hasattr(v, "item") else v
                           for v in d[x].tolist()}
            n_valid.append(len(d))
            n_pruned.append(int(_prune_mask(d).sum()))
            if any(d[x].nunique() > 1 for x in ALIGN):
                align_free += 1
        sizes = {x: len(dom[x]) for x in AXES}
        cart = int(np.prod(list(sizes.values())))
        frac = [v / cart for v in n_valid]
        keep = [p / v for p, v in zip(n_pruned, n_valid, strict=True)]
        r = {"n_shapes": len(shapes), "axis_sizes": sizes,
             "cartesian": cart,
             "valid_median": int(st.median(n_valid)),
             "valid_range": [min(n_valid), max(n_valid)],
             "valid_frac_median": round(st.median(frac), 4),
             "pruned_median": int(st.median(n_pruned)),
             "prune_keeps_mean": round(float(np.mean(keep)), 4),
             # ★ 0 means every shape fixes all three alignment axes.
             "shapes_with_a_free_align_axis": align_free}
        res["tables"][gpu] = r
        print(f"\n  ── {gpu}  {r['n_shapes']} 형상")
        print("     축 " + " · ".join(f"{x} {sizes[x]}" for x in AXES))
        print(f"     데카르트 {cart:,}  유효 중앙 {r['valid_median']:,} "
              f"({r['valid_range'][0]:,}~{r['valid_range'][1]:,}) "
              f"= {r['valid_frac_median']:.2%}")
        print(f"     가지치기 후 중앙 {r['pruned_median']:,} "
              f"({r['prune_keeps_mean']:.1%} 남음)")
        print(f"     ⛔ align 이 자유인 형상 {align_free}/{len(shapes)}")
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
