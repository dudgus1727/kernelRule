"""★ Observing F2's stage 3 — the three items written in the
pre-registration. 0 LLM calls.

    python3 experiments/f2_observe.py

```
1. does a 7-term seed become 8 terms (and when?)
   In the D-54 investigation, F1's #9 started at 7 terms and filled itself to
   8 at r1. If it fills up even with room to spare, "starting at 7 terms is
   not an advantage" follows

2. does the np.where selection structure survive
   It was not in the seed. If it is kept, it is the F2 library's own
   contribution
   If it is dropped, the reason is read from the changes field

3. are the 7 new axes still used after the evolution
```
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np

#: What stage 1 judged to be a "new axis" (sp <= 0.95 against the human 24).
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
    """`if` reweighting / `np.where` selection / the features used."""
    t = ast.parse(code)
    where_sel = 0
    for n in ast.walk(t):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "where" and "p." in ast.unparse(n.args[0])):
            # ★ Two things are told apart — selection (both sides features)
            #   and gating (one side a constant)
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
    print("F2 stage-3 observation — the pre-registration items")
    print("=" * 78)
    print(f"  seed {seed['source']}  {s0['n_terms']} terms  "
          f"if {s0['if']}  where-select {s0['where_select']}\n")

    print("1. ★ does a 7-term seed become 8 terms")
    print(f"   {'run':6s} {'r0':>3} {'r1':>3} {'r2':>3} {'r3':>3} {'r4':>3} "
          f"{'r5':>3} {'r6':>3} {'r7':>3} {'r8':>3} {'r9':>3} {'r10':>4} "
          f"{'r11':>4}   the final best")
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
              + f"   {n_terms(best['code'])} terms"
              + (f"  (8 terms first at r{first8})" if first8 is not None else
                 "  (8 terms never)"))
    n_hit = sum(1 for x in hit8 if x is not None)
    print(f"\n   -> runs where 8 terms appeared {n_hit}/{len(hit8)}"
          + (f", median round of first appearance "
             f"{int(np.median([x for x in hit8 if x is not None]))}"
             if n_hit else ""))

    print("\n2. ★ does the np.where selection structure survive")
    print(f"   {'run':6s} {'best rule':>10} {'if':>3} {'where-select':>13}  "
          f"changes")
    for s in range(6):
        f = Path("runs") / f"{tag}-s{s}" / "archive.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(ln) for ln in f.open() if ln.strip()]
        best = min(rows, key=lambda e: e["regret"])
        st = structure(best["code"])
        print(f"   s{s:<5d} {st['n_terms']:10d} {st['if']:3d} "
              f"{st['where_select']:13d}  {str(best.get('changes'))[:44]}")

    print("\n3. ★ are the new axes still used after the evolution")
    used: set[str] = set()
    for s in range(6):
        f = Path("runs") / f"{tag}-s{s}" / "archive.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(ln) for ln in f.open() if ln.strip()]
        best = min(rows, key=lambda e: e["regret"])
        st = structure(best["code"])
        print(f"   s{s}  new axes {len(st['feats'] & NEW_AXES)}  "
              f"known5 {len((st['feats'] | st['shape_vals']) & KNOWN5)}  "
              f"{sorted(st['feats'] & NEW_AXES)}")
        used |= st["feats"] & NEW_AXES
    print(f"\n   -> new axes used in the best rule of the 6 runs "
          f"{len(used)}/{len(NEW_AXES)}")
    print(f"      not used: {sorted(NEW_AXES - used)}")


if __name__ == "__main__":
    main()
