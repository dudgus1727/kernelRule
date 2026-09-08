"""★ It **re-checks a refused proposal with the fixed checker**. 0 LLM calls.

    python3 experiments/revalidate.py runs/F2rw-p8/stage1-features

## Why re-checking, not re-proposing

```
re-proposing (N LLM calls)   only that area gets more proposals, so **the
                             condition changes**
★ re-checking (0 LLM calls)  the same code through the fixed checker. The
                             condition does not change
```

**What the checker threw away is not what the LLM could not make.** The same
was done at D-37/D-38 — what an `inspect.getsource` defect had thrown away was
fixed and passed again.

**What was revived and what is still refused are recorded separately.**
"it was thrown away by a checker defect" and "it really does not work" are
different facts.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import FeatureRegistry
from kernelrule.features.generated import FeatureRejected, register_generated
from kernelrule.features.loader import load_generated
from kernelrule.features.validate import alt_hw

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"




def main(stage1: str) -> None:
    warnings.simplefilter("ignore")
    d = Path(stage1)
    path = d / "proposals.jsonl"
    rows = [json.loads(ln) for ln in path.open() if ln.strip()]
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    # ★ The accepted ones go in first — the duplicate check has to run against
    #   them too.
    gen = FeatureRegistry("revalidate")
    for f in load_generated(path, table=table, exclude=set()):
        gen.add(f)
    n_before = len(gen._items)
    matrix = FeatureMatrix(table, gen) if gen._items else None
    hw_alt = alt_hw(table.hw)

    print("=" * 78)
    print(f"re-checking — {stage1}   (0 LLM calls)")
    print("=" * 78)
    print(f"  {n_before} already accepted.  "
          f"{sum(1 for r in rows if not r.get('accepted'))} refusals are "
          f"looked at again\n")

    revived, still = [], []
    for r in rows:
        if r.get("accepted") or not r.get("code"):
            continue
        name = r.get("name") or "?"
        try:
            f = register_generated(
                r["code"], registry=gen,
                meta={k: r.get(k) for k in
                      ("name", "unit", "direction", "expected_range")},
                table=table,
                matrix=matrix or FeatureMatrix(table, FeatureRegistry("e")),
                hw_alt=hw_alt)
            revived.append(f.name)
            print(f"  ★ revived  {f.name}")
            print(f"       before: {str(r.get('error'))[:80]}")
        except FeatureRejected as e:
            still.append((name, str(e)[:90]))
            print(f"     still refused  {name}")
            print(f"       {str(e)[:88]}")
        except Exception as e:                              # noqa: BLE001
            still.append((name, f"{type(e).__name__}: {e}"[:90]))
            print(f"     still refused  {name}  {type(e).__name__}")

    print()
    print(f"  ★ revived {len(revived)}: {revived}")
    print(f"     still refused {len(still)}")
    print(f"     the library {n_before} -> {len(gen._items)}")

    if revived:
        # ★ What was revived is kept in **a separate file**. Overwriting
        #   `proposals.jsonl` would erase "what was refused at the time"
        #   (documentation rule 2).
        out = d / "revalidated.jsonl"
        with out.open("w") as fh:
            for r in rows:
                if r.get("name") in revived:
                    fh.write(json.dumps({**r, "accepted": True,
                                         "revalidated": True,
                                         "original_error": r.get("error")},
                                        ensure_ascii=False) + "\n")
        print(f"\n  recorded: {out}  (the original proposals.jsonl is left "
              f"alone)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "runs/F2rw-p8/stage1-features")
