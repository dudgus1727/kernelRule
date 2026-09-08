"""★ F1 — can the LLM **make** a physical quantity? The fundamental question
of this project.

    python3 experiments/feature_writer.py [n_proposals] [condition] [model]

## Why it is the fundamental question

In every run so far the LLM has only **combined the 24 a human wrote**.
**Defining** `tail_waste` is the act of understanding the physics, and now
that §29 has peeled the weights off too, what remains is "pick 8 out of 24".

    F0  no features       can it put physics into code from scratch
    F1  raw values only   can it make a derived physical quantity   ★ here
    F2  the base 5        can it build on top of them
    F3  all 24            combination only (= what has been done so far)

Making `tail_waste` requires deriving
`ceil(M/tile_m)*ceil(N/tile_n)/sm_count`, and that is **evidence of having
understood wave quantization**.

## The evaluation — rediscovery

A different name still counts as a rediscovery if it is **mathematically the
same**. The criterion is the same as §8.4: if Spearman **and** Pearson are
both above 0.95, they measure the same thing. (Both are looked at in order to
tell a monotone transform from a linear relation — `sm_idle_cost` and
`tail_waste` have a Spearman of 1.0 but a low Pearson.)

**A partial success is a result too.** Rediscovering half is evidence that
"the LLM can derive a physical quantity", and a human can fill in the rest.
**A failure is a result too** — it confirms that "features defined by a
human, combination by the LLM" is the accurate division of labour.

## ⚠️ The structural holdout is not looked at (§12.3d)

This experiment is judged only by **the number of rediscoveries** and **the
in-sample regret**. The structural holdout is looked at once, after F1 is
over.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.agents.openai_client import DEFAULT_MODEL, Budget, LLMConfig, OpenAILLM
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY, FeatureRegistry
from kernelrule.features.generated import (
    FeatureRejected,
    _reference_columns,
    register_generated,
)
from kernelrule.features.validate import _pearson, _spearman, alt_hw

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
OUT = Path("runs")

#: The rediscovery criterion (the same as §8.4). Spearman **and** Pearson,
#: both.
RHO = 0.95




def _columns(reg: FeatureRegistry, table, matrix, shapes) -> dict:
    """The value vector per feature. Used for the rediscovery comparison."""
    out: dict[str, list] = {n: [] for n in reg._items}
    for p in shapes:
        feats, info = matrix.for_shape(p)
        for n, acc in out.items():
            f = reg[n]
            v = getattr(info, n) if f.shape_level else getattr(feats, n)
            acc.append(np.full(1, float(v)) if f.shape_level
                       else np.asarray(v, float))
    return {n: np.concatenate(v) for n, v in out.items()}


def main(n_proposals: int = 20, condition: str = "F1",
         model: str = DEFAULT_MODEL) -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    hw_alt = alt_hw(table.hw)

    gen = FeatureRegistry(f"generated-{condition}")
    llm = OpenAILLM(LLMConfig(model=model, concurrency=4),
                    feature_names=[], shape_values=[], registry=REGISTRY,
                    budget=Budget(max_calls=200), cache=False)

    d = OUT / f"featwriter-{condition}-{model}"
    d.mkdir(parents=True, exist_ok=True)
    log = d / "proposals.jsonl"
    log.write_text("")

    print("=" * 78)
    print(f"F1 — FeatureWriter, condition {condition}, {n_proposals} "
          f"proposals  [{model}]")
    print("=" * 78)
    print("  ★ only raw values are given. Not one existing feature name is in "
          "the prompt"
          if condition == "F1" else "  the existing features are shown")
    print()

    # ★ The reference columns are **accumulated**. It used to rebuild the
    #   FeatureMatrix over the whole generated registry on every acceptance,
    #   which took a proposal from 40 seconds to 4 minutes (O(n²)).
    #   Only the new feature's column is computed once and merged in.
    ref_cols = _reference_columns(table, matrix, FeatureRegistry("empty"))

    def _add_columns(f) -> None:
        one = FeatureRegistry(f"col-{f.name}")
        one.add(f)
        ref_cols.update(_reference_columns(table, FeatureMatrix(table, one),
                                           FeatureRegistry("empty")))

    t0 = time.perf_counter()
    failures: list[tuple[str, str]] = []
    for i in range(n_proposals):
        made = sorted(gen._items)
        # ★ The failures are fed back. Without them it **repeats the same
        #   proposal** — in the first run 16 of 20 were the same feature
        #   (D-38). It is the same disease as the 132/240 retry exhaustion:
        #   what you do not know is wrong you cannot fix.
        recent = ""
        if failures:
            lines = "\n".join(f"- `{n}` — {e}" for n, e in failures[-5:])
            recent = ("\n\n## ★ What was just rejected — do not propose the "
                      f"same thing again\n\n{lines}\n\nRead the rejection "
                      "reasons and find **a different axis**, or if it is the "
                      "same axis, fix that reason and propose again.")
        task = ("## What to make this time\n\nPropose one feature."
                + (f"\n\nWhat has been made so far: {made}\nFind **an axis "
                   "different from these**." if made else "") + recent)
        row: dict = {"i": i}
        try:
            out = llm.complete("feature", "", condition=condition, task=task,
                               registry=REGISTRY)
            row |= {k: out.get(k) for k in
                    ("name", "code", "rationale", "unit", "expected_range",
                     "direction")}
            f = register_generated(out["code"], registry=gen, meta=out,
                                   table=table, matrix=matrix, hw_alt=hw_alt,
                                   others=ref_cols)
            _add_columns(f)
            row["accepted"] = True
            print(f"  #{i:02d}  ✓ {f.name:28s} {f.expected_range}", flush=True)
        except FeatureRejected as e:
            row |= {"accepted": False, "error": str(e)}
            failures.append((str(row.get("name") or "?"), str(e)[:180]))
            print(f"  #{i:02d}  ✗ {str(e)[:80]}", flush=True)
        except Exception as e:                              # noqa: BLE001
            row |= {"accepted": False, "error": f"{type(e).__name__}: {e}"}
            print(f"  #{i:02d}  ! {type(e).__name__}: {str(e)[:70]}")
        with log.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    # ★ An LLM call cannot be made again (D-33). It is kept before the scoring.
    llm.dump(d / "llm_calls")

    print(f"\n  accepted {len(gen._items)}/{n_proposals}   "
          f"{time.perf_counter() - t0:.0f}s   calls {llm.budget.calls}"
          f"  input {llm.budget.input_tokens:,}  "
          f"output {llm.budget.output_tokens:,}")
    if not gen._items:
        return

    # -- the rediscovery judgement -----------------------------------------
    shapes = list(table.shapes())[:8]
    mine = _columns(gen, table, FeatureMatrix(table, gen), shapes)
    ref = _columns(REGISTRY, table, matrix, shapes)

    print(f"\n{'=' * 78}")
    print(f"rediscovery — how many of the {len(ref)} a human wrote were made "
          f"again")
    print(f"  the criterion: Spearman **and** Pearson both > {RHO} (§8.4)")
    print("=" * 78)
    found: dict[str, tuple[str, float, float]] = {}
    for gname, gv in mine.items():
        for rname, rv in ref.items():
            if len(gv) != len(rv):
                continue
            sp, pe = abs(_spearman(gv, rv)), abs(_pearson(gv, rv))
            if sp > RHO and pe > RHO:
                prev = found.get(rname)
                if prev is None or sp + pe > prev[1] + prev[2]:
                    found[rname] = (gname, sp, pe)
    for rname, (gname, sp, pe) in sorted(found.items()):
        print(f"  {rname:26s} <- {gname:26s} sp {sp:.3f}  pe {pe:.3f}")
    print(f"\n  ★ rediscovered {len(found)}/{len(ref)}   "
          f"new axes {len(gen._items) - len(found)}")

    novel = [n for n in sorted(gen._items)
             if n not in {g for g, _, _ in found.values()}]
    if novel:
        print(f"\n  axes not among the existing 24: {novel}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20,
         sys.argv[2] if len(sys.argv) > 2 else "F1",
         sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL)
