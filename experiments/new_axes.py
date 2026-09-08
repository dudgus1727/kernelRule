"""★ Item 2 — is a generated new axis **useful**? It settles F1's standing.

    python3 experiments/new_axes.py [rounds] [n_seeds]

## Why this comes first

F1 showed "it can be made". **"It is useful" is a different question, and
without the latter the former stays a "proof of possibility".**

```
condition A   the human-written 24              (= the current best, no seed
                                                 rule + descriptions)
condition B   the 24 + N generated new axes
the same 3 seeds / the same rounds / no seed rule / descriptions on
```

## What counts as a "new axis"

Among the generated features, **those with Spearman > 0.95 against an
existing one are dropped** (the §8.4 duplicate criterion). Putting a
rediscovery in gives the same information twice and muddies "is a new axis
useful". The list is not written by hand but **computed here** — writing it
down makes it go out of step the next time F1 is re-run.

## The verdict

```
comparing the in-sample regret
★ is a generated feature actually used in the final rule (how many, which)
the structural holdout is looked at once, after this experiment (§12.3d)
```

**"Is it used" matters as much as regret.** If 37 were given and the final
rule uses only the existing 24, that means the generated features are not
useful.
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.agents.openai_client import DEFAULT_MODEL, Budget, LLMConfig, OpenAILLM
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY, FeatureRegistry
from kernelrule.features.loader import extended_registry, load_generated
from kernelrule.features.validate import _spearman

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
MODEL = DEFAULT_MODEL   # ★ a single source (D-45)
#: ★ A placeholder. `<model>` has to be replaced with a real run for this to
#: work — the `featwriter-F1-gpt-5.4` this script used was deleted (D-52).
PROPOSALS = Path("runs/featwriter-F1-<model>/proposals.jsonl")
DUP_RHO = 0.95


def _columns(reg: FeatureRegistry, table, shapes) -> dict:
    mat = FeatureMatrix(table, reg)
    out: dict[str, list] = {n: [] for n in reg._items}
    for p in shapes:
        fe, info = mat.for_shape(p)
        for n, acc in out.items():
            f = reg[n]
            acc.append(np.full(int(info.n_candidates), float(getattr(info, n)))
                       if f.shape_level else np.asarray(getattr(fe, n), float))
    return {n: np.concatenate(v) for n, v in out.items()}


def novel_axes(table, shapes) -> list:
    if "<model>" in str(PROPOSALS) or not PROPOSALS.exists():
        raise SystemExit(
            f"the feature proposals file is missing: {PROPOSALS}\n"
            "Run F1 with the instructed model first "
            "(experiments/feature_writer.py) and change PROPOSALS to that "
            "path. The earlier gpt-5.4 artefacts were deleted (D-52).")
    """The generated features that **are not duplicates of an existing one**.
    The list is not written by hand."""
    gen = load_generated(PROPOSALS, table=table)
    ref = _columns(REGISTRY, table, shapes)
    tmp = FeatureRegistry("gen-probe")
    for f in gen:
        tmp.add(f)
    mine = _columns(tmp, table, shapes)
    novel, dup = [], []
    for f in gen:
        gv = mine[f.name]
        hit = next((rn for rn, rv in ref.items()
                    if len(rv) == len(gv) and abs(_spearman(gv, rv)) > DUP_RHO),
                   None)
        (dup if hit else novel).append((f, hit))
    print(f"  generated {len(gen)} -> new axes {len(novel)} / "
          f"duplicates {len(dup)}")
    for f, hit in dup:
        print(f"    duplicate  {f.name:30s} ~ {hit}")
    return [f for f, _ in novel]


def used_features(code: str) -> set[str]:
    return {n.attr for n in ast.walk(ast.parse(code.strip()))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id in ("f", "p")}


def main(rounds: int = 12, n_seeds: int = 3) -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")

    print("=" * 78)
    print(f"item 2 — the usefulness of a new axis.  {n_seeds} seeds x "
          f"{rounds} rounds  [{MODEL}]")
    print("=" * 78)
    novel = novel_axes(table, list(table.shapes())[:12])
    ext = extended_registry(REGISTRY, novel)
    new_names = {f.name for f in novel}
    print(f"  condition A features {len(REGISTRY._items)}  /  condition B "
          f"features {len(ext._items)}\n")

    budget = Budget(max_calls=2000, max_input_tokens=40_000_000,
                    max_output_tokens=5_000_000)
    t0 = time.perf_counter()
    for cond, reg in (("A-base", REGISTRY), ("B-extended", ext)):
        matrix = FeatureMatrix(table, reg)
        for s in range(n_seeds):
            run_id = f"newaxes-{cond}-s{s}"
            if (Path("runs") / run_id / "archive.jsonl").exists():
                print(f"  [{run_id}] already there. Skipped")
                continue
            llm = OpenAILLM(LLMConfig(model=MODEL, concurrency=6),
                            feature_names=matrix.feature_names(),
                            shape_values=matrix.shape_value_names(),
                            registry=reg, budget=budget, cache=False)
            loop = RoundLoop(cfg=LoopConfig(run_id=run_id, max_rounds=rounds,
                                            n_rules_per_round=12, seed=7 + s),
                             table=table, matrix=matrix, splits=splits,
                             llm=llm)
            print(f"\n  --- {run_id} (features {len(reg._items)}) ---",
                  flush=True)
            try:
                loop.run(rounds)
            except Exception as e:                          # noqa: BLE001
                print(f"  ★ stopped: {type(e).__name__}: {str(e)[:100]}")
            print(f"  cumulative calls {budget.calls} "
                  f"input {budget.input_tokens:,}"
                  f"  {time.perf_counter() - t0:.0f}s", flush=True)

    # -- were the generated features actually used -------------------------
    print(f"\n{'=' * 78}")
    print("were the generated features used in the final rule — it matters as "
          "much as regret")
    print("=" * 78)
    for s in range(n_seeds):
        d = Path("runs") / f"newaxes-B-extended-s{s}" / "archive.jsonl"
        if not d.exists():
            continue
        with d.open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        best = min(arc, key=lambda e: e["regret"])
        used_best = used_features(best["code"]) & new_names
        used_any: set[str] = set()
        for e in arc:
            used_any |= used_features(e["code"]) & new_names
        print(f"  s{s}  the best rule uses {len(used_best)} "
              f"{sorted(used_best)}")
        print(f"      the whole archive {len(used_any)}/{len(new_names)} "
              f"{sorted(used_any)}")

    print(f"\n  total {time.perf_counter() - t0:.0f}s  "
          f"calls {budget.calls}")
    print("  scoring: python3 experiments/score_new_axes.py")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12,
         int(sys.argv[2]) if len(sys.argv) > 2 else 3)
