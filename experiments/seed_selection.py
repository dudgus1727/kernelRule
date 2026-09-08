"""★ Is picking a seed skill, or selection bias?

    python3 experiments/seed_selection.py [n_seeds] [seed_base] [tag]

## What is checked

Across 6 seeds of the same condition, **the in-sample score predicted the
structural holdout well** (Spearman 0.943). Picking the in-sample minimum
seed gives a holdout of 1.0518, far better than the median of a random seed,
1.0817, and ahead of the vendor's 1.0737 as well.

**But that 1.0518 is not an unbiased estimate.** The minimum of 6 was picked,
so its holdout value is likely to be on the good side of the 6 too. The
higher the correlation, the more so.

So **the procedure is fixed and it is measured again on a new set of seeds.**

```
6 new seeds -> the one in-sample minimum seed -> its holdout value
  near 1.05  it is skill. Settled as a procedure
  near 1.08  it was selection bias
```

## Why the procedure does not break §10.2

The selection signal comes **entirely from the training split**. It is the
same structure as a hyperparameter search — choose on training, measure once
on the holdout.

## Why the prediction works only between seeds

The rules **inside** an archive resemble each other, so a 0.001 difference in
the training score is noise (even picking the holdout best out of the top 10
only goes 1.0817 -> 1.0778). **When the seed differs, the evolutionary
trajectory itself differs and the quality difference is real.** "Choosing
inside an archive" and "choosing a run" are different problems.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.agents.openai_client import DEFAULT_MODEL, Budget, LLMConfig, OpenAILLM
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.canonical import canonical_score
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.scoring import evaluate
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
MODEL = DEFAULT_MODEL   # ★ a single source (D-45)

#: The two earlier sets. Used for the pooled 12-seed distribution.
#: ⚠️ The Korean in the first run id is **a directory name** — it is not
#: translated (D-146).
PRIOR = [f"seedabl-desc-다-noseed-s{s}" for s in range(3)] + \
        [f"newaxes-A-base-s{s}" for s in range(3)]


def _setup(table):
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def main(n_seeds: int = 6, seed_base: int = 20260823, tag: str = "selB",
         feature_detail: str = "full") -> None:
    """⚠️ `seed_base` is **no longer the LLM seed** (D-47).

    The Responses endpoint has no `seed` parameter, and a reasoning model
    refuses `temperature` too. So **the randomness on the LLM side is not
    controlled.** "N seeds" now means N runs with a different
    `LoopConfig.seed` (our own RNG for parent selection and so on), and the
    LLM's non-determinism sits on top of that.

    Part of the seed spread 0.0977 comes from here — it is the part that
    cannot be controlled.
    """
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    splits = _setup(table)

    budget = Budget(max_calls=3000, max_input_tokens=60_000_000,
                    max_output_tokens=8_000_000)
    print("=" * 76)
    print(f"seed selection check — {n_seeds} new seeds  [{MODEL}]  tag={tag}")
    print("=" * 76)
    print(f"  the condition: no seed rule + the base 24 features + "
          f"feature_detail={feature_detail}")
    print(f"  training {len(splits.train.shapes)} / structural holdout "
          f"{len(splits.val.shapes)}\n")

    t0 = time.perf_counter()
    for s in range(n_seeds):
        run_id = f"{tag}-s{s}"
        if (Path("runs") / run_id / "archive.jsonl").exists():
            print(f"  [{run_id}] already there. Skipped")
            continue
        llm = OpenAILLM(LLMConfig(model=MODEL, concurrency=6,
                                  feature_detail=feature_detail),
                        feature_names=matrix.feature_names(),
                        shape_values=matrix.shape_value_names(),
                        registry=REGISTRY, budget=budget, cache=False)
        loop = RoundLoop(cfg=LoopConfig(run_id=run_id, max_rounds=12,
                                        n_rules_per_round=12, seed=100 + s),
                         table=table, matrix=matrix, splits=splits, llm=llm)
        print(f"\n  --- {run_id} ---", flush=True)
        try:
            loop.run(12)
        except Exception as e:                              # noqa: BLE001
            print(f"  ★ stopped: {type(e).__name__}: {str(e)[:100]}")
        print(f"  cumulative calls {budget.calls}  "
              f"{time.perf_counter() - t0:.0f}s",
              flush=True)

    # -- scoring ----------------------------------------------------------
    def score(run_id: str):
        f = Path("runs") / run_id / "archive.jsonl"
        if not f.exists():
            return None
        with f.open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        if not arc:
            # ★ An empty archive is not "a bad run" but **a run that did not
            #   happen**. Quietly entering it as 0 in the scoring pollutes
            #   the distribution (§26.4).
            print(f"  {run_id:16s} ⚠️ the archive is empty — excluded from "
                  f"the scoring")
            return None
        best = min(arc, key=lambda e: e["regret"])
        return canonical_score(best["code"], best["w"], table=table,
                               matrix=matrix, splits=splits)

    v = evaluate(vendor_order_fn(table, load_vendor(VENDOR),
                                 mapping="nearest"),
                 table, list(splits.val.shapes), ks=(1,))

    print(f"\n{'=' * 76}")
    print(f"the new set of {n_seeds} seeds — ★ chosen in-sample, and the "
          f"holdout is looked at only once")
    print("=" * 76)
    print(f"  {'run':16s} {'in-sample':>10} {'struct HO':>10}")
    new = []
    for s in range(n_seeds):
        r = score(f"{tag}-s{s}")
        if r is None:
            continue
        new.append((f"{tag}-s{s}", r.in_sample, r.holdout))
        print(f"  {f'{tag}-s{s}':16s} {r.in_sample:10.4f} {r.holdout:10.4f}")
    if not new:
        return
    pick = min(new, key=lambda x: x[1])
    ho = np.array([x[2] for x in new])
    print(f"\n  ★ the in-sample minimum seed: {pick[0]}   "
          f"struct HO {pick[2]:.4f}")
    print(f"     random-seed median {np.median(ho):.4f}  "
          f"worst {ho.max():.4f}")
    print(f"     vendor {v.at(1):.4f}   the earlier set's chosen value "
          f"1.0518")
    # ★ It does not judge automatically (D-46, D-50). "the minimum of N" is
    #   an optimism bias in itself, and the threshold came from **another
    #   model's data**. The judgement only comes from fixing the procedure
    #   and measuring on **a new set**.
    print(f"     ⚠️ this value is the minimum of {len(new)} — that is an "
          f"optimism bias in itself (D-50).\n        Before the procedure is "
          f"fixed and it is measured on **a new set**, it is not an estimate.")

    # -- pooling the sets ---------------------------------------------------
    #    ★ Only **the same model and endpoint** are pooled (the fifth axis of
    #      D-31). PRIOR is gpt-5.4 + chat, so it must not be mixed with the
    #      luna runs.
    import json as _json
    def _model_of(run: str) -> str:
        f = Path("runs") / run / "config.json"
        if f.exists():
            try:
                c = _json.loads(f.read_text()).get("llm", {})
                return f"{c.get('model','?')}/{c.get('endpoint','chat')}"
            except Exception:                               # noqa: BLE001
                pass
        # From the time before config.json — **only the model** is recovered
        # from llm_calls.
        # ★ The endpoint is not left anywhere. Asserting `/chat` labels a
        #   responses run as chat and **breaks D-31 without our even knowing
        #   it was broken.** What is unknown is written down as unknown
        #   (§26.4).
        for g in sorted((Path("runs") / run / "llm_calls").glob("*.json"))[:1]:
            try:
                m = _json.loads(g.read_text()).get("model", "?")
                return f"{m}/endpoint-unknown"
            except Exception:                               # noqa: BLE001
                pass
        return "unknown"

    here = _model_of(f"{tag}-s0")
    same = [r for r in PRIOR if _model_of(r) == here]
    print(f"\n{'=' * 76}")
    print(f"pooling the sets — ★ only within the same condition ({here})")
    print("=" * 76)
    if not same:
        print("  the earlier sets have a different condition — they are not "
              "pooled (D-31).")
        print(f"  This set of {len(new)} seeds alone cannot estimate the "
              f"spread.")
        ho2 = sorted(x[2] for x in new)
        print(f"  {tag} struct HO  " + "  ".join(f"{x:.4f}" for x in ho2)
              + f"   width {ho2[-1] - ho2[0]:.4f}")
        return
    allr = []
    for run in same:
        r = score(run)
        if r:
            allr.append((run, r.in_sample, r.holdout))
    allr += new
    a = np.array([[x[1], x[2]] for x in allr])
    from kernelrule.features.validate import _pearson, _spearman
    print(f"  n={len(allr)}   struct HO median {np.median(a[:, 1]):.4f}  "
          f"min {a[:, 1].min():.4f}  max {a[:, 1].max():.4f}  "
          f"width {a[:, 1].max() - a[:, 1].min():.4f}")
    print(f"  in-sample vs holdout  Spearman {_spearman(a[:, 0], a[:, 1]):.3f}"
          f"  Pearson {_pearson(a[:, 0], a[:, 1]):.3f}")
    print(f"  picking the in-sample minimum over everything: "
          f"{min(allr, key=lambda x: x[1])[0]} "
          f"-> {min(allr, key=lambda x: x[1])[2]:.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6,
         int(sys.argv[2]) if len(sys.argv) > 2 else 20260823,
         sys.argv[3] if len(sys.argv) > 3 else "selB",
         sys.argv[4] if len(sys.argv) > 4 else "full")
