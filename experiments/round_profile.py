"""★ It splits one round's time into segments (§5-1). **0 LLM calls**.

    python3 experiments/round_profile.py

Before parallelising, it looks at **what the bottleneck is**. If most of it is
waiting on the LLM, parallelising the scoring and the fitting gains little.

## The two axes are measured separately

```
LLM wait   ★ measured from the `seconds` recorded in a real run's
           `llm_calls/*.json`
           The concurrency is 6, so the wall clock is close to the sum / 6
the rest   ★ measured per segment by running one round with MockLLM
           the static check / the sandbox / fit_weights / the scoring
```

**Measuring both in one run would need LLM calls.** So they are measured
separately and read together — and that fact is written into the result.
"""

from __future__ import annotations

import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
RUNS = Path("runs")


def llm_share() -> None:
    """In a real run, what % of the wall clock is spent waiting on the LLM."""
    print("=" * 74)
    print("A. the LLM wait — from the real run records")
    print("=" * 74)
    print(f"  {'run':28s} {'rounds':>7} {'wall/R':>9} "
          f"{'LLM sum/R':>11} {'/conc 6':>9} {'share':>6}")
    for d in sorted(RUNS.glob("*/rounds.jsonl")):
        run = d.parent
        R = [json.loads(x) for x in d.read_text().splitlines() if x.strip()]
        if not R:
            continue
        secs = [float(json.loads(g.read_text()).get("seconds") or 0.0)
                for g in (run / "llm_calls").glob("*.json")]
        if not any(secs):
            continue
        wall = sum(r["seconds"] for r in R) / len(R)
        llm = sum(secs) / len(R)
        print(f"  {run.name:28s} {len(R):5d} {wall:9.1f} {llm:9.1f} "
              f"{llm / 6:9.1f} {llm / 6 / wall:6.0%}")
    print("  ★ the concurrency is 6, so '/conc 6' is the lower bound of the "
          "LLM wall clock")
    print("    (with retries and serial stretches the real value is larger)")


def non_llm_breakdown() -> None:
    """It runs one round with MockLLM and measures the segments outside the
    LLM."""
    print("\n" + "=" * 74)
    print("B. outside the LLM — one MockLLM round, per segment")
    print("=" * 74)
    warnings.simplefilter("ignore")
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core import loop as loop_mod
    from kernelrule.core import scoring as scoring_mod
    from kernelrule.core import weights as weights_mod
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import Split, SplitSet
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY

    t0 = time.perf_counter()
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    print(f"  loading the table + feature matrix "
          f"{time.perf_counter() - t0:.1f}s "
          "(it is not redone every round)")

    def aligned(p) -> bool:
        x = table.frame_for(p)
        return bool((x.align_a == 8).all() and (x.align_b == 8).all()
                    and (x.align_c == 8).all())

    sh = [p for p in table.shapes() if aligned(p)]
    train = [p for p in sh if 11008 not in (p.N, p.K)]
    val = [p for p in sh if 11008 in (p.N, p.K)]
    splits = SplitSet(train=Split("train", tuple(train)),
                      val=Split("val", tuple(val)))

    acc: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)

    def timed(mod, name, key):
        orig = getattr(mod, name)

        def wrap(*a, **k):
            s = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                acc[key] += time.perf_counter() - s
                cnt[key] += 1
        setattr(mod, name, wrap)
        return orig

    olds = [(loop_mod, "check_rule",
             timed(loop_mod, "check_rule", "static check")),
            (loop_mod, "fit_weights", timed(loop_mod, "fit_weights",
                                            "fit_weights")),
            (loop_mod, "run_isolated", timed(loop_mod, "run_isolated",
                                             "sandbox")),
            (loop_mod, "evaluate_scores",
             timed(loop_mod, "evaluate_scores", "scoring")),
            (loop_mod, "build_report", timed(loop_mod, "build_report",
                                             "diagnostic report"))]
    _ = (scoring_mod, weights_mod)

    from kernelrule.agents.mock import MockLLM
    from kernelrule.core.loop import LoopConfig, RoundLoop

    seed = json.loads(
        (RUNS / "F3rw-p8" / "stage2-rule-writer"
         / "chosen.json").read_text())
    cfg = LoopConfig(run_id="profile", n_rules_per_round=12, max_rounds=1,
                     seed=0, out_dir="/tmp")
    llm = MockLLM("mutate", seed=1, feature_names=matrix.feature_names(),
                  shape_values=matrix.shape_value_names())
    lp = RoundLoop(cfg=cfg, table=table, matrix=matrix, splits=splits, llm=llm)
    lp.seed(seed["code"], seed["w0"])
    # ★ Scoring the seed also calls `fit_weights` — without clearing it, the
    #   round time comes out at 118% (that really was printed).
    acc.clear()
    cnt.clear()
    t0 = time.perf_counter()
    r = lp.run_round()
    wall = time.perf_counter() - t0
    for mod, name, orig in olds:
        setattr(mod, name, orig)

    print(f"\n  round wall clock {wall:.1f}s   proposed {r.n_proposed} "
          f"scored {r.n_scored}")
    print(f"  {'segment':20s} {'sec':>8} {'share':>6} {'calls':>6}")
    for k in ("diagnostic report", "static check", "sandbox", "fit_weights",
              "scoring"):
        print(f"  {k:20s} {acc[k]:8.1f} {acc[k] / wall:6.0%} {cnt[k]:6d}")
    other = wall - sum(acc[k] for k in acc)
    print(f"  {'the rest':20s} {other:8.1f} {other / wall:6.0%}")
    if cnt["fit_weights"]:
        print(f"\n  fit_weights per candidate "
              f"{acc['fit_weights'] / cnt['fit_weights']:.1f}s"
              f"  ({cnt['fit_weights']} candidates scored)")
        print(f"  ★ if all 12 got scored it would be "
              f"{acc['fit_weights'] / cnt['fit_weights'] * 12:.0f}s — "
              "the more duplicates there are, the cheaper this segment looks")
    print("\n  ★ with MockLLM the LLM wait is 0. In a real round these values "
          "are diluted by A's share")


def main() -> None:
    llm_share()
    non_llm_breakdown()
    parallel_speedup()


if __name__ == "__main__":
    main()


def parallel_speedup(workers: int = 6) -> None:
    """★ It runs the same round serially and in parallel and compares **the
    time and the values** (D-95).

    Time alone must not be looked at — a gain only counts if the values are
    the same (principle 29).
    """
    print("\n" + "=" * 74)
    print(f"C. parallelisation — serial vs {workers} workers (MockLLM, the "
          f"same seed)")
    print("=" * 74)
    warnings.simplefilter("ignore")
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.mock import MockLLM
    from kernelrule.core.loop import LoopConfig, RoundLoop
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import Split, SplitSet
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        x = table.frame_for(p)
        return bool((x.align_a == 8).all() and (x.align_b == 8).all()
                    and (x.align_c == 8).all())

    sh = [p for p in table.shapes() if aligned(p)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in sh if 11008 not in (p.N, p.K))),
        val=Split("val", tuple(p for p in sh if 11008 in (p.N, p.K))))
    seed = json.loads(
        (RUNS / "F3rw-p8" / "stage2-rule-writer"
         / "chosen.json").read_text())

    out = {}
    for n in (0, workers):
        llm = MockLLM("mutate", seed=1, feature_names=matrix.feature_names(),
                      shape_values=matrix.shape_value_names())
        lp = RoundLoop(
            cfg=LoopConfig(run_id=f"prof{n}", n_rules_per_round=12,
                           max_rounds=1, seed=0, out_dir="/tmp", n_workers=n),
            table=table, matrix=matrix, splits=splits, llm=llm)
        lp.seed(seed["code"], seed["w0"])
        t0 = time.perf_counter()
        r = lp.run_round()
        el = sorted(lp.archive.cells.values(), key=lambda e: e.rule_id)
        out[n] = (time.perf_counter() - t0, r.n_scored,
                  [(e.rule_id, round(e.regret, 12), tuple(e.w)) for e in el])
        if lp._pool_exec is not None:
            lp._pool_exec.shutdown(wait=True)
    (ts, ns, es), (tp, np_, ep) = out[0], out[workers]
    print(f"  serial   {ts:6.1f}s  scored {ns}")
    print(f"  parallel {tp:6.1f}s  scored {np_}   ★ {ts / tp:.1f}x")
    print(f"  ★ are the values the same: "
          f"{'yes' if es == ep else '★no — there is hidden state'}")
    print("  ⚠️ faster is not better — it is a gain only when the values are "
          "the same (principle 29)")
