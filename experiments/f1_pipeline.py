"""★ The F0~F3 pipeline — the LLM builds everything from features to rules
(§30.9).

    python3 experiments/f1_pipeline.py F1 --dry-run
    python3 experiments/f1_pipeline.py F1 --n-features 20 --n-seeds 3
    python3 experiments/f1_pipeline.py F1 --stage 2   # reuse stage-1 output

## Why it is needed

Every evolution run so far **combined material a human made**.

    features  the 24 in `features/physical.py`  <- written by a human from
                                                   the physics documents
    seed      `rules/human_guided.py`           <- written by a human. It
                                                   uses 6 of those 24
    loop      picking 8 of the 24

`experiments/feature_writer.py` confirmed "the LLM can build features". But
those features were only ever tested **on top of the human 24**. What has not
been done yet is the complete form of the fundamental question.

    "can a rule be built from F1 features alone"

## Three stages

    1  FeatureWriter  raw values only -> N features   -> stage1-features/
    2  RuleWriter     the stage-1 feature list only -> a seed rule
                                                      -> stage2-rule-writer/
    3  RoundLoop      the stage-1 registry + the stage-2 seed
                                                      -> stage3-evolution/

**Under F1/F0 the 24 a human made enter none of the three stages.** What
guarantees that is "no registry default" in `features/__init__.py`, and the
AST check in `tests/test_features.py` pins it.

## The conditions

    F3  REGISTRY (the human 24) + a seed        = every run so far
    F2  ★ the 5 public facts + extension by FeatureWriter
        (F1-K before the rename, D-128)
    F1  start from 0 -> FeatureWriter -> a RuleWriter seed  ★ the fundamental
                                                              question

**It is a 0 -> 5 -> 24 ladder.** All three have been run.

★ The old name `F0` (no features) and the old `F2` (5 raw physical
quantities) were **deleted** — both have 0 runs. The old
`F1-K` is today's `F2` (D-128).

F3 must run **through this path** too. Running it through another script
mixes path differences into the result (§26.2).
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import kernelrule.features.physical  # noqa: F401  — it fills REGISTRY
from kernelrule.agents.mock import MockLLM
from kernelrule.agents.openai_client import DEFAULT_MODEL, Budget, LLMConfig
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet, check_balance, experiment_shapes
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY, FeatureRegistry
from kernelrule.features.generated import (
    SHAPE_LEVEL_REASON,
    FeatureRejected,
    register_generated,
)
from kernelrule.features.validate import alt_hw
from kernelrule.report.table_facts import TableFacts
from kernelrule.rules.checks import fitter_for

#: The default table. ★ Changed with `--bundle` / `--env-hash` — from the
#: 5090 transfer on it must not be hardcoded. **It is recorded in
#: config.json per run.**
BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
BUNDLE_HASH = "c63710df"
OUT = Path("runs")

#: ★ The F2 pre-registration. It is **the same content** as
#: `docs/artifacts/f2-preregistration.md`, and `tests/test_f2_prereg.py`
#: checks that they do not diverge.
#: **It was nailed down without calling the LLM even once** — written just
#: before a run, the feel picked up while building the plumbing seeps into
#: the criteria (D-50).
#:
#: ⚠️ 2026-09-08 (D-146): the values were translated into English together
#: with the document. It is a frozen record, so nothing was deleted — the
#: Korean original is at commit `ee53b4d`.
F2_PREREG = {
    "purpose": ("given known axes, does it make more new ones. And does the "
                "library get better"),
    "expected": ("the number of new axes is larger than F1's (no budget is "
                 "spent on rediscovery). ★ Whether the evolution performance "
                 "catches up with F3 is not known. Not catching up is not a "
                 "failure. Whether it is better than F1 is an open question "
                 "too."),
    "start_library": 5,
    "areas": 7,
    "per_category": 3,
    # ⚠️ This key is a **frozen record**. D-93 renamed the roles, but the
    #   pre-registration was written under the name of that time — changing
    #   it would be rewriting the record (documentation rule 2). The live
    #   name is `--n-rule-writer`.
    "n_architect": 10,
    "n_seeds": 6,
    "rounds": 12,
    "primary_metrics": [
        "the number of new axes and their correlation against the human 24",
        ("the structural holdout after evolution (the same final scoring "
         "procedure as F1)")],
    "not_a_criterion": ("the number of rediscoveries — five were given, so of "
                        "course it drops. The condition is not evaluated by "
                        "this"),
    "two_variables": ("the start 5 + the GPU example. **They are not "
                      "separated** — both are part of 'public knowledge is "
                      "given'. It is an exception to D-31, and 'which of the "
                      "two did it' cannot be separated by this experiment"),
    "on_failure": {
        "an area refused 3 times in a row": "skip it and record it. Continue",
        "acceptance under half": (
            "stop and ★ look at the refusal-reason distribution first — "
            "mostly duplicates is normal (five were given), mostly §8.3 "
            "failures is a checker or field problem, mostly schema failures "
            "is a prompt problem. The order is infrastructure -> checker -> "
            "subject (principle 8)"),
        "all RuleWriter calls refused": "stop and report",
        "3 runs in a row with an empty archive": "stop"},
    "not_doing": [
        "the remaining 19 are not put in — that is the F3 condition",
        "--recategorize is not used — the fixed seven",
        ("the F1 results are not deleted — they are what it is compared "
         "against"),
        "the model is not changed (D-45, principle 25)",
        "the prompt is not fixed after seeing the results (§12.3d)"],
    "threshold_rationale": ("half is 'the minimum for the experiment to hold', "
                            "not a level tightened against F1's measurement "
                            "(80%). F2 was given five, so duplicate refusals "
                            "can rise and that is normal behaviour"),
    "budget_calls": 990,
    "discrimination_note": ("because of the seed spread sigma=0.0274, a "
                            "difference of the 0.02 class between conditions "
                            "cannot be told apart (D-53). If they are similar, "
                            "'indistinguishable' is the honest description"),
}

#: The range for how many areas the LLM partitions into. **A human does not
#: decide the areas** — deciding them hands over prior knowledge. Only the
#: count range is given (§30.10).
CAT_MIN, CAT_MAX = 5, 8

#: After this many consecutive refusals in one area, that area is skipped.
#: "this area is hard to express from raw values" stays in the record.
CAT_GIVE_UP = 3


def _plan(cats: list[dict], n_features: int, per_cat: int) -> list[str | None]:
    """What to build and how many times. ★ The count is **derived from the
    number of areas** (§30.10).

    It used to be fixed at 20. That is arbitrary, and forced features appear
    just to fill it. With 6 areas it is `6 x per_cat`, and with no areas
    (free generation) it is `n_features` times, as before.
    """
    if not cats:
        return [None] * n_features
    plan: list[str | None] = []
    for _ in range(per_cat):   # round robin — so one area does not pile up
        plan.extend(c["name"] for c in cats)
    return plan


#: ★ How many **branchable** axes the FeatureWriter is asked for over a
#: session — shape level (computed from `p`/`hw` alone) and not constant on
#: the scored shapes (D-170 §4-3).
#:
#: **3.** known7 already carries three (`roofline_ratio`, `log_min_dim`,
#: `log_flops`), so three generated ones bring F2 to six, against F3's
#: eight. The point is to get within reach of the human library's branching
#: material, not past it.
#:
#: ⛔ Not raised above 3. There is a cheap way to satisfy a large quota —
#: alignment has four distinct values on this table, so any number of
#: "alignment variants" would count while measuring one thing. The
#: duplication check refuses them one by one; a bigger quota just pays the
#: model to keep trying.
MIN_BRANCHABLE = 3


def _n_branchable(gen: FeatureRegistry, base: FeatureRegistry) -> int:
    """How many branchable axes have been built **so far in this session**.

    ★ "not constant on the scored shapes" needs no measurement here: since
    D-170 §2 the constant check runs over the whole experiment population,
    so a shape-level feature that survived registration necessarily varies
    on it. Before that change this count would have been a lie — the ten
    libraries of the stage-1 sweep each registered a shape-level axis and
    every one of them was constant on the scored shapes.
    """
    return sum(1 for n in set(gen._items) - set(base._items)
               if gen[n].shape_level)


def _branch_block(gen: FeatureRegistry, base: FeatureRegistry) -> str:
    """The live branchable-axis count for the prompt (D-170 §4-3)."""
    n = _n_branchable(gen, base)
    names = sorted(n_ for n_ in set(gen._items) - set(base._items)
                   if gen[n_].shape_level)
    if n >= MIN_BRANCHABLE:
        return (f"\n\n★ Branchable axes so far: **{n}** of the "
                f"{MIN_BRANCHABLE} asked for ({names}). The quota is met — "
                f"build whatever this area needs.")
    return (f"\n\n★ Branchable axes so far: **{n}** of the "
            f"{MIN_BRANCHABLE} asked for"
            + (f" ({names})." if names else ".")
            + " A branchable axis is computed from `p` and `hw` only — no "
              "`cfg` — so a rule can split on it. If this area admits one, "
              "make this one branchable.")


def _task(cat: str | None, cats: list[dict], made_in: dict[str, list[str]],
          gen: FeatureRegistry, base: FeatureRegistry) -> str:
    """The instruction for this proposal. With an area, it builds within
    that area."""
    if cat is None:
        made = sorted(set(gen._items) - set(base._items))
        tail = (f"\n\nBuilt so far: {made}. Find an axis different from "
                "these." if made else "")
        return ("## What to build now\n\nPropose one feature." + tail
                + _branch_block(gen, base))
    desc = next(c["description"] for c in cats if c["name"] == cat)
    mine = made_in.get(cat, [])
    other = sorted(set(gen._items) - set(base._items) - set(mine))
    return (f"## What to build now\n\n**Area: `{cat}`** — {desc}\n\n"
            f"Propose one physical quantity in this area.\n\n"
            f"```\nalready built in this area: {mine or 'none'}\n"
            f"from other areas (for duplicate checking): {other or 'none'}\n"
            "```\n\n"
            "★ Stay **within this area**. Wandering into another area leaves "
            "nothing to build\nwhen that area's turn comes."
            + _branch_block(gen, base))


#: ★ How many times the FeatureWriter is asked again when §8.3 refuses its
#: candidate (D-170 §3).
#:
#: **3, the same as everywhere else** (`LLMConfig.max_retries`, and what
#: D-164 gave RuleWriter). The evidence we have for a number is D-164's:
#: with the refusal fed back at 3 retries, stage 2 went from 3/10 seeds
#: surviving to 10/10. Picking a different number here would mean claiming
#: something that measurement has not said.
#:
#: ⛔ Not unlimited. A slot that is refused 4 times **stays empty** — the
#: library gets smaller and that fact is recorded (`rejections` in
#: summary.json). An axis the model cannot express after four tries is a
#: result, not something to spend budget on.
FEATURE_RETRIES = 3

#: How many times to call again when a RuleWriter output is caught by the
#: static checks.
#: ★ If all of them fail it is an **error** — it does not silently proceed
#: without a seed (§26.4).
ARCH_RETRIES = 3


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def _aligned_shapes(table: PerfTable) -> list:
    """⚠️ 2026-09-11 (D-167 §R): the body moved to
    `kernelrule.core.splits.experiment_shapes`. The same predicate was
    copied into 36 places across 35 files; the name is kept here because
    the artefacts and the documents refer to it. That move changed no value
    — 61 shapes on the A6000 table before and after.

    ⚠️ 2026-09-13 (D-170 §1): **the criterion behind that function changed**
    — alignment 8 became "more than one kernel family", and the A6000
    population went 61 -> 65. The name `_aligned_shapes` is now a
    misnomer kept for the artefacts that refer to it; it is not renamed
    because the recorded runs point at it. What it returns is whatever
    `experiment_shapes` says today.
    """
    return experiment_shapes(table)


def _population_criterion() -> str:
    """The population criterion **as the library states it today** (D-170
    §1). A run records this string, so a run recorded before 2026-09-13 and
    one recorded after can be told apart without guessing."""
    from kernelrule.core.splits import (
        KERNEL_FAMILY_COLUMN,
        MIN_KERNEL_FAMILIES,
    )

    return (f"at least {MIN_KERNEL_FAMILIES} distinct "
            f"{KERNEL_FAMILY_COLUMN} in the candidate space "
            f"(kernelrule.core.splits.experiment_shapes, D-170 §1)")


def _wire_feature_retry(llm, *, table, matrix, hw_alt, gen, train_shapes
                        ) -> bool:
    """★ Put the §8.3 refusal **inside the retry path** (D-170 §3).

    `f1_pipeline.py:535` caught `FeatureRejected` and moved to the next
    slot, so a refusal never reached the model and the slot simply emptied.
    Fixing what the constant check looks at (D-170 §2) without this would
    not have produced different axes — it would have produced a smaller
    library.

    Returns whether it was wired (a `MockLLM` has no hook — a dry run is not
    silently treated as if it had one).
    """
    from kernelrule.features.generated import check_generated

    if not hasattr(llm, "set_feature_check"):
        return False

    def check(name: str, code: str, meta: dict) -> None:
        check_generated(code, registry=gen, meta=meta, table=table,
                        matrix=matrix, hw_alt=hw_alt,
                        train_shapes=train_shapes)

    llm.set_feature_check(check)
    return True


def _shape_population(table: PerfTable, splits: SplitSet) -> dict:
    """★ 2026-09-11 (D-167 §Q): **which shapes the run actually used.**

    `config.json` recorded `split_kind` and stopped there, so a reader saw
    `kfold0-seed12345` and understood "the 66 shapes in three parts". The
    run is on 61 — `_aligned_shapes` drops 5 — and nothing said so. A
    number needs its procedure attached (documentation rule 1).

    ⚠️ Everything here is **computed from `table` and `splits`.** Writing
    66/61/5 as literals makes the record a lie on the next bundle.
    """
    used = _aligned_shapes(table)
    keep = {p.key for p in used}
    dropped = [p for p in table.shapes() if p.key not in keep]
    return {
        "table": len(table.shapes()), "used": len(used),
        "excluded": len(dropped),
        # ★ Read out of the library, not written here. When the criterion
        #   changed (D-170 §1) a literal here would have kept claiming the
        #   old one.
        "criterion": _population_criterion(),
        "excluded_shapes": [f"{p.M}x{p.N}x{p.K}" for p in dropped],
        "n_train": len(splits.train.shapes), "n_val": len(splits.val.shapes)}


#: ★ Which shape population a run was scored on (D-170 §1).
#:
#:   "family"  the current criterion — more than one kernel family (65 on
#:             the A6000 table)
#:   "align8"  the criterion until 2026-09-13 — alignment 8 everywhere (61)
#:
#: ⛔ A **recorded** run must be re-scored on the population it ran on.
#: `_population_of` reads that out of the run's own `config.json`, so
#: re-scoring an old rule does not silently move its number.
POPULATIONS = ("family", "align8")


def _population_of(cfg: dict) -> str:
    """Which population a recorded run used, from its `config.json`.

    A run from before 2026-09-11 has no `shape_population` block at all
    (D-167 §Q added it) and a run from before 2026-09-13 records the
    alignment criterion — **both are `align8`**. Only the string naming the
    kernel-family criterion means `family`.
    """
    crit = ((cfg.get("shape_population") or {}).get("criterion") or "")
    return "family" if "pipeline_kind" in crit else "align8"


def _population_shapes(table: PerfTable, population: str) -> list:
    from kernelrule.core.splits import aligned_shapes

    if population not in POPULATIONS:
        raise ValueError(f"unknown population: {population!r}. {POPULATIONS}")
    return (experiment_shapes(table) if population == "family"
            else aligned_shapes(table))


def _splits(table: PerfTable, *, fold: int | None = None,
            split_seed: int = 12345, k: int = 3,
            population: str = "family",
            design: str = "kfold") -> SplitSet:
    """The split. The default is the **structural split** (the 11008 layer
    held out whole, §10.1).

    ★ With `fold` it is a **stratified random k-fold** (D-144). The two
    splits ask different things:

    ```
    k-fold   "does the result hold when the split changes"  — random, so
             M·N·K are mixed across both sides
    11008    "does it work on a dimension value never seen" — one value is
             removed whole
    ```

    ⚠️ `split_seed` is **the randomness that builds the folds**. Separating
    it from the loop's evolution seed is what lets "because the split
    differed" be told from "because the evolution differed".
    """
    # ⚠️ 2026-09-13 (D-170 §1): the default population is the **current**
    #   one (65). Re-scoring a rule from a recorded run needs
    #   `population=_population_of(cfg)` — the folds are built from the
    #   shape list, so a different population is a different split, and the
    #   number would move with no sign of it.
    shapes = _population_shapes(table, population)
    if design not in ("kfold", "nkgroup"):
        raise ValueError(f"unknown split design: {design!r}")
    if design == "nkgroup":
        # ★ D-171 §1 — a whole (N,K) group on one side. `fold` picks which.
        #   ⚠️ It takes **no seed**: the assignment is deterministic, so
        #   there is no split randomness to separate from the evolution
        #   seed here.
        from kernelrule.core.splits import nk_group_folds

        if fold is None:
            raise ValueError(
                "the nkgroup design has no default fold — state --fold. "
                "Guessing one would put a different holdout under the same "
                "name (§26.4)")
        s = nk_group_folds(shapes, k=k)[fold]
        check_balance(s.train, table.hw)
        return s
    if fold is not None:
        from kernelrule.core.splits import stratified_kfold

        s = stratified_kfold(shapes, table.hw, k=k, seed=split_seed,
                             fold=fold)[0]
    else:
        held = [p for p in shapes if 11008 in (p.N, p.K)]
        s = SplitSet(train=Split("train", tuple(p for p in shapes
                                                if p not in held)),
                     val=Split("val", tuple(held)), kind="nk11008")
    check_balance(s.train, table.hw)
    return s


def _base_registry(condition: str) -> FeatureRegistry:
    """The **starting registry** the condition decides.

    ⚠️ 2026-09-11 (D-166 §M): the body moved to
    `kernelrule.features.loader.base_registry` — `run_registry` rebuilds a
    finished run's axis list and needs the same first layer. One copy
    (principle 2); the three conditions and the 0 -> 5 -> 24 ladder are
    unchanged.
    """
    from kernelrule.features.loader import base_registry
    # ★ `human=` is explicit: the library must not reach for the global
    #   registry (§30.9), so the human list is handed in from here.
    return base_registry(condition, human=REGISTRY)


def _make_llm(a, *, registry: FeatureRegistry, budget: Budget,
              table=None):
    """★ `registry` is mandatory — which feature list goes into the prompt
    is the experimental condition itself (§30.9). MockLLM receives the same
    list.

    ★ `table` decides **which bundle the hardware facts are built from**
    (D-113). Without it, calling RuleWriter fails — falling back to a
    default silently sends another GPU's facts.
    """
    hw_text = None
    if table is not None:
        from kernelrule.agents.hwprompt import (
            check_hw_prompt,
            hw_prompt_from_bundle,
        )
        # ★ D-166: no `table=`. The measurement-limit section was what
        #   needed it, and that section (with its answer-derived `min_ms`)
        #   is gone.
        hw_text, _facts = hw_prompt_from_bundle(
            a.bundle, env_hash=getattr(a, "env_hash", None))
        # ★ Is what was built the same hardware as this table? Verified in
        #   reverse — name + this bundle's effective numbers (D-166 §E-③).
        check_hw_prompt(hw_text, table.hw)
    names = sorted(n for n in registry._items if not registry[n].shape_level)
    svals = sorted(n for n in registry._items if registry[n].shape_level)
    if a.dry_run:
        return MockLLM("mutate", seed=a.seed, feature_names=names,
                       shape_values=svals)
    from kernelrule.agents.openai_client import OpenAILLM
    # ★ The objective is passed all the way to the prompt (D-101). Without
    #   it the acceptance criterion and what the model is told diverge.
    #   ★ Since D-128 evolution is regret only — the rank loss stays as a
    #   metric (D-118 · D-121).
    return OpenAILLM(LLMConfig(model=a.model, concurrency=6,
                               objective="regret",
                               # ★ D-170 §3 — stated rather than inherited.
                               #   The FeatureWriter's §8.3 refusals now use
                               #   this path, so the number belongs where it
                               #   can be read.
                               max_retries=FEATURE_RETRIES,
                               parameters=getattr(a, "parameters", None),
                               product_hint=getattr(
                                   a, "product_hint", False),
                               power_hint=getattr(
                                   a, "power_hint", False),
                               hw_text=hw_text),
                     feature_names=names, shape_values=svals,
                     registry=registry, budget=budget, cache=False)


def _dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------
# Stage 1 — FeatureWriter
# ---------------------------------------------------------------------------
def stage1(a, d: Path, table, matrix, base: FeatureRegistry,
           train_shapes=None) -> FeatureRegistry:
    """Builds features. ★ Each proposal is appended immediately (D-33).

    **The areas are partitioned first** (§30.10). It used to loop over
    `range(20)` instructing only "an axis different from what you have built
    so far", and with no direction the first few circled near where they
    landed — of luna's 17, 3 were `split_k_*` and 5 were `cta_*`. And the
    number 20 is arbitrary, so forced features appear just to fill it.

        1  partition the areas (1 LLM call)  -> categories.json
        2  generate per area (N per area)    -> the count is derived from
                                                the number of areas

    **A human does not give the areas** — that would hand over prior
    knowledge. How the LLM partitions is itself an object of observation.
    """
    out = d / "stage1-features"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "proposals.jsonl"
    log.write_text("")

    gen = FeatureRegistry(f"{a.condition}-generated")
    for n in sorted(base._items):     # F2/F3 build on top of the base
        gen.add(base[n])

    # ★ It **extends** an existing library (the D-63 follow-up). The
    #   accepted features of the earlier run must go into the registry first
    #   so that (1) the duplication verdict is made against them too and
    #   (2) the artefact is the union. The earlier proposal history is
    #   copied in front as it is, so the format `load_generated` reads is
    #   preserved.
    prior_lines: list[str] = []
    if a.extend_from:
        src = Path(a.extend_from) / "proposals.jsonl"
        if not src.exists():
            raise SystemExit(f"{src} does not exist.")
        from kernelrule.features.loader import load_generated
        for f in load_generated(src, exclude=set(base._items), table=table):
            gen.add(f)
        prior_lines = [ln for ln in src.read_text().splitlines() if ln.strip()]
        print(f"  ★ carried over {len(gen._items) - len(base._items)} from "
              f"{src} — the artefact is the **union**\n")
    n_base = len(gen._items)
    n_prior = n_base - len(base._items)
    if prior_lines:               # the carried-over history goes in front
        log.write_text("\n".join(prior_lines) + "\n")

    llm = _make_llm(a, registry=gen, budget=Budget(max_calls=a.n_features * 4),
                    table=table)
    hw_alt = alt_hw(table.hw)
    # ★ D-170 §3 — a refusal is handed back to the model instead of
    #   emptying the slot.
    retry_wired = _wire_feature_retry(llm, table=table, matrix=matrix,
                                      hw_alt=hw_alt, gen=gen,
                                      train_shapes=train_shapes)
    print(f"  ★ §8.3 refusals go back to the model: {retry_wired} "
          f"(up to {FEATURE_RETRIES} retries per slot, D-170 §3)")
    rejects: dict[str, int] = {}
    t0 = time.perf_counter()

    # -- 1. partition the areas -------------------------------------------
    cats: list[dict] = []
    cat_notes = ""
    if (a.categorize or a.categorize_only) and not a.recategorize:
        # ★ The fixed list (§30.18). 0 LLM calls.
        from kernelrule.agents.openai_client import load_prompt

        block = load_prompt("areas.md")
        body = block[block.index("```") + 3:block.rindex("```")]
        # Split on `name | description`. Splitting on whitespace would cut
        # "compute throughput" at "compute".
        cats = [{"name": ln.split("|", 1)[0].strip(),
                 "description": ln.split("|", 1)[1].strip()}
                for ln in body.splitlines() if "|" in ln]
        cat_notes = ("the fixed list (prompts/areas.md). --recategorize "
                     "draws it again")
        _dump_json(out / "categories.json",
                   {"categories": cats, "notes": cat_notes, "source": "fixed"})
        print(f"  ★ {len(cats)} fixed areas (0 LLM calls):")
        for c in cats:
            print(f"     {c['name']:16s} {c['description'][:56]}")
        print()
        if a.categorize_only:
            raise SystemExit(0)
    elif a.categorize or a.categorize_only:
        try:
            res = llm.complete("categorize", "", n_min=CAT_MIN, n_max=CAT_MAX)
            cats = [dict(c) for c in res["categories"]]
            cat_notes = res.get("notes", "")
        except Exception as e:                              # noqa: BLE001
            # ★ It does not silently fall back to free generation — that
            #   changes the condition (§26.4)
            raise RuntimeError(
                f"partitioning the areas failed: {type(e).__name__}: {e}. "
                f"It does not silently fall back to free generation — that "
                f"is a different condition (F1-free). State it with "
                f"`--no-categorize`.") from e
        _dump_json(out / "categories.json",
                   {"categories": cats, "notes": cat_notes,
                    "n_min": CAT_MIN, "n_max": CAT_MAX})
        print(f"  ★ the LLM partitioned into {len(cats)} areas:")
        for c in cats:
            print(f"     {c['name']:32s} {c['description'][:60]}")
        if cat_notes:
            print(f"     (left out) {cat_notes[:100]}")
        print()
        if a.categorize_only:
            # ★ Diagnosis only. Nothing is generated (D-63).
            print("  ★ --categorize-only — stopping here. "
                  f"record: {out / 'categories.json'}")
            raise SystemExit(0)

    #: Area -> the names built in that area. Fed back into the prompt.
    made_in: dict[str, list[str]] = {c["name"]: [] for c in cats}
    #: Area -> consecutive refusals. At 3 that area is skipped.
    streak: dict[str, int] = dict.fromkeys(made_in, 0)
    skipped: list[str] = []

    def dump() -> None:
        made = sorted(set(gen._items) - set(base._items))
        _dump_json(out / "summary.json", {
            "shape_level": {n: SHAPE_LEVEL_REASON[n]
                            for n in sorted(gen._items)
                            if gen[n].shape_level and n in SHAPE_LEVEL_REASON},
            "shape_level_needs_recheck": sorted(
                n for n in gen._items if gen[n].shape_level
                and "Re-judgement" in SHAPE_LEVEL_REASON.get(n, "")),
            "categories": [c["name"] for c in cats],
            "made_by_category": made_in,
            "skipped_categories": skipped,
            "condition": a.condition, "model": a.model, "dry_run": a.dry_run,
            "n_planned": n_planned, "n_base": n_base, "n_prior": n_prior,
            "extend_from": a.extend_from, "only_category": a.only_category,
            "per_category": a.per_category, "categorize": a.categorize,
            "n_accepted": len(made) - n_prior,  # ★ only what was built now
            "n_total": len(made),               # including carried-over
            "rejections": rejects, "seconds": round(time.perf_counter() - t0, 1),
            # ★ D-170 §3: did the refusal actually reach the model, and how
            #   often. A retry that leaves no record cannot be counted.
            "feature_retry_wired": retry_wired,
            "feature_retries_allowed": FEATURE_RETRIES,
            "violations": (llm.violation_report()
                           if hasattr(llm, "violation_report") else None),
            "feature_names": made,
            "physics_coverage": _physics_coverage(table, gen, base)})
        (out / "features.py").write_text(_features_module(gen, base))
        if hasattr(llm, "dump"):
            llm.dump(out / "llm_calls")

    if a.only_category:
        # A partial match. The LLM names the areas afresh every time, so
        # they cannot be pinned exactly — it is a compromise for
        # strengthening a particular axis **while keeping the principle that
        # a human does not define the areas** (D-63).
        # Several keywords with `|`. **The LLM names the areas afresh every
        #  time and the language changes too** — the first round was Korean
        #  (`산술_대역폭_압력`) and the second English
        #  (`roofline_pressure`). Several conceptual keywords are used.
        keys = [k.strip() for k in a.only_category.split("|") if k.strip()]
        hit = [c for c in cats
               if any(k in c["name"] or k in c["description"] for k in keys)]
        if not hit:
            raise SystemExit(
                f"no area matches the keywords {keys}. "
                f"What came out this time: {[c['name'] for c in cats]}")
        cats = hit[:1]
        made_in = {cats[0]["name"]: []}
        streak = {cats[0]["name"]: 0}
        print(f"  ★ generating for the area {cats[0]['name']!r} only — a "
              f"**separate condition**. Do not mix it into the comparison "
              f"table\n")
    plan = _plan(cats, a.n_features, a.per_category)
    n_planned = len(plan)   # ★ area-based, the count is decided here
    try:
        for i, cat in enumerate(plan):
            if cat is not None and cat in skipped:
                continue
            row: dict = {"i": i, "category": cat}
            try:
                res = llm.complete("feature", "", condition=a.condition,
                                   registry=gen,
                                   task=_task(cat, cats, made_in, gen, base))
                row.update({k: res.get(k) for k in
                            ("name", "code", "unit", "direction",
                             "expected_range", "rationale")})
                f = register_generated(res["code"], registry=gen, meta=res,
                                       table=table, matrix=matrix,
                                       hw_alt=hw_alt,
                                       # ★ D-166 — see the loop's call site
                                       train_shapes=train_shapes)
                row["accepted"] = True
                row["shape_level"] = f.shape_level
                if f.shape_level:
                    row["shape_level_reason"] = SHAPE_LEVEL_REASON.get(f.name)
                if cat is not None:
                    made_in[cat].append(f.name)
                    streak[cat] = 0
                print(f"  #{i:02d}  ✓ {f.name}"
                      + (f"   [{cat}]" if cat else ""))
            except FeatureRejected as e:
                row.update(accepted=False, error=str(e)[:200])
                key = str(e).split(":")[0][:40]
                rejects[key] = rejects.get(key, 0) + 1
                print(f"  #{i:02d}  ✗ {str(e)[:90]}")
                if cat is not None:
                    streak[cat] += 1
                    if streak[cat] >= CAT_GIVE_UP:
                        skipped.append(cat)
                        print(f"       ★ skipping the area {cat!r} — "
                              f"{CAT_GIVE_UP} consecutive refusals. "
                              f"'hard to express from raw values' is "
                              f"recorded")
            except Exception as e:                          # noqa: BLE001
                row.update(accepted=False, error=f"{type(e).__name__}: {e}"[:200])
                rejects[type(e).__name__] = rejects.get(type(e).__name__, 0) + 1
                print(f"  #{i:02d}  ✗ {type(e).__name__}: {str(e)[:70]}")
            with log.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        dump()               # ★ it survives even if it dies midway (D-33)

    print(f"\n  accepted this time {len(gen._items) - n_base}/{n_planned}  "
          f"registry total {len(gen._items)}"
          + (f"  ({len(cats)} areas x {a.per_category})" if cats else ""))
    return gen


#: The six terms `human_guided` uses. **Written in this one place only**
#: (principle 2).
_SEED_TERMS = ("traffic_amplification", "sm_idle_cost", "smem_pressure",
               "has_spill", "split_k_cost", "pipeline_warmup_frac")


def _range_report(matrix, reg: FeatureRegistry, splits) -> dict:
    """★ Declared range vs the range actually taken on the training configs
    (D-160). 0 LLM calls.

    ⛔ **Not for a prompt.** The `include_observed` line in
    `features/__init__.py` is what separates condition A from B, and this
    value is on the B side. It exists so that "was the declaration far off"
    can be answered, and the answer is read from the artefact by a human.
    """
    obs = matrix.observed_ranges(splits.train.shapes)
    rows = {}
    for n in sorted(reg._items):
        f = reg[n]
        lo_d, hi_d = (float(x) for x in f.expected_range)
        lo_o, hi_o = obs[n]
        span_d, span_o = hi_d - lo_d, hi_o - lo_o
        rows[n] = {
            "declared": [lo_d, hi_d], "observed": [lo_o, hi_o],
            "shape_level": bool(f.shape_level), "unit": f.unit,
            "direction": f.direction,
            # How many times wider the declaration is than the truth. `null`
            # when the observed span is 0 (a constant axis) — a ratio would
            # be a division by zero and **is not filled in by guessing**.
            "span_ratio": (span_d / span_o) if span_o > 0 else None,
            "outside": bool(lo_o < lo_d or hi_o > hi_d),
            "constant": span_o == 0.0}
    n_out = sum(1 for r in rows.values() if r["outside"])
    ratios = [r["span_ratio"] for r in rows.values() if r["span_ratio"]]
    return {"n_train_shapes": len(splits.train.shapes),
            "split_kind": splits.kind,
            "note": ("★ measured on the TRAINING shapes only, and it never "
                     "goes into a prompt (D-160 §2-3)"),
            "n_features": len(rows),
            "n_outside_the_declaration": n_out,
            "n_constant": sum(1 for r in rows.values() if r["constant"]),
            "span_ratio_median": (sorted(ratios)[len(ratios) // 2]
                                  if ratios else None),
            "features": rows}


def _physics_coverage(table, gen: FeatureRegistry,
                      base: FeatureRegistry) -> dict:
    """★ Does the F1 library cover the hand seed's physics — **a result to
    be measured.**

    It is not a problem to fix. `has_spill` alone moved 1.1637 to 3.1841
    (§8.2), so the absence of a counterpart is itself the result. It is
    computed with 0 LLM calls.

    For each seed term it finds the generated feature with the highest
    Spearman and Pearson. The criterion is the same as §8.4 — both must
    exceed 0.95 to count as "covered".

    ## ★ 2026-09-12 (D-169): why both, and what `covered` does not mean

    The criterion was checked against `c21-lib`, where three terms have
    **Spearman 1.000 and still read as not covered**:

    ```
    has_spill     <- spill_traffic_ratio            sp 1.000  pe 0.545
    split_k_cost  <- partial_sum_combine_work_ratio sp 1.000  pe 0.925
    sm_idle_cost  <- cta_residency_waves            sp 0.949  pe 0.319
    ```

    **The criterion is right and was not changed.** A rule is a *weighted
    sum then sorted*, and a fitted weight can only rescale an axis
    linearly. An axis that orders configs identically but non-linearly
    ranks the same **alone** and differently **inside a sum with other
    terms**, so it is not a drop-in replacement for the seed term. Rank
    agreement alone would say "covered" about an axis the rule cannot
    actually substitute.

    ⚠️ But two things follow that the single number `_n_covered` hides.

    ```
    ★ `monotone_only` is already computed and is a real third state —
      "the library orders configs the same way, but not linearly".
      `_n_monotone_only` is reported next to `_n_covered` from now on
    ★ `has_spill` takes **two values** (measured: binary, [0, 1]).
      Pearson against a continuous axis is then capped by the
      point-biserial coefficient, so that term is near-uncoverable
      whatever the library is — `_n_covered` has a structural ceiling
      below `_n_terms`
    ```

    ⛔ `_n_covered` is still the strict count. It is not widened to include
    `monotone_only` — that would change what the number has meant in every
    earlier artefact.
    """
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features.generated import _reference_columns
    from kernelrule.features.validate import _pearson, _spearman

    made = sorted(set(gen._items) - set(base._items))
    if not made:
        return {"note": "there are no generated features"}

    from kernelrule.features.generated import DUP_SHAPES, _spread_shapes

    # ★ 2026-09-13 (D-170) — **build the columns only on the shapes that are
    #   read.**
    #
    #   `FeatureMatrix(table, reg)` is not lazy: it computes every column of
    #   every shape at construction. `_reference_columns` then reads
    #   `_spread_shapes(table, DUP_SHAPES)` — 21 of the 66 — so two thirds of
    #   the work was thrown away, and this function built such a matrix
    #   **once per generated axis** (20 registries of one feature). D-161
    #   fixed exactly this shape of waste inside `_reference_columns`
    #   ("what must be built is built for those n_shapes alone") and did not
    #   look at its callers (principle 23).
    #
    #   ⚠️ One matrix for 20 axes and 20 matrices for one axis cost the
    #   **same** — the saving is `shapes=`, not the merge. Measured on the
    #   13-library comparison table: 22 minutes and still running, against
    #   `n_covered` values that are unchanged.
    look_at = _spread_shapes(table, DUP_SHAPES)
    human = FeatureRegistry("seed-terms")
    for n in _SEED_TERMS:
        if n in REGISTRY._items:
            human.add(REGISTRY[n])
    ref = _reference_columns(table, FeatureMatrix(table, human,
                                                  shapes=look_at),
                             FeatureRegistry("empty"))
    all_made = FeatureRegistry("generated")
    for n in made:
        all_made.add(gen[n])
    mine = _reference_columns(table, FeatureMatrix(table, all_made,
                                                   shapes=look_at),
                              FeatureRegistry("empty"))

    out: dict = {}
    for name, rv in ref.items():
        best = None
        for gname, gv in mine.items():
            if len(gv) != len(rv):
                continue
            sp, pe = abs(_spearman(gv, rv)), abs(_pearson(gv, rv))
            if best is None or sp > best[1]:
                best = (gname, float(sp), float(pe))
        if best is None:
            out[name] = {"covered": False, "note": "not comparable"}
            continue
        out[name] = {"nearest": best[0], "spearman": round(best[1], 3),
                     "pearson": round(best[2], 3),
                     "covered": best[1] > 0.95 and best[2] > 0.95,
                     "monotone_only": best[1] > 0.95 and best[2] <= 0.95}
    out["_n_covered"] = sum(1 for v in out.values()
                            if isinstance(v, dict) and v.get("covered"))
    # ★ D-169 — the third state, beside "covered" and "not covered".
    out["_n_monotone_only"] = sum(
        1 for v in out.values()
        if isinstance(v, dict) and v.get("monotone_only"))
    out["_n_terms"] = len(ref)
    return out


def _features_module(gen: FeatureRegistry, base: FeatureRegistry) -> str:
    """Records the generated features **in a re-registrable form**
    (§11.4)."""
    made = sorted(set(gen._items) - set(base._items))
    head = ['"""The stage-1 FeatureWriter output. This file is a **record**.',
            "",
            ("To use them again, read `proposals.jsonl` through"
             " `features/loader.py::load_generated`"),
            "— that is the reference, and this is for humans to read.",
            '"""', "", "import numpy as np  # noqa: F401", ""]
    body = [f"# {n}\n{gen[n].source or '(no source)'}\n" for n in made]
    return "\n".join(head) + "\n".join(body)


# ---------------------------------------------------------------------------
# Stage 2 — RuleWriter
# ---------------------------------------------------------------------------
def stage2(a, d: Path, table, matrix, reg: FeatureRegistry, splits) -> dict:
    """Builds the seed rule. It picks the best training score — **it does
    not look at the holdout**."""
    from kernelrule.agents.schemas import validate_rule_proposal
    from kernelrule.core.splits import is_unsealed
    from kernelrule.rules.checks import check_rule, limits_for

    out = d / "stage2-rule-writer"
    (out / "candidates").mkdir(parents=True, exist_ok=True)

    if a.condition == "F3" and a.seed_source == "human_guided":
        # ★ F3 uses the hand seed by definition. It still takes **the same
        #   path**.
        from kernelrule.rules.human_guided import CODE, W0
        chosen = {"source": "human_guided", "code": CODE, "w0": list(W0),
                  "fit_regret": None,
                  "why": "for F3 the hand seed is the condition (every run "
                         "so far)"}
        _dump_json(out / "chosen.json", chosen)
        _dump_json(out / "summary.json", {"condition": "F3", "n_tries": 0,
                                          "source": "human_guided"})
        return chosen

    llm = _make_llm(a, registry=reg, budget=Budget(max_calls=a.n_rule_writer * 3),
                    table=table)
    facts = TableFacts.compute(table, splits.train)
    loop = _loop(a, table, matrix, splits, llm, run_id=f"arch-{a.condition}")
    rows: list[dict] = []
    t0 = time.perf_counter()

    def dump() -> None:
        _dump_json(out / "summary.json", {
            "condition": a.condition, "model": a.model, "dry_run": a.dry_run,
            "n_tries": a.n_rule_writer, "n_ok": sum(r["ok"] for r in rows),
            "seconds": round(time.perf_counter() - t0, 1), "tries": rows})
        if hasattr(llm, "dump"):
            llm.dump(out / "llm_calls")

    try:
        for i in range(a.n_rule_writer):
            row = {"i": i, "ok": False}
            for attempt in range(ARCH_RETRIES):
                try:
                    res = llm.complete("rule_writer", "", condition="A",
                                       table_facts=facts, registry=reg)
                    prop = validate_rule_proposal(
                        res, parameters=getattr(a, "parameters", None))
                    check_rule(prop.code, feature_names=matrix.feature_names(),
                               shape_value_names=matrix.shape_value_names(),
                               n_weights=len(prop.w0),
                               limits=limits_for(
                                   getattr(a, "parameters", None))
                               ).raise_if_bad()
                    e = loop.score_only(prop.code, prop.w0)
                    row.update(ok=True, code=prop.code, w0=list(prop.w0),
                               fit_regret=e, attempt=attempt)
                    (out / "candidates" / f"try{i:02d}.py").write_text(prop.code)
                    print(f"  arch #{i:02d}  ✓ train {e:.4f}")
                    break
                except Exception as ex:                     # noqa: BLE001
                    row.update(error=f"{type(ex).__name__}: {ex}"[:200],
                               attempt=attempt)
                    print(f"  arch #{i:02d}  ✗ (attempt {attempt + 1}) "
                          f"{type(ex).__name__}: {str(ex)[:70]}")
            rows.append(row)
    finally:
        dump()

    ok = [r for r in rows if r["ok"]]
    if not ok:
        # ★ It does not silently proceed without a seed (§26.4). Without a
        #   seed, round 1 starts from an empty parent, and then the question
        #   "can it read the report and fix it" does not hold at all.
        raise RuntimeError(
            f"all {a.n_rule_writer} RuleWriter calls x {ARCH_RETRIES} "
            f"retries failed. It does not proceed without a seed — that the "
            f"F1 features are not enough to build a rule is itself a "
            f"**result**, so it stops here. The artefacts are in {out}.")
    best = min(ok, key=lambda r: r["fit_regret"])
    chosen = {"source": f"rule_writer-try{best['i']:02d}", "code": best["code"],
              "w0": best["w0"], "fit_regret": best["fit_regret"],
              "why": f"the best training score among {len(ok)}/"
                     f"{a.n_rule_writer} successes",
              "all_fit_regret": sorted(r["fit_regret"] for r in ok),
              # ★ 4-3 — what was seen at selection time is left **in the
              #   record**. Procedurally it is held, but evidence is needed
              #   later.
              "selected_on": "train_split_regret_only",
              "holdout_seen_at_selection": False,
              "unsealed": is_unsealed(),
              "_note": ("the seed was picked purely on the training-split "
                        "regret from `RoundLoop.score_only()`. That function "
                        "does not return the holdout — if the selection sees "
                        "the holdout, that holdout is not a holdout "
                        "(principle 6, D-40/D-46/D-50).")}
    _dump_json(out / "chosen.json", chosen)
    print(f"\n  seed: {chosen['source']}  train {best['fit_regret']:.4f}")
    return chosen


# ---------------------------------------------------------------------------
# Stage 3 — RoundLoop
# ---------------------------------------------------------------------------
def _loop(a, table, matrix, splits, llm, *, run_id: str) -> RoundLoop:
    return RoundLoop(
        cfg=LoopConfig(run_id=run_id, max_rounds=a.rounds,
                       # ★ The proposal count uses LoopConfig's default (6)
                       #   (D-144)
                       seed=a.seed,
                       max_new_features_per_round=getattr(
                           a, "max_new_features",
                           LoopConfig.max_new_features_per_round),
                       feature_condition=a.condition,
                       use_analyst=not getattr(a, "no_analyst", False),
                       n_workers=getattr(a, "workers",
                                         LoopConfig.n_workers),
                       objective="regret",
                       parameters=getattr(a, "parameters", None),
                       # ★ The fitter is **decided by the parameter count**
                       #   (D-128)
                       **fitter_for(getattr(a, "parameters", None)),
                       hypothesis_pool=tuple(
                           getattr(a, "hypothesis_pool", []) or ())),
        table=table, matrix=matrix, splits=splits, llm=llm)


def stage3(a, d: Path, table, matrix, reg, splits, seed_rule: dict) -> None:
    out = d / "stage3-evolution"
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(max_calls=3000, max_input_tokens=60_000_000,
                    max_output_tokens=8_000_000)
    for s in range(a.n_seeds):
        run_id = f"{d.name}-s{s}"
        llm = _make_llm(a, registry=reg, budget=budget, table=table)
        loop = RoundLoop(
            cfg=LoopConfig(run_id=run_id, max_rounds=a.rounds,
                           # ★ The proposal count is LoopConfig's default
                           #   (6) (D-144)
                           seed=100 + a.seed + s,
                           max_new_features_per_round=a.max_new_features,
                           feature_condition=a.condition,
                           use_analyst=not a.no_analyst,
                           n_workers=a.workers,
                           objective="regret",
                           parameters=a.parameters,
                           # ★ The fitter is **decided by the parameter
                           #   count** (D-128)
                           **fitter_for(a.parameters),
                           hypothesis_pool=tuple(a.hypothesis_pool)),
            table=table, matrix=matrix, splits=splits, llm=llm)
        loop.seed(seed_rule["code"], seed_rule["w0"],
                  changes="the stage-2 seed")
        print(f"\n  --- {run_id} ---", flush=True)
        try:
            loop.run(a.rounds)      # RoundLoop.run dumps in its finally
        except Exception as e:                              # noqa: BLE001
            print(f"  ★ stopped: {type(e).__name__}: {str(e)[:100]}")
        # ★ `llm_calls/` is a directory. Copying only with `glob("*")` +
        #   `is_file()` **loses the LLM call records entirely** — a path by
        #   which something unrepeatable disappears silently (D-33).
        src = OUT / run_id
        if src.exists():
            import shutil
            dst = out / f"s{s}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
class Terminated(KeyboardInterrupt):
    """Turns SIGTERM into an exception — otherwise `finally` **does not
    run**.

    ★ The default SIGTERM handler dies immediately without unwinding the
    stack. Even code written to leave artefacts through `try/finally` never
    reaches that `finally` (D-33). Confirmed in practice: killed with
    `timeout 25`, only the incrementally appended `proposals.jsonl`
    survived and `summary.json` was never written.

    The incremental append is the first line of defence (an LLM call can
    never be revived), and this is the second.
    """


def _install_signal_handlers() -> None:
    def _die(signum, _frame):
        raise Terminated(
            f"terminating on signal {signal.Signals(signum).name}")

    import contextlib

    # ★ Only SIGTERM is caught. **SIGHUP must not be** — it is the normal
    #   signal that arrives on detaching into the background, and turning it
    #   into a termination exception kills a perfectly good run the moment
    #   it starts. Two F1 runs really were lost that way.
    #   (The twin of principle 1 — a safe-looking handler was making a
    #   verdict.)
    with contextlib.suppress(OSError, ValueError):   # platform/thread limits
        signal.signal(signal.SIGTERM, _die)


def main() -> None:
    _install_signal_handlers()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("condition", choices=("F1", "F2", "F3"))
    ap.add_argument("--n-features", type=int, default=20,
                    help="used only for free generation (--no-categorize). "
                         "With areas, the count is derived from the number "
                         "of areas")
    ap.add_argument("--per-category", type=int, default=3,
                    help="proposals per area. Total = number of areas x "
                         "this value")
    ap.add_argument("--categorize", dest="categorize", action="store_true",
                    help="partition the areas and generate per area. "
                         "★ Off by default — the purpose (reducing "
                         "concentration) was already achieved by the new "
                         "prompt alone, and rediscovery actually fell "
                         "(D-63)")
    ap.add_argument("--recategorize", action="store_true",
                    help="★ **draw the areas again** with the LLM (1 "
                         "call). By default the fixed seven of "
                         "`prompts/areas.md` are used — drawing them every "
                         "time makes that run a different condition, and it "
                         "happens silently (§30.18). A run that used it "
                         "stays in the config")
    ap.add_argument("--categorize-only", action="store_true",
                    help="★ partition the areas **once** and stop (for "
                         "diagnosis). It is not used for generation. A cheap "
                         "device that makes the LLM say for itself 'what it "
                         "cannot build' — this is what found the dtype gap "
                         "(D-63)")
    ap.add_argument("--only-category", metavar="NAME",
                    help="generate for this area only (several keywords "
                         "with `|`, searched in both the name and the "
                         "description). Used to strengthen an existing "
                         "library along a particular axis. ★ The addition "
                         "is a **separate condition**, so do not mix it into "
                         "the comparison table")
    ap.set_defaults(categorize=False)
    ap.add_argument("--n-rule-writer", type=int, default=10)
    ap.add_argument("--seed-source", choices=("rule_writer", "human_guided"),
                    default=None,
                    help="where the seed comes from. The default is "
                         "human_guided for F3 and architect otherwise. "
                         "★ Giving F3 an architect makes it a **control "
                         "arm** — the human 24 and the F1 library can be "
                         "compared under the same prompt")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=12,
                    help="the number of rounds. ★ 2026-09-06 (D-140): "
                         "reverted 24 -> 12. The evidence supporting 24 "
                         "(+0.0395, 6/6, p=0.0156) came from the **wrong "
                         "curve**, and re-measured on bests.jsonl it is "
                         "+0.0088, 9/12, p=0.146 (D-139). ★ It is 'there is "
                         "no evidence 24 is better', not 'it converges by "
                         "12' — the cost is 2x, so 12 is used. Default "
                         "history: 12 (~D-129) -> 24 (D-129) -> 12 (D-140). "
                         "patience is 0 (D-132)")
    # ★ The Analyst -> FeatureWriter path (D-75). **0 is the default = off**
    #   — the same condition as the runs so far. Switched on, runs from that
    #   point are a **separate family**.
    # ★ Uses another campaign's seed as it is (D-83). The 6 runs of one
    #   campaign share a single stage-2 seed, so when campaigns are compared
    #   the seed is a confounder — with respect to the seed the effective
    #   sample is 1 versus 1 (D-82, principle 28).
    ap.add_argument("--seed-from", metavar="RUN_DIR",
                    help="use another campaign's "
                         "stage2-rule-writer/chosen.json as this campaign's "
                         "seed. The provenance is recorded in chosen.json — "
                         "it must not be mistaken for something stage 2 "
                         "built")
    # ★ The §16.1 ablation. With it off there is neither a diagnostic report
    #   nor hypotheses (D-89).
    ap.add_argument("--no-analyst", action="store_true",
                    help="switch the Analyst off (the §16.1 ablation). The "
                         "diagnostic report is not even built")
    # ★ The §16.1 control arm C (D-91). The Analyst is not called and
    #   **someone else's hypotheses** go in.
    ap.add_argument("--hypothesis-pool", nargs="+", default=[],
                    metavar="HYPOTHESES_JSONL",
                    help="another run's hypotheses.jsonl. Without the "
                         "Analyst, those hypotheses are borrowed per round "
                         "— runs with the same seed number are excluded "
                         "automatically")
    # ★ Parallel scoring and fitting (D-95). 0 = sequential (the default).
    #   The results must be identical.
    ap.add_argument("--workers", type=int,
                    default=LoopConfig.n_workers, metavar="N",
                    help="score and fit in N processes (0=sequential). ★ The "
                         "default is LoopConfig's (6, D-163). ⚠️ It was 0 "
                         "here while LoopConfig said 6, so the pipeline kept "
                         "running sequential — the default has to be set in "
                         "**both** places (D-164). The result equals the "
                         "sequential one — "
                         "test_parallel_matches_sequential_on_the_cma_path "
                         "pins it")
    ap.add_argument("--max-new-features", type=int,
                    default=LoopConfig.max_new_features_per_round,
                    metavar="N",
                    help="new axes buildable per round (0=no path, D-75). "
                         "★ The default is LoopConfig's (3, D-160) — the "
                         "same for every condition. The §21 feature-matrix "
                         "cache is invalidated on each new axis")
    ap.add_argument("--bundle", default=BUNDLE,
                    help="the measurement table. ★ Changing to another GPU "
                         "is a **different condition** — do not mix it into "
                         "the comparison table (§3.4)")
    ap.add_argument("--env-hash", default=BUNDLE_HASH,
                    help="the table's env_hash prefix. Not a join key but "
                         "an isolation boundary")
    ap.add_argument("--parameters", type=int, default=None,
                    help="the parameter cap (weights + numeric literals). "
                         "⚠️ It caps nothing since D-150 — every run "
                         "so far is under that condition. The checker and "
                         "the prompt see the same value")
    ap.add_argument("--power-hint", action="store_true",
                    help="★ Experiment (b) (D-112). It states that a "
                         "weight may sit **in the exponent slot**. The base "
                         "is a single f.<name> and the exponent weight is "
                         "bounded by EXPONENT_BOUNDS (0~4) — normalisation, "
                         "not a hyperparameter. The guard applies always, "
                         "regardless of the hint")
    ap.add_argument("--product-hint", action="store_true",
                    help="★ Experiment B (D-110). It **states** that two "
                         "features may be multiplied within a term. The "
                         "static checks never blocked it — the condition is "
                         "not 'loosening' but 'saying so'. It is carried on "
                         "all four surfaces (the system and user prompts, "
                         "the output schema, the checker)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    # ★ The split (D-144). With `--fold` it is a stratified 3-fold;
    #   without it, the structural split.
    ap.add_argument("--split-design", choices=("kfold", "nkgroup"),
                    default="kfold",
                    help="★ D-171 §1: `nkgroup` keeps a whole (N,K) group "
                         "on one side — the layer is the unit a deployment "
                         "changes, and a random cut puts the neighbours of "
                         "one sweep on both sides. Needs --fold")
    ap.add_argument("--fold", type=int, default=None,
                    help="which fold of the stratified k-fold (0..k-1). "
                         "Without it, the structural split (11008)")
    ap.add_argument("--folds", type=int, default=3, help="the k of the k-fold")
    ap.add_argument("--split-seed", type=int, default=12345,
                    help="★ the randomness that builds the folds. "
                         "**Separated from the evolution seed**")
    ap.add_argument("--tag", default=None,
                    help="★ the artefact directory name **verbatim** "
                         "(the D-128 tag rule: "
                         "<features><seed>-p<parameters>[-<expressivity>]"
                         "[-<experiment>], e.g. F3rw-p8 / F3rw-p16 / "
                         "F3rw-p8-prod). Without it, it falls back to the "
                         "old name and can overwrite the same directory — "
                         "a new campaign **must** give it")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the plumbing only, with MockLLM. 0 LLM "
                         "calls")
    ap.add_argument("--stage", type=int, choices=(1, 2, 3),
                    help="run only this stage. It reads the previous "
                         "stage's artefacts")
    ap.add_argument("--extend-from", metavar="STAGE1_DIR",
                    help="carry over an existing stage-1 artefact and "
                         "**extend** it. The duplication verdict is made "
                         "against those too and the artefact is the union. "
                         "★ The extension is a separate condition, so do not "
                         "mix it into the comparison table")
    ap.add_argument("--import-featwriter", metavar="RUN_DIR",
                    help="import an `experiments/feature_writer.py` "
                         "artefact as stage 1. The format is the same, so a "
                         "copy suffices — it saves 20 calls. After importing, "
                         "start with --stage 2")
    a = ap.parse_args()
    # ★ The fitter is decided by the parameter count (D-128). If `cma` is
    #   needed and missing, it stops **now** (principle 1) — seeing it at the
    #   first fit after 500 calls is too late.
    _fit = fitter_for(a.parameters)
    if _fit["fit_method"] == "cma":
        try:
            import cma  # noqa: F401
        except ImportError:
            raise SystemExit(
                f"{a.parameters} parameters are fitted with CMA-ES, but "
                f"the `cma` package is missing. `pip install cma` (the "
                f"`fit` group in pyproject)."
            ) from None
    print(f"  fitter: {_fit['fit_method']} / restarts "
          f"{_fit['fit_restarts']} / "
          f"evals {_fit['max_evals']}  <- decided by "
          f"{a.parameters if a.parameters is not None else 8} parameters")
    if a.seed_source is None:
        a.seed_source = "human_guided" if a.condition == "F3" else "rule_writer"

    # ★ `--tag` is **the directory name verbatim** (the D-128 tag rule).
    #   It used to be assembled as `f1pipe-<condition>-<tag>`, which could
    #   not follow the tag rule. Only without `--tag` does it fall back to
    #   the old name.
    tag = a.tag
    d = OUT / tag if tag else OUT / (
        f"f1pipe-{a.condition}-" + ("mock" if a.dry_run else a.model))
    # ★ Overwriting is blocked. An LLM call cannot be remade (D-33).
    if d.exists() and any(d.iterdir()) and a.stage in (None, 1):
        raise SystemExit(
            f"{d} already exists and is not empty. Overwriting loses the "
            f"previous run's LLM calls (D-33). Give another name with "
            f"`--tag`, or delete it and run again.")
    d.mkdir(parents=True, exist_ok=True)

    table = PerfTable.from_bundle(a.bundle, env_hash=a.env_hash,
                                  ok_only=False)
    splits = _splits(table, fold=getattr(a, "fold", None),
                     split_seed=getattr(a, "split_seed", 12345),
                     k=getattr(a, "folds", 3),
                     design=getattr(a, "split_design", "kfold"))
    base = _base_registry(a.condition)

    print("=" * 78)
    print(f"the F0~F3 pipeline — condition {a.condition}  [{tag}]"
          + ("  ★ DRY RUN (0 LLM calls)" if a.dry_run else ""))
    print("=" * 78)
    print(f"  starting registry {base.name!r}: {len(base._items)}")
    print(f"  split {splits.kind}  train {len(splits.train.shapes)} / "
          f"holdout {len(splits.val.shapes)}")
    print(f"  artefacts {d}\n")

    if a.import_featwriter:
        src = Path(a.import_featwriter) / "proposals.jsonl"
        if not src.exists():
            raise SystemExit(f"{src} does not exist.")
        dst = d / "stage1-features"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "proposals.jsonl").write_text(src.read_text())
        _dump_json(dst / "summary.json", {
            "condition": a.condition, "imported_from": str(src),
            "note": ("the `feature_writer.py` artefact was imported as is. "
                     "The format is the same — the keys `load_generated` "
                     "reads are identical. physics_coverage was not computed "
                     "(stage 1 was not run)")})
        n = sum(1 for ln in src.open()
                if ln.strip() and json.loads(ln).get("accepted"))
        print(f"  ★ imported {n} accepted features from {src} as stage 1. "
              f"Start with --stage 2.\n")
        return

    stages = (a.stage,) if a.stage else (1, 2, 3)

    # ★ F3 is **the human 24 as they are, by definition** (every run so
    #   far). Running FeatureWriter here would make the registry 27 and it
    #   would stop being "the existing condition". It does not skip silently
    #   but **says so** (§26.4).
    if a.condition == "F3" and 1 in stages:
        stages = tuple(x for x in stages if x != 1)
        print("  ★ F3 does not run stage 1 (FeatureWriter) — because the "
              "condition is 'the human 24 as they are'. To add a new axis, "
              "use F2 or state `--stage 1`.\n")
    if a.stage == 1 and a.condition == "F3":
        raise SystemExit(
            "--stage 1 with F3 contradicts the condition. F3's condition is "
            "the human 24 as they are, and adding features to them makes it "
            "F2.")

    # Stage 1
    if 1 in stages:
        # ★ FeatureWriter sees **a matrix built from the base registry**.
        #   Under F1 it is an empty matrix, so no human feature value appears
        #   anywhere.
        m0 = FeatureMatrix(table, base)
        print("--- stage 1 FeatureWriter ---")
        reg = stage1(a, d, table, m0, base,
                     train_shapes=splits.train.shapes)
    elif a.condition == "F3":
        reg = base                  # the human 24 as they are
        print(f"--- no stage 1 (F3) — {len(reg._items)} human features ---")
    else:
        reg = _load_stage1(d, base, a.condition, table)
        n_sh = sum(1 for n in reg._items if reg[n].shape_level)
        print(f"--- stage 1 skipped — {len(reg._items)} stored features "
              f"(shape level {n_sh}, config level "
              f"{len(reg._items) - n_sh}) ---")

    matrix = FeatureMatrix(table, reg)
    # ★ What the axes **actually** take on the training configs (D-160).
    #   The declaration is what the model said; this is what the table says.
    #   ⛔ It is written to the artefact and nowhere near a prompt — a range
    #   read off this table would make the run condition B.
    _dump_json(d / "stage1-features" / "observed-ranges.json",
               _range_report(matrix, reg, splits))
    from kernelrule.core.splits import is_unsealed

    _dump_json(d / "config.json", {
        "condition": a.condition, "model": a.model, "dry_run": a.dry_run,
        "seed_source": a.seed_source,
        # ★ Were the areas drawn again with the LLM (§30.18)? If so, that
        #   run is a **different condition** from one using the fixed list.
        "recategorize": a.recategorize,
        # ★ Did this run go with the final split open (§30.15)
        "unsealed": is_unsealed(),
        "seed": a.seed, "rounds": a.rounds, "n_seeds": a.n_seeds,
        "n_features": a.n_features, "n_rule_writer": a.n_rule_writer,
        "bundle": a.bundle, "env_hash": a.env_hash,
        "split_kind": splits.kind,
        # ★ D-167 §Q — the shape population, not just how it was cut.
        "shape_population": _shape_population(table, splits),
        "registry": {"name": reg.name, "n": len(reg._items),
                     "names": sorted(reg._items)},
        "human_features_present": sorted(set(reg._items) & set(REGISTRY._items))
        if a.condition == "F1"
        else "N/A (F2/F3 include them deliberately)"})

    # Stage 2 — ★ if the later stages will not run, it **is not even read**.
    #   It used to read `chosen.json` even under `--stage 1` and die with
    #   FileNotFoundError. The stage-1 artefacts were fine, but an exit code
    #   of 1 looks like a failure.
    # ★ Importing the seed from another campaign (D-83). It is handled
    #   **before** stage 2 so that `--stage 3` alone holds.
    if a.seed_from:
        src = Path(a.seed_from) / "stage2-rule-writer" / "chosen.json"
        if not src.exists():
            raise SystemExit(
                f"{src} does not exist — there is no seed to import")
        got = json.loads(src.read_text())
        got["copied_from"] = str(src)
        got["source"] = f"{got.get('source', '?')} (copied: {a.seed_from})"
        dst = d / "stage2-rule-writer"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "chosen.json").write_text(
            json.dumps(got, ensure_ascii=False, indent=1))
        print(f"  ★ the seed was imported from {src} — it was not built by "
              f"this campaign's stage 2")

    if 3 in stages or 2 in stages:
        if 2 in stages:
            print("\n--- stage 2 RuleWriter ---")
            chosen = stage2(a, d, table, matrix, reg, splits)
        else:
            chosen = json.loads(
                (d / "stage2-rule-writer" / "chosen.json").read_text())
            print(f"\n--- stage 2 skipped — the stored seed "
                  f"{chosen['source']} ---")

        # Stage 3
        if 3 in stages:
            print("\n--- stage 3 evolution ---")
            stage3(a, d, table, matrix, reg, splits, chosen)

    print(f"\ndone. artefacts {d}")


def _load_stage1(d: Path, base: FeatureRegistry, condition: str, table):
    """Restores the stored stage-1 features. ★ Without them it does not
    silently use the base alone.

    ★ `table` must be passed — so that `shape_level` is **re-judged**
    (§30.12). Without it the recorded value is used (mostly absent = False),
    and then stages 2 and 3 run **with 0 shape-level features**. F1 stage 2
    really was run once in that state — `p.*` branches were 0/10, and that
    was not an observation about the LLM but this defect (D-67).
    """
    from kernelrule.features.loader import extended_registry, load_generated

    path = d / "stage1-features" / "proposals.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. --stage 2/3 need the stage-1 "
            f"artefacts. It does not silently fall back to the base registry "
            f"— that would run with the condition changed (§26.4).")
    made = load_generated(path, exclude=set(base._items), table=table)
    # ★ It also reads what was thrown away by a checker defect and
    #   **revived by re-checking** (`experiments/revalidate.py`). The
    #   original `proposals.jsonl` is not edited — "what was refused at the
    #   time" must not disappear (documentation rule 2).
    revive = path.parent / "revalidated.jsonl"
    if revive.exists():
        extra = load_generated(revive, exclude=set(base._items) |
                               {f.name for f in made}, table=table)
        if extra:
            print(f"  ★ adding {len(extra)} revived by re-checking: "
                  f"{[f.name for f in extra]}")
        made = [*made, *extra]
    return extended_registry(base, made, name=f"{condition}-loaded")


if __name__ == "__main__":
    main()
