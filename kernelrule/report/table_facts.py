"""Block 3.5's table-structure observations — computed **on the training
split only** (§12.3).

## Why a separate module

`build_report(table_facts=[...])` used to take a **list of free strings**.
The report generator itself rejects `train.role != "train"`, but that
argument **bypassed the check entirely** — a caller could drop in a sentence
computed on the full table, and that is exactly what happened.

    The block 3.5 of the first real run
       (`runs/real-gpt-5.4-mini-2026-03-17`) held values computed on all 66
       shapes / on a888's 61 shapes. The validation and final splits went
       into the prompt. See `docs/artifacts/first-real-run.md`.

§12.3 only said "do not put holdout **scores** in", and **aggregates slipped
through.** Score or aggregate, nothing that came out of the holdout may enter
the prompt — if a human looks at it and fixes the system, the end result is
tuning to the holdout (§10.2).

## Structural enforcement

`TableFacts` is built only through `compute()`, and `compute()` accepts only
a training split. `build_report` now rejects a list of strings — **the bypass
path is removed** (§26.4: do not roll on silently in a bad state).

## Minimum supporting shape count

An aggregate resting on few shapes is indistinguishable from memorisation.
"stages matter at K<512" with only 4 such shapes is not physics, it is those
4. So a sentence coming from fewer than `MIN_SUPPORT` shapes is **not
emitted, and if there was one that should have been, that fact is written
down** — it does not drop out silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kernelrule.core.splits import Split, SplitError, regime_of
from kernelrule.core.table import PerfTable

__all__ = ["TableFacts", "MIN_SUPPORT"]

#: An aggregate coming from fewer shapes than this is not presented.
MIN_SUPPORT = 15


@dataclass(frozen=True, slots=True)
class TableFacts:
    """Table-structure observations computed on the training split.

    ★ Built only through `compute()`. There is no path for inserting free
    strings.
    """

    lines: tuple[str, ...]
    n_shapes: int
    #: Feature name -> the observation to attach to that feature. Used by
    #: `features.render_features`.
    #: ★ It came from the training split, so it is left out of the condition-A
    #: prompt.
    by_feature: dict[str, list[str]] = field(default_factory=dict)
    #: Observations that had to be dropped for lack of supporting shapes.
    #: **They do not disappear silently.**
    withheld: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def compute(cls, table: PerfTable, train: Split) -> TableFacts:
        """★ Accepts the training split only. Any other role raises."""
        if not isinstance(train, Split) or train.role != "train":
            raise SplitError(
                "Table-structure observations are computed on the training "
                "split only (§12.3). Aggregates do not cross the holdout "
                "either — blocking only scores lets it leak.")

        shapes = list(train.shapes)
        n = len(shapes)
        lines: list[str] = []
        withheld: list[str] = []

        def emit(text: str, support: int) -> None:
            (lines if support >= MIN_SUPPORT else withheld).append(text)

        lines.append(f"Computed on the training split only, {n} shapes (§12.3).")

        # ★ The "composition of the answer set" that used to be here has
        #   been **deleted** (2026-08-26, §12.3).
        #
        #   It was a line like `shapes whose answer set contains a spilling
        #   kernel: 0/61`. That is **an answer summary that names an axis** —
        #   it reads "you need not look at has_spill" off the table and hands
        #   it over, answering on the LLM's behalf what the LLM should find
        #   itself. It even got attached to each feature's description through
        #   `by_feature`.
        #
        #   For the same reason `design.md`'s "put GBDT feature importances
        #   into block 3.5" was withdrawn too (§30.6 correction).
        #
        #   The criterion (§12.3b):
        #     allowed   "one fixed config reaches top-1 1.115"   size of the room
        #     refused   "0 shapes where a spilling kernel is best"  names an axis
        #     refused   "GBDT considered mainloop_iters important"
        #
        #   **It is worse in F0~F3.** The features the LLM writes all have
        #   different names, so the `feat_of` mapping does not match at all,
        #   and yet the axis names stay in the prompt.
        by_feature: dict[str, list[str]] = {}

        # -- How far one fixed config gets you ------------------------------
        from kernelrule.baselines.static_topk import StaticTopK

        res = StaticTopK(table, shapes, coverage="union").run(ks=(1, 3, 8))
        emit("How far one fixed config gets you (shape-independent):  "
             f"top-1 {res.by_k[1]['all']:.3f}   top-3 {res.by_k[3]['all']:.3f}"
             f"   top-8 {res.by_k[8]['all']:.3f}", n)

        # -- Broken down by regime. ★ Size comes first (§30.5) -------------
        fast = [p for p in shapes if regime_of(p, table.hw) == "short"]
        slow = [p for p in shapes if regime_of(p, table.hw) == "long"]
        for group, label in ((fast, "fast regime (SOL<0.5ms)"),
                             (slow, "slow regime (SOL>=0.5ms)")):
            if not group:
                continue
            r = StaticTopK(table, group, coverage="union").run(ks=(1,))
            emit(f"  {label}, {len(group):2d} shapes only: top-1 "
                 f"{r.by_k[1]['all']:.3f}", len(group))

        # -- Sharpness of the answer — shapes whose ranking vanishes inside
        #    the noise ------------------------------------------------------
        n_flat = sum(1 for p in shapes if int(table.answer_mask(p).sum()) > 1)
        emit(f"Shapes with no single answer (ties inside the noise): "
             f"{n_flat}/{n}", n)
        sizes = np.array([int(table.answer_mask(p).sum()) for p in shapes])
        emit(f"  tie width: median {int(np.median(sizes))}, max "
             f"{int(sizes.max())}", n)

        if withheld:
            lines.append(f"★ {len(withheld)} observations were withheld "
                         f"(fewer than {MIN_SUPPORT} supporting shapes, or a "
                         "missing column). They do not disappear silently "
                         "(§26.4).")

        return cls(lines=tuple(lines), n_shapes=n, by_feature=by_feature,
                   withheld=tuple(withheld))
