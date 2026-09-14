"""Holdout splits (§10).

## ★ The training split and the holdout split are under **different
constraints**

    holdout   must be a block.                     to block interpolation
    training  must hold enough of every regime.    to stop a minority regime
                                                   being sacrificed

**When the two demands conflict, training wins.** A regime training never saw
has no business being evaluated in the first place.

This section originally said only "use a block split, random is
interpolation". It missed **that a block split can push a whole regime onto
one side.** `M > 2048` is exactly that shape — all 11 long shapes go to the
holdout and training becomes 82% short shapes.

Measured (same loop / seed / budget, only the split changed):

| training composition | val gap after 12 rounds | regret over all 61 shapes |
|---|---:|---:|
| short 82% / long 18% | **+2.629** | 1.390 (**worse** than the hand rule's 1.177) |
| short 69% / long 31% | +0.009 | 1.143 |

**Evolution makes whatever trade the regime composition of the training split
allows.** If sacrificing a minority regime helps the training score, it does
that. And if the holdout does not overlap that minority regime, **the blowup
simply is not visible while the rule is still bad on that regime.**

⚠️ **No random splits.** If M=4095 is in training, M=4096 is not a test. Use
a block split.

The reason this file is in stage 1 is the `Split` **type**. For `fit_weights`
to enforce "it only takes a training split", the role has to be nailed into
the type (§29.7). Writing "please pass a training split" in the documentation
is not enforcement (§30.8).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from kernelrule.core.types import Problem

__all__ = ["Split", "SplitSet", "SplitError", "by_predicate",
           "nk_groups", "nk_group_folds", "NK_LAYER",
           "nk_band_folds", "MAIN_BAND", "in_main_band",
           "experiment_shapes", "aligned_shapes", "kernel_families",
           "ALIGNMENT_REQUIRED", "MIN_KERNEL_FAMILIES",
           "KERNEL_FAMILY_COLUMN",
           "stratified_kfold",
           "split_by_M_range", "split_by_K_range", "split_by_alignment",
           "split_by_size", "split_by_waves", "SPLITS",
           "RegimeBalance", "regime_of", "check_balance", "describe",
           "MIN_REGIME_FRAC"]

Role = Literal["train", "val", "test"]


class SplitError(RuntimeError):
    """The split is wrong. **Do not proceed with an empty set** (§26.4)."""


#: ★ The environment variable that unseals the final split (§30.15).
#:
#:   "exactly once at the end" was **an intention, not enforcement.** One
#:   could simply read `splits.test.shapes`. The seal is made out of code —
#:   so it cannot be opened by accident.
#:
#:   A run that opened it is recorded in `config.json` as `unsealed: true`,
#:   and its numbers are marked **possibly contaminated**.
UNSEAL_ENV = "KERNELRULE_UNSEAL"


def is_unsealed() -> bool:
    """Is the final split open? This exists to be recorded in
    `config.json`."""
    import os

    return os.environ.get(UNSEAL_ENV, "") not in ("", "0", "false", "False")


@dataclass(frozen=True, slots=True)
class Split:
    """A subset of shapes + a **role**. The point is that the role lives in
    the type.

        train  used for the diagnostic report and weight fitting. The LLM
               sees it
        val    scored every round. The LLM cannot see it, but a human can
        test   exactly once at the end of the project. ★ **sealed** —
               `KERNELRULE_UNSEAL` required

    ★ The `shapes` of `role="test"` are sealed (§30.15). Reaching them
    requires the environment variable to be set explicitly, and that fact
    stays in `config.json`.
    **`len()` and `role` are unaffected by the seal** — that a split exists
    and looking inside it are different things.
    """

    role: Role
    #: ⚠️ **Do not read this directly.** For `role="test"` it bypasses the
    #: seal check. Use the `shapes` property.
    _shapes: tuple[Problem, ...]
    name: str = ""

    @property
    def shapes(self) -> tuple[Problem, ...]:
        """The shapes. ★ For `role="test"` it goes through the seal check
        (§30.15)."""
        if self.role == "test" and not is_unsealed():
            raise SplitError(
                "the final split is **sealed** (§10.2). It is opened "
                f"exactly once, at the end of the project.\n  To open it, "
                f"set {UNSEAL_ENV}=1 **explicitly**.\n"
                "  That run is recorded in config.json as `unsealed: true`, "
                "and the numbers from it are marked **possibly "
                "contaminated**.\n"
                "  ⚠️ Once seen it cannot be undone — a human now knows it "
                "and makes the next decision knowing it (which is why §10.2 "
                "uses three splits).")
        return self._shapes

    def __post_init__(self) -> None:
        if self.role not in ("train", "val", "test"):
            raise SplitError(f"unknown role: {self.role!r}")
        if not self._shapes:
            raise SplitError(
                f"split {self.name or self.role!r} is empty. An empty set "
                f"is not returned and waved through (§26.4).")

    def __len__(self) -> int:
        # ★ Unaffected by the seal — knowing the size and looking inside
        # are different things.
        return len(self._shapes)

    def __iter__(self):
        return iter(self.shapes)          # goes through the seal check


@dataclass(frozen=True, slots=True)
class SplitSet:
    """The three-way split (§10.2). The middle one is needed because **the
    human is the contaminant**."""

    train: Split
    val: Split
    test: Split | None = None
    kind: str = ""

    def __post_init__(self) -> None:
        a = {p.key for p in self.train}
        b = {p.key for p in self.val}
        overlap = a & b
        if overlap:
            raise SplitError(
                f"train and val share {len(overlap)} shapes: "
                f"{sorted(overlap)[:5]}. That is not a holdout.")
        if self.test is not None:
            c = {p.key for p in self.test}
            if (a & c) or (b & c):
                raise SplitError(
                    "test overlaps train/val. The seal is broken.")


def by_predicate(shapes: Sequence[Problem],
                 held_out: Callable[[Problem], bool], *,
                 name: str = "", val_frac_of_heldout: float = 1.0) -> SplitSet:
    """A block split from a single predicate. The shapes where it is true
    are the holdout.

    In stage 2, `split_by_M_range` / `split_by_K_range` /
    `split_by_alignment` / `split_by_arch` sit on top of this.
    """
    train = tuple(p for p in shapes if not held_out(p))
    out = tuple(p for p in shapes if held_out(p))
    if not train or not out:
        raise SplitError(
            f"split {name!r} left one side empty (train={len(train)}, "
            f"heldout={len(out)}). It raises — it does not proceed "
            f"(§26.4).")
    n_val = max(1, int(round(len(out) * val_frac_of_heldout)))
    return SplitSet(
        train=Split("train", train, name=f"{name}:train"),
        val=Split("val", out[:n_val], name=f"{name}:val"),
        test=(Split("test", out[n_val:], name=f"{name}:test")
              if out[n_val:] else None),
        kind=name)


def stratified_kfold(shapes: Sequence[Problem], hw, *, k: int = 3,
                     seed: int = 0, fold: int | None = None,
                     name: str = "kfold") -> list[SplitSet]:
    """★ A random k-fold that preserves the memory/compute ratio (D-144).

    ## Why stratified

    Splitting purely at random makes the composition wobble per fold.

    ```
    of 61 shapes  memory 20 / compute 41   (t_memory > t_compute)
    drawing 20 at random can produce a fold with 2 memory shapes
    -> ★ that fold's val effectively measures compute alone
    ```

    So each stratum is shuffled separately and cut into k parts, and fold
    `i`'s val is the union of part `i` of every stratum. A ±1 difference in
    val size is accepted.

    ## ★ The split seed and the evolution seed are separated

    `seed` is **the randomness that builds the folds**. Do not give it the
    same value as the loop's `cfg.seed` — if the two are entangled, "because
    the split differed" cannot be told from "because the evolution differed".
    """
    if k < 2:
        raise SplitError(f"folds cannot be built with k={k}")
    import numpy as np

    strata: dict[str, list] = {}
    for p in shapes:
        strata.setdefault(regime_of(p, hw, axis="roofline"), []).append(p)
    rng = np.random.default_rng(seed)
    chunks: dict[str, list[list]] = {}
    for nm, group in sorted(strata.items()):
        g = [group[i] for i in rng.permutation(len(group))]
        # ★ The earlier parts are one larger (7/7/6). The remainder is not
        # thrown away.
        chunks[nm] = [g[i::k] for i in range(k)]
    out: list[SplitSet] = []
    for i in range(k):
        val = tuple(p for nm in sorted(chunks) for p in chunks[nm][i])
        vk = {p.key for p in val}
        train = tuple(p for p in shapes if p.key not in vk)
        if not train or not val:
            raise SplitError(f"fold {i} left one side empty (§26.4)")
        out.append(SplitSet(
            train=Split("train", train, name=f"{name}{i}:train"),
            val=Split("val", val, name=f"{name}{i}:val"),
            kind=f"{name}{i}-seed{seed}"))
    return out if fold is None else [out[fold]]


# ---------------------------------------------------------------------------
# Block splits (§10.1)
# ---------------------------------------------------------------------------
def split_by_M_range(shapes: Sequence[Problem], *, m_threshold: int = 2048
                     ) -> SplitSet:
    """M > 2048 as the holdout. It tests **extrapolation**.

    kernelTab's main GBDT metric is this split (11 holdout shapes, 1.011).
    The 0.8 percentage-point gap against the shape-level 5-fold (1.019) is
    the measure of **how hard shape generalisation is** — the 5-fold puts
    M=1024 in training and M=1000 in validation, which is interpolation in
    practice.
    """
    return by_predicate(shapes, lambda p: m_threshold < p.M,
                        name=f"M>{m_threshold}")


def split_by_K_range(shapes: Sequence[Problem], *, k_threshold: int = 8192
                     ) -> SplitSet:
    """Layer B's K band as the holdout. Extrapolation in mainloop
    depth."""
    return by_predicate(shapes, lambda p: k_threshold < p.K,
                        name=f"K>{k_threshold}")


def split_by_alignment(shapes: Sequence[Problem]) -> SplitSet:
    """All of layer D (alignment < 8) as the holdout.

    An alignment-1 shape cannot use cp.async, so only stages=2 is possible —
    a band where only kernel families the rule has never seen remain.
    """
    def held(p: Problem) -> bool:
        return (p.K % 8 != 0) or (p.N % 8 != 0)
    return by_predicate(shapes, held, name="align<8")


def split_by_size(shapes: Sequence[Problem], hw, *, ms: float = 0.5
                  ) -> SplitSet:
    """Short shapes as the holdout (the §30.5 tension).

    Almost all the room is under 0.5ms, and that band is where the
    measurement resolution is worst. It directly tests **whether what was
    learned on long shapes transfers to short ones**.

    ⚠️ The boundary is taken from the **roofline lower bound**, not from
    `best_ms` (the answer). `best_ms` is `ANSWER_COLS`, so using it in a
    split definition leaks the answer in.
    """
    import math

    from kernelrule.features.physical import log_sol_ms

    thresh = math.log2(ms)

    def held(p: Problem) -> bool:
        return log_sol_ms(p, hw, _DUMMY_CFG) < thresh
    return by_predicate(shapes, held, name=f"sol<{ms}ms")


def split_by_waves(shapes: Sequence[Problem], hw, *, tile: int = 128,
                   waves_threshold: float = 1.0) -> SplitSet:
    """Cuts on waves at a reference tile.

    ★ Layer C derives M **backwards** from `sm_count`, so a split on the
    absolute M cuts different things on different GPUs (§10.1). Transfer
    experiments use this path.
    """
    import math

    def held(p: Problem) -> bool:
        w = (math.ceil(p.M / tile) * math.ceil(p.N / tile)) / hw.sm_count
        return w < waves_threshold
    return by_predicate(shapes, held, name=f"waves<{waves_threshold}")


#: Name -> constructor. The report runs all of them.
SPLITS = {
    "M_range": split_by_M_range,
    "K_range": split_by_K_range,
    "alignment": split_by_alignment,
}

#: The placeholder config `split_by_size` needs in order to call a
#: shape-level feature. A shape-level feature does not look at `cfg` (that is
#: its definition).
_DUMMY_CFG = None


def _make_dummy():
    from kernelrule.core.types import Config
    return Config(tile_m=128, tile_n=128, tile_k=32, align_a=8, align_b=8,
                  align_c=8, split_k=1, split_k_mode="serial", arch="sm_86",
                  kernel_id="_dummy", regs_per_thread=128, threads=256,
                  smem_bytes=32768, spill_bytes=0, max_blocks_per_sm=2,
                  pipeline_kind="multistage",
                  # ★ D-161: `stages` is a field now. 3 is the smallest
                  #   multistage depth — it keeps the dummy internally
                  #   consistent. Nothing reads it here (the roofline needs
                  #   no stage count); a mismatched 0 would just be a lie
                  #   waiting to be read.
                  stages=3)


_DUMMY_CFG = _make_dummy()


# ---------------------------------------------------------------------------
# ★ Regime balance (§10.1) — it is not waved through silently (§26.4)
# ---------------------------------------------------------------------------
#: Warns when the training split holds some regime below this fraction.
#:
#: Measured (training fixed at 24, 3 seeds, fixed test bench = 12 long
#: shapes):
#:
#:     long frac   median   worst
#:      8%          6.57    16.33
#:     17%          1.17    10.41
#:     25%          1.20     2.19
#:     33%          1.22     1.79
#:
#: **Balance shrinks the tail; it cannot lift the median.** 25% is a lower
#: bound, not a safety line — even at 33% the worst is 1.79. Run any
#: experiment that changes the split with at least 3 seeds and **report the
#: worst value alongside.**
MIN_REGIME_FRAC = 0.25


@dataclass(frozen=True, slots=True)
class RegimeBalance:
    """The regime composition of a split. **Always printed.**"""

    axis: str
    counts: dict
    n: int

    @property
    def fractions(self) -> dict:
        return {k: v / self.n for k, v in self.counts.items()} if self.n else {}

    def minority(self) -> tuple[str, float]:
        f = self.fractions
        if not f:
            return ("", 0.0)
        k = min(f, key=lambda x: f[x])
        return (k, f[k])

    @property
    def ok(self) -> bool:
        return self.minority()[1] >= MIN_REGIME_FRAC

    def __str__(self) -> str:
        parts = " / ".join(f"{k} {v}({v / self.n:.0%})"
                           for k, v in sorted(self.counts.items()))
        mark = "" if self.ok else "   ⚠️ minority regime too small"
        return f"[{self.axis}] n={self.n}  {parts}{mark}"


def regime_of(p: Problem, hw, *, axis: str = "size") -> str:
    """The shape's regime. **It does not use the answer** — it cuts on the
    roofline lower bound.

    `best_ms` is `ANSWER_COLS`, so putting it into a split definition
    contaminates the holdout.
    """
    import math

    from kernelrule.features.physical import is_memory_bound, log_sol_ms

    if axis == "size":
        return ("short" if log_sol_ms(p, hw, _DUMMY_CFG) < math.log2(0.5)
                else "long")
    if axis == "roofline":
        return ("mem" if is_memory_bound(p, hw, _DUMMY_CFG) else "comp")
    raise ValueError(f"unknown regime axis: {axis!r}")


def check_balance(split: Split, hw, *, axis: str = "size",
                  strict: bool = False) -> RegimeBalance:
    """Does the training split hold enough of every regime?

    ⚠️ If not it **warns** (an error with `strict=True`). It is not waved
    through silently — a rule that sacrificed a minority regime looks like an
    improvement by the training score.
    """
    import warnings
    from collections import Counter

    c = Counter(regime_of(p, hw, axis=axis) for p in split.shapes)
    bal = RegimeBalance(axis=axis, counts=dict(c), n=len(split.shapes))
    if split.role == "train" and not bal.ok:
        k, f = bal.minority()
        msg = (f"regime {k!r} of the training split "
               f"{split.name or split.role!r} is only {f:.0%} "
               f"(threshold {MIN_REGIME_FRAC:.0%}).\n"
               f"  {bal}\n"
               "  Evolution can sacrifice a minority regime and still look "
               "like an improvement by the training score. In measurement, "
               "the 18% composition worsened overall regret from 1.177 to "
               "1.390 while the training score improved from 1.201 to 1.118 "
               "(§10.1).")
        if strict:
            raise SplitError(msg)
        warnings.warn(msg, stacklevel=2)
    return bal


def describe(ss: SplitSet, hw, *, axis: str = "size") -> str:
    """Renders a split's regime composition for a human. **Always
    printed.**"""
    lines = [f"split {ss.kind or '(unnamed)'}"]
    for sp in (ss.train, ss.val, ss.test):
        if sp is None:
            continue
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bal = check_balance(sp, hw, axis=axis)
        lines.append(f"  {sp.role:5s} {bal}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ★ 2026-09-11 (D-167 §R) — the shape population the experiments run on
# ★ 2026-09-13 (D-170 §1) — **the criterion changed**: alignment -> kernel
#   family. 61 -> 65 on the A6000 table.
# ---------------------------------------------------------------------------
#: ★ The **old** criterion (D-167 §R, superseded 2026-09-13). Every candidate
#: of a shape had to carry this alignment on all three operands.
#:
#: ⛔ It is kept, not deleted: every number recorded before 2026-09-13 — the
#: 21-run campaign included — has the 61-shape population as its denominator,
#: and `aligned_shapes()` below is what reproduces it. Nothing in the
#: experiment path calls it any more.
ALIGNMENT_REQUIRED = 8

#: ★ The **current** criterion (D-170 §1). A shape is used unless its
#: candidate space holds fewer than this many kernel families.
MIN_KERNEL_FAMILIES = 2

#: What counts as a kernel family. `pipeline_kind` is `multistage` /
#: `pipelined` — the two mainloop structures, which differ in whether
#: `cp.async` is used at all, not in a parameter.
KERNEL_FAMILY_COLUMN = "pipeline_kind"


def aligned_shapes(table) -> list:
    """★ The **old** shape population — every candidate aligned to 8 on A, B
    and C (D-167 §R). 61 of the A6000 table's 66.

    ⛔ Superseded by `experiment_shapes` on 2026-09-13 (D-170 §1). It stays
    so that a number recorded under the old condition can still be
    reproduced — the 21-run campaign is scored on exactly this set. **Do not
    call it from a new experiment.**
    """
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == ALIGNMENT_REQUIRED).all()
                    and (d.align_b == ALIGNMENT_REQUIRED).all()
                    and (d.align_c == ALIGNMENT_REQUIRED).all())

    return [p for p in table.shapes() if aligned(p)]


def kernel_families(table, p) -> list[str]:
    """The distinct kernel families in one shape's candidate space."""
    return sorted(set(table.frame_for(p)[KERNEL_FAMILY_COLUMN].tolist()))


#: ★ The population per table, memoised (D-170 §1).
#:
#: `table.frame_for(p)` materialises a ~17,000 x 68 slice, so walking every
#: shape costs **0.69s**. Since D-170 §2 the validation asks for the
#: population on **every candidate feature**, twice (the retry check and the
#: registration), and `_spread_shapes` asks again — measured at about +2.8s
#: per proposal, which is most of the cost the population change appeared to
#: add.
#:
#: ⚠️ Keyed on the table **object**. A `PerfTable` is loaded once and never
#: mutated (`from_bundle` builds it whole), so identity is the right key; a
#: second `from_bundle` of the same bundle is a different object and is
#: computed again, which is correct rather than merely safe.
_POPULATION: dict[int, tuple[object, list]] = {}


def experiment_shapes(table) -> list:
    """The shapes the experiments run on: **every shape whose candidate
    space holds more than one kernel family**.

    ★ Why this function exists: the predicate was copied into **36 places
    across 35 files** (every `experiments/*.py` that builds a split). One
    judgement in 36 copies is the largest instance of principle 2 in this
    repository, and the vendor-preset case (D-158 fixed one of two copies)
    showed how that ends. Unifying them (D-167 §R) changed no value; **this
    change of criterion does** — see below.

    ## ★ 2026-09-13 (D-170 §1) — from alignment to kernel family

    The old criterion was `align_a == align_b == align_c == 8` on every
    candidate. Its grounds were **never recorded anywhere** — not in
    `design.md`, not in `decisions.md`; it was in the code from the first
    commit and no decision entry introduces it (that absence is itself
    written down here rather than guessed, principle 39).

    What D-167 §R measured is what made it untenable:

    ```
    shape                 candidates  ext_stages  kernel families      align
    (1024,4096,4097)           4,800     2..2      pipelined  ★ one   (1,1,8)
    (1024,4096,4098)          17,250     2..8      multistage+pipelined (2,2,8)
    (1024,4096,4100)          17,250     2..8      multistage+pipelined (4,4,8)
    (1024,4100,4096)          16,315     2..8      multistage+pipelined (8,8,4)
    (1024,4098,4096)          16,315     2..8      multistage+pipelined (8,8,2)
    the 61 old-included    3,465~17,325  2..8      multistage+pipelined (8,8,8)
    ```

    **4 of the 5 excluded shapes have the same candidate-space shape as the
    included ones** — same stage range, same two families, a comparable
    candidate count. Only `(1024,4096,4097)` is structurally different: one
    family, one stage count, a quarter of the candidates. Alignment 8 was
    standing in for "the kernel space is structurally different" and missing
    on 4 of 5, so the criterion now says that directly.

    Two things followed from the old criterion that were not intended:

    ```
    scoring        excluded the 5   (61)
    feature check  included them    (66)
    -> ★ a shape that is never scored was deciding what went into the library
    ```

    and the vendor heuristic — which knows nothing of our splits — is **not
    worse** on the excluded shapes than on the included ones:

    ```
    vendor geomean   61 included 1.0797   /   5 excluded 1.1238
      (1024,4096,4100) 1.2031   (1024,4096,4097) 1.1875  (1024,4096,4098) 1.1565
      (1024,4100,4096) 1.0432 ★ better than the 61-shape mean
      (1024,4098,4096) 1.0400 ★ better
    ★ the worst included shape is (1024,4096,256) at 1.2703 — worse than any
      excluded one. The excluded five are not a "special zone"; the
      difficulty ranges overlap.
    ```

    ⚠️ K alignment and N alignment are not the same thing physically. K
    misalignment touches what the mainloop reads on every iteration; N
    misalignment touches only the epilogue (writing C out). The two N-side
    shapes are the two the vendor does best on.

    ## What this costs

    The population is the denominator of every number in the repository.
    ⛔ Numbers recorded before 2026-09-13 are **not** recomputed — they are
    marked as the old condition and left standing (D-170 §9). `aligned_shapes`
    reproduces that population exactly.

    ## Measured on all four tables (2026-09-13, 0 LLM calls)

    ```
    table   table shapes   old (align 8)   ★ new (family)   dropped
    a6000        66             61              65          (1024,4096,4097)
    5090         66             61              65          (1024,4096,4097)
    4090         64             59              63          (1024,4096,4097)
    h100         64             59              63          (1024,4096,4097)
    ```

    ★ The same one shape in all four tables, and in all four it is both the
    only one-family shape and the only one-stage-count shape. The judgement
    is made per table (`table.frame_for`), not by copying the A6000 list.
    """
    hit = _POPULATION.get(id(table))
    # The table is held alongside the answer so that a recycled `id()`
    # cannot hand back another table's population.
    if hit is not None and hit[0] is table:
        return list(hit[1])
    out = [p for p in table.shapes()
           if len(kernel_families(table, p)) >= MIN_KERNEL_FAMILIES]
    _POPULATION[id(table)] = (table, out)
    return list(out)


# ---------------------------------------------------------------------------
# ★ 2026-09-13 (D-171) — the (N, K) group split
# ---------------------------------------------------------------------------
#: ★ The layer dimension that decides fold 0 (D-171 §1-2). It is the same
#: 11008 the `nk11008` structural split has always used — the FFN width of
#: the model this grid was built around. ⛔ Not a knob: changing it changes
#: which shapes are the layer holdout, and with it every transfer number.
NK_LAYER = 11008


def nk_groups(shapes: Sequence[Problem]) -> dict[tuple[int, int], list]:
    """Shapes grouped by `(N, K)`, in the order they first appear.

    ★ It looks at the **shape definition only**. `best_ms` is `ANSWER_COLS`,
    so a split built from the answer contaminates the holdout (§10.1).
    """
    out: dict[tuple[int, int], list] = {}
    for p in shapes:
        out.setdefault((p.N, p.K), []).append(p)
    return out


def nk_group_folds(shapes: Sequence[Problem], *, k: int = 4,
                   name: str = "nkgroup") -> list[SplitSet]:
    """★ k folds in which **a whole `(N, K)` group is on one side**
    (D-171 §1).

    ## Why not a random k-fold

    This table is **not a sample, it is a designed grid.** Every shape is a
    probe placed on purpose:

    ```
    a_workload  40 shapes   (N,K) 4 kinds x M 10 kinds
    b_kvary     12 shapes   M in {128,1024}, a K sweep
    c_waves     11 shapes   N=K=4096, ★ a fine M sweep (aimed at wave
                            quantisation)
    d_alignment  4 shapes   misaligned
    e_square     5 shapes   M=N=K
    ```

    Cut at random and the neighbours of one sweep land on both sides —
    M=1500 in training and M=1536 in validation. The layer `(N, K)` is the
    unit a deployment actually changes, so it is the unit the split uses.

    ## The layout on the A6000's 65 shapes

    ```
    ★ fold 0   (4096,11008) + (11008,4096)      val 20
      fold 1   (4096,4096)  17 shapes           val 17
      fold 2~3 the remaining 15 groups          val 14 · 14
    ```

    ★ Fold 0 is **exactly the old `nk11008` holdout** — those 20 shapes are
    two groups, so the structural split we have been using all along is one
    fold of this design rather than a separate experiment.

    ⚠️ `(4096,4096)` holds 17 shapes and **cannot be split**, so raising `k`
    does not improve the balance: at k=5 a 9-shape fold appears. k=4 is the
    largest k whose smallest fold still holds 14.

    ## How the rest is assigned

    Groups are sorted by size (descending, ties by `(N, K)`) and each goes to
    the fold holding the fewest shapes so far, ties to the lowest index —
    after folds 0 and 1 are claimed by the two fixed groups above. It is
    deterministic and takes no randomness at all, so there is **no split
    seed** to separate from the evolution seed.

    ⚠️ What this measures is **"a layer shape never seen"**, not "a shape
    never seen". A validation shape has no `(N, K)` sibling in training by
    construction, but its **M siblings remain** — M=1024 appears in almost
    every group. That limit is unavoidable on this grid and is stated rather
    than papered over.
    """
    if k < 2:
        raise SplitError(f"folds cannot be built with k={k}")
    groups = nk_groups(shapes)
    if len(groups) < k:
        raise SplitError(
            f"there are only {len(groups)} (N,K) groups and {k} folds were "
            f"asked for. A fold would be empty (§26.4)")
    assigned: list[list] = [[] for _ in range(k)]
    # ★ Fold 0 is **the 11008 layer, both orientations together** — the two
    #   groups `(4096,11008)` and `(11008,4096)`. That is not a greedy
    #   choice: those 20 shapes are exactly the `nk11008` holdout this
    #   repository has been using since §10.1, so pinning them as fold 0
    #   makes the structural split **one fold of this design** instead of a
    #   separate experiment (D-171 §1-2).
    fixed0 = [g for g in groups if NK_LAYER in g]
    if not fixed0:
        raise SplitError(
            f"no (N,K) group carries {NK_LAYER}, so fold 0 cannot be the "
            f"layer holdout. Groups: {sorted(groups)}")
    for g in fixed0:
        assigned[0].extend(groups[g])
    rest_groups = {g: v for g, v in groups.items() if g not in fixed0}
    order = sorted(rest_groups, key=lambda g: (-len(rest_groups[g]), g))
    # ★ Fold 1 takes the largest remaining group whole — `(4096,4096)` with
    #   17 shapes on this table. It cannot be split (that is what caps `k`),
    #   so it decides a fold rather than being spread over several.
    if k >= 2 and order:
        assigned[1].extend(rest_groups[order[0]])
        order = order[1:]
    groups = rest_groups
    for g in order:
        i = min(range(k), key=lambda j: (len(assigned[j]), j))
        assigned[i].extend(groups[g])
    out: list[SplitSet] = []
    for i in range(k):
        val = tuple(p for p in shapes if p in assigned[i])
        vk = {p.key for p in val}
        train = tuple(p for p in shapes if p.key not in vk)
        if not train or not val:
            raise SplitError(f"fold {i} left one side empty (§26.4)")
        out.append(SplitSet(
            train=Split("train", train, name=f"{name}{i}:train"),
            val=Split("val", val, name=f"{name}{i}:val"),
            kind=f"{name}{i}-k{k}"))
    return out


# ---------------------------------------------------------------------------
# ★ 2026-09-14 (D-174 §1) — the (N,K) band split, **without pinning the
#   layer to one fold**
# ---------------------------------------------------------------------------
#: The N and K values the grid sweeps as a "main band" — the layer widths a
#: transformer block actually uses. A group is in the band when **both** N
#: and K are one of these.
#:
#: ⛔ It is not a tuning knob. It names which groups are the large ones so
#: they can be spread one per fold; the values come from the grid's design,
#: not from any score.
MAIN_BAND = (4096, 11008, 12288)


def in_main_band(nk: tuple[int, int]) -> bool:
    return nk[0] in MAIN_BAND and nk[1] in MAIN_BAND


def nk_band_folds(shapes: Sequence[Problem], *, k: int = 4,
                  name: str = "nkband") -> list[SplitSet]:
    """★ k folds over whole `(N, K)` groups, **balanced by size and by
    band** (D-174 §1).

    ## Why this replaced pinning the layer to fold 0

    `nk_group_folds` put both 11008 groups in fold 0 so that fold 0 was the
    old `nk11008` holdout. The grid has **four large groups** (17 · 10 · 10
    · 10 shapes on the A6000), and spending two of them on one fold leaves
    two to cover three folds — the last fold gets the leftovers:

    ```
    방식                   형상 수              대역 밖         최대 비율
    k=3 · nk11008 고정   [24, 23, 18]        [4, 6, 8]         44%
    k=4 · nk11008 고정   [22, 20, 15,  8]    [2, 3, 5, 8]      44%
    ★ k=4 · 고정 없음    [17, 16, 16, 16]  ★ [0, 6, 6, 6]   ★ 28%
    ```

    ★ Lowering `k` does not fix it. Releasing the pin does.

    That leftover fold is where the campaign's rules collapsed, and the
    baselines say it was not the shapes:

    ```
              n    vendor   static top-1   native
    fold3    12   1.1055     1.0475      ★ 1.4717
    fold0-2  36   1.1060     1.0644        1.0583
    ```

    ## The rule — dimensions only

    ```
    ★ a whole (N,K) group stays on one side      ⛔ never split
    ★ the four largest groups go ★ one per fold
    ★ the rest, largest first, to the fold with the fewest shapes and then
      the fewest out-of-band groups
    ⛔ the layer is not pinned anywhere
    ```

    ⚠️ It reads `N` and `K` and nothing else — no score, no `best_ms`.

    ⚠️ **The old `nk11008` holdout is no longer one fold.** Under
    cross-validation every shape is held out exactly once, so those 20
    shapes still get a value; it is a **different procedure** from the old
    "train 45 -> holdout 20 once" and must be reported as such (§1-5).
    """
    if k < 2:
        raise SplitError(f"folds cannot be built with k={k}")
    groups = nk_groups(shapes)
    if len(groups) < k:
        raise SplitError(
            f"there are only {len(groups)} (N,K) groups and {k} folds were "
            f"asked for (§26.4)")
    order = sorted(groups, key=lambda g: (-len(groups[g]), g))
    assigned: list[list] = [[] for _ in range(k)]
    out_of_band: list[int] = [0] * k
    # ★ The k largest groups take one fold each.
    for i, g in enumerate(order[:k]):
        assigned[i].extend(groups[g])
        if not in_main_band(g):
            out_of_band[i] += len(groups[g])
    for g in order[k:]:
        i = min(range(k),
                key=lambda j: (len(assigned[j]), out_of_band[j], j))
        assigned[i].extend(groups[g])
        if not in_main_band(g):
            out_of_band[i] += len(groups[g])
    out: list[SplitSet] = []
    for i in range(k):
        keep = {p.key for p in assigned[i]}
        val = tuple(p for p in shapes if p.key in keep)
        train = tuple(p for p in shapes if p.key not in keep)
        if not train or not val:
            raise SplitError(f"fold {i} left one side empty (§26.4)")
        out.append(SplitSet(
            train=Split("train", train, name=f"{name}{i}:train"),
            val=Split("val", val, name=f"{name}{i}:val"),
            kind=f"{name}{i}-k{k}"))
    return out
