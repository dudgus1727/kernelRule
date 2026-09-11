"""Core types — **there is nowhere for a measured time to enter** (§3.3, §6.1).

This file exists as a **defence**, not a convenience. It is the first of four
layers keeping the rule function from seeing the answer, and even if the other
three (loader / static check / behavioural check) are all breached, it holds
here — **because the objects a rule can touch have no times on them.**

    Problem       shape.       no times
    Hardware      hardware.    no times. peak/bandwidth are **effective** (§6.2)
    Config        candidate.   no times
    CandidateSet  all candidates of one shape, column-wise.
                  **no times + tie-break included**

`CandidateSet` is where the §30.7 answer leak is blocked structurally. The
only thing the scorer may use to order is the `tiebreak` integer array, and
that array is built from config identity alone (kernel_id, split_k,
split_k_mode). Putting time into the tie-break would require a field this
class does not have, so it raises `AttributeError`.

This is a hot path, so frozen dataclasses are used. Pydantic lives only at the
LLM boundary (§11.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "CandidateSet",
    "Config",
    "Hardware",
    "Problem",
    "ShapeKey",
    "config_key",
    "hardware_from_env",
    "shape_key",
]

#: Shape join key. (M, N, K, dtype). The only path linking table and candidates.
ShapeKey = tuple[int, int, int, str]

#: Config join key. (kernel_id, split_k, split_k_mode).
#: **The tie-break uses only this** (§30.7).
ConfigKey = tuple[str, int, str]



#: dtype name -> bytes per element. **An unknown name raises** — falling back
#: to a default silently makes the whole roofline wrong (§26.4).
_DTYPE_BYTES: dict[str, float] = {
    "f16": 2.0, "bf16": 2.0, "f32": 4.0, "tf32": 4.0, "f64": 8.0,
    "i8": 1.0, "u8": 1.0, "f8": 1.0, "i32": 4.0,
}

@dataclass(frozen=True, slots=True)
class Problem:
    """GEMM shape. D[MxN] = A[MxK] @ B[KxN].

    ⚠️ Do not add `time_ms` / `difficulty` / `n_distinct_times` here. All
    three are `ANSWER_COLS` — derived from the answer and unknown at
    deployment time.
    """

    M: int
    N: int
    K: int
    dtype: str = "f16"
    acc_dtype: str = "f32"
    layout_a: str = "row"
    layout_b: str = "col"
    layout_c: str = "row"

    @property
    def key(self) -> ShapeKey:
        return (self.M, self.N, self.K, self.dtype)

    @property
    def bytes_per_element(self) -> float:
        """Bytes of one A/B/C element. **Derived from `dtype` — it is not
        new information** (§30.11).

        Why it is exposed: the roofline is `FLOP / byte`, and getting bytes
        requires turning the dtype into bytes. But `p.dtype` is a string and
        the feature sandbox has no `np.dtype(...).itemsize`. Because of that,
        in F1 the LLM **failed three times in a row** on the
        arithmetic/bandwidth-pressure area (D-63).

        **Listing a field you cannot use is telling the model it exists.**
        """
        return _DTYPE_BYTES[self.dtype]

    @property
    def acc_bytes_per_element(self) -> float:
        """Bytes of one accumulator element. This is the size of a parallel
        split-K partial sum."""
        return _DTYPE_BYTES[self.acc_dtype]


@dataclass(frozen=True, slots=True)
class Hardware:
    """GPU. `peak_tflops_f16` / `bandwidth_gbps` are **effective values**
    (§6.2).

    Using spec values (at boost clock) puts the ridge point 26% above reality
    and flips the `is_memory_bound` verdict near the boundary. Use
    `hardware_from_env()` as the **only entry point** —
    `Hardware(**env["hardware"])` gives you the spec values.
    """

    name: str
    arch: str
    sm_count: int
    smem_per_block: int
    max_threads_per_sm: int
    regs_per_sm: int
    peak_tflops_f16: float   # effective
    bandwidth_gbps: float    # effective
    l2_bytes: int

    @property
    def ridge_point(self) -> float:
        """The roofline knee [FLOP/byte]. peak_flops / bandwidth."""
        return (self.peak_tflops_f16 * 1e12) / (self.bandwidth_gbps * 1e9)


@dataclass(frozen=True, slots=True)
class Config:
    """What the rule orders. **No measured times.**

    Common fields + kernel attributes known at build time + `ext`.

    Keeping `ext` a dict is deliberate (§6.1). The fields differ per
    architecture, and a feature aiming at architecture transfer must not look
    at `ext` — making them dataclass fields would only encourage access.
    """

    # Common — physical features must compute from these alone for transfer
    # to hold (§4.3)
    tile_m: int
    tile_n: int
    tile_k: int
    align_a: int
    align_b: int
    align_c: int
    split_k: int
    split_k_mode: str        # "serial" | "parallel"
    arch: str
    # Kernel attributes known at build time — no execution needed, so usable
    # (§3.2)
    kernel_id: str
    regs_per_thread: int
    threads: int
    smem_bytes: int
    spill_bytes: int
    max_blocks_per_sm: int
    pipeline_kind: str       # "pipelined" | "multistage"
    #: ★ 2026-09-11 (D-161): the operand-buffer stage count, **lifted out of
    #: `ext`**.
    #:
    #: `pipeline_kind` is this value projected onto two levels — measured on
    #: all four tables, `pipelined` is exactly `stages == 2` and
    #: `multistage` is exactly `stages >= 3`, with no row disagreeing. So
    #: the depth was already exposed, flattened.
    #:
    #: It stays out of `ext` under the §4.3 rule only if it does not
    #: transfer, and it does: the value set is {2,3,4,5,6,7,8} on the A6000,
    #: the 5090, the 4090 and the H100 alike, with **0 missing rows**. The
    #: `swizzle` fields stay in `ext` — `identity`/`horizontal` are SM80
    #: words (D-75).
    #:
    #: ⚠️ `0` means **the bundle has no `ext_stages` column**. Every bundle
    #: measured so far has it, and so does the synthetic generator; a
    #: feature that sees 0 is looking at a table that cannot answer, not at
    #: a kernel with zero stages.
    stages: int = 0
    #: SASS instruction count. Known at build time, common across
    #: architectures. GBDT ranked it highly but the hand rule never used it
    #: (§30.6b).
    inst_total: int = 0
    # Architecture-specific. A transferable rule must not reference it.
    ext: dict = field(default_factory=dict)

    @property
    def key(self) -> ConfigKey:
        return (self.kernel_id, self.split_k, self.split_k_mode)


def shape_key(p: Problem) -> ShapeKey:
    return p.key


def config_key(c: Config) -> ConfigKey:
    return c.key


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """All candidates of one shape, as column arrays.

    ★ **That this object has no times is the whole of the §30.7 defence.**

    Code such as `idxmin` or `sorted(..., key=lambda c: (score, time))` needs
    a field that is not here, so it fails immediately with `AttributeError`.
    The scorer obtains times only through `PerfTable.times_of()`, and that
    value is used purely for indexing **after the order is already decided**.

    `tiebreak` is an integer rank built from config identity alone. Fed as the
    primary key of `np.lexsort` it makes score ties deterministic — and that
    matters in practice, because timer quantisation produces ties in bulk
    (at 512³ one modal value covers 9.2%).
    """

    n: int
    kernel_id: np.ndarray        # (n,) object/str
    split_k: np.ndarray          # (n,) int
    split_k_mode: np.ndarray     # (n,) object/str
    #: Deterministic rank built from config identity alone.
    #: **Independent of the answer.**
    tiebreak: np.ndarray         # (n,) int64
    #: Config objects (for the deployment shim and reports). May be lazy.
    configs: tuple[Config, ...] = ()
    #: Row indices in the source table. Used only by the scorer to find times.
    row_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("kernel_id", "split_k", "split_k_mode", "tiebreak"):
            arr = getattr(self, name)
            if len(arr) != self.n:
                raise ValueError(
                    f"CandidateSet.{name} length {len(arr)} != n {self.n}")

    def order_by(self, score: np.ndarray) -> np.ndarray:
        """Indices sorted by ascending score. **Ties break on config
        identity only.**

        ⚠️ Do not add time as a secondary key here — that was the §30.7 bug.
        It is structurally impossible (this class has no times), but the
        discipline is written next to the code.
        """
        score = np.asarray(score, dtype=np.float64)
        if score.shape != (self.n,):
            raise ValueError(
                f"score shape {score.shape} != number of candidates "
                f"({self.n},). The rule must return a score vector over all "
                "candidates of one shape.")
        if not np.all(np.isfinite(score)):
            # Do not silently push nan to the back (§26.4). The rule is broken.
            n_bad = int((~np.isfinite(score)).sum())
            raise ValueError(
                f"{n_bad} non-finite values in the score (nan/inf). "
                "Rejecting the rule.")
        # In lexsort the last key is primary. (tiebreak, score) -> score wins.
        return np.lexsort((self.tiebreak, score))

    def top_k(self, score: np.ndarray, k: int) -> np.ndarray:
        """Top k only. **Exactly the same result as a full sort**, in O(n).

        Scoring looks only at the top k (k <= 10), and sorting all 15,000
        every time makes that cost dominate over 200 weight fits x 66 shapes.
        Measured at 14.5 s per rule, which breaks §29.2's premise that
        "scoring is essentially free".

        Correctness: collecting only elements with score <= `kth` suffices.
        Any other element has a score above `kth`, and there are already at
        least k elements at or below `kth`, so it cannot make the top k.
        **Ties still break on the tie-break alone.**
        """
        score = np.asarray(score, dtype=np.float64)
        if score.shape != (self.n,):
            raise ValueError(
                f"score shape {score.shape} != number of candidates "
                f"({self.n},).")
        if not np.all(np.isfinite(score)):
            raise ValueError(
                f"{int((~np.isfinite(score)).sum())} non-finite values in "
                "the score. Rejecting.")
        k = int(k)
        if k >= self.n:
            return np.lexsort((self.tiebreak, score))[:k]
        kth = np.partition(score, k - 1)[k - 1]
        pool = np.flatnonzero(score <= kth)
        return pool[np.lexsort((self.tiebreak[pool], score[pool]))][:k]


def make_tiebreak(kernel_id, split_k, split_k_mode) -> np.ndarray:
    """Build a deterministic integer rank from config identity.
    **Independent of the answer.**

    It does not depend on the table's row order — `groupby.idxmin()` did, and
    that made "the best config per shape" change with the tie-break, which was
    observed (29 of 66 shapes have ties at the best time, up to 84-way).
    """
    # `np.unique(..., return_inverse=True)` codes are **lexicographic**
    # (pandas' factorize is first-appearance order, which depends on row order
    #  — do not use it).
    kid = np.unique(np.asarray(kernel_id, dtype=object).astype(str),
                    return_inverse=True)[1]
    mode = np.unique(np.asarray(split_k_mode, dtype=object).astype(str),
                     return_inverse=True)[1]
    sk = np.asarray(split_k, dtype=np.int64)
    order = np.lexsort((mode, sk, kid))          # kernel_id is the primary key
    rank = np.empty(len(sk), dtype=np.int64)
    rank[order] = np.arange(len(sk), dtype=np.int64)
    return rank


def hardware_from_env(env: dict) -> Hardware:
    """`env.json` -> `Hardware`. **Every caller uses only this** (§6.2).

    Delegates to kernelTab's function of the same name, then copies into our
    dataclass — so the effective-value correction is not reimplemented.
    """
    from kerneltab.core.hardware import hardware_from_env as _kt

    kt = _kt(env)
    hw = Hardware(
        name=kt.name, arch=kt.arch, sm_count=kt.sm_count,
        smem_per_block=kt.smem_per_block,
        max_threads_per_sm=kt.max_threads_per_sm,
        regs_per_sm=kt.regs_per_sm,
        peak_tflops_f16=kt.peak_tflops_f16,
        bandwidth_gbps=kt.bandwidth_gbps,
        l2_bytes=kt.l2_bytes,
    )
    # Check that the effective correction actually happened. Spec values
    # coming through unchanged means env.json has no *_effective, and then the
    # ridge point is wrong. Do not let it pass silently (§26.4).
    spec = env.get("hardware", {})
    if (env.get("peak_tflops_f16_effective") is None
            and spec.get("peak_tflops_f16") is not None):
        import warnings
        warnings.warn(
            "env.json has no peak_tflops_f16_effective. The spec value "
            "(boost clock) is used and the ridge point comes out above "
            "reality — the is_memory_bound verdict flips near the boundary "
            "(§6.2).", stacklevel=2)
    return hw


def config_from_row(row: dict[str, Any]) -> Config:
    """One adapter-normalised row -> Config. Called by `core/adapter.py`."""
    ext = {k[len("ext_"):]: v for k, v in row.items() if k.startswith("ext_")}
    return Config(
        tile_m=int(row["tile_m"]), tile_n=int(row["tile_n"]),
        tile_k=int(row["tile_k"]),
        align_a=int(row["align_a"]), align_b=int(row["align_b"]),
        align_c=int(row["align_c"]),
        split_k=int(row["split_k"]), split_k_mode=str(row["split_k_mode"]),
        arch=str(row["arch"]), kernel_id=str(row["kernel_id"]),
        regs_per_thread=int(row["regs_per_thread"]),
        threads=int(row["threads"]),
        smem_bytes=int(row["smem_bytes"]),
        spill_bytes=int(row["spill_bytes"]),
        max_blocks_per_sm=int(row["max_blocks_per_sm"]),
        pipeline_kind=str(row["pipeline_kind"]),
        # ★ D-161. `ext` keeps its copy — `ext` is the raw record of the
        #   table and nothing that reads it should change.
        stages=int(row.get("ext_stages") or 0),
        inst_total=int(row.get("inst_total") or 0),
        ext=ext,
    )
