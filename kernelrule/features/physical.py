"""Physical features written by a human (§8.2). **They do not judge.**

There are two reasons this file must exist before the LLM loop. (1) Depending
on LLM generation means nothing works at the start. (2) It is the baseline
against which "did the features the LLM added contribute" is measured (§16.1
ablation).

## Writing rules — enforced identically on the LLM

    Pure function. Computed from (Problem, Hardware, Config) only
    A single float. **Unify the direction so that larger is worse** — the
                 rule is then always "weighted sum, then ascending sort", and
                 there is no room for the LLM to get the sign confused
    Read hardware constants from hw.*. No hardcoding of 84 / 101376
    No reference to cfg.ext — the architecture-transfer premise (§4.3)
    At most 10 lines

## The vectorisation (`vec=`) is **the same physics written as arrays**

The scalar implementation is the contract and the vectorisation is speed. If
the two diverge, training (the matrix) and deployment (the scalar) use
different functions, so `verify_vectorized()` compares them on a sample and
**a mismatch is a rejection** (§26.4).

## Facts confirmed on this table (§18.3) — what the features must capture

    A spilling kernel was picked as optimal 0 times (rel median 13.6, max 37.2)
    warp_m=128 is optimal 0 times — it points at the same thing as spilling
    stages=2 (MmaPipelined) is a different kernel family from stages>=3
      (multistage)
    An alignment-1 shape cannot use cp.async, so only stages=2 is possible
    split_k_mode is serial in 66/66. parallel occupies 400k rows and is
      optimal 0 times
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np

from kernelrule.core.types import Config, Hardware, Problem
from kernelrule.features import REGISTRY
from kernelrule.features import feature as _feature
from kernelrule.features import shape_feature as _shape_feature

# ★ **Every** feature in this module goes into `REGISTRY`. That binding is
#   nailed down here in one line — `feature()` itself has no default
#   (§30.9). This avoids repeating `registry=REGISTRY` on every decorator
#   while still making where things register visible at a glance at the top
#   of the file.
feature = partial(_feature, registry=REGISTRY)
shape_feature = partial(_shape_feature, registry=REGISTRY)

__all__ = ["REGISTRY"]

_DTYPE_BYTES = {"f16": 2, "bf16": 2, "f8": 1, "f32": 4}


def _ebytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(str(dtype), 2)


def _v_ebytes(df) -> np.ndarray:
    return np.asarray([_DTYPE_BYTES.get(str(d), 2) for d in df["dtype"]],
                      dtype=np.float64)


# ---------------------------------------------------------------------------
# Grid and waves — how many times the GPU is filled
# ---------------------------------------------------------------------------
def _v_tiles_mn(df):
    return (np.ceil(df["M"].to_numpy(np.float64) / df["tile_m"].to_numpy(np.float64)),
            np.ceil(df["N"].to_numpy(np.float64) / df["tile_n"].to_numpy(np.float64)))


def _v_grid_tiles(df):
    gm, gn = _v_tiles_mn(df)
    return gm * gn * df["split_k"].to_numpy(np.float64)


@feature(unit="count", expected_range=(1.0, 1e7), direction="neutral",
         vec=lambda df, hw, p: np.log2(1.0 + _v_grid_tiles(df)))
def log_grid_tiles(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The number of CTAs (log2). How large the grid is. No direction — it
    is a scale indicator."""
    tiles = (math.ceil(p.M / cfg.tile_m) * math.ceil(p.N / cfg.tile_n)
             * cfg.split_k)
    return math.log2(1.0 + tiles)


def _v_waves(df, hw):
    occ = np.maximum(1.0, df["max_blocks_per_sm"].to_numpy(np.float64))
    return _v_grid_tiles(df) / (hw.sm_count * occ)


@feature(unit="dimensionless", expected_range=(0.0, 1e5), direction="neutral",
         vec=lambda df, hw, p: _v_waves(df, hw))
def waves(p: Problem, hw: Hardware, cfg: Config) -> float:
    """How many times the grid fills the GPU. It accounts for occupancy.

    Below 1, SMs are left over. No direction — larger is not worse.
    """
    tiles = (math.ceil(p.M / cfg.tile_m) * math.ceil(p.N / cfg.tile_n)
             * cfg.split_k)
    return tiles / (hw.sm_count * max(1, cfg.max_blocks_per_sm))


def _v_tail_waste(df, hw):
    w = np.maximum(_v_waves(df, hw), 1e-12)
    full = np.ceil(w)
    return (full - w) / full


@feature(expected_range=(0.0, 1.0), direction="higher_is_worse",
         vec=lambda df, hw, p: _v_tail_waste(df, hw))
def tail_waste(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The fraction of SM slots idle in the last wave. 0~1, larger is
    worse.

    When waves is large it naturally approaches 0 — so this term fades away
    by itself on large shapes, without any conditional branch (the "good
    fix" of §2.3).
    """
    w = max(1e-12, waves(p, hw, cfg))
    full = math.ceil(w)
    return (full - w) / full


@feature(expected_range=(0.0, 10.0), direction="higher_is_worse",
         vec=lambda df, hw, p: 1.0 / np.maximum(1.0 - _v_tail_waste(df, hw), 1e-3)
         - 1.0)
def sm_idle_cost(p: Problem, hw: Hardware, cfg: Config) -> float:
    """**How many times** worse wave quantisation makes it, minus 1. The
    non-linear form of `tail_waste`.

    Using 128x128 on 512³ gives 16 tiles, so only 16 of 84 SMs run and the
    factor is 5.3x. A linear term (`tail_waste=0.81`) cannot produce that
    magnitude — the hand rule really did miss this, and went from 1.221 down
    to 1.192 (kernelTab baselines.md).
    """
    return 1.0 / max(1e-3, 1.0 - tail_waste(p, hw, cfg)) - 1.0


# ---------------------------------------------------------------------------
# What a tile actually does — the shape x config interaction
# ---------------------------------------------------------------------------
def _v_edge_waste(df):
    gm, gn = _v_tiles_mn(df)
    tm = df["tile_m"].to_numpy(np.float64)
    tn = df["tile_n"].to_numpy(np.float64)
    M = df["M"].to_numpy(np.float64)
    N = df["N"].to_numpy(np.float64)
    return (gm * tm / M) * (gn * tn / N) - 1.0


@feature(expected_range=(0.0, 300.0), direction="higher_is_worse",
         vec=lambda df, hw, p: _v_edge_waste(df))
def edge_waste(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The multiple of **wasted work** where a tile crosses the shape
    boundary, minus 1.

    A tile computes everything it covers even outside the shape. With a
    128-row tile on M=1, 99.2% of the work is thrown away (value 127). This
    is why a small M prefers a small tile_m, and it is where the shape and
    the config meet.
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    return (gm * cfg.tile_m / p.M) * (gn * cfg.tile_n / p.N) - 1.0


@feature(unit="bytes", expected_range=(0.0, 100.0), direction="higher_is_worse",
         vec=lambda df, hw, p: (lambda gm, gn: np.log2(
             1.0 + gm * gn * (df["tile_m"].to_numpy(np.float64)
                              + df["tile_n"].to_numpy(np.float64))
             * df["K"].to_numpy(np.float64) * _v_ebytes(df)))(*_v_tiles_mn(df)))
def log_dram_traffic(p: Problem, hw: Hardware, cfg: Config) -> float:
    """How many bytes of A/B are read from DRAM (log2). The tiling decides
    it.

    `ceil(M/tm)*ceil(N/tn)*(tm+tn)*K*elem` — a larger tile means more reuse,
    so the total traffic falls. At M=1, `gm=1`, so raising tm does not
    reduce the tile count and only `(tm+tn)` grows, which makes it
    **automatically a penalty**.
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    b = gm * gn * (cfg.tile_m + cfg.tile_n) * p.K * _ebytes(p.dtype)
    return math.log2(1.0 + b)


@feature(expected_range=(0.0, 200.0), direction="higher_is_worse",
         vec=lambda df, hw, p: (lambda gm, gn: (
             gm * gn * (df["tile_m"].to_numpy(np.float64)
                        + df["tile_n"].to_numpy(np.float64))
             * df["K"].to_numpy(np.float64) * _v_ebytes(df))
             / np.maximum(_v_ebytes(df) * (df["M"].to_numpy(np.float64)
                                           * df["K"].to_numpy(np.float64)
                                           + df["K"].to_numpy(np.float64)
                                           * df["N"].to_numpy(np.float64)), 1.0)
         )(*_v_tiles_mn(df)))
def traffic_amplification(p: Problem, hw: Hardware, cfg: Config) -> float:
    """Actual A/B traffic / the theoretical minimum. 1.0 is perfect reuse.

    The dimensionless form of `log_dram_traffic`. Dividing out the shape
    size makes it transferable (§8.1).
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    eb = _ebytes(p.dtype)
    actual = gm * gn * (cfg.tile_m + cfg.tile_n) * p.K * eb
    ideal = eb * (p.M * p.K + p.K * p.N)
    return actual / max(1.0, ideal)


@feature(expected_range=(0.0, 8.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.abs(np.log2(
             df["tile_m"].to_numpy(np.float64)
             / df["tile_n"].to_numpy(np.float64))))
def tile_aspect_imbalance(p: Problem, hw: Hardware, cfg: Config) -> float:
    """How elongated the tile is. |log2(tm/tn)|. A square is 0.

    At equal area, a square minimises A/B traffic.
    """
    return abs(math.log2(cfg.tile_m / cfg.tile_n))


# ---------------------------------------------------------------------------
# mainloop depth and split-K
# ---------------------------------------------------------------------------
@feature(unit="count", expected_range=(0.0, 16.0), direction="neutral",
         vec=lambda df, hw, p: np.log2(np.maximum(1.0,
             df["K"].to_numpy(np.float64)
             / (df["tile_k"].to_numpy(np.float64)
                * df["split_k"].to_numpy(np.float64)))))
def log_mainloop_iters(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The number of mainloop iterations (log2). `K / (tile_k * split_k)`.

    The axis GBDT ranked most important (§30.6b). When it is short the
    pipeline warm-up cost cannot be paid back; when it is long A/B reuse
    works well.
    """
    return math.log2(max(1.0, p.K / (cfg.tile_k * cfg.split_k)))


def _v_stages_est(df):
    """Derives the pipeline depth backwards from smem. **It does not look
    at `ext`.**"""
    denom = np.maximum(1.0, df["tile_k"].to_numpy(np.float64)
                       * (df["tile_m"].to_numpy(np.float64)
                          + df["tile_n"].to_numpy(np.float64)) * _v_ebytes(df))
    return np.maximum(1.0, df["smem_bytes"].to_numpy(np.float64) / denom)


@feature(expected_range=(0.0, 4.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.clip(
             _v_stages_est(df)
             / np.maximum(1.0, df["K"].to_numpy(np.float64)
                          / (df["tile_k"].to_numpy(np.float64)
                             * df["split_k"].to_numpy(np.float64))), 0.0, 4.0))
def pipeline_warmup_frac(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The fraction of the mainloop taken up by filling the pipeline.

    ★ The depth is **derived backwards from smem** instead of being read
    from an architecture-specific extension field. CUTLASS's mainloop smem
    is `stages * tile_k * (tile_m+tile_n) * elem`, so dividing gives the
    depth. Because it does not look at `ext` it transfers across
    architectures (§4.3), and where the notion of stages differs, as on
    SM90, it still means something as "how deep the smem was stacked".

    A deep pipeline cannot pay back its warm-up when the mainloop is short.
    """
    denom = max(1.0, cfg.tile_k * (cfg.tile_m + cfg.tile_n) * _ebytes(p.dtype))
    stages = max(1.0, cfg.smem_bytes / denom)
    iters = max(1.0, p.K / (cfg.tile_k * cfg.split_k))
    return min(4.0, stages / iters)


@feature(expected_range=(0.0, 1.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.log2(df["split_k"].to_numpy(np.float64)) / 4.0)
def split_k_cost(p: Problem, hw: Hardware, cfg: Config) -> float:
    """A proxy for the split-K reduction cost. log2(sk)/4, and 0 at sk=1.

    serial split-K round-trips D per partition, so the cost grows with the
    number of partitions. The log is used because the difference from sk=1
    to 2 is larger than from 8 to 16.
    """
    return math.log2(cfg.split_k) / 4.0


@feature(expected_range=(0.0, 40.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.log2(
             1.0 + np.where(df["split_k_mode"].to_numpy().astype(str) == "parallel",
                            _v_ebytes(df) * df["M"].to_numpy(np.float64)
                            * df["N"].to_numpy(np.float64)
                            * df["split_k"].to_numpy(np.float64), 0.0)))
def log_workspace_bytes(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The reduction traffic of parallel split-K (log2). 0 for serial.

    GBDT ranked it highly and the hand rule did not use it (§30.6b).
    parallel writes M*N*sk partials to DRAM and reads them back — this can
    explain why serial is optimal on 66/66 shapes.
    """
    if cfg.split_k_mode != "parallel":
        return 0.0
    return math.log2(1.0 + _ebytes(p.dtype) * p.M * p.N * cfg.split_k)


# ---------------------------------------------------------------------------
# Resource pressure — kernel properties knowable at build time (§3.2)
# ---------------------------------------------------------------------------
@feature(expected_range=(0.0, 1.5), direction="higher_is_worse",
         vec=lambda df, hw, p: df["smem_bytes"].to_numpy(np.float64)
         / hw.smem_per_block)
def smem_pressure(p: Problem, hw: Hardware, cfg: Config) -> float:
    """How much of the smem budget it uses. Filling it reduces the resident
    blocks per SM."""
    return cfg.smem_bytes / hw.smem_per_block


@feature(expected_range=(0.0, 4.0), direction="higher_is_worse",
         vec=lambda df, hw, p: df["regs_total_per_block"].to_numpy(np.float64)
         / hw.regs_per_sm if "regs_total_per_block" in df.columns
         else df["regs_per_thread"].to_numpy(np.float64)
         * df["threads"].to_numpy(np.float64) / hw.regs_per_sm)
def reg_pressure(p: Problem, hw: Hardware, cfg: Config) -> float:
    """How many times the SM register file one block demands."""
    return cfg.regs_per_thread * cfg.threads / hw.regs_per_sm


@feature(expected_range=(0.0, 1.0), direction="higher_is_worse",
         vec=lambda df, hw, p: 1.0 - np.clip(
             df["max_blocks_per_sm"].to_numpy(np.float64)
             * df["threads"].to_numpy(np.float64) / hw.max_threads_per_sm,
             0.0, 1.0))
def occupancy_deficit(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The fraction of thread slots per SM left unfilled. 0 is perfect.

    `max_blocks_per_sm` is a build-time value, so it may be used (§3.2).
    """
    used = cfg.max_blocks_per_sm * cfg.threads / hw.max_threads_per_sm
    return 1.0 - min(1.0, max(0.0, used))


@feature(expected_range=(0.0, 1.0), direction="higher_is_worse",
         vec=lambda df, hw, p: (df["spill_bytes"].to_numpy(np.float64) > 0
                                ).astype(np.float64))
def has_spill(p: Problem, hw: Hardware, cfg: Config) -> float:
    """Is there a register spill? 0 or 1.

    ★ In this table a spilling kernel was **picked as optimal 0 times**, and
    its rel median is 13.6 (max 37.2). It is 7.4% of all rows, and the whole
    long tail comes from here.
    """
    return 1.0 if cfg.spill_bytes > 0 else 0.0


@feature(expected_range=(0.0, 10.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.log2(
             1.0 + df["spill_bytes"].to_numpy(np.float64)) / 4.0)
def spill_magnitude(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The size of the spill (log2/4). Not just whether, but how much."""
    return math.log2(1.0 + cfg.spill_bytes) / 4.0


@feature(expected_range=(0.0, 1.0), direction="higher_is_worse",
         vec=lambda df, hw, p: (df["pipeline_kind"].to_numpy().astype(str)
                                == "pipelined").astype(np.float64))
def is_two_stage(p: Problem, hw: Hardware, cfg: Config) -> float:
    """Is it a two-stage pipeline (MmaPipelined)? A **different kernel
    family** from multistage.

    It looks at `pipeline_kind` rather than `ext.stages` — that one is a
    field common across architectures, so it transfers (§4.3).
    """
    return 1.0 if cfg.pipeline_kind == "pipelined" else 0.0


@feature(unit="count", expected_range=(0.0, 30.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.log2(1.0 + df["inst_total"].to_numpy(np.float64))
         if "inst_total" in df.columns else np.zeros(len(df)))
def log_inst_total(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The number of SASS instructions (log2). A proxy for kernel
    complexity.

    GBDT ranked it highly and the hand rule did not use it (§30.6b). It is
    knowable at build time, so it may be used. If absent from the table it
    is 0 — in that case this term becomes a constant and trips `validate`'s
    "constant" check.
    """
    return math.log2(1.0 + float(cfg.inst_total))


# ---------------------------------------------------------------------------
# Shape level — **scalars**, so a rule may use `if` (the §8.1 replacement)
# ---------------------------------------------------------------------------
@shape_feature(unit="flop/byte", expected_range=(0.0, 1e5), direction="neutral")
def arith_intensity(p: Problem, hw: Hardware, cfg: Config) -> float:
    """A function of the shape alone. 2MNK / (bytes read and written)."""
    eb = _ebytes(p.dtype)
    return 2.0 * p.M * p.N * p.K / max(1.0, eb * (p.M * p.K + p.K * p.N
                                                  + p.M * p.N))


@shape_feature(expected_range=(0.0, 1e4), direction="neutral")
def roofline_ratio(p: Problem, hw: Hardware, cfg: Config) -> float:
    """AI / ridge point. Below 1 it is memory-bound.

    `hw.ridge_point` is computed from the **effective** values (§6.2). Using
    the spec values is off by 26% and flips the class of shapes near the
    boundary.
    """
    return arith_intensity(p, hw, cfg) / hw.ridge_point


@shape_feature(expected_range=(0.0, 1.0), direction="neutral")
def is_memory_bound(p: Problem, hw: Hardware, cfg: Config) -> float:
    """Is it memory-bound? 0 or 1. **A rule writes
    `if p.is_memory_bound:`.**"""
    return 1.0 if roofline_ratio(p, hw, cfg) < 1.0 else 0.0


def log_sol_ms(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The roofline lower-bound time (log2, ms). **Computed from the
    shape, not measured.**

    ⚠️ 2026-09-10 (D-156): **no longer a registered feature.** It used to
    carry `@shape_feature`, so a rule could branch on `p.log_sol_ms` — and
    the diagnostic report handed the model the SOL 0.5 ms split every round,
    so it did. That is **our** axis, chosen in 2026-08, and D-143 had already
    found it indefensible as a regime boundary. Feeding it back through the
    report made the model rediscover it every round (the r9 rule of D-155
    branches on `p.log_sol_ms < -5`, then drifts to `log_flops > 20/40/30`).

    ★ The function stays because the **scoring** path uses it: the canonical
    procedure refits per regime on this split (`core/canonical.py`,
    §10.2 · D-69), and every committed number was produced that way.
    ⚠️ So the axis is gone from what the model sees, **not** from how we
    score. That asymmetry is deliberate and is written down in D-156.

    If it matters, FeatureWriter can invent it again — and then it is the
    model's axis, not ours.
    """
    eb = _ebytes(p.dtype)
    t_c = 2.0 * p.M * p.N * p.K / (hw.peak_tflops_f16 * 1e12) * 1e3
    t_m = eb * (p.M * p.K + p.K * p.N + p.M * p.N) / (hw.bandwidth_gbps
                                                      * 1e9) * 1e3
    return math.log2(max(1e-6, t_c, t_m))


# ---------------------------------------------------------------------------
# ★ 2026-09-08 (D-144) — four more shape-level values to branch on.
#   There were only four so far (`is_memory_bound` · `roofline_ratio` ·
#   `log_sol_ms` · `arith_intensity`), too little material for the model to
#   find a regime with.
#   ★ All are determined **by the shape alone** — no config enters.
# ---------------------------------------------------------------------------
@shape_feature(unit="log2 flop", expected_range=(0.0, 80.0),
               direction="neutral")
def log_flops(p: Problem, hw: Hardware, cfg: Config) -> float:
    """log2(2·M·N·K). The absolute size of the problem."""
    return math.log2(max(1.0, 2.0 * p.M * p.N * p.K))


@shape_feature(expected_range=(-30.0, 30.0), direction="neutral")
def aspect_MN(p: Problem, hw: Hardware, cfg: Config) -> float:
    """log2(M/N). How elongated the shape is. 0 is square."""
    return math.log2(max(1.0, float(p.M)) / max(1.0, float(p.N)))


@shape_feature(expected_range=(0.0, 1e5), direction="neutral")
def reuse_ratio(p: Problem, hw: Hardware, cfg: Config) -> float:
    """M·N·K / (M·K + K·N + M·N). The amount of data reuse.

    Its form resembles `arith_intensity`, but **the dtype bytes do not
    enter** — it is a pure shape quantity.
    """
    return (float(p.M) * p.N * p.K
            / max(1.0, float(p.M) * p.K + float(p.K) * p.N
                  + float(p.M) * p.N))


@shape_feature(unit="log2", expected_range=(0.0, 30.0), direction="neutral")
def log_min_dim(p: Problem, hw: Hardware, cfg: Config) -> float:
    """log2(min(M,N,K)). The shortest axis — it separates skinny
    shapes."""
    return math.log2(max(1.0, float(min(p.M, p.N, p.K))))


@shape_feature(expected_range=(0.0, 1.0), direction="neutral")
def can_use_cp_async(p: Problem, hw: Hardware, cfg: Config) -> float:
    """Does the alignment permit cp.async? At 0, only stages=2 is possible.

    ★ A physical fact confirmed in the table (§18.3). An alignment-1 shape
    has no multistage kernel at all.
    """
    return 1.0 if min(p.M, p.N, p.K) and _align_ok(p) else 0.0


def _align_ok(p: Problem) -> bool:
    """Does the K-direction access satisfy 16-byte alignment (the cp.async
    requirement)?"""
    elems = 16 // _ebytes(p.dtype)
    return (p.K % elems == 0) and (p.N % elems == 0)


# ---------------------------------------------------------------------------
# ★ Physical meaning — not "what does it measure" but "why does it drive
# performance" (§12.3b)
# ---------------------------------------------------------------------------
# Previously only the one-line summary (`doc`) went into the prompt.
# `has_spill` looked like this:
#
#     f.has_spill    [0, 1]    Is there a register spill? 0 or 1.
#
# The range is the same as `tail_waste`'s, so it reads as **a penalty of
# similar size**. In reality they are orders of magnitude apart — removing
# that term takes regret from 1.1637 to 3.1841
# (`docs/artifacts/spill-term.md`). RuleWriter under condition A did not pick
# that term, and that one thing trapped condition (a) of the seed experiment.
#
# ⚠️ What is written here is only **what is knowable without the table**
# (§12.3b).
#     physics (allowed)  "registers overflow into local memory. An access is
#                         tens of times slower"
#     observed (banned)  "in this table a spilling kernel never entered the
#                         answer set"
#
# Magnitudes are written **only where they follow from the formula** (for
# `1/(1-x)`, 0.5 gives 2x). Multiples that came out of measurement are not
# written.

_PHYSICS: dict[str, str] = {
    # -- resource limits. A cliff, not a slope --------------------------------
    "has_spill":
        "Registers overflow into local memory (= DRAM). If a register access "
        "is one cycle, local is hundreds, and it happens every mainloop "
        "iteration. ★ Once it is on, no other advantage easily offsets it — "
        "give it the largest penalty",
    "spill_magnitude":
        "How much spilling (log2 bytes / 4). Separates how bad it is once "
        "`has_spill` is on. A 4 means 16x more bytes round-tripping",
    "reg_pressure":
        "How many times the SM register file one block demands. Above 1, no "
        "block fits on an SM or it spills — the cliff is right around 1",
    "smem_pressure":
        "How much of the smem budget is used. Filling it drops residency to "
        "one block per SM, so memory latency cannot be hidden behind another "
        "block. A staircase, not a cliff",

    # -- amount of work. Grows linearly ---------------------------------------
    "traffic_amplification":
        "Actual A/B traffic / theoretical minimum. Small tiles re-read the "
        "same data many times. On memory-bound shapes it is nearly "
        "proportional to time. ★ The value is large, so taking log2 matches "
        "its magnitude to the other terms",
    "log_dram_traffic":
        "Bytes read from DRAM (log2). When bandwidth is the ceiling, this is "
        "the time",
    "edge_waste":
        "Work multiplier from tiles overhanging the shape, minus 1. A 128-row "
        "tile on M=1 wastes 99.2% of the work. ★ The range is wide (hundreds), "
        "so use a correspondingly small weight or saturate it",
    "log_inst_total":
        "SASS instruction count (log2). A proxy for time when compute-bound",
    "log_mainloop_iters":
        "Mainloop iterations (log2) = K/(tile_k*split_k). When small, filling "
        "the pipeline costs relatively more",

    # -- how full the machine is ----------------------------------------------
    "tail_waste":
        "Fraction of SM slots idle on the last wave. The loss is roughly "
        "1/(1-x) — 2x at 0.5, 5x at 0.8. ★ Used linearly, that magnitude does "
        "not come out",
    "sm_idle_cost":
        "The non-linear form of the above: 1/(1-tail_waste) - 1. The actual "
        "loss multiplier. It diverges as tail_waste approaches 1, so the "
        "implementation clips it",
    "occupancy_deficit":
        "Fraction of per-SM thread slots that cannot be filled. Less room to "
        "hide memory latency. Matters less when compute-bound",
    "waves":
        "How many times the grid fills the GPU. Below 1 the GPU idles; the "
        "closer to an integer, the less last-wave waste",
    "log_grid_tiles":
        "CTA count (log2). A scale indicator with no direction — better used "
        "as a condition on other terms",

    # -- pipelining and kernel family -----------------------------------------
    "is_two_stage":
        "Is this a 2-stage pipeline (MmaPipelined)? It is a **different "
        "kernel family** from multistage, with entirely different performance "
        "characteristics. Without cp.async, only this is possible",
    "can_use_cp_async":
        "Does alignment allow cp.async (16 bytes)? At 0, only stages=2 is "
        "possible and the global->smem copy must go through registers",
    "pipeline_warmup_frac":
        "Share of the mainloop spent filling the pipeline. Larger when the "
        "mainloop is short — it marks where adding stages starts to cost",

    # -- the price of splitting -----------------------------------------------
    "split_k_cost":
        "Proxy for the split-K reduction cost. Dividing K buys parallelism "
        "but the partials must be combined. serial round-trips in fp16 "
        "(precision loss); parallel writes M*N*sk to DRAM and reads it back",
    "log_workspace_bytes":
        "Partial-sum bytes parallel split-K writes to DRAM (log2). 0 for "
        "serial",

    # -- tile shape -----------------------------------------------------------
    "tile_aspect_imbalance":
        "How elongated the tile is. |log2(tm/tn)|. Square is better for "
        "reuse, but on an elongated shape an elongated tile wastes less at "
        "the edges",

    # -- shape level (for branching) ------------------------------------------
    "is_memory_bound":
        "Is arithmetic intensity below the ridge point? If so the traffic "
        "terms dominate; otherwise instruction count and residency do. "
        "★ This is the main branch for deciding which terms to weight heavily",
    "roofline_ratio":
        "AI / ridge point. Below 1 is memory-bound. The continuous version of "
        "is_memory_bound, so the boundary can be handled smoothly",
    "arith_intensity":
        "2MNK / bytes moved. A function of the shape alone, so no config can "
        "change it",
    "log_flops":
        "log2(2*M*N*K). The absolute size of the problem",
    "aspect_MN":
        "log2(M/N). How elongated the shape is. 0 means square",
    "reuse_ratio":
        "M*N*K / (M*K + K*N + M*N). The amount of data reuse",
    "log_min_dim":
        "log2(min(M,N,K)). The shortest axis — separates skinny shapes",
}

for _name, _text in _PHYSICS.items():
    REGISTRY.annotate(_name, physical_meaning=_text)

# ★ Fixes the two whose declared range diverged from reality (§12.3b — it
#   comes from the formula, not the table)
#   log_grid_tiles: it returns log2(1+tiles) but declared a **linear** range.
#                   At 1e7 tiles, log2 is 24.
#   sm_idle_cost:   it is 1/max(1e-3, 1-x) - 1, so the implementation's upper
#                   bound is 999. It was declared as 10, so actual values
#                   exceeded the declaration.
REGISTRY.annotate("log_grid_tiles", expected_range=(0.0, 24.0))
REGISTRY.annotate("sm_idle_cost", expected_range=(0.0, 999.0))
