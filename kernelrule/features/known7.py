"""★ The starting library of condition F2 — **the seven public facts**
(§30.17).

★ Renamed 2026-09-04: this condition's old name was `F1-K` (D-128).
★ Renamed 2026-09-13 (D-170 §4): `known5` -> `known7`, `KNOWN5` -> `KNOWN7`,
  `F2-known5` -> `F2-known7`, `examples/known5.md` -> `examples/known7.md`.
  **The old names are not deleted from the record** — every run recorded
  before that date says `F2-known5` and its library really is the five.
  ⚠️ F2 is the same condition *name* with **different content**; a number
  from before and a number from after are not on the same condition.

## ★ Why two were added (D-170 §4)

The question is what the model builds on top of what is known, and a rule
branches with `if p.<x>:`. Counting the axes that can actually carry a
branch — shape level **and** not constant on the scored population:

```
F3 (the human 24)  ★ 8   arith_intensity · aspect_MN · can_use_cp_async
                         is_memory_bound · log_flops · log_min_dim
                         reuse_ratio · roofline_ratio
F2 (known5)        ★ 1   roofline_ratio
                         + whatever stage 1 generated -> ★ 10/10 libraries
                           of the sweep had 0 that varied -> effectively 1
```

So "F3 does better than F2" may be a statement about **branching material,
1 against 8**, and not about library size at all. That is a hypothesis for
the next experiment, written down before it is run.

Which two: **the two the evolution picked for itself** when D-156 removed
our SOL axis — `log_min_dim` 7 times and `log_flops` 7 times, with the
branch threshold stable from r0 to r11 (the SOL-based D-155 wandered
20 -> 40 -> 30). The model chose them; we are not inventing a pair.

Both are shape arithmetic — `log2(min(M,N,K))` and `log2(2MNK)` — so they
break no part of the "knowable without this table" condition.

## Why the base library is given at all

In F1 (starting from 0), nearly half the budget went into **re-inventing
physics that is already known** — of 21, 6 strict rediscoveries + 3 monotone
ones (D-63).

And those are **things one knows without this table.**

```
wave quantization      CUDA C++ Best Practices Guide
occupancy calculation  CUDA Occupancy Calculator
arithmetic intensity   Williams et al. (2009) roofline
tile edge waste        CUTLASS documentation, predication
register spilling      CUDA C++ Best Practices, Register Pressure
★ problem dimensions   the GEMM shape itself — M, N, K are the inputs
★ problem size         2MNK is the definition of GEMM's flop count
```

★ The last two are not "knowledge" in the same sense as the first five —
they are **the arithmetic of the inputs**, available to anyone holding the
shape. That is precisely why giving them costs the condition nothing.

**That knowledge is there when porting to a new GPU too.** "Does the LLM
rediscover these on its own" has been answered (largely, it reproduces them),
and what we want to know is **"what does it build on top of what is
known"**.

## Which seven — **ones of different shapes**

```
tail_waste          ratio      normalised to 0~1 with a physical upper bound
occupancy_deficit   ratio      uses a build-time value (max_blocks_per_sm)
roofline_ratio      threshold  ★ shape level (p.*). Read against 1
edge_waste          absolute   unbounded (0~300). Compression must be
                               considered
has_spill           binary     once on, the magnitude changes
★ log_min_dim       log        shape level. The shortest axis — it is what
                               separates M=1 from M=4096 on this table
★ log_flops         log        shape level. The absolute problem size
```

The three shape-level ones are deliberate — a rule branches with
`if p.<x>:`, and one such axis is not a regime, it is a switch.

## ★ The docstrings are a cleaned-up version — not the original

The docstrings in `physical.py` have **measurement results from this table**
mixed in. Sending them into the prompt as they are violates §12.3.

```
as it stands (physical.py)
    ★ In this table a spilling kernel was picked as optimal 0 times, and its
    rel median is 13.6 (max 37.2). It is 7.4% of all rows and ...
    -> all three numbers are measurement results. You need the table to know
```

The criterion:

```
Can it be known without the table?
  yes  -> physics. Keep it   "a 128-row tile on M=1 wastes 99.2%" (from the
                             formula)
  no   -> measurement. Drop  "a spilling kernel's rel median is 13.6x"
```

**The source is written on each feature.** For two reasons — (1) "is this
public knowledge" can be verified later. A description whose source cannot
be written is one that came out of this table. (2) The habit of writing
sources is itself passed on to the LLM.

⚠️ **`physical.py` is not modified.** Its docstrings are development
documentation for humans, and table observations belong there. **Only the
path that goes into the prompt** uses the cleaned-up version.
"""

from __future__ import annotations

import math

import numpy as np

from kernelrule.core.types import Config, Hardware, Problem
from kernelrule.features import FeatureRegistry
from kernelrule.features import feature as _feature
from kernelrule.features import shape_feature as _shape_feature

__all__ = ["KNOWN7", "SOURCES", "source_of"]

#: The starting registry of F2. **Only seven go in** — the other 17 are
#: condition F3.
KNOWN7 = FeatureRegistry("known7")

#: Feature -> public source. It is appended to the description in the
#: prompt.
SOURCES: dict[str, str] = {
    "tail_waste": 'CUDA C++ Best Practices Guide, "Thread and Block Heuristics"',
    "occupancy_deficit": "CUDA Occupancy Calculator",
    "roofline_ratio": 'Williams, Waterman, Patterson (2009), "Roofline"',
    "edge_waste": "CUTLASS documentation, predication",
    "has_spill": 'CUDA C++ Best Practices Guide, "Register Pressure"',
    # ★ D-170 §4. The source is the problem statement itself: a GEMM is
    #   given as (M, N, K), and 2MNK is the flop count in its definition
    #   (BLAS level-3 GEMM; NVIDIA's Matrix Multiplication Background User
    #   Guide states both). Nothing here needs this table.
    "log_min_dim": ("the GEMM problem statement — M, N, K are the inputs "
                    "(NVIDIA Matrix Multiplication Background User Guide)"),
    "log_flops": ("the definition of GEMM's flop count, 2MNK (NVIDIA "
                  "Matrix Multiplication Background User Guide)"),
}


def source_of(name: str) -> str:
    """The source. Missing means an exception — **a feature whose source
    cannot be named cannot come in here.**"""
    if name not in SOURCES:
        raise KeyError(
            f"there is no public source for {name!r}. F2's starting "
            f"library holds only 'what is known without this table' "
            f"(§30.17).")
    return SOURCES[name]


def _v_waves(df, hw):
    gm = np.ceil(df["M"].to_numpy(np.float64)
                 / df["tile_m"].to_numpy(np.float64))
    gn = np.ceil(df["N"].to_numpy(np.float64)
                 / df["tile_n"].to_numpy(np.float64))
    tiles = gm * gn * np.maximum(df["split_k"].to_numpy(np.float64), 1.0)
    return tiles / (hw.sm_count
                    * np.maximum(df["max_blocks_per_sm"].to_numpy(np.float64), 1.0))


def _v_tail(df, hw):
    w = np.maximum(_v_waves(df, hw), 1e-12)
    full = np.ceil(w)
    return (full - w) / full


@_feature(registry=KNOWN7, expected_range=(0.0, 1.0),
          direction="higher_is_worse",
          vec=lambda df, hw, p: _v_tail(df, hw))
def tail_waste(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The fraction of SM slots idle in the last wave. 0~1, larger is
    worse.

    CTAs are distributed across the SMs, and in the last batch some SMs sit
    idle. When waves is large it naturally approaches 0.
    Source: CUDA C++ Best Practices Guide, "Thread and Block Heuristics"
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    tiles = gm * gn * max(1, cfg.split_k)
    w = max(1e-12, tiles / (hw.sm_count * max(1, cfg.max_blocks_per_sm)))
    full = math.ceil(w)
    return (full - w) / full


@_feature(registry=KNOWN7, expected_range=(0.0, 1.0),
          direction="higher_is_worse",
          vec=lambda df, hw, p: 1.0 - np.clip(
              df["max_blocks_per_sm"].to_numpy(np.float64)
              * df["threads"].to_numpy(np.float64) / hw.max_threads_per_sm,
              0.0, 1.0))
def occupancy_deficit(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The fraction of thread slots per SM left unfilled. 0 is perfect.

    Empty slots leave less room to hide memory latency behind other warps.
    `max_blocks_per_sm` is fixed at compile time, so it may be used.
    Source: CUDA Occupancy Calculator
    """
    used = cfg.max_blocks_per_sm * cfg.threads / hw.max_threads_per_sm
    return 1.0 - min(1.0, max(0.0, used))


@_shape_feature(registry=KNOWN7, expected_range=(0.0, 1e4),
                direction="neutral")
def roofline_ratio(p: Problem, hw: Hardware, cfg: Config) -> float:
    """Arithmetic intensity / ridge point. Below 1 it is memory-bound.

    Arithmetic intensity is FLOP / bytes moved, and the ridge point is the
    hardware's peak_flops / bandwidth. Where the two meet is the boundary
    between the compute bottleneck and the bandwidth bottleneck. **It is a
    shape-level value, so it can be used to branch, as in
    `if p.roofline_ratio < 1.0:`.**
    Source: Williams, Waterman, Patterson (2009), "Roofline"
    """
    eb = p.bytes_per_element
    ai = 2.0 * p.M * p.N * p.K / max(
        1.0, eb * (p.M * p.K + p.K * p.N + p.M * p.N))
    return ai / hw.ridge_point


def _v_edge(df):
    gm = np.ceil(df["M"].to_numpy(np.float64)
                 / df["tile_m"].to_numpy(np.float64))
    gn = np.ceil(df["N"].to_numpy(np.float64)
                 / df["tile_n"].to_numpy(np.float64))
    return (gm * df["tile_m"].to_numpy(np.float64) / df["M"].to_numpy(np.float64)
            ) * (gn * df["tile_n"].to_numpy(np.float64)
                 / df["N"].to_numpy(np.float64)) - 1.0


@_feature(registry=KNOWN7, expected_range=(0.0, 300.0),
          direction="higher_is_worse", vec=lambda df, hw, p: _v_edge(df))
def edge_waste(p: Problem, hw: Hardware, cfg: Config) -> float:
    """The multiple of work thrown away where a tile crosses the shape
    boundary, minus 1. Larger is worse.

    A tile computes everything it covers even outside the shape. With a
    128-row tile on M=1, 99.2% of the work is thrown away (value 127). The
    range is wide, so give it a correspondingly small weight or compress it
    with a log.
    Source: CUTLASS documentation, predication
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    return (gm * cfg.tile_m / p.M) * (gn * cfg.tile_n / p.N) - 1.0


@_feature(registry=KNOWN7, expected_range=(0.0, 1.0),
          direction="higher_is_worse",
          vec=lambda df, hw, p: (df["spill_bytes"].to_numpy(np.float64) > 0
                                 ).astype(np.float64))
def has_spill(p: Problem, hw: Hardware, cfg: Config) -> float:
    """Is there a register spill? 0 or 1.

    When registers exceed the SM limit they are pushed out to local memory.
    Local is physically DRAM, and it is accessed on every iteration inside
    the mainloop. If a register access is one cycle, local is hundreds.
    Source: CUDA C++ Best Practices Guide, "Register Pressure"
    """
    return 1.0 if cfg.spill_bytes > 0 else 0.0


@_shape_feature(registry=KNOWN7, expected_range=(0.0, 30.0),
                direction="neutral")
def log_min_dim(p: Problem, hw: Hardware, cfg: Config) -> float:
    """log2 of the shortest of M, N, K. It separates skinny shapes from
    square ones.

    A GEMM is given as three dimensions, and when one of them is small the
    problem is a different shape of problem — a tile that is wider than the
    dimension it covers throws work away no matter what else is chosen. **It
    is shape level, so it can be used to branch, as in
    `if p.log_min_dim < 4.0:`.**
    Source: the GEMM problem statement — M, N, K are the inputs
    """
    return math.log2(max(1.0, float(min(p.M, p.N, p.K))))


@_shape_feature(registry=KNOWN7, expected_range=(0.0, 80.0),
                direction="neutral")
def log_flops(p: Problem, hw: Hardware, cfg: Config) -> float:
    """log2(2·M·N·K) — the absolute size of the problem.

    2MNK is the flop count in the definition of GEMM. It says nothing about
    which config is good; it says **how large the problem is**, which is the
    axis along which fixed overheads stop mattering. **Shape level, so it
    can be used to branch.**
    Source: the definition of GEMM's flop count, 2MNK
    """
    return math.log2(max(1.0, 2.0 * float(p.M) * float(p.N) * float(p.K)))


# ★ The descriptions and ranges are attached for the prompt. Those of
#   `physical.py` are not used as they are — table observations are mixed in
#   there (§12.3).
for _n, _text in {
    "tail_waste": ("The fraction of SM slots idle in the last wave. The "
                   "loss is roughly 1/(1-x)x — 2x at 0.5, 5x at 0.8. Used "
                   "linearly, that magnitude does not come out"),
    "occupancy_deficit": ("The fraction of thread slots per SM left "
                          "unfilled. There is less room to hide memory "
                          "latency. It matters less when compute-bound"),
    "roofline_ratio": ("Arithmetic intensity / ridge point. Below 1 it is "
                       "memory-bound. It is shape level, so it can be used "
                       "to branch, and it handles the boundary more smoothly "
                       "than a binary verdict"),
    "edge_waste": ("The multiple of work thrown away where a tile crosses "
                   "the shape boundary, minus 1. With a 128-row tile on M=1 "
                   "the value is 127. The range is wide (hundreds), so give "
                   "it a small weight or compress it with a log"),
    "has_spill": ("Registers overflow into local memory (= DRAM). If a "
                  "register access is one cycle, local is hundreds, and it "
                  "happens on every iteration inside the mainloop"),
    "log_min_dim": ("log2 of the shortest of M, N, K. It is shape level, so "
                    "it can be used to branch. A dimension shorter than the "
                    "tile that covers it throws work away whatever else is "
                    "chosen"),
    "log_flops": ("log2(2MNK), the absolute size of the problem. It is "
                  "shape level, so it can be used to branch. It is the axis "
                  "along which fixed overheads stop mattering"),
}.items():
    KNOWN7.annotate(_n, physical_meaning=f"{_text}. Source: {source_of(_n)}")
