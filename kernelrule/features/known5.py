"""★ The starting library of condition F2 — **the five public facts**
(§30.17).

★ Renamed 2026-09-04: this condition's old name was `F1-K` (D-128). The
module name is unchanged — its content is "the 5 public facts", separate from
the condition's name.

## Why five are given

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
```

**That knowledge is there when porting to a new GPU too.** "Does the LLM
rediscover these on its own" has been answered (largely, it reproduces them),
and what we want to know is **"what does it build on top of what is
known"**.

## Which five — **ones of different shapes**

```
tail_waste          ratio      normalised to 0~1 with a physical upper bound
occupancy_deficit   ratio      uses a build-time value (max_blocks_per_sm)
roofline_ratio      threshold  ★ shape level (p.*). Read against 1
edge_waste          absolute   unbounded (0~300). Compression must be
                               considered
has_spill           binary     once on, the magnitude changes
```

`roofline_ratio` being shape-level is deliberate too — it shows that a rule
can branch with `if p.<x>:`.

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

__all__ = ["KNOWN5", "SOURCES", "source_of"]

#: The starting registry of F2. **Only five go in** — the other 19 are
#: condition F3.
KNOWN5 = FeatureRegistry("known5")

#: Feature -> public source. It is appended to the description in the
#: prompt.
SOURCES: dict[str, str] = {
    "tail_waste": 'CUDA C++ Best Practices Guide, "Thread and Block Heuristics"',
    "occupancy_deficit": "CUDA Occupancy Calculator",
    "roofline_ratio": 'Williams, Waterman, Patterson (2009), "Roofline"',
    "edge_waste": "CUTLASS documentation, predication",
    "has_spill": 'CUDA C++ Best Practices Guide, "Register Pressure"',
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


@_feature(registry=KNOWN5, expected_range=(0.0, 1.0),
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


@_feature(registry=KNOWN5, expected_range=(0.0, 1.0),
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


@_shape_feature(registry=KNOWN5, expected_range=(0.0, 1e4),
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


@_feature(registry=KNOWN5, expected_range=(0.0, 300.0),
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


@_feature(registry=KNOWN5, expected_range=(0.0, 1.0),
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
}.items():
    KNOWN5.annotate(_n, physical_meaning=f"{_text}. Source: {source_of(_n)}")
