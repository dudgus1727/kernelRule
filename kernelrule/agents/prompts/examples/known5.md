## Shape examples — ★ **these are already in the library**

The three below are already in "existing features" above. **Do not rebuild
them** — they are rejected as duplicates and only cost you parameters.

Look only at how physics is turned into code, how the source is cited, and
**how the three shapes differ**.

```python
# (1) ratio form — normalised to 0~1, with a physical upper bound
def tail_waste(p, hw, cfg) -> float:
    """Fraction of SM slots idle on the last wave. Larger is worse.

    CTAs are distributed across SMs, and on the last batch some SMs idle.
    Source: CUDA C++ Best Practices Guide, "Thread and Block Heuristics"
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    tiles = gm * gn * max(1, cfg.split_k)
    w = max(1e-12, tiles / (hw.sm_count * max(1, cfg.max_blocks_per_sm)))
    full = math.ceil(w)
    return (full - w) / full


# (2) absolute form — unbounded. May need compression
def edge_waste(p, hw, cfg) -> float:
    """Work multiplier from tiles overhanging the shape, minus 1.
    Larger is worse.

    A tile computes everything it covers even outside the shape.
    A 128-row tile on M=1 wastes 99.2% of the work (value 127).
    Source: CUTLASS documentation, predication
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    return (gm * cfg.tile_m / p.M) * (gn * cfg.tile_n / p.N) - 1.0


# (3) binary form — when switching it on changes the magnitude
def has_spill(p, hw, cfg) -> float:
    """Is there register spilling? 0 or 1.

    When registers exceed the SM limit they spill to local memory. Local is
    physically DRAM and is touched every mainloop iteration. A register
    access is one cycle; local is hundreds.
    Source: CUDA C++ Best Practices Guide, "Register Pressure"
    """
    return 1.0 if cfg.spill_bytes > 0 else 0.0
```

The difference between the three is the point — **ratio / absolute / binary.**
Decide which form your physics has, then write the formula.

**Cite a source in `rationale` if you can.** For published physics you can;
if you cannot, it means you derived it yourself, so write the derivation.
