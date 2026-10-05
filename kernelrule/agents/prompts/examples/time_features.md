<!-- D-193 — condition F4's FeatureWriter example. F4 drops the three axes
     known7.md shows, so the examples are time features that are in the F4
     library. Code copied verbatim from kernelrule/features/time_features.jsonl. -->
## Shape examples — ★ **these are already in the library**

The three below are already in "existing features" above. **Do not rebuild
them** — they are rejected as duplicates and only cost you parameters.

Look only at how physics is turned into code, and **how the three shapes
differ**.

```python
# (1) a time ratio — bytes over a bandwidth, divided by the ideal time
def tm_l2_ratio(p, hw, cfg) -> float:
    """L2->SM operand (+spill) traffic time at DRAM bandwidth over ideal compute time. Larger is worse."""
    gm, gn, sk = math.ceil(p.M / cfg.tile_m), math.ceil(p.N / cfg.tile_n), max(1, cfg.split_k)
    it = math.ceil(math.ceil(p.K / sk) / cfg.tile_k); ctas = gm * gn * sk
    moved = ctas * it * ((cfg.tile_m + cfg.tile_n) * cfg.tile_k * p.bytes_per_element + cfg.threads * cfg.spill_bytes * 2.0)
    t0 = 2.0 * p.M * p.N * p.K / (hw.peak_tflops_f16 * 1e12)
    return moved / (hw.bandwidth_gbps * 1e9) / t0


# (2) a time with two effects pulling in opposite directions — more splits
#     fill idle SMs (time falls), and pay the per-round prologue/epilogue and
#     the fix-up traffic more often (time rises). Lowest inside the range
def splitk_roofline_log_time(p, hw, cfg) -> float:
    """log2 of a split-K-aware critical-path time over the ideal 2MNK/peak. Busiest SM runs ceil(tiles*sk/SMs) CTAs of ceil(K/(tk*sk)) k-steps; each co-resident round exposes pipeline fill (stages-1) and the epilogue (tile write, plus read for serial slices) once; DRAM roof adds the serial fix-up round trips; parallel mode adds a reduction pass. Larger is worse."""
    s = max(1, cfg.split_k); ser = cfg.split_k_mode != "parallel" or s == 1; b = p.bytes_per_element; sm = hw.sm_count
    T = math.ceil(p.M / cfg.tile_m) * math.ceil(p.N / cfg.tile_n); n = math.ceil(T * s / sm); I = math.ceil(p.K / (cfg.tile_k * s))
    step_bytes = (cfg.tile_m + cfg.tile_n) * cfg.tile_k * b; tile_out = cfg.tile_m * cfg.tile_n * b
    o = (cfg.stages - 1) + ((2.0 - 1.0 / s) if ser else 1.0) * tile_out / step_bytes
    t_step = 2.0 * cfg.tile_m * cfg.tile_n * cfg.tile_k / (hw.peak_tflops_f16 * 1e12 / sm)
    t_sm = (n * I + math.ceil(n / max(1, cfg.max_blocks_per_sm)) * o) * t_step
    out = p.M * p.N * b; bw = hw.bandwidth_gbps * 1e9
    t_dram = ((p.M * p.K + p.K * p.N) * b + out + (2.0 * (s - 1) * out if ser else 0.0)) / bw
    t = max(t_sm, t_dram) + (0.0 if ser else (2.0 * s + 1.0) * out / bw)
    return math.log2(t / (2.0 * p.M * p.N * p.K / (hw.peak_tflops_f16 * 1e12)))


# (3) binary form — an effect the time estimate does not contain
def tm_regstaged(p, hw, cfg) -> float:
    """1 for the register-staged two-stage mainloop (one iteration of compute covers each global load), else 0."""
    return 1.0 if cfg.stages <= 2 else 0.0
```

The difference between the three is the point — **a time ratio / a time
with an interior minimum / a binary correction.** Decide which form your
physics has, then write the formula.

**Cite a source in `rationale` if you can.** For published physics you can;
if you cannot, it means you derived it yourself, so write the derivation.
