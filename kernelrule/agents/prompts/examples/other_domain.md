## Shape examples — ★ from a different domain

The three below are from a **domain unrelated to this problem**.
Look only at how physics is turned into code, how the explanation is written,
and how magnitudes are matched. **Do not carry these concepts over to GEMM.**

```python
# (1) ratio form — normalised to 0~1, with a physical upper bound
def branch_divergence_cost(warp_size, active_lanes) -> float:
    """Cost of a divergent branch within a warp. Larger is worse.

    SIMT executes per warp, so a divergent branch runs both sides in
    sequence. The fewer active lanes, the more lanes idle.
    """
    return 1.0 - active_lanes / max(warp_size, 1)


# (2) absolute form — unbounded, so compress with a log
def queue_backlog(arrival_rate, service_rate) -> float:
    """How much the queue backs up. Larger is worse.

    If arrivals outpace service it grows without bound. It degrades sharply
    near saturation, so compress with a log to match the other terms.
    """
    rho = arrival_rate / max(service_rate, 1e-9)
    return math.log2(1.0 + rho / max(1.0 - min(rho, 0.999), 1e-9))


# (3) binary form — when switching it on changes the magnitude
def page_fault_present(working_set, ram_bytes) -> float:
    """Does the working set exceed physical memory? If so the magnitude
    changes.

    A disk access is tens of thousands of times slower than memory, so
    binary is enough.
    """
    return 1.0 if working_set > ram_bytes else 0.0
```

The difference between the three is the point — **normalised / log-compressed
/ binary.** Decide which form your physics has, then write the formula.
