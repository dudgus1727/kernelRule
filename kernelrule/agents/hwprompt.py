"""★ The hardware-facts prompt is **built from the bundle** (D-113).

## Why it must not be a file

`hw/` held a single hand-written `sm_86.md`, and the default of
`LLMConfig.arch_prompt` was **pinned** to it. `f1_pipeline` did not change
that value. So the §29.5 (c) regeneration, which ran RuleWriter on the 5090
table, **received the A6000 hardware facts.**

```
what RuleWriter got   A6000 / SM 84 / L2 6 MB / ridge 159.1 / tick 1.024us
the actual 5090       SM 170 / L2 96 MB / ridge 117.9 / tick 0.016us
```

The condition was nailed into the code as a constant, and nobody looked at
that constant as a condition. Another form of principle 2.

## What is arch-independent and what comes from the bundle

```
execution-model section  arch-independent — CTA distribution / tile
                         boundaries / split-K / stages
numbers                  all from env.json
measurement-limit section  ★ the tick and the "what % per kernel length"
                           table are **computed** from tick_ms
```

`hw/sm_86.md` was deleted on 2026-09-08 (D-146). It had been dead since D-113
— nothing generates from it — and keeping it forced an exception into every
prompt test. The old runs' condition is their `config.json`, and the file
itself is in the history (`ee53b4d`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kernelrule.core.types import Hardware, hardware_from_env

__all__ = ["render_hw_prompt", "hw_prompt_from_bundle", "check_hw_prompt",
           "HwPromptError"]


class HwPromptError(ValueError):
    """The bundle and the prompt have diverged. **Do not proceed silently**
    (§26.4)."""


#: **Reference** lengths [ms] for the tick table.
#:
#: The verdict is made not from this list but at `min_ms` — **that table's
#: actual minimum best_ms** (D-117). 14us being the observed floor is true
#: of the A6000 only, and judging at a length that is not in the table is a
#: criterion unrelated to the table (principle 2).
_TICK_ROWS = (0.5, 1.3)


def _fmt_bytes(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f} MB"
    return f"{n / (1 << 10):.0f} KB"


def render_hw_prompt(hw: Hardware, *, noise, env: dict,
                     min_ms: float) -> str:
    """`Hardware` + noise model -> the prompt body. **Not written by
    hand.**

    ★ The **conclusion of the measurement-limit section differs per table**
    (D-116). The noise floor is `max(statistical term, tick term)`, and
    which one wins differs per table:

    ```
    A6000   at 11.3us  tick 9.09% vs statistical 3.36%  -> ★ the tick is the limit
    5090    at 28.7us  tick 0.06% vs statistical 0.10%  -> the tick is not the limit
    ```

    ★ The length at which the verdict is made is **that table's minimum
    `best_ms`** (`min_ms`). Judging at a length that is not in the table
    makes it a criterion unrelated to the table (D-117).

    Sending the A6000's conclusion ("short shapes are buried inside the
    tick") to the 5090 as is is a **wrong warning**. The verdict is made by
    comparing the two terms `NoiseModel` already holds; no new criterion is
    invented here (principle 2).
    """
    tick_ms = float(noise.tick_ms)
    if tick_ms <= 0:
        raise HwPromptError(
            f"tick_ms is {tick_ms}. The tick section cannot be built.")
    spec_t = env.get("peak_tflops_f16_spec")
    spec_b = env.get("bandwidth_gbps_spec")
    sm_mhz = env.get("locked_mhz") or env.get("sm_clock_mhz")
    mem_mhz = env.get("mem_clock_used_mhz") or env.get("mem_clock_mhz")
    at_sm = f" @{sm_mhz:.0f}MHz" if sm_mhz else ""
    at_mem = f" @{mem_mhz:.0f}MHz" if mem_mhz else ""
    spec_line = ""
    if spec_t and spec_b:
        spec_line = (f"**These are effective values.** Not the spec "
                     f"({spec_t:.1f} TFLOP/s, {spec_b:.0f} GB/s) but "
                     "measured with the clocks locked.\nWithout this "
                     "correction the memory-bound verdict is wrong.\n")
    if not (min_ms and min_ms > 0):
        raise HwPromptError(
            f"min_ms is {min_ms}. The verdict must be made at **this "
            "table's shortest kernel** (D-117).")
    lens = (min_ms, *_TICK_ROWS)
    rows = "\n".join(
        f"  {ms * 1000:>6.1f} us kernel   tick {noise.tick_pct(ms):7.3%}"
        f"   statistical {noise.sigma(ms):7.3%}"
        + ("   <- this table's minimum" if ms == min_ms else "")
        for ms in lens)
    # ★ Which term is the limit — read at **this table's shortest
    #   kernel**.
    tick_binds = noise.tick_pct(min_ms) > noise.sigma(min_ms)
    limit_note = (
        """**A difference inside one tick may as well not exist.** Refining
the rule below that is learning noise. It is better to block the paths that
**lose for certain**."""
        if tick_binds else
        """★ **On this table the tick is not the limit.** As the table above
shows, the statistical term is larger at every length — so what sets the
limit is not the timer but **the spread that repetition averages down**.
Short shapes are not especially hard to get right.

Do not aim below the noise floor (the larger of the two). That is still
learning noise.""")
    return f"""# Hardware facts — do not recall from memory, use what follows

This project got it wrong from memory four times (the meaning of ScaleType,
split-K constraints, swizzle compatibility, the serial partial type). This is
the target hardware.

```
GPU        {hw.name} ({hw.arch})
SMs        {hw.sm_count}
smem       {_fmt_bytes(hw.smem_per_block)} per block ({hw.smem_per_block:,} B)
threads    up to {hw.max_threads_per_sm:,} per SM
registers  {hw.regs_per_sm:,} per SM
L2         {_fmt_bytes(hw.l2_bytes)}
effective  {hw.peak_tflops_f16:.1f} TFLOP/s{at_sm}   \
{hw.bandwidth_gbps:.1f} GB/s{at_mem}
ridge      {hw.ridge_point:.1f} FLOP/byte
```

{spec_line}
## Execution model

```
CTAs are distributed across SMs, and on the last wave some SMs idle.

A tile computes **everything it covers**, even outside the shape.
  A 128-row tile on M=1 wastes 99.2% of the work.

split-K divides K to create more tiles, at the cost of a reduction.
  serial    round-trips D in fp16 per partition (precision loss)
  parallel  writes M*N*sk partials to DRAM and reads them back

stages=2 (MmaPipelined) and stages>=3 (multistage) are **different kernel
families**. If alignment does not reach 16 bytes, cp.async is unavailable and
only 2 stages are possible.
```

## Limits of measurement — this affects your judgement

```
Time is only recorded in units of the CUDA event timer's tick ({tick_ms * 1000:.3f} us).
Differences smaller than that **cannot be distinguished by measurement.**

The noise floor is the **larger** of two terms:
  tick        tick/t                    repetition does not shrink it
                                        (a resolution limit)
  statistical sigma_abs/t + sigma_rel   repetition averages it down

{rows}
```

**This is a property of the hardware and the timer.** It appears in the same
form on any GPU — but **which term wins differs per table.**

{limit_note}
"""


def hw_prompt_from_bundle(bundle: str | Path, *, env_hash: str | None = None,
                          table=None, min_ms: float | None = None
                          ) -> tuple[str, dict]:
    """Bundle path -> (prompt body, facts summary). **The only entry
    point.**

    ★ One of `table` or `min_ms` must be given (D-117) — because the tick
    verdict is made at **that table's shortest kernel**. There is no
    default.
    """
    p = Path(bundle) / "env.json"
    if not p.exists():
        raise HwPromptError(
            f"{p} is missing. The hardware facts cannot be built, so this "
            "does not proceed — falling back to a default sends another "
            "GPU's facts (D-113).")
    env = json.loads(p.read_text())
    hw = hardware_from_env(env)
    from kerneltab.core.bundle import load_bundle

    from kernelrule.core.noise import NoiseModel

    # ★ Uses **the same entry point** as `PerfTable` (principle 2).
    #   Reading it separately here could diverge from the coefficients the
    #   table uses.
    b = load_bundle(str(bundle), verify=True)
    if env_hash and not str(b.env_hash).startswith(str(env_hash)):
        raise HwPromptError(
            f"env_hash mismatch. requested {env_hash!r}, bundle "
            f"{str(b.env_hash)[:16]!r}")
    noise = NoiseModel.from_bundle(b)
    tick_ms = float(noise.tick_ms)
    if min_ms is None:
        if table is None:
            raise HwPromptError(
                "neither `table` nor `min_ms` is given. Whether the tick "
                "is the limit must be judged at **that table's shortest "
                "kernel** (D-117).")
        min_ms = float(min(table.best_time(q) for q in table.shapes()))
    txt = render_hw_prompt(hw, noise=noise, env=env, min_ms=min_ms)
    return txt, {"name": hw.name, "arch": hw.arch, "sm_count": hw.sm_count,
                 "l2_bytes": hw.l2_bytes, "ridge_point": hw.ridge_point,
                 "tick_ms": tick_ms, "source": str(p),
                 # ★ Recorded because it is a condition — the
                 #   **conclusion** of the measurement-limit section changes
                 #   with it.
                 "min_ms": float(min_ms),
                 "tick_binds": bool(noise.tick_pct(min_ms)
                                    > noise.sigma(min_ms)),
                 "tick_pct_at_min": float(noise.tick_pct(min_ms)),
                 "sigma_at_min": float(noise.sigma(min_ms))}


def check_hw_prompt(text: str, hw: Hardware, tick_ms: float) -> None:
    """Does the prompt speak of **this table's** hardware? If not, raise
    (§26.4).

    ★ It checks both the name and the tick. Checking only the name lets
    another bundle of the same GPU (with a different timer tick) pass.
    """
    if hw.name not in text:
        raise HwPromptError(
            f"the hardware prompt does not speak of {hw.name!r}. "
            "Another GPU's facts are going out (D-113).")
    m = re.search(r"tick \(([\d.]+) us\)", text)
    if not m:
        raise HwPromptError("the hardware prompt has no tick section.")
    got, want = float(m.group(1)), tick_ms * 1000
    if abs(got - want) > 0.5e-3 * max(1.0, want):
        raise HwPromptError(
            f"the prompt's tick {got} us differs from the bundle's "
            f"{want:.3f} us.")
