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
from pathlib import Path

from kernelrule.core.types import Hardware, hardware_from_env

__all__ = ["render_hw_prompt", "hw_prompt_from_bundle", "check_hw_prompt",
           "HwPromptError"]


class HwPromptError(ValueError):
    """The bundle and the prompt have diverged. **Do not proceed silently**
    (§26.4)."""


def _fmt_bytes(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f} MB"
    return f"{n / (1 << 10):.0f} KB"


def render_hw_prompt(hw: Hardware, *, noise=None, env: dict) -> str:
    """`Hardware` -> the prompt body. **Not written by hand.**

    ## ⚠️ 2026-09-11 (D-166): the measurement-limit section is gone

    It used to end with a tick table and a verdict ("a difference inside one
    tick may as well not exist — refining below it is learning noise"). Two
    reasons it left:

    ```
    1 ★ RuleWriter cannot refine. It writes once from a blank page with no
      parent, no score and no report. Telling it "do not refine below the
      tick" addresses an ability it does not have. This repository already
      keeps the objective block out of RuleWriter for exactly that reason
      (`openai_client.py`, §30.10)
    2 ★ it carried an answer-derived number. The verdict was read at
      `min(table.best_time(q) for q in table.shapes())` — a **measured**
      minimum over **every** shape, holdout included. On the H100 that
      minimum is a holdout shape and differs from the training minimum
      (42.848us vs 43.296us)
    ```

    ⛔ Narrowing `min_ms` to the training split was considered and rejected:
    it launders the answer-derived number instead of answering why it is
    there at all (D-166 §E).

    The noise and tick information is **not gone from the system** — it goes
    to the Analyst and the RuleEditor through the diagnostic report, which is
    where a role that can refine sees it. D-113 · D-116 · D-117 keep their
    records; their tests moved to that renderer.

    `noise` is still accepted so callers do not break, and it is unused.
    """
    _ = noise
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

"""


def hw_prompt_from_bundle(bundle: str | Path, *, env_hash: str | None = None
                          ) -> tuple[str, dict]:
    """Bundle path -> (prompt body, facts summary). **The only entry
    point.**

    ⚠️ 2026-09-11 (D-166): `table` and `min_ms` are gone. They existed for
    the measurement-limit section's verdict, that section is gone, and an
    argument nothing reads is the shape of leftover this clean-up is about
    (the same as `vendor.extract`). The caller in `f1_pipeline` no longer
    passes a table.
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
    txt = render_hw_prompt(hw, noise=noise, env=env)
    # ★ `tick_ms` stays in the facts — it is a property of this bundle and
    #   the trace records it. What left with the section is the **verdict**
    #   made from the answer (`min_ms` · `tick_binds` · the two percentages).
    return txt, {"name": hw.name, "arch": hw.arch, "sm_count": hw.sm_count,
                 "l2_bytes": hw.l2_bytes, "ridge_point": hw.ridge_point,
                 "tick_ms": tick_ms, "source": str(p)}


def check_hw_prompt(text: str, hw: Hardware, tick_ms: float | None = None
                    ) -> None:
    """Does the prompt speak of **this table's** hardware? If not, raise
    (§26.4).

    ★ **Two legs.** The name alone lets another bundle of the same GPU
    through, and two bundles of one GPU differ exactly where it matters.

    ⚠️ 2026-09-11 (D-166): the second leg used to be the timer tick, and the
    tick section left with the measurement-limit section (§E). It is now the
    **effective numbers** — which is closer to what D-113 was defending: the
    prompt's claim is `peak_tflops_f16` / `bandwidth_gbps` / `ridge_point`,
    and those differ between two bundles of the same card whose clocks were
    locked differently. Two of the three must be present and match.

    ⛔ Do not reduce this to one leg. `tick_ms` is still accepted so callers
    do not break, and it is no longer read.
    """
    _ = tick_ms
    if hw.name not in text:
        raise HwPromptError(
            f"the hardware prompt does not speak of {hw.name!r}. "
            "Another GPU's facts are going out (D-113).")
    want = {"peak_tflops_f16": f"{hw.peak_tflops_f16:.1f}",
            "bandwidth_gbps": f"{hw.bandwidth_gbps:.1f}",
            "ridge_point": f"{hw.ridge_point:.1f}"}
    missing = [k for k, v in want.items() if v not in text]
    if len(want) - len(missing) < 2:
        raise HwPromptError(
            f"the hardware prompt does not carry this bundle's effective "
            f"numbers ({', '.join(f'{k}={v}' for k, v in want.items())}); "
            f"absent: {missing}. Another bundle's facts are going out "
            f"(D-113 · D-166).")
