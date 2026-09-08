"""★ 하드웨어 사실 프롬프트를 **번들에서 만든다** (D-113).

## 왜 파일로 두면 안 되나

`hw/` 에 손으로 쓴 `sm_86.md` 하나뿐이었고 `LLMConfig.arch_prompt` 의
기본값이 그것으로 **고정**돼 있었다. `f1_pipeline` 은 그 값을 바꾸지
않았다. 그래서 5090 표로 RuleWriter 를 돌린 §29.5 (c) 재생성이
**A6000 하드웨어 사실을 받았다.**

```
RuleWriter 가 받은 것   A6000 / SM 84 / L2 6 MB / ridge 159.1 / 눈금 1.024us
실제 5090              SM 170 / L2 96 MB / ridge 117.9 / 눈금 0.016us
```

조건이 코드에 상수로 박혀 있었고, 그 상수가 조건이라는 것을 아무도 안
봤다. 원칙 2 의 또 다른 형태다.

## 무엇이 arch 무관이고 무엇이 번들에서 오나

```
실행 모델 절      arch 무관 — CTA 배분 / 타일 경계 / split-K / stages
숫자             전부 env.json 에서
측정 한계 절      ★ 눈금과 "커널 길이별 몇 %" 표를 tick_ms 로 **계산**한다
```

`hw/sm_86.md` 는 **지우지 않고 둔다** — 2026-09-03 이전 실행의 조건이
그 파일이고, 지우면 그 실행들을 되짚을 수 없다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kernelrule.core.types import Hardware, hardware_from_env

__all__ = ["render_hw_prompt", "hw_prompt_from_bundle", "check_hw_prompt",
           "HwPromptError"]


class HwPromptError(ValueError):
    """번들과 프롬프트가 달라졌다. **조용히 진행하지 않는다** (§26.4)."""


#: 눈금 표에 쓸 **참고** 길이 [ms].
#:
#: 판정은 이 목록이 아니라 `min_ms` — **그 표의 실제 최소 best_ms** — 에서
#: 한다 (D-117). 14us 가 관측 하한인 것은 A6000 뿐이고, 표에 없는 길이에서
#: 판정하면 그것은 표와 무관한 기준이다 (원칙 2).
_TICK_ROWS = (0.5, 1.3)


def _fmt_bytes(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f} MB"
    return f"{n / (1 << 10):.0f} KB"


def render_hw_prompt(hw: Hardware, *, noise, env: dict,
                     min_ms: float) -> str:
    """`Hardware` + 노이즈 모델 -> 프롬프트 본문. **손으로 쓰지 않는다.**

    ★ 측정 한계 절의 **결론이 표마다 다르다** (D-116). 노이즈 바닥은
    `max(통계항, 눈금항)` 인데 어느 쪽이 이기는지가 표마다 달라진다:

    ```
    A6000   11.3us 에서 눈금 9.09% vs 통계 3.36%   -> ★ 눈금이 한계다
    5090    28.7us 에서 눈금 0.06% vs 통계 0.10%   -> 눈금은 한계가 아니다
    ```

    ★ 판정하는 길이는 **그 표의 최소 `best_ms`** 다 (`min_ms`). 표에 없는
    길이에서 판정하면 표와 무관한 기준이 된다 (D-117).

    5090 에 A6000 의 결론("짧은 형상은 눈금 안에 묻힌다")을 그대로
    보내면 **틀린 경고**다. 판정은 `NoiseModel` 이 이미 들고 있는 두 항을
    비교해서 하고, 여기서 새 기준을 만들지 않는다 (원칙 2).
    """
    tick_ms = float(noise.tick_ms)
    if tick_ms <= 0:
        raise HwPromptError(f"tick_ms 가 {tick_ms} 다. 눈금 절을 못 만든다.")
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
            f"min_ms 가 {min_ms} 다. 이 표의 **가장 짧은 커널**에서 "
            "판정해야 한다 (D-117).")
    lens = (min_ms, *_TICK_ROWS)
    rows = "\n".join(
        f"  {ms * 1000:>6.1f} us 커널   눈금 {noise.tick_pct(ms):7.3%}"
        f"   통계 {noise.sigma(ms):7.3%}"
        + ("   ← 이 표의 최솟값" if ms == min_ms else "")
        for ms in lens)
    # ★ 어느 항이 한계인가 — **이 표의 가장 짧은 커널**에서 본다.
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
    """번들 경로 -> (프롬프트 본문, 사실 요약). **유일한 진입점**이다.

    ★ `table` 이나 `min_ms` 중 하나는 있어야 한다 (D-117) — 눈금 판정을
    **그 표의 가장 짧은 커널**에서 하기 때문이다. 기본값을 두지 않는다.
    """
    p = Path(bundle) / "env.json"
    if not p.exists():
        raise HwPromptError(
            f"{p} 가 없다. 하드웨어 사실을 만들 수 없으므로 진행하지 "
            "않는다 — 기본값으로 떨어지면 다른 GPU 의 사실이 간다 (D-113).")
    env = json.loads(p.read_text())
    hw = hardware_from_env(env)
    from kerneltab.core.bundle import load_bundle

    from kernelrule.core.noise import NoiseModel

    # ★ `PerfTable` 과 **같은 진입점**을 쓴다 (원칙 2). 여기서 따로 읽으면
    #   표가 쓰는 계수와 달라질 수 있다.
    b = load_bundle(str(bundle), verify=True)
    if env_hash and not str(b.env_hash).startswith(str(env_hash)):
        raise HwPromptError(
            f"env_hash 불일치. 요청 {env_hash!r}, 번들 "
            f"{str(b.env_hash)[:16]!r}")
    noise = NoiseModel.from_bundle(b)
    tick_ms = float(noise.tick_ms)
    if min_ms is None:
        if table is None:
            raise HwPromptError(
                "`table` 도 `min_ms` 도 없다. 눈금이 한계인지를 **그 표의 "
                "가장 짧은 커널**에서 판정해야 한다 (D-117).")
        min_ms = float(min(table.best_time(q) for q in table.shapes()))
    txt = render_hw_prompt(hw, noise=noise, env=env, min_ms=min_ms)
    return txt, {"name": hw.name, "arch": hw.arch, "sm_count": hw.sm_count,
                 "l2_bytes": hw.l2_bytes, "ridge_point": hw.ridge_point,
                 "tick_ms": tick_ms, "source": str(p),
                 # ★ 조건이므로 남긴다 — 측정 한계 절의 **결론**이 달라진다.
                 "min_ms": float(min_ms),
                 "tick_binds": bool(noise.tick_pct(min_ms)
                                    > noise.sigma(min_ms)),
                 "tick_pct_at_min": float(noise.tick_pct(min_ms)),
                 "sigma_at_min": float(noise.sigma(min_ms))}


def check_hw_prompt(text: str, hw: Hardware, tick_ms: float) -> None:
    """프롬프트가 **이 표의** 하드웨어를 말하는가. 아니면 예외 (§26.4).

    ★ 이름과 눈금 둘 다 본다. 이름만 보면 같은 GPU 의 다른 번들(다른
    타이머 눈금)이 통과한다.
    """
    if hw.name not in text:
        raise HwPromptError(
            f"하드웨어 프롬프트가 {hw.name!r} 를 말하지 않는다. "
            "다른 GPU 의 사실이 가고 있다 (D-113).")
    # ★ 2026-09-08 (D-146): 프롬프트가 영어가 됐다. **옛 한글 형태도 받는다**
    #   — `hw/sm_86.md` 는 얼린 파일이라 한글이고, 그것도 검사할 수 있어야 한다.
    m = (re.search(r"tick \(([\d.]+) us\)", text)
         or re.search(r"눈금\(([\d.]+) us\)", text))
    if not m:
        raise HwPromptError(
            "the hardware prompt has no tick section / "
            "하드웨어 프롬프트에 눈금 절이 없다.")
    got, want = float(m.group(1)), tick_ms * 1000
    if abs(got - want) > 0.5e-3 * max(1.0, want):
        raise HwPromptError(
            f"프롬프트의 눈금 {got} us 가 번들의 {want:.3f} us 와 다르다.")
