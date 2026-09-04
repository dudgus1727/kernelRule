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
        spec_line = (f"**실효값입니다.** 스펙({spec_t:.1f} TFLOP/s, "
                     f"{spec_b:.0f} GB/s)이 아니라 클럭을 고정한 상태의 "
                     "관측값입니다.\n이 보정이 없으면 memory-bound 판정이 "
                     "어긋납니다.\n")
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
        """**눈금 안의 차이는 존재하지 않는 것과 같습니다.** 그 아래를 겨냥해
규칙을 정교하게 만드는 것은 노이즈를 배우는 일입니다. 그보다 **확실히
지는 경로**를 막는 편이 낫습니다."""
        if tick_binds else
        """★ **이 표에서는 눈금이 한계가 아닙니다.** 위 표에서 보듯 어느
길이에서도 통계 항이 더 큽니다 — 즉 한계를 정하는 것은 타이머가 아니라
**반복 측정으로 줄어드는 산포**입니다. 짧은 형상이라고 해서 특별히 못
맞추는 것이 아닙니다.

노이즈 바닥(둘 중 큰 쪽) 아래를 겨냥하지 마세요. 그것은 여전히
노이즈를 배우는 일입니다.""")
    return f"""# 하드웨어 사실 — 기억에서 꺼내지 말고 아래를 쓰세요

이 프로젝트에서 기억에 의존해 네 번 틀렸습니다 (ScaleType 의미, split-K
제약, swizzle 호환성, serial 부분합 타입). 아래가 대상 하드웨어입니다.

```
GPU        {hw.name} ({hw.arch})
SM         {hw.sm_count}개
smem       블록당 {_fmt_bytes(hw.smem_per_block)} ({hw.smem_per_block:,} B)
스레드      SM 당 최대 {hw.max_threads_per_sm:,}
레지스터    SM 당 {hw.regs_per_sm:,}
L2         {_fmt_bytes(hw.l2_bytes)}
실효 성능   {hw.peak_tflops_f16:.1f} TFLOP/s{at_sm}   \
{hw.bandwidth_gbps:.1f} GB/s{at_mem}
ridge      {hw.ridge_point:.1f} FLOP/byte
```

{spec_line}
## 실행 모델

```
CTA 가 SM 에 배분되고, 마지막 wave 에서 SM 일부가 논다.

타일은 형상 경계를 넘어도 그 부분을 **전부 계산한다.**
  M=1 에 128행 타일이면 일의 99.2% 가 버려진다.

split-K 는 K 를 나눠 타일 수를 늘리되 리덕션 비용이 붙는다.
  serial   파티션마다 fp16 으로 D 를 왕복한다 (정밀도 손실)
  parallel 부분합 M*N*sk 개를 DRAM 에 쓰고 다시 읽는다

stages=2 (MmaPipelined) 와 stages>=3 (multistage) 는 **다른 커널 계열**이다.
alignment 가 16바이트를 못 맞추면 cp.async 를 못 써서 2단만 가능하다.
```

## 측정의 한계 — 이것이 판단에 영향을 줍니다

```
시간은 CUDA 이벤트 타이머의 눈금({tick_ms * 1000:.3f} us) 단위로만 기록된다.
그보다 작은 차이는 **측정으로 구분할 수 없다.**

노이즈 바닥은 두 항 중 **큰 쪽**이다:
  눈금   tick/t          반복 측정해도 안 줄어든다 (분해 한계)
  통계   sigma_abs/t + sigma_rel   반복하면 평균으로 줄어든다

{rows}
```

**이것은 하드웨어와 타이머의 성질입니다.** 어느 GPU 로 가도
같은 형태로 나타납니다 — 다만 **어느 항이 이기는지는 표마다 다릅니다.**

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
    m = re.search(r"눈금\(([\d.]+) us\)", text)
    if not m:
        raise HwPromptError("하드웨어 프롬프트에 눈금 절이 없다.")
    got, want = float(m.group(1)), tick_ms * 1000
    if abs(got - want) > 0.5e-3 * max(1.0, want):
        raise HwPromptError(
            f"프롬프트의 눈금 {got} us 가 번들의 {want:.3f} us 와 다르다.")
