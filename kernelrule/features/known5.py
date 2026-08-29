"""★ F1-K 조건의 시작 라이브러리 — **공개 지식 다섯** (§30.17).

## 왜 다섯을 주나

F1(0개 시작)에서 예산의 절반 가까이가 **이미 알려진 물리를 다시 만드는
데** 쓰였다 — 21개 중 엄격 재발견 6 + 단조 3 (D-63).

그런데 그것들은 **이 표 없이도 아는 것들이다.**

```
wave quantization      CUDA C++ Best Practices Guide
occupancy 계산          CUDA Occupancy Calculator
arithmetic intensity   Williams et al. (2009) roofline
타일 경계 낭비          CUTLASS 문서, predication
레지스터 스필           CUDA C++ Best Practices, Register Pressure
```

**새 GPU 로 이식할 때도 그 지식은 있다.** "LLM 이 이것들을 스스로
재발견하는가" 는 답이 나왔고(대체로 재현한다), 우리가 궁금한 것은
**"알려진 것 위에 무엇을 더 만드는가"** 다.

## 어느 다섯인가 — **형태가 서로 다른 것**

```
tail_waste          비율형   0~1 로 정규화되고 물리적 상한이 있다
occupancy_deficit   비율형   빌드 시점 값(max_blocks_per_sm)을 쓴다
roofline_ratio      문턱형   ★ 형상 수준(p.*). 1 을 경계로 읽는다
edge_waste          절대량형 상한이 없다 (0~300). 압축을 고려해야 한다
has_spill           이진형   켜지면 자릿수가 달라진다
```

`roofline_ratio` 가 형상 수준인 것도 의도다 — 규칙이 `if p.<x>:` 로
분기할 수 있다는 것을 보여준다.

## ★ docstring 은 정리본이다 — 원본이 아니다

`physical.py` 의 docstring 에는 **이 표에서 나온 측정 결과**가 섞여
있다. 그대로 프롬프트에 내보내면 §12.3 위반이다.

```
지금 (physical.py)
    ★ 이 표에서 스필 커널은 최적으로 뽑힌 적이 0회이고 rel 중앙값이
    13.6(최대 37.2)이다. 전체 행의 7.4% 이며 ...
    -> 세 숫자 전부 측정 결과다. 표를 봐야 안다
```

판단 기준:

```
표 없이 알 수 있는가?
  예   -> 물리. 남긴다      "128행 타일에 M=1 이면 99.2% 낭비" (식에서 나온다)
  아니오 -> 측정. 뺀다       "스필 커널의 rel 중앙값 13.6배"
```

**출처를 각 피처에 적는다.** 이유가 둘이다 — (1) "공개 지식인가" 를
나중에 검증할 수 있다. 출처를 못 적는 설명이 있으면 그것은 이 표에서
나온 것이다. (2) 출처를 적는 습관 자체가 LLM 에게 전달된다.

⚠️ **`physical.py` 를 고치지 않는다.** 그쪽 docstring 은 사람이 읽는
개발 문서이고 표 관측이 거기 있는 것은 정상이다. **프롬프트로 나가는
경로만** 정리본을 쓴다.
"""

from __future__ import annotations

import math

import numpy as np

from kernelrule.core.types import Config, Hardware, Problem
from kernelrule.features import FeatureRegistry
from kernelrule.features import feature as _feature
from kernelrule.features import shape_feature as _shape_feature

__all__ = ["KNOWN5", "SOURCES", "source_of"]

#: F1-K 의 시작 레지스트리. **다섯만 들어간다** — 나머지 19개는 F3 조건이다.
KNOWN5 = FeatureRegistry("known5")

#: 피처 -> 공개 출처. 프롬프트의 설명 끝에 붙는다.
SOURCES: dict[str, str] = {
    "tail_waste": 'CUDA C++ Best Practices Guide, "Thread and Block Heuristics"',
    "occupancy_deficit": "CUDA Occupancy Calculator",
    "roofline_ratio": 'Williams, Waterman, Patterson (2009), "Roofline"',
    "edge_waste": "CUTLASS 문서, predication",
    "has_spill": 'CUDA C++ Best Practices Guide, "Register Pressure"',
}


def source_of(name: str) -> str:
    """출처. 없으면 예외 — **출처를 못 대는 피처는 여기 못 들어온다.**"""
    if name not in SOURCES:
        raise KeyError(
            f"{name!r} 의 공개 출처가 없다. F1-K 의 시작 라이브러리는 "
            "'이 표 없이도 아는 것' 만 담는다 (§30.17).")
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
    """마지막 wave 에서 노는 SM 슬롯의 비율. 0~1, 클수록 나쁘다.

    CTA 가 SM 에 나뉘어 배분되는데, 마지막 묶음에서 일부 SM 이 논다.
    waves 가 크면 자연히 0 에 가까워진다.
    출처: CUDA C++ Best Practices Guide, "Thread and Block Heuristics"
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
    """SM 당 스레드 슬롯 중 채우지 못하는 비율. 0 이면 만점.

    슬롯이 비면 메모리 지연을 다른 워프로 가릴 여지가 줄어든다.
    `max_blocks_per_sm` 은 컴파일 시점에 정해지는 값이라 써도 된다.
    출처: CUDA Occupancy Calculator
    """
    used = cfg.max_blocks_per_sm * cfg.threads / hw.max_threads_per_sm
    return 1.0 - min(1.0, max(0.0, used))


@_shape_feature(registry=KNOWN5, expected_range=(0.0, 1e4),
                direction="neutral")
def roofline_ratio(p: Problem, hw: Hardware, cfg: Config) -> float:
    """산술강도 / ridge point. 1 미만이면 메모리 바운드다.

    산술강도는 FLOP / 이동 바이트이고, ridge point 는 하드웨어의
    peak_flops / bandwidth 다. 둘이 같아지는 지점이 계산 병목과 대역폭
    병목의 경계다. **형상 수준 값이라 `if p.roofline_ratio < 1.0:` 처럼
    분기에 쓸 수 있다.**
    출처: Williams, Waterman, Patterson (2009), "Roofline"
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
    """타일이 형상 경계를 넘어 버려지는 일의 배수 - 1. 클수록 나쁘다.

    타일은 형상 밖으로 튀어나가도 그 부분을 전부 계산한다. M=1 에 128행
    타일이면 일의 99.2% 가 버려진다 (값 127). 범위가 넓으니 가중치를
    그만큼 작게 주거나 로그로 압축해서 쓴다.
    출처: CUTLASS 문서, predication
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    return (gm * cfg.tile_m / p.M) * (gn * cfg.tile_n / p.N) - 1.0


@_feature(registry=KNOWN5, expected_range=(0.0, 1.0),
          direction="higher_is_worse",
          vec=lambda df, hw, p: (df["spill_bytes"].to_numpy(np.float64) > 0
                                 ).astype(np.float64))
def has_spill(p: Problem, hw: Hardware, cfg: Config) -> float:
    """레지스터 스필이 있는가. 0 또는 1.

    레지스터가 SM 한계를 넘으면 로컬 메모리로 밀려난다. 로컬은 물리적으로
    DRAM 이고, mainloop 안에서 매 반복 접근한다. 레지스터 접근이
    1사이클이면 로컬은 수백 사이클이다.
    출처: CUDA C++ Best Practices Guide, "Register Pressure"
    """
    return 1.0 if cfg.spill_bytes > 0 else 0.0


# ★ 설명과 범위를 프롬프트용으로 붙인다. `physical.py` 의 것을 그대로
#   쓰지 않는다 — 거기에는 표 관측이 섞여 있다 (§12.3).
for _n, _text in {
    "tail_waste": ("마지막 wave 에서 노는 SM 슬롯의 비율. 손해는 대략 "
                   "1/(1-x) 배다 — 0.5 면 2배, 0.8 면 5배. 선형으로 쓰면 "
                   "그 크기가 안 나온다"),
    "occupancy_deficit": ("SM 당 스레드 슬롯 중 못 채우는 비율. 메모리 "
                          "지연을 가릴 여지가 줄어든다. 연산 바운드에서는 "
                          "덜 중요하다"),
    "roofline_ratio": ("산술강도 / ridge point. 1 미만이면 메모리 바운드. "
                       "형상 수준이라 분기에 쓸 수 있고, 이진 판정보다 "
                       "경계 근처를 부드럽게 다룬다"),
    "edge_waste": ("타일이 형상 경계를 넘어 버려지는 일의 배수 - 1. "
                   "M=1 에 128행 타일이면 값이 127 이다. 범위가 넓으니 "
                   "(수백) 가중치를 작게 주거나 로그로 압축해라"),
    "has_spill": ("레지스터가 넘쳐 로컬 메모리(=DRAM)로 나간다. 레지스터 "
                  "접근이 1사이클이면 로컬은 수백 사이클이고 mainloop "
                  "안에서 매 반복 일어난다"),
}.items():
    KNOWN5.annotate(_n, physical_meaning=f"{_text}. 출처: {source_of(_n)}")
