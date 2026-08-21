"""사람이 작성한 물리 피처 (§8.2). **판단하지 않는다.**

이 파일이 LLM 루프보다 먼저 있어야 하는 이유가 둘이다. (1) LLM 생성에
의존하면 초반에 아무것도 못 한다. (2) "LLM 이 추가한 피처가 기여했는가" 를
재는 기준선이 된다 (§16.1 ablation).

## 작성 규칙 — LLM 에게도 동일하게 강제한다

    순수 함수. (Problem, Hardware, Config) 만으로 계산
    float 하나. **클수록 나쁜 방향으로 통일** — 규칙이 항상 "가중합 후
                 오름차순 정렬" 이 되어 LLM 이 부호를 헷갈릴 여지가 없다
    하드웨어 상수는 hw.* 에서 읽기. 84 / 101376 하드코딩 금지
    cfg.ext 참조 금지 — 아키텍처 전이 전제 (§4.3)
    10줄 이내

## 벡터화 (`vec=`) 는 **같은 물리를 배열로** 쓴 것이다

스칼라 구현이 계약이고 벡터화는 속도다. 둘이 어긋나면 학습(행렬)과
배포(스칼라)가 다른 함수를 쓰게 되므로 `verify_vectorized()` 가 표본에서
대조하고 **불일치는 기각**이다 (§26.4).

## 이 표에서 확인된 사실 (§18.3) — 피처가 담아야 할 것

    스필 커널은 최적으로 뽑힌 적이 0회다 (rel 중앙 13.6, 최대 37.2)
    warp_m=128 은 최적 0회 — 스필과 같은 것을 가리킨다
    stages=2(MmaPipelined)는 stages>=3(multistage)과 다른 커널 계열이다
    alignment 1 형상은 cp.async 를 못 써서 stages=2 만 가능하다
    split_k_mode 는 66/66 이 serial. parallel 은 40만 줄을 쓰고 최적 0회
"""

from __future__ import annotations

import math

import numpy as np

from kernelrule.core.types import Config, Hardware, Problem
from kernelrule.features import REGISTRY, feature, shape_feature

__all__ = ["REGISTRY"]

_DTYPE_BYTES = {"f16": 2, "bf16": 2, "f8": 1, "f32": 4}


def _ebytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(str(dtype), 2)


def _v_ebytes(df) -> np.ndarray:
    return np.asarray([_DTYPE_BYTES.get(str(d), 2) for d in df["dtype"]],
                      dtype=np.float64)


# ---------------------------------------------------------------------------
# 그리드와 wave — GPU 를 몇 번 채우는가
# ---------------------------------------------------------------------------
def _v_tiles_mn(df):
    return (np.ceil(df["M"].to_numpy(np.float64) / df["tile_m"].to_numpy(np.float64)),
            np.ceil(df["N"].to_numpy(np.float64) / df["tile_n"].to_numpy(np.float64)))


def _v_grid_tiles(df):
    gm, gn = _v_tiles_mn(df)
    return gm * gn * df["split_k"].to_numpy(np.float64)


@feature(unit="count", expected_range=(1.0, 1e7), direction="neutral",
         vec=lambda df, hw, p: np.log2(1.0 + _v_grid_tiles(df)))
def log_grid_tiles(p: Problem, hw: Hardware, cfg: Config) -> float:
    """CTA 개수(log2). 그리드가 얼마나 큰가. 방향성 없음 — 규모 지표다."""
    tiles = (math.ceil(p.M / cfg.tile_m) * math.ceil(p.N / cfg.tile_n)
             * cfg.split_k)
    return math.log2(1.0 + tiles)


def _v_waves(df, hw):
    occ = np.maximum(1.0, df["max_blocks_per_sm"].to_numpy(np.float64))
    return _v_grid_tiles(df) / (hw.sm_count * occ)


@feature(unit="dimensionless", expected_range=(0.0, 1e5), direction="neutral",
         vec=lambda df, hw, p: _v_waves(df, hw))
def waves(p: Problem, hw: Hardware, cfg: Config) -> float:
    """그리드가 GPU 를 몇 번 채우는가. occupancy 를 반영한다.

    1 미만이면 SM 이 남는다. 방향성 없음 — 크다고 나쁜 것이 아니다.
    """
    tiles = (math.ceil(p.M / cfg.tile_m) * math.ceil(p.N / cfg.tile_n)
             * cfg.split_k)
    return tiles / (hw.sm_count * max(1, cfg.max_blocks_per_sm))


def _v_tail_waste(df, hw):
    w = np.maximum(_v_waves(df, hw), 1e-12)
    full = np.ceil(w)
    return (full - w) / full


@feature(expected_range=(0.0, 1.0), direction="higher_is_worse",
         vec=lambda df, hw, p: _v_tail_waste(df, hw))
def tail_waste(p: Problem, hw: Hardware, cfg: Config) -> float:
    """마지막 wave 에서 노는 SM 슬롯의 비율. 0~1, 클수록 나쁨.

    waves 가 크면 자연히 0 에 가까워진다 — 그래서 조건부 분기 없이도
    큰 형상에서 이 항이 알아서 사라진다 (§2.3 의 "좋은 수정").
    """
    w = max(1e-12, waves(p, hw, cfg))
    full = math.ceil(w)
    return (full - w) / full


@feature(expected_range=(0.0, 10.0), direction="higher_is_worse",
         vec=lambda df, hw, p: 1.0 / np.maximum(1.0 - _v_tail_waste(df, hw), 1e-3)
         - 1.0)
def sm_idle_cost(p: Problem, hw: Hardware, cfg: Config) -> float:
    """wave 양자화로 **몇 배** 손해인가 - 1. `tail_waste` 의 비선형 형태.

    512³ 에 128x128 을 쓰면 타일 16개로 84 SM 중 16개만 돌아 계수가 5.3배다.
    선형 항(`tail_waste=0.81`)으로는 그 크기가 안 나온다 — 손규칙이 실제로
    이걸 놓쳐서 1.221 에서 1.192 로 내려갔다 (kernelTab baselines.md).
    """
    return 1.0 / max(1e-3, 1.0 - tail_waste(p, hw, cfg)) - 1.0


# ---------------------------------------------------------------------------
# 타일이 실제로 하는 일 — 형상 x config 상호작용
# ---------------------------------------------------------------------------
def _v_edge_waste(df):
    gm, gn = _v_tiles_mn(df)
    tm = df["tile_m"].to_numpy(np.float64)
    tn = df["tile_n"].to_numpy(np.float64)
    M = df["M"].to_numpy(np.float64)
    N = df["N"].to_numpy(np.float64)
    return (gm * tm / M) * (gn * tn / N) - 1.0


@feature(expected_range=(0.0, 300.0), direction="higher_is_worse",
         vec=lambda df, hw, p: _v_edge_waste(df))
def edge_waste(p: Problem, hw: Hardware, cfg: Config) -> float:
    """타일이 형상 경계를 넘어 **버려지는 일**의 배수 - 1.

    타일은 형상 밖으로 튀어나가도 그 부분을 전부 계산한다. M=1 에 128행
    타일이면 일의 99.2% 가 버려진다 (값 127). 작은 M 이 작은 tile_m 을
    선호하는 이유이고, 형상과 config 가 만나는 지점이다.
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    return (gm * cfg.tile_m / p.M) * (gn * cfg.tile_n / p.N) - 1.0


@feature(unit="bytes", expected_range=(0.0, 100.0), direction="higher_is_worse",
         vec=lambda df, hw, p: (lambda gm, gn: np.log2(
             1.0 + gm * gn * (df["tile_m"].to_numpy(np.float64)
                              + df["tile_n"].to_numpy(np.float64))
             * df["K"].to_numpy(np.float64) * _v_ebytes(df)))(*_v_tiles_mn(df)))
def log_dram_traffic(p: Problem, hw: Hardware, cfg: Config) -> float:
    """A/B 를 DRAM 에서 몇 바이트 읽는가 (log2). 타일링이 결정한다.

    `ceil(M/tm)*ceil(N/tn)*(tm+tn)*K*elem` — 타일이 클수록 재사용이 커져
    총 트래픽이 준다. M=1 이면 `gm=1` 이라 tm 을 키워도 타일 수가 안 줄고
    `(tm+tn)` 만 늘어 **자동으로 벌점**이 된다.
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    b = gm * gn * (cfg.tile_m + cfg.tile_n) * p.K * _ebytes(p.dtype)
    return math.log2(1.0 + b)


@feature(expected_range=(0.0, 200.0), direction="higher_is_worse",
         vec=lambda df, hw, p: (lambda gm, gn: (
             gm * gn * (df["tile_m"].to_numpy(np.float64)
                        + df["tile_n"].to_numpy(np.float64))
             * df["K"].to_numpy(np.float64) * _v_ebytes(df))
             / np.maximum(_v_ebytes(df) * (df["M"].to_numpy(np.float64)
                                           * df["K"].to_numpy(np.float64)
                                           + df["K"].to_numpy(np.float64)
                                           * df["N"].to_numpy(np.float64)), 1.0)
         )(*_v_tiles_mn(df)))
def traffic_amplification(p: Problem, hw: Hardware, cfg: Config) -> float:
    """실제 A/B 트래픽 / 이론 최소치. 1.0 이 완벽한 재사용이다.

    `log_dram_traffic` 의 무차원 형태. 형상 크기를 나눠서 전이가 된다 (§8.1).
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    eb = _ebytes(p.dtype)
    actual = gm * gn * (cfg.tile_m + cfg.tile_n) * p.K * eb
    ideal = eb * (p.M * p.K + p.K * p.N)
    return actual / max(1.0, ideal)


@feature(expected_range=(0.0, 8.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.abs(np.log2(
             df["tile_m"].to_numpy(np.float64)
             / df["tile_n"].to_numpy(np.float64))))
def tile_aspect_imbalance(p: Problem, hw: Hardware, cfg: Config) -> float:
    """타일이 얼마나 길쭉한가. |log2(tm/tn)|. 정방형이 0.

    같은 면적이면 정방형이 A/B 트래픽을 최소화한다.
    """
    return abs(math.log2(cfg.tile_m / cfg.tile_n))


# ---------------------------------------------------------------------------
# mainloop 깊이와 split-K
# ---------------------------------------------------------------------------
@feature(unit="count", expected_range=(0.0, 16.0), direction="neutral",
         vec=lambda df, hw, p: np.log2(np.maximum(1.0,
             df["K"].to_numpy(np.float64)
             / (df["tile_k"].to_numpy(np.float64)
                * df["split_k"].to_numpy(np.float64)))))
def log_mainloop_iters(p: Problem, hw: Hardware, cfg: Config) -> float:
    """mainloop 반복 수(log2). `K / (tile_k * split_k)`.

    GBDT 가 가장 중요하게 꼽은 축이다 (§30.6b). 짧으면 파이프라인 워밍업
    비용을 못 갚고, 길면 A/B 재사용이 잘 된다.
    """
    return math.log2(max(1.0, p.K / (cfg.tile_k * cfg.split_k)))


def _v_stages_est(df):
    """smem 에서 파이프라인 깊이를 역산한다. **`ext` 를 안 본다.**"""
    denom = np.maximum(1.0, df["tile_k"].to_numpy(np.float64)
                       * (df["tile_m"].to_numpy(np.float64)
                          + df["tile_n"].to_numpy(np.float64)) * _v_ebytes(df))
    return np.maximum(1.0, df["smem_bytes"].to_numpy(np.float64) / denom)


@feature(expected_range=(0.0, 4.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.clip(
             _v_stages_est(df)
             / np.maximum(1.0, df["K"].to_numpy(np.float64)
                          / (df["tile_k"].to_numpy(np.float64)
                             * df["split_k"].to_numpy(np.float64))), 0.0, 4.0))
def pipeline_warmup_frac(p: Problem, hw: Hardware, cfg: Config) -> float:
    """파이프라인 채우기가 mainloop 에서 차지하는 비율.

    ★ 깊이를 아키텍처 전용 확장 필드에서 읽지 않고 **smem 에서 역산한다.**
    CUTLASS 의 mainloop smem 이 `stages * tile_k * (tile_m+tile_n) * elem`
    이므로 나누면 깊이가 나온다. `ext` 를 안 보므로 아키텍처 전이가 되고
    (§4.3), SM90 처럼 stages 개념이 다른 곳에서도 "smem 을 얼마나 깊이
    쌓았는가" 로 여전히 의미가 있다.

    깊은 파이프라인은 mainloop 이 짧을 때 워밍업을 못 갚는다.
    """
    denom = max(1.0, cfg.tile_k * (cfg.tile_m + cfg.tile_n) * _ebytes(p.dtype))
    stages = max(1.0, cfg.smem_bytes / denom)
    iters = max(1.0, p.K / (cfg.tile_k * cfg.split_k))
    return min(4.0, stages / iters)


@feature(expected_range=(0.0, 1.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.log2(df["split_k"].to_numpy(np.float64)) / 4.0)
def split_k_cost(p: Problem, hw: Hardware, cfg: Config) -> float:
    """split-K 리덕션 비용의 대리 지표. log2(sk)/4, sk=1 이면 0.

    serial split-K 는 파티션마다 D 를 왕복하므로 파티션 수에 비례해 비용이
    붙는다. 로그를 쓰는 이유는 sk=1->2 의 차이가 8->16 보다 크기 때문이다.
    """
    return math.log2(cfg.split_k) / 4.0


@feature(expected_range=(0.0, 40.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.log2(
             1.0 + np.where(df["split_k_mode"].to_numpy().astype(str) == "parallel",
                            _v_ebytes(df) * df["M"].to_numpy(np.float64)
                            * df["N"].to_numpy(np.float64)
                            * df["split_k"].to_numpy(np.float64), 0.0)))
def log_workspace_bytes(p: Problem, hw: Hardware, cfg: Config) -> float:
    """parallel split-K 의 리덕션 트래픽(log2). serial 이면 0.

    GBDT 가 상위로 꼽았는데 손규칙은 안 썼다 (§30.6b). parallel 은 부분합
    M*N*sk 개를 DRAM 에 쓰고 다시 읽는다 — 66/66 형상에서 최적이 serial 인
    이유를 이것이 설명할 수 있다.
    """
    if cfg.split_k_mode != "parallel":
        return 0.0
    return math.log2(1.0 + _ebytes(p.dtype) * p.M * p.N * cfg.split_k)


# ---------------------------------------------------------------------------
# 자원 압력 — 빌드 시점에 알 수 있는 커널 속성 (§3.2)
# ---------------------------------------------------------------------------
@feature(expected_range=(0.0, 1.5), direction="higher_is_worse",
         vec=lambda df, hw, p: df["smem_bytes"].to_numpy(np.float64)
         / hw.smem_per_block)
def smem_pressure(p: Problem, hw: Hardware, cfg: Config) -> float:
    """smem 예산을 얼마나 쓰는가. 꽉 채우면 SM 당 상주 블록이 줄어든다."""
    return cfg.smem_bytes / hw.smem_per_block


@feature(expected_range=(0.0, 4.0), direction="higher_is_worse",
         vec=lambda df, hw, p: df["regs_total_per_block"].to_numpy(np.float64)
         / hw.regs_per_sm if "regs_total_per_block" in df.columns
         else df["regs_per_thread"].to_numpy(np.float64)
         * df["threads"].to_numpy(np.float64) / hw.regs_per_sm)
def reg_pressure(p: Problem, hw: Hardware, cfg: Config) -> float:
    """블록 하나가 SM 레지스터 파일의 몇 배를 요구하는가."""
    return cfg.regs_per_thread * cfg.threads / hw.regs_per_sm


@feature(expected_range=(0.0, 1.0), direction="higher_is_worse",
         vec=lambda df, hw, p: 1.0 - np.clip(
             df["max_blocks_per_sm"].to_numpy(np.float64)
             * df["threads"].to_numpy(np.float64) / hw.max_threads_per_sm,
             0.0, 1.0))
def occupancy_deficit(p: Problem, hw: Hardware, cfg: Config) -> float:
    """SM 당 스레드 슬롯 중 채우지 못하는 비율. 0 이면 만점.

    `max_blocks_per_sm` 은 빌드 시점 값이라 써도 된다 (§3.2).
    """
    used = cfg.max_blocks_per_sm * cfg.threads / hw.max_threads_per_sm
    return 1.0 - min(1.0, max(0.0, used))


@feature(expected_range=(0.0, 1.0), direction="higher_is_worse",
         vec=lambda df, hw, p: (df["spill_bytes"].to_numpy(np.float64) > 0
                                ).astype(np.float64))
def has_spill(p: Problem, hw: Hardware, cfg: Config) -> float:
    """레지스터 스필이 있는가. 0 또는 1.

    ★ 이 표에서 스필 커널은 **최적으로 뽑힌 적이 0회**이고 rel 중앙값이
    13.6(최대 37.2)이다. 전체 행의 7.4% 이며 긴 꼬리가 전부 여기서 나온다.
    """
    return 1.0 if cfg.spill_bytes > 0 else 0.0


@feature(expected_range=(0.0, 10.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.log2(
             1.0 + df["spill_bytes"].to_numpy(np.float64)) / 4.0)
def spill_magnitude(p: Problem, hw: Hardware, cfg: Config) -> float:
    """스필의 크기(log2/4). 있고 없고만이 아니라 얼마나인지."""
    return math.log2(1.0 + cfg.spill_bytes) / 4.0


@feature(expected_range=(0.0, 1.0), direction="higher_is_worse",
         vec=lambda df, hw, p: (df["pipeline_kind"].to_numpy().astype(str)
                                == "pipelined").astype(np.float64))
def is_two_stage(p: Problem, hw: Hardware, cfg: Config) -> float:
    """2단 파이프라인(MmaPipelined)인가. multistage 와 **다른 커널 계열**이다.

    `ext.stages` 를 안 보고 `pipeline_kind` 를 본다 — 그쪽이 아키텍처
    공통 필드라 전이가 된다 (§4.3).
    """
    return 1.0 if cfg.pipeline_kind == "pipelined" else 0.0


@feature(unit="count", expected_range=(0.0, 30.0), direction="higher_is_worse",
         vec=lambda df, hw, p: np.log2(1.0 + df["inst_total"].to_numpy(np.float64))
         if "inst_total" in df.columns else np.zeros(len(df)))
def log_inst_total(p: Problem, hw: Hardware, cfg: Config) -> float:
    """SASS 명령어 수(log2). 커널 복잡도의 대리 지표.

    GBDT 가 상위로 꼽았는데 손규칙은 안 썼다 (§30.6b). 빌드 시점에 알 수
    있으므로 써도 된다. 표에 없으면 0 — 그 경우 이 항은 상수가 되어
    `validate` 의 "상수" 검사에 걸린다.
    """
    return math.log2(1.0 + float(cfg.inst_total))


# ---------------------------------------------------------------------------
# 형상 수준 — **스칼라**라서 규칙이 `if` 를 쓸 수 있다 (§8.1 대체본)
# ---------------------------------------------------------------------------
@shape_feature(unit="flop/byte", expected_range=(0.0, 1e5), direction="neutral")
def arith_intensity(p: Problem, hw: Hardware, cfg: Config) -> float:
    """형상만의 함수. 2MNK / (읽고 쓰는 바이트)."""
    eb = _ebytes(p.dtype)
    return 2.0 * p.M * p.N * p.K / max(1.0, eb * (p.M * p.K + p.K * p.N
                                                  + p.M * p.N))


@shape_feature(expected_range=(0.0, 1e4), direction="neutral")
def roofline_ratio(p: Problem, hw: Hardware, cfg: Config) -> float:
    """AI / ridge point. 1 미만이면 메모리 바운드다.

    `hw.ridge_point` 는 **실효값**에서 계산된다 (§6.2). 스펙값을 쓰면
    26% 어긋나 경계 근처 형상의 분류가 뒤집힌다.
    """
    return arith_intensity(p, hw, cfg) / hw.ridge_point


@shape_feature(expected_range=(0.0, 1.0), direction="neutral")
def is_memory_bound(p: Problem, hw: Hardware, cfg: Config) -> float:
    """메모리 바운드인가. 0 또는 1. **규칙이 `if p.is_memory_bound:` 로 쓴다.**"""
    return 1.0 if roofline_ratio(p, hw, cfg) < 1.0 else 0.0


@shape_feature(expected_range=(-25.0, 15.0), direction="neutral")
def log_sol_ms(p: Problem, hw: Hardware, cfg: Config) -> float:
    """roofline 하한 시간(log2, ms). **측정값이 아니라 형상에서 계산한다.**

    형상이 얼마나 짧은가의 대리 지표다. 크기 층화(§30.4)가 지표에서
    중요한데 `best_ms` 는 `ANSWER_COLS` 라 규칙이 볼 수 없다. 이것은
    형상과 하드웨어만으로 계산되므로 배포 시점에도 알 수 있다.
    """
    eb = _ebytes(p.dtype)
    t_c = 2.0 * p.M * p.N * p.K / (hw.peak_tflops_f16 * 1e12) * 1e3
    t_m = eb * (p.M * p.K + p.K * p.N + p.M * p.N) / (hw.bandwidth_gbps
                                                      * 1e9) * 1e3
    return math.log2(max(1e-6, t_c, t_m))


@shape_feature(expected_range=(0.0, 1.0), direction="neutral")
def can_use_cp_async(p: Problem, hw: Hardware, cfg: Config) -> float:
    """alignment 가 cp.async 를 허용하는가. 0 이면 stages=2 만 가능하다.

    ★ 표에서 확인된 물리적 사실이다 (§18.3). alignment 1 형상은
    multistage 커널이 아예 없다.
    """
    return 1.0 if min(p.M, p.N, p.K) and _align_ok(p) else 0.0


def _align_ok(p: Problem) -> bool:
    """K 방향 접근이 16바이트 정렬을 만족하는가 (cp.async 요건)."""
    elems = 16 // _ebytes(p.dtype)
    return (p.K % elems == 0) and (p.N % elems == 0)
