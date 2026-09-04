"""핵심 타입 — **측정 시간이 들어올 자리가 없다** (§3.3, §6.1).

이 파일의 존재 이유는 편의가 아니라 **방어**다. 규칙 함수가 정답을 볼 수 없게
하는 네 겹 중 첫 번째 겹이며, 나머지 셋(로더 / 정적 검사 / 행동 검사)이 전부
뚫려도 여기서 막힌다 — 규칙이 손댈 수 있는 객체에 시간이 **없기 때문**이다.

    Problem   형상.        시간 없음
    Hardware  하드웨어.    시간 없음. peak/bandwidth 는 **실효값** (§6.2)
    Config    후보.        시간 없음
    CandidateSet  형상 하나의 후보 전체를 배열로. **시간 없음 + tie-break 포함**

`CandidateSet` 이 §30.7 의 정답 누출을 구조적으로 막는 자리다. 채점기가
순서를 정할 때 쓸 수 있는 것이 `tiebreak` 정수 배열뿐이고, 그 배열은 config
정체성(kernel_id, split_k, split_k_mode)만으로 만들어진다. 시간을 tie-break 에
넣으려면 이 클래스에 없는 필드를 참조해야 하므로 `AttributeError` 가 난다.

뜨거운 경로이므로 frozen dataclass 를 쓴다. Pydantic 은 LLM 경계에서만 (§11.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "CandidateSet",
    "Config",
    "Hardware",
    "Problem",
    "ShapeKey",
    "config_key",
    "hardware_from_env",
    "shape_key",
]

#: 형상 조인 키. (M, N, K, dtype). 표와 후보를 잇는 유일한 경로다.
ShapeKey = tuple[int, int, int, str]

#: config 조인 키. (kernel_id, split_k, split_k_mode).
#: **tie-break 도 이것만으로 한다** (§30.7).
ConfigKey = tuple[str, int, str]



#: dtype 이름 -> 원소당 바이트. **알 수 없는 이름이면 예외다** — 조용히
#: 기본값으로 떨어지면 roofline 이 통째로 틀린다 (§26.4).
_DTYPE_BYTES: dict[str, float] = {
    "f16": 2.0, "bf16": 2.0, "f32": 4.0, "tf32": 4.0, "f64": 8.0,
    "i8": 1.0, "u8": 1.0, "f8": 1.0, "i32": 4.0,
}

@dataclass(frozen=True, slots=True)
class Problem:
    """GEMM 형상. D[MxN] = A[MxK] @ B[KxN].

    ⚠️ 여기에 `time_ms` / `difficulty` / `n_distinct_times` 를 넣지 마라.
    셋 다 `ANSWER_COLS` 다 — 정답에서 유도된 값이고 배포 시점에 알 수 없다.
    """

    M: int
    N: int
    K: int
    dtype: str = "f16"
    acc_dtype: str = "f32"
    layout_a: str = "row"
    layout_b: str = "col"
    layout_c: str = "row"

    @property
    def key(self) -> ShapeKey:
        return (self.M, self.N, self.K, self.dtype)

    @property
    def bytes_per_element(self) -> float:
        """A/B/C 원소 하나의 바이트. **`dtype` 에서 유도된다 — 새 정보가
        아니다** (§30.11).

        노출하는 이유: roofline 은 `FLOP / byte` 인데 바이트를 얻으려면
        dtype 을 바이트로 바꿔야 한다. 그런데 `p.dtype` 은 문자열이고,
        피처 샌드박스에는 `np.dtype(...).itemsize` 가 없다. 그래서 F1 에서
        LLM 이 `산술_대역폭_압력` 영역을 **세 번 연속 실패**했다 (D-63).

        **쓸 수 없는 필드를 목록에 올리는 것은 있다고 말하는 것이다.**
        """
        return _DTYPE_BYTES[self.dtype]

    @property
    def acc_bytes_per_element(self) -> float:
        """누산기 원소 하나의 바이트. split-K parallel 의 부분합이 이 크기다."""
        return _DTYPE_BYTES[self.acc_dtype]


@dataclass(frozen=True, slots=True)
class Hardware:
    """GPU. `peak_tflops_f16` / `bandwidth_gbps` 는 **실효값**이다 (§6.2).

    스펙값(부스트 클럭 기준)을 쓰면 ridge point 가 실제보다 26% 높게 나오고
    `is_memory_bound` 판정이 경계 근처에서 뒤집힌다. `hardware_from_env()` 를
    **유일한 진입점**으로 쓴다 — `Hardware(**env["hardware"])` 는 스펙값이다.
    """

    name: str
    arch: str
    sm_count: int
    smem_per_block: int
    max_threads_per_sm: int
    regs_per_sm: int
    peak_tflops_f16: float   # 실효
    bandwidth_gbps: float    # 실효
    l2_bytes: int

    @property
    def ridge_point(self) -> float:
        """roofline 의 무릎 [FLOP/byte]. peak_flops / bandwidth."""
        return (self.peak_tflops_f16 * 1e12) / (self.bandwidth_gbps * 1e9)


@dataclass(frozen=True, slots=True)
class Config:
    """규칙이 순위를 매기는 대상. **측정 시간 없음.**

    공통 필드 + 빌드 시점에 알 수 있는 커널 속성 + `ext`.

    `ext` 를 dict 로 두는 것은 의도다 (§6.1). 아키텍처마다 필드가 다르고,
    아키텍처 전이를 노리는 피처는 `ext` 를 보면 안 된다 — dataclass 필드로
    만들면 오히려 접근을 부추긴다.
    """

    # 공통 — 물리 피처가 이것만으로 계산되어야 전이가 성립한다 (§4.3)
    tile_m: int
    tile_n: int
    tile_k: int
    align_a: int
    align_b: int
    align_c: int
    split_k: int
    split_k_mode: str        # "serial" | "parallel"
    arch: str
    # 빌드 시점에 알 수 있는 커널 속성 — 실행 불필요하므로 써도 된다 (§3.2)
    kernel_id: str
    regs_per_thread: int
    threads: int
    smem_bytes: int
    spill_bytes: int
    max_blocks_per_sm: int
    pipeline_kind: str       # "pipelined" | "multistage"
    #: SASS 명령어 수. 빌드 시점에 알 수 있고 아키텍처 공통이다.
    #: GBDT 가 상위로 꼽은 축인데 손규칙은 안 썼다 (§30.6b).
    inst_total: int = 0
    # 아키텍처 전용. 전이 규칙은 참조 금지.
    ext: dict = field(default_factory=dict)

    @property
    def key(self) -> ConfigKey:
        return (self.kernel_id, self.split_k, self.split_k_mode)


def shape_key(p: Problem) -> ShapeKey:
    return p.key


def config_key(c: Config) -> ConfigKey:
    return c.key


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """형상 하나의 후보 전체. 열 지향 배열.

    ★ **이 객체에 시간이 없다는 것이 §30.7 방어의 전부다.**

    `idxmin` / `sorted(..., key=lambda c: (score, time))` 같은 코드를 쓰려면
    여기 없는 필드가 필요하므로 `AttributeError` 로 즉시 실패한다.
    채점기는 `PerfTable.times_of()` 로만 시간을 얻고, 그 값은 **순서가 이미
    정해진 뒤에** 인덱싱에만 쓰인다.

    `tiebreak` 는 config 정체성으로만 만든 정수 순위다. `np.lexsort` 의
    1차 키로 넣으면 점수 동점이 결정론적으로 달라진다 — 타이머 양자화 때문에
    동점이 대량 발생하므로(512³ 에서 최빈값 하나에 9.2%) 이게 실제로 중요하다.
    """

    n: int
    kernel_id: np.ndarray        # (n,) object/str
    split_k: np.ndarray          # (n,) int
    split_k_mode: np.ndarray     # (n,) object/str
    #: config 정체성만으로 만든 결정론적 순위. **정답과 무관하다.**
    tiebreak: np.ndarray         # (n,) int64
    #: Config 객체 배열 (배포 shim 과 리포트용). 지연 생성 가능.
    configs: tuple[Config, ...] = ()
    #: 원본 표에서의 행 인덱스. 채점기가 시간을 찾는 데만 쓴다.
    row_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("kernel_id", "split_k", "split_k_mode", "tiebreak"):
            arr = getattr(self, name)
            if len(arr) != self.n:
                raise ValueError(
                    f"CandidateSet.{name} 길이 {len(arr)} != n {self.n}")

    def order_by(self, score: np.ndarray) -> np.ndarray:
        """점수 오름차순 정렬 인덱스. **동점은 config 정체성으로만 가른다.**

        ⚠️ 여기에 시간을 2차 키로 넣지 마라 — 그게 §30.7 의 버그다.
        구조적으로 불가능하게 해 두었지만(이 클래스에 시간이 없다) 규율을
        코드 옆에 남긴다.
        """
        score = np.asarray(score, dtype=np.float64)
        if score.shape != (self.n,):
            raise ValueError(
                f"score 형태 {score.shape} != 후보 수 ({self.n},). "
                "규칙이 형상 하나의 모든 후보에 대한 점수 벡터를 내야 한다.")
        if not np.all(np.isfinite(score)):
            # 조용히 nan 을 뒤로 미루지 않는다 (§26.4). 규칙이 망가진 것이다.
            n_bad = int((~np.isfinite(score)).sum())
            raise ValueError(
                f"점수에 비유한 값 {n_bad}개 (nan/inf). 규칙을 기각한다.")
        # lexsort 는 마지막 키가 1차. (tiebreak, score) -> score 우선.
        return np.lexsort((self.tiebreak, score))

    def top_k(self, score: np.ndarray, k: int) -> np.ndarray:
        """상위 k개만. **전체 정렬과 정확히 같은 결과**를 O(n) 에 낸다.

        채점은 상위 k개만 보는데(k <= 10) 매번 15,000개를 전부 정렬하면
        가중치 적합 200회 x 66형상에서 그 비용이 지배한다. 실측으로 규칙당
        14.5초가 나왔고, 그러면 "채점이 사실상 공짜" 라는 §29.2 의 전제가
        무너진다.

        정확성: 점수가 `kth` 이하인 것만 모아도 충분하다. 그 밖의 원소는
        점수가 `kth` 보다 크고, `kth` 이하인 원소가 이미 k개 이상 있으므로
        상위 k에 들 수 없다. **동점은 여전히 tie-break 로만 달라진다.**
        """
        score = np.asarray(score, dtype=np.float64)
        if score.shape != (self.n,):
            raise ValueError(
                f"score 형태 {score.shape} != 후보 수 ({self.n},).")
        if not np.all(np.isfinite(score)):
            raise ValueError(
                f"점수에 비유한 값 {int((~np.isfinite(score)).sum())}개. 기각한다.")
        k = int(k)
        if k >= self.n:
            return np.lexsort((self.tiebreak, score))[:k]
        kth = np.partition(score, k - 1)[k - 1]
        pool = np.flatnonzero(score <= kth)
        return pool[np.lexsort((self.tiebreak[pool], score[pool]))][:k]


def make_tiebreak(kernel_id, split_k, split_k_mode) -> np.ndarray:
    """config 정체성으로 결정론적 정수 순위를 만든다. **정답 무관.**

    표의 행 순서에 의존하지 않는다 — `groupby.idxmin()` 이 표 행 순서에
    의존해서 "형상별 최적 config" 가 tie-break 마다 달라지는 것을 실제로
    확인했다 (66형상 중 29개가 최적시간 동점, 최대 84중 동점).
    """
    # `np.unique(..., return_inverse=True)` 의 코드는 **사전순**이다
    # (pandas 의 factorize 는 등장순이라 행 순서에 의존한다 — 쓰면 안 된다).
    kid = np.unique(np.asarray(kernel_id, dtype=object).astype(str),
                    return_inverse=True)[1]
    mode = np.unique(np.asarray(split_k_mode, dtype=object).astype(str),
                     return_inverse=True)[1]
    sk = np.asarray(split_k, dtype=np.int64)
    order = np.lexsort((mode, sk, kid))          # kernel_id 가 1차 키
    rank = np.empty(len(sk), dtype=np.int64)
    rank[order] = np.arange(len(sk), dtype=np.int64)
    return rank


def hardware_from_env(env: dict) -> Hardware:
    """`env.json` -> `Hardware`. **모든 호출부는 이것만 쓴다** (§6.2).

    kernelTab 의 같은 이름 함수에 위임한 뒤 우리 dataclass 로 옮긴다.
    실효값 보정 로직을 재구현하지 않기 위해서다.
    """
    from kerneltab.core.hardware import hardware_from_env as _kt

    kt = _kt(env)
    hw = Hardware(
        name=kt.name, arch=kt.arch, sm_count=kt.sm_count,
        smem_per_block=kt.smem_per_block,
        max_threads_per_sm=kt.max_threads_per_sm,
        regs_per_sm=kt.regs_per_sm,
        peak_tflops_f16=kt.peak_tflops_f16,
        bandwidth_gbps=kt.bandwidth_gbps,
        l2_bytes=kt.l2_bytes,
    )
    # 실효 보정이 실제로 일어났는지 확인한다. 스펙값이 그대로 오면
    # env.json 에 *_effective 가 없다는 뜻이고, 그러면 ridge point 가 틀린다.
    # 조용히 넘어가지 않는다 (§26.4).
    spec = env.get("hardware", {})
    if (env.get("peak_tflops_f16_effective") is None
            and spec.get("peak_tflops_f16") is not None):
        import warnings
        warnings.warn(
            "env.json 에 peak_tflops_f16_effective 가 없다. 스펙값(부스트 클럭)이 "
            "쓰이며 ridge point 가 실제보다 높게 나온다 — is_memory_bound 판정이 "
            "경계 근처에서 뒤집힌다 (§6.2).", stacklevel=2)
    return hw


def config_from_row(row: dict[str, Any]) -> Config:
    """어댑터가 정규화한 행 하나 -> Config. `core/adapter.py` 가 호출한다."""
    ext = {k[len("ext_"):]: v for k, v in row.items() if k.startswith("ext_")}
    return Config(
        tile_m=int(row["tile_m"]), tile_n=int(row["tile_n"]),
        tile_k=int(row["tile_k"]),
        align_a=int(row["align_a"]), align_b=int(row["align_b"]),
        align_c=int(row["align_c"]),
        split_k=int(row["split_k"]), split_k_mode=str(row["split_k_mode"]),
        arch=str(row["arch"]), kernel_id=str(row["kernel_id"]),
        regs_per_thread=int(row["regs_per_thread"]),
        threads=int(row["threads"]),
        smem_bytes=int(row["smem_bytes"]),
        spill_bytes=int(row["spill_bytes"]),
        max_blocks_per_sm=int(row["max_blocks_per_sm"]),
        pipeline_kind=str(row["pipeline_kind"]),
        inst_total=int(row.get("inst_total") or 0),
        ext=ext,
    )
