"""홀드아웃 분할 (§10).

## ★ 학습 분할과 홀드아웃 분할은 **다른 제약**을 받는다

    홀드아웃   블록이어야 한다.               보간을 막기 위해서
    학습      모든 체제를 충분히 담아야 한다.   소수 체제 희생을 막기 위해서

**두 요구가 충돌하면 학습 쪽이 우선이다.** 학습이 못 본 체제는 애초에
평가할 자격이 없다.

원래 이 절은 "블록 분할을 쓰라, 무작위는 보간이다" 만 말했다. **블록 분할이
체제를 통째로 한쪽에 몰아넣을 수 있다는 것**을 놓쳤다. `M > 2048` 이 정확히
그 형태다 — 긴 형상 11개가 전부 홀드아웃으로 가고 학습은 82% 가 짧은 형상이
된다.

실측 (같은 루프/시드/예산, 분할만 바꿈):

| 학습 구성 | 12라운드 후 검증 격차 | 전체 61형상 regret |
|---|---:|---:|
| 짧은 82% / 긴 18% | **+2.629** | 1.390 (손규칙 1.177 보다 **나쁘다**) |
| 짧은 69% / 긴 31% | +0.009 | 1.143 |

**진화는 학습 분할의 체제 구성이 허용하는 거래를 한다.** 소수 체제를
희생하는 것이 학습 점수에 유리하면 그렇게 한다. 그리고 홀드아웃이 그 소수
체제와 겹치지 않으면 **폭발이 안 보일 뿐 규칙은 여전히 그 체제에서 나쁘다.**

⚠️ **무작위 분할 금지.** M=4095 가 학습에 있으면 M=4096 은 시험이 아니다.
블록 분할을 쓴다.

이 파일이 1단계에 있는 이유는 `Split` **타입** 때문이다. `fit_weights` 가
"학습 분할만 받는다" 를 강제하려면 역할이 타입에 박혀 있어야 한다 (§29.7).
문서에 '학습 분할을 넣으세요' 라고 적는 것은 강제가 아니다 (§30.8).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from kernelrule.core.types import Problem

__all__ = ["Split", "SplitSet", "SplitError", "by_predicate",
           "split_by_M_range", "split_by_K_range", "split_by_alignment",
           "split_by_size", "split_by_waves", "SPLITS",
           "RegimeBalance", "regime_of", "check_balance", "describe",
           "MIN_REGIME_FRAC"]

Role = Literal["train", "val", "test"]


class SplitError(RuntimeError):
    """분할이 잘못됐다. **빈 집합으로 진행하지 않는다** (§26.4)."""


@dataclass(frozen=True, slots=True)
class Split:
    """형상 부분집합 + **역할**. 역할이 타입에 있는 것이 요점이다.

        train  진단 리포트와 가중치 적합에 쓰인다. LLM 이 본다
        val    라운드마다 채점. LLM 은 못 보지만 사람은 본다
        test   프로젝트 끝에 딱 한 번. 봉인
    """

    role: Role
    shapes: tuple[Problem, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if self.role not in ("train", "val", "test"):
            raise SplitError(f"알 수 없는 역할: {self.role!r}")
        if not self.shapes:
            raise SplitError(
                f"분할 {self.name or self.role!r} 이 빈 집합이다. "
                "빈 집합을 반환하고 통과시키지 않는다 (§26.4).")

    def __len__(self) -> int:
        return len(self.shapes)

    def __iter__(self):
        return iter(self.shapes)


@dataclass(frozen=True, slots=True)
class SplitSet:
    """3분할 (§10.2). 가운데가 필요한 이유는 **사람이 오염원**이기 때문이다."""

    train: Split
    val: Split
    test: Split | None = None
    kind: str = ""

    def __post_init__(self) -> None:
        a = {p.key for p in self.train}
        b = {p.key for p in self.val}
        overlap = a & b
        if overlap:
            raise SplitError(
                f"train 과 val 이 {len(overlap)}개 형상을 공유한다: "
                f"{sorted(overlap)[:5]}. 홀드아웃이 아니다.")
        if self.test is not None:
            c = {p.key for p in self.test}
            if (a & c) or (b & c):
                raise SplitError("test 가 train/val 과 겹친다. 봉인이 깨졌다.")


def by_predicate(shapes: Sequence[Problem],
                 held_out: Callable[[Problem], bool], *,
                 name: str = "", val_frac_of_heldout: float = 1.0) -> SplitSet:
    """술어 하나로 블록 분할한다. 술어가 참인 형상이 홀드아웃이다.

    2단계에서 `split_by_M_range` / `split_by_K_range` / `split_by_alignment` /
    `split_by_arch` 가 이 위에 올라간다.
    """
    train = tuple(p for p in shapes if not held_out(p))
    out = tuple(p for p in shapes if held_out(p))
    if not train or not out:
        raise SplitError(
            f"분할 {name!r} 이 한쪽을 비웠다 (train={len(train)}, "
            f"heldout={len(out)}). 에러로 낸다 — 진행하지 않는다 (§26.4).")
    n_val = max(1, int(round(len(out) * val_frac_of_heldout)))
    return SplitSet(
        train=Split("train", train, name=f"{name}:train"),
        val=Split("val", out[:n_val], name=f"{name}:val"),
        test=(Split("test", out[n_val:], name=f"{name}:test")
              if out[n_val:] else None),
        kind=name)


# ---------------------------------------------------------------------------
# 블록 분할 (§10.1)
# ---------------------------------------------------------------------------
def split_by_M_range(shapes: Sequence[Problem], *, m_threshold: int = 2048
                     ) -> SplitSet:
    """M > 2048 홀드아웃. **외삽**을 시험한다.

    kernelTab 의 GBDT 주 지표가 이 분할이다 (홀드아웃 11형상, 1.011).
    형상 단위 5-fold(1.019)와의 0.8%p 격차가 **형상 일반화가 얼마나
    어려운가**의 척도다 — 5-fold 는 M=1024 가 학습에, M=1000 이 검증에
    들어가는 사실상 보간이다.
    """
    return by_predicate(shapes, lambda p: p.M > m_threshold,
                        name=f"M>{m_threshold}")


def split_by_K_range(shapes: Sequence[Problem], *, k_threshold: int = 8192
                     ) -> SplitSet:
    """층 B 의 K 구간 홀드아웃. mainloop 깊이 외삽."""
    return by_predicate(shapes, lambda p: p.K > k_threshold,
                        name=f"K>{k_threshold}")


def split_by_alignment(shapes: Sequence[Problem]) -> SplitSet:
    """층 D 전체(alignment < 8) 홀드아웃.

    alignment 1 형상은 cp.async 를 못 써서 stages=2 만 가능하다 — 규칙이
    본 적 없는 커널 계열만 남는 구간이다.
    """
    def held(p: Problem) -> bool:
        return (p.K % 8 != 0) or (p.N % 8 != 0)
    return by_predicate(shapes, held, name="align<8")


def split_by_size(shapes: Sequence[Problem], hw, *, ms: float = 0.5
                  ) -> SplitSet:
    """짧은 형상 홀드아웃 (§30.5 의 긴장).

    여지가 거의 전부 0.5ms 미만에 있는데 그 구간이 측정 분해능이 가장
    나쁜 곳이다. **긴 형상으로 배운 것이 짧은 형상에 전이되는가** 를
    직접 시험한다.

    ⚠️ 경계를 `best_ms`(정답)가 아니라 **roofline 하한**으로 잡는다.
    `best_ms` 는 `ANSWER_COLS` 라 분할 정의에 쓰면 정답이 새어 들어간다.
    """
    from kernelrule.features.physical import log_sol_ms
    import math

    thresh = math.log2(ms)

    def held(p: Problem) -> bool:
        return log_sol_ms(p, hw, _DUMMY_CFG) < thresh
    return by_predicate(shapes, held, name=f"sol<{ms}ms")


def split_by_waves(shapes: Sequence[Problem], hw, *, tile: int = 128,
                   waves_threshold: float = 1.0) -> SplitSet:
    """참조 타일 기준 waves 로 자른다.

    ★ 층 C 는 `sm_count` 에서 M 을 **역산**하므로 M 절대값 기준 분할이
    GPU 마다 다른 것을 자른다 (§10.1). 전이 실험에서는 이 경로를 쓴다.
    """
    import math

    def held(p: Problem) -> bool:
        w = (math.ceil(p.M / tile) * math.ceil(p.N / tile)) / hw.sm_count
        return w < waves_threshold
    return by_predicate(shapes, held, name=f"waves<{waves_threshold}")


#: 이름 -> 생성자. 리포트가 전부 돌린다.
SPLITS = {
    "M_range": split_by_M_range,
    "K_range": split_by_K_range,
    "alignment": split_by_alignment,
}

#: `split_by_size` 가 형상 수준 피처를 부르는 데 필요한 자리표시자 config.
#: 형상 수준 피처는 `cfg` 를 보지 않는다 (그것이 정의다).
_DUMMY_CFG = None


def _make_dummy():
    from kernelrule.core.types import Config
    return Config(tile_m=128, tile_n=128, tile_k=32, align_a=8, align_b=8,
                  align_c=8, split_k=1, split_k_mode="serial", arch="sm_86",
                  kernel_id="_dummy", regs_per_thread=128, threads=256,
                  smem_bytes=32768, spill_bytes=0, max_blocks_per_sm=2,
                  pipeline_kind="multistage")


_DUMMY_CFG = _make_dummy()


# ---------------------------------------------------------------------------
# ★ 체제 균형 (§10.1) — 조용히 통과시키지 않는다 (§26.4)
# ---------------------------------------------------------------------------
#: 학습 분할이 어떤 체제를 이 비율 미만으로 담으면 경고한다.
#:
#: 실측 (학습 24개 고정, 시드 3개, 고정 시험대 = 긴 형상 12개):
#:
#:     긴 비율   중앙    최악
#:      8%      6.57   16.33
#:     17%      1.17   10.41
#:     25%      1.20    2.19
#:     33%      1.22    1.79
#:
#: **균형은 꼬리를 줄이지 중앙을 못 올린다.** 25% 는 하한이지 안전선이
#: 아니다 — 33% 에서도 최악이 1.79 다. 분할을 바꾸는 실험은 시드 3개
#: 이상으로 돌리고 **최악값을 함께 보고하라.**
MIN_REGIME_FRAC = 0.25


@dataclass(frozen=True, slots=True)
class RegimeBalance:
    """분할의 체제 구성. **항상 출력한다.**"""

    axis: str
    counts: dict
    n: int

    @property
    def fractions(self) -> dict:
        return {k: v / self.n for k, v in self.counts.items()} if self.n else {}

    def minority(self) -> tuple[str, float]:
        f = self.fractions
        if not f:
            return ("", 0.0)
        k = min(f, key=lambda x: f[x])
        return (k, f[k])

    @property
    def ok(self) -> bool:
        return self.minority()[1] >= MIN_REGIME_FRAC

    def __str__(self) -> str:
        parts = " / ".join(f"{k} {v}({v / self.n:.0%})"
                           for k, v in sorted(self.counts.items()))
        mark = "" if self.ok else "   ⚠️ 소수 체제 부족"
        return f"[{self.axis}] n={self.n}  {parts}{mark}"


def regime_of(p: Problem, hw, *, axis: str = "size") -> str:
    """형상의 체제. **정답을 쓰지 않는다** — roofline 하한으로 자른다.

    `best_ms` 는 `ANSWER_COLS` 라 분할 정의에 들어가면 홀드아웃이 오염된다.
    """
    import math

    from kernelrule.features.physical import is_memory_bound, log_sol_ms

    if axis == "size":
        return ("short" if log_sol_ms(p, hw, _DUMMY_CFG) < math.log2(0.5)
                else "long")
    if axis == "roofline":
        return ("mem" if is_memory_bound(p, hw, _DUMMY_CFG) else "comp")
    raise ValueError(f"알 수 없는 체제 축: {axis!r}")


def check_balance(split: Split, hw, *, axis: str = "size",
                  strict: bool = False) -> RegimeBalance:
    """학습 분할이 모든 체제를 충분히 담는가.

    ⚠️ 부족하면 **경고**한다 (`strict=True` 면 에러). 조용히 통과시키지
    않는다 — 소수 체제를 희생한 규칙이 학습 점수로는 개선처럼 보인다.
    """
    import warnings
    from collections import Counter

    c = Counter(regime_of(p, hw, axis=axis) for p in split.shapes)
    bal = RegimeBalance(axis=axis, counts=dict(c), n=len(split.shapes))
    if split.role == "train" and not bal.ok:
        k, f = bal.minority()
        msg = (f"학습 분할 {split.name or split.role!r} 의 체제 {k!r} 가 "
               f"{f:.0%} 뿐이다 (기준 {MIN_REGIME_FRAC:.0%}).\n"
               f"  {bal}\n"
               "  진화가 소수 체제를 희생하고도 학습 점수는 개선처럼 보인다. "
               "실측에서 18% 구성이 전체 regret 을 1.177 -> 1.390 으로 "
               "악화시키면서 학습 점수는 1.201 -> 1.118 로 좋아졌다 (§10.1).")
        if strict:
            raise SplitError(msg)
        warnings.warn(msg, stacklevel=2)
    return bal


def describe(ss: SplitSet, hw, *, axis: str = "size") -> str:
    """분할의 체제 구성을 사람이 읽게 낸다. **항상 찍는다.**"""
    lines = [f"분할 {ss.kind or '(이름 없음)'}"]
    for sp in (ss.train, ss.val, ss.test):
        if sp is None:
            continue
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bal = check_balance(sp, hw, axis=axis)
        lines.append(f"  {sp.role:5s} {bal}")
    return "\n".join(lines)
