"""채점 — regret, 난이도 층화, 유의성 (§7, §30.4, §30.5).

## 인터페이스가 곧 방어다

채점기가 받는 것은 **순서를 내는 함수** 하나다.

    order_fn(p: Problem, cand: CandidateSet) -> np.ndarray   # 후보 인덱스 순열

`cand` 에 시간이 없고(§types) `order_fn` 은 표를 받지 않는다. 규칙도 베이스라인도
GBDT 도 벤더 휴리스틱도 전부 이 시그니처를 만족하며, **누구도 시간을 볼 수 없다.**
시간은 `evaluate()` 안에서 순서가 **이미 정해진 뒤에** 인덱싱에만 쓰인다.

## 보고 규약 (§7.3, §30.4, §30.5)

전체 geomean 만 내는 경로를 만들지 않는다. 모든 결과는 세 축으로 쪼개진다.

    ★ 크기  t_best >= 0.5ms / < 0.5ms      — **먼저 본다** (아래)
      난이도 상위 절반 / 하위 절반          — k=1 에서만 의미가 있다
      k      1 / 3 / 5 / 10                 — k=1 과 k>=3 은 다른 배포 시나리오다

## ★ 크기 층화가 난이도 층화보다 5배 더 갈린다 (§30.5)

정적 top-1 의 정본 값에서:

    난이도 상/하    1.099  vs  1.132     차이 0.03
    크기 >=/<0.5ms  1.021  vs  1.164     차이 0.14   <- 5배

**고정 config 의 손해는 거의 전부 0.5ms 미만 형상에서 온다.** 그런데 그
구간이 정확히 **측정 분해능이 가장 나쁜 곳**이다 (§30.2 — 0.5ms 에서 눈금
하나가 0.2%, 14µs 에서 7.3%).

    노릴 여지가 있는 곳  =  측정으로 확인하기 가장 어려운 곳

이것이 이 프로젝트의 가장 큰 긴장이고, 지표 설계에 그대로 들어간다.
그래서 `stratified()` 는 크기를 먼저 내고, `report()` 도 크기를 먼저 찍는다.
`hit_rate`(정답 집합 적중)를 regret 과 함께 보는 이유도 이것이다 — 짧은
형상에서는 regret 차이가 노이즈일 수 있지만 "구분 불가능한 집합 안에
들어갔는가" 는 노이즈 바닥을 이미 반영한 판정이다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from kernelrule.core.table import PerfTable
from kernelrule.core.types import CandidateSet, Problem

__all__ = [
    "Evaluation",
    "OrderFn",
    "ScoreOf",
    "Strata",
    "evaluate",
    "evaluate_scores",
    "geomean",
    "is_significant",
]

#: 임의의 랭커. 후보 인덱스의 **순열**을 낸다. **표를 받지 않는다.**
#: 벤더 휴리스틱처럼 점수가 없는 베이스라인도 이걸로 표현된다.
OrderFn = Callable[[Problem, CandidateSet], np.ndarray]

#: 점수 기반 랭커. 후보별 점수 배열을 낸다 (낮을수록 좋다).
#: 규칙은 전부 이쪽이며, 상위 k개만 뽑는 빠른 경로를 탄다.
ScoreOf = Callable[[Problem, CandidateSet], np.ndarray]

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def geomean(x) -> float:
    """기하평균. regret 은 비율 척도이므로 산술평균을 쓰지 않는다."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(a))))


@dataclass(frozen=True, slots=True)
class Strata:
    """평가 층. `PerfTable` 의 정답 쪽 통계로 만든다. **규칙에 넘기지 마라.**"""

    shapes: tuple[Problem, ...]
    hard: np.ndarray          # bool (n_shapes,) 난이도 상위 절반
    small: np.ndarray         # bool (n_shapes,) t_best < 0.5ms
    difficulty: np.ndarray    # float
    best_ms: np.ndarray       # float
    layer: tuple[str, ...] = ()

    @classmethod
    def build(cls, table: PerfTable,
              shapes: Sequence[Problem] | None = None) -> Strata:
        shapes = tuple(shapes if shapes is not None else table.shapes())
        if not shapes:
            # 빈 집합으로 진행하지 않는다 (§26.4).
            raise ValueError("Strata.build 에 형상이 하나도 없다.")
        st = [table.stats(p) for p in shapes]
        diff = np.asarray([s.difficulty for s in st], dtype=np.float64)
        best = np.asarray([s.best_ms for s in st], dtype=np.float64)
        # 중앙값 초과를 "어려운 절반" 으로 본다. 홀수여도 결정론적이다.
        hard = diff > np.median(diff)
        small = np.asarray([s.is_small for s in st])

        layers = table.meta.get("shape_layers") or {}
        lut: dict[tuple[int, int, int], str] = {}
        for name, rows in layers.items():
            for r in rows:
                lut.setdefault((int(r[0]), int(r[1]), int(r[2])), name)
        layer = tuple(lut.get((p.M, p.N, p.K), "?") for p in shapes)
        return cls(shapes=shapes, hard=hard, small=small, difficulty=diff,
                   best_ms=best, layer=layer)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """채점 결과. 형상별 원값 + 층별 집계."""

    ks: tuple[int, ...]
    shapes: tuple[Problem, ...]
    #: (n_shapes, n_ks) — 형상별 regret@k
    regret: np.ndarray
    #: (n_shapes, n_ks) — 상위 k 안에 정답 집합 원소가 있었는가
    hit: np.ndarray
    strata: Strata
    #: 형상별 2σ 허용치. 유의성 판정에 쓴다.
    tol: np.ndarray
    label: str = ""
    extra: dict = field(default_factory=dict)

    def _col(self, k: int) -> int:
        try:
            return self.ks.index(k)
        except ValueError:
            raise KeyError(f"k={k} 는 채점되지 않았다. 채점된 k: {self.ks}"
                           ) from None

    def at(self, k: int = 1, *, mask: np.ndarray | None = None) -> float:
        v = self.regret[:, self._col(k)]
        if mask is not None:
            v = v[mask]
            if v.size == 0:
                return float("nan")
        return geomean(v)

    def hit_rate(self, k: int = 1, *, mask: np.ndarray | None = None) -> float:
        v = self.hit[:, self._col(k)]
        if mask is not None:
            v = v[mask]
        return float(v.mean()) if v.size else float("nan")

    def stratified(self, k: int = 1) -> dict[str, float]:
        """§30.4 + §7.3 의 필수 층화. **이 dict 를 통째로 보고한다.**

        ★ 크기 층화가 먼저다 (§30.5). 난이도보다 5배 더 갈린다.
        """
        s = self.strata
        return {
            "all": self.at(k),
            # 크기 — 먼저 본다
            "large(>=0.5ms)": self.at(k, mask=~s.small),
            "small(<0.5ms)": self.at(k, mask=s.small),
            "n_small": float(int(s.small.sum())),
            # 난이도
            "hard": self.at(k, mask=s.hard),
            "easy": self.at(k, mask=~s.hard),
            "n_shapes": float(len(s.shapes)),
        }

    def size_gap(self, k: int = 1) -> float:
        """짧은 형상과 긴 형상의 regret 격차. **주 진단량이다** (§30.5).

        정적 top-1 에서 0.14 였다. 규칙이 이 격차를 줄이고 있는지가
        "짧은 형상에서 실제로 뭘 배웠는가" 의 직접 신호다.
        """
        s = self.strata
        return self.at(k, mask=s.small) - self.at(k, mask=~s.small)

    def difficulty_gap(self, k: int = 1) -> float:
        s = self.strata
        return self.at(k, mask=s.hard) - self.at(k, mask=~s.hard)

    def by_layer(self, k: int = 1) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in sorted(set(self.strata.layer)):
            m = np.asarray([x == name for x in self.strata.layer])
            out[name] = self.at(k, mask=m)
        return out

    def report(self) -> str:
        """★ 크기 층화를 먼저 찍는다 (§30.5)."""
        st = self.strata
        lines = [f"== {self.label or 'evaluation'} =="
                 f"  ({len(st.shapes)}형상, <0.5ms {int(st.small.sum())}개)"]
        for k in self.ks:
            s = self.stratified(k)
            lines.append(
                f"  regret@{k:<2d} 전체 {s['all']:.4f} "
                f"| >=0.5ms {s['large(>=0.5ms)']:.4f} "
                f"| <0.5ms {s['small(<0.5ms)']:.4f} (격차 {self.size_gap(k):+.4f}) "
                f"| 어려움 {s['hard']:.4f} 쉬움 {s['easy']:.4f} "
                f"| hit {self.hit_rate(k):.3f}")
        return "\n".join(lines)


def evaluate(order_fn: OrderFn, table: PerfTable,
             shapes: Sequence[Problem] | None = None, *,
             ks: Sequence[int] = DEFAULT_KS, label: str = "",
             strata: Strata | None = None) -> Evaluation:
    """`order_fn` 을 채점한다.

    ★ `order_fn` 은 `(Problem, CandidateSet)` 만 받는다. 표도 시간도 안 준다.
    시간은 순서가 정해진 **뒤에** 여기서만 인덱싱된다 (§30.7).
    """
    shapes = tuple(shapes if shapes is not None else table.shapes())
    if not shapes:
        raise ValueError("evaluate 에 형상이 하나도 없다. 빈 집합으로 진행하지 "
                         "않는다 (§26.4).")
    ks = tuple(int(k) for k in ks)
    strata = strata or Strata.build(table, shapes)

    regret = np.empty((len(shapes), len(ks)), dtype=np.float64)
    hit = np.zeros((len(shapes), len(ks)), dtype=bool)
    tol = np.empty(len(shapes), dtype=np.float64)

    for i, p in enumerate(shapes):
        cand = table.candidates(p)
        order = np.asarray(order_fn(p, cand))
        _check_order(order, cand, p)

        t = table.times_of(p)              # ← 시간이 등장하는 유일한 지점
        ans = table.answer_mask(p)
        st = table.stats(p)
        tol[i] = st.answer_tol
        for j, k in enumerate(ks):
            top = order[:k]
            regret[i, j] = float(t[top].min()) / st.best_ms
            hit[i, j] = bool(ans[top].any())

    return Evaluation(ks=ks, shapes=shapes, regret=regret, hit=hit,
                      strata=strata, tol=tol, label=label)


def evaluate_scores(score_of: ScoreOf, table: PerfTable,
                    shapes: Sequence[Problem] | None = None, *,
                    ks: Sequence[int] = DEFAULT_KS, label: str = "",
                    strata: Strata | None = None) -> Evaluation:
    """점수 기반 규칙을 채점한다. `evaluate` 와 **결과가 같고 훨씬 빠르다.**

    전체 정렬 대신 `CandidateSet.top_k` 를 쓴다. 형상당 후보가 15,000개인데
    상위 10개만 보므로 전체 정렬은 낭비다 — 실측으로 규칙당 2.7초가 나왔고
    라운드당 12규칙이면 32초라 LLM 호출과 맞먹었다.

    ★ tie-break 는 그대로 config 정체성으로만 한다 (§30.7).
    """
    shapes = tuple(shapes if shapes is not None else table.shapes())
    if not shapes:
        raise ValueError("evaluate_scores 에 형상이 하나도 없다 (§26.4).")
    ks = tuple(int(k) for k in ks)
    kmax = max(ks)
    strata = strata or Strata.build(table, shapes)

    regret = np.empty((len(shapes), len(ks)), dtype=np.float64)
    hit = np.zeros((len(shapes), len(ks)), dtype=bool)
    tol = np.empty(len(shapes), dtype=np.float64)

    for i, p in enumerate(shapes):
        cand = table.candidates(p)
        top = cand.top_k(np.asarray(score_of(p, cand)), kmax)
        t = table.times_of(p)              # ← 시간이 등장하는 유일한 지점
        ans = table.answer_mask(p)
        st = table.stats(p)
        tol[i] = st.answer_tol
        for j, k in enumerate(ks):
            sel = top[:k]
            regret[i, j] = float(t[sel].min()) / st.best_ms
            hit[i, j] = bool(ans[sel].any())

    return Evaluation(ks=ks, shapes=shapes, regret=regret, hit=hit,
                      strata=strata, tol=tol, label=label)


def _check_order(order: np.ndarray, cand: CandidateSet, p: Problem) -> None:
    """순열인지 검사한다. **조용히 넘어가지 않는다** (§26.4).

    LLM 이 만든 규칙은 후보를 빠뜨리거나 중복시키는 코드를 실제로 낸다.
    그대로 두면 regret 이 좋아 보이는 방향(후보를 줄이면 top-k 가 유리)으로
    틀린다.
    """
    if order.ndim != 1 or order.size != cand.n:
        raise ValueError(
            f"{p.key}: 순서 길이 {order.size} != 후보 수 {cand.n}. "
            "규칙은 모든 후보를 정렬해 돌려줘야 한다.")
    if order.dtype.kind not in "iu":
        raise ValueError(f"{p.key}: 순서가 정수 인덱스가 아니다 ({order.dtype}).")
    seen = np.zeros(cand.n, dtype=bool)
    seen[order] = True
    if not seen.all():
        raise ValueError(
            f"{p.key}: 순서가 순열이 아니다 (누락 {int((~seen).sum())}개). "
            "후보를 빠뜨리면 top-k 가 유리해져 채점이 틀린다.")


def is_significant(delta: float, ev: Evaluation, *,
                   mask: np.ndarray | None = None) -> bool:
    """이 regret 차이가 측정 노이즈보다 큰가 (§7.4).

    ⚠️ 고정 임계값을 쓰지 마라. 형상마다 다르다. 여기서는 평가에 포함된
    형상들의 2σ 허용치를 기하평균해 문턱으로 쓴다.

    ⚠️ 계산할 수 없으면 **유의하지 않다**고 본다 (§26.4 — 실패 쪽으로 기운다).
    아카이브를 갱신하지 않는 쪽이 노이즈를 개선으로 착각하는 것보다 안전하다.
    """
    tol = ev.tol if mask is None else ev.tol[mask]
    if tol.size == 0 or not np.all(np.isfinite(tol)):
        return False
    return abs(float(delta)) > geomean(tol)
