"""가중치 최적화 — 구조와 파라미터의 분리 (§29).

## 왜 나누는가

규칙은 두 가지가 섞여 있다.

    구조     이산, 조합적. "어떤 피처를 어떻게 엮을까"   -> LLM
    가중치   연속.         "2.0 이 맞나 2.7 이 맞나"      -> 수치 최적화기

## ★ 언제 돌리는가가 핵심 (§29.3)

    잘못:  LLM 규칙 생성 -> 채점 -> 아카이브
    맞음:  LLM 규칙 생성 -> **가중치 최적화** -> 채점 -> 아카이브

안 하면 좋은 구조가 나쁜 초기값 때문에 버려지고 평범한 구조가 좋은 초기값으로
살아남는다. 진화가 구조가 아니라 **가중치 운**을 선택하게 된다.

## gradient descent 를 쓸 수 없다 (§29.2)

목적함수 regret 은 `argmin` 을 거쳐 나오므로 가중치에 대해 **계단 함수**다.
가중치를 조금 바꿔도 순위가 안 바뀌면 regret 이 그대로고, 어느 순간 순위가
뒤집히며 점프한다. 기울기가 0이거나 정의되지 않는다. Nelder-Mead 를 쓴다.

## 학습 분할만 받는다 (§29.7)

`Split` 의 `role` 을 검사한다. 검증/최종 분할이 목적함수에 들어가는 경로를
만들지 않는다 — 문서에 적는 것은 강제가 아니다 (§30.8).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from kernelrule.core.matrix import Feats, FeatureMatrix, ShapeInfo
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, SplitError
from kernelrule.core.table import PerfTable
from kernelrule.core.types import Hardware, Problem

__all__ = ["FitError", "FittedRule", "ScoreFn", "fit_weights", "make_order_fn",
           "make_score_of"]

#: LLM 이 쓰는 함수. `w` 는 **수치 최적화기가 맞춘다** (§8.1 대체본).
ScoreFn = Callable[[Feats, ShapeInfo, Hardware, np.ndarray], np.ndarray]


class FitError(RuntimeError):
    """가중치를 적합할 수 없다. 규칙을 기각한다."""


@dataclass(frozen=True, slots=True)
class FittedRule:
    w: np.ndarray
    w0: np.ndarray
    fit_regret: float
    n_evals: int
    n_infeasible: int
    sensitivity: np.ndarray
    seconds: float
    val_regret: float = float("nan")
    code: str = ""
    method: str = "nelder-mead"

    @property
    def gap(self) -> float:
        """학습 - 검증 격차. 라운드마다 기록한다 (§29.4).

        벌어지면 파라미터 수를 줄여야 한다.
        """
        return self.val_regret - self.fit_regret

    @property
    def dead_terms(self) -> list[int]:
        """0 근처로 수렴했거나 둔감한 항. 피처 정리 후보다 (§29.6)."""
        return [i for i in range(len(self.w))
                if abs(self.w[i]) < 1e-3 or self.sensitivity[i] < 1e-6]

    def __str__(self) -> str:
        return (f"FittedRule(fit={self.fit_regret:.4f}, "
                f"val={self.val_regret:.4f}, evals={self.n_evals}, "
                f"w={np.array2string(self.w, precision=3)})")


class _Problem:
    """적합용 사전 계산. 형상별로 피처/tie-break/시간을 한 번만 꺼내 둔다.

    ★ `score_fn` 은 여기서도 시간을 못 본다. 시간은 `_regret` 안에서
    **순서가 정해진 뒤** 인덱싱에만 쓰인다.
    """

    __slots__ = ("hw", "items", "k")

    def __init__(self, matrix: FeatureMatrix, table: PerfTable,
                 shapes: Sequence[Problem], k: int) -> None:
        self.hw = matrix.hw
        self.k = k
        self.items = []
        for p in shapes:
            f, info = matrix.for_shape(p)
            cand = table.candidates(p)
            t = np.asarray(table.times_of(p), dtype=np.float64)
            self.items.append((f, info, cand, t, float(table.best_time(p))))

    def regret(self, score_fn: ScoreFn, w: np.ndarray) -> float:
        rs = np.empty(len(self.items), dtype=np.float64)
        for i, (f, info, cand, t, best) in enumerate(self.items):
            s = np.asarray(score_fn(f, info, self.hw, w), dtype=np.float64)
            if s.shape != (cand.n,) or not np.all(np.isfinite(s)):
                return float("inf")      # 이 가중치에서 실행 불가 (기각 아님)
            # ★ 상위 k개만 뽑는다. 전체 정렬은 이 루프에서 비용이 지배한다.
            rs[i] = float(t[cand.top_k(s, self.k)].min()) / best
        return geomean(rs)


def fit_weights(score_fn: ScoreFn, matrix: FeatureMatrix, table: PerfTable,
                split: Split, w0: Sequence[float], *,
                method: str = "nelder-mead", max_evals: int = 200,
                k: int = 1, val_split: Split | None = None,
                n_restarts: int = 4,
                sensitivity_delta: float = 0.5) -> FittedRule:
    """구조를 고정하고 가중치만 맞춘다.

    `split` 은 **`role="train"` 이어야 한다.** 검증/최종이 목적함수에
    들어가는 경로는 없다 (§29.7).

    `val_split` 은 적합이 끝난 뒤 **보고용으로만** 채점된다. 목적함수에
    관여하지 않는다 — 격차(`FittedRule.gap`)를 라운드마다 기록하기 위한 것이다.
    """
    if not isinstance(split, Split):
        raise SplitError(
            "fit_weights 는 Split 을 받는다. 형상 리스트를 넘기면 어느 분할인지 "
            "알 수 없다 — 명시하지 않으면 에러다 (§26.4).")
    if split.role != "train":
        raise SplitError(
            f"fit_weights 에 role={split.role!r} 분할이 들어왔다. "
            "학습 분할만 받는다 (§29.7). 검증/최종으로 적합하면 홀드아웃이 "
            "아니게 된다.")
    if val_split is not None and val_split.role != "val":
        raise SplitError(
            f"val_split 의 role 이 {val_split.role!r} 다. 'val' 이어야 한다.")

    w0 = np.asarray(list(w0), dtype=np.float64)
    if w0.ndim != 1 or w0.size == 0:
        raise FitError(f"W0 형태가 잘못됐다: {w0.shape}")
    if not np.all(np.isfinite(w0)):
        raise FitError("W0 에 비유한 값이 있다.")

    t0 = time.perf_counter()
    prob = _Problem(matrix, table, split.shapes, k)
    n_eval = 0
    n_inf = 0

    def obj(w: np.ndarray) -> float:
        nonlocal n_eval, n_inf
        n_eval += 1
        v = prob.regret(score_fn, np.asarray(w, dtype=np.float64))
        if not np.isfinite(v):
            n_inf += 1
            return 1e6      # 이 가중치는 실행 불가. 구조 기각은 아니다.
        return v

    m = method.lower()
    if m not in ("nelder-mead", "neldermead", "powell"):
        raise FitError(f"알 수 없는 최적화기: {method!r}. "
                       "nelder-mead | powell 중 하나여야 한다.")

    # ★ 재시작이 필요하다. 목적함수가 **계단 함수**라 단순 심플렉스는
    #   평평한 지대에서 수축해 멈춘다 (§29.2). 실제로 초기값에서 한 발도
    #   못 움직이는 경우가 나온다. 매 재시작마다 심플렉스를 다시 부풀리고,
    #   몇 개는 무작위 출발점에서 시작한다. 시드가 고정이라 결정론적이다.
    best_w = w0.copy()
    best_v = obj(w0)
    rng = np.random.default_rng(_RESTART_SEED)
    per = max(20, max_evals // max(1, n_restarts))
    for r in range(n_restarts):
        if n_eval >= max_evals:
            break
        start = best_w if r == 0 else best_w + rng.normal(
            0.0, 0.35 * np.maximum(np.abs(best_w), 1.0))
        res = _minimize_once(obj, start, m, per, r)
        v = obj(np.asarray(res.x, dtype=np.float64))
        if np.isfinite(v) and v < best_v:
            best_v, best_w = v, np.asarray(res.x, dtype=np.float64)

    if n_inf >= n_eval:
        raise FitError(
            f"모든 가중치에서 규칙이 유효한 점수를 내지 못했다 "
            f"({n_inf}/{n_eval}). 구조를 기각한다.")

    w = best_w
    fit_regret = float(prob.regret(score_fn, w))
    if not np.isfinite(fit_regret) or fit_regret >= 1e6:
        raise FitError("적합된 가중치에서 regret 이 유한하지 않다. 기각한다.")

    sens = _sensitivity(prob, score_fn, w, fit_regret, sensitivity_delta)

    val = float("nan")
    if val_split is not None:
        vp = _Problem(matrix, table, val_split.shapes, k)
        val = float(vp.regret(score_fn, w))

    return FittedRule(w=w, w0=w0, fit_regret=fit_regret, n_evals=n_eval,
                      n_infeasible=n_inf, sensitivity=sens,
                      seconds=time.perf_counter() - t0, val_regret=val,
                      method=m)


#: 재시작 출발점의 시드. 고정이라 `fit_weights` 는 결정론적이다.
_RESTART_SEED = 20260820


def _minimize_once(obj, start, method: str, budget: int, r: int):
    """재시작 한 번. 심플렉스를 **다시 부풀려서** 시작한다."""
    from scipy.optimize import minimize

    start = np.asarray(start, dtype=np.float64)
    if method == "powell":
        return minimize(obj, start, method="Powell",
                        options={"maxfev": budget, "xtol": 1e-4, "ftol": 1e-6})
    n = len(start)
    step = 0.6 * (0.5 ** r) * np.maximum(np.abs(start), 1.0)
    simplex = np.tile(start, (n + 1, 1))
    for i in range(n):
        simplex[i + 1, i] += step[i]
    return minimize(obj, start, method="Nelder-Mead",
                    options={"maxfev": budget, "xatol": 1e-4, "fatol": 1e-9,
                             "adaptive": True, "initial_simplex": simplex})


def _sensitivity(prob: _Problem, score_fn: ScoreFn, w: np.ndarray,
                 base: float, delta: float) -> np.ndarray:
    """각 `w[i]` 를 ±delta 만큼 흔들었을 때의 regret 변화 (§29.6).

    둔감한 항은 그 피처가 쓸모없다는 뜻이고, 아주 민감한 항은 그 물리량이
    지배적이라는 뜻이다. 둘 다 진단 리포트에 들어갈 정보다.
    """
    out = np.zeros(len(w), dtype=np.float64)
    for i in range(len(w)):
        d = abs(w[i]) * delta if abs(w[i]) > 1e-9 else delta
        vals = []
        for sign in (+1.0, -1.0):
            wp = w.copy()
            wp[i] += sign * d
            v = prob.regret(score_fn, wp)
            if np.isfinite(v):
                vals.append(abs(v - base))
        out[i] = max(vals) if vals else 0.0
    return out


def make_score_of(score_fn: ScoreFn, matrix: FeatureMatrix,
                  w: Sequence[float]):
    """`score_fn` + 가중치 -> `evaluate_scores` 가 받는 `score_of`.

    채점 뜨거운 경로에서는 이쪽을 쓴다 (상위 k개만 뽑는다).
    """
    w = np.asarray(list(w), dtype=np.float64)
    hw = matrix.hw

    def score_of(p: Problem, cand) -> np.ndarray:
        f, info = matrix.for_shape(p)
        return np.asarray(score_fn(f, info, hw, w), dtype=np.float64)

    return score_of


def make_order_fn(score_fn: ScoreFn, matrix: FeatureMatrix,
                  w: Sequence[float]):
    """`score_fn` + 가중치 -> 채점기가 받는 `order_fn` (§scoring).

    ★ 학습과 배포가 **같은 `score_fn`** 을 쓴다. 변환이 없으므로 "학습 때와
    배포 때가 다르다" 는 오류가 원천 차단된다 (§8.1 대체본).
    """
    w = np.asarray(list(w), dtype=np.float64)
    hw = matrix.hw

    def order_fn(p: Problem, cand) -> np.ndarray:
        f, info = matrix.for_shape(p)
        s = np.asarray(score_fn(f, info, hw, w), dtype=np.float64)
        return cand.order_by(s)

    return order_fn
