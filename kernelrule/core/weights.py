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
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from kernelrule.core.matrix import Feats, FeatureMatrix, ShapeInfo
from kernelrule.core.numerics import approx_zero
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, SplitError
from kernelrule.core.table import PerfTable
from kernelrule.core.types import Hardware, Problem

__all__ = ["FitError", "FitWarning", "FittedRule", "ScoreFn", "fit_weights",
           "make_order_fn",
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
    #: 항별 **실효 기여도** = |w_i| x (그 피처 열의 표준편차) (D-70).
    #: 절대 배율과 달리 피처 스케일에 불변이라 라이브러리를 바꿔도
    #: 같은 기준으로 읽힌다. 계산할 수 없으면 `None`.
    contrib: np.ndarray | None = None
    #: ★ **다듬기 전** 평가 횟수. `n_evals` 는 다듬기까지 합한 값이라
    #: `max_evals` 와 견줄 수 없다 — 견주면 다듬기가 켜진 순간 상한 경고가
    #: **항상** 뜬다. 상한에 닿았는지는 이 값으로 본다.
    n_fit_evals: int = 0
    #: ★ 대리 손실로 **초기점만** 만드는 데 쓴 평가 수 (`init_objective`).
    #: `n_evals` 에는 더해져 있고 `n_fit_evals` 에는 없다 — 상한 판정은
    #: 참 목적함수의 평가로만 한다. 예산 비교에는 `n_evals` 를 쓴다.
    n_init_evals: int = 0

    @property
    def moved(self) -> bool:
        """★ 적합기가 실제로 움직였는가 (D-54).

        `False` 면 **초기값으로 채점된 것**이고, §29.3 의 "가중치 최적화
        후에 채점한다 — 안 하면 진화가 구조가 아니라 가중치 운을 선택한다"
        가 그 후보에 대해 성립하지 않는다.

        24회 중 13회가 그랬다. 로그에는 적합 후 regret 만 남아서 **찾으려고
        해야 보였다** — 규칙을 내보내는 다른 작업이 우연히 드러냈다.
        """
        return not np.allclose(self.w, self.w0)

    def invariants(self) -> list[str]:
        """이 적합이 이상한 이유들. **비어 있어야 정상이다** (D-54).

        계산 단계마다 "아무것도 안 했는가" 를 스스로 알리게 한다.
        로그는 사후에 찾아야 보이고, 찾으려면 무엇을 찾을지 알아야 한다.
        """
        out: list[str] = []
        if not self.moved:
            out.append(f"적합기가 움직이지 않았다 (n_evals={self.n_evals}) "
                       "— 초기값으로 채점된다")
        w = np.asarray(self.w)
        # ★ **절대 배율(|w|/|w0|)은 지표가 아니다** (D-70).
        #
        #   F1 라이브러리는 피처가 대부분 [0, 0.2] 라 적합기가 배율을 크게
        #   키운다 — |w| 최대가 4,159,634 까지 갔다. 사람이 쓴 24개는
        #   [0, 300] 짜리가 섞여 있어 8.6~7,663 이다. **같은 100배 기준을
        #   쓰면 F1 팔에서 상시 발화해 감시가 신호를 잃는다** (원칙 11).
        #
        #   순위만 보는 목적함수라 전체 배율은 무해하다. 문제가 되는 것은
        #   **한 항이 다른 항들을 압도하는 것**이고, 그것은 실효 기여도로
        #   재야 한다: |w_i| x (그 피처의 표준편차).
        if self.contrib is not None:
            c = np.asarray(self.contrib, dtype=np.float64)
            pos = c[c > 0]
            if pos.size >= 2:
                med = float(np.median(pos))
                dom = np.flatnonzero(c > _DOMINANCE * med)
                if dom.size:
                    out.append(
                        f"한 항이 다른 항들을 압도한다: "
                        f"{[int(i) for i in dom]} — 실효 기여도가 중앙값의 "
                        f"{_DOMINANCE:.0f}배를 넘는다")
                dead = np.flatnonzero([approx_zero(x) for x in c])
                if dead.size:
                    out.append(f"실효 기여도가 0 인 항: "
                               f"{[int(i) for i in dead]} — 순위에 관여하지 "
                               "않는다")
        neg = np.flatnonzero(w < 0)
        if neg.size:
            # 전 피처가 "클수록 나쁨" 이므로 음수는 방향이 뒤집힌 것이다.
            # 형상 분기로 재가중하는 경우 정당할 수 있어 **경고만** 한다.
            out.append(f"음수 가중치: {[int(i) for i in neg]} — 피처는 전부 "
                       "'클수록 나쁨' 이다 (§8.2)")
        return out

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

    __slots__ = ("hw", "items", "k", "_pairs", "n_pairs", "n_dropped",
                 "_pairs1", "n_pairs1")

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
        self._pairs = None
        self.n_pairs = 0
        self.n_dropped = 0
        #: ★ 참 1등 대 나머지 쌍 (D-109). `rank_lambda` 가 0 이면 안 만든다.
        self._pairs1 = None
        self.n_pairs1 = 0

    # -- 순위 손실 (D-101) -------------------------------------------------
    def _make_pairs(self, table: PerfTable, top_k: int, *,
                    anchor_best: bool = False) -> tuple[list, int, int]:
        """★ 참 상위 `top_k` 안의 쌍. **노이즈로 못 가르는 쌍은 뺀다.**

        가중치는 `|t_j - t_i| / t_best` — **실제 손해**다. 튜닝할 값이
        없고, 노이즈 바닥 이내면 자동으로 0 에 가까워진다.

        `anchor_best` 면 **참 1등이 낀 쌍만** 남긴다 — `regret` 의 부드러운
        대리다 (D-109). 같은 쌍 집합의 부분집합이므로 정규화가 같고
        `lambda` 가 순수한 비율이 된다.

        ⚠️ `NoiseModel.resolvable` 을 쓴다. 판정을 새로 정의하지 않는다
        (원칙 2).
        """
        pairs, n_ok, n_drop = [], 0, 0
        for _f, _info, _cand, t, best in self.items:
            top = np.argsort(t, kind="stable")[:top_k]
            tt = t[top]
            n = len(top)
            iu, ju = np.triu_indices(n, k=1)          # t[iu] <= t[ju]
            if iu.size == 0:
                pairs.append(None)
                continue
            a, b = tt[iu], tt[ju]
            ok = table.noise.resolvable(a, b) & (b > a)
            if anchor_best:
                ok = ok & (iu == 0)
            n_drop += int((~ok).sum())
            if not ok.any():
                pairs.append(None)
                continue
            iu, ju = iu[ok], ju[ok]
            w = (tt[ju] - tt[iu]) / best              # ★ 실제 손해
            n_ok += int(iu.size)
            pairs.append((top, iu, ju, w, float(w.sum())))
        return pairs, n_ok, n_drop

    def build_pairs(self, table: PerfTable, top_k: int) -> None:
        self._pairs, self.n_pairs, self.n_dropped = self._make_pairs(
            table, top_k)

    def build_top1_pairs(self, table: PerfTable, top_k: int) -> None:
        """★ 참 1등 대 나머지 — `regret` 의 부드러운 대리 (D-109)."""
        self._pairs1, self.n_pairs1, _ = self._make_pairs(
            table, top_k, anchor_best=True)

    def rank_loss(self, score_fn: ScoreFn, w: np.ndarray) -> float:
        """가중 로지스틱 쌍 손실. **작을수록 좋다** 규약이므로 s_i < s_j."""
        if self._pairs is None:
            raise FitError("build_pairs 를 먼저 불러야 한다.")
        return self._loss_on(self._pairs, score_fn, w)

    def rank_loss_top1(self, score_fn: ScoreFn, w: np.ndarray) -> float:
        if self._pairs1 is None:
            raise FitError("build_top1_pairs 를 먼저 불러야 한다.")
        return self._loss_on(self._pairs1, score_fn, w)

    def _loss_on(self, pairs, score_fn: ScoreFn, w: np.ndarray) -> float:
        tot = den = 0.0
        for (f, info, cand, _t, _best), pr in zip(self.items, pairs,
                                                  strict=True):
            if pr is None:
                continue
            top, iu, ju, pw, wsum = pr
            s = np.asarray(score_fn(f, info, self.hw, w), dtype=np.float64)
            if s.shape != (cand.n,) or not np.all(np.isfinite(s)):
                return float("inf")
            st = s[top]
            # softplus(s_i - s_j): s_i 가 작아야(=좋아야) 손실이 준다
            tot += float((pw * np.logaddexp(0.0, st[iu] - st[ju])).sum())
            den += wsum
        return tot / den if den > 0 else float("inf")

    def regret(self, score_fn: ScoreFn, w: np.ndarray) -> float:
        rs = np.empty(len(self.items), dtype=np.float64)
        for i, (f, info, cand, t, best) in enumerate(self.items):
            s = np.asarray(score_fn(f, info, self.hw, w), dtype=np.float64)
            if s.shape != (cand.n,) or not np.all(np.isfinite(s)):
                return float("inf")      # 이 가중치에서 실행 불가 (기각 아님)
            # ★ 상위 k개만 뽑는다. 전체 정렬은 이 루프에서 비용이 지배한다.
            rs[i] = float(t[cand.top_k(s, self.k)].min()) / best
        return geomean(rs)


class FitWarning(UserWarning):
    """적합이 이상하다. **기각은 아니지만 조용히 넘기지 않는다** (D-54)."""


def fit_weights(score_fn: ScoreFn, matrix: FeatureMatrix, table: PerfTable,
                split: Split, w0: Sequence[float], *,
                method: str = "nelder-mead", max_evals: int = 200,
                k: int = 1, val_split: Split | None = None,
                n_restarts: int = 4,
                warn_invariants: bool = True,
                polish: bool = True,          # ★ D-55/D-56, 기본 켜짐
                polish_budget: int = 600,   # ★ 적합 305 의 2배 이내 (D-59)
                sensitivity_delta: float = 0.5,
                objective: str = "regret",    # ★ D-128: 다시 regret
                rank_top_k: int = 100,
                rank_lambda: float = 0.0,
                init_objective: str | None = None,
                init_evals: int = 0,
                bounds: list | None = None) -> FittedRule:
    """구조를 고정하고 가중치만 맞춘다.

    `split` 은 **`role="train"` 이어야 한다.** 검증/최종이 목적함수에
    들어가는 경로는 없다 (§29.7).

    `val_split` 은 적합이 끝난 뒤 **보고용으로만** 채점된다. 목적함수에
    관여하지 않는다 — 격차(`FittedRule.gap`)를 라운드마다 기록하기 위한 것이다.

    ## ★ `objective` — 기본은 `"regret"` 다 (D-128 에서 되돌렸다)

    ```
    "rank"    ★ 기본. 참 상위 `rank_top_k` 안의 가중 쌍 손실.
              미분 가능 -> L-BFGS-B
    "regret"  argmin 하나의 상대 시간. 계단 함수라 Nelder-Mead + 재시작
    ```

    ### 기본이 두 번 바뀌었다 — 지금은 `"regret"` 이다

    ```
    ~2026-09-01   "regret"   지금까지의 모든 결과가 통과한 경로
     2026-09-01   "rank"     그때 하는 실험이 순위 손실이었다 (D-101)
    ★2026-09-04   "regret"   순위 손실은 **틀린 목적함수**로 결론났다
                             (D-118·D-121). 진화 경로에서 뺀다 (D-128)
    ```

    ⚠️ `"rank"` 는 **함수로는 남는다** — `rank_loss` / `rank_loss_top1` /
    `tau` 는 지표로 쓰고, 옛 실행을 재현하려면 명시해서 부르면 된다.
    **진화 루프는 거부한다** (`LoopConfig`).

    `"rank"` 에서도 `fit_regret` 은 계속 `regret` 으로 계산해 기록한다 —
    **채점 기준은 안 바꾼다** (실험 계획서 `rank-evo-prereg.md` §3).

    ## `init_objective` — 대리 손실로 **초기점만** 만든다

    ```
    init_objective="rank_top1"   1단계: rank_loss_top1 로 L-BFGS-B
                                 (`init_evals` 회). 미분 가능하다
                                 2단계: 그 점에서 **regret 으로** 본 적합
    ```

    ⚠️ **채택은 regret 이다.** 1단계 값은 반환값에 남지 않고 `best_v` /
    `fit_regret` / 다듬기 전부가 참 목적함수로 다시 잰다. 조건이 섞이므로
    쓸 때 실험 계획서에 명시한다 — "초기점 생성에만 쓰고 채택은 regret@1"
    (`fitter-regret-prereg.md` §2).

    `objective="regret"` 에서만 받는다. 순위 손실 경로에서 순위 손실로
    초기점을 만드는 것은 같은 목적함수를 두 번 부르는 것이다.
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
    # ★ 가중치별 경계 (D-112). `None` 이면 아무것도 안 바뀐다 — 지수 자리를
    #   안 쓰는 규칙은 옛 실행과 **같은 조건**이어야 한다 (원칙 36).
    if bounds is None:
        def _proj(x):
            return x
    else:
        if len(bounds) != w0.size:
            raise FitError(
                f"bounds 길이 {len(bounds)} != 가중치 {w0.size}")
        _blo = np.array([b[0] for b in bounds], dtype=np.float64)
        _bhi = np.array([b[1] for b in bounds], dtype=np.float64)

        def _proj(x):
            return np.clip(x, _blo, _bhi)

        w0 = _proj(w0)
    if w0.ndim != 1 or w0.size == 0:
        raise FitError(f"W0 형태가 잘못됐다: {w0.shape}")
    if not np.all(np.isfinite(w0)):
        raise FitError("W0 에 비유한 값이 있다.")

    t0 = time.perf_counter()
    prob = _Problem(matrix, table, split.shapes, k)
    n_eval = 0
    n_inf = 0

    # ★ **본 것 중 최선**을 직접 붙든다 (D-55). 최적화기의 `res.x` 만 받으면
    #   탐색 도중 들른 더 좋은 점이 버려진다 — 목적함수가 계단이라 심플렉스가
    #   좋은 꼭짓점을 밟고 지나쳐도 수축은 다른 곳에서 끝난다. 24회 중 5회가
    #   그랬고, 최대 0.0277 을 버렸다. 평가는 이미 지불했으니 공짜 회수다.
    seen_v = float("inf")
    seen_w = w0.copy()

    if objective not in ("regret", "rank"):
        raise FitError(f"알 수 없는 목적함수: {objective!r}. "
                       "regret | rank 중 하나여야 한다.")
    if rank_lambda < 0:
        raise FitError(f"rank_lambda 는 0 이상이어야 한다: {rank_lambda}")
    if objective != "rank" and rank_lambda:
        raise FitError(
            f"rank_lambda={rank_lambda} 인데 objective={objective!r} 다. "
            "람다는 순위 손실 위에만 얹는다 (D-109).")
    if objective == "rank":
        prob.build_pairs(table, rank_top_k)
        if prob.n_pairs == 0:
            raise FitError(
                "순위 손실에 쓸 쌍이 하나도 없다 — 노이즈 바닥으로 가를 "
                "수 있는 쌍이 상위 "
                f"{rank_top_k} 안에 없다. 조용히 진행하지 않는다.")
        if rank_lambda:
            prob.build_top1_pairs(table, rank_top_k)
            if prob.n_pairs1 == 0:
                raise FitError(
                    "1등 쌍이 하나도 없다 — 참 1등을 노이즈 바닥으로 "
                    f"2등과 못 가른다 (상위 {rank_top_k}). "
                    "조용히 진행하지 않는다.")

    def value_at(w: np.ndarray) -> float:
        """★ **적합하는 목적함수**의 값. 세지 않는다.

        ⚠️ 다듬기도 이것을 쓴다 (D-122). 예전에는 `_polish` 안에
        `prob.regret` 이 박혀 있어서, `objective="rank"` 일 때 **순위 손실
        기준값과 regret 을 견주고** 있었다 — regret(1.2) > 순위 손실(0.24)
        이라 어떤 걸음도 채택되지 않아 다듬기가 조용히 아무것도 안 했다.
        목적함수 값 함수를 **한 곳에만** 둔다 (원칙 2).
        """
        # ★ **경계 안으로 접어서** 잰다. Nelder-Mead 는 경계를 모르고
        #   다듬기도 마찬가지라, 여기서 접어야 한 곳에서만 강제된다.
        wa = _proj(np.asarray(w, dtype=np.float64))
        if objective == "regret":
            return prob.regret(score_fn, wa)
        v = prob.rank_loss(score_fn, wa)
        if rank_lambda and np.isfinite(v):
            v += rank_lambda * prob.rank_loss_top1(score_fn, wa)
        return v

    def obj(w: np.ndarray) -> float:
        nonlocal n_eval, n_inf, seen_v, seen_w
        n_eval += 1
        wa = _proj(np.asarray(w, dtype=np.float64))
        v = value_at(wa)
        if not np.isfinite(v):
            n_inf += 1
            return 1e6      # 이 가중치는 실행 불가. 구조 기각은 아니다.
        if v < seen_v:
            seen_v, seen_w = v, wa.copy()
        return v

    m = method.lower()
    if m not in ("nelder-mead", "neldermead", "powell", "cma"):
        raise FitError(f"알 수 없는 최적화기: {method!r}. "
                       "nelder-mead | powell | cma 중 하나여야 한다.")

    # ★ 대리 손실로 **초기점만** 만든다. 여기서 나온 값은 아무 데도 남지
    #   않는다 — 아래 `best_v` 부터 전부 참 목적함수로 다시 잰다.
    n_init = 0
    start0 = w0.copy()
    if init_objective is not None:
        if init_objective != "rank_top1":
            raise FitError(
                f"알 수 없는 초기점 목적함수: {init_objective!r}. "
                "rank_top1 뿐이다.")
        if objective != "regret":
            raise FitError(
                f"init_objective 는 regret 경로에만 쓴다 "
                f"(objective={objective!r}). 순위 손실로 적합하면서 순위 "
                "손실로 초기점을 만드는 것은 같은 것을 두 번 부르는 것이다.")
        if init_evals <= 0:
            raise FitError(
                f"init_objective={init_objective!r} 인데 init_evals="
                f"{init_evals} 다. 예산을 명시해야 한다 — 초기점 단계도 "
                "evals 예산을 쓴다 (팔 비교의 공정성).")
        prob.build_top1_pairs(table, rank_top_k)
        if prob.n_pairs1 == 0:
            raise FitError(
                "1등 쌍이 하나도 없다 — 참 1등을 노이즈 바닥으로 2등과 "
                f"못 가른다 (상위 {rank_top_k}). 조용히 진행하지 않는다.")
        seen_s = float("inf")
        seen_sw = w0.copy()

        def _sobj(x: np.ndarray) -> float:
            nonlocal n_init, seen_s, seen_sw
            n_init += 1
            xa = _proj(np.asarray(x, dtype=np.float64))
            v = prob.rank_loss_top1(score_fn, xa)
            if not np.isfinite(v):
                return 1e6
            if v < seen_s:
                seen_s, seen_sw = v, xa.copy()
            return v

        from scipy.optimize import minimize as _min_init
        _min_init(_sobj, w0, method="L-BFGS-B", bounds=bounds,
                  options={"maxiter": 500, "maxfun": init_evals})
        if np.isfinite(seen_s):
            start0 = _proj(seen_sw)

    # ★ 재시작이 필요하다. 목적함수가 **계단 함수**라 단순 심플렉스는
    #   평평한 지대에서 수축해 멈춘다 (§29.2). 실제로 초기값에서 한 발도
    #   못 움직이는 경우가 나온다. 매 재시작마다 심플렉스를 다시 부풀리고,
    #   몇 개는 무작위 출발점에서 시작한다. 시드가 고정이라 결정론적이다.
    best_w = start0.copy()
    best_v = obj(best_w)
    #: 실제로 시작한 재시작 횟수. 아래에서 **세어서 경고한다**.
    n_started = 0
    if objective == "rank":
        # ★ 계단이 아니므로 준뉴턴을 먼저 쓴다. 이것이 이 설계의 이점이다.
        #
        # ⚠️ **예산을 나눠 준다.** 처음에는 `maxfun=max_evals` 로 뒀는데,
        #    실측에서 L-BFGS 혼자 209/200 을 써서 **재시작이 한 번도 안
        #    돌았다.** "재시작은 그대로 둔다" 가 거짓이 되고, 전역 탐색
        #    없이 국소해 하나로 끝난다 (원칙 1 — 장치가 안 돈다).
        from scipy.optimize import minimize as _min
        for r in range(n_restarts):
            if n_eval >= max_evals:
                break
            n_started += 1
            start = (best_w if r == 0 else best_w + np.random.default_rng(
                _RESTART_SEED + r).normal(
                    0.0, 0.35 * np.maximum(np.abs(best_w), 1.0)))
            _min(obj, _proj(start), method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 500,
                          "maxfun": max(20, max_evals // n_restarts)})
            if seen_v < best_v:
                best_v, best_w = seen_v, seen_w.copy()
    rng = np.random.default_rng(_RESTART_SEED)
    per = max(20, max_evals // max(1, n_restarts))
    # ★ **마지막 재시작이 예산을 다 쓰면서도 아직 나아지고 있었는가.**
    #   "예산을 다 썼다" 자체는 경고가 아니다 — 재시작 일정이 `max_evals`
    #   를 **설계상 전부 쓰게** 돼 있어 언제나 참이다(원칙 11). 신호는
    #   "잘리는 순간까지 개선 중이었다" 쪽이다.
    cut_while_improving = False
    for r in range(n_restarts):
        n_started += 1
        if n_eval >= max_evals:
            break
        before_v, before_n = best_v, n_eval
        start = best_w if r == 0 else best_w + rng.normal(
            0.0, 0.35 * np.maximum(np.abs(best_w), 1.0))
        res = _minimize_once(obj, start, m, per, r, bounds=bounds)
        obj(np.asarray(res.x, dtype=np.float64))
        if seen_v < best_v:                 # ★ res.x 가 아니라 '본 것 중 최선'
            best_v, best_w = seen_v, seen_w.copy()
        cut_while_improving = (best_v < before_v - 1e-12
                               and n_eval - before_n >= per)

    # ★ 재시작이 **실제로 돌았는가** (2026-09-01). `n_restarts=4` 라고
    #   적어 놓고 1회만 도는 일이 실제로 있었다 — rank 경로에서 L-BFGS
    #   가 `maxfun=max_evals` 로 예산을 혼자 다 썼다. 주석은 "재시작은
    #   그대로 둔다" 였고 **거짓이었다** (원칙 1).
    if n_restarts > 1 and n_started < 2:
        warnings.warn(
            f"fit_weights: 재시작이 {n_started}회만 돌았다 "
            f"(n_restarts={n_restarts}). 첫 최적화가 예산 "
            f"{max_evals} 을 혼자 쓴다 — 전역 탐색이 없다.",
            FitWarning, stacklevel=2)
    if n_inf >= n_eval:
        raise FitError(
            f"모든 가중치에서 규칙이 유효한 점수를 내지 못했다 "
            f"({n_inf}/{n_eval}). 구조를 기각한다.")

    n_fit = n_eval          # ★ 다듬기 전. 상한 판정은 이 값으로 한다.
    w = _proj(best_w)
    if polish:
        w, best_v, n_pol = _polish(prob, score_fn, w, best_v, polish_budget,
                                   value=value_at)
        n_eval += n_pol
        # ★ 다듬기는 경계를 모른다. **여기서 한 번 더 접는다** — 접힌
        #   값으로 아래 regret/민감도를 다시 재므로 보고값과 반환값이
        #   같은 가중치에서 나온다.
        w = _proj(w)
    # ★ 채점 기준은 언제나 regret 이다 — `objective="rank"` 여도 그렇다.
    #   "채점은 regret, 학습은 순위 손실" (rank-evo-prereg.md §3)
    fit_regret = float(prob.regret(score_fn, w))
    if not np.isfinite(fit_regret) or fit_regret >= 1e6:
        raise FitError("적합된 가중치에서 regret 이 유한하지 않다. 기각한다.")

    sens = _sensitivity(prob, score_fn, w, fit_regret, sensitivity_delta)
    contrib = _contributions(prob, score_fn, w)

    val = float("nan")
    if val_split is not None:
        vp = _Problem(matrix, table, val_split.shapes, k)
        val = float(vp.regret(score_fn, w))

    # ★ 두 가지를 고쳤다 (D-76).
    #   (1) 다듬기 평가를 뺀 `n_fit` 으로 견준다. `n_eval` 로 견주면 다듬기
    #       예산 600 이 상한 300 을 언제나 넘는다.
    #   (2) 예산 소진만으로는 경고하지 않는다 — 재시작 일정이 예산을 전부
    #       쓰게 돼 있어 그것도 언제나 참이다. **잘리는 순간까지 개선 중**
    #       이었을 때만 경고한다. 그래야 "수렴 전 중단" 이 사실이 된다.
    hit_cap = n_fit >= max_evals and cut_while_improving
    out = FittedRule(w=w, w0=w0, fit_regret=fit_regret,
                     # ★ 예산 비교는 이 값으로 한다 — 초기점 단계도 센다.
                     n_evals=n_eval + n_init,
                     n_infeasible=n_inf, sensitivity=sens,
                     seconds=time.perf_counter() - t0, val_regret=val,
                     method=m, contrib=contrib, n_fit_evals=n_fit,
                     n_init_evals=n_init)
    if warn_invariants:
        msgs = out.invariants()
        if hit_cap:
            # 상한을 조금 넘길 수 있다 — 재시작 진입 전에만 검사하고
            # `obj(res.x)` 가 몇 번 더 불린다. 넘긴 양이 아니라 **닿았다는
            # 사실**이 신호다: 예산을 다 쓸 때까지 개선을 찾고 있었다.
            msgs.append(f"평가 상한에서 잘렸는데 아직 나아지고 있었다 "
                        f"({n_fit}/{max_evals}, 다듬기 포함 총 {n_eval}회) "
                        "— 예산을 늘리면 더 좋아질 수 있다")
        for msg in msgs:
            warnings.warn(f"fit_weights: {msg}", FitWarning, stacklevel=2)
    return out


#: 재시작 출발점의 시드. 고정이라 `fit_weights` 는 결정론적이다.
_RESTART_SEED = 20260820

#: 초기 심플렉스 스텝 = 이 값 x |start|. 재시작마다 절반이 된다.
#: ★ 기본 0.6 에서 24회 중 13회가 **한 발짝도 못 움직였다** (D-54).
SIMPLEX_SCALE = 0.6
#: 0 보다 크면 **절대 스텝**을 쓴다 (|start| 무시). 지배적 항 포화를 시험한다.
SIMPLEX_ABS = 0.0

#: 한 항의 실효 기여도가 **다른 항 중앙값의 이 배수**를 넘으면 경고한다.
#: 절대 배율(100배)을 쓰면 F1 라이브러리에서 상시 발화한다 (D-70).
_DOMINANCE = 50.0

#: 좌표 다듬기의 스텝 배율 (`polish=True` 일 때). 큰 것부터 훑고 줄인다.
#: 좌표 다듬기의 스텝 배율. **로그 스케일로 넓게** 훑는다 (D-59).
#:
#: 처음에는 (0.5, 0.25, 0.1) 이었다 — 전부 1보다 작아서 **한 좌표를 크게
#: 흔드는 시도가 아예 없었다.** 도달 실패 6건의 격차가 0.0085~0.0308 인데,
#: 계단 함수에서 그 정도 격차는 좌표 하나를 배로 키우거나 반으로 줄이면
#: 넘어갈 수 있다. 큰 것부터 훑어 큰 계단을 먼저 넘고, 작은 것으로 다듬는다.
_POLISH_DELTAS = (10.0, 3.0, 1.0, 0.5, 0.25, 0.1)

#: 쌍좌표 시도의 부호 조합. 단일 좌표로 못 넘는 계단이 있을 수 있다 —
#: 두 항이 서로를 상쇄해 순위가 안 바뀌는 경우가 그렇다.
_PAIR_SIGNS = ((+1.0, +1.0), (+1.0, -1.0), (-1.0, +1.0), (-1.0, -1.0))


def _polish(prob, score_fn: ScoreFn, w: np.ndarray, base: float,
            budget: int, *, pairs: bool = True, value=None
            ) -> tuple[np.ndarray, float, int]:
    """★ 좌표 하강으로 다듬는다 (D-55, D-59 강화).

    Nelder-Mead 가 멈춘 점은 **좌표 방향으로도 국소 최적이 아니었다** — 24회 중
    9회가 좌표 하나를 흔드는 것만으로 나아졌고, 최대 0.0513 이었다. 목적함수가
    계단이라 심플렉스는 평평한 지대에서 수축해 멈추는데, 축 방향의 계단 하나를
    넘으면 값이 떨어진다.

    세 가지를 한다.

    ```
    1  단일 좌표 x delta      delta 는 10 ~ 0.1 로그 스케일 (D-59)
    2  쌍좌표                 1이 한 바퀴 아무것도 못 찾았을 때만
    3  반복                   개선되면 그 delta 로 다시 한 바퀴
    ```

    쌍좌표는 **단일 좌표가 막혔을 때만** 켠다. `n^2 x 4` 라 비싸고, 단일
    좌표로 풀리는 동안 쓰면 예산만 먹는다.

    ⚠️ **훈련 형상만 본다** — `prob` 이 학습 분할로 만들어진다. 홀드아웃이
    들어오는 인자가 없고 `test_polish_only_sees_the_training_split` 이
    그것을 고정한다 (§29.7).

    ## ★ `value` — 적합하는 목적함수를 받는다 (D-122)

    `None` 이면 `prob.regret` 이다. **`objective="rank"` 로 부를 때는
    반드시 넘겨야 한다** — 안 넘기면 순위 손실 기준값과 regret 을 견주게
    되고, 값의 자릿수가 달라 어떤 걸음도 채택되지 않는다. 그 상태로
    D-101~D-112 의 순위 손실 실행 전부가 **다듬기 없이** 돌았다.
    """
    val = value if value is not None else (
        lambda t: prob.regret(score_fn, t))
    w = w.copy()
    n_ev = 0

    def try_step(t: np.ndarray) -> bool:
        nonlocal base, w, n_ev
        v = val(t)
        n_ev += 1
        if np.isfinite(v) and v < base - 1e-12:
            base, w = v, t
            return True
        return False

    n = len(w)
    for d in _POLISH_DELTAS:
        improved = True
        while improved and n_ev < budget:
            improved = False
            for i in range(n):
                for sgn in (+1.0, -1.0):
                    t = w.copy()
                    t[i] = t[i] + sgn * d * max(abs(t[i]), 1.0)
                    improved |= try_step(t)
        # ★ 단일 좌표가 이 delta 에서 막혔다. 쌍좌표로 한 바퀴 돈다.
        if not pairs or n_ev >= budget or n < 2:
            continue
        for i in range(n - 1):
            for j in range(i + 1, n):
                for si, sj in _PAIR_SIGNS:
                    if n_ev >= budget:
                        break
                    t = w.copy()
                    t[i] = t[i] + si * d * max(abs(t[i]), 1.0)
                    t[j] = t[j] + sj * d * max(abs(t[j]), 1.0)
                    try_step(t)
    return w, base, n_ev


def _cma_once(obj, start: np.ndarray, budget: int, r: int, *,
              bounds: list | None = None):
    """CMA-ES 한 번 (D-123). **초기 스텝을 Nelder-Mead 와 같게 준다.**

    `sigma0=1.0` 에 `CMA_stds` 로 좌표별 스텝을 주면 `_minimize_once` 의
    심플렉스 스텝과 같은 크기에서 출발한다 — 팔 비교에서 스텝 크기가
    교락이 되지 않는다.

    ⚠️ **예산을 정확히 맞추지 못한다.** 세대 단위로 끝나므로 몇 회
    넘긴다 (16차원 popsize 12 -> 최대 11회). 실제 평가 수를 보고한다
    (`n_evals`) — 예산이 같았다고 가정하지 않는다 (원칙 38).
    """
    try:
        import cma as _cma
    except ImportError as e:      # pragma: no cover - 선택 의존성
        raise FitError(
            "method='cma' 인데 `cma` 패키지가 없다. "
            "`pip install cma` (pyproject 의 `fit` 추가 그룹). "
            "조용히 다른 최적화기로 넘어가지 않는다.") from e

    step = SIMPLEX_SCALE * (0.5 ** r) * np.maximum(np.abs(start), 1.0)
    opts = {"CMA_stds": [float(x) for x in step], "maxfevals": int(budget),
            "verbose": -9, "seed": int(_RESTART_SEED + r), "verb_log": 0,
            "verb_disp": 0}
    if bounds is not None:
        opts["bounds"] = [[float(b[0]) for b in bounds],
                          [float(b[1]) for b in bounds]]
    es = _cma.CMAEvolutionStrategy([float(x) for x in start], 1.0, opts)
    es.optimize(obj)
    xb = es.result.xbest
    x = np.asarray(xb if xb is not None else start, dtype=np.float64)
    return _Res(x)


@dataclass(frozen=True, slots=True)
class _Res:
    """`scipy` 결과 객체의 자리를 메운다 — 부르는 쪽은 `.x` 만 본다."""

    x: np.ndarray


def _minimize_once(obj, start, method: str, budget: int, r: int, *,
                   bounds: list | None = None):
    """재시작 한 번. 심플렉스를 **다시 부풀려서** 시작한다."""
    from scipy.optimize import minimize

    start = np.asarray(start, dtype=np.float64)
    if method == "cma":
        return _cma_once(obj, start, budget, r, bounds=bounds)
    if method == "powell":
        return minimize(obj, start, method="Powell",
                        options={"maxfev": budget, "xtol": 1e-4, "ftol": 1e-6})
    n = len(start)
    # ★ 스텝 크기. `SIMPLEX_SCALE` 로 쓸어볼 수 있게 뺐다 (D-54).
    #   상대 스텝(|start| 비례)은 **지배적 항을 덜 흔든다** — 큰 항이 이미
    #   순위를 포화시켰으면 더 흔들어도 순위가 안 바뀐다. `SIMPLEX_ABS` 는
    #   그것을 시험한다 (절대 스텝).
    if SIMPLEX_ABS > 0.0:
        step = np.full(n, SIMPLEX_ABS * (0.5 ** r))
    else:
        step = SIMPLEX_SCALE * (0.5 ** r) * np.maximum(np.abs(start), 1.0)
    simplex = np.tile(start, (n + 1, 1))
    for i in range(n):
        simplex[i + 1, i] += step[i]
    return minimize(obj, start, method="Nelder-Mead",
                    options={"maxfev": budget, "xatol": 1e-4, "fatol": 1e-9,
                             "adaptive": True, "initial_simplex": simplex})


def _contributions(prob: _Problem, score_fn: ScoreFn,
                   w: np.ndarray) -> np.ndarray | None:
    """항별 **실효 기여도** = |w_i| x (그 항이 점수에 만드는 산포) (D-70).

    `w_i` 하나만 0 으로 두고 점수를 다시 계산해, 원래 점수와의 차이가
    형상 안에서 얼마나 흩어지는지를 잰다. **형상 안의 산포만 본다** —
    형상 상수는 순위를 안 바꾸므로 기여도가 0 이어야 한다 (절대 규칙 2).

    ★ **시간을 안 본다.** `score_fn` 만 부르고 `prob.regret` 은 안 부른다 —
    정답이 들어오는 경로가 없다 (§3).

    절대 배율(|w|/|w0|)과 달리 **피처 스케일에 불변**이라, 라이브러리를
    바꿔도 같은 기준으로 읽힌다. F1(피처 [0,0.2])과 사람 24개
    (피처 [0,300])에서 |w| 자릿수가 셋 이상 달라 절대 기준이 무의미했다.
    """
    out = np.zeros(len(w), dtype=np.float64)
    n = 0
    # ★ `items` 는 `(feats, info, cand, times, best)` 다. 뒤의 둘은
    #   **정답**이므로 이름을 `_` 로 받아 손댈 수 없게 한다 (§3).
    for f, info, _cand, _times, _best in prob.items:
        try:
            base = np.asarray(score_fn(f, info, prob.hw, w),
                              dtype=np.float64)
        except Exception:                                   # noqa: BLE001
            return None
        if base.ndim != 1 or base.size < 2:
            continue
        for i in range(len(w)):
            w2 = np.asarray(w, dtype=np.float64).copy()
            w2[i] = 0.0
            try:
                d = base - np.asarray(
                    score_fn(f, info, prob.hw, w2), dtype=np.float64)
            except Exception:                               # noqa: BLE001
                return None
            if np.isfinite(d).all():
                out[i] += float(np.std(d))
        n += 1
    return out / max(n, 1) if n else None


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
