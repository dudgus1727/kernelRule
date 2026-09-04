"""최종 채점 — **루프의 분할을 그대로 쓴다** (§10.2 / D-36).

## 왜 별도 모듈인가

채점기와 루프가 분할을 **각자 정하고 있었다.** 채점기는 61형상 전체에서
체제별로 3개마다 1개를 뽑아 "홀드아웃" 이라고 불렀는데, 그 19형상 중
11개(58%)가 루프의 학습 형상이었다.

    구조는 그 형상들을 보고 진화했다. 홀드아웃인 것은 **가중치뿐**이었다.

그 상태로 "통과 조건을 넘었다" 고 썼고, 실제로는 구분 불가였다
(`docs/artifacts/feature-descriptions.md` 의 정정).

**고침: 임의로 형상을 뽑는 경로를 없앤다.** 이 함수는 `SplitSet` 을
받아야만 돌고, 겹치면 예외다.

## 무엇의 홀드아웃인가 (D-36)

"홀드아웃" 은 하나가 아니다. 단계마다 학습하는 것이 다르다.

    구조     RuleEditor/Analyst 가 진화시킨다  -> `splits.val` 로만 잰다
    가중치   fit_weights 가 맞춘다            -> 어떤 분할이든 그 안에서만
    프롬프트  사람이 고친다                    -> `splits.test` 로만 (§10.2)

이 함수는 **구조 홀드아웃**을 낸다: 가중치를 `splits.train` 에서 체제별로
적합하고, `splits.val` 에서 평가한다. 루프는 val 을 조기 종료 판정에만
썼으므로 구조가 그 형상에 맞춰지지 않았다.

## 체제 판정

`regime_of(axis="size")` — SOL 대리 지표다. **`t_best` 를 쓰지 않는다**
(§10.1). 배포 시점에 계산되지 않는 경계로 자르면 그 숫자는 오라클이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from kernelrule.core.scoring import Evaluation, evaluate_scores, geomean
from kernelrule.core.splits import Split, SplitError, SplitSet, regime_of
from kernelrule.core.table import PerfTable

__all__ = ["CanonicalScore", "canonical_score"]

#: 체제당 이 수 미만이면 그 체제의 적합을 믿을 수 없다 (§10.1).
MIN_PER_REGIME = 8


@dataclass(frozen=True, slots=True)
class CanonicalScore:
    """구조 홀드아웃 점수. **무엇을 안 봤는지 함께 들고 다닌다** (D-36)."""

    #: `splits.val` 기하평균. ★ 이것이 보고할 값이다.
    holdout: float
    #: `splits.train` 기하평균. 참고용 — 구조가 여기 맞춰졌다.
    in_sample: float
    #: 체제별 홀드아웃 기하평균.
    by_regime: dict[str, float]
    #: 홀드아웃 평가 결과. 유의성 판정(`compare`)에 그대로 쓴다.
    evaluation: Evaluation
    #: ★ 체제별로 **적합된** 가중치. 이것이 없으면 규칙을 파일로 내보낼 때
    #: 초기값을 적게 되고, 그 파일은 재현되지 않는다 — 파일이 거짓말을 한다.
    weights: dict[str, list[float]] = field(default_factory=dict)
    n_holdout: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def line(self) -> str:
        r = "  ".join(f"{k} {v:.4f}" for k, v in sorted(self.by_regime.items()))
        return (f"구조HO {self.holdout:.4f} (n={self.n_holdout})  "
                f"표본내 {self.in_sample:.4f}  [{r}]")


def canonical_score(code: str, w0, *, table: PerfTable, matrix,
                    splits: SplitSet, max_evals: int = 300) -> CanonicalScore:
    """★ `splits` 없이는 부를 수 없다. 임의 분할 경로를 두지 않는다.

    체제마다 `splits.train` 에서 가중치를 적합하고 `splits.val` 에서 잰다.
    """
    if not isinstance(splits, SplitSet):
        raise SplitError(
            "최종 채점은 루프의 SplitSet 을 받아야 한다 (D-36). 형상을 "
            "따로 뽑으면 구조 홀드아웃이 학습 형상과 겹친다 — 실제로 19 중 "
            "11 이 겹쳤고 '통과 조건 통과' 를 잘못 보고했다.")

    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import fit_weights, make_score_of

    train = list(splits.train.shapes)
    val = list(splits.val.shapes)
    if not val:
        raise SplitError("검증 분할이 비었다. 구조 홀드아웃을 낼 수 없다.")

    fn = compile_rule(code)
    warns: list[str] = []
    reg_tr: dict = {}
    reg_ho: dict = {}
    tol_ho: dict = {}
    fitted: dict[str, list[float]] = {}

    for name in ("short", "long"):
        g_tr = [p for p in train if regime_of(p, table.hw) == name]
        g_ho = [p for p in val if regime_of(p, table.hw) == name]
        if not g_tr:
            if g_ho:
                warns.append(f"체제 {name!r}: 학습 형상이 0개인데 홀드아웃에 "
                             f"{len(g_ho)}개 있다 — 그 형상은 채점할 수 없다")
            continue
        if len(g_tr) < MIN_PER_REGIME:
            warns.append(f"체제 {name!r}: 학습 형상 {len(g_tr)}개 < "
                         f"{MIN_PER_REGIME}. 그 체제의 가중치는 믿기 어렵다")
        fit = fit_weights(fn, matrix, table, Split("train", tuple(g_tr)),
                          w0, max_evals=max_evals,
                          # ★ **최종 채점은 언제나 regret 이다** (D-103).
                          #   `fit_weights` 의 기본값이 `rank` 로 바뀌었으므로
                          #   여기서 **명시**해야 한다. 안 하면 이 프로젝트의
                          #   모든 수치가 조용히 다른 것이 된다.
                          objective="regret")
        fitted[name] = [float(x) for x in fit.w]
        so = make_score_of(fn, matrix, fit.w)
        e_tr = evaluate_scores(so, table, g_tr, ks=(1,))
        for i, p in enumerate(e_tr.shapes):
            reg_tr[p] = e_tr.regret[i, 0]
        if not g_ho:
            continue
        e_ho = evaluate_scores(so, table, g_ho, ks=(1,))
        for i, p in enumerate(e_ho.shapes):
            reg_ho[p], tol_ho[p] = e_ho.regret[i, 0], e_ho.tol[i]

    scored = [p for p in val if p in reg_ho]
    if not scored:
        raise SplitError("홀드아웃 형상을 하나도 채점하지 못했다 (§26.4).")
    if len(scored) < len(val):
        warns.append(f"홀드아웃 {len(val)}개 중 {len(scored)}개만 채점됐다")

    by_regime = {}
    for name in ("short", "long"):
        v = [reg_ho[p] for p in scored if regime_of(p, table.hw) == name]
        if v:
            by_regime[name] = geomean(np.array(v))

    # 유의성 판정에 그대로 쓸 수 있게 `Evaluation` 형태로 담는다
    base = evaluate_scores(make_score_of(fn, matrix, np.asarray(w0, float)),
                           table, scored, ks=(1,))
    ev = replace(base,
                 regret=np.array([reg_ho[p] for p in scored]).reshape(-1, 1),
                 tol=np.array([tol_ho[p] for p in scored]), label="canonical")

    return CanonicalScore(
        holdout=geomean(np.array([reg_ho[p] for p in scored])),
        in_sample=geomean(np.array([reg_tr[p] for p in train if p in reg_tr])),
        by_regime=by_regime, weights=fitted, evaluation=ev,
        n_holdout=len(scored), warnings=tuple(warns))
