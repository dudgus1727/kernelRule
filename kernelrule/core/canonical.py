"""Final scoring — **it uses the loop's own split** (§10.2 / D-36).

## Why a separate module

The scorer and the loop were **each deciding the split on their own.** The
scorer took every third shape per regime out of all 61 and called that the
"holdout", and 11 of those 19 shapes (58%) were the loop's training shapes.

    The structure had evolved while looking at those shapes. Only the
    **weights** were held out.

In that state "the gate was passed" was written, when it was actually
indistinguishable (see the correction in
`docs/artifacts/feature-descriptions.md`).

**The fix: remove the path that picks shapes arbitrarily.** This function
only runs when given a `SplitSet`, and overlap raises.

## A holdout of what (D-36)

"Holdout" is not one thing. Each stage learns something different.

    structure   evolved by RuleEditor/Analyst  -> measured on `splits.val` only
    weights     fitted by fit_weights          -> within whichever split
    prompt      edited by a human              -> `splits.test` only (§10.2)

This function produces the **structure holdout**: ⛔ it **fits nothing**. It
takes the rule and the weights the loop ended with and reads them on all of
`splits.val`.

★ 2026-09-11 (D-166): ~~The loop used val only for the early-stop
decision~~ — **the loop does not look at val at all.** Early stopping was
turned off at D-132 and the path was sealed at D-144: `should_stop` returns
`(False, "")` always and raises if `patience > 0`. Acceptance, parent
selection and the objective switch all read the training score. So the
structure is not fitted to these shapes by any route — the old sentence
understated the guarantee, and understating a guarantee is its own kind of
wrong (principle 9).

```
the representative runs   F3rw-p8-nan, 6 seeds   patience = 0 (all six)
                          -> none of them stopped on val
⚠️ runs from before D-132 (2026-09-04) may have had an early stop that did
   read val. **Which runs those are has not been checked** — do not assume
   (principle 39).
```

## ⛔ 2026-09-18 (D-182) — ★ 채점은 적합하지 않는다

채점은 **완성품을 재는 자리**다. 거기서 가중치를 다시 맞추면 재는 것이 아니다.

```
★ 루프가 ★ 그 학습 분할에서 이미 가중치를 맞췄다
⛔ 그런데 채점기가 ★ 같은 형상으로 ★ 300회를 다시 돌리고 있었다
⛔ 루프의 답은 ★ 출발점(w0)으로만 쓰였다
```

**왜 그렇게 됐나.** D-69 가 체제별 적합을 넣으면서, 루프가 만든 **한 벌**을
채점기가 `short`/`long` **두 벌**로 다시 만들었다 — ★ 채점이 루프가 안 한 일을
대신 해준 것이다. D-179 가 두 벌을 한 벌로 줄이면서 **적합 호출은 남겼다.**

★ 이제 `fit_weights` 호출이 없다. 받은 `w` 로 홀드아웃을 잴 뿐이다.

## ⛔ 2026-09-17 (D-179) — ★ 체제를 나누지 않는다

예전에는 `regime_of(axis="size")` 로 학습 분할을 `short`/`long` 으로 갈라
**가중치를 두 벌** 맞추고 홀드아웃도 갈라 채점했다. 넷이 잘못이었다.

```
⛔ 경계 0.5ms 를 ★ 우리가 표를 보고 골랐다 (§10.1 "경계 탐색")
⛔ 그 경계로 가중치를 두 벌 쓴다 — ★ 벤더에는 그 자유도가 없다
⛔ ★ 루프는 한 벌로 진화했는데 채점은 두 벌이었다
   -> 최적화한 목적과 보고한 값이 ★ 다른 함수였다
⛔ 0.5 는 ★ a6000 을 보고 고른 값인데 네 표에 그대로 썼다
   빠른 GPU 일수록 SOL 이 작아져 ★ long 이 줄어든다
   (실측: h100 fold1·fold3 의 학습 long 이 ★ 8개 — MIN_PER_REGIME 과 같다)
```

★ **이제 학습 분할 전체로 가중치 한 벌을 맞추고 홀드아웃 전체를 그 한 벌로
채점한다.** 루프가 최적화한 것과 보고하는 값이 같은 함수다.

그 SOL 값을 계산하던 함수는 코드에서 **완전히 지웠다** — `regime_of` 의
`axis="size"` 분기도, 그것으로 자르던 분할 함수도 함께 없앴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from kernelrule.core.scoring import Evaluation, evaluate_scores, geomean
from kernelrule.core.splits import SplitError, SplitSet, regime_of
from kernelrule.core.table import PerfTable

__all__ = ["CanonicalScore", "canonical_score"]

#: ⛔ 2026-09-18 (D-182): `MIN_TRAIN_SHAPES` 를 없앴다.
#:
#: D-179 가 체제별 문턱(`MIN_PER_REGIME`)을 없애면서 "학습 형상이 너무 적으면
#: 조용히 넘어가지 않는다" 를 분할 전체에 옮겨 놨었다. ★ 그런데 채점이 더는
#: 학습 분할에서 적합하지 않으므로 **여기서 할 말이 아니다** — 가중치가 믿을
#: 만한지는 ★ 그것을 만든 루프가 판정할 일이다 (`fit_weights` 의 경고).


@dataclass(frozen=True, slots=True)
class CanonicalScore:
    """The structure-holdout score. **It carries what was not looked at**
    (D-36)."""

    #: Geometric mean over `splits.val`. ★ This is the value to report.
    holdout: float
    #: Geometric mean over `splits.train`, read with the **same** weights.
    #: ★ For reference — the structure was evolved here, so
    #: `holdout - in_sample` is the structure's own generalisation gap.
    #: ⛔ 2026-09-18 (D-182): it is no longer "the score the fit reached" —
    #: nothing is fitted here.
    in_sample: float
    #: ⚠️ ★ **읽기용** roofline 쪼개기 (`mem`/`comp`). ⛔ 적합을 나누는 근거가
    #: 아니다 — 가중치는 한 벌이고 이 값은 그 한 벌의 결과를 갈라 본 것뿐이다.
    #: 예전의 `short`/`long`(SOL 축)은 D-179 에서 사라졌다.
    by_regime: dict[str, float]
    #: The holdout evaluation. Used as is for significance (`compare`).
    evaluation: Evaluation
    #: ★ The weights **as scored** — since D-182 these are the ones handed
    #: in, not a refit. Kept so that exporting a rule writes the numbers it
    #: was actually scored with; a file without them does not reproduce.
    #: ⛔ D-179: 키는 `"all"` 하나다 (예전에는 `short`/`long` 두 벌).
    weights: dict[str, list[float]] = field(default_factory=dict)
    n_holdout: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def line(self) -> str:
        r = "  ".join(f"{k} {v:.4f}" for k, v in sorted(self.by_regime.items()))
        return (f"structHO {self.holdout:.4f} (n={self.n_holdout})  "
                f"in-sample {self.in_sample:.4f}  [{r}]")


def canonical_score(code: str, w, *, table: PerfTable, matrix,
                    splits: SplitSet) -> CanonicalScore:
    """★ Score a finished rule on the holdout. ⛔ **It fits nothing.**

    ```
    받는 것   규칙 코드 + ★ 루프가 끝낸 가중치
    하는 일   그 가중치로 ★ splits.val 전체를 채점한다
              참고로 splits.train 도 ★ 같은 가중치로 읽는다
    ⛔ 안 하는 일  적합. `fit_weights` 를 부르지 않는다
    ```

    ★ It cannot be called without `splits`. There is no arbitrary-split
    path — that is what D-36 fixed.

    ⚠️ **이름은 그대로 둔다.** `canonical_score` 로 낸 수치가 결정 기록과
    산출물 수십 곳에 있고, 이름을 바꾸면 그 인용이 전부 미아가 된다. ★ 바뀐
    것은 절차이고, 그 절차는 이 docstring 과 D-182 에 적혀 있다.

    ⛔ 2026-09-18 (D-182): `max_evals` 인자를 없앴다 — 적합이 없으니 쓸 데가
    없다. 넘기던 호출자는 없었다.
    """
    if not isinstance(splits, SplitSet):
        raise SplitError(
            "final scoring must receive the loop's SplitSet (D-36). Picking "
            "shapes separately makes the structure holdout overlap the "
            "training shapes — 11 of 19 actually did, and 'the gate was "
            "passed' was reported wrongly.")

    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import make_score_of

    train = list(splits.train.shapes)
    val = list(splits.val.shapes)
    if not val:
        raise SplitError("the validation split is empty. No structure "
                         "holdout can be produced.")

    fn = compile_rule(code)
    warns: list[str] = []

    # ★ D-182 — ⛔ no fitting. The weights are the loop's answer, used as
    #   they are. `w` used to be only a starting point for a 300-evaluation
    #   refit on the very shapes the loop had already fitted on.
    w = np.asarray(w, float)
    so = make_score_of(fn, matrix, w)
    e_ho = evaluate_scores(so, table, val, ks=(1,))
    reg_ho = {p: e_ho.regret[i, 0] for i, p in enumerate(e_ho.shapes)}
    # ★ The training shapes, read with the **same** weights. ⛔ Not a fit —
    #   it is what makes `holdout - in_sample` the structure's own gap
    #   instead of a gap between two different weight vectors.
    e_tr = evaluate_scores(so, table, train, ks=(1,))
    reg_tr = {p: e_tr.regret[i, 0] for i, p in enumerate(e_tr.shapes)}

    scored = [p for p in val if p in reg_ho]
    if not scored:
        raise SplitError("not a single holdout shape could be scored "
                         "(§26.4).")
    if len(scored) < len(val):
        warns.append(f"only {len(scored)} of {len(val)} holdout shapes were "
                     "scored")

    # ⚠️ ★ 읽기용일 뿐이다 (D-179 §2-3). ⛔ 우리가 고른 경계가 아니라
    #    roofline ridge point 다.
    by_regime = {}
    for name in ("mem", "comp"):
        v = [reg_ho[p] for p in scored
             if regime_of(p, table.hw, axis="roofline") == name]
        if v:
            by_regime[name] = geomean(np.array(v))

    # Packed as an `Evaluation` so significance testing can use it directly
    ev = replace(evaluate_scores(so, table, scored, ks=(1,)),
                 label="canonical")
    # ★ The weights **as scored** — the same ones that came in. Exporting a
    #   rule without them writes a file that does not reproduce.
    fitted: dict[str, list[float]] = {"all": [float(x) for x in w]}

    return CanonicalScore(
        holdout=geomean(np.array([reg_ho[p] for p in scored])),
        in_sample=geomean(np.array([reg_tr[p] for p in train if p in reg_tr])),
        by_regime=by_regime, weights=fitted, evaluation=ev,
        n_holdout=len(scored), warnings=tuple(warns))
