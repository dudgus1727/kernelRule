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

This function produces the **structure holdout**: ★ one weight vector is
fitted on **all** of `splits.train` and evaluated on **all** of
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
from kernelrule.core.splits import Split, SplitError, SplitSet, regime_of
from kernelrule.core.table import PerfTable

__all__ = ["CanonicalScore", "canonical_score"]

#: ⛔ 2026-09-17 (D-179): `MIN_PER_REGIME` 은 체제별 적합이 있을 때의 문턱이라
#: 없앴다. 그런데 "학습 형상이 너무 적으면 조용히 넘어가지 않는다" 는 보증은
#: 남겨야 해서, ★ 같은 수 8 을 **분할 전체**에 적용한다.
#:
#: ⚠️ 이것은 ★ 무엇을 자르는 경계가 아니다 — **경고만** 한다. 예전에는 체제마다
#: 8 을 요구했으므로 이 문턱은 그때보다 ★ 느슨하다.
MIN_TRAIN_SHAPES = 8


@dataclass(frozen=True, slots=True)
class CanonicalScore:
    """The structure-holdout score. **It carries what was not looked at**
    (D-36)."""

    #: Geometric mean over `splits.val`. ★ This is the value to report.
    holdout: float
    #: Geometric mean over `splits.train`. For reference — the structure was
    #: fitted here.
    in_sample: float
    #: ⚠️ ★ **읽기용** roofline 쪼개기 (`mem`/`comp`). ⛔ 적합을 나누는 근거가
    #: 아니다 — 가중치는 한 벌이고 이 값은 그 한 벌의 결과를 갈라 본 것뿐이다.
    #: 예전의 `short`/`long`(SOL 축)은 D-179 에서 사라졌다.
    by_regime: dict[str, float]
    #: The holdout evaluation. Used as is for significance (`compare`).
    evaluation: Evaluation
    #: ★ The weights **as fitted**. Without them, exporting the rule to a
    #: file writes the initial values, and that file does not reproduce —
    #: the file lies.
    #: ⛔ 2026-09-17 (D-179): 한 벌이므로 키는 `"all"` 하나다. 예전에는
    #: `short`/`long` 두 벌이었다.
    weights: dict[str, list[float]] = field(default_factory=dict)
    n_holdout: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def line(self) -> str:
        r = "  ".join(f"{k} {v:.4f}" for k, v in sorted(self.by_regime.items()))
        return (f"structHO {self.holdout:.4f} (n={self.n_holdout})  "
                f"in-sample {self.in_sample:.4f}  [{r}]")


def canonical_score(code: str, w0, *, table: PerfTable, matrix,
                    splits: SplitSet, max_evals: int = 300) -> CanonicalScore:
    """★ It cannot be called without `splits`. There is no arbitrary-split
    path.

    ★ 2026-09-17 (D-179): weights are fitted on **all** of `splits.train`
    as **one** vector and measured on **all** of `splits.val`. ⛔ There is no
    regime split any more — see the module docstring.
    """
    if not isinstance(splits, SplitSet):
        raise SplitError(
            "final scoring must receive the loop's SplitSet (D-36). Picking "
            "shapes separately makes the structure holdout overlap the "
            "training shapes — 11 of 19 actually did, and 'the gate was "
            "passed' was reported wrongly.")

    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import fit_weights, make_score_of

    train = list(splits.train.shapes)
    val = list(splits.val.shapes)
    if not val:
        raise SplitError("the validation split is empty. No structure "
                         "holdout can be produced.")

    fn = compile_rule(code)
    warns: list[str] = []
    if len(train) < MIN_TRAIN_SHAPES:
        warns.append(f"{len(train)} training shapes < {MIN_TRAIN_SHAPES}. "
                     "These weights are hard to trust")

    # ★ D-179 — ★ one fit on the whole training split, read on the whole
    #   holdout. ⛔ No regime split: the loop optimised one weight vector, so
    #   the reported number must come from one weight vector too.
    fit = fit_weights(fn, matrix, table, Split("train", tuple(train)),
                      w0, max_evals=max_evals,
                      # ★ **Final scoring is always regret** (D-103).
                      #   `fit_weights`'s default once changed to `rank`, so
                      #   it must be **stated** here. Otherwise every number
                      #   in this project silently becomes a different thing.
                      objective="regret")
    fitted: dict[str, list[float]] = {"all": [float(x) for x in fit.w]}
    so = make_score_of(fn, matrix, fit.w)
    e_tr = evaluate_scores(so, table, train, ks=(1,))
    reg_tr = {p: e_tr.regret[i, 0] for i, p in enumerate(e_tr.shapes)}
    e_ho = evaluate_scores(so, table, val, ks=(1,))
    reg_ho = {p: e_ho.regret[i, 0] for i, p in enumerate(e_ho.shapes)}
    tol_ho = {p: e_ho.tol[i] for i, p in enumerate(e_ho.shapes)}

    scored = [p for p in val if p in reg_ho]
    if not scored:
        raise SplitError("not a single holdout shape could be scored "
                         "(§26.4).")
    if len(scored) < len(val):
        warns.append(f"only {len(scored)} of {len(val)} holdout shapes were "
                     "scored")

    # ⚠️ ★ 읽기용일 뿐이다 (D-179 §2-3). 적합은 이미 한 벌로 끝났고, 이
    #    쪼개기는 `roofline` 축 — ⛔ 우리가 고른 경계가 아니라 ridge point 다.
    by_regime = {}
    for name in ("mem", "comp"):
        v = [reg_ho[p] for p in scored
             if regime_of(p, table.hw, axis="roofline") == name]
        if v:
            by_regime[name] = geomean(np.array(v))

    # Packed as an `Evaluation` so significance testing can use it directly
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
