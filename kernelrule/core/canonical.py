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

This function produces the **structure holdout**: weights are fitted per
regime on `splits.train` and evaluated on `splits.val`. The loop used val
only for the early-stop decision, so the structure was not fitted to those
shapes.

## Deciding the regime

`regime_of(axis="size")` — an SOL proxy. **It does not use `t_best`**
(§10.1). Cutting on a boundary that cannot be computed at deployment time
makes the number an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from kernelrule.core.scoring import Evaluation, evaluate_scores, geomean
from kernelrule.core.splits import Split, SplitError, SplitSet, regime_of
from kernelrule.core.table import PerfTable

__all__ = ["CanonicalScore", "canonical_score"]

#: Below this many per regime, that regime's fit cannot be trusted (§10.1).
MIN_PER_REGIME = 8


@dataclass(frozen=True, slots=True)
class CanonicalScore:
    """The structure-holdout score. **It carries what was not looked at**
    (D-36)."""

    #: Geometric mean over `splits.val`. ★ This is the value to report.
    holdout: float
    #: Geometric mean over `splits.train`. For reference — the structure was
    #: fitted here.
    in_sample: float
    #: Per-regime holdout geometric mean.
    by_regime: dict[str, float]
    #: The holdout evaluation. Used as is for significance (`compare`).
    evaluation: Evaluation
    #: ★ The weights **as fitted** per regime. Without them, exporting the
    #: rule to a file writes the initial values, and that file does not
    #: reproduce — the file lies.
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

    Per regime, weights are fitted on `splits.train` and measured on
    `splits.val`.
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
    reg_tr: dict = {}
    reg_ho: dict = {}
    tol_ho: dict = {}
    fitted: dict[str, list[float]] = {}

    for name in ("short", "long"):
        g_tr = [p for p in train if regime_of(p, table.hw) == name]
        g_ho = [p for p in val if regime_of(p, table.hw) == name]
        if not g_tr:
            if g_ho:
                warns.append(f"regime {name!r}: 0 training shapes but "
                             f"{len(g_ho)} in the holdout — those shapes "
                             "cannot be scored")
            continue
        if len(g_tr) < MIN_PER_REGIME:
            warns.append(f"regime {name!r}: {len(g_tr)} training shapes < "
                         f"{MIN_PER_REGIME}. That regime's weights are hard "
                         "to trust")
        fit = fit_weights(fn, matrix, table, Split("train", tuple(g_tr)),
                          w0, max_evals=max_evals,
                          # ★ **Final scoring is always regret** (D-103).
                          #   `fit_weights`'s default once changed to `rank`,
                          #   so it must be **stated** here. Otherwise every
                          #   number in this project silently becomes a
                          #   different thing.
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
        raise SplitError("not a single holdout shape could be scored "
                         "(§26.4).")
    if len(scored) < len(val):
        warns.append(f"only {len(scored)} of {len(val)} holdout shapes were "
                     "scored")

    by_regime = {}
    for name in ("short", "long"):
        v = [reg_ho[p] for p in scored if regime_of(p, table.hw) == name]
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
