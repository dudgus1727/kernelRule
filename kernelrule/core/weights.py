"""Weight optimisation — separating structure from parameters (§29).

## Why they are separated

A rule mixes two things.

    structure   discrete, combinatorial. "which features, wired how" -> the LLM
    weights     continuous.              "is 2.0 right, or 2.7"      -> a
                                          numerical optimiser

## ★ When it runs is the crux (§29.3)

    wrong:  LLM writes a rule -> score -> archive
    right:  LLM writes a rule -> **optimise the weights** -> score -> archive

Without it, a good structure is thrown away because of bad initial values
while a mediocre structure survives on good ones. Evolution then selects not
structure but **luck in the weights**.

## Gradient descent cannot be used (§29.2)

The objective, regret, comes out through an `argmin`, so it is a **step
function** in the weights. Changing a weight slightly leaves regret unchanged
while the ranking holds, then it jumps as the ranking flips. The gradient is
0 or undefined. Nelder-Mead is used.

## It takes the training split only (§29.7)

It checks the `role` of the `Split`. There is no path by which the validation
or final split enters the objective — writing it in the documentation is not
enforcement (§30.8).
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

#: The function the LLM writes. `w` is **fitted by the numerical
#: optimiser** (the §8.1 replacement).
ScoreFn = Callable[[Feats, ShapeInfo, Hardware, np.ndarray], np.ndarray]


class FitError(RuntimeError):
    """The weights cannot be fitted. The rule is rejected."""


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
    #: The **effective contribution** per term = |w_i| x (the standard
    #: deviation of that feature's column) (D-70). Unlike the absolute
    #: magnitude it is invariant to feature scale, so it reads on the same
    #: basis even when the library changes. `None` if it cannot be computed.
    contrib: np.ndarray | None = None
    #: ★ The evaluation count **before polish**. `n_evals` includes polish,
    #: so it cannot be compared against `max_evals` — comparing it makes the
    #: cap warning fire **always** the moment polish is on. Whether the cap
    #: was reached is read from this value.
    n_fit_evals: int = 0
    #: ★ The evaluations spent building **only the starting point** from a
    #: surrogate loss (`init_objective`). They are added into `n_evals` and
    #: not into `n_fit_evals` — the cap verdict is made on evaluations of
    #: the true objective alone. For budget comparisons use `n_evals`.
    n_init_evals: int = 0

    @property
    def moved(self) -> bool:
        """★ Did the fitter actually move (D-54)?

        `False` means it was **scored on the initial values**, and §29.3's
        "score after optimising the weights — otherwise evolution selects
        luck in the weights rather than structure" does not hold for that
        candidate.

        13 of 24 were like that. The logs kept only the post-fit regret, so
        **you had to go looking to see it** — another task, exporting rules,
        revealed it by accident.
        """
        return not np.allclose(self.w, self.w0)

    def invariants(self) -> list[str]:
        """The reasons this fit is odd. **Normal means empty** (D-54).

        It makes each computation stage announce "did I do nothing" itself.
        Logs have to be searched after the fact, and searching requires
        knowing what to search for.
        """
        out: list[str] = []
        if not self.moved:
            out.append(f"the fitter did not move (n_evals={self.n_evals}) "
                       "— it is scored on the initial values")
        w = np.asarray(self.w)
        # ★ **The absolute magnitude (|w|/|w0|) is not an indicator**
        #   (D-70).
        #
        #   In the F1 library most features are [0, 0.2], so the fitter
        #   raises the magnitude greatly — max |w| went as high as
        #   4,159,634. The 24 a human wrote include [0, 300] ones, so theirs
        #   are 8.6~7,663. **Using the same 100x criterion makes it fire
        #   constantly on the F1 arm and the watchdog loses its signal**
        #   (principle 11).
        #
        #   The objective looks only at ranking, so the overall magnitude is
        #   harmless. What matters is **one term overwhelming the others**,
        #   and that has to be measured by effective contribution:
        #   |w_i| x (that feature's standard deviation).
        if self.contrib is not None:
            c = np.asarray(self.contrib, dtype=np.float64)
            pos = c[c > 0]
            if pos.size >= 2:
                med = float(np.median(pos))
                dom = np.flatnonzero(c > _DOMINANCE * med)
                if dom.size:
                    out.append(
                        f"one term overwhelms the others: "
                        f"{[int(i) for i in dom]} — its effective "
                        f"contribution exceeds {_DOMINANCE:.0f}x the median")
                dead = np.flatnonzero([approx_zero(x) for x in c])
                if dead.size:
                    out.append(f"terms with zero effective contribution: "
                               f"{[int(i) for i in dead]} — they take no "
                               f"part in the ranking")
        neg = np.flatnonzero(w < 0)
        if neg.size:
            # Every feature is "larger is worse", so a negative flips the
            # direction. It can be legitimate when reweighting under a shape
            # branch, so this **only warns**.
            out.append(f"negative weights: {[int(i) for i in neg]} — every "
                       f"feature is 'larger is worse' (§8.2)")
        return out

    @property
    def gap(self) -> float:
        """The training-validation gap. Recorded every round (§29.4).

        If it widens, the parameter count has to come down.
        """
        return self.val_regret - self.fit_regret

    @property
    def dead_terms(self) -> list[int]:
        """Terms that converged near 0 or are insensitive. Candidates for
        feature cleanup (§29.6).

        ⚠️ 2026-09-12 (D-169): this counts **two different things** and the
        two say opposite things about the library. `dead_by_weight` and
        `dead_by_sensitivity` split them; this one is kept because
        `n_dead_terms` of the 21-run campaign was recorded through it and
        that number must stay reproducible (documentation rule 2).
        """
        return [i for i in range(len(self.w))
                if abs(self.w[i]) < 1e-3 or self.sensitivity[i] < 1e-6]

    @property
    def dead_by_weight(self) -> list[int]:
        """★ The fitter pushed the weight to ~0. **That is the fitter
        working** — it found the term useless and removed it."""
        return [i for i in range(len(self.w)) if abs(self.w[i]) < 1e-3]

    @property
    def dead_by_sensitivity(self) -> list[int]:
        """★ The term carries a weight and still cannot move the ranking.

        Perturbing it changes nothing, so the axis had no ordering power
        here in the first place — that is a statement about the **library
        or the rule's structure**, not about the fitter. `|w| >= 1e-3`
        excludes the terms the fitter already zeroed, so the two counts do
        not overlap.
        """
        return [i for i in range(len(self.w))
                if self.sensitivity[i] < 1e-6 and abs(self.w[i]) >= 1e-3]

    def __str__(self) -> str:
        return (f"FittedRule(fit={self.fit_regret:.4f}, "
                f"val={self.val_regret:.4f}, evals={self.n_evals}, "
                f"w={np.array2string(self.w, precision=3)})")


class _Problem:
    """Precomputation for fitting. Features / tie-break / times are pulled
    out once per shape.

    ★ `score_fn` cannot see the times here either. Times are used only for
    indexing inside `_regret`, **after** the order is fixed.
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
        #: ★ The true-first-place versus the rest pairs (D-109). Not built
        #: when `rank_lambda` is 0.
        self._pairs1 = None
        self.n_pairs1 = 0

    # -- The rank loss (D-101) --------------------------------------------
    def _make_pairs(self, table: PerfTable, top_k: int, *,
                    anchor_best: bool = False) -> tuple[list, int, int]:
        """★ Pairs within the true top `top_k`. **Pairs noise cannot
        separate are dropped.**

        The weight is `|t_j - t_i| / t_best` — **the actual loss**. There is
        nothing to tune, and inside the noise floor it automatically
        approaches 0.

        With `anchor_best`, only **pairs involving the true first place**
        are kept — a smooth surrogate for `regret` (D-109). It is a subset
        of the same pair set, so the normalisation matches and `lambda`
        becomes a pure ratio.

        ⚠️ It uses `NoiseModel.resolvable`. The verdict is not redefined
        (principle 2).
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
            w = (tt[ju] - tt[iu]) / best              # ★ the actual loss
            n_ok += int(iu.size)
            pairs.append((top, iu, ju, w, float(w.sum())))
        return pairs, n_ok, n_drop

    def build_pairs(self, table: PerfTable, top_k: int) -> None:
        self._pairs, self.n_pairs, self.n_dropped = self._make_pairs(
            table, top_k)

    def build_top1_pairs(self, table: PerfTable, top_k: int) -> None:
        """★ True first place versus the rest — a smooth surrogate for
        `regret` (D-109)."""
        self._pairs1, self.n_pairs1, _ = self._make_pairs(
            table, top_k, anchor_best=True)

    def rank_loss(self, score_fn: ScoreFn, w: np.ndarray) -> float:
        """The weighted logistic pairwise loss. The convention is **lower
        is better**, so s_i < s_j."""
        if self._pairs is None:
            raise FitError("build_pairs must be called first.")
        return self._loss_on(self._pairs, score_fn, w)

    def rank_loss_top1(self, score_fn: ScoreFn, w: np.ndarray) -> float:
        if self._pairs1 is None:
            raise FitError("build_top1_pairs must be called first.")
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
            # softplus(s_i - s_j): the loss falls as s_i gets small
            # (= good)
            tot += float((pw * np.logaddexp(0.0, st[iu] - st[ju])).sum())
            den += wsum
        return tot / den if den > 0 else float("inf")

    def regret(self, score_fn: ScoreFn, w: np.ndarray) -> float:
        rs = np.empty(len(self.items), dtype=np.float64)
        for i, (f, info, cand, t, best) in enumerate(self.items):
            s = np.asarray(score_fn(f, info, self.hw, w), dtype=np.float64)
            if s.shape != (cand.n,) or not np.all(np.isfinite(s)):
                return float("inf")   # unrunnable at these weights (not a
                                      # rejection)
            # ★ Only the top k are picked. A full sort dominates the cost in
            #   this loop.
            rs[i] = float(t[cand.top_k(s, self.k)].min()) / best
        return geomean(rs)


class FitWarning(UserWarning):
    """The fit is odd. **Not a rejection, but not passed over silently**
    (D-54)."""


def fit_weights(score_fn: ScoreFn, matrix: FeatureMatrix, table: PerfTable,
                split: Split, w0: Sequence[float], *,
                method: str = "nelder-mead", max_evals: int = 200,
                k: int = 1, val_split: Split | None = None,
                n_restarts: int = 4,
                warn_invariants: bool = True,
                polish: bool = True,        # ★ D-55/D-56, on by default
                polish_budget: int = 600,   # ★ within 2x the fit's 305 (D-59)
                sensitivity_delta: float = 0.5,
                objective: str = "regret",  # ★ D-128: back to regret
                rank_top_k: int = 100,
                rank_lambda: float = 0.0,
                init_objective: str | None = None,
                init_evals: int = 0,
                bounds: list | None = None) -> FittedRule:
    """Fixes the structure and fits only the weights.

    `split` **must have `role="train"`.** There is no path by which the
    validation or final split enters the objective (§29.7).

    `val_split` is scored **for reporting only**, after the fit has
    finished. It takes no part in the objective — it exists so the gap
    (`FittedRule.gap`) can be recorded every round.

    ## ★ `objective` — the default is `"regret"` (reverted in D-128)

    ```
    "rank"    ★ default. The weighted pairwise loss within the true top
              `rank_top_k`. Differentiable -> L-BFGS-B
    "regret"  the relative time of the single argmin. A step function, so
              Nelder-Mead + restarts
    ```

    ### The default changed twice — it is now `"regret"`

    ```
    ~2026-09-01   "regret"   the path every result so far passed through
     2026-09-01   "rank"     the experiment being run then was the rank loss
                             (D-101)
    ★2026-09-04   "regret"   the rank loss was concluded to be the **wrong
                             objective** (D-118 · D-121). It is taken out of
                             the evolution path (D-128)
    ```

    ⚠️ `"rank"` **remains as a function** — `rank_loss` / `rank_loss_top1` /
    `tau` are used as metrics, and reproducing an old run only needs it
    passed explicitly. **The evolution loop refuses it** (`LoopConfig`).

    Even under `"rank"`, `fit_regret` keeps being computed and recorded as
    `regret` — **the scoring criterion does not change** (the experiment
    plan `rank-evo-prereg.md` §3).

    ## `init_objective` — builds **only the starting point** from a
    surrogate loss

    ```
    init_objective="rank_top1"   stage 1: L-BFGS-B on rank_loss_top1
                                 (`init_evals` evaluations). It is
                                 differentiable
                                 stage 2: from that point, a fit seen
                                 **through regret**
    ```

    ⚠️ **Acceptance is by regret.** The stage-1 value does not survive into
    the return value; `best_v` / `fit_regret` / the polish all re-measure
    through the true objective. It mixes conditions, so using it must be
    stated in the experiment plan — "used only to generate the starting
    point; acceptance is regret@1" (`fitter-regret-prereg.md` §2).

    It is accepted only under `objective="regret"`. Building the starting
    point from the rank loss on the rank-loss path would be calling the same
    objective twice.
    """
    if not isinstance(split, Split):
        raise SplitError(
            "fit_weights takes a Split. Passing a list of shapes leaves it "
            "unknown which split it is — unstated means an error (§26.4).")
    if split.role != "train":
        raise SplitError(
            f"a split with role={split.role!r} came into fit_weights. "
            f"Only the training split is accepted (§29.7). Fitting on the "
            f"validation or final split stops it from being a holdout.")
    if val_split is not None and val_split.role != "val":
        raise SplitError(
            f"val_split has role {val_split.role!r}. It must be 'val'.")

    w0 = np.asarray(list(w0), dtype=np.float64)
    # ★ Per-weight bounds (D-112). With `None` nothing changes — a rule
    #   that uses no exponent slot must be under **the same conditions** as
    #   the old runs (principle 36).
    if bounds is None:
        def _proj(x):
            return x
    else:
        if len(bounds) != w0.size:
            raise FitError(
                f"bounds length {len(bounds)} != weights {w0.size}")
        _blo = np.array([b[0] for b in bounds], dtype=np.float64)
        _bhi = np.array([b[1] for b in bounds], dtype=np.float64)

        def _proj(x):
            return np.clip(x, _blo, _bhi)

        w0 = _proj(w0)
    if w0.ndim != 1 or w0.size == 0:
        raise FitError(f"the shape of W0 is wrong: {w0.shape}")
    if not np.all(np.isfinite(w0)):
        raise FitError("W0 contains a non-finite value.")

    t0 = time.perf_counter()
    prob = _Problem(matrix, table, split.shapes, k)
    n_eval = 0
    n_inf = 0

    # ★ It holds on to **the best it has seen** itself (D-55). Taking only
    #   the optimiser's `res.x` throws away better points visited during the
    #   search — the objective is a step function, so the simplex can step
    #   on a good vertex and pass over it while the contraction ends
    #   elsewhere. 5 of 24 were like that, discarding up to 0.0277. The
    #   evaluations are already paid for, so recovering them is free.
    seen_v = float("inf")
    seen_w = w0.copy()

    if objective not in ("regret", "rank"):
        raise FitError(f"unknown objective: {objective!r}. It must be one "
                       f"of regret | rank.")
    if rank_lambda < 0:
        raise FitError(f"rank_lambda must be 0 or more: {rank_lambda}")
    if objective != "rank" and rank_lambda:
        raise FitError(
            f"rank_lambda={rank_lambda} but objective={objective!r}. "
            f"Lambda is only laid on top of the rank loss (D-109).")
    if objective == "rank":
        prob.build_pairs(table, rank_top_k)
        if prob.n_pairs == 0:
            raise FitError(
                "there is not one pair to use for the rank loss — no pair "
                f"separable by the noise floor lies within the top "
                f"{rank_top_k}. This does not proceed silently.")
        if rank_lambda:
            prob.build_top1_pairs(table, rank_top_k)
            if prob.n_pairs1 == 0:
                raise FitError(
                    "there is not one first-place pair — the true first "
                    f"place cannot be separated from the second by the "
                    f"noise floor (top {rank_top_k}). This does not proceed "
                    f"silently.")

    def value_at(w: np.ndarray) -> float:
        """★ The value of **the objective being fitted**. It does not
        count.

        ⚠️ Polish uses this too (D-122). `prob.regret` used to be nailed
        into `_polish`, so under `objective="rank"` it was **comparing a
        rank-loss reference value against regret** — regret (1.2) > the rank
        loss (0.24), so no step was ever accepted and polish silently did
        nothing. The objective-value function lives in **exactly one place**
        (principle 2).
        """
        # ★ It measures **after folding into the bounds**. Nelder-Mead does
        #   not know about bounds and neither does polish, so folding here is
        #   what enforces them in one place.
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
            return 1e6   # unrunnable at these weights. Not a structural
                         # rejection.
        if v < seen_v:
            seen_v, seen_w = v, wa.copy()
        return v

    m = method.lower()
    if m not in ("nelder-mead", "neldermead", "powell", "cma"):
        raise FitError(f"unknown optimiser: {method!r}. It must be one of "
                       f"nelder-mead | powell | cma.")

    # ★ Builds **only the starting point** from the surrogate loss. The
    #   value from here survives nowhere — from `best_v` below, everything
    #   is re-measured through the true objective.
    n_init = 0
    start0 = w0.copy()
    if init_objective is not None:
        if init_objective != "rank_top1":
            raise FitError(
                f"unknown starting-point objective: {init_objective!r}. "
                f"Only rank_top1 exists.")
        if objective != "regret":
            raise FitError(
                f"init_objective is used only on the regret path "
                f"(objective={objective!r}). Fitting with the rank loss and "
                f"building the starting point from the rank loss is calling "
                f"the same thing twice.")
        if init_evals <= 0:
            raise FitError(
                f"init_objective={init_objective!r} but "
                f"init_evals={init_evals}. The budget must be stated — the "
                f"starting-point stage spends the evals budget too (fairness "
                f"of the arm comparison).")
        prob.build_top1_pairs(table, rank_top_k)
        if prob.n_pairs1 == 0:
            raise FitError(
                "there is not one first-place pair — the true first place "
                f"cannot be separated from the second by the noise floor "
                f"(top {rank_top_k}). This does not proceed silently.")
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

    # ★ Restarts are needed. The objective is a **step function**, so a
    #   plain simplex contracts on a flat region and stops (§29.2). Cases
    #   really do occur where it cannot move a single step from the initial
    #   values. Each restart re-inflates the simplex, and some start from a
    #   random point. The seed is fixed, so it is deterministic.
    best_w = start0.copy()
    best_v = obj(best_w)
    #: How many restarts actually started. Below, it **counts and warns**.
    n_started = 0
    if objective == "rank":
        # ★ It is not a step function, so a quasi-Newton method comes
        #   first. That is the benefit of this design.
        #
        # ⚠️ **The budget is divided.** It was first set to
        #    `maxfun=max_evals`, and in measurement L-BFGS alone used 209/200
        #    so that **not one restart ran.** "the restarts are left as they
        #    are" became false, and it ends at a single local optimum with no
        #    global search (principle 1 — the device does not run).
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
    # ★ **Was the last restart still improving as it used up the budget?**
    #   "the budget was used up" is not itself a warning — the restart
    #   schedule is **designed to spend all of** `max_evals`, so it is always
    #   true (principle 11). The signal is "it was still improving at the
    #   moment it was cut".
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
        if seen_v < best_v:            # ★ not res.x but 'the best seen'
            best_v, best_w = seen_v, seen_w.copy()
        cut_while_improving = (best_v < before_v - 1e-12
                               and n_eval - before_n >= per)

    # ★ Did the restarts **actually run** (2026-09-01)? It really happened
    #   that `n_restarts=4` was written down and only 1 ran — on the rank
    #   path, L-BFGS used the whole budget alone through
    #   `maxfun=max_evals`. The comment said "the restarts are left as they
    #   are" and it was **false** (principle 1).
    if n_restarts > 1 and n_started < 2:
        warnings.warn(
            f"fit_weights: only {n_started} restarts ran "
            f"(n_restarts={n_restarts}). The first optimisation uses the "
            f"whole budget of {max_evals} — there is no global search.",
            FitWarning, stacklevel=2)
    if n_inf >= n_eval:
        raise FitError(
            f"the rule produced no valid score at any weights "
            f"({n_inf}/{n_eval}). The structure is rejected.")

    n_fit = n_eval    # ★ before polish. The cap verdict uses this value.
    w = _proj(best_w)
    if polish:
        w, best_v, n_pol = _polish(prob, score_fn, w, best_v, polish_budget,
                                   value=value_at)
        n_eval += n_pol
        # ★ Polish does not know about bounds. **It folds once more here**
        #   — regret and sensitivity below are re-measured on the folded
        #   values, so the reported and returned numbers come from the same
        #   weights.
        w = _proj(w)
    # ★ The scoring criterion is always regret — even under
    #   `objective="rank"`. "score by regret, train by the rank loss"
    #   (rank-evo-prereg.md §3)
    fit_regret = float(prob.regret(score_fn, w))
    if not np.isfinite(fit_regret) or fit_regret >= 1e6:
        raise FitError(
            "regret is not finite at the fitted weights. Rejected.")

    sens = _sensitivity(prob, score_fn, w, fit_regret, sensitivity_delta)
    contrib = _contributions(prob, score_fn, w)

    val = float("nan")
    if val_split is not None:
        vp = _Problem(matrix, table, val_split.shapes, k)
        val = float(vp.regret(score_fn, w))

    # ★ Two things were fixed (D-76).
    #   (1) It compares against `n_fit`, which excludes the polish
    #       evaluations. Compared against `n_eval`, the polish budget of 600
    #       always exceeds the cap of 300.
    #   (2) Exhausting the budget alone does not warn — the restart schedule
    #       is designed to spend it all, so that is always true too. It warns
    #       only when it was **still improving at the moment it was cut**.
    #       Only then is "cut before convergence" a fact.
    hit_cap = n_fit >= max_evals and cut_while_improving
    out = FittedRule(w=w, w0=w0, fit_regret=fit_regret,
                     # ★ Budget comparisons use this value — the
                     #   starting-point stage counts too.
                     n_evals=n_eval + n_init,
                     n_infeasible=n_inf, sensitivity=sens,
                     seconds=time.perf_counter() - t0, val_regret=val,
                     method=m, contrib=contrib, n_fit_evals=n_fit,
                     n_init_evals=n_init)
    if warn_invariants:
        msgs = out.invariants()
        if hit_cap:
            # It can overshoot the cap slightly — the check happens only
            # before entering a restart, and `obj(res.x)` is called a few
            # more times. The signal is not the amount of overshoot but
            # **the fact that it was reached**: it was still finding
            # improvements when the budget ran out.
            msgs.append(f"cut at the evaluation cap while still improving "
                        f"({n_fit}/{max_evals}, {n_eval} in total including "
                        f"polish) — a larger budget could do better")
        for msg in msgs:
            warnings.warn(f"fit_weights: {msg}", FitWarning, stacklevel=2)
    return out


#: The seed for the restart starting points. It is fixed, so `fit_weights`
#: is deterministic.
_RESTART_SEED = 20260820

#: The initial simplex step = this value x |start|. It halves per restart.
#: ★ At the default of 0.6, 13 of 24 **could not move a single step**
#: (D-54).
SIMPLEX_SCALE = 0.6
#: Above 0 it uses an **absolute step** (ignoring |start|). It tests
#: saturation of a dominant term.
SIMPLEX_ABS = 0.0

#: Warns when one term's effective contribution exceeds **this multiple of
#: the median of the others**. Using an absolute magnitude (100x) fires
#: constantly on the F1 library (D-70).
_DOMINANCE = 50.0

#: Step multipliers for the coordinate polish (when `polish=True`). It
#: sweeps from large to small.
#: The step multipliers of the coordinate polish. It sweeps **widely, on a
#: log scale** (D-59).
#:
#: They were first (0.5, 0.25, 0.1) — all below 1, so **there was no attempt
#: at all to shake one coordinate hard.** The six reach failures had gaps of
#: 0.0085~0.0308, and on a step function a gap of that size can be crossed by
#: doubling or halving one coordinate. Sweeping from large first crosses the
#: big steps, and the small ones do the fine work.
_POLISH_DELTAS = (10.0, 3.0, 1.0, 0.5, 0.25, 0.1)

#: The sign combinations for two-coordinate attempts. There can be steps a
#: single coordinate cannot cross — when two terms cancel each other and the
#: ranking does not move.
_PAIR_SIGNS = ((+1.0, +1.0), (+1.0, -1.0), (-1.0, +1.0), (-1.0, -1.0))


def _polish(prob, score_fn: ScoreFn, w: np.ndarray, base: float,
            budget: int, *, pairs: bool = True, value=None
            ) -> tuple[np.ndarray, float, int]:
    """★ Polishes by coordinate descent (D-55, strengthened in D-59).

    The point where Nelder-Mead stopped **was not a local optimum along the
    coordinate directions either** — 9 of 24 improved by shaking a single
    coordinate, by up to 0.0513. The objective is a step function, so the
    simplex contracts and stops on a flat region; crossing one step along an
    axis drops the value.

    It does three things.

    ```
    1  single coordinate x delta   delta on a 10 ~ 0.1 log scale (D-59)
    2  two coordinates             only when 1 found nothing in a full pass
    3  repeat                      on improvement, another pass at that delta
    ```

    Two-coordinate steps are switched on **only when single coordinates are
    stuck**. It is `n^2 x 4` and therefore expensive, and using it while
    single coordinates still work only eats budget.

    ⚠️ **It sees the training shapes only** — `prob` is built from the
    training split. There is no argument by which the holdout could come in,
    and `test_polish_only_sees_the_training_split` pins that (§29.7).

    ## ★ `value` — it takes the objective being fitted (D-122)

    With `None` it is `prob.regret`. **When called with
    `objective="rank"` it must be passed** — without it, a rank-loss
    reference value is compared against regret, and the magnitudes differ so
    that no step is ever accepted. In that state every rank-loss run of
    D-101~D-112 ran **without polish**.
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
        # ★ Single coordinates are stuck at this delta. Do a pass with
        #   coordinate pairs.
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
    """One CMA-ES run (D-123). **The initial step is given the same as
    Nelder-Mead's.**

    With `sigma0=1.0` and per-coordinate steps through `CMA_stds`, it starts
    from the same size as `_minimize_once`'s simplex step — so the step size
    is not a confounder in the arm comparison.

    ⚠️ **The budget cannot be matched exactly.** It ends on generation
    boundaries, so it overshoots by a few (16 dimensions, popsize 12 -> at
    most 11). The actual evaluation count is reported (`n_evals`) — it does
    not assume the budgets were equal (principle 38).
    """
    try:
        import cma as _cma
    except ImportError as e:      # pragma: no cover - optional dependency
        raise FitError(
            "method='cma' but the `cma` package is missing. "
            "`pip install cma` (the `fit` extra group in pyproject). "
            "It does not silently fall back to another optimiser.") from e

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
    """Stands in for a `scipy` result object — the caller looks only at
    `.x`."""

    x: np.ndarray


def _minimize_once(obj, start, method: str, budget: int, r: int, *,
                   bounds: list | None = None):
    """One restart. It starts by **re-inflating** the simplex."""
    from scipy.optimize import minimize

    start = np.asarray(start, dtype=np.float64)
    if method == "cma":
        return _cma_once(obj, start, budget, r, bounds=bounds)
    if method == "powell":
        return minimize(obj, start, method="Powell",
                        options={"maxfev": budget, "xtol": 1e-4, "ftol": 1e-6})
    n = len(start)
    # ★ The step size. Pulled out as `SIMPLEX_SCALE` so it can be swept
    #   (D-54). A relative step (proportional to |start|) **shakes a
    #   dominant term less** — if a large term has already saturated the
    #   ranking, shaking it more does not move the ranking. `SIMPLEX_ABS`
    #   tests that (an absolute step).
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
    """The **effective contribution** per term = |w_i| x (the spread that
    term creates in the score) (D-70).

    It sets only `w_i` to 0, recomputes the score, and measures how much the
    difference from the original scatters within a shape. **Only the spread
    within a shape is looked at** — a shape constant does not change the
    ranking, so its contribution must be 0 (absolute rule 2).

    ★ **It does not look at the times.** It calls only `score_fn` and never
    `prob.regret` — there is no path by which the answer comes in (§3).

    Unlike the absolute magnitude (|w|/|w0|), it is **invariant to feature
    scale**, so it reads on the same basis even when the library changes.
    Between F1 (features [0,0.2]) and the human 24 (features [0,300]) the
    magnitude of |w| differed by more than three orders, which made an
    absolute criterion meaningless.
    """
    out = np.zeros(len(w), dtype=np.float64)
    n = 0
    # ★ `items` is `(feats, info, cand, times, best)`. The last two are
    #   **the answer**, so they are bound to `_` names to keep them
    #   untouchable (§3).
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
    """The change in regret when each `w[i]` is shaken by ±delta (§29.6).

    An insensitive term means that feature is useless; a very sensitive one
    means that physical quantity dominates. Both belong in the diagnostic
    report.
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
    """`score_fn` + weights -> the `score_of` that `evaluate_scores`
    takes.

    This is what the hot scoring path uses (it picks only the top k).
    """
    w = np.asarray(list(w), dtype=np.float64)
    hw = matrix.hw

    def score_of(p: Problem, cand) -> np.ndarray:
        f, info = matrix.for_shape(p)
        return np.asarray(score_fn(f, info, hw, w), dtype=np.float64)

    return score_of


def make_order_fn(score_fn: ScoreFn, matrix: FeatureMatrix,
                  w: Sequence[float]):
    """`score_fn` + weights -> the `order_fn` the scorer takes
    (§scoring).

    ★ Training and deployment use **the same `score_fn`**. There is no
    conversion, so the error "training differed from deployment" is blocked
    at the source (the §8.1 replacement).
    """
    w = np.asarray(list(w), dtype=np.float64)
    hw = matrix.hw

    def order_fn(p: Problem, cand) -> np.ndarray:
        f, info = matrix.for_shape(p)
        s = np.asarray(score_fn(f, info, hw, w), dtype=np.float64)
        return cand.order_by(s)

    return order_fn
