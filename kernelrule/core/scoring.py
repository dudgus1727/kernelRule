"""Scoring — regret, difficulty stratification, significance (§7, §30.4,
§30.5).

## The interface is the defence

What the scorer receives is one **function that produces an order**.

    order_fn(p: Problem, cand: CandidateSet) -> np.ndarray   # a permutation
                                                             # of candidate
                                                             # indices

`cand` has no times (§types) and `order_fn` is not given the table. Rules,
baselines, GBDT and the vendor heuristic all satisfy this signature, and
**none of them can see the times.** Times are used only for indexing inside
`evaluate()`, **after** the order is already fixed.

## Reporting convention (§7.3, §30.4, §30.5)

There is no path that reports the overall geomean alone. Every result is
split along three axes.

    ★ size        t_best >= 0.5ms / < 0.5ms  — **looked at first** (below)
      difficulty  upper half / lower half    — only meaningful at k=1
      k           1 / 3 / 5 / 10             — k=1 and k>=3 are different
                                               deployment scenarios

## ★ Size stratification moves 5x more than difficulty stratification (§30.5)

At the reference value of the static top-1:

    difficulty hi/lo    1.099  vs  1.132     difference 0.03
    size >=/<0.5ms      1.021  vs  1.164     difference 0.14   <- 5x

**Almost all of a fixed config's loss comes from shapes under 0.5ms.** And
that band is exactly **where the measurement resolution is worst** (§30.2 —
one tick is 0.2% at 0.5ms and 7.3% at 14µs).

    where there is room to gain  =  where it is hardest to confirm by
                                    measurement

This is the biggest tension in this project, and it goes straight into the
design of the metric. That is why `stratified()` puts size first and
`report()` prints size first. It is also why `hit_rate` (hitting the answer
set) is read alongside regret — on short shapes a regret difference may be
noise, but "did it land inside the indistinguishable set" is a verdict that
already accounts for the noise floor.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from kernelrule.core.table import PerfTable
from kernelrule.core.types import CandidateSet, Problem

__all__ = [
    "Comparison",
    "compare",
    "Evaluation",
    "OrderFn",
    "ScoreOf",
    "Strata",
    "evaluate",
    "evaluate_scores",
    "geomean",
    "is_significant",
]

#: An arbitrary ranker. It produces a **permutation** of candidate indices.
#: **It is not given the table.** A baseline with no scores, like the vendor
#: heuristic, is expressed this way too.
OrderFn = Callable[[Problem, CandidateSet], np.ndarray]

#: A score-based ranker. It produces a per-candidate score array (lower is
#: better). Every rule is one of these, and it takes the fast path that only
#: picks the top k.
ScoreOf = Callable[[Problem, CandidateSet], np.ndarray]

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def geomean(x) -> float:
    """Geometric mean. regret is a ratio scale, so no arithmetic mean."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(a))))


@dataclass(frozen=True, slots=True)
class Strata:
    """Evaluation strata. Built from `PerfTable`'s answer-side statistics.
    **Do not hand it to a rule.**"""

    shapes: tuple[Problem, ...]
    hard: np.ndarray          # bool (n_shapes,) the harder half
    small: np.ndarray         # bool (n_shapes,) t_best < 0.5ms
    difficulty: np.ndarray    # float
    best_ms: np.ndarray       # float
    layer: tuple[str, ...] = ()

    @classmethod
    def build(cls, table: PerfTable,
              shapes: Sequence[Problem] | None = None) -> Strata:
        shapes = tuple(shapes if shapes is not None else table.shapes())
        if not shapes:
            # Do not proceed with an empty set (§26.4).
            raise ValueError("Strata.build was given no shapes at all.")
        st = [table.stats(p) for p in shapes]
        diff = np.asarray([s.difficulty for s in st], dtype=np.float64)
        best = np.asarray([s.best_ms for s in st], dtype=np.float64)
        # Above the median counts as "the hard half". Deterministic even
        # with an odd count.
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
    """A scoring result. Per-shape raw values + per-stratum aggregates."""

    ks: tuple[int, ...]
    shapes: tuple[Problem, ...]
    #: (n_shapes, n_ks) — per-shape regret@k
    regret: np.ndarray
    #: (n_shapes, n_ks) — was there a member of the answer set in the top k
    hit: np.ndarray
    strata: Strata
    #: Per-shape 2σ tolerance. Used for the significance verdict.
    tol: np.ndarray
    label: str = ""
    extra: dict = field(default_factory=dict)

    def _col(self, k: int) -> int:
        try:
            return self.ks.index(k)
        except ValueError:
            raise KeyError(f"k={k} was not scored. scored k: {self.ks}"
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
        """The mandatory stratification of §30.4 + §7.3. **Report this
        dict whole.**

        ★ Size stratification comes first (§30.5). It moves 5x more than
        difficulty.
        """
        s = self.strata
        return {
            "all": self.at(k),
            # size — looked at first
            "large(>=0.5ms)": self.at(k, mask=~s.small),
            "small(<0.5ms)": self.at(k, mask=s.small),
            "n_small": float(int(s.small.sum())),
            # difficulty
            "hard": self.at(k, mask=s.hard),
            "easy": self.at(k, mask=~s.hard),
            "n_shapes": float(len(s.shapes)),
        }

    def size_gap(self, k: int = 1) -> float:
        """The regret gap between short and long shapes. **The main
        diagnostic** (§30.5).

        It was 0.14 for the static top-1. Whether a rule is narrowing this
        gap is the direct signal of "what did it actually learn on short
        shapes".
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
        """★ Prints the size stratification first (§30.5)."""
        st = self.strata
        lines = [(f"== {self.label or 'evaluation'} =="
                 f"  ({len(st.shapes)} shapes, "
                 f"{int(st.small.sum())} under 0.5ms)")]
        for k in self.ks:
            s = self.stratified(k)
            lines.append(
                f"  regret@{k:<2d} all {s['all']:.4f} "
                f"| >=0.5ms {s['large(>=0.5ms)']:.4f} "
                f"| <0.5ms {s['small(<0.5ms)']:.4f} "
                f"(gap {self.size_gap(k):+.4f}) "
                f"| hard {s['hard']:.4f} easy {s['easy']:.4f} "
                f"| hit {self.hit_rate(k):.3f}")
        return "\n".join(lines)


def evaluate(order_fn: OrderFn, table: PerfTable,
             shapes: Sequence[Problem] | None = None, *,
             ks: Sequence[int] = DEFAULT_KS, label: str = "",
             strata: Strata | None = None) -> Evaluation:
    """Scores `order_fn`.

    ★ `order_fn` receives only `(Problem, CandidateSet)`. Neither the table
    nor the times are given. Times are indexed only here, **after** the
    order is fixed (§30.7).
    """
    shapes = tuple(shapes if shapes is not None else table.shapes())
    if not shapes:
        raise ValueError("evaluate was given no shapes at all. It does not "
                         "proceed with an empty set (§26.4).")
    ks = tuple(int(k) for k in ks)
    strata = strata or Strata.build(table, shapes)

    regret = np.empty((len(shapes), len(ks)), dtype=np.float64)
    hit = np.zeros((len(shapes), len(ks)), dtype=bool)
    tol = np.empty(len(shapes), dtype=np.float64)

    for i, p in enumerate(shapes):
        cand = table.candidates(p)
        order = np.asarray(order_fn(p, cand))
        _check_order(order, cand, p)

        t = table.times_of(p)          # <- the only place times appear
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
    """Scores a score-based rule. **Same result as `evaluate`, much
    faster.**

    Instead of a full sort it uses `CandidateSet.top_k`. There are 15,000
    candidates per shape and only the top 10 are looked at, so a full sort is
    waste — measured at 2.7 s per rule, which at 12 rules per round is 32 s,
    on a par with the LLM calls.

    ★ The tie-break is still made on config identity alone (§30.7).
    """
    shapes = tuple(shapes if shapes is not None else table.shapes())
    if not shapes:
        raise ValueError(
            "evaluate_scores was given no shapes at all (§26.4).")
    ks = tuple(int(k) for k in ks)
    kmax = max(ks)
    strata = strata or Strata.build(table, shapes)

    regret = np.empty((len(shapes), len(ks)), dtype=np.float64)
    hit = np.zeros((len(shapes), len(ks)), dtype=bool)
    tol = np.empty(len(shapes), dtype=np.float64)

    for i, p in enumerate(shapes):
        cand = table.candidates(p)
        top = cand.top_k(np.asarray(score_of(p, cand)), kmax)
        t = table.times_of(p)          # <- the only place times appear
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
    """Checks that it is a permutation. **It does not pass silently**
    (§26.4).

    Rules the LLM writes really do produce code that drops or duplicates
    candidates. Left alone, that errs in the direction that makes regret look
    good (shrinking the candidates favours top-k).
    """
    if order.ndim != 1 or order.size != cand.n:
        raise ValueError(
            f"{p.key}: order length {order.size} != candidate count "
            f"{cand.n}. A rule must return every candidate, ordered.")
    if order.dtype.kind not in "iu":
        raise ValueError(
            f"{p.key}: the order is not integer indices ({order.dtype}).")
    seen = np.zeros(cand.n, dtype=bool)
    seen[order] = True
    if not seen.all():
        raise ValueError(
            f"{p.key}: the order is not a permutation "
            f"({int((~seen).sum())} missing). Dropping candidates favours "
            f"top-k and makes the scoring wrong.")


def is_significant(delta: float, ev: Evaluation, *,
                   mask: np.ndarray | None = None) -> bool:
    """Is this regret difference larger than the measurement noise (§7.4)?

    ⚠️ Do not use a fixed threshold. It differs per shape. Here the geometric
    mean of the 2σ tolerances of the shapes in the evaluation is used as the
    threshold.

    ⚠️ When it cannot be computed it is taken as **not significant** (§26.4
    — leaning towards failure). Not updating the archive is safer than
    mistaking noise for an improvement.
    """
    tol = ev.tol if mask is None else ev.tol[mask]
    if tol.size == 0 or not np.all(np.isfinite(tol)):
        return False
    return abs(float(delta)) > geomean(tol)


# ---------------------------------------------------------------------------
# ★ Comparing two methods — won/lost is not declared from a geomean
# difference alone (§7.4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Comparison:
    """The result of comparing A and B **per shape**.

    "1.085 > 1.080, so it lost" is imprecise. In a 61-shape geomean, 0.5% is
    a size that can appear when a few shapes move by a single tick.

    **"It lost significantly on N shapes" is the precise statement.**
    """

    name_a: str
    name_b: str
    geo_a: float
    geo_b: float
    #: Per shape, (t_A - t_B) / (t_best * noise_floor(t_best)). Larger
    #: means A is worse.
    sigma: np.ndarray
    #: Per-shape regret difference (A - B). Sigma says "is it real", this
    #: says "how large".
    delta: np.ndarray
    shapes: tuple[Problem, ...]
    k_sigma: float = 2.0

    @property
    def a_loses(self) -> np.ndarray:
        """Shapes where A lost to B **significantly**."""
        return self.sigma > self.k_sigma

    @property
    def a_wins(self) -> np.ndarray:
        return self.sigma < -self.k_sigma

    @property
    def tied(self) -> np.ndarray:
        return np.abs(self.sigma) <= self.k_sigma

    def report(self) -> str:
        n = len(self.shapes)
        n_win = int(self.a_wins.sum())
        n_lose = int(self.a_loses.sum())
        n_tie = int(self.tied.sum())
        lines = [
            (f"{self.name_a} {self.geo_a:.4f}  vs  {self.name_b} "
             f"{self.geo_b:.4f}   "
             f"(difference {self.geo_a - self.geo_b:+.4f})"),
            (f"  of {n} shapes — {self.name_a} won significantly on {n_win}, "
             f"lost on {n_lose}, indistinguishable on {n_tie}   "
             f"(at {self.k_sigma} sigma)"),
        ]
        if n:
            # ★ Sigma says "is it real", not "how large". On long shapes
            #   the noise floor is 0.05%, so even a small difference is
            #   hundreds of sigma. The size (the regret difference) has to
            #   be shown alongside or it gets misread.
            mw = int(np.argmin(self.delta))
            ml = int(np.argmax(self.delta))
            pw, pl = self.shapes[mw], self.shapes[ml]
            lines.append(f"  largest gain {pw.M}x{pw.N}x{pw.K} "
                         f"regret {self.delta[mw]:+.3f} "
                         f"({-self.sigma[mw]:.0f} sigma)")
            lines.append(f"  largest loss {pl.M}x{pl.N}x{pl.K} "
                         f"regret {self.delta[ml]:+.3f} "
                         f"({self.sigma[ml]:.0f} sigma)")
            big = int((self.delta > 0.05).sum())
            if big:
                lines.append(
                    f"  {big} shapes lost more than 0.05 of regret — the "
                    f"geomean is dragged by a handful like these")
        return "\n".join(lines)


def compare(a: Evaluation, b: Evaluation, table: PerfTable, *,
            name_a: str = "A", name_b: str = "B", k: int = 1,
            k_sigma: float = 2.0) -> Comparison:
    """Compares two scoring results **in units of the per-shape noise
    floor**.

    ⚠️ They must have been measured on the same shape set (§30.8 — same
    procedure / denominator / aggregation).
    """
    if tuple(p.key for p in a.shapes) != tuple(p.key for p in b.shapes):
        raise ValueError(
            "the two evaluations have different shape sets. A comparison "
            "only holds when measured on the same set (§30.8).")
    ci = a.ks.index(k), b.ks.index(k)
    sig = np.empty(len(a.shapes), dtype=np.float64)
    dlt = np.empty(len(a.shapes), dtype=np.float64)
    for i, p in enumerate(a.shapes):
        st = table.stats(p)
        r_a, r_b = a.regret[i, ci[0]], b.regret[i, ci[1]]
        dlt[i] = r_a - r_b
        # regret is t/best, so convert back to t
        denom = st.best_ms * st.noise_floor
        sig[i] = ((r_a - r_b) * st.best_ms / denom) if denom > 0 else 0.0
    return Comparison(name_a=name_a, name_b=name_b, geo_a=a.at(k),
                      geo_b=b.at(k), sigma=sig, delta=dlt, shapes=a.shapes,
                      k_sigma=k_sigma)
