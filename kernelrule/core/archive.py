"""The MAP-Elites archive (§13, §27).

## Why a single best will not do

Always starting from the best only searches nearby. It reaches the top of one
hill but never sees the higher mountain beside it. **A rule that is best in
one region is kept alive even if it is poor overall.**

    overall best      rule #47   1.12
    mem-bound best    rule #31   1.08 (overall 1.24)  <- discarded by a
                                                          single-best scheme
    compute best      rule #40   1.03 (overall 1.31)

And **combining the two can produce a rule that is good at both.** That is
the crossover, and it is where the leaps come from.

## Cell axes (§27) — ★ changed to the size regime

    code_len         the number of AST nodes            4 bands
    short_objective  the **short** shapes in the        4 bands
                     training split
    long_objective   the **long** shapes in the         4 bands
                     training split
                                                        -> 64 cells

**It was originally mem-bound / compute-bound.** It was changed to follow the
§30.5 result that size stratification moves 5x more than difficulty
stratification, and the measurement (§10.1) that evolution **sacrifices a
minority size regime**. The purpose is to preserve rules that transfer in
their own cells — even under balanced training, 1 in 9 still blows up.

⚠️ **Using the validation split as a cell axis contaminates the holdout**
(§10.2). The axes split regimes **within the training split**.

Adjust by watching how many cells fill over the first 20 rounds — **under 10
means the boundaries are too coarse, over 50 too fine.**

## Updates are judged against the noise floor (§7.4, §13.4)

Updating on "slightly better" makes the archive accumulate noise.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Archive", "Elite", "CELL_AXIS_NAMES", "N_QUANTILES"]

#: ★ The three cell axes (2026-09-10, D-155).
#:
#: ```
#: 2026-09-08 (D-144)  mem_objective · comp_objective · all_objective
#: ★ now               regret · n_weights · regime_skew
#: ```
#:
#: ⚠️ **All three of the old axes were scores.** A good rule is in the top
#: band of all three and lands in `(0,0,0)`; a bad one scatters and takes a
#: cell of its own — the grid ran backwards, spending its cells on bad
#: rules. Measured at D-154: 4 of the 5 held cells were rules at regret
#: 1.29~1.41 while every good proposal (1.10~1.14) fell into the one cell
#: that already held 1.1002, and the archive stopped moving for five rounds.
#:
#: ```
#: regret       how good it is
#: n_terms      how complex it is
#: regime_skew  mem - comp: which band it leans to
#:              band 0 = better on memory · 1 = balanced · 2 = better on
#:              compute
#: ```
#:
#: ⚠️ 2026-09-10 (D-156): axis 2 was `len(w0)`. Reusing a weight across terms
#: is allowed now, so twenty terms can run on one index and `len(w0)` stops
#: measuring size. **The term count is the honest measure.**
#:
#: ★ None of the three depends on which features exist, so a FeatureWriter
#: run that invents new axes needs no human to re-map them.
#:
#: ⚠️ The path count is **not** an axis — it correlates with the rule size at
#: r = +0.867 (measured on the D-154 run), so it would be the same axis
#: twice.
CELL_AXIS_NAMES = ("regret", "n_terms", "regime_skew")

#: Bands per axis. ★ Dynamic tertiles — **there are no absolute
#: boundaries** (see below). ⚠️ 3 is arbitrary.
N_QUANTILES = 3

#: ★ Only the top fraction by the acceptance key may enter the archive
#: (D-155).
#:
#: The archive is where **different kinds** are kept, not where bad rules
#: are kept. Without a cut the lowest band is forced to hold whatever is
#: worst, and those rules then occupy cells and are handed out as parents.
#:
#: ⚠️ **0.30 is arbitrary.** It was not swept. On the D-154 data it lets
#: 1 of 6 rules in at r0 and 20 of 69 by r11, and the line keeps falling as
#: the population improves — that is the property being bought, not the
#: number.
TOP_FRACTION = 0.30


@dataclass
class Elite:
    rule_id: str
    code: str
    w: list[float]
    regret: float
    #: The objective value on the **memory band** (t_memory > t_compute)
    #: within the training split.
    #: ★ Name history: `short_regret` -> `short_objective` (D-101) ->
    #: `mem_objective` (D-144). The axis changed from size (SOL 0.5ms) to
    #: the roofline.
    mem_objective: float
    #: The objective value on the **compute band** within the training
    #: split
    comp_objective: float
    #: ★ The objective value over the **whole** training split (the axis
    #: added in D-144)
    all_objective: float
    code_len: int
    round: int
    #: ★ The number of terms — `w[i]` **uses**, from `CheckReport.n_terms`
    #: (D-156). It is the archive's size axis; `len(w0)` stopped measuring
    #: size when weight reuse was allowed. 0 falls back to `len(w)` so an
    #: Elite built by hand in a test still works.
    code_terms: int = 0
    changes: str = ""
    hypothesis_id: str = ""
    parent_ids: list[str] = field(default_factory=list)
    val_regret: float = float("nan")
    #: ★ The rank loss (D-101). It becomes the acceptance criterion when
    #: `Archive(select_by="rank")`. `regret` **keeps being filled in** even
    #: then — both are recorded.
    rank_loss: float = float("nan")

    @property
    def regime_gap(self) -> float:
        """The gap between the compute band and the memory band. **A
        transfer signal.**

        When it is large, that rule is sacrificing one band.
        """
        return abs(self.comp_objective - self.mem_objective)

    @property
    def n_weights(self) -> int:
        """How many fitted coefficients. ⚠️ **Not the size axis** since
        D-156 — with reuse allowed it undercounts a big rule."""
        return len(self.w)

    @property
    def n_terms(self) -> int:
        """Cell axis 2 — how complex the rule is (D-155 · D-156).

        ★ It counts `w[i]` **uses**, not distinct indices, so reusing one
        weight across twenty terms still reads as twenty.
        """
        return int(self.code_terms or len(self.w))

    @property
    def regime_skew(self) -> float:
        """Cell axis 3 — **signed**, unlike `regime_gap` (D-155).

        Negative means it does better on the memory band, positive on the
        compute band. The sign is the point: `regime_gap` puts a
        memory-specialist and a compute-specialist in the same cell.
        """
        return self.mem_objective - self.comp_objective

    def to_dict(self) -> dict:
        """★ `cell` cannot be built here — the tertiles are decided by the
        **population**. `Archive.dump` fills in the cell at that moment
        (D-144)."""
        return dict(self.__dict__)


class Archive:
    """One best per cell + the overall best."""

    def __init__(self, noise_tol: float = 0.0, *,
                 select_by: str = "regret") -> None:
        #: The smallest improvement that counts as an update. Given by
        #: `is_significant` (§7.4).
        self.noise_tol = float(noise_tol)
        #: ★ What acceptance is judged on (D-101). The default is `regret`
        #: — every run so far is under that condition. `"rank"` runs **only
        #: when stated explicitly**.
        #:
        #: ⚠️ The cell **axes** do not change (code length / per-regime
        #: regret). The axes are the device that creates diversity and
        #: acceptance sets the goal — changing both together makes two
        #: variables (`rank-evo-prereg.md` correction).
        if select_by not in ("regret", "rank"):
            raise ValueError(f"unknown acceptance criterion: {select_by!r}")
        self.select_by = select_by
        # ★ The cells are **always dynamic tertiles** (D-144).
        #   `cell_mode="absolute"` was removed — with absolute boundaries
        #   [1.0, 1.05, 1.15, 1.35, inf], everything piles into one cell as
        #   evolution proceeds (measured: 2 cells occupied / 3-12 accepted /
        #   tau -0.093 over the whole range, the third candidate of D-42).
        #
        #   ⚠️ The price: what a cell means changes every round. As the
        #   whole improves, the absolute value of the first tertile falls.
        #   In exchange the cells fill evenly. When the boundaries move,
        #   **every held elite is re-placed** (`_consider_quantile`).
        self.cells: dict[tuple, Elite] = {}
        #: ★ The tertiles are computed from **every rule ever scored**, not
        #: from the elites that happen to be alive (D-155). With the live
        #: elites as the population the thing fed back on itself: a good
        #: candidate could not get in, so the population did not change, so
        #: the boundaries did not move. Only the axis values are kept — three
        #: floats per rule, 69 rules over 12 rounds.
        self._seen_axes: list[tuple[float, ...]] = []
        #: The acceptance key of every rule ever scored. The cut line is a
        #: quantile of this.
        self._seen_keys: list[float] = []
        self.best: Elite | None = None
        self.history: list[dict] = []
        self.n_seen = 0
        self.n_accepted = 0
        #: The round in which a cell was newly filled. Used for the
        #: early-stop verdict (§14.3).
        self.last_new_cell_round = -1

    def _key(self, e: Elite) -> float:
        """The value acceptance uses. **Lower is better.**"""
        if self.select_by == "regret":
            return e.regret
        v = e.rank_loss
        if not math.isfinite(v):
            raise ValueError(
                "select_by='rank' but Elite.rank_loss is missing. It does "
                "not silently fall back to regret (§26.4).")
        return v

    @property
    def _tol(self) -> float:
        """★ `noise_tol` is not used for the rank loss.

        `noise_tol` is a value `is_significant` produced at the scale of
        regret, and the rank loss is at a different scale. Besides, **the
        pairs behind the rank loss have already been filtered by
        `resolvable`**, so it already accounts for noise — adding a
        tolerance on top would subtract it twice.
        """
        return self.noise_tol if self.select_by == "regret" else 0.0

    def _boundaries(self) -> list[tuple[float, ...]]:
        """★ The tertile boundaries per axis, from **every rule scored so
        far** (D-155).

        There are no absolute boundaries — set at the scale of regret they
        stop meaning anything as the whole population improves (D-42's third
        candidate: 2 cells occupied, tau -0.093). The price is that a cell
        means something different every round, so **every held elite is
        re-placed** whenever they move.
        """
        out = []
        for j in range(len(CELL_AXIS_NAMES)):
            vals = sorted(v[j] for v in self._seen_axes)
            if not vals:
                out.append(())
                continue
            n = len(vals)
            out.append(tuple(vals[min(n - 1, (n * q) // N_QUANTILES)]
                             for q in range(1, N_QUANTILES)))
        return out

    def _cell_of(self, e: Elite, bounds: list[tuple[float, ...]]) -> tuple:
        """The band per axis. **Ties fall in the lower band** — the same
        rule as before (D-41), a value equal to a boundary does not create a
        cell of its own."""
        cell = []
        for j, name in enumerate(CELL_AXIS_NAMES):
            v = getattr(e, name)
            band = 0
            for b in bounds[j]:
                if v > b:
                    band += 1
            cell.append(min(N_QUANTILES - 1, band))
        return tuple(cell)

    def _cut_line(self) -> float:
        """★ The acceptance key a rule must beat to enter the archive at all
        (D-155). `inf` until something has been scored."""
        if not self._seen_keys:
            return float("inf")
        vals = sorted(self._seen_keys)
        k = max(1, int(TOP_FRACTION * len(vals)))
        return vals[k - 1]

    def _consider_quantile(self, e: Elite) -> list[str]:
        """Re-places every elite against the current boundaries and keeps
        the best per cell. If `e` survives, it won."""
        bounds = self._boundaries()
        pool = [*self.cells.values(), e]
        best: dict[tuple, Elite] = {}
        for x in pool:
            c = self._cell_of(x, bounds)
            cur = best.get(c)
            if cur is None or self._key(x) < self._key(cur):
                best[c] = x
        won: list[str] = []
        if e in best.values():
            won.append("new_cell" if len(best) > len(self.cells) else "cell")
        self.cells = best
        if won:
            self.last_new_cell_round = e.round
        return won

    def consider(self, e: Elite) -> list[str]:
        """Tries to insert it. Returns which slots it took. An empty list
        means discarded.

        ⚠️ The check happens **first**. `self.best is None or _key(e) < ...`
        short-circuits past `_key` when the archive is empty — a fail-open
        where only the first candidate got in unchecked (a test caught it).
        """
        self.n_seen += 1
        k = self._key(e)      # ★ the check. The value is used again below
        # ★ The population every boundary is computed from is **everything
        #   scored**, so it grows even when the candidate is refused (D-155).
        self._seen_axes.append(tuple(float(getattr(e, nm))
                                     for nm in CELL_AXIS_NAMES))
        self._seen_keys.append(k)
        won: list[str] = []
        if self.best is None or self._key(e) < self._key(self.best) - self._tol:
            won.append("best")
            self.best = e
        # ★ Only the top `TOP_FRACTION` may take a cell. The overall best is
        #   decided above and is not subject to it — the best is never lost.
        if k <= self._cut_line():
            won.extend(self._consider_quantile(e))
        # ★ The tertile cell is decided by the **population**, so an Elite
        #   alone does not know its own cell. What is recorded is the cell
        #   it actually landed in. An empty tuple if it was pushed out.
        c = next((k for k, v in self.cells.items() if v is e), ())
        if won:
            self.n_accepted += 1
        self.history.append({"round": e.round, "rule_id": e.rule_id,
                             "regret": e.regret, "rank_loss": e.rank_loss,
                             "select_by": self.select_by, "cell": list(c),
                             "won": won, "changes": e.changes,
                             "cut_line": self._cut_line(),
                             "axes": {nm: getattr(e, nm)
                                      for nm in CELL_AXIS_NAMES}})
        return won

    # -- Parent selection (§13.3) -----------------------------------------
    def parents(self, n: int, rng) -> list[tuple[str, list[Elite]]]:
        """6 steady improvements / 3 explorations of another hill / 3
        crossovers (the ratio at n=12)."""
        elites = list(self.cells.values())
        if not elites:
            return [("fresh", []) for _ in range(n)]
        n_exploit = max(1, round(n * 0.5))
        n_random = max(1, round(n * 0.25))
        n_cross = max(0, n - n_exploit - n_random)
        out: list[tuple[str, list[Elite]]] = []
        for _ in range(n_exploit):
            out.append(("exploit", [self.best or elites[0]]))
        for _ in range(n_random):
            out.append(("explore", [elites[int(rng.integers(len(elites)))]]))
        for _ in range(n_cross):
            if len(elites) >= 2:
                i, j = rng.choice(len(elites), size=2, replace=False)
                out.append(("cross", [elites[int(i)], elites[int(j)]]))
            else:
                out.append(("exploit", [self.best or elites[0]]))
        return out[:n]

    # -- State ------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def summary(self) -> dict:
        return {"n_cells": self.n_cells, "n_seen": self.n_seen,
                "n_accepted": self.n_accepted,
                "best_regret": self.best.regret if self.best else float("nan"),
                "cut_line": self._cut_line(),
                "last_new_cell_round": self.last_new_cell_round}

    def dump(self, path: str | Path) -> None:
        """★ Writes the final archive **state** (D-139 — not the history of
        improvements).

        The cells are tertiles, so an Elite alone does not know them — the
        current placement is written alongside.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            for c, e in self.cells.items():
                d = e.to_dict()
                d["cell"] = list(c)
                # ★ The raw axis values beside the cell (D-155) — without
                #   them "why did the cells come out that way" cannot be read
                #   back from the artefact.
                d["axes"] = {nm: getattr(e, nm) for nm in CELL_AXIS_NAMES}
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
