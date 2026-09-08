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

#: ★ The three cell axes (2026-09-08, D-144).
#:
#: ```
#: old   code_len · short_objective(SOL<0.5ms) · long_objective   4x4x4
#: ★ new mem_objective · comp_objective · all_objective           3x3x3
#: ```
#:
#: The bands are cut by `t_memory > t_compute` (a comparison of the two
#: roofline lower-bound terms) — **no arbitrary threshold** such as
#: `SOL 0.5 ms` is used (D-143 showed that threshold cannot be defended).
#:
#: `all_objective` was confirmed not to be redundant — within the same
#: (mem, comp) cell, the overall band differed in 13 of 16 cells.
CELL_AXIS_NAMES = ("mem_objective", "comp_objective", "all_objective")

#: Bands per axis. ★ Dynamic tertiles — **there are no absolute
#: boundaries** (see below).
N_QUANTILES = 3


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

    def _quantile_cells(self, pool: list[Elite]) -> dict:
        """★ Sorts the held elites + the candidate **by the per-regime
        values** and assigns quantile cells.

        ⚠️ What is sorted are the three of `CELL_AXIS_NAMES`, and those
        values are **always per-regime regret** (`ev.at(1, mask=...)`). That
        holds even when the objective is `rank` — exactly as designed:
        **the axes are the diversity device and acceptance sets the goal**
        (D-101). It was first written as "sorted by the objective", which
        was inaccurate (corrected in D-104).

        The point is that there are no boundary values — absolute boundaries
        are set at the scale of regret, so when the value distribution moves
        everything piles into one cell. The archive holds at most 64, so
        sorting is free.

        **Ties share a cell** (`argsort(argsort(.))` is not used, D-41).
        """
        n = len(pool)
        out: dict[int, tuple] = {}
        axes = {}
        for name in CELL_AXIS_NAMES:
            vals = [getattr(x, name) for x in pool]
            order = sorted(range(n), key=lambda i: (vals[i], i))
            rank = [0] * n
            r = 0
            for pos, i in enumerate(order):
                if pos and vals[i] > vals[order[pos - 1]]:
                    r = pos
                rank[i] = r
            axes[name] = [min(N_QUANTILES - 1, x * N_QUANTILES // max(n, 1))
                          for x in rank]
        for i, _x in enumerate(pool):
            out[i] = tuple(axes[nm][i] for nm in CELL_AXIS_NAMES)
        return out

    def _consider_quantile(self, e: Elite) -> list[str]:
        """Re-assigns the cells and keeps only the best per cell. If `e`
        survives, it won."""
        pool = [*self.cells.values(), e]
        cells = self._quantile_cells(pool)
        best: dict[tuple, int] = {}
        for i, x in enumerate(pool):
            c = cells[i]
            cur = best.get(c)
            if cur is None or self._key(x) < self._key(pool[cur]):
                best[c] = i
        new = {c: pool[i] for c, i in best.items()}
        won: list[str] = []
        if e in new.values():
            won.append("new_cell" if len(new) > len(self.cells) else "cell")
        self.cells = new
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
        self._key(e)          # ★ the check. The value is used again below
        won: list[str] = []
        if self.best is None or self._key(e) < self._key(self.best) - self._tol:
            won.append("best")
            self.best = e
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
                             "won": won, "changes": e.changes})
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
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
