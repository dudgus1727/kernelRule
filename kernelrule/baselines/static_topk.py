"""Static top-k — the regret of using k fixed configs, **independent of the
shape**.

## ★ The procedure changes the answer (§30.5b)

How you set the `status` filter and the cover definition splits the answer
three ways. Measured:

    ok only + individual full cover      k=1 1.394   k=3 1.383   k=8 1.383  <- saturates
    ok only + union cover                k=1 1.394   k=3 1.060   k=8 1.009
    all + union cover (representative)   k=1 1.115   k=3 1.031   k=8 1.006

**Compute all three and report them side by side, naming the representative
one. Do not report a single number.**

### (1) The `status` filter

`high_outlier_frac` is a property of **that one measurement**, not of the
config. More repetitions only raise the chance that one sample lands outside
the IQR, and the median time stays valid. Keeping only `ok` demands "a config
whose measurement happened to be clean on every shape", which leaves **3** of
17,325 configs ok across all 61 shapes.

### (2) The cover definition

Requiring each individual config to be valid on all 61 shapes effectively
excludes `split_k>1` — `split_k=3` is valid only where K is a multiple of 3.
A real library of course includes such configs and uses whichever is valid
per shape. **It is enough that the union of the k covers.**

### (3) Relaxing it collapses the other way

Using "score only on covered shapes and report the cover rate" (§9.1) as is
lets the greedy escape to 23% cover and report 1.074. So we **require the
union to cover every shape**, and report failure to do so as a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kernelrule.core.scoring import Strata, geomean
from kernelrule.core.table import PerfTable

__all__ = ["StaticTopK", "TopKResult", "PROCEDURES", "run_all_procedures"]

#: Penalty multiplier for an uncovered shape. It makes the greedy secure
#: **cover first**. Relaxing it (scoring only covered shapes) lets it escape
#: to 23% cover.
_UNCOVERED_PENALTY = 1e3

#: The three procedures reported side by side. The last is representative.
PROCEDURES = (
    ("ok_individual", dict(ok_only=True, coverage="individual"),
     "ok only + individual full cover"),
    ("ok_union", dict(ok_only=True, coverage="union"),
     "ok only + union cover"),
    ("canonical", dict(ok_only=False, coverage="union"),
     "★ all statuses + union cover (representative)"),
)


@dataclass
class TopKResult:
    procedure: str
    description: str
    ks: tuple[int, ...]
    #: k -> per-stratum regret dict
    by_k: dict[int, dict[str, float]] = field(default_factory=dict)
    #: k -> cover rate
    coverage: dict[int, float] = field(default_factory=dict)
    n_shapes: int = 0
    n_configs_considered: int = 0
    n_configs_total: int = 0
    chosen: list[tuple] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"[{self.procedure}] {self.description}",
                 (f"    {self.n_shapes} shapes, candidate configs "
                 f"{self.n_configs_considered}/{self.n_configs_total} "
                 f"({100 * self.n_configs_considered / max(1, self.n_configs_total):.2f}%)"),
                 (f"    {'k':>3} {'all':>7} {'>=0.5ms':>8} {'<0.5ms':>8} "
                 f"{'hard':>7} {'easy':>7} {'cover':>7}")]
        for k in self.ks:
            d = self.by_k[k]
            lines.append(
                f"    {k:>3} {d['all']:7.3f} {d['large(>=0.5ms)']:8.3f} "
                f"{d['small(<0.5ms)']:8.3f} {d['hard']:7.3f} {d['easy']:7.3f} "
                f"{self.coverage[k]:7.1%}")
        return "\n".join(lines)


class StaticTopK:
    """Facility-location greedy. Submodular, so it has the (1-1/e)
    guarantee."""

    def __init__(self, table: PerfTable, shapes=None, *,
                 coverage: str = "union") -> None:
        if coverage not in ("union", "individual"):
            raise ValueError(f"unknown cover definition: {coverage!r}")
        self.table = table
        self.coverage = coverage
        self.shapes = tuple(shapes if shapes is not None else table.shapes())
        if not self.shapes:
            raise ValueError("there are no shapes at all (§26.4).")
        self.strata = Strata.build(table, self.shapes)
        self._build()

    def _build(self) -> None:
        """config x shape matrix. Unmeasured entries are NaN."""
        keys: dict[tuple, int] = {}
        cols = []
        for p in self.shapes:
            cand = self.table.candidates(p)
            t = np.asarray(self.table.times_of(p), dtype=np.float64)
            best = self.table.best_time(p)
            rel = t / best
            col: dict[int, float] = {}
            for i in range(cand.n):
                key = (str(cand.kernel_id[i]), int(cand.split_k[i]),
                       str(cand.split_k_mode[i]))
                j = keys.get(key)
                if j is None:
                    j = keys[key] = len(keys)
                # The same key should not appear twice for one shape, but if
                # it does, take the faster one.
                if j not in col or rel[i] < col[j]:
                    col[j] = float(rel[i])
            cols.append(col)

        n_cfg, n_sh = len(keys), len(self.shapes)
        A = np.full((n_cfg, n_sh), np.nan, dtype=np.float64)
        for s, col in enumerate(cols):
            idx = np.fromiter(col.keys(), dtype=np.int64, count=len(col))
            val = np.fromiter(col.values(), dtype=np.float64, count=len(col))
            A[idx, s] = val
        self.keys = list(keys)
        self.n_configs_total = n_cfg
        if self.coverage == "individual":
            keep = np.flatnonzero(~np.isnan(A).any(axis=1))
            if keep.size == 0:
                raise ValueError(
                    "no config was measured on every shape. Static top-k "
                    "cannot be defined with the 'individual' cover.")
            self.rows = keep
        else:
            self.rows = np.arange(n_cfg)
        self.A = A[self.rows]
        self.logA = np.log(self.A)

    def run(self, ks=(1, 2, 3, 5, 8, 10, 20)) -> TopKResult:
        ks = tuple(sorted(int(k) for k in ks))
        n_sh = len(self.shapes)
        cur = np.full(n_sh, np.nan)
        chosen: list[tuple] = []
        res = TopKResult(procedure="", description="", ks=ks,
                         n_shapes=n_sh,
                         n_configs_considered=int(self.rows.size),
                         n_configs_total=self.n_configs_total)
        pen = np.log(_UNCOVERED_PENALTY)
        for k in range(1, max(ks) + 1):
            # The objective with each candidate added. Uncovered shapes are
            # filled with the penalty.
            cand = np.fmin(np.broadcast_to(cur, self.logA.shape), self.logA)
            filled = np.where(np.isnan(cand), pen, cand)
            obj = filled.mean(axis=1)
            i = int(np.argmin(obj))
            cur = np.fmin(cur, self.logA[i])
            chosen.append(self.keys[self.rows[i]])
            if k in ks:
                covered = ~np.isnan(cur)
                res.coverage[k] = float(covered.mean())
                rel = np.exp(cur)
                res.by_k[k] = self._strat(rel, covered)
        res.chosen = chosen
        return res

    def _strat(self, rel: np.ndarray, covered: np.ndarray) -> dict:
        s = self.strata

        def g(mask):
            m = mask & covered
            return geomean(rel[m]) if m.any() else float("nan")

        allm = np.ones(len(rel), dtype=bool)
        return {"all": g(allm), "hard": g(s.hard), "easy": g(~s.hard),
                "large(>=0.5ms)": g(~s.small), "small(<0.5ms)": g(s.small)}


def run_all_procedures(bundle_ref, env_hash: str, *, shapes_filter=None,
                       ks=(1, 2, 3, 5, 8, 10, 20)) -> list[TopKResult]:
    """Run all three procedures. **Do not emit only the representative one**
    (§30.5b)."""
    out = []
    for name, kw, desc in PROCEDURES:
        table = PerfTable.from_bundle(bundle_ref, env_hash=env_hash,
                                      ok_only=kw["ok_only"])
        shapes = ([p for p in table.shapes() if shapes_filter(p, table)]
                  if shapes_filter else None)
        try:
            r = StaticTopK(table, shapes, coverage=kw["coverage"]).run(ks)
        except ValueError as e:
            r = TopKResult(procedure=name, description=f"{desc} — failed: {e}",
                           ks=tuple(ks))
            out.append(r)
            continue
        r.procedure, r.description = name, desc
        out.append(r)
    return out
