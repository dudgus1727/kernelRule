"""`PerfTable` — table lookups. **The measured times exist only here**
(§7.1, §3).

## Structural isolation

What goes out to the rule/feature side and the times are **different
objects**.

    table.candidates(p)  -> CandidateSet    no times. rules/features see this
    table.times_of(p)    -> np.ndarray      only the scorer calls it. read-only

`CandidateSet` has no time field, so writing `sorted(..., key=(score, time))`
or `idxmin()` requires referencing a field that does not exist, and that is
an `AttributeError`. This is how the §30.7 bug is blocked **at the data
structure level**.

It was confirmed on this table: of 66 shapes, **29 have an exact tie at the
best time**, with up to 84 tied (timer quantisation). "The optimal config for
that shape" is a function of the tie-break rule, not a physical fact. So this
class **does not provide** `best_config()` — what can be defined is only
`best_time()` (a scalar, tie-break independent) and `answer_mask()` (a set).

## env_hash is not a join key but an isolation boundary (§3.4)

`env_hash` is a **required argument with no default**. kernelTab stepped into
this trap five times, every one of them "aggregating data with several
conditions mixed in, without a filter", and every one **silently wrong, with
no error.**
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kernelrule.core.adapter import normalize
from kernelrule.core.noise import NoiseModel
from kernelrule.core.types import (
    CandidateSet,
    Config,
    Hardware,
    Problem,
    ShapeKey,
    config_from_row,
    hardware_from_env,
    make_tiebreak,
)

__all__ = ["PerfTable", "TableError"]

#: The join key. Used to check that the ranking table and the scoring table
#: point at the same rows.
_JOIN = ("M", "N", "K", "kernel_id", "split_k", "split_k_mode")

#: The size-stratification boundary (§30.4). Shorter than this and noise
#: dominates the ranking.
SIZE_STRATA_MS = 0.5

#: **Condition metadata** that has a single value across the whole table.
#: There is no reason to carry it per row. 980,915 rows x a string exceeds
#: 100MB. It is moved to `PerfTable.meta`.
_CONSTANT_META = ("bundle_id", "gpu_name", "cutlass_commit", "nvcc_arch",
                  "clock_locked", "locked_mhz", "sm_count",
                  "peak_tflops_used", "ridge_point_spec")

#: Short repeated strings. Keeping them as categories cuts memory by an
#: order of magnitude.
_CATEGORICAL = ("dtype", "acc_dtype", "layout_a", "layout_b", "layout_c",
                "split_k_mode", "pipeline_kind", "arch", "ext_swizzle_type",
                "workspace_dtype", "partials_dtype")


class TableError(RuntimeError):
    """The table cannot be trusted. **Do not proceed with a default.**"""


@dataclass(frozen=True, slots=True)
class ShapeStats:
    """Answer-side statistics for one shape. **For the scorer and
    stratification only. Do not hand it to a rule.**

    `difficulty` is an `ANSWER_COLS` quantity — a rule that knows "this one
    is hard, be careful" has peeked at the answer. It is used only for
    evaluation stratification (§7.3).
    """

    key: ShapeKey
    n_candidates: int
    best_ms: float
    median_ms: float
    difficulty: float
    noise_floor: float
    answer_tol: float
    n_answers: int
    n_tied_at_best: int
    n_distinct_times: int

    @property
    def distinct_time_frac(self) -> float:
        """A measurement-resolution indicator (§30.4b). **A different axis**
        from difficulty.

        low difficulty  = performance really is similar (physics)
        low here        = the measurement cannot tell them apart (instrument)
        """
        return self.n_distinct_times / self.n_candidates if self.n_candidates else 0.0

    @property
    def is_small(self) -> bool:
        """Is it in the noise-dominated band (§30.4)? 45 of the 66 shapes
        in the A6000 table."""
        return self.best_ms < SIZE_STRATA_MS


class PerfTable:
    """(shape, config) -> time lookup.

    ⚠️ Do not call the constructor directly — use `from_bundle` /
    `from_frames`.
    """

    __slots__ = ("_X", "__times", "_rows", "_stats", "_cands", "_order",
                 "_hw", "_noise", "_meta", "_env_hash")

    def __init__(self, X: pd.DataFrame, times: np.ndarray, *,
                 hw: Hardware | None, noise: NoiseModel, env_hash: str,
                 meta: dict) -> None:
        if len(X) != len(times):
            raise TableError(
                f"feature rows {len(X)} != time rows {len(times)}. The "
                f"join is misaligned.")
        self._X = X.reset_index(drop=True)
        # Name mangling + read-only. Makes it hard for this to slip into a
        # sort key by accident.
        t = np.asarray(times, dtype=np.float64).copy()
        t.setflags(write=False)
        self.__times = t
        self._hw = hw
        self._noise = noise
        self._env_hash = env_hash
        self._meta = dict(meta)

        # ★ The shape groups are built **vectorised**. Building a million
        #   tuples and lists in a Python loop uses more memory than the
        #   table itself (measured at 3.7GB -> ).
        idx = self._X.groupby(["M", "N", "K", "dtype"], sort=False,
                              observed=True).indices
        self._rows = {(int(k[0]), int(k[1]), int(k[2]), str(k[3])):
                      np.asarray(v, dtype=np.int64) for k, v in idx.items()}
        self._order = list(self._rows)
        self._cands: dict[ShapeKey, CandidateSet] = {}
        self._stats: dict[ShapeKey, ShapeStats] = {}
        self._build_stats()

    # -- Construction -----------------------------------------------------
    @classmethod
    def from_bundle(cls, ref: str | Path, *, env_hash: str,
                    ok_only: bool = False,
                    unexpected: str = "warn") -> PerfTable:
        """Builds from a kernelTab bundle.

        `env_hash` is **required** (§3.4). There is no default.

        Why `ok_only=False` is the default: `high_outlier_frac` is a valid
        measurement too (10.7% of the total). This decision must be recorded
        in `RunConfig`, and the gate-condition report gives both.
        """
        from kerneltab.core.bundle import load_bundle

        b = load_bundle(ref, verify=True)
        full = str(b.env_hash)
        if not full.startswith(str(env_hash)):
            raise TableError(
                f"env_hash mismatch. requested {env_hash!r}, bundle "
                f"{full[:16]!r}\n"
                "  env_hash is not a join key but an isolation boundary "
                "(§3.4). Do not mix data from different conditions.")

        noise = NoiseModel.from_bundle(b)
        env = b.env()
        hw = hardware_from_env(env)

        X = b.ranking(ok_only=ok_only, unknown_columns="warn")
        y = b.scoring(ok_only=ok_only)
        meta = {
            "bundle_id": b.info.get("bundle_id"),
            "schema_version": b.schema_version,
            "gpu_name": b.info.get("gpu_name"),
            "arch": b.info.get("arch"),
            "sm_count": b.info.get("sm_count"),
            "ok_only": ok_only,
            "shape_layers": b.shape_layers(),
            "tick_is_fallback": noise.tick_is_fallback,
        }
        return cls.from_frames(X, y, hw=hw, noise=noise, env_hash=full,
                              meta=meta, unexpected=unexpected)

    @classmethod
    def from_frames(cls, X: pd.DataFrame, y: pd.DataFrame, *,
                    hw: Hardware | None, noise: NoiseModel, env_hash: str,
                    meta: dict | None = None,
                    unexpected: str = "warn") -> PerfTable:
        """Builds after normalisation and join verification.

        `X` must be the result of `load_for_ranking` (no answers) and `y`
        that of `load_for_scoring` (answers included). **It checks that the
        two frames point at the same rows** — if they diverge because
        `ok_only` was given differently, scoring is silently wrong.
        """
        if len(X) != len(y):
            raise TableError(
                f"ranking {len(X)} rows != scoring {len(y)} rows. Check "
                f"that both were loaded with the same ok_only.")
        Xn = normalize(X, unexpected=unexpected)
        for c in _JOIN:
            if c not in y.columns:
                raise TableError(
                    f"the scoring table has no join key {c!r}.")
            a = Xn[c].to_numpy()
            b_ = y[c].to_numpy()
            if a.dtype.kind in "OU" or b_.dtype.kind in "OU":
                same = np.asarray([str(u) == str(v)
                                   for u, v in zip(a, b_, strict=True)])
            else:
                same = a == b_
            if not same.all():
                bad = int((~same).sum())
                raise TableError(
                    f"the ranking/scoring join is misaligned: {bad} rows "
                    f"differ on {c!r}. The premise that the row order is the "
                    f"same is broken.")
        if "time_ms" not in y.columns:
            raise TableError("the scoring table has no time_ms.")
        if "env_hash" in y.columns and y["env_hash"].nunique() > 1:
            raise TableError(
                f"the scoring table mixes {y['env_hash'].nunique()} "
                f"env_hash values. Do not aggregate across conditions "
                f"(§3.4).")

        meta = dict(meta or {})
        # Constant metadata is taken out of the rows and moved to `meta`.
        drop = ["env_hash"]
        for c in _CONSTANT_META:
            if c in Xn.columns:
                vals = Xn[c].unique()
                if len(vals) == 1:
                    meta.setdefault(c, vals[0])
                    drop.append(c)
        Xn = Xn.drop(columns=[c for c in drop if c in Xn.columns])
        for c in _CATEGORICAL:
            if c in Xn.columns and str(Xn[c].dtype) != "category":
                Xn[c] = Xn[c].astype("category")
        return cls(Xn, y["time_ms"].to_numpy(dtype=np.float64),
                   hw=hw, noise=noise, env_hash=env_hash, meta=meta)

    # -- Lookup -----------------------------------------------------------
    @property
    def hw(self) -> Hardware:
        if self._hw is None:
            raise TableError(
                "this table has no Hardware (env.json was not supplied).")
        return self._hw

    @property
    def noise(self) -> NoiseModel:
        return self._noise

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    @property
    def env_hash(self) -> str:
        return self._env_hash

    def shapes(self) -> list[Problem]:
        """The list of shapes. It keeps the order they appear in the table
        (determinism)."""
        return [Problem(M=k[0], N=k[1], K=k[2], dtype=k[3])
                for k in self._order]

    def frame_for(self, p: Problem) -> pd.DataFrame:
        """The **feature rows** of one shape. No answers. Used by
        FeatureMatrix."""
        return self._X.iloc[self._rows[self._k(p)]]

    def candidates(self, p: Problem) -> CandidateSet:
        """The candidate set handed to rules/features. **No times.**"""
        k = self._k(p)
        cs = self._cands.get(k)
        if cs is None:
            rows = self._rows[k]
            sub = self._X.iloc[rows]
            kid = sub["kernel_id"].astype(str).to_numpy()
            sk = sub["split_k"].to_numpy(dtype=np.int64)
            mode = sub["split_k_mode"].astype(str).to_numpy()
            cs = CandidateSet(
                n=len(rows), kernel_id=kid, split_k=sk, split_k_mode=mode,
                tiebreak=make_tiebreak(kid, sk, mode), row_index=rows)
            self._cands[k] = cs
        return cs

    def configs(self, p: Problem) -> tuple[Config, ...]:
        """An array of `Config` objects. For the deployment shim and
        reports. Not used on the scoring path."""
        sub = self.frame_for(p)
        return tuple(config_from_row(r) for r in sub.to_dict("records"))

    # -- The answer side (scorer only) ------------------------------------
    def times_of(self, p: Problem) -> np.ndarray:
        """★ **Scorer only.** The time array aligned with the candidate
        order (read-only).

        Handing this array to a rule function breaks §3's isolation. The
        call sites are limited to three (scoring / baselines / report).
        """
        v = self.__times[self._rows[self._k(p)]]
        v.setflags(write=False)
        return v

    def best_time(self, p: Problem) -> float:
        return self._stats[self._k(p)].best_ms

    def stats(self, p: Problem) -> ShapeStats:
        return self._stats[self._k(p)]

    def all_stats(self) -> list[ShapeStats]:
        return [self._stats[k] for k in self._order]

    def difficulty(self, p: Problem) -> float:
        """Median time / best time. **An `ANSWER_COLS` quantity. Used for
        stratification only.**"""
        return self._stats[self._k(p)].difficulty

    def answer_mask(self, p: Problem) -> np.ndarray:
        """A boolean mask of the candidates accepted as answers.

        The tolerance is **2σ of the per-shape noise floor** (§30.3), not a
        fixed 1% — on a 15µs kernel 1% is a difference that does not
        reproduce, so it would split noise into right and wrong answers.

        ⚠️ Use this instead of `kerneltab.core.table.answer_set()`. That one
        uses a **module-level global constant**, so on another GPU's bundle
        it silently uses the A6000 tick (see `core/noise.py`).
        """
        t = self.times_of(p)
        st = self._stats[self._k(p)]
        return t <= st.best_ms * (1.0 + st.answer_tol)

    # -- Internals --------------------------------------------------------
    def _k(self, p: Problem) -> ShapeKey:
        k = p.key if isinstance(p, Problem) else tuple(p)
        if k not in self._rows:
            raise KeyError(f"shape not in the table: {k}")
        return k

    def _build_stats(self) -> None:
        t_all = self.__times
        for k, rows in self._rows.items():
            t = t_all[rows]
            finite = t[np.isfinite(t) & (t > 0)]
            if finite.size == 0:
                raise TableError(
                    f"shape {k} has not one valid measurement. Check the "
                    f"table.")
            best = float(finite.min())
            med = float(np.median(finite))
            tol = self._noise.answer_tol(best)
            self._stats[k] = ShapeStats(
                key=k, n_candidates=int(t.size), best_ms=best, median_ms=med,
                difficulty=med / best,
                noise_floor=float(self._noise.floor(best)),
                answer_tol=tol,
                n_answers=int((t <= best * (1.0 + tol)).sum()),
                n_tied_at_best=int((t == best).sum()),
                n_distinct_times=int(np.unique(finite).size),
            )

    def summary(self) -> pd.DataFrame:
        """Per-shape answer-side statistics. **For reports and
        stratification only.**"""
        return pd.DataFrame([{
            "M": s.key[0], "N": s.key[1], "K": s.key[2], "dtype": s.key[3],
            "n_candidates": s.n_candidates, "best_ms": s.best_ms,
            "difficulty": s.difficulty, "noise_floor": s.noise_floor,
            "answer_tol": s.answer_tol, "n_answers": s.n_answers,
            "n_tied_at_best": s.n_tied_at_best,
            "distinct_time_frac": s.distinct_time_frac,
            "is_small": s.is_small,
        } for s in self.all_stats()])
