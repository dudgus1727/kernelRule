"""Precomputing the feature matrix (§21) — makes scoring 300x faster.

## Why it is needed (appendix ★ fix 1)

The original design said "scoring takes a few milliseconds", and that was
wrong. There are about 15,000 candidates per shape, so `score()` is called
12 rules x 66 shapes x 15,000 ~= 11.9 million times per round. In pure
Python that is 2 minutes per round, 100 minutes of scoring alone over 50
rounds. That breaks the design premise that "scoring is free, so 12 rules
are produced in parallel".

A feature is a **pure function** of `(Problem, Hardware, Config)` and has
nothing to do with the rule. Whatever rule comes along, the value for the
same (shape, config) pair is the same. **Compute it once, lift it into a
matrix, and let the rule become numpy operations over that matrix.**

## The asymmetry between Feats and ShapeInfo is the design (§8.1 replacement)

    f.<name>  ->  a (n_candidates,) array   `if f.waves < 1:` is a ValueError
    p.<name>  ->  a scalar                   `if p.is_memory_bound:` works

It makes config-level conditional specialisation (= the road to a lookup
table) **syntactically** hard, while shape-level branching (= the kind that
generalises) stays allowed.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kernelrule.core.table import PerfTable
from kernelrule.core.types import Hardware, Problem, config_from_row
from kernelrule.features import FeatureRegistry

__all__ = ["FeatureMatrix", "Feats", "ShapeInfo"]


class _DictAttr:
    """Fetches values by name, but **a typo does not pass silently**
    (§21.3)."""

    __slots__ = ("_cols", "_kind")

    def __init__(self, cols: dict, kind: str) -> None:
        object.__setattr__(self, "_cols", cols)
        object.__setattr__(self, "_kind", kind)

    def __getattr__(self, name: str):
        # ★ Names starting with an underscore are refused immediately.
        #   Otherwise, in a state where `_cols` does not exist yet (e.g.
        #   restoring from a pickle), `__getattr__("_cols")` calls itself
        #   again and falls into **infinite recursion**. The sandbox caught
        #   this.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            cols = object.__getattribute__(self, "_cols")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return cols[name]
        except KeyError:
            kind = object.__getattribute__(self, "_kind")
            raise AttributeError(
                f"unregistered {kind}: {name!r}. "
                f"available: {sorted(cols)}") from None

    def __setattr__(self, name, value):     # pragma: no cover
        raise AttributeError(f"{self._kind} is read-only.")

    # -- pickle. Used when the sandbox hands this to a child process -------
    # The `__slots__` + forbidden `__setattr__` combination breaks the
    # default path.
    def __getstate__(self):
        return {"_cols": object.__getattribute__(self, "_cols"),
                "_kind": object.__getattribute__(self, "_kind")}

    def __setstate__(self, state):
        object.__setattr__(self, "_cols", state["_cols"])
        object.__setattr__(self, "_kind", state["_kind"])

    def __contains__(self, name: str) -> bool:
        return name in self._cols

    def __len__(self) -> int:
        return len(self._cols)

    def keys(self):
        return self._cols.keys()


#: ★ Shape fields that are always present, **independent of** the registry.
#: They are properties of the problem itself, so `p.M` and the like can be
#: used under any condition (F0~F3). The registry-swap check must treat them
#: as an exception — they are not features (§30.9).
INTRINSIC_SHAPE_FIELDS = ("M", "N", "K", "n_candidates")


class Feats(_DictAttr):
    """Config-level features. Each value is an `(n_candidates,)` numpy
    array.

    `if f.tail_waste < 0.1:` raises `ValueError` — because it is an array.
    The pressure to use `np.where` instead of conditional specialisation
    comes out of the syntax.
    """

    def __init__(self, cols: dict[str, np.ndarray]) -> None:
        super().__init__(cols, "feature")


class ShapeInfo(_DictAttr):
    """Shape-level values. All **scalars**, so `if` works directly."""

    def __init__(self, cols: dict[str, float]) -> None:
        super().__init__(cols, "shape-level value")


@dataclass
class MatrixStats:
    n_shapes: int
    n_rows: int
    n_features: int
    build_seconds: float
    bytes_: int
    from_cache: bool

    def __str__(self) -> str:
        src = "cache" if self.from_cache else "computed"
        return (f"FeatureMatrix({src}): {self.n_shapes} shapes x "
                f"{self.n_rows} rows "
                f"x {self.n_features} features, {self.bytes_/1e6:.0f}MB, "
                f"{self.build_seconds:.1f}s")


def _configs_of(df) -> list:
    """One shape's rows as `Config` objects. ★ Built once and handed to
    every feature (D-161) — the conversion is what a column costs."""
    return [config_from_row(r) for r in df.to_dict("records")]


class FeatureMatrix:
    """(shape, config) -> every feature value. Computed once when the table
    is loaded."""

    def __init__(self, table: PerfTable, registry: FeatureRegistry, *,
                 hw: Hardware | None = None,
                 cache_dir: str | Path | None = None,
                 shapes=None,
                 verbose: bool = False) -> None:
        self.table = table
        self.registry = registry
        self.hw = hw if hw is not None else table.hw
        self.cache_dir = Path(cache_dir) if cache_dir else None
        #: ★ 2026-09-11 (D-161): which shapes this matrix holds. `None` is
        #: every shape in the table — what every scoring path wants.
        #: ⚠️ **Not `_shapes`** — that name is `Split`'s sealed private field
        #: and `test_nothing_reads_the_private_field` forbids reading it
        #: anywhere outside `splits.py` (§30.15). A matrix's shape list is a
        #: different thing and must not weaken that guard.
        #: A **probe** matrix built to answer one question about a candidate
        #: feature does not need all of them, and the duplication check only
        #: ever looks at 4 shapes while paying for 66 (`_reference_columns`).
        #: ⚠️ A partial matrix is not cached — the cache key does not carry
        #: the shape list and a partial one would be served as complete.
        self._held = (list(table.shapes()) if shapes is None
                        else list(shapes))
        self.partial = shapes is not None
        self._cols: dict[tuple, dict[str, np.ndarray]] = {}
        self._info: dict[tuple, dict[str, float]] = {}
        self.stats: MatrixStats | None = None
        self._build(verbose=verbose)

    def shapes(self) -> list:
        """The shapes this matrix actually holds."""
        return list(self._held)

    def has_column(self, name: str) -> bool:
        """Is this axis already computed? Used to skip a recomputation that
        would produce the identical column (D-161)."""
        if not self._cols:
            return False
        k = self._held[0].key
        return name in self._cols[k] or name in self._info[k]

    # -- Lookup -----------------------------------------------------------
    def for_shape(self, p: Problem) -> tuple[Feats, ShapeInfo]:
        k = p.key
        return Feats(self._cols[k]), ShapeInfo(self._info[k])

    def feature_names(self) -> list[str]:
        return self.registry.names(shape_level=False)

    def shape_value_names(self) -> list[str]:
        return sorted(next(iter(self._info.values())))

    def feature_mins(self) -> dict:
        """The **minimum of the declared range** per axis. Used by the
        exponent-slot guard (D-112).

        ★ It is the declared range, not the observed one. Judging by
        observation lets an axis that happened to be positive in this table
        turn negative in another.
        """
        return {n: float(self.registry[n].expected_range[0])
                for n in self.registry.names(shape_level=False)}

    def observed_ranges(self, shapes=None) -> dict[str, tuple[float, float]]:
        """★ The **actual** min/max each axis takes on those shapes' configs
        (D-160).

        The declared `expected_range` is what the model said; this is what
        the table says. Measured on the F1 run, 15 of 20 declarations were
        the schema default [0,1].

        ⛔ **It must not go into a prompt.** A range read off this table is
        an observation of this table, and putting it in front of the model
        makes the run condition B (`describe_with(include_observed=...)`
        holds that line). It is written to the artefacts and the trace only
        — and it is not put in `Feature.observed`, which is the field that
        does reach the prompt.

        `shapes` defaults to every shape in the table. The caller passes the
        **training** shapes so that nothing from the holdout is measured.
        """
        shapes = list(self._held if shapes is None else shapes)
        out: dict[str, tuple[float, float]] = {}
        for n in self.registry.names(shape_level=False):
            lo = min(float(self._cols[p.key][n].min()) for p in shapes)
            hi = max(float(self._cols[p.key][n].max()) for p in shapes)
            out[n] = (lo, hi)
        for n in self.registry.names(shape_level=True):
            vals = [float(self._info[p.key][n]) for p in shapes]
            out[n] = (min(vals), max(vals))
        return out

    def column(self, name: str) -> np.ndarray:
        """The column with every shape concatenated. For the GBDT baseline
        and correlation analysis."""
        return np.concatenate([self._cols[p.key][name]
                               for p in self._held])

    def adopt(self, other: FeatureMatrix, name: str) -> None:
        """Take one already-computed column from another matrix built on the
        **same table** (D-161).

        Registering an axis used to compute the same column three times: the
        probe matrix inside `register_generated`, `detect_shape_level`, and
        then `invalidate` here. The values are identical by construction —
        same function, same rows — so they are copied instead.
        """
        if other.table is not self.table:
            raise ValueError(
                "adopt() only works between matrices on the same table — "
                "copying a column across tables would silently mix two "
                "measurements")
        f = self.registry[name]
        for p in self._held:
            k = p.key
            if f.shape_level:
                self._info[k][name] = other._info[k][name]
            else:
                self._cols[k][name] = other._cols[k][name]

    def invalidate(self, feature_name: str) -> None:
        """When a feature is added or changed, recompute **only that
        column** (§21.2)."""
        f = self.registry[feature_name]
        for p in self._held:
            df = self.table.frame_for(p)
            info = self._info[p.key]
            if f.shape_level:
                info[f.name] = self._scalar(f, p, df)
            else:
                self._cols[p.key][f.name] = self._vector(f, p, df, info)

    # -- Computation ------------------------------------------------------
    def _cache_key(self) -> str:
        m = self.table.meta
        h = hashlib.sha256()
        h.update(str(self.table.env_hash).encode())
        h.update(str(m.get("ok_only")).encode())
        h.update(str(m.get("bundle_id")).encode())
        h.update(self.registry.lock_hash().encode())
        return h.hexdigest()[:16]

    def _build(self, *, verbose: bool) -> None:
        t0 = time.perf_counter()
        # ★ A partial matrix is never cached (D-161) — the key does not
        #   carry the shape list, so it would later be served as a complete
        #   one.
        path = (self.cache_dir / f"featmat-{self._cache_key()}.npz"
                if self.cache_dir and not self.partial else None)
        if path is not None and path.exists():
            self._load_cache(path)
            self._finish_stats(t0, from_cache=True)
            return

        cfg_feats = self.registry.items(shape_level=False)
        shp_feats = self.registry.items(shape_level=True)
        for p in self._held:
            df = self.table.frame_for(p)
            info: dict[str, float] = dict(zip(
                INTRINSIC_SHAPE_FIELDS,
                (float(p.M), float(p.N), float(p.K), float(len(df))),
                strict=True))
            for f in shp_feats:
                info[f.name] = self._scalar(f, p, df)
            cols: dict[str, np.ndarray] = {}
            # ★ 2026-09-11 (D-161): the rows are turned into `Config`
            #   objects **once per shape**, not once per feature. Measured
            #   on 8 shapes / 124,740 rows: `to_dict` 1.4s + `config_from_row`
            #   0.8s against 0.02s for the feature function itself — the
            #   conversion *was* the cost, and every feature paid it again.
            cfgs = _configs_of(df) if cfg_feats else []
            for f in cfg_feats:
                cols[f.name] = self._vector(f, p, df, info, cfgs=cfgs)
            self._cols[p.key] = cols
            self._info[p.key] = info
            if verbose:
                print(f"  {p.key} {len(df)} rows", flush=True)

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._save_cache(path)
        self._finish_stats(t0, from_cache=False)

    def _scalar(self, f, p: Problem, df) -> float:
        """A shape-level feature. Computed from one representative config
        (it must be Config-independent).

        ⚠️ Exceptions are not swallowed (§26.4). A feature that blows up is
        a rejection, not an approval.
        """
        cfg = config_from_row(df.iloc[0].to_dict())
        v = float(f.fn(p, self.hw, cfg))
        if not np.isfinite(v):
            raise ValueError(
                f"{f.name}({p.key}) produced the non-finite value {v}. "
                f"Rejected.")
        return v

    def _vector(self, f, p: Problem, df, info: dict, cfgs=None) -> np.ndarray:
        if f.vec is not None:
            out = np.asarray(f.vec(df, self.hw, ShapeInfo(info)),
                             dtype=np.float64)
            if out.shape != (len(df),):
                raise ValueError(
                    f"{f.name}: the vectorised implementation produced "
                    f"{out.shape}. It must be ({len(df)},).")
        else:
            rows = _configs_of(df) if cfgs is None else cfgs
            out = np.asarray([float(f.fn(p, self.hw, c)) for c in rows],
                             dtype=np.float64)
        if not np.all(np.isfinite(out)):
            n_bad = int((~np.isfinite(out)).sum())
            raise ValueError(
                f"{f.name}({p.key}) has {n_bad} non-finite values. "
                f"Rejected (§26.4).")
        return out

    def _finish_stats(self, t0: float, *, from_cache: bool) -> None:
        n_rows = sum(len(next(iter(c.values()))) if c else 0
                     for c in self._cols.values())
        nfeat = len(self.feature_names())
        self.stats = MatrixStats(
            n_shapes=len(self._cols), n_rows=n_rows, n_features=nfeat,
            build_seconds=time.perf_counter() - t0,
            bytes_=n_rows * nfeat * 8, from_cache=from_cache)

    # -- Cache ------------------------------------------------------------
    def _save_cache(self, path: Path) -> None:
        blob: dict[str, np.ndarray] = {}
        for k, cols in self._cols.items():
            tag = "|".join(str(x) for x in k)
            for name, arr in cols.items():
                blob[f"c::{tag}::{name}"] = arr
            blob[f"i::{tag}"] = np.asarray(
                [self._info[k][n] for n in sorted(self._info[k])],
                dtype=np.float64)
            blob[f"n::{tag}"] = np.asarray(sorted(self._info[k]), dtype=object)
        np.savez_compressed(path, **blob, allow_pickle=True)

    def _load_cache(self, path: Path) -> None:
        z = np.load(path, allow_pickle=True)
        for key in z.files:
            if key.startswith("c::"):
                _, tag, name = key.split("::", 2)
                k = self._tag_to_key(tag)
                self._cols.setdefault(k, {})[name] = z[key]
            elif key.startswith("i::"):
                tag = key[3:]
                k = self._tag_to_key(tag)
                names = list(z[f"n::{tag}"])
                self._info[k] = dict(zip([str(n) for n in names],
                                         [float(v) for v in z[key]],
                                         strict=True))

    @staticmethod
    def _tag_to_key(tag: str) -> tuple:
        m, n, kk, dt = tag.split("|")
        return (int(m), int(n), int(kk), dt)
