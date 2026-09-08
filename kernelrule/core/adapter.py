"""The schema contract and the adapter (§23).

kernelTab is still changing (`migration_plan.md` lists 11 items, `env_hash`
was redefined, the move to `schema_version` 2). Coding against today's schema
breaks when the main campaign's table arrives. So there is one thin layer
between the table and our types.

Principles (§26.4 — everything leans towards failing):

    a required column is missing   -> error     (no papering over with defaults)
    a new column appeared          -> warn and continue
    found via an alias             -> record it and continue
    a derived column cannot be built -> error
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pandas as pd

__all__ = [
    "ALIASES",
    "DERIVED",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "SchemaError",
    "SchemaReport",
    "check_schema",
    "normalize",
]

#: Our contract version. A different axis from kernelTab's `BUNDLE.json`
#: schema_version.
SCHEMA_VERSION = "1.0"


class SchemaError(RuntimeError):
    """The table does not satisfy the contract. **Do not proceed
    silently.**"""


#: Columns that must exist in the rule input (`load_for_ranking`).
#: Without them a `Config` cannot be built or a physical feature cannot be
#: computed.
REQUIRED_COLUMNS: frozenset[str] = frozenset({
    # shape
    "M", "N", "K", "dtype",
    # config (architecture-common) — transferable features must compute from
    # these alone (§4.3)
    "tile_m", "tile_n", "tile_k",
    "align_a", "align_b", "align_c",
    "split_k", "split_k_mode",
    "kernel_id", "arch",
    # Kernel attributes known at build time (§3.2 — allowed, no execution)
    "regs_per_thread", "threads", "max_blocks_per_sm", "pipeline_kind",
})

#: Used when present, skipped when not. **Absence is not an error.**
OPTIONAL_COLUMNS: frozenset[str] = frozenset({
    "acc_dtype", "layout_a", "layout_b", "layout_c",
    "ext_warp_m", "ext_warp_n", "ext_warp_k", "ext_stages",
    "ext_swizzle_type", "ext_swizzle_n",
    "hmma_count", "inst_total", "ldg_count", "lds_count", "sts_count",
    "ldsm_count", "cpasync_count",
    "workspace_bytes", "workspace_dtype", "partials_dtype",
    "theoretical_occupancy", "regs_total_per_block", "launchable",
    "local_bytes", "res_regs", "res_local", "build_seconds",
    "smem_matches", "hmma_matches", "expected_hmma", "cutlass_max_blocks",
    "env_hash", "bundle_id", "gpu_name", "sm_count", "clock_locked",
    "cutlass_commit", "nvcc_arch", "locked_mhz",
})

#: The table's names need not map 1:1 onto ours.
#: Value: (our name) -> (candidates to look for in the table, in order)
ALIASES: dict[str, tuple[str, ...]] = {
    "smem_bytes": ("smem_bytes", "smem_dynamic", "smem_computed",
                   "smem_static_bytes"),
}

#: Derived columns — built by the adapter (§23.3). Keeping the name the rule
#: uses separate from the table's name means only this changes when kernelTab
#: splits or merges a column.
DERIVED: dict[str, tuple[str, ...]] = {
    # Spills are recorded separately for loads and stores. The rule only
    # needs the total.
    "spill_bytes": ("spill_stores", "spill_loads"),
}


@dataclass
class SchemaReport:
    ok: bool
    n_rows: int
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    aliased: dict[str, str] = field(default_factory=dict)
    derived: list[str] = field(default_factory=list)

    def raise_if_bad(self) -> SchemaReport:
        if not self.ok:
            raise SchemaError(
                "the table does not satisfy the contract. Missing required "
                f"columns: {self.missing}\n"
                "  Check REQUIRED_COLUMNS / ALIASES / DERIVED in "
                "core/adapter.py. No papering over with defaults (§26.4).")
        return self

    def __str__(self) -> str:
        parts = [f"rows={self.n_rows}", f"ok={self.ok}"]
        if self.missing:
            parts.append(f"missing={self.missing}")
        if self.aliased:
            parts.append(f"aliased={self.aliased}")
        if self.derived:
            parts.append(f"derived={self.derived}")
        if self.unexpected:
            parts.append(f"new={len(self.unexpected)}")
        return "SchemaReport(" + ", ".join(parts) + ")"


def _resolve(df: pd.DataFrame, name: str) -> str | None:
    """Resolve `name` to the table's actual column name, scanning
    aliases."""
    if name in df.columns:
        return name
    for cand in ALIASES.get(name, ()):
        if cand in df.columns:
            return cand
    return None


def check_schema(df: pd.DataFrame, *, unexpected: str = "warn") -> SchemaReport:
    """Report required columns, aliases, derivability and new columns.

    `unexpected="warn"` is the default — kernelTab adding a column is normal,
    and blowing up each time would make the table unusable. It can be raised
    to `"raise"`.
    """
    cols = set(df.columns)
    missing: list[str] = []
    aliased: dict[str, str] = {}
    derived: list[str] = []

    for name in sorted(REQUIRED_COLUMNS):
        got = _resolve(df, name)
        if got is None:
            missing.append(name)
        elif got != name:
            aliased[name] = got

    for name in sorted(ALIASES):
        if name in REQUIRED_COLUMNS:
            continue
        got = _resolve(df, name)
        if got is None:
            missing.append(name)
        elif got != name:
            aliased[name] = got

    for name, sources in DERIVED.items():
        if name in cols:
            continue
        if all(s in cols for s in sources):
            derived.append(name)
        else:
            have = [s for s in sources if s in cols]
            missing.append(f"{name} (cannot derive: of {sources} only {have} "
                           "is present)")

    known = (REQUIRED_COLUMNS | OPTIONAL_COLUMNS | set(ALIASES)
             | set(DERIVED) | {s for v in DERIVED.values() for s in v}
             | {c for v in ALIASES.values() for c in v})
    unexpected_cols = sorted(cols - known)

    rep = SchemaReport(ok=not missing, n_rows=len(df), missing=missing,
                       unexpected=unexpected_cols, aliased=aliased,
                       derived=derived)
    if unexpected_cols:
        msg = (f"{len(unexpected_cols)} columns in the table are not in the "
               "contract: "
               f"{unexpected_cols[:12]}{' ...' if len(unexpected_cols) > 12 else ''}\n"
               "  kernelTab may have added a column. If it is derived from "
               "the answer it belongs in kernelTab's ANSWER_COLS; if it is a "
               "feature, add it to OPTIONAL_COLUMNS here. Until then it is "
               "ignored.")
        if unexpected == "raise":
            raise SchemaError(msg)
        if unexpected == "warn":
            warnings.warn(msg, stacklevel=2)
    return rep


def normalize(df: pd.DataFrame, *, unexpected: str = "warn") -> pd.DataFrame:
    """Normalise the table to our names: resolve aliases, build derived
    columns.

    ⚠️ **It neither creates nor carries answer columns.** The input must be
    the result of `load_for_ranking`. If `time_ms` is mixed in it raises here
    — an adapter that passes the answer through would make §3's isolation
    meaningless.
    """
    from kerneltab.core.table import ANSWER_COLS

    leaked = sorted(set(df.columns) & set(ANSWER_COLS))
    if leaked:
        raise SchemaError(
            f"answer columns reached normalize(): {leaked}\n"
            "  Pass the result of load_for_ranking(). The adapter does not "
            "pass the answer through (§3.2).")

    check_schema(df, unexpected=unexpected).raise_if_bad()

    out = df
    renames = {}
    for name in sorted(set(REQUIRED_COLUMNS) | set(ALIASES)):
        got = _resolve(df, name)
        if got is not None and got != name:
            renames[got] = name
    if renames:
        out = out.rename(columns=renames)

    for name in DERIVED:
        if name in out.columns:
            continue
        if name == "spill_bytes":
            out = out.assign(spill_bytes=(out["spill_stores"].fillna(0)
                                          + out["spill_loads"].fillna(0)
                                          ).astype("int64"))
        else:  # pragma: no cover - adding to DERIVED means filling this too
            raise SchemaError(f"no derivation rule is implemented: {name}")

    for name in ("acc_dtype", "layout_a", "layout_b", "layout_c"):
        if name not in out.columns:
            default = {"acc_dtype": "f32", "layout_a": "row",
                       "layout_b": "col", "layout_c": "row"}[name]
            out = out.assign(**{name: default})
    return out
