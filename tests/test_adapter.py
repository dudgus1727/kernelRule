"""The schema contract (§23). Everything **leans towards failing** (§26.4)."""
from __future__ import annotations

import pandas as pd
import pytest

from kernelrule.core.adapter import (
    REQUIRED_COLUMNS,
    SchemaError,
    check_schema,
    normalize,
)


def _minimal() -> pd.DataFrame:
    row = dict.fromkeys(REQUIRED_COLUMNS, 1)
    row.update({"dtype": "f16", "split_k_mode": "serial", "kernel_id": "k0",
                "arch": "sm_86", "pipeline_kind": "multistage",
                "smem_dynamic": 32768, "spill_stores": 0, "spill_loads": 0})
    return pd.DataFrame([row])


def test_minimal_frame_satisfies_contract():
    rep = check_schema(_minimal(), unexpected="ignore")
    assert rep.ok, rep.missing


def test_missing_required_column_is_an_error():
    """A missing required column is an **error**. No papering over with
    defaults."""
    df = _minimal().drop(columns=["tile_k"])
    rep = check_schema(df, unexpected="ignore")
    assert not rep.ok and "tile_k" in rep.missing
    with pytest.raises(SchemaError, match="tile_k"):
        rep.raise_if_bad()
    with pytest.raises(SchemaError):
        normalize(df, unexpected="ignore")


def test_alias_resolution():
    """`smem_bytes` arrives as `smem_dynamic`."""
    rep = check_schema(_minimal(), unexpected="ignore")
    assert rep.aliased.get("smem_bytes") == "smem_dynamic"
    out = normalize(_minimal(), unexpected="ignore")
    assert "smem_bytes" in out.columns


def test_derived_spill_bytes():
    df = _minimal()
    df["spill_stores"], df["spill_loads"] = 12, 30
    out = normalize(df, unexpected="ignore")
    assert out["spill_bytes"].iloc[0] == 42


def test_derived_impossible_is_an_error():
    df = _minimal().drop(columns=["spill_loads"])
    rep = check_schema(df, unexpected="ignore")
    assert not rep.ok and any("spill_bytes" in m for m in rep.missing)


def test_new_column_warns_but_proceeds():
    """kernelTab adding a column is normal. Blowing up makes the table
    unusable."""
    df = _minimal()
    df["ext_cluster_m"] = 2
    with pytest.warns(UserWarning, match="not in the\n *contract|not in the contract"):
        rep = check_schema(df, unexpected="warn")
    assert rep.ok and "ext_cluster_m" in rep.unexpected
    with pytest.raises(SchemaError):
        check_schema(df, unexpected="raise")


def test_normalize_refuses_answer_columns():
    """★ An adapter that passes the answer through makes §3's isolation
    meaningless."""
    df = _minimal()
    df["time_ms"] = 0.5
    with pytest.raises(SchemaError, match="answer columns"):
        normalize(df, unexpected="ignore")


def test_normalize_refuses_difficulty():
    """`difficulty` is an answer too — derived from it and unknown at
    deployment time."""
    df = _minimal()
    df["difficulty"] = 1.5
    with pytest.raises(SchemaError, match="answer columns"):
        normalize(df, unexpected="ignore")


@pytest.mark.needs_bundle
def test_real_bundle_matches_contract(real_bundle_path):
    """A real bundle satisfies the contract (§23.4). Without a bundle it is
    skipped, and that is shown."""
    import warnings

    from kerneltab.core.bundle import load_bundle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = load_bundle(real_bundle_path).ranking(ok_only=False,
                                                  unknown_columns="ignore")
        rep = check_schema(X, unexpected="ignore")
    assert rep.ok, f"required columns missing from the real bundle: {rep.missing}"
    out = normalize(X, unexpected="ignore")
    assert {"smem_bytes", "spill_bytes"} <= set(out.columns)


def test_synthetic_table_matches_contract(synth_bundles):
    """The synthetic table must satisfy the same contract for the
    loader/adapter to be verified (§22.3)."""
    import warnings

    from kerneltab.core.bundle import load_bundle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = load_bundle(synth_bundles["normal"]).ranking(
            ok_only=False, unknown_columns="ignore")
    assert check_schema(X, unexpected="ignore").ok
