"""스키마 계약 (§23). 전부 **실패 쪽으로 기운다** (§26.4)."""
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
    """필수 컬럼이 없으면 **에러**다. 기본값으로 때우지 않는다."""
    df = _minimal().drop(columns=["tile_k"])
    rep = check_schema(df, unexpected="ignore")
    assert not rep.ok and "tile_k" in rep.missing
    with pytest.raises(SchemaError, match="tile_k"):
        rep.raise_if_bad()
    with pytest.raises(SchemaError):
        normalize(df, unexpected="ignore")


def test_alias_resolution():
    """`smem_bytes` 는 `smem_dynamic` 으로 들어온다."""
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
    """kernelTab 이 컬럼을 추가하는 것은 정상이다. 터지면 표를 못 쓴다."""
    df = _minimal()
    df["ext_cluster_m"] = 2
    with pytest.warns(UserWarning, match="계약에 없는 컬럼"):
        rep = check_schema(df, unexpected="warn")
    assert rep.ok and "ext_cluster_m" in rep.unexpected
    with pytest.raises(SchemaError):
        check_schema(df, unexpected="raise")


def test_normalize_refuses_answer_columns():
    """★ 어댑터가 정답을 통과시키는 경로가 되면 §3 의 격리가 무의미해진다."""
    df = _minimal()
    df["time_ms"] = 0.5
    with pytest.raises(SchemaError, match="정답 컬럼"):
        normalize(df, unexpected="ignore")


def test_normalize_refuses_difficulty():
    """`difficulty` 도 정답이다 — 정답에서 유도됐고 배포 시점에 알 수 없다."""
    df = _minimal()
    df["difficulty"] = 1.5
    with pytest.raises(SchemaError, match="정답 컬럼"):
        normalize(df, unexpected="ignore")


@pytest.mark.needs_bundle
def test_real_bundle_matches_contract(real_bundle_path):
    """실제 번들이 계약을 만족한다 (§23.4). 번들이 없으면 스킵되며 표시된다."""
    import warnings

    from kerneltab.core.bundle import load_bundle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = load_bundle(real_bundle_path).ranking(ok_only=False,
                                                  unknown_columns="ignore")
        rep = check_schema(X, unexpected="ignore")
    assert rep.ok, f"실제 번들에 누락된 필수 컬럼: {rep.missing}"
    out = normalize(X, unexpected="ignore")
    assert {"smem_bytes", "spill_bytes"} <= set(out.columns)


def test_synthetic_table_matches_contract(synth_bundles):
    """합성 표도 같은 계약을 만족해야 로더/어댑터가 검증된다 (§22.3)."""
    import warnings

    from kerneltab.core.bundle import load_bundle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = load_bundle(synth_bundles["normal"]).ranking(
            ok_only=False, unknown_columns="ignore")
    assert check_schema(X, unexpected="ignore").ok
