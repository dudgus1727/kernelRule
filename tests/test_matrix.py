"""The feature matrix (§21) — typos and array `if` are caught by the
syntax."""
from __future__ import annotations

import numpy as np
import pytest

from kernelrule.core.matrix import FeatureMatrix
from kernelrule.features import FeatureRegistry, feature, shape_feature


@pytest.fixture(scope="module")
def reg():
    r = FeatureRegistry("test")

    @feature(registry=r, expected_range=(0.0, 1.0),
             vec=lambda df, hw, p: (np.ceil(df.M.to_numpy() / df.tile_m.to_numpy())
                                    * np.ceil(df.N.to_numpy() / df.tile_n.to_numpy())
                                    * df.split_k.to_numpy() / hw.sm_count))
    def waves_like(p, hw, cfg) -> float:
        """How many times the grid fills the GPU."""
        import math
        return (math.ceil(p.M / cfg.tile_m) * math.ceil(p.N / cfg.tile_n)
                * cfg.split_k / hw.sm_count)

    @feature(registry=r,
             vec=lambda df, hw, p: df.smem_bytes.to_numpy() / hw.smem_per_block)
    def smem_pressure(p, hw, cfg) -> float:
        """How much of the smem budget it uses."""
        return cfg.smem_bytes / hw.smem_per_block

    @shape_feature(registry=r, expected_range=(0.0, 1e6), unit="flop/byte")
    def arith_intensity(p, hw, cfg) -> float:
        """A function of the shape alone. It is a **scalar**, so a rule may
        use `if`."""
        return 2.0 * p.M * p.N * p.K / (2.0 * (p.M * p.K + p.K * p.N + p.M * p.N))

    return r


def test_feats_typo_raises_immediately(synth_table, reg):
    """★ `f.tail_wast` does not pass silently (§21.3)."""
    m = FeatureMatrix(synth_table, reg)
    f, info = m.for_shape(synth_table.shapes()[0])
    assert isinstance(f.waves_like, np.ndarray)
    with pytest.raises(AttributeError, match="unregistered feature"):
        _ = f.waves_lik


def test_shape_info_typo_raises(synth_table, reg):
    m = FeatureMatrix(synth_table, reg)
    _, info = m.for_shape(synth_table.shapes()[0])
    with pytest.raises(AttributeError, match="unregistered"):
        _ = info.arith_intensity_typo


def test_config_level_if_is_a_type_error(synth_table, reg):
    """★ Config-level conditional specialisation is **syntactically** hard
    (the §8.1 replacement).

    `f.<feature>` is an array, so `if` raises ValueError.
    """
    m = FeatureMatrix(synth_table, reg)
    f, _ = m.for_shape(synth_table.shapes()[0])
    with pytest.raises(ValueError):
        if f.waves_like < 1.0:      # noqa: SIM103
            pass


def test_shape_level_if_works(synth_table, reg):
    """Conversely, a shape-level condition is a scalar, so `if` works
    directly."""
    m = FeatureMatrix(synth_table, reg)
    _, info = m.for_shape(synth_table.shapes()[0])
    assert isinstance(info.arith_intensity, float)
    if info.arith_intensity > 0:
        pass


def test_vectorized_matches_scalar(synth_table, reg):
    """★ If the vectorised implementation differs from the scalar one,
    training and deployment use different functions."""
    from kernelrule.features import verify_vectorized

    p = synth_table.shapes()[0]
    df = synth_table.frame_for(p)
    m = FeatureMatrix(synth_table, reg)
    _, info = m.for_shape(p)
    for name in ("waves_like", "smem_pressure"):
        verify_vectorized(reg[name], df, synth_table.hw, info, n=64)


def test_vectorized_mismatch_is_rejected(synth_table, reg):
    from kernelrule.features import Feature, verify_vectorized

    p = synth_table.shapes()[0]
    df = synth_table.frame_for(p)
    bad = Feature(**{**reg["smem_pressure"].__dict__,
                     "vec": lambda d, hw, i: np.zeros(len(d)) + 0.5})
    m = FeatureMatrix(synth_table, reg)
    _, info = m.for_shape(p)
    with pytest.raises(ValueError,
                       match="differs from the scalar one"):
        verify_vectorized(bad, df, synth_table.hw, info, n=64)


def test_registry_refuses_in_place_modification(reg):
    """★ In-place modification is forbidden (§8.4). Past experiments would
    become invalid."""
    with pytest.raises(ValueError, match="In-place"):
        @feature(registry=reg)
        def smem_pressure(p, hw, cfg) -> float:      # noqa: F811
            return 0.0


def test_deprecate_keeps_the_feature_runnable(reg):
    r = FeatureRegistry("dep")

    @feature(registry=r)
    def dup(p, hw, cfg) -> float:
        return 1.0

    r.deprecate("dup", at_round=47,
                reason="Spearman 0.97 with waves, a duplicate")
    assert "dup" not in r.names()                # drops out of new prompts
    assert "dup" in r.names(active_only=False)   # existing rules keep running
    assert r["dup"].deprecation_reason


def test_nonfinite_feature_is_rejected(synth_table):
    """★ A feature producing a non-finite value is a **rejection**, not an
    approval (§26.4)."""
    r = FeatureRegistry("bad")

    @feature(registry=r, vec=lambda df, hw, p: np.full(len(df), np.nan))
    def broken(p, hw, cfg) -> float:
        return float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        FeatureMatrix(synth_table, r)


def test_matrix_cache_roundtrip(synth_table, reg, tmp_path):
    a = FeatureMatrix(synth_table, reg, cache_dir=tmp_path)
    assert a.stats.from_cache is False
    b = FeatureMatrix(synth_table, reg, cache_dir=tmp_path)
    assert b.stats.from_cache is True
    p = synth_table.shapes()[0]
    fa, ia = a.for_shape(p)
    fb, ib = b.for_shape(p)
    assert np.allclose(fa.waves_like, fb.waves_like)
    assert ia.arith_intensity == pytest.approx(ib.arith_intensity)


def test_scale_invariance_hw_actually_used(synth_table, reg, hw_other):
    """★ If `hw` changes and the result does not, the hardware is not being
    used (§8.3 item 6)."""
    a = FeatureMatrix(synth_table, reg)
    b = FeatureMatrix(synth_table, reg, hw=hw_other)
    p = synth_table.shapes()[0]
    fa, _ = a.for_shape(p)
    fb, _ = b.for_shape(p)
    assert not np.allclose(fa.waves_like, fb.waves_like)
    assert not np.allclose(fa.smem_pressure, fb.smem_pressure)
