"""피처 행렬 (§21) — 오타와 배열 `if` 가 문법에서 걸린다."""
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
        """그리드가 GPU 를 몇 번 채우는가."""
        import math
        return (math.ceil(p.M / cfg.tile_m) * math.ceil(p.N / cfg.tile_n)
                * cfg.split_k / hw.sm_count)

    @feature(registry=r,
             vec=lambda df, hw, p: df.smem_bytes.to_numpy() / hw.smem_per_block)
    def smem_pressure(p, hw, cfg) -> float:
        """smem 예산을 얼마나 쓰는가."""
        return cfg.smem_bytes / hw.smem_per_block

    @shape_feature(registry=r, expected_range=(0.0, 1e6), unit="flop/byte")
    def arith_intensity(p, hw, cfg) -> float:
        """형상만의 함수. **스칼라**라서 규칙이 `if` 를 쓸 수 있다."""
        return 2.0 * p.M * p.N * p.K / (2.0 * (p.M * p.K + p.K * p.N + p.M * p.N))

    return r


def test_feats_typo_raises_immediately(synth_table, reg):
    """★ `f.tail_wast` 는 조용히 통과하지 않는다 (§21.3)."""
    m = FeatureMatrix(synth_table, reg)
    f, info = m.for_shape(synth_table.shapes()[0])
    assert isinstance(f.waves_like, np.ndarray)
    with pytest.raises(AttributeError, match="등록되지 않은 피처"):
        _ = f.waves_lik


def test_shape_info_typo_raises(synth_table, reg):
    m = FeatureMatrix(synth_table, reg)
    _, info = m.for_shape(synth_table.shapes()[0])
    with pytest.raises(AttributeError, match="등록되지 않은"):
        _ = info.arith_intensity_typo


def test_config_level_if_is_a_type_error(synth_table, reg):
    """★ config 수준 조건부 특수화가 **문법적으로** 어렵다 (§8.1 대체본).

    `f.<피처>` 는 배열이므로 `if` 가 ValueError 를 낸다.
    """
    m = FeatureMatrix(synth_table, reg)
    f, _ = m.for_shape(synth_table.shapes()[0])
    with pytest.raises(ValueError):
        if f.waves_like < 1.0:      # noqa: SIM103
            pass


def test_shape_level_if_works(synth_table, reg):
    """반대로 형상 수준 조건은 스칼라라 `if` 가 그대로 된다."""
    m = FeatureMatrix(synth_table, reg)
    _, info = m.for_shape(synth_table.shapes()[0])
    assert isinstance(info.arith_intensity, float)
    if info.arith_intensity > 0:
        pass


def test_vectorized_matches_scalar(synth_table, reg):
    """★ 벡터화 구현이 스칼라와 다르면 학습과 배포가 다른 함수를 쓴다."""
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
    with pytest.raises(ValueError, match="벡터화 구현이 스칼라와 다르다"):
        verify_vectorized(bad, df, synth_table.hw, info, n=64)


def test_registry_refuses_in_place_modification(reg):
    """★ in-place 수정 금지 (§8.4). 과거 실험이 무효가 된다."""
    with pytest.raises(ValueError, match="in-place 수정 금지"):
        @feature(registry=reg)
        def smem_pressure(p, hw, cfg) -> float:      # noqa: F811
            return 0.0


def test_deprecate_keeps_the_feature_runnable(reg):
    r = FeatureRegistry("dep")

    @feature(registry=r)
    def dup(p, hw, cfg) -> float:
        return 1.0

    r.deprecate("dup", at_round=47, reason="waves 와 스피어만 0.97, 중복")
    assert "dup" not in r.names()                  # 새 규칙의 프롬프트에서 빠지고
    assert "dup" in r.names(active_only=False)     # 기존 규칙은 계속 돈다
    assert r["dup"].deprecation_reason


def test_nonfinite_feature_is_rejected(synth_table):
    """★ 피처가 비유한 값을 내면 **기각**이다. 승인이 아니다 (§26.4)."""
    r = FeatureRegistry("bad")

    @feature(registry=r, vec=lambda df, hw, p: np.full(len(df), np.nan))
    def broken(p, hw, cfg) -> float:
        return float("nan")

    with pytest.raises(ValueError, match="비유한"):
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
    """★ `hw` 를 바꿨는데 결과가 안 바뀌면 하드웨어를 안 쓰는 것이다 (§8.3 6번)."""
    a = FeatureMatrix(synth_table, reg)
    b = FeatureMatrix(synth_table, reg, hw=hw_other)
    p = synth_table.shapes()[0]
    fa, _ = a.for_shape(p)
    fb, _ = b.for_shape(p)
    assert not np.allclose(fa.waves_like, fb.waves_like)
    assert not np.allclose(fa.smem_pressure, fb.smem_pressure)
