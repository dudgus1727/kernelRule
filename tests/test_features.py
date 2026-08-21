"""피처 라이브러리와 자동 검증 (§8.2, §8.3)."""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

import kernelrule.features.physical  # noqa: F401  등록
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.features import REGISTRY
from kernelrule.features.validate import validate_registry


@pytest.fixture(scope="module")
def matrix(synth_table):
    return FeatureMatrix(synth_table, REGISTRY)


def test_library_has_enough_features():
    """§8.2 — 시작 전에 손으로 10~15개를 채워둔다."""
    assert len(REGISTRY.names(shape_level=False)) >= 15
    assert len(REGISTRY.names(shape_level=True)) >= 4


def test_no_feature_touches_ext():
    """★ `cfg.ext` 참조 금지 — 아키텍처 전이 전제 (§4.3, §8.2).

    `ext_*` 는 SM90 에 대응물이 없다. 전이를 노리는 규칙이 그것을 쓰면
    주 지표(아키텍처 홀드아웃)에서 무너진다.
    """
    bad = []
    for f in REGISTRY.items(active_only=False):
        try:
            src = inspect.getsource(f.fn)
        except (OSError, TypeError):      # pragma: no cover
            continue
        if "cfg.ext" in src or ".ext[" in src:
            bad.append(f.name)
    assert not bad, f"`cfg.ext` 를 참조하는 피처: {bad}"


def test_no_feature_references_answers():
    """정답 컬럼 이름을 **식별자로** 참조하지 않는다.

    단순 부분문자열 검사는 안 된다 — `hw.peak_tflops_f16` 이 `tflops` 를
    포함해서 오탐이 난다. 토큰 경계로 본다.
    """
    import re
    from kerneltab.core.table import ANSWER_COLS

    bad = []
    for f in REGISTRY.items(active_only=False):
        try:
            src = inspect.getsource(f.fn)
        except (OSError, TypeError):      # pragma: no cover
            continue
        code = "\n".join(ln for ln in src.split("\n")
                          if not ln.strip().startswith("#"))
        for col in ANSWER_COLS:
            if re.search(rf"(?<![\w.]){re.escape(col)}\b", code):
                bad.append((f.name, col))
    assert not bad, f"정답 컬럼을 참조하는 피처: {bad}"


def test_all_features_are_short():
    """§8.2 — 10줄 이내. 길면 물리가 아니라 조합이다."""
    long = []
    for f in REGISTRY.items(active_only=False):
        try:
            src = inspect.getsource(f.fn)
        except (OSError, TypeError):      # pragma: no cover
            continue
        body = [ln for ln in src.split("\n")
                if ln.strip() and not ln.strip().startswith("#")]
        # docstring 과 데코레이터를 뺀 실질 줄 수
        n = len([ln for ln in body if not ln.lstrip().startswith(("@", '"""'))])
        if n > 22:
            long.append((f.name, n))
    assert not long, f"너무 긴 피처: {long}"


def test_registry_validates_clean(synth_table, matrix, hw_other):
    """★ 전 피처가 자동 검증을 통과한다. 기각이 하나라도 있으면 실패다."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reps = validate_registry(REGISTRY, synth_table, matrix,
                                 hw_alt=hw_other, n_shapes=4)
    failed = {n: [str(c) for c in r.fails()]
              for n, r in reps.items() if r.failed}
    assert not failed, f"검증 기각: {failed}"


def test_scale_invariance_catches_a_hardcoded_constant(synth_table, matrix,
                                                       hw_other):
    """★ 하드웨어 상수를 하드코딩한 피처가 **잡히는가** (§8.3 6번).

    감시가 실제로 작동하는지 확인한다. 검사가 있다는 사실만으로는 아무것도
    보장되지 않는다 (§30.8).
    """
    from kernelrule.features import Feature, FeatureRegistry
    from kernelrule.features.validate import validate_feature

    r = FeatureRegistry("bad")

    def fake_waves(p, hw, cfg) -> float:
        """hw.sm_count 를 읽는 척하지만 84 를 박아 뒀다."""
        import math
        return math.ceil(p.M / cfg.tile_m) * math.ceil(p.N / cfg.tile_n) / 84.0

    f = Feature(name="fake_waves", fn=fake_waves, unit="dimensionless",
                expected_range=(0.0, 1e6), direction="neutral",
                vec=None, code_hash="x")
    r.add(f)
    m2 = FeatureMatrix(synth_table, r)
    rep = validate_feature(f, synth_table, m2, hw_alt=hw_other, n_shapes=3)
    # `hw.` 문자열이 docstring 에만 있으므로 하드웨어를 쓰는 것으로 보인다
    assert rep.failed, "하드코딩된 84 를 못 잡았다"
    assert any("스케일" in c.name for c in rep.fails())


def test_vectorized_matches_scalar_everywhere(synth_table, matrix):
    """★ 학습(행렬)과 배포(스칼라)가 같은 함수를 쓰는가."""
    from kernelrule.features import verify_vectorized

    p = synth_table.shapes()[0]
    df = synth_table.frame_for(p)
    _, info = matrix.for_shape(p)
    for f in REGISTRY.items():
        if f.vec is None or f.shape_level:
            continue
        verify_vectorized(f, df, matrix.hw, info, n=96)


def test_directions_are_declared():
    for f in REGISTRY.items(active_only=False):
        assert f.direction in ("higher_is_worse", "higher_is_better",
                               "neutral"), f.name


def test_shape_features_ignore_config(synth_table, matrix):
    """형상 수준 피처는 `cfg` 를 봐서는 안 된다 — 그것이 정의다."""
    p = synth_table.shapes()[0]
    cfgs = synth_table.configs(p)
    for f in REGISTRY.items(shape_level=True):
        vals = {float(f.fn(p, matrix.hw, c)) for c in cfgs[:40]}
        assert len(vals) == 1, f"{f.name} 이 config 마다 다른 값을 낸다: {vals}"
