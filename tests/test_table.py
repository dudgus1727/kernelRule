"""`PerfTable` — 조인, 격리 경계, 형상 통계."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from kernelrule.core.table import PerfTable, TableError
from toy import make_table


def test_join_mismatch_is_an_error():
    """ranking/scoring 이 어긋나면 채점이 조용히 틀린다."""
    import pandas as pd
    t = make_table({(1024, 4096, 4096): [1.0, 2.0]})
    X = t.frame_for(t.shapes()[0]).copy()
    y = pd.DataFrame({"kernel_id": ["k001", "k000"], "M": [1024, 1024],
                      "N": [4096, 4096], "K": [4096, 4096],
                      "split_k": [1, 1], "split_k_mode": ["serial"] * 2,
                      "time_ms": [1.0, 2.0]})
    from kernelrule.core.noise import NoiseModel
    with pytest.raises(TableError, match="조인이 어긋났다"):
        PerfTable.from_frames(X, y, hw=None,
                              noise=NoiseModel.a6000_reference(),
                              env_hash="x", unexpected="ignore")


def test_shape_with_no_valid_measurement_is_an_error():
    with pytest.raises(TableError, match="유효한 측정이 하나도 없다"):
        make_table({(1024, 4096, 4096): [0.0, float("nan")]})


def test_unknown_shape_raises():
    from kernelrule.core.types import Problem
    t = make_table({(1024, 4096, 4096): [1.0, 2.0]})
    with pytest.raises(KeyError, match="표에 없는 형상"):
        t.candidates(Problem(999, 1, 1))


def test_summary_has_both_axes():
    """난이도(물리)와 distinct_time_frac(계측)은 **다른 축**이다 (§30.4b)."""
    t = make_table({(1024, 4096, 4096): [1.0, 1.0, 2.0, 4.0]})
    s = t.summary().iloc[0]
    assert s.difficulty == pytest.approx(1.5)
    assert s.n_tied_at_best == 2
    assert s.distinct_time_frac == pytest.approx(3 / 4)


def test_size_stratum_boundary():
    small = make_table({(512, 512, 512): [0.2, 0.4]})
    large = make_table({(4096, 4096, 4096): [1.2, 2.0]})
    assert small.all_stats()[0].is_small
    assert not large.all_stats()[0].is_small


@pytest.mark.needs_bundle
def test_real_bundle_reproduces_known_counts(real_bundle_path):
    """§2 의 '확인된 사실' 중 tie-break 와 무관한 것들을 고정한다."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                  ok_only=False)
    assert len(t.shapes()) == 66
    s = t.summary()
    assert len(s) == 66
    assert s.best_ms.min() == pytest.approx(0.011264, abs=1e-6)
    assert s.best_ms.max() == pytest.approx(9.730048, abs=1e-4)
    assert int(s.is_small.sum()) == 45            # 45/66 이 0.5ms 미만
    assert 1.60 < s.difficulty.median() < 1.75


@pytest.mark.needs_bundle
def test_real_bundle_has_massive_ties_at_the_optimum(real_bundle_path):
    """★ 66형상 중 절반 가까이가 최적시간에 **정확한 동점**이다.

    그래서 "형상별 최적 config" 는 tie-break 규칙의 함수이지 물리적 사실이
    아니다. `best_config` 를 제공하지 않는 이유다.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                  ok_only=False)
    n_tied = np.array([s.n_tied_at_best for s in t.all_stats()])
    assert (n_tied > 1).sum() >= 20
    assert n_tied.max() >= 50
