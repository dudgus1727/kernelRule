"""정준 채점이 루프의 분할을 그대로 쓰는가 (§10.2 / D-36).

채점기와 루프가 분할을 각자 정하다가 홀드아웃 19형상 중 11개가 루프의
학습 형상이 됐다. 구조는 그것들을 보고 진화했으므로 홀드아웃이 아니었다.
**임의 분할 경로 자체를 없앴고, 그 사실을 여기서 고정한다.**
"""

from __future__ import annotations

import numpy as np
import pytest

from kernelrule.core.canonical import canonical_score
from kernelrule.core.splits import Split, SplitError, SplitSet
from tests.toy import make_table

_CODE = "def score(f, p, hw, w):\n    return f.idx * w[0]\n"


def _setup():
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import Feature, FeatureRegistry

    reg = FeatureRegistry("canon")
    reg.add(Feature(name="idx", fn=lambda p, hw, c: 0.0, unit="dimensionless",
                    expected_range=(0.0, 10.0), direction="neutral",
                    vec=lambda df, hw, p: np.arange(len(df), dtype=float),
                    code_hash="x"))
    t = make_table({(1024, 4096, 4096): [1.0, 2.0],
                    (512, 4096, 4096): [1.0, 1.5],
                    (2048, 4096, 4096): [2.0, 2.2],
                    (4096, 4096, 4096): [1.0, 3.0]})
    return t, FeatureMatrix(t, reg)


def test_canonical_requires_the_loop_splitset():
    """★ 형상을 따로 뽑는 경로를 두지 않는다."""
    t, m = _setup()
    shapes = list(t.shapes())
    with pytest.raises(SplitError, match="SplitSet"):
        canonical_score(_CODE, [1.0], table=t, matrix=m, splits=shapes)
    with pytest.raises(SplitError, match="SplitSet"):
        canonical_score(_CODE, [1.0], table=t, matrix=m,
                        splits=Split("train", tuple(shapes)))


def test_holdout_never_overlaps_the_training_shapes():
    """겹치면 SplitSet 이 만들어질 때 이미 막힌다 — 그것을 고정한다."""
    t, _ = _setup()
    shapes = list(t.shapes())
    with pytest.raises(SplitError):
        SplitSet(train=Split("train", tuple(shapes)),
                 val=Split("val", (shapes[0],)))


def test_canonical_scores_only_the_val_shapes():
    """★ 홀드아웃 점수는 `splits.val` 에서만 나온다 (D-36)."""
    t, m = _setup()
    shapes = list(t.shapes())
    splits = SplitSet(train=Split("train", tuple(shapes[:3])),
                      val=Split("val", tuple(shapes[3:])))
    r = canonical_score(_CODE, [1.0], table=t, matrix=m, splits=splits)
    assert r.n_holdout == len(shapes) - 3
    assert tuple(r.evaluation.shapes) == tuple(shapes[3:])
    # 표본내는 학습 형상에서 나온다 — 둘이 같은 집합이면 안 된다
    assert set(r.evaluation.shapes).isdisjoint(splits.train.shapes)


def test_thin_regime_warns_instead_of_pretending():
    """체제당 형상이 적으면 **조용히 넘어가지 않는다** (§10.1 / §26.4)."""
    t, m = _setup()
    shapes = list(t.shapes())
    splits = SplitSet(train=Split("train", tuple(shapes[:3])),
                      val=Split("val", tuple(shapes[3:])))
    r = canonical_score(_CODE, [1.0], table=t, matrix=m, splits=splits)
    assert r.warnings, "형상 3개짜리 학습인데 경고가 없다"
    assert any("학습 형상" in w for w in r.warnings)
