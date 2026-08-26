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


# ---------------------------------------------------------------------------
# 커밋된 규칙과 기록된 점수가 어긋나지 않는가
# ---------------------------------------------------------------------------
# `runs/` 는 .gitignore 라 문서의 숫자를 대조할 방법이 없었다. 규칙과
# **적합된** 가중치를 커밋해 두면 채점이 결정론적이므로 검증할 수 있다.

def test_exported_rules_match_their_index():
    """★ `rules/*.py` 와 `index.json` 이 짝이 맞는가.

    전체 재채점은 `experiments/verify_rules.py` 가 한다 (번들이 필요하고
    수 분 걸린다). 여기서는 **파일과 기록이 어긋나지 않는지**만 본다 —
    내보내기를 깜빡하면 문서가 조용히 낡는다.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "docs/artifacts/rules"
    idx = root / "index.json"
    if not idx.exists():
        pytest.skip("내보낸 규칙이 없다 — experiments/export_rules.py")
    index = json.loads(idx.read_text())
    assert index, "index.json 이 비었다"
    for row in index:
        f = root / f"{row['run']}.py"
        assert f.exists(), f"{row['run']} 의 규칙 파일이 없다"
        src = f.read_text()
        assert "def score(" in src and "W_FITTED" in src
        w = row["weights"]
        assert set(w) == {"short", "long"}, f"{row['run']}: 체제가 빠졌다"
        assert all(len(v) > 0 for v in w.values())
        # ⚠️ 두 체제의 가중치가 **같을 수 있다.** Nelder-Mead 가 계단형
        #   목적함수에서 한 발짝도 못 움직이는 경우가 실재한다 (D-54) —
        #   12개 중 여러 개가 그렇다. 그것은 내보내기 버그가 아니므로
        #   여기서 실패시키지 않는다. 감시는 `index.json` 의 `w_moved` 다.
