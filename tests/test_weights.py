"""가중치 최적화 (§29). 구조와 파라미터의 분리."""
from __future__ import annotations

import numpy as np
import pytest

from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.scoring import evaluate
from kernelrule.core.splits import Split, SplitError, SplitSet, by_predicate
from kernelrule.core.weights import FitError, fit_weights, make_order_fn
from kernelrule.features import FeatureRegistry, feature
from toy import make_table

#: 생성 계수를 **알고 있는** 표를 만든다. 시간 = exp(w_true . f).
W_TRUE = np.array([2.0, 0.5, 1.5])


@pytest.fixture(scope="module")
def known():
    """알려진 계수로 만든 표 + 같은 구조의 규칙 (§26.2)."""
    rng = np.random.default_rng(0)
    n_cfg, n_shape = 24, 12

    # ★ 형상마다 피처값이 다르다. 같으면 모든 형상의 최적 config 가 같아져
    #   regret@1 이 24개 값만 갖는 계단이 되고, 그건 실제 표와 다르다.
    times, cols = {}, {"f0": [], "f1": [], "f2": []}
    for s in range(n_shape):
        F = rng.uniform(0.0, 1.0, size=(n_cfg, 3))
        base = 0.5 + 0.1 * s
        times[(128 * (s + 1), 4096, 4096)] = list(base * np.exp(F @ W_TRUE))
        for i in range(3):
            cols[f"f{i}"].extend(F[:, i].tolist())
    t = make_table(times, feature_cols=cols)

    r = FeatureRegistry("known")
    for i in range(3):
        def mk(i=i):
            @feature(registry=r, vec=lambda df, hw, p, i=i:
                     df[f"f{i}"].to_numpy(np.float64))
            def _f(p, hw, cfg, i=i) -> float:
                return 0.0
            _f.__name__ = f"f{i}"
            return _f
        # 이름을 바꾼 뒤 다시 등록해야 하므로 직접 만든다
        from kernelrule.features import Feature
        r.add(Feature(name=f"f{i}", fn=lambda p, hw, cfg: 0.0,
                      unit="dimensionless", expected_range=(0.0, 1.0),
                      direction="higher_is_worse",
                      vec=(lambda df, hw, p, i=i: df[f"f{i}"].to_numpy(np.float64)),
                      code_hash=f"h{i}"))
    m = FeatureMatrix(t, r)

    def score(f, p, hw, w):
        return f.f0 * w[0] + f.f1 * w[1] + f.f2 * w[2]

    return t, m, score


def _all_train(table) -> Split:
    return Split("train", tuple(table.shapes()), name="all")


def test_weight_fit_recovers_known_optimum(known):
    """★ 알려진 계수와 같은 구조를 주면 최적화기가 그 최적을 되찾아야 한다.

    regret 은 **순서**만 보므로 `w` 는 스케일까지만 식별된다. 방향(정규화한
    벡터)이 맞고 regret 이 1.0 에 도달하는지로 판정한다.
    """
    t, m, score = known
    w0 = np.array([1.0, 1.0, 1.0])          # 일부러 틀린 초기값
    fr = fit_weights(score, m, t, _all_train(t), w0, max_evals=400)
    assert fr.fit_regret == pytest.approx(1.0, abs=1e-9), fr
    cos = float(fr.w @ W_TRUE / (np.linalg.norm(fr.w) * np.linalg.norm(W_TRUE)))
    assert cos > 0.99, f"복원된 방향이 다르다: {fr.w} vs {W_TRUE} (cos={cos:.4f})"


def test_bad_initial_weights_would_have_lost_a_good_structure(known):
    """★ §29.3 의 근거. 가중치 최적화 없이 채점하면 좋은 구조가 버려진다."""
    t, m, score = known
    w_bad = np.array([1.0, 4.0, -3.0])
    before = evaluate(make_order_fn(score, m, w_bad), t, ks=(1,)).at(1)
    fr = fit_weights(score, m, t, _all_train(t), w_bad, max_evals=400)
    assert before > 1.05, "초기값이 충분히 나쁘지 않아 시험이 성립하지 않는다"
    assert fr.fit_regret < before
    assert fr.fit_regret == pytest.approx(1.0, abs=1e-9)


def test_fit_never_worse_than_initial(known):
    """계단 함수라 Nelder-Mead 가 초기값보다 나쁜 곳에 멈출 수 있다."""
    t, m, score = known
    w0 = W_TRUE.copy()
    fr = fit_weights(score, m, t, _all_train(t), w0, max_evals=30)
    base = evaluate(make_order_fn(score, m, w0), t, ks=(1,)).at(1)
    assert fr.fit_regret <= base + 1e-12


def test_weight_fit_uses_train_split_only(known):
    """★ 검증/최종 분할이 목적함수에 들어가는 경로가 없다 (§29.7)."""
    t, m, score = known
    shapes = t.shapes()
    for role in ("val", "test"):
        bad = Split(role, tuple(shapes))
        with pytest.raises(SplitError, match="학습 분할만"):
            fit_weights(score, m, t, bad, [1.0, 1.0, 1.0])


def test_weight_fit_refuses_a_bare_shape_list(known):
    """분할을 명시하지 않으면 에러다. 어느 분할인지 알 수 없다 (§26.4)."""
    t, m, score = known
    with pytest.raises(SplitError, match="Split 을 받는다"):
        fit_weights(score, m, t, t.shapes(), [1.0, 1.0, 1.0])


def test_val_split_must_be_val(known):
    t, m, score = known
    tr = _all_train(t)
    with pytest.raises(SplitError, match="'val' 이어야"):
        fit_weights(score, m, t, tr, [1.0, 1.0, 1.0], val_split=tr)


def test_gap_is_recorded(known):
    """학습/검증 격차를 라운드마다 기록한다 (§29.4)."""
    t, m, score = known
    shapes = t.shapes()
    tr = Split("train", tuple(shapes[:6]))
    va = Split("val", tuple(shapes[6:]))
    fr = fit_weights(score, m, t, tr, [1.0, 1.0, 1.0], val_split=va,
                     max_evals=300)
    assert np.isfinite(fr.val_regret)
    assert np.isfinite(fr.gap)


def test_sensitivity_flags_dead_terms(known):
    """0 근처로 수렴하거나 둔감한 항은 피처 정리 후보다 (§29.6)."""
    t, m, score = known

    def score4(f, p, hw, w):
        # w[3] 은 아무 데도 안 쓰인다 -> 완전히 둔감해야 한다
        return f.f0 * w[0] + f.f1 * w[1] + f.f2 * w[2] + 0.0 * w[3]

    fr = fit_weights(score4, m, t, _all_train(t), [1.0, 1.0, 1.0, 1.0],
                     max_evals=300)
    assert fr.sensitivity[3] == 0.0
    assert 3 in fr.dead_terms


def test_structure_that_never_scores_is_rejected(known):
    """모든 가중치에서 유효한 점수를 못 내면 **구조를 기각**한다 (§26.4)."""
    t, m, score = known

    def broken(f, p, hw, w):
        return np.full(len(f.f0), np.nan)

    with pytest.raises(FitError, match="구조를 기각"):
        fit_weights(broken, m, t, _all_train(t), [1.0], max_evals=20)


def test_make_order_fn_uses_the_same_score_fn(known):
    """★ 학습과 배포가 같은 `score()` 를 쓴다 (§8.1 대체본)."""
    t, m, score = known
    fr = fit_weights(score, m, t, _all_train(t), [1.0, 1.0, 1.0], max_evals=400)
    ev = evaluate(make_order_fn(score, m, fr.w), t, ks=(1,))
    assert ev.at(1) == pytest.approx(fr.fit_regret, abs=1e-12)


def test_split_refuses_empty_and_overlap():
    """분할이 빈 집합이거나 겹치면 에러다 (§26.4, §10)."""
    from kernelrule.core.types import Problem
    a = Problem(1024, 4096, 4096)
    b = Problem(2048, 4096, 4096)
    with pytest.raises(SplitError, match="빈 집합"):
        Split("train", ())
    with pytest.raises(SplitError, match="공유"):
        SplitSet(train=Split("train", (a, b)), val=Split("val", (b,)))
    with pytest.raises(SplitError, match="한쪽을 비웠다"):
        by_predicate([a, b], lambda p: False, name="none")
