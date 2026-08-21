"""채점 — 손계산 고정값 (§26.2)."""
from __future__ import annotations

import numpy as np
import pytest
from toy import make_table, order_by_index

from kernelrule.core.scoring import Strata, evaluate, geomean, is_significant


@pytest.fixture
def t3():
    """형상 3개 x config 5개. 시간을 손으로 지정했다."""
    return make_table({
        (1024, 4096, 4096): [1.0, 1.2, 2.0, 4.0, 8.0],
        (2048, 4096, 4096): [2.0, 2.5, 3.0, 3.5, 10.0],
        (4096, 4096, 4096): [4.0, 4.4, 4.8, 20.0, 40.0],
    })


def test_best_and_difficulty_hand_computed(t3):
    shapes = t3.shapes()
    assert [t3.best_time(p) for p in shapes] == [1.0, 2.0, 4.0]
    # 난이도 = 중앙값/최적. 중앙값은 [2.0, 3.0, 4.8]
    assert [t3.difficulty(p) for p in shapes] == [2.0, 1.5, 1.2]


def test_regret_hand_computed(t3):
    """규칙이 각 형상에서 인덱스 [2,1,0,3,4] 순서를 낸다.

        형상1 top1 = idx2 -> 2.0 / 1.0 = 2.0     top3 = min(2.0,1.2,1.0) = 1.0
        형상2 top1 = idx2 -> 3.0 / 2.0 = 1.5     top3 = min(3.0,2.5,2.0)/2.0 = 1.0
        형상3 top1 = idx2 -> 4.8 / 4.0 = 1.2     top3 = min(4.8,4.4,4.0)/4.0 = 1.0
    """
    ev = evaluate(order_by_index([2, 1, 0, 3, 4]), t3, ks=(1, 3))
    assert list(ev.regret[:, 0]) == [2.0, 1.5, 1.2]
    assert list(ev.regret[:, 1]) == [1.0, 1.0, 1.0]
    assert ev.at(1) == pytest.approx((2.0 * 1.5 * 1.2) ** (1 / 3))
    assert ev.at(3) == pytest.approx(1.0)


def test_perfect_rule_gives_exactly_one(t3):
    """정답을 그대로 내는 규칙의 regret 이 **정확히** 1.0 이다."""
    ev = evaluate(order_by_index([0, 1, 2, 3, 4]), t3, ks=(1, 3, 5))
    assert ev.at(1) == 1.0
    assert np.all(ev.regret == 1.0) is np.True_ or ev.regret[:, 0].tolist() == [1.0] * 3
    assert ev.hit_rate(1) == 1.0


def test_worst_rule_matches_max_over_best(t3):
    """역순 규칙의 regret 이 (최악/최적) 과 일치한다."""
    ev = evaluate(order_by_index([4, 3, 2, 1, 0]), t3, ks=(1,))
    assert list(ev.regret[:, 0]) == [8.0, 5.0, 10.0]
    assert ev.at(1) == pytest.approx(geomean([8.0, 5.0, 10.0]))


def test_geomean_not_arithmetic():
    """regret 은 비율 척도다. 산술평균을 쓰면 큰 값이 지배한다."""
    assert geomean([1.0, 4.0]) == pytest.approx(2.0)
    assert geomean([1.0, 4.0]) != pytest.approx(2.5)


def test_order_must_be_a_permutation(t3):
    """★ 후보를 빠뜨리면 top-k 가 유리해진다. 조용히 넘어가지 않는다 (§26.4)."""
    with pytest.raises(ValueError, match="순서 길이"):
        evaluate(order_by_index([0, 1, 2]), t3, ks=(1,))
    with pytest.raises(ValueError, match="순열이 아니다"):
        evaluate(order_by_index([0, 0, 1, 2, 3]), t3, ks=(1,))


def test_empty_shape_set_is_an_error(t3):
    with pytest.raises(ValueError, match="형상이 하나도 없다"):
        evaluate(order_by_index([0, 1, 2, 3, 4]), t3, shapes=[], ks=(1,))


def test_strata_splits_by_difficulty_and_size(t3):
    s = Strata.build(t3)
    # 난이도 [2.0, 1.5, 1.2] -> 중앙 1.5 초과가 어려운 절반
    assert list(s.hard) == [True, False, False]
    # 최적시간 [1.0, 2.0, 4.0] -> 전부 0.5ms 이상
    assert not s.small.any()


def test_stratified_report_always_has_all_axes(t3):
    ev = evaluate(order_by_index([1, 0, 2, 3, 4]), t3, ks=(1,))
    st = ev.stratified(1)
    assert {"all", "large(>=0.5ms)", "small(<0.5ms)", "hard", "easy"} <= set(st)
    # ★ 크기 층화가 먼저 나온다 (§30.5)
    keys = list(st)
    assert keys.index("large(>=0.5ms)") < keys.index("hard")


def test_significance_uses_shape_noise_not_fixed_threshold():
    """★ 고정 임계값을 쓰지 않는다. 형상마다 다르다 (§7.4)."""
    from kernelrule.core.noise import NoiseModel
    m = NoiseModel.a6000_reference()
    big = make_table({(4096, 4096, 4096): [4.0, 4.1]}, noise=m)
    small = make_table({(512, 512, 512): [0.012, 0.013]}, noise=m)
    ev_b = evaluate(order_by_index([0, 1]), big, ks=(1,))
    ev_s = evaluate(order_by_index([0, 1]), small, ks=(1,))
    # 같은 0.5% 차이가 큰 형상에서는 유의하고 작은 형상에서는 아니다
    assert is_significant(0.005, ev_b)
    assert not is_significant(0.005, ev_s)


def test_significance_fails_closed():
    """계산할 수 없으면 **유의하지 않다**고 본다 (§26.4)."""
    t = make_table({(1024, 4096, 4096): [1.0, 2.0]})
    ev = evaluate(order_by_index([0, 1]), t, ks=(1,))
    ev.tol[:] = np.nan
    assert is_significant(999.0, ev) is False


def test_answer_mask_uses_noise_floor_not_one_percent():
    """정답 집합이 형상별 노이즈 바닥으로 만들어진다 (§30.3)."""
    from kernelrule.core.noise import NoiseModel
    m = NoiseModel.a6000_reference()
    # 11.3µs 형상: 바닥이 9% 이므로 2σ = 18% 안이 전부 정답
    t = make_table({(512, 512, 512): [0.0113, 0.0120, 0.0130, 0.0200]}, noise=m)
    p = t.shapes()[0]
    assert t.answer_mask(p).tolist() == [True, True, True, False]
    # 4ms 형상: 바닥이 0.053% 이므로 1% 차이는 명백한 오답
    t2 = make_table({(4096, 4096, 4096): [4.0, 4.04]}, noise=m)
    assert t2.answer_mask(t2.shapes()[0]).tolist() == [True, False]
