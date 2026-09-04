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


# ---------------------------------------------------------------------------
# ★ 두 방법 비교 — geomean 차이만으로 이겼다/졌다 하지 않는다 (§7.4)
# ---------------------------------------------------------------------------
def test_comparison_uses_shape_noise_not_geomean():
    """★ `1.085 > 1.080 이니 졌다` 는 부정확하다.

    61형상 geomean 에서 0.5% 면 형상 몇 개가 눈금 하나 차이로 달라져도 나온다.
    **"유의하게 진 형상이 N개" 가 정확한 서술이다.**
    """
    from kernelrule.core.noise import NoiseModel
    from kernelrule.core.scoring import compare

    m = NoiseModel.a6000_reference()
    t = make_table({
        # 4ms 형상: 바닥 0.053%. 1% 차이는 명백하다
        (4096, 4096, 4096): [4.0, 4.04],
        # 11us 형상: 바닥 9.1%. 6% 차이도 구분 불가다
        (512, 512, 512): [0.0113, 0.0120],
    }, noise=m)
    a = evaluate(order_by_index([1, 0]), t, ks=(1,), label="A")
    b = evaluate(order_by_index([0, 1]), t, ks=(1,), label="B")
    c = compare(a, b, t, name_a="A", name_b="B")
    assert int(c.a_loses.sum()) == 1        # 큰 형상만
    assert int(c.tied.sum()) == 1           # 작은 형상은 구분 불가
    assert "유의하게" in c.report()


def test_comparison_reports_magnitude_not_just_significance():
    """★ 시그마는 "실재하는가" 이지 "얼마나 큰가" 가 아니다.

    긴 형상은 노이즈 바닥이 0.05% 라 작은 차이도 수백 시그마다.
    크기(regret 차이)를 함께 보여야 오독하지 않는다.
    """
    from kernelrule.core.noise import NoiseModel
    from kernelrule.core.scoring import compare

    # 4ms 형상, 바닥 0.053%. 1% 차이 -> 약 19시그마인데 regret 차이는 0.01 뿐.
    # (0.1% 로 잡았다가 1.87시그마가 나왔다 — 유의하지 않은 것이 맞다.
    #  기대가 틀렸지 코드가 틀린 것이 아니었다.)
    t = make_table({(4096, 4096, 4096): [4.0, 4.04]},
                   noise=NoiseModel.a6000_reference())
    a = evaluate(order_by_index([1, 0]), t, ks=(1,))
    b = evaluate(order_by_index([0, 1]), t, ks=(1,))
    c = compare(a, b, t)
    assert c.sigma[0] > 10, "1% 차이가 큰 형상에서는 유의해야 한다"
    assert abs(c.delta[0]) < 0.02, "그런데 크기는 작다"
    rep = c.report()
    assert "regret" in rep and "시그마" in rep


def test_comparison_refuses_mismatched_shape_sets():
    """같은 형상 집합에서 잰 것이어야 비교가 성립한다 (§30.8)."""
    from kernelrule.core.scoring import compare

    t = make_table({(1024, 4096, 4096): [1.0, 2.0],
                    (2048, 4096, 4096): [1.0, 2.0]})
    sh = t.shapes()
    a = evaluate(order_by_index([0, 1]), t, sh, ks=(1,))
    b = evaluate(order_by_index([0, 1]), t, sh[:1], ks=(1,))
    with pytest.raises(ValueError, match="형상 집합이 다르다"):
        compare(a, b, t)
