"""가중치 최적화 (§29). 구조와 파라미터의 분리."""
from __future__ import annotations

import numpy as np
import pytest
from toy import make_table

from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.scoring import evaluate
from kernelrule.core.splits import Split, SplitError, SplitSet, by_predicate
from kernelrule.core.weights import FitError, fit_weights, make_order_fn
from kernelrule.features import FeatureRegistry, feature

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


# ---------------------------------------------------------------- D-54/D-55
# 적합기가 "아무것도 안 했는가" 를 스스로 신고해야 한다. 24회 중 13회가
# 초기값 그대로였는데 아무도 몰랐다 — 그 침묵을 막는 검사다.

def test_fitted_rule_reports_that_it_did_not_move():
    from kernelrule.core.weights import FittedRule

    fr = FittedRule(w=np.array([1.0, 2.0]), w0=np.array([1.0, 2.0]),
                    fit_regret=1.1, n_evals=305, n_infeasible=0,
                    sensitivity=np.zeros(2), seconds=1.0)
    assert not fr.moved
    assert any("움직이지 않았다" in m for m in fr.invariants())


def test_fitted_rule_flags_dominance_and_negative_weights():
    """★ 절대 배율이 아니라 **실효 기여도**로 지배를 잡는다 (D-70)."""
    from kernelrule.core.weights import FittedRule

    fr = FittedRule(w=np.array([500.0, -3.0, 1.0]), w0=np.array([1.0, 2.0, 1.0]),
                    fit_regret=1.1, n_evals=10, n_infeasible=0,
                    sensitivity=np.zeros(3), seconds=1.0,
                    contrib=np.array([1000.0, 1.0, 1.0]))
    assert fr.moved
    msgs = " ".join(fr.invariants())
    assert "압도한다" in msgs and "음수" in msgs

    # 기여도가 고르면 |w| 가 아무리 커도 경고하지 않는다 — 배율은 무해하다
    ok = FittedRule(w=np.array([5e6, 4e6, 6e6]), w0=np.ones(3),
                    fit_regret=1.1, n_evals=10, n_infeasible=0,
                    sensitivity=np.zeros(3), seconds=1.0,
                    contrib=np.array([1.0, 0.9, 1.1]))
    assert not any("압도" in m for m in ok.invariants())


def test_dead_term_is_flagged():
    """실효 기여도 0 = 순위에 관여하지 않는 항 (절대 규칙 2)."""
    from kernelrule.core.weights import FittedRule

    fr = FittedRule(w=np.ones(3), w0=np.ones(3), fit_regret=1.1, n_evals=10,
                    n_infeasible=0, sensitivity=np.zeros(3), seconds=1.0,
                    contrib=np.array([1.0, 0.0, 1.2]))
    assert any("기여도가 0" in m for m in fr.invariants())


def test_fit_weights_warns_about_its_own_invariants(known):
    """적합이 이상하면 **조용히 넘어가지 않는다** (D-54).

    ★ 예전에는 "평가 상한에 닿았다" 로 이것을 확인했다. 그 경고가 **늘
    떠 있었기 때문**이고, 그래서 시험은 통과했지만 아무것도 안 지키고
    있었다 (D-76). 이제는 **죽은 항**으로 확인한다 — `w[2]` 가 점수에
    안 들어가므로 실효 기여도가 0 이고, 그것은 진짜 이상 신호다.
    """
    import warnings

    from kernelrule.core.weights import FitWarning

    t, m, _score = known

    def dead_term(f, p, hw, w):
        return f.f0 * w[0] + f.f1 * w[1] + f.f2 * 0.0 * w[2]

    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        fit_weights(dead_term, m, t, _all_train(t), [1.0, 1.0, 1.0],
                    max_evals=200)
    msgs = [str(w.message) for w in got if issubclass(w.category, FitWarning)]
    assert any("실효 기여도가 0" in x for x in msgs), \
        f"죽은 항을 신고하지 않았다: {msgs}"


def test_polish_never_worsens_training_regret(known):
    """좌표 다듬기는 훈련 regret 을 **개선하거나 같아야** 한다 (D-55).

    받아들이는 조건이 `v < base` 라 구조적으로 그렇다. 이 검사가 깨지면
    다듬기가 훈련 아닌 것을 보고 있다는 뜻이다 (§29.7).
    """
    t, m, score = known
    tr = _all_train(t)
    a = fit_weights(score, m, t, tr, [1.0, 1.0, 1.0], max_evals=120,
                    warn_invariants=False)
    b = fit_weights(score, m, t, tr, [1.0, 1.0, 1.0], max_evals=120,
                    warn_invariants=False, polish=True)
    assert b.fit_regret <= a.fit_regret + 1e-12


def test_polish_only_sees_the_training_split(known):
    """다듬기에 검증 분할을 흘리는 경로가 없다 — 인자 자체가 없다 (§29.7)."""
    import inspect

    from kernelrule.core.weights import _polish

    names = set(inspect.signature(_polish).parameters)
    assert "val_split" not in names and "splits" not in names


def test_contributions_are_scale_invariant(known):
    """★ 실효 기여도는 **가중치를 통째로 배로 키워도** 비율이 그대로다.

    절대 배율(|w|/|w0|)은 피처 스케일에 따라 자릿수가 달라져 라이브러리를
    바꾸면 기준이 무의미해진다 — F1(피처 [0,0.2])에서 |w| 최대가
    770,164 이고 사람 24개에서는 45.1 이었다 (D-70).
    """
    from kernelrule.core.weights import _contributions, _Problem

    t, m, score = known
    prob = _Problem(m, t, t.shapes(), 1)
    w = np.array([1.0, 2.0, 0.5])
    a = _contributions(prob, score, w)
    b = _contributions(prob, score, w * 1000.0)
    assert a is not None and b is not None
    # 절대값은 1000배, **비율**은 같다
    ra, rb = a / a.max(), b / b.max()
    assert np.allclose(ra, rb, atol=1e-9), (ra, rb)


def test_contribution_of_a_shape_constant_term_is_zero(known):
    """형상 상수 항은 순위를 안 바꾸므로 기여도가 **정확히 0** 이어야 한다.

    `_rules_common.md` 절대 규칙 2 가 말하는 no-op 항을 잡는 검사다.
    """
    from kernelrule.core.weights import _contributions, _Problem

    t, m, _ = known

    def score_with_noop(f, p, hw, w):
        # w[1] 항은 형상 상수라 그 형상 안에서 순위를 못 바꾼다
        return f.f0 * w[0] + p.n_candidates * w[1]

    prob = _Problem(m, t, t.shapes(), 1)
    c = _contributions(prob, score_with_noop, np.array([1.0, 5.0]))
    assert c is not None
    assert c[0] > 0.0
    assert c[1] == 0.0, f"형상 상수 항의 기여도가 0 이 아니다: {c[1]}"


def test_contributions_never_touch_the_answer():
    """★ 시간을 보는 경로가 없다 (§3)."""
    import ast
    import inspect

    from kernelrule.core.weights import _contributions

    # ★ 독스트링을 빼고 **본문만** 본다. 문서에 "prob.regret 을 안 부른다"
    #   라고 적어 두면 문자열 검사가 그걸 잡는다 (원칙 14 — 계측이 만드는 오탐).
    tree = ast.parse(inspect.getsource(_contributions).strip())
    fn = tree.body[0]
    body = fn.body[1:] if (isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)
                           ) else fn.body
    src = "\n".join(ast.unparse(n) for n in body)
    assert "prob.regret" not in src, "정답을 통과하는 regret 을 부른다"
    assert "_times" in src and "_best" in src, (
        "정답 자리를 `_` 로 안 받는다 — 이름이 없어야 손댈 수 없다")
    # `score_fn` 만 부른다
    assert src.count("score_fn(") == 2

def test_cap_warning_ignores_polish_evals(known):
    """★ 상한 경고는 **다듬기 전** 평가로 판정한다.

    `n_evals` 는 다듬기까지 합한 값이다. 그것으로 `max_evals` 와 견주면
    다듬기 예산(600)이 상한(300)을 언제나 넘어 **모든 적합에서** "평가
    상한에 닿았다 — 수렴 전 중단" 이 뜬다. 다듬기가 기본으로 켜진 뒤
    (D-56) 이 경고는 늘 켜져 있어 신호가 아니었다 (원칙 11).
    """
    import warnings as _w

    from kernelrule.core.weights import FitWarning

    t, m, score = known
    sp = _all_train(t)
    with _w.catch_warnings(record=True) as got:
        _w.simplefilter("always")
        fr = fit_weights(score, m, t, sp, [1.0, 1.0, 1.0], max_evals=3000,
                         polish=True, polish_budget=400)
    assert fr.n_fit_evals < fr.n_evals, "다듬기가 안 돌았다 — 시험이 무의미하다"
    assert fr.n_fit_evals < 3000, "적합만으로 상한에 닿았다 — 예산을 올려라"
    caps = [str(x.message) for x in got
            if issubclass(x.category, FitWarning) and "평가 상한" in str(x.message)]
    assert not caps, f"다듬기 평가 때문에 상한 경고가 떴다: {caps}"


def test_cap_warning_needs_actual_improvement_at_cutoff(known):
    """★ 예산을 다 쓴 것만으로는 경고하지 않는다.

    재시작 일정이 `max_evals` 를 **설계상 전부 쓰게** 돼 있어 "상한에
    닿았다" 는 언제나 참이다. 늘 참인 것은 감시가 아니다 (원칙 11).
    경고는 **잘리는 순간까지 나아지고 있었을 때**만 뜬다.

    생성 계수 `W_TRUE` 로 출발하면 더 나아질 곳이 없다 — 예산은 다 쓰지만
    개선은 없으므로 경고가 없어야 한다.
    """
    import warnings as _w

    from kernelrule.core.weights import FitWarning

    t, m, score = known
    with _w.catch_warnings(record=True) as got:
        _w.simplefilter("always")
        fr = fit_weights(score, m, t, _all_train(t), W_TRUE.tolist(),
                         max_evals=120, polish=False)
    assert fr.n_fit_evals >= 120, "예산을 다 쓰지 않았다 — 시험이 무의미하다"
    caps = [str(x.message) for x in got
            if issubclass(x.category, FitWarning) and "상한" in str(x.message)]
    assert not caps, f"예산 소진만으로 경고가 떴다: {caps}"
