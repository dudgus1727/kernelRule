"""★ 정답 누출 방어 (§3, §22.5, §30.7).

이 모듈이 스킵되면 `conftest.py` 의 감시가 세션을 실패시킨다 (§26.3).
"""
from __future__ import annotations

import dataclasses
import inspect
import warnings

import numpy as np
import pytest
from toy import constant_score_order

from kernelrule.core.scoring import evaluate
from kernelrule.core.types import CandidateSet, Config, Hardware, Problem

ANSWERISH = ("time", "ms", "difficulty", "cublas", "tflops", "regret",
             "outlier", "distinct", "peak", "elapsed")

#: 이름은 정답처럼 보이지만 **측정 조건 상수**인 것들. 측정 전에 정해지고
#: 표 전체에서 값이 하나다. 아래 테스트가 "정말로 상수인가" 를 검사하므로
#: 이 목록은 단순 화이트리스트가 아니다 — 어긋나면 걸린다.
SAFE_CONDITION_COLS = {"peak_tflops_used", "locked_mhz", "ridge_point",
                       "ridge_point_spec", "build_seconds"}


# ---------------------------------------------------------------------------
# 1. 자료구조 수준 — 규칙이 손댈 수 있는 객체에 시간이 없다 (§3.3)
# ---------------------------------------------------------------------------
def test_rule_facing_types_have_no_time_field():
    """`Problem` / `Config` / `CandidateSet` 어디에도 측정 시간이 없다."""
    for cls in (Problem, Config, CandidateSet):
        names = [f.name for f in dataclasses.fields(cls)]
        bad = [n for n in names
               if any(k in n.lower() for k in ("time", "cublas", "difficulty",
                                               "regret", "tflops"))]
        assert not bad, f"{cls.__name__} 에 정답스러운 필드: {bad}"


def test_hardware_has_no_measurement():
    names = [f.name for f in dataclasses.fields(Hardware)]
    assert not any("time" in n or "difficulty" in n for n in names)


def test_candidate_set_cannot_reach_times(synth_table):
    """★ `CandidateSet` 으로는 시간에 도달할 수 없다.

    `sorted(..., key=lambda c: (score, time))` 을 쓰려면 없는 속성이 필요하다.
    """
    p = synth_table.shapes()[0]
    cand = synth_table.candidates(p)
    for attr in ("time_ms", "time", "times", "best_time", "difficulty"):
        assert not hasattr(cand, attr), f"CandidateSet.{attr} 가 존재한다"


def test_order_fn_signature_excludes_the_table():
    """채점기가 규칙에 넘기는 인자가 `(Problem, CandidateSet)` 뿐이다."""
    src = inspect.getsource(evaluate)
    assert "order_fn(p, cand)" in src, \
        "order_fn 호출 인자가 바뀌었다 — 표나 시간이 넘어가지 않는지 확인하라"


def test_times_of_is_read_only(synth_table):
    """채점기가 받는 시간 배열도 쓰기 금지다. 실수로 손대는 것을 막는다."""
    p = synth_table.shapes()[0]
    t = synth_table.times_of(p)
    with pytest.raises(ValueError):
        t[0] = 0.0


def test_perftable_has_no_best_config():
    """★ "형상별 최적 config" 를 제공하지 않는다.

    이 표에서 66형상 중 29개가 최적시간에 **정확한 동점**이고 최대 84중
    동점이다. 그러면 "최적 config" 는 tie-break 규칙의 함수이지 물리적
    사실이 아니다. 정의 가능한 것은 `best_time` 과 `answer_mask` 뿐이다.
    """
    from kernelrule.core.table import PerfTable
    assert not hasattr(PerfTable, "best_config")
    assert not hasattr(PerfTable, "argbest")


# ---------------------------------------------------------------------------
# 2. tie-break 가 정답을 보지 않는다 (§30.7)
# ---------------------------------------------------------------------------
def test_constant_score_gives_random_performance(synth_table):
    """★ 모든 config 의 점수를 상수로 만들면 regret 이 무작위 선택과 동등해야 한다.

    좋게 나오면 tie-break 가 정답을 본다. kernelTab 베이스라인 실험에서
    실제로 발생한 버그다 (§30.7).
    """
    ev = evaluate(constant_score_order, synth_table, ks=(1,), label="constant")
    got = ev.at(1)

    rng = np.random.default_rng(0)
    draws = []
    for _ in range(24):
        def rnd(p, cand, rng=rng):
            return rng.permutation(cand.n)
        draws.append(evaluate(rnd, synth_table, ks=(1,)).at(1))
    lo, hi = float(np.min(draws)), float(np.max(draws))
    assert lo * 0.7 <= got <= hi * 1.3, (
        f"상수 점수 regret {got:.3f} 이 무작위 범위 [{lo:.3f}, {hi:.3f}] 밖이다 — "
        "tie-break 가 정답을 보고 있다")


def test_tiebreak_is_independent_of_row_order(synth_table):
    """tie-break 가 **표의 행 순서**에 의존하지 않는다.

    `groupby.idxmin()` 은 행 순서에 의존한다. 실제로 그 때문에 "형상별 최적
    config" 의 축 분포가 절차마다 달라지는 것을 확인했다.
    """
    from kernelrule.core.types import make_tiebreak

    p = synth_table.shapes()[0]
    c = synth_table.candidates(p)
    perm = np.random.default_rng(3).permutation(c.n)
    tb2 = make_tiebreak(c.kernel_id[perm], c.split_k[perm],
                        c.split_k_mode[perm])
    # 같은 config 는 섞여도 같은 상대 순위를 갖는다
    assert np.array_equal(np.argsort(c.tiebreak[perm]), np.argsort(tb2))


def test_order_by_rejects_nonfinite_scores(synth_table):
    """nan 을 뒤로 미루고 조용히 진행하지 않는다. 규칙이 망가진 것이다."""
    p = synth_table.shapes()[0]
    c = synth_table.candidates(p)
    s = np.zeros(c.n)
    s[3] = np.nan
    with pytest.raises(ValueError, match="비유한"):
        c.order_by(s)


# ---------------------------------------------------------------------------
# 3. ★ null 프리셋 — 유일한 자동 누출 탐지기 (§22.5)
# ---------------------------------------------------------------------------
def test_null_preset_gives_no_improvement(null_table):
    """★ config 가 성능과 무관한 표에서 **어떤 규칙도 1.0 을 크게 밑돌 수 없다.**

    밑돌면 어딘가에서 정답이 새고 있다. 이것이 가장 중요한 테스트다.
    """
    orders = {
        "constant": constant_score_order,
        "by_tiebreak": lambda p, c: c.order_by(np.zeros(c.n)),
        "by_splitk": lambda p, c: c.order_by(c.split_k.astype(float)),
        "random": lambda p, c: np.random.default_rng(5).permutation(c.n),
    }
    for name, fn in orders.items():
        ev = evaluate(fn, null_table, ks=(1,), label=name)
        assert ev.at(1) >= 0.999, (
            f"{name}: null 표에서 regret {ev.at(1):.4f} < 1.0 — 정답이 새고 있다")


def test_null_preset_difficulty_is_near_one(null_table):
    """null 표에서는 난이도가 1 근처여야 한다 (§22.5)."""
    d = np.array([s.difficulty for s in null_table.all_stats()])
    assert d.max() < 1.15, f"null 표의 난이도가 {d.max():.3f} — 구조가 남아 있다"


def test_null_preset_best_equals_typical(null_table):
    """null 표에서 최적과 중앙값의 차이는 노이즈뿐이다."""
    for s in null_table.all_stats():
        assert s.difficulty - 1.0 < 4.0 * s.noise_floor + 0.1


# ---------------------------------------------------------------------------
# 4. 로더 계약 (§3.2)
# ---------------------------------------------------------------------------
def test_ranking_loader_has_no_answers(real_bundle_path):
    from kerneltab.core.bundle import load_bundle
    from kerneltab.core.table import ANSWER_COLS, assert_no_answers

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = load_bundle(real_bundle_path).ranking(ok_only=False,
                                                  unknown_columns="ignore")
    assert_no_answers(X)
    assert not (set(X.columns) & set(ANSWER_COLS))


def test_perftable_feature_frame_has_no_answers(synth_table):
    """`PerfTable.frame_for` 는 피처만 준다 — 피처 행렬의 입력이다."""
    from kerneltab.core.table import ANSWER_COLS, OUTCOME_COLS

    p = synth_table.shapes()[0]
    cols = set(synth_table.frame_for(p).columns)
    assert not (cols & set(ANSWER_COLS))
    assert not (cols & set(OUTCOME_COLS))
    suspect = sorted(c for c in cols
                     if any(k in c.lower() for k in ANSWERISH))
    unexplained = [c for c in suspect if c not in SAFE_CONDITION_COLS]
    assert not unexplained, (
        f"규칙 입력에 정답스러운 컬럼이 있다: {unexplained}. "
        "측정 조건 상수라면 SAFE_CONDITION_COLS 에 넣고 이유를 적어라.")
    # 조건 상수라고 주장한 것이 **정말로 상수인지** 확인한다.
    df = synth_table.frame_for(p)
    for c in suspect:
        assert df[c].nunique(dropna=False) == 1, (
            f"{c!r} 는 측정 조건 상수로 분류돼 있는데 형상 안에서 "
            f"{df[c].nunique()}개 값을 갖는다 — 정답에서 유도된 값일 수 있다.")


def test_env_hash_is_required(real_bundle_path):
    """`env_hash` 는 조인 키가 아니라 격리 경계다. 기본값이 없다 (§3.4)."""
    from kernelrule.core.table import PerfTable

    sig = inspect.signature(PerfTable.from_bundle)
    assert sig.parameters["env_hash"].default is inspect.Parameter.empty
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception, match="env_hash"):
            PerfTable.from_bundle(real_bundle_path, env_hash="deadbeef")


# ---------------------------------------------------------------------------
# 피처 함수가 받는 프레임에 정답이 없는가 (§3 5번째 겹)
# ---------------------------------------------------------------------------
# `FeatureMatrix` 는 `table.frame_for(p)` 를 피처 함수에 그대로 넘긴다.
# 그 프레임이 정답을 담고 있으면 **피처가 측정 시간을 볼 수 있다.**
#
# 지금 24개는 사람이 썼으니 안 쓰지만, **FeatureWriter 가 만든 피처는
# 다르다** — 그쪽은 우리가 안 본 코드다. 구조로 막혀 있어야 한다.

@pytest.mark.needs_bundle
def test_feature_frame_has_no_answer_column(real_bundle_path):
    """★ `frame_for` 가 `ANSWER_COLS` 를 한 칸도 담지 않는다."""
    from kerneltab.core.table import ANSWER_COLS

    from kernelrule.core.table import PerfTable
    t = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                              ok_only=False)
    cols = set(t.frame_for(t.shapes()[0]).columns)
    leaked = cols & set(ANSWER_COLS)
    assert not leaked, (
        f"피처 함수가 받는 프레임에 정답 컬럼이 있다: {sorted(leaked)}. "
        "`PerfTable.from_bundle` 이 `bundle.ranking()` 을 쓰는지 확인하라 (§3)")


def test_generated_feature_touching_answers_is_rejected():
    """★ 정답 컬럼을 참조하는 생성 피처를 검사기가 잡는가 (§11.4).

    프레임에 없으므로 실행하면 어차피 터지지만, **AST 단계에서 사유가
    분명하게** 잡혀야 한다 — 실행 예외로 터지면 "모델이 나쁜 피처를 냈다"
    로 읽힌다 (D-49).
    """
    from kernelrule.features.generated import FeatureRejected, check_feature_code

    for col in ("time_ms", "cublas_ms", "difficulty", "tflops"):
        code = (f"def peek(p, hw, cfg) -> float:\n"
                f"    return float(cfg.{col})\n")
        with pytest.raises(FeatureRejected):
            check_feature_code(code, known=frozenset())
