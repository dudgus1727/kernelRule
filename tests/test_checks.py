"""정적 검사 (§8.3 + 부록 §8.1). **adversarial 케이스가 전부 걸려야 한다** (§24.3)."""
from __future__ import annotations

import pytest

from kernelrule.rules.checks import LIMITS, CheckReport, RuleCheckError, check_rule

FEAT = {"tail_waste", "smem_pressure", "waves", "has_spill", "edge_waste",
        "traffic_amplification", "sm_idle_cost", "split_k_cost",
        "pipeline_warmup_frac"}
SHAPE = {"is_memory_bound", "arith_intensity", "log_sol_ms", "M", "N", "K"}


def chk(code: str, n_weights: int = 1) -> CheckReport:
    return check_rule(code, feature_names=FEAT, shape_value_names=SHAPE,
                      n_weights=n_weights)


def test_well_formed_rule_passes():
    code = ("def score(f, p, hw, w):\n"
            "    s = f.tail_waste * w[0]\n"
            "    s = s + f.smem_pressure * w[1]\n"
            "    if p.is_memory_bound:\n"
            "        s = s + f.edge_waste * w[2]\n"
            "    return s + np.where(f.has_spill > 0, w[3], 0.0)\n")
    r = chk(code, 4)
    assert r.ok, r.violations
    assert r.features_used == {"tail_waste", "smem_pressure", "edge_waste",
                               "has_spill"}


def test_weight_indices_do_not_eat_the_literal_budget():
    """★ `w[0]` 의 `0` 은 리터럴이 아니다 — 가중치는 `n_weights` 로 이미 센다."""
    code = ("def score(f, p, hw, w):\n"
            "    return (f.tail_waste * w[0] + f.waves * w[1]"
            " + f.has_spill * w[2] + f.edge_waste * w[3])\n")
    r = chk(code, 4)
    assert r.n_literals == 0 and r.n_weights == 4 and r.ok


# ---------------------------------------------------------------------------
# §24.3 의 adversarial 케이스 — 하나라도 통과하면 방어에 구멍이 있다
# ---------------------------------------------------------------------------
ADVERSARIAL = [
    ("암기", ("def score(f, p, hw, w):\n"
             "    if p.M == 4096:\n"
             "        return f.waves * w[0]\n"
             "    return f.waves * w[0]\n"), 1, "직접 비교"),
    ("정답 누출", ("def score(f, p, hw, w):\n"
                  "    return f.waves * w[0] + time_ms\n"), 1, "금지된 이름"),
    ("난이도 참조", ("def score(f, p, hw, w):\n"
                    "    return f.waves * w[0] * difficulty\n"), 1, "금지된 이름"),
    ("표 접근", ("def score(f, p, hw, w):\n"
                "    return TABLE[0] * w[0]\n"), 1, "금지된 이름"),
    ("무한 루프 소스", ("def score(f, p, hw, w):\n"
                       "    while f.waves > 0:\n"
                       "        pass\n"
                       "    return f.waves * w[0]\n"), 1, "config 수준"),
    ("샌드박스 탈출", ("def score(f, p, hw, w):\n"
                      "    import os\n"
                      "    return f.waves * w[0]\n"), 1, "import"),
    ("오타", ("def score(f, p, hw, w):\n"
             "    return f.tail_wast * w[0]\n"), 1, "등록되지 않은 피처"),
    ("배열에 if", ("def score(f, p, hw, w):\n"
                  "    if f.waves < 1:\n"
                  "        return f.waves * w[0]\n"
                  "    return f.waves * w[0]\n"), 1, "config 수준"),
    ("배열에 삼항", ("def score(f, p, hw, w):\n"
                    "    return (w[0] if f.waves < 1 else w[0]) * f.waves\n"),
     1, "config 수준"),
    ("비결정론", ("def score(f, p, hw, w):\n"
                 "    return np.random.rand(3) * w[0]\n"), 1, "numpy"),
    ("던더 우회", ("def score(f, p, hw, w):\n"
                  "    return f.waves * w[0] + score.__globals__['x']\n"),
     1, "던더"),
    ("w 슬라이싱", ("def score(f, p, hw, w):\n"
                   "    return f.waves * w[0] + sum(w[1:])\n"), 2, "상수 인덱스"),
    ("w 통째로", ("def score(f, p, hw, w):\n"
                 "    return f.waves * len(w)\n"), 1, "통째로"),
    ("구문 오류", "def score(f, p, hw, w)\n    return 1\n", 1, "파싱 실패"),
    ("함수 여러 개", ("def helper():\n    return 1\n"
                     "def score(f, p, hw, w):\n"
                     "    return f.waves * w[0]\n"), 1, "하나만"),
    ("시그니처 변경", ("def score(problem, hw, candidates):\n"
                      "    return 0.0\n"), 1, "시그니처"),
    ("컴프리헨션", ("def score(f, p, hw, w):\n"
                   "    return sum([x for x in f.waves]) * w[0]\n"),
     1, "컴프리헨션"),
    ("미등록 형상값", ("def score(f, p, hw, w):\n"
                      "    if p.secret_difficulty:\n"
                      "        return f.waves * w[0]\n"
                      "    return f.waves * w[0]\n"), 1, "형상 수준"),
]


@pytest.mark.parametrize("name,code,nw,expect",
                         ADVERSARIAL, ids=[c[0] for c in ADVERSARIAL])
def test_adversarial_case_is_rejected(name, code, nw, expect):
    r = chk(code, nw)
    assert not r.ok, f"{name} 가 통과했다 — 방어에 구멍이 있다"
    assert any(expect in v for v in r.violations), \
        f"{name}: 기대한 이유({expect})가 아니라 {r.violations}"


def test_literal_budget_includes_weights():
    """★ `len(W0)` 를 리터럴 예산에 합산한다 (§29.4)."""
    code = ("def score(f, p, hw, w):\n"
            "    return f.waves * w[0] + f.tail_waste * w[1]\n")
    assert chk(code, 2).ok
    # 0 리터럴 + 9 가중치 = 9 > 8
    r = check_rule(code.replace("w[1]", "w[8]"), feature_names=FEAT,
                   shape_value_names=SHAPE, n_weights=9)
    assert not r.ok
    assert any("리터럴" in v for v in r.violations)


def test_unused_weights_are_rejected():
    code = "def score(f, p, hw, w):\n    return f.waves * w[0]\n"
    r = chk(code, 3)
    assert not r.ok and any("최대 인덱스" in v for v in r.violations)


def test_parse_failure_is_rejection_not_pass():
    """★ 파싱 실패는 **거부**다. 통과가 아니다 (§26.4)."""
    r = chk("this is not python", 1)
    assert not r.ok
    with pytest.raises(RuleCheckError):
        r.raise_if_bad()


def test_shape_level_if_is_allowed():
    """형상 수준 분기는 **허용**한다. 그것이 일반화되는 종류다."""
    code = ("def score(f, p, hw, w):\n"
            "    s = f.waves * w[0]\n"
            "    if p.is_memory_bound:\n"
            "        s = s + f.edge_waste * w[1]\n"
            "    return s\n")
    assert chk(code, 2).ok


def test_human_guided_rule_obeys_the_same_constraints():
    """★ 사람 기준선도 규칙과 **동일한 제약**을 받는다 (§9.4).

    같은 조건이어야 비교가 공정하다.
    """
    import kernelrule.features.physical  # noqa: F401  등록
    from kernelrule.features import REGISTRY
    from kernelrule.rules.human_guided import CODE, W0

    r = check_rule(CODE, feature_names=REGISTRY.names(shape_level=False),
                   shape_value_names=REGISTRY.names(shape_level=True),
                   n_weights=len(W0))
    assert r.ok, r.violations
    assert r.parameters_used <= LIMITS["parameters"]


# ---------------------------------------------------------------------------
# A-1 — 형상 수준 분기 no-op 경고 (거부가 아니다)
# ---------------------------------------------------------------------------
NOOP = [
    ("스칼라 곱", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                  "    if p.is_memory_bound:\n        s = s * w[1]\n"
                  "    return s\n"), 2),
    ("스칼라 덧셈", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                    "    if p.is_memory_bound:\n        s = s + w[1]\n"
                    "    return s\n"), 2),
    ("리터럴 곱", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                  "    if p.is_memory_bound:\n        s = s * 2.0\n"
                  "    return s\n"), 1),
    ("hw 상수", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                "    if p.is_memory_bound:\n        s = s / hw.sm_count\n"
                "    return s\n"), 1),
    ("AugAssign", """def score(f, p, hw, w):
    s = f.waves * w[0]
    if p.is_memory_bound:
        s *= w[1]
    return s
""", 2),
]


@pytest.mark.parametrize("name,code,nw", NOOP, ids=[c[0] for c in NOOP])
def test_noop_shape_branch_warns_but_passes(name, code, nw):
    """★ 형상 상수를 누적 점수에 곱/더하면 그 형상 안의 순위가 안 바뀐다.

    **거부가 아니라 경고**다 — 문법적으로 합법이고, 일반적인 경우는 못
    잡는다. 진짜 판정은 채점기와 §12 진단 리포트가 한다.
    """
    r = chk(code, nw)
    assert r.ok, f"{name} 을 거부하면 안 된다 (경고여야 한다)"
    assert r.warnings, f"{name} 이 no-op 인데 경고가 없다"
    assert "순위를 바꾸지 못한다" in r.warnings[0]


VALID_BRANCH = [
    ("항 재가중", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                  "    if p.is_memory_bound:\n"
                  "        s = s + f.edge_waste * w[1]\n    return s\n"), 2),
    ("AugAssign 항", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                     "    if p.is_memory_bound:\n"
                     "        s += f.edge_waste * w[1]\n    return s\n"), 2),
    ("분기 없음", "def score(f, p, hw, w):\n    return f.waves * w[0]\n", 1),
]


@pytest.mark.parametrize("name,code,nw", VALID_BRANCH,
                         ids=[c[0] for c in VALID_BRANCH])
def test_meaningful_shape_branch_is_not_warned(name, code, nw):
    """config 수준 항의 **가중치를 바꾸는** 분기는 경고하지 않는다."""
    r = chk(code, nw)
    assert r.ok and not r.warnings, r.warnings


def test_warnings_do_not_affect_ok():
    """경고는 `ok` 에 영향을 주지 않는다. 그래야 진화가 안 막힌다."""
    code = ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
            "    if p.is_memory_bound:\n        s = s * w[1]\n    return s\n")
    r = chk(code, 2)
    assert r.ok and len(r.warnings) == 1
    r.raise_if_bad()      # 예외가 나면 안 된다


# ---------------------------------------------------------------------------
# ★ 리터럴 예산 우회 — 실제 LLM 이 뚫었던 구멍 (§29.4)
# ---------------------------------------------------------------------------
def test_weight_reuse_is_rejected():
    """★ 가중치 8개로 항 19개를 만든 규칙이 실제로 통과했었다.

    `len(W0) == max_index + 1` 만 보면 못 잡는다. 예산의 목적은 "파라미터가
    많으면 어떤 구조든 비슷한 regret 에 도달해 구조 비교가 무의미해진다" 를
    막는 것인데, 항을 늘려 그 제한을 피하면 목적이 무너진다.
    """
    code = ("def score(f, p, hw, w):\n"
            "    s = f.waves * w[0]\n"
            "    s = s + f.tail_waste * w[0]\n"
            "    s = s + f.has_spill * w[1]\n"
            "    return s\n")
    r = chk(code, 2)
    assert not r.ok
    assert any("재사용" in v for v in r.violations)
    assert r.n_terms == 3 and r.n_weights == 2


def test_term_count_is_reported():
    code = ("def score(f, p, hw, w):\n"
            "    s = f.waves * w[0]\n"
            "    return s + f.tail_waste * w[1]\n")
    r = chk(code, 2)
    assert r.ok and r.n_terms == 2


def test_shape_branch_reweighting_is_still_allowed():
    """★ 같은 피처에 **다른** 가중치를 주는 재가중은 정당하다.

    "같은 식이 두 번 나오면 거부" 를 넣었다가 이 패턴을 오탐했다.
    이것이 §A-1 이 권장하는 바로 그 형태다.
    """
    code = ("def score(f, p, hw, w):\n"
            "    s = f.traffic_amplification * w[0]\n"
            "    if p.is_memory_bound:\n"
            "        s = s + f.traffic_amplification * w[1]\n"
            "    return s\n")
    r = chk(code, 2)
    assert r.ok, r.violations


def test_interaction_term_is_allowed():
    """피처 곱은 실재하는 물리(상호작용)다. 막지 않는다."""
    code = ("def score(f, p, hw, w):\n"
            "    s = f.waves * w[0]\n"
            "    return s + f.has_spill * f.smem_pressure * w[1]\n")
    assert chk(code, 2).ok


def test_the_actual_evasive_rule_is_now_rejected():
    """실제 실행에서 나온 규칙을 회귀로 고정한다."""
    code = ("def score(f, p, hw, w):\n"
            "    s = np.log2(f.traffic_amplification) * w[0]\n"
            "    s = s + f.sm_idle_cost * w[1]\n"
            "    s = s + f.smem_pressure * w[2]\n"
            "    s = s + f.has_spill * w[3]\n"
            "    s = s + f.edge_waste * w[0]\n"
            "    s = s + f.waves * w[1]\n"
            "    return s\n")
    r = chk(code, 4)
    assert not r.ok and any("재사용" in v for v in r.violations)


def test_human_guided_rule_uses_one_weight_per_term():
    """사람 기준선이 새 규칙을 만족하는지 회귀로 고정한다."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.features import REGISTRY
    from kernelrule.rules.human_guided import CODE, W0

    r = check_rule(CODE, feature_names=REGISTRY.names(shape_level=False),
                   shape_value_names=REGISTRY.names(shape_level=True),
                   n_weights=len(W0))
    assert r.ok, r.violations
    assert r.n_terms == len(W0), f"항 {r.n_terms} != 가중치 {len(W0)}"


# ---------------------------------------------------------------------------
# D-78 — 분기 비교 상수는 예산에서 뺀다
# ---------------------------------------------------------------------------

def test_branch_comparison_constant_is_free():
    """★ `p.roofline_ratio < 1` 의 `1` 은 예산에 안 든다 (D-78).

    합산 예산이 물리 상수까지 막아서, 진화가 `1` 을 안 쓰고 우회했다 —
    `np.square(x) < x`, `x < np.sqrt(x)`, `x < np.sign(x)`,
    `np.isfinite(x)`. 넷 다 `x < 1` 과 같고 사람이 읽기 어렵다.
    """
    code = ("def score(f, p, hw, w):\n"
            "    s = np.where(p.roofline_ratio < 1, f.waves, f.tail_waste) * w[0]\n"
            "    s = s + f.edge_waste * w[1]\n"
            "    return s\n")
    r = check_rule(code, feature_names=FEAT,
                   shape_value_names=SHAPE | {"roofline_ratio"}, n_weights=2)
    assert r.ok, r.violations
    assert r.n_literals == 0, "비교 상수가 예산에 들어갔다"
    assert r.branch_constants == [1], "면제한 상수를 기록하지 않았다"


def test_non_comparison_constant_still_costs():
    """중첩된 식 안의 상수는 여전히 예산이다 — 면제는 비교 피연산자만."""
    code = ("def score(f, p, hw, w):\n"
            "    s = np.where((f.waves - 3.0) < 1, f.waves, f.tail_waste) * w[0]\n"
            "    return s\n")
    r = check_rule(code, feature_names=FEAT,
                   shape_value_names=SHAPE | {"roofline_ratio"}, n_weights=1)
    assert r.n_literals == 1, f"3.0 을 세지 않았다: {r}"
    assert r.branch_constants == [1]


def test_shape_size_comparison_is_still_banned():
    """★ 면제는 **암기를 열어주지 않는다.** `p.M > 1024` 는 그대로 거부."""
    code = ("def score(f, p, hw, w):\n"
            "    s = np.where(p.M > 1024, f.waves, f.tail_waste) * w[0]\n"
            "    return s\n")
    r = check_rule(code, feature_names=FEAT, shape_value_names=SHAPE,
                   n_weights=1)
    assert not r.ok
    assert any("형상 크기를 직접 비교" in v for v in r.violations), r.violations


def test_both_budget_counters_agree():
    """★ LLM 경계와 정적 검사가 **같은 수를 세야 한다** (D-37 계열).

    갈리면 경계는 통과시키고 정적 검사가 조용히 버린다 — 그때 모델은
    무엇이 틀렸는지 끝내 듣지 못한다.
    """
    from kernelrule.rules.checks import PARAMETERS, literal_parameter_message

    cases = [
        ("def score(f, p, hw, w):\n    return f.waves * w[0]\n", 1),
        (("def score(f, p, hw, w):\n"
          "    return np.where(p.roofline_ratio < 1, f.waves, f.tail_waste)"
          " * w[0]\n"), 8),
        (("def score(f, p, hw, w):\n"
          "    return (f.waves - 2.0) * w[0]\n"), 8),
        ("def score(f, p, hw, w):\n    return f.waves * w[0] * 1.5\n", 8),
    ]
    for code, nw in cases:
        r = check_rule(code, feature_names=FEAT,
                       shape_value_names=SHAPE | {"roofline_ratio"},
                       n_weights=nw)
        over_static = r.parameters_used > PARAMETERS
        over_llm = literal_parameter_message(code, nw) is not None
        assert over_static == over_llm, (
            f"두 계수기가 갈렸다: 정적 {r.parameters_used}/{PARAMETERS} vs "
            f"경계 {over_llm}\n{code}")


# ---------------------------------------------------------------------------
# D-92 — 항등 변환으로 상수를 만드는 것을 막는다
# ---------------------------------------------------------------------------
IDENTITY = [
    ("isfinite",
     ("def score(f, p, hw, w):\n"
     "    return np.nan_to_num(np.isfinite(f.tail_waste)\n"
     "                         / (np.isfinite(f.tail_waste) - f.tail_waste)) * w[0]\n")),
    ("sign 을 비교에",
     ("def score(f, p, hw, w):\n"
     "    return np.where(p.roofline_ratio < np.sign(p.roofline_ratio),\n"
     "                    f.waves, f.tail_waste) * w[0]\n")),
    ("x < sqrt(x)",
     ("def score(f, p, hw, w):\n"
     "    return np.where(p.roofline_ratio < np.sqrt(p.roofline_ratio),\n"
     "                    f.waves, f.tail_waste) * w[0]\n")),
    ("square(x) < x",
     ("def score(f, p, hw, w):\n"
     "    return np.where(np.square(p.roofline_ratio) < p.roofline_ratio,\n"
     "                    f.waves, f.tail_waste) * w[0]\n")),
]


@pytest.mark.parametrize(("name", "code"), IDENTITY,
                         ids=[c[0] for c in IDENTITY])
def test_identity_transform_is_rejected(name, code):
    """★ 상수를 만드는 항등 변환은 **결함**이다 (D-92).

    넷 다 수학적으로 상수/단순 비교와 같은데 사람이 읽기 어렵다.
    "해석 가능한 규칙" 이 이 연구의 주장이므로 그것을 갉아먹는다.
    """
    from kernelrule.rules.checks import identity_transform_message

    msg = identity_transform_message(code)
    assert msg, f"{name} 를 못 잡는다"
    # ★ 대안을 함께 말해야 한다. 금지만 말하면 또 다른 우회를 만든다.
    assert "면제" in msg and "D-78" in msg
    r = check_rule(code, feature_names=FEAT,
                   shape_value_names=SHAPE | {"roofline_ratio"}, n_weights=1)
    assert not r.ok, f"{name}: 메시지는 나오는데 거부가 안 된다"


LEGIT = [
    ("리터럴 비교",
     ("def score(f, p, hw, w):\n"
     "    return np.where(p.roofline_ratio < 1, f.waves, f.tail_waste) * w[0]\n")),
    ("정당한 sqrt", "def score(f, p, hw, w):\n    return np.sqrt(f.waves) * w[0]\n"),
    ("정당한 square",
     ("def score(f, p, hw, w):\n    return np.square(f.reg_pressure) * w[0]\n")),
]


@pytest.mark.parametrize(("name", "code"), LEGIT, ids=[c[0] for c in LEGIT])
def test_legitimate_uses_are_not_rejected(name, code):
    """★ `sqrt` / `square` 자체는 정당하다 — **같은 인자와 비교할 때**만 결함이다.

    이것을 통째로 막으면 비선형 변환을 못 쓴다 (§30.10 의 "1/(1-x) 나
    log2(x) 는 같은 물리량의 다른 형태다").
    """
    from kernelrule.rules.checks import identity_transform_message

    assert identity_transform_message(code) is None, f"{name} 를 오탐한다"


def test_identity_check_catches_the_real_archive():
    """★ 되돌려서 잡는지 확인한다 (D-39 계열).

    합성 사례만으로는 "만든 검사기가 만든 사례를 잡는다" 밖에 안 된다.
    **실제로 진화가 만든 규칙**을 잡아야 한다.
    """
    import json
    from pathlib import Path

    from kernelrule.rules.checks import identity_transform_message

    root = Path(__file__).resolve().parents[1] / "runs"
    codes = []
    for f in sorted(root.glob("*/archive.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                codes.append(json.loads(ln)["code"])
    if not codes:
        pytest.skip("runs/ 가 없다 (gitignore) — 클론 직후에는 못 돈다")
    hit = sum(1 for c in codes if identity_transform_message(c))
    assert hit > 0, ("아카이브에서 하나도 못 잡는다 — 검사기가 실제 우회를 "
                     "못 본다는 뜻이다")
