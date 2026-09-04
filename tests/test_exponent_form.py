"""★ 가중치를 **지수 자리**에 두는 형태 (D-112).

지금까지 넓힌 것은 항 수 / 노드 수 / 곱이고, **가중치가 선형 자리에만
있다**는 것은 안 건드렸다. 그것이 벽의 마지막 후보다 (D-111).

여기 시험이 지키는 것 셋:

```
수치 가드     밑이 음수가 될 수 있으면 nan 이고 **조용한 무력화**다
지수 경계     EXPONENT_BOUNDS 는 하이퍼파라미터가 아니라 정규화다
네 면        검사기 / 시스템 프롬프트 / 사용자 프롬프트 / 출력 스키마
```
"""
from __future__ import annotations

import os

import pytest

FN = ["reg_pressure", "split_k_cost"]
SV = ["is_memory_bound"]
MINS = {"reg_pressure": 0.0, "split_k_cost": 0.0}


def _check(code: str, n_weights: int, mins=MINS):
    from kernelrule.rules.checks import check_rule

    return check_rule(code, feature_names=FN, shape_value_names=SV,
                      n_weights=n_weights, feature_mins=mins)


_OK = ("def score(f, p, hw, w):\n"
       "    s = np.power(f.reg_pressure, w[0]) * w[1]\n"
       "    s = s + f.split_k_cost * w[2]\n"
       "    return s\n")


def test_exponent_form_is_allowed():
    assert _check(_OK, 3).ok


def test_base_must_be_a_bare_feature():
    """식을 밑으로 쓰면 음수가 될 수 있고 그러면 `nan` 이다."""
    code = ("def score(f, p, hw, w):\n"
            "    s = np.power(f.reg_pressure - f.split_k_cost, w[0]) * w[1]\n"
            "    return s\n")
    r = _check(code, 2)
    assert not r.ok and any("밑은" in v for v in r.violations), r.violations


def test_exponent_must_be_a_bare_weight():
    """식을 지수로 올리면 경계를 못 붙인다."""
    code = ("def score(f, p, hw, w):\n"
            "    s = np.power(f.reg_pressure, w[0] + 1.0) * w[1]\n"
            "    return s\n")
    r = _check(code, 2)
    assert not r.ok and any("지수 자리" in v for v in r.violations), r.violations


def test_negative_range_feature_is_refused_as_a_base():
    r = _check(_OK, 3, mins={"reg_pressure": -1.0, "split_k_cost": 0.0})
    assert not r.ok and any("음수" in v for v in r.violations), r.violations


def test_star_star_operator_is_treated_the_same():
    code = ("def score(f, p, hw, w):\n"
            "    s = (f.reg_pressure - f.split_k_cost) ** w[0] * w[1]\n"
            "    return s\n")
    assert not _check(code, 2).ok


# ---------------------------------------------------------------------------
# ★ 경계 — 하이퍼파라미터가 아니라 정규화다
# ---------------------------------------------------------------------------


def test_bounds_are_none_without_an_exponent():
    """★ 지수를 안 쓰는 규칙은 **옛 실행과 같은 조건**이어야 한다 (원칙 36).

    경계를 항상 붙이면 지금까지의 모든 실행이 조용히 다른 조건이 된다.
    """
    from kernelrule.rules.checks import weight_bounds

    plain = ("def score(f, p, hw, w):\n"
             "    s = f.reg_pressure * w[0]\n"
             "    return s\n")
    assert weight_bounds(plain, 1) is None


def test_bounds_apply_only_to_exponent_indices():
    from kernelrule.rules.checks import EXPONENT_BOUNDS, weight_bounds

    b = weight_bounds(_OK, 3)
    assert b[0] == EXPONENT_BOUNDS
    assert b[1] == (float("-inf"), float("inf"))
    assert b[2] == (float("-inf"), float("inf"))


def test_fit_respects_the_bounds(synth_table):
    """★ 실제로 적합해서 확인한다 — 숫자만 맞춰 두면 소용없다 (원칙 38)."""
    import numpy as np

    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.splits import Split
    from kernelrule.core.weights import fit_weights
    from kernelrule.features import Feature, FeatureRegistry
    from kernelrule.rules.checks import EXPONENT_BOUNDS, weight_bounds

    t = synth_table
    r = FeatureRegistry("known")
    for i in range(2):
        r.add(Feature(name=f"f{i}", fn=lambda p, hw, cfg: 0.0,
                      unit="dimensionless", expected_range=(0.0, 1.0),
                      direction="higher_is_worse",
                      # ★ 표 열에 안 기대는 vec — 이 시험은 **경계**를
                      #   보는 것이지 피처를 보는 것이 아니다.
                      vec=(lambda df, hw, p, i=i:
                           np.linspace(0.1, 1.0, len(df)) ** (i + 1)),
                      code_hash=f"h{i}"))
    m = FeatureMatrix(t, r)
    code = ("def score(f, p, hw, w):\n"
            "    s = np.power(f.f0, w[0]) * w[1]\n"
            "    return s\n")
    b = weight_bounds(code, 2)
    # ★ 경계 **밖에서** 출발시킨다 — 접히지 않으면 그대로 남는다
    fr = fit_weights(compile_rule(code), m, t,
                     Split("train", tuple(t.shapes())), [99.0, 1.0],
                     max_evals=60, objective="regret", polish=False, bounds=b)
    lo, hi = EXPONENT_BOUNDS
    assert lo <= float(fr.w[0]) <= hi, f"지수가 경계 밖이다: {fr.w[0]}"
    assert np.isfinite(fr.fit_regret)


# ---------------------------------------------------------------------------
# ★ 네 면 (D-105 / D-107 / D-110)
# ---------------------------------------------------------------------------


def test_guard_holds_at_the_llm_boundary_regardless_of_the_hint():
    """가드는 **조건이 아니라 수치 안전**이라 힌트와 무관하게 걸린다."""
    from kernelrule.agents.schemas import rule_output_for

    bad = ("def score(f, p, hw, w):\n"
           "    s = np.power(f.reg_pressure - f.split_k_cost, w[0]) * w[1]\n"
           "    return s\n")
    for hint in (False, True):
        with pytest.raises(Exception, match="밑은"):
            rule_output_for(8, power_hint=hint)(
                code=bad, w0=[1.0, 1.0], changes="", hypothesis_id="")


def test_power_hint_lands_on_every_surface_and_is_off_by_default():
    from kernelrule.agents.openai_client import assemble_instructions
    from kernelrule.agents.schemas import rule_output_for

    os.environ.setdefault("OPENAI_API_KEY", "t")
    kw = {"objective": "rank", "parameters": 8}
    off = assemble_instructions("rule_editor", **kw)
    on = assemble_instructions("rule_editor", power_hint=True, **kw)
    assert "{power_block}" not in off and "{power_note}" not in off
    assert "지수 자리에 둘 수 있습니다" not in off
    assert "지수 자리에 둘 수 있습니다" in on   # 시스템
    assert "np.power(f.<이름>, w[i])` 로 **바꿔도 됩니다**" in on
    for ph in (False, True):
        d = rule_output_for(8, power_hint=ph).model_json_schema()
        has = "지수 자리" in d["properties"]["code"]["description"]
        assert has is ph
