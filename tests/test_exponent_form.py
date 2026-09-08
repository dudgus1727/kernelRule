"""★ The form that puts a weight **in the exponent slot** (D-112).

What has been widened so far is the term count / the node count / products,
and **that weights sit only in linear positions** was never touched. That is
the last candidate for the wall (D-111).

Three things these tests hold:

```
numerical guard   a base that can go negative is nan, and that is a **silent
                  neutering**
exponent bounds   EXPONENT_BOUNDS is normalisation, not a hyperparameter
four surfaces     the checker / the system prompt / the user prompt / the
                  output schema
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
    """An expression as the base can go negative, and then it is
    `nan`."""
    code = ("def score(f, p, hw, w):\n"
            "    s = np.power(f.reg_pressure - f.split_k_cost, w[0]) * w[1]\n"
            "    return s\n")
    r = _check(code, 2)
    assert not r.ok and any("base of `np.power`" in v
                            for v in r.violations), r.violations


def test_exponent_must_be_a_bare_weight():
    """Raising an expression to the exponent leaves no way to bound it."""
    code = ("def score(f, p, hw, w):\n"
            "    s = np.power(f.reg_pressure, w[0] + 1.0) * w[1]\n"
            "    return s\n")
    r = _check(code, 2)
    assert not r.ok and any("exponent slot" in v
                            for v in r.violations), r.violations


def test_negative_range_feature_is_refused_as_a_base():
    r = _check(_OK, 3, mins={"reg_pressure": -1.0, "split_k_cost": 0.0})
    assert not r.ok and any("can be negative" in v
                            for v in r.violations), r.violations


def test_star_star_operator_is_treated_the_same():
    code = ("def score(f, p, hw, w):\n"
            "    s = (f.reg_pressure - f.split_k_cost) ** w[0] * w[1]\n"
            "    return s\n")
    assert not _check(code, 2).ok


# ---------------------------------------------------------------------------
# ★ The bounds — normalisation, not a hyperparameter
# ---------------------------------------------------------------------------


def test_bounds_are_none_without_an_exponent():
    """★ A rule that uses no exponent must be **under the same conditions
    as the old runs** (principle 36).

    Always attaching bounds would silently make every run so far a different
    condition.
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
    """★ Verified by actually fitting — matching the numbers alone is
    useless (principle 38)."""
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
                      # ★ A vec that does not lean on table columns — this
                      #   test looks at the **bounds**, not at the feature.
                      vec=(lambda df, hw, p, i=i:
                           np.linspace(0.1, 1.0, len(df)) ** (i + 1)),
                      code_hash=f"h{i}"))
    m = FeatureMatrix(t, r)
    code = ("def score(f, p, hw, w):\n"
            "    s = np.power(f.f0, w[0]) * w[1]\n"
            "    return s\n")
    b = weight_bounds(code, 2)
    # ★ It starts **outside** the bounds — if it is not folded in, it stays
    fr = fit_weights(compile_rule(code), m, t,
                     Split("train", tuple(t.shapes())), [99.0, 1.0],
                     max_evals=60, objective="regret", polish=False, bounds=b)
    lo, hi = EXPONENT_BOUNDS
    assert lo <= float(fr.w[0]) <= hi, (
        f"the exponent is outside the bounds: {fr.w[0]}")
    assert np.isfinite(fr.fit_regret)


# ---------------------------------------------------------------------------
# ★ The four surfaces (D-105 / D-107 / D-110)
# ---------------------------------------------------------------------------


def test_guard_holds_at_the_llm_boundary_regardless_of_the_hint():
    """The guard is **numerical safety, not a condition**, so it applies
    regardless of the hint."""
    from kernelrule.agents.schemas import rule_output_for

    bad = ("def score(f, p, hw, w):\n"
           "    s = np.power(f.reg_pressure - f.split_k_cost, w[0]) * w[1]\n"
           "    return s\n")
    for hint in (False, True):
        with pytest.raises(Exception, match="base of `np.power`"):
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
    assert "a weight may sit in the exponent" not in off
    assert "a weight may sit in the exponent" in on   # system
    assert "np.power(f.<name>, w[i])`" in on
    for ph in (False, True):
        d = rule_output_for(8, power_hint=ph).model_json_schema()
        has = "in the exponent" in d["properties"]["code"]["description"]
        assert has is ph
