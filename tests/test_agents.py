"""MockLLM and the schema boundary (§24, §11.7)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from kernelrule.agents.mock import (
    ADVERSARIAL_CASES,
    MockLLM,
    _parse_terms,
    _render_rule,
)
from kernelrule.agents.schemas import (
    RuleProposal,
    SchemaViolation,
    validate_rule_proposal,
)

FEATS = ["traffic_amplification", "has_spill", "is_two_stage",
         "log_workspace_bytes", "sm_idle_cost", "split_k_cost",
         "smem_pressure"]


# ---------------------------------------------------------------------------
# The schema boundary — **no partial acceptance** (§26.4)
# ---------------------------------------------------------------------------
BAD_RESPONSES = [
    ("no function", {"code": "x = 1", "w0": [1.0]}),
    ("answer reference",
     {"code": "def score(f,p,hw,w): return time_ms", "w0": [1.0]}),
    ("difficulty reference",
     {"code": "def score(f,p,hw,w): return difficulty", "w0": [1.0]}),
    ("import", {"code": "def score(f,p,hw,w):\n import os\n return 1",
                "w0": [1.0]}),
    ("empty w0", {"code": "def score(f,p,hw,w): return 1", "w0": []}),
    ("string w0", {"code": "def score(f,p,hw,w): return 1", "w0": ["a"]}),
    ("runaway w0", {"code": "def score(f,p,hw,w): return 1", "w0": [1e9]}),
    ("not a dict", ["code"]),
]


@pytest.mark.parametrize("name,obj", BAD_RESPONSES,
                         ids=[c[0] for c in BAD_RESPONSES])
def test_schema_violation_is_discarded(name, obj):
    with pytest.raises(SchemaViolation):
        validate_rule_proposal(obj)


def test_valid_response_passes():
    p = validate_rule_proposal(
        {"code": "def score(f, p, hw, w):\n    return f.waves * w[0]\n",
         "w0": [2.0], "changes": "x"})
    assert isinstance(p, RuleProposal) and p.w0 == [2.0]


# ---------------------------------------------------------------------------
# ★ adversarial — one of them passing means a hole in the defence
# (§24.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,code,w0", ADVERSARIAL_CASES,
                         ids=[c[0] for c in ADVERSARIAL_CASES])
def test_adversarial_case_is_blocked_somewhere(name, code, w0):
    """It must be caught **somewhere** — the schema, the static checks or
    the sandbox."""
    from kernelrule.rules.checks import check_rule

    blocked = False
    try:
        validate_rule_proposal({"code": code, "w0": w0})
    except SchemaViolation:
        blocked = True
    if not blocked:
        rep = check_rule(code, feature_names=set(FEATS) | {"waves",
                         "tail_waste", "edge_waste"},
                         shape_value_names={"is_memory_bound", "M"},
                         n_weights=len(w0))
        blocked = not rep.ok
    if not blocked:
        from kernelrule.core.matrix import Feats, ShapeInfo
        from kernelrule.core.sandbox import run_isolated
        f = Feats({n: np.ones(4) for n in FEATS + ["waves"]})
        out = run_isolated(code, (f, ShapeInfo({"is_memory_bound": 0.0}),
                                  None, np.asarray(w0)), timeout=6.0)
        blocked = not out.ok
    assert blocked, f"{name} passed every defence"


# ---------------------------------------------------------------------------
# mutate — ★ it perturbs the structure
# ---------------------------------------------------------------------------
def test_render_parse_roundtrip():
    code, w0 = _render_rule(["f.a", "f.b"], ("is_memory_bound", "f.c"))
    terms, branch = _parse_terms(code)
    assert terms == ["f.a", "f.b"] and branch == ("is_memory_bound", "f.c")
    assert len(w0) == 3


def test_mutate_changes_structure_not_just_weights():
    """★ Shaking only the weights tests nothing — `fit_weights` fits
    them."""
    m = MockLLM("mutate", seed=1, feature_names=FEATS)
    parent = RuleProposal(code="def score(f, p, hw, w):\n"
                               "    s = f.traffic_amplification * w[0]\n"
                               "    return s\n", w0=[1.0])
    codes = set()
    for _ in range(12):
        out = m.complete("rule_editor", "x", parent=parent,
                         hypothesis={"measurable_with": ["has_spill"]})
        codes.add(out["code"])
    assert len(codes) > 1, "the structure never changes"
    # The term count really does vary
    sizes = {len(_parse_terms(c)[0]) for c in codes}
    assert len(sizes) > 1, f"the term count is fixed: {sizes}"


def test_mutate_follows_the_hypothesis():
    """It adds the feature the hypothesis named first — the path by which
    the diagnosis contributes."""
    m = MockLLM("mutate", seed=3, feature_names=FEATS)
    parent = RuleProposal(code="def score(f, p, hw, w):\n"
                               "    s = f.traffic_amplification * w[0]\n"
                               "    return s\n", w0=[1.0])
    hits = 0
    for _ in range(20):
        out = m.complete("rule_editor", "x", parent=parent,
                         hypothesis={"measurable_with": ["is_two_stage"]})
        if "f.is_two_stage" in out["code"]:
            hits += 1
    assert hits >= 8, f"only {hits}/20 followed the hypothesis"


def test_mutate_respects_the_literal_budget():
    m = MockLLM("mutate", seed=5, feature_names=FEATS * 3)
    p = None
    for _ in range(40):
        out = m.complete("rule_editor", "x", parent=p,
                         hypothesis={"measurable_with": FEATS})
        assert len(out["w0"]) <= 8, out["w0"]
        p = validate_rule_proposal(out)


def test_diagnose_reads_the_unused_column():
    """★ The diagnosis reads the report's `★ unused` column. A test of the
    loop plumbing."""
    m = MockLLM("mutate", seed=0, feature_names=FEATS)
    report = ("is_two_stage       1.0   0.0  ★ unused\n"
              "log_workspace_bytes 22.0 0.0  ★ unused\n"
              "split_k_cost       0.5   0.0  in use\n")
    out = m.complete("analyze", report)
    names = [h["measurable_with"][0] for h in out["hypotheses"]]
    assert "is_two_stage" in names and "log_workspace_bytes" in names
    assert "split_k_cost" not in names


# ---------------------------------------------------------------------------
# replay — deterministic reproduction
# ---------------------------------------------------------------------------
def test_replay_reproduces_exactly(tmp_path):
    a = MockLLM("mutate", seed=11, feature_names=FEATS)
    outs = [a.complete("rule_editor", f"p{i}") for i in range(6)]
    a.dump(tmp_path / "calls")
    b = MockLLM("replay", replay_dir=tmp_path / "calls")
    assert [b.complete("rule_editor", f"p{i}") for i in range(6)] == outs


def test_replay_missing_dir_is_an_error(tmp_path):
    """★ It does not silently fall back to canned (§26.4)."""
    with pytest.raises(FileNotFoundError):
        MockLLM("replay", replay_dir=tmp_path / "nope")


def test_replay_detects_a_changed_loop(tmp_path):
    a = MockLLM("mutate", seed=2, feature_names=FEATS)
    a.complete("rule_editor", "x")
    a.dump(tmp_path / "c")
    b = MockLLM("replay", replay_dir=tmp_path / "c")
    b.complete("rule_editor", "x")
    with pytest.raises(SchemaViolation, match="replay"):
        b.complete("rule_editor", "y")


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError, match="unknown mode"):
        MockLLM("wishful")


def test_deterministic_across_instances():
    a = MockLLM("mutate", seed=9, feature_names=FEATS)
    b = MockLLM("mutate", seed=9, feature_names=FEATS)
    assert ([a.complete("rule_editor", "x") for _ in range(5)]
            == [b.complete("rule_editor", "x") for _ in range(5)])


# ---------------------------------------------------------------------------
# Do the three enforcement sites agree (D-26) — the §30.8 pattern
# ---------------------------------------------------------------------------
# The description, the validator and the error message ran on all three
# different. They were tied to one constant, so **the test pins that fact** —
# the purpose is to stop someone fixing only one of them later.

def test_hypothesis_count_desc_and_validator_share_one_constant():
    from kernelrule.agents import schemas as S
    if not S.HAVE_PYDANTIC:
        pytest.skip("no pydantic")
    # ★ 2026-09-08 (D-144): it is **fixed** at 3. The requirement that the
    #   description, the validation and the error say the same constant is
    #   unchanged (D-26).
    exact = S.N_HYP_MIN == S.N_HYP_MAX
    want = (f"xactly {S.N_HYP_MIN}" if exact   # "Exactly" in the description,
                                               # "exactly" in the error
            else f"{S.N_HYP_MIN}~{S.N_HYP_MAX}")
    desc = S.AnalysisOutput.model_fields["hypotheses"].description
    assert want in desc

    def mk(n):
        return S.AnalysisOutput(hypotheses=[{"claim": f"hypothesis {i}"}
                                             for i in range(n)])

    mk(S.N_HYP_MIN)                                 # the lower bound passes
    for n in (S.N_HYP_MIN - 1, S.N_HYP_MAX + 1):    # outside is refused
        with pytest.raises(Exception) as ei:
            mk(n)
        # The error message must say the same constant
        assert want in str(ei.value)


def test_there_is_no_weight_cap():
    """★ 2026-09-09 (D-150): the parameter cap is gone.

    It used to assert `LIMITS["parameters"] == MAX_WEIGHTS` — one source for
    one cap. There is no cap now, so what is pinned is its absence: a long
    `w0` is accepted, and `LIMITS` no longer carries a budget at all.
    """
    from kernelrule.agents.schemas import validate_rule_proposal
    from kernelrule.rules.checks import LIMITS

    assert "parameters" not in LIMITS
    code = ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
            + "".join(f"    s = s + f.tail_waste * w[{i}]\n"
                      for i in range(1, 20))
            + "    return s\n")
    p = validate_rule_proposal({"code": code, "w0": [1.0] * 20})
    assert len(p.w0) == 20


# ---------------------------------------------------------------------------
# Does banned-word substring matching avoid catching comments (D-27)
# ---------------------------------------------------------------------------

_HDR = "def score(f, p, hw, w):\n"


@pytest.mark.parametrize("code,banned", [
    # What is inside a comment or a string does not run -> not caught
    (_HDR + "    # this shape has high difficulty\n"
            "    return f.tail_waste * w[0]\n", None),
    (_HDR + "    note = 'import os is forbidden'\n"
            "    return f.waves * w[0]\n", None),
    # Real code is still caught
    (_HDR + "    return f.difficulty * w[0]\n", "difficulty"),
    ("import os\n" + _HDR + "    return f.waves * w[0]\n", "import "),
    (_HDR + "    return TABLE.time_ms * w[0]\n", "time_ms"),
])
def test_banned_check_ignores_comments_and_strings(code, banned):
    from kernelrule.agents.schemas import check_banned
    assert check_banned(code) == banned


def test_banned_check_never_skips_on_tokenize_failure():
    """On a syntax error it **checks the original conservatively**
    (§26.4)."""
    from kernelrule.agents.schemas import check_banned
    assert check_banned("def score(  # unclosed\n  import os") == "import "


# ---------------------------------------------------------------------------
# Does it announce Pydantic's absence at the moment of use (4-5)
# ---------------------------------------------------------------------------

def test_missing_pydantic_fails_loudly():
    from kernelrule.agents.schemas import _NoPydantic
    stub = _NoPydantic("AnalysisOutput")
    with pytest.raises(ImportError, match="is \\*\\*disabled\\*\\*"):
        stub()
    with pytest.raises(ImportError, match="is \\*\\*disabled\\*\\*"):
        _ = stub.model_validate    # attribute access alone announces it


def test_changes_is_optional():
    """For lineage tracking. Throwing a rule away for leaving it empty only
    burns retries (4-4)."""
    from kernelrule.agents import schemas as S
    if not S.HAVE_PYDANTIC:
        pytest.skip("no pydantic")
    out = S.RuleOutput(code="def score(f, p, hw, w):\n    return f.waves*w[0]",
                       w0=[1.0])
    assert out.changes == ""


# ---------------------------------------------------------------------------
# FeatureWriter — the F1~F3 conditions (§11.4)
# ---------------------------------------------------------------------------
# The fundamental question is "can the LLM **build** physical quantities".
# For that, what is given under each condition has to be exact — if even one
# existing feature name leaks into F1, that experiment has merely re-asked
# "did it only combine".

def _feat_client():
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import REGISTRY
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    return OpenAILLM(LLMConfig(model="m"), feature_names=[], shape_values=[],
                     registry=REGISTRY, cache=False), REGISTRY


@pytest.mark.parametrize("cond", ["F1"])
def test_f0_f1_leak_no_existing_feature_name(cond):
    """★ No existing feature name may enter the raw-values condition."""
    c, reg = _feat_client()
    text = c._feature_prompt(condition=cond)
    leaked = [n for n in reg._items if n in text]
    assert not leaked, (
        f"an existing feature leaked into the {cond} prompt: {leaked}")


def test_f3_shows_the_existing_features():
    c, reg = _feat_client()
    text = c._feature_prompt(condition="F3")
    assert "tail_waste" in text and "has_spill" in text


def test_unknown_condition_is_rejected():
    c, _ = _feat_client()
    with pytest.raises(ValueError, match="unknown condition"):
        c._feature_prompt(condition="F9")


def test_feature_prompt_example_uses_no_real_feature(monkeypatch):
    """D-35 — **F1's** shape example must not hand over a real feature.

    The conditions that give public knowledge (F2/F3) show the real features
    down to the code — that is the definition of the condition (§30.17).
    Those use `examples/known5.md`.
    """
    from kernelrule.agents.openai_client import load_prompt
    from kernelrule.features import REGISTRY
    body = load_prompt("examples/other_domain.md")
    start = body.index("```python")
    example = body[start:body.index("```", start + 3)]
    leaked = [n for n in REGISTRY._items if n in example]
    assert not leaked, (
        f"the shape example contains a real feature: {leaked}")


def test_prompts_state_no_parameter_cap(monkeypatch):
    """★ 2026-09-09 (D-150): the budget number is gone from the prompts.

    This test used to change the budget constant to 16 and demand that every
    prompt follow. Now there is nothing to follow — what it pins is that no
    prompt states a cap the checker does not enforce, which is the D-105
    failure in the other direction.
    """
    from kernelrule.agents.openai_client import load_prompt

    for f in ("role/_rules_common.md", "role/_rules_edit.md",
              "role/rule_editor.md", "role/rule_writer.md"):
        txt = load_prompt(f)
        assert "{parameters}" not in txt, f"{f}: a budget slot is left"
        for line in txt.splitlines():
            low = line.lower()
            if "no cap" in low:
                continue
            assert "per execution path" not in low, f"{f}: {line}"
            assert "8 parameters" not in low, f"{f}: {line}"


def test_prompt_states_no_size_limit_at_all():
    """★ 2026-09-10 (D-151): this used to require the sentence "there is no
    cap" to reach the model.

    It was itself the problem — **a negation activates what it negates**, and
    the schema field description still carried "at most 8 per execution
    path", which the model quoted back when asked. What is pinned now is that
    no surface the model sees states a size limit at all: not the prompts,
    not the output schema.
    """
    import json

    from kernelrule.agents.openai_client import assemble_instructions
    from kernelrule.agents.schemas import rule_output_for

    surfaces = {"schema": json.dumps(rule_output_for().model_json_schema(),
                                     ensure_ascii=False)}
    for role in ("rule_writer", "rule_editor", "analyze"):
        kw = {"objective": "regret"}
        if role == "rule_writer":
            kw["hw_text"] = "GPU: T\n"
        surfaces[role] = assemble_instructions(role, **kw)
    for name, txt in surfaces.items():
        low = txt.lower()
        for phrase in ("per execution path", "at most 8", "no cap",
                       "{parameters}"):
            assert phrase not in low, f"{name}: {phrase!r} is still there"


# ---------------------------------------------------------------------------
# ★ The five surfaces that reach the model (D-152)
# ---------------------------------------------------------------------------
#: Everything the model reads. A rule stated on one of them and not enforced —
#: or enforced and not stated — is the D-105/107/151/152 family of mistakes.
#:
#:   1 the system prompt
#:   2 the user prompt
#:   3 the output schema's field descriptions
#:   4 ★ the validation failure message — pydantic-ai returns it to the model
#:     on the retry
#:   5 ★ the static checker's refusal message, on the same retry path
def test_no_size_limit_on_any_of_the_five_surfaces():
    """★ Checked **by behaviour**, not by wording (D-152).

    A cap of 8 survived two removals by living in a validator: the words were
    gone from every prompt while `len(w0) > 8` still raised. So this test
    feeds a 20-weight rule through each surface that can refuse.
    """
    import json

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import assemble_instructions
    from kernelrule.agents.schemas import (
        rule_output_for,
        validate_rule_proposal,
    )
    from kernelrule.features import REGISTRY
    from kernelrule.rules.checks import check_rule, limits_for

    code = ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
            + "".join(f"    s = s + f.tail_waste * w[{i}]\n"
                      for i in range(1, 20))
            + "    return s\n")
    w0 = [0.1] * 20

    # 3 + 4: the schema type the model fills, and its validators
    rule_output_for()(code=code, w0=w0)
    rule_output_for(product_hint=True)(code=code, w0=w0)
    # the dict path (MockLLM and anything without structured output)
    assert len(validate_rule_proposal({"code": code, "w0": w0}).w0) == 20
    # 5: the static checker
    rep = check_rule(code, feature_names=REGISTRY.names(shape_level=False),
                     shape_value_names=REGISTRY.names(shape_level=True),
                     n_weights=20, limits=limits_for())
    assert rep.ok, rep.violations
    # a branch, 25 weights
    branched = ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                "    if p.is_memory_bound:\n"
                + "".join(f"        s = s + f.tail_waste * w[{i}]\n"
                          for i in range(1, 13))
                + "    else:\n"
                + "".join(f"        s = s + f.sm_idle_cost * w[{i}]\n"
                          for i in range(13, 25))
                + "    return s\n")
    rule_output_for()(code=branched, w0=[0.1] * 25)
    assert check_rule(branched,
                      feature_names=REGISTRY.names(shape_level=False),
                      shape_value_names=REGISTRY.names(shape_level=True),
                      n_weights=25, limits=limits_for()).ok

    # 1 + 2 + 3: and no surface *states* a limit either
    blob = json.dumps(rule_output_for().model_json_schema(),
                      ensure_ascii=False).lower()
    for role in ("rule_writer", "rule_editor", "analyze"):
        kw = {"objective": "regret"}
        if role == "rule_writer":
            kw["hw_text"] = "GPU: T\n"
        blob += assemble_instructions(role, **kw).lower()
    for phrase in ("per execution path", "at most 8", "the budget is",
                   "no cap", "{parameters}"):
        assert phrase not in blob, f"{phrase!r} is still on a surface"


def test_the_numeric_safety_checks_stay():
    """⚠️ Removing the cap must not remove these — they are not a budget."""
    import pytest as _pytest

    from kernelrule.agents.schemas import rule_output_for

    code = "def score(f, p, hw, w):\n    return f.waves * w[0]\n"
    cls = rule_output_for()
    with _pytest.raises(Exception, match="empty"):
        cls(code=code, w0=[])
    with _pytest.raises(Exception, match="abnormally large"):
        cls(code=code, w0=[1e9])


# ---------------------------------------------------------------------------
# ★ D-160 — the feature metadata is required, and the measured range is not
#           a prompt
# ---------------------------------------------------------------------------
def test_feature_metadata_has_no_defaults():
    """★ With defaults there is no telling "chosen" from "not filled in".

    On the F1 run `expected_range` was the schema default [0,1] in 15 of 20
    and `direction` was the default in **20 of 20** (D-159 §3-1). Checked by
    behaviour: leaving one out must fail validation, and the failure message
    is what the model gets back on the retry.
    """
    from kernelrule.agents import schemas as S
    if not S.HAVE_PYDANTIC:
        pytest.skip("no pydantic")
    import pydantic

    base = {"name": "probe", "code": "def probe(p, hw, cfg) -> float:\n"
                                     "    return 0.0\n",
            "rationale": "why"}
    for missing in ("unit", "expected_range", "direction"):
        kw = dict(base, unit="ratio", expected_range=(0.0, 4.0),
                  direction="higher_is_worse")
        kw.pop(missing)
        with pytest.raises(pydantic.ValidationError) as ei:
            S.FeatureOutput(**kw)
        assert missing in str(ei.value)
    assert S.FeatureOutput(**base, unit="ratio", expected_range=(0.0, 4.0),
                           direction="higher_is_worse").unit == "ratio"


def test_the_measured_range_never_reaches_the_prompt():
    """⛔ D-160 §2-3 — the observed range is recorded, never shown.

    A range read off this table is an observation **of this table**, and
    putting it in front of the model makes the run condition B. Checked by
    behaviour: the numbers `observed_ranges` returns must not appear in the
    rendered feature block, not even with `include_observed=True`.
    """
    import warnings

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY, FeatureRegistry, render_features

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = PerfTable.from_bundle("datasets/rtx-a6000-sm_86-c63710df",
                                  env_hash="c63710df", ok_only=False)
    reg = FeatureRegistry("probe")
    for n in ("log_grid_tiles", "edge_waste"):
        reg.add(REGISTRY[n])
    m = FeatureMatrix(t, reg)
    obs = m.observed_ranges(list(t.shapes())[:4])
    text = render_features(reg, include_observed=True)
    for n, (lo, hi) in obs.items():
        for v in (lo, hi):
            # A measured bound that is not also a declared bound must be
            # absent from the text.
            if v in reg[n].expected_range:
                continue
            assert f"{v:.4g}" not in text, (
                f"the measured range of {n} is in the prompt: {v}")
