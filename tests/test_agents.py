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


def test_weight_cap_has_one_source_of_truth():
    from kernelrule.agents.schemas import MAX_WEIGHTS
    from kernelrule.rules.checks import LIMITS
    assert LIMITS["parameters"] == MAX_WEIGHTS


def test_mock_and_real_paths_enforce_the_same_budget():
    """§24 — `validate_rule_proposal` alone had no w0 length check.

    Developing on the mock, a budget overrun went uncaught and was caught
    only on the real LLM.
    """
    from kernelrule.agents.schemas import (
        MAX_WEIGHTS,
        SchemaViolation,
        validate_rule_proposal,
    )
    code = "def score(f, p, hw, w):\n    return f.waves * w[0]\n"
    validate_rule_proposal({"code": code, "w0": [1.0] * MAX_WEIGHTS})
    with pytest.raises(SchemaViolation, match="budget"):
        validate_rule_proposal({"code": code, "w0": [1.0] * (MAX_WEIGHTS + 1)})


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


def test_prompts_never_hardcode_the_budget_number(monkeypatch):
    """★ The budget number has **one** source: `checks.PARAMETERS`.

    Five prompt files, the schema and the checker each wrote their own 8. A
    change misses one — it would be the sixth after `is_reference` /
    `top_k` / `DEFAULT_MODEL` / `REGISTRY` / `load_generated`.

    If the prompt follows when `checks.PARAMETERS` changes, it is a single
    source.
    """
    from kernelrule.agents.openai_client import load_prompt
    from kernelrule.rules import checks

    files = ["role/_rules_common.md", "role/_rules_edit.md",
             "role/rule_editor.md", "role/rule_writer.md"]
    monkeypatch.setattr(checks, "PARAMETERS", 16)
    for f in files:
        txt = load_prompt(f)
        assert "{budget}" not in txt, f"{f}: the substitution did not happen"
        assert "16" in txt, f"{f}: the budget does not flow into the prompt"
        # No old number may remain in a budget sentence
        for line in txt.splitlines():
            low = line.lower()
            if "budget" in low or "cap" in low or "literal" in low:
                assert " 8 " not in line and "8 parameters" not in line \
                    and "<= 8" not in line, (
                        f"{f}: a frozen 8 remains — {line}")


def test_prompt_tells_the_model_branch_constants_are_free():
    """★ When the rule changes, **the model must know too** (D-78).

    Loosening only the checker and leaving the prompt as it is keeps the
    model going around — because it does not know the constraint was
    loosened.

    ★ 2026-09-08 (D-145): it looks at the **assembled prompt**. Which file
    the explanation lives in depends on deduplication (§3-4), and what the
    model receives is the assembled result. Checking per file breaks the
    test every time a duplicate is removed.
    """
    from kernelrule.agents.openai_client import assemble_instructions

    for role in ("rule_writer", "rule_editor"):
        kw = {"objective": "regret", "parameters": 8}
        if role == "rule_writer":
            kw["hw_text"] = "GPU: T\n"
        txt = assemble_instructions(role, **kw)
        assert "branch condition" in txt and "not parameters" in txt, (
            f"{role}: the exemption is not explained")
