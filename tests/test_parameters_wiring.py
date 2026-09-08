"""★ The parameter budget goes to **every prompt** as the same value
(D-105).

A campaign was run with `--rule-budget 16`, and only the checker was 16 —
**the role files of the system prompt and the user prompt rendered 8.**
Within the same prompt, `_rules_common.md` said "at most 16" while
`rule_editor.md` said "a term cap of 8" (the wording changed in D-128 to "a
parameter cap of N" — what is checked is **whether the four surfaces say the
same number**). All 36 rules stopped at 8 terms, and it was nearly read as
"raising the budget does not raise the term count".

The cause was that there were several places (principle 23):

    load_prompt(..., parameters=)   only `assemble_instructions` passed it
    load_prompt("role/...")         `_agent` and `_optimize_prompt` did not
    checks.PARAMETERS               the user prompt **imported it directly**

The old test `test_rule_writers_get_the_budget` was `"8" in body` — 8 also
appears in the feature descriptions, so it passes even when changed to 16.
**A changing value must not be searched for as a constant.**
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "kernelrule/agents/openai_client.py"


def test_every_load_prompt_call_passes_the_budget():
    """★ It counts **every** call site — one missed and the prompt
    diverges."""
    tree = ast.parse(SRC.read_text())
    bad = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
           and n.func.id == "load_prompt"
           and "parameters" not in {k.arg for k in n.keywords}]
    assert not bad, (
        f"there is a call without `parameters=`: {SRC.name} lines {bad}. "
        f"Falling back to the default ignores `parameters` (D-105).")


#: The functions allowed to read `checks.PARAMETERS`. Read anywhere else it
#: ignores `parameters` — the condition silently returns to 8.
_MAY_READ_PARAMETERS = {"load_prompt", "__init__"}


def test_only_two_functions_read_the_module_constant():
    """★ It **counts** the places that read `checks.PARAMETERS`. There is
    one effective budget."""
    tree = ast.parse(SRC.read_text())
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if fn.name in _MAY_READ_PARAMETERS:
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and n.id in ("BUDGET", "_CHECK_BUDGET"):
                bad.append((fn.name, n.lineno))
    assert not bad, (
        f"places that read `checks.PARAMETERS` directly: {bad}. "
        f"Only `self._budget` may be looked at (D-105).")


def _llm(budget: int | None):
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import FeatureRegistry

    os.environ.setdefault("OPENAI_API_KEY", "t")
    return OpenAILLM(LLMConfig(parameters=budget), feature_names=[],
                     shape_values=[], registry=FeatureRegistry("F1"))


@pytest.mark.parametrize("budget", [8, 16])
def test_user_and_system_prompts_agree_on_the_budget(budget):
    """Do the system and user prompts say **the same number**?"""
    from kernelrule.agents.openai_client import assemble_instructions

    llm = _llm(budget)
    other = 16 if budget == 8 else 8
    sys_p = assemble_instructions("rule_editor", objective="rank",
                                  parameters=llm._parameters)
    usr_p = llm._user_prompt("rule_editor", "", parent=None,
                             parent_n_terms=0, analyst=False)
    for name, txt in (("system", sys_p), ("user", usr_p)):
        # ★ Wording history: "a term cap of N" -> "a parameter cap of N"
        #   (D-128) -> "N per execution path" (D-144). The requirement that
        #   the number comes from one place is unchanged.
        assert (f"{budget} per execution path" in txt
                or f"at most {budget}" in txt), (
            f"the budget {budget} is not visible in the {name} prompt")
        assert (f"{other} per execution path" not in txt
                and f"at most {other}" not in txt), (
            f"the {name} prompt says {other} — the condition changed "
            f"(D-105)")


def test_budget_reaches_the_saturation_notice():
    """The saturation notice must see the effective budget too — an
    8-term parent told "the budget is full" under a budget of 16 can never
    add a term.

    ★ 2026-09-08 (D-144): with the budget now **per path**, the room left is
    counted from `parent_path_params` (the heaviest path). Counting from
    `parent_n_terms` (the total term count) lies to a parent that split its
    branches.
    """
    llm = _llm(16)
    txt = llm._user_prompt("rule_editor", "", parent=None,
                           parent_n_terms=8, parent_path_params=8,
                           analyst=False)
    assert "is at the cap" not in txt, (
        "budget is 16 but a parent with 8 on its path was told it is at the "
        "cap (D-105)")
    assert "Room left on the heaviest path: 8" in txt, txt[:400]

    # ★ A parent that split its branches — 14 terms in total but the
    #   heaviest path is 8
    txt2 = llm._user_prompt("rule_editor", "", parent=None,
                            parent_n_terms=14, parent_path_params=8,
                            analyst=False)
    assert "is at the cap" not in txt2, (
        "8 per path but told 'at the cap' from the total of 14 terms (D-144)")


# ---------------------------------------------------------------------------
# ★ The caps **attached** to the budget (D-106)
# ---------------------------------------------------------------------------
#
# An 8-term rule has a measured median of 271 AST nodes and a maximum of 383,
# against a cap of 400. Raising only the budget to 16 makes 16-term rules
# **refused at the node cap** — what gets measured is not "a budget of 16 has
# no effect" but "16 terms could not be used".


def test_limits_scale_with_the_budget():
    from kernelrule.rules.checks import LIMITS, limits_for

    assert limits_for(None) == LIMITS
    assert limits_for(8)["ast_nodes"] == LIMITS["ast_nodes"]
    assert limits_for(16)["ast_nodes"] == 2 * LIMITS["ast_nodes"]
    assert limits_for(16)["max_lines"] == 2 * LIMITS["max_lines"]


def test_prompt_states_the_scaled_node_cap():
    """If the prompt says the constant 400 it diverges from the checker
    (800)."""
    from kernelrule.agents.openai_client import load_prompt
    from kernelrule.rules.checks import limits_for

    for b in (8, 16):
        txt = load_prompt("role/_rules_edit.md", parameters=b)
        n = limits_for(b)["ast_nodes"]
        assert f"{n} AST nodes" in txt, (
            f"the node cap does not match at budget {b}")


def test_a_sixteen_term_rule_fits_only_under_the_raised_cap():
    """★ Confirmed with a real 16-term rule — matching the numbers alone is
    useless."""
    from kernelrule.rules.checks import check_rule, limits_for

    terms = "\n".join(
        f"    s = s + np.where(p.is_memory_bound, f.log_dram_traffic, "
        f"f.log_inst_total) * w[{i}]" for i in range(1, 16))
    code = ("def score(f, p, hw, w):\n"
            "    s = f.reg_pressure * w[0]\n" + terms + "\n    return s")
    kw = {"feature_names": ["reg_pressure", "log_dram_traffic",
                            "log_inst_total"],
          "shape_value_names": ["is_memory_bound"], "n_weights": 16}
    lo = check_rule(code, limits=limits_for(8), **kw)
    hi = check_rule(code, limits=limits_for(16), **kw)
    assert not lo.ok, (
        "if 16 terms pass at a budget of 8, the checker is not filtering")
    assert hi.n_nodes > limits_for(8)["ast_nodes"], (
        f"there are only {hi.n_nodes} nodes, so the cap check is never "
        f"touched — this test cannot measure what it meant to")
    assert hi.ok, f"refused even at a budget of 16: {hi.violations}"


# ---------------------------------------------------------------------------
# ★ The output schema (D-107) — the fourth place
# ---------------------------------------------------------------------------
#
# `pydantic-ai` **sends `RuleOutput`'s field descriptions to the model as the
# tool schema.** With "★ at most 8 terms" frozen in there, all 29 rules of
# the budget-16 campaign had 8 terms even though the prompt said "a cap of
# 16". Schema refusals were 0 across all 36 rounds — the model never even
# tried.


def _twelve_terms() -> str:
    body = "".join(f"    s = s + f.edge_waste * w[{i}]\n" for i in range(1, 12))
    return ("def score(f, p, hw, w):\n"
            "    s = f.reg_pressure * w[0]\n" + body + "    return s")


@pytest.mark.parametrize("budget", [8, 16])
def test_output_schema_states_the_budget(budget):
    from kernelrule.agents.schemas import rule_output_for

    schema = rule_output_for(budget).model_json_schema()
    for fld in ("code", "w0"):
        d = schema["properties"][fld]["description"]
        assert f"At most {budget}" in d or f"at most {budget}" in d, (
            f"the output schema's {fld} description does not state the "
            f"budget {budget} — the model sets the term count from this "
            "sentence (D-107)")


def test_output_schema_validation_follows_the_budget():
    """★ If only the description is fixed and the validation stays at 8,
    the model tries and gets refused."""
    from kernelrule.agents.schemas import rule_output_for

    kw = {"code": _twelve_terms(), "w0": [1.0] * 12, "changes": "",
          "hypothesis_id": ""}
    with pytest.raises(Exception, match="12"):
        rule_output_for(8)(**kw)
    rule_output_for(16)(**kw)          # at a budget of 16 it must pass


def test_dict_path_validation_follows_the_budget():
    """The MockLLM path and any path not using structured output must see
    the same budget."""
    from kernelrule.agents.schemas import (
        SchemaViolation,
        validate_rule_proposal,
    )

    d = {"code": _twelve_terms(), "w0": [1.0] * 12}
    with pytest.raises(SchemaViolation):
        validate_rule_proposal(d)
    validate_rule_proposal(d, parameters=16)


#: ★ **Every surface** the budget number goes out on. One missed and the
#: condition changes.
#:
#:   D-105  only the checker was reached (the prompt files / the user prompt)
#:   D-106  the attached cap (ast_nodes) did not follow
#:   D-107  the output schema's description was frozen at 8
def test_all_four_surfaces_say_the_same_budget():
    from kernelrule.agents.openai_client import assemble_instructions
    from kernelrule.agents.schemas import rule_output_for
    from kernelrule.rules.checks import limits_for

    for b in (8, 16):
        llm = _llm(b)
        surfaces = {
            "system prompt": assemble_instructions(
                "rule_editor", objective="rank", parameters=llm._parameters),
            "user prompt": llm._user_prompt(
                "rule_editor", "", parent=None, parent_n_terms=0,
                analyst=False),
            "output schema": json.dumps(
                rule_output_for(b).model_json_schema(), ensure_ascii=False),
        }
        for name, txt in surfaces.items():
            assert (f"{b} per execution path" in txt
                    or f"at most {b}" in txt
                    or f"At most {b}" in txt), (
                f"{name} does not state the budget {b}")
        assert limits_for(b)["parameters"] == b


# ---------------------------------------------------------------------------
# ★ The numbers of the goal definition (k, lambda) and the product term
# (D-109 / D-110)
# ---------------------------------------------------------------------------
#
# The `rank` goal block wrote "100 configs" as **a constant**. Running the
# `k` sweep as it was would have made the prompt a fifth surface saying 100
# on its own.


@pytest.mark.parametrize("k", [10, 20, 100])
def test_objective_block_states_the_running_k(k):
    from kernelrule.agents.openai_client import assemble_instructions

    txt = assemble_instructions("rule_editor", objective="rank", parameters=8,
                                rank_top_k=k)
    assert f"the {k} genuinely fastest" in txt, (
        f"the objective block does not state k={k}")
    for other in (10, 20, 100):
        if other != k:
            assert f"the {other} genuinely fastest" not in txt


def test_objective_block_states_lambda_only_when_set():
    from kernelrule.agents.openai_client import assemble_instructions

    kw = {"objective": "rank", "parameters": 8}
    assert ("getting the true first place right"
            not in assemble_instructions("rule_editor", **kw))
    on = assemble_instructions("rule_editor", rank_lambda=1.0, **kw)
    assert "getting the true first place right" in on


def test_product_hint_is_off_by_default_and_lands_on_every_surface():
    from kernelrule.agents.openai_client import assemble_instructions
    from kernelrule.agents.schemas import rule_output_for

    off = assemble_instructions("rule_editor", objective="rank", parameters=8)
    on = assemble_instructions("rule_editor", objective="rank", parameters=8,
                               product_hint=True)
    assert "{product_block}" not in off and "{product_note}" not in off
    assert "you may multiply features" not in off
    assert "you may multiply features" in on and (
        "multiplying two features" in on)
    for ph in (False, True):
        d = rule_output_for(8, product_hint=ph).model_json_schema()
        has = ("may multiply two features"
               in d["properties"]["code"]["description"])
        assert has is ph, (
            f"schema product_hint={ph} but the product sentence is {has}")


# ---------------------------------------------------------------------------
# ★ The fitter is **decided by the parameter count** (D-128)
# ---------------------------------------------------------------------------
def test_fitter_is_derived_from_the_parameter_count():
    """At most 8, Nelder-Mead; above it, CMA. **Decided in one place
    only.**

    Rationale: in 8 dimensions the two fitters are indistinguishable
    (D-125), and in 16 dimensions Nelder-Mead's reach rate is 92%, short of
    the mark (D-77 · D-123).
    """
    from kernelrule.rules.checks import PARAMETERS, fitter_for

    for n in (None, 4, 8):
        f = fitter_for(n)
        assert f == {"fit_method": "nelder-mead", "fit_restarts": 4,
                     "max_evals": 200}, n
    for n in (9, 16, 32):
        f = fitter_for(n)
        assert f == {"fit_method": "cma", "fit_restarts": 1,
                     "max_evals": 300}, n
    assert fitter_for(PARAMETERS)["fit_method"] == "nelder-mead"


def test_fitter_keys_are_loopconfig_fields():
    """★ `**fitter_for(n)` must splat straight in — it leaves no place to
    diverge."""
    import dataclasses

    from kernelrule.core.loop import LoopConfig
    from kernelrule.rules.checks import fitter_for

    fields = {f.name for f in dataclasses.fields(LoopConfig)}
    assert set(fitter_for(8)) <= fields
    cfg = LoopConfig(run_id="x", parameters=16, **fitter_for(16))
    assert (cfg.fit_method, cfg.fit_restarts, cfg.max_evals) == ("cma", 1, 300)


def test_pipeline_has_no_fitter_or_rank_flags():
    """★ `--fit-method` / `--objective` are **gone** (D-128).

    The fitter is derived from the parameters, and the evolution objective is
    only regret. A remaining flag would mean "the condition can be given by
    hand", and the rule leaks.
    """
    src = (Path(__file__).resolve().parents[1]
           / "experiments" / "f1_pipeline.py").read_text()
    for flag in ('"--fit-method"', '"--fit-restarts"', '"--max-evals"',
                 '"--objective"', '"--objective-switch"', '"--rank-top-k"',
                 '"--rank-lambda"'):
        assert flag not in src, f"{flag} is still there"
