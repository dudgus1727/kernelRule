"""★ The rule-writing surfaces must say **the same thing** (D-105).

A campaign once ran with `--rule-budget 16` while only the checker was 16 —
the role files and the user prompt rendered 8, all 36 rules stopped at 8
terms, and it was nearly read as "raising the budget does not raise the term
count". The lesson is that a rule the model must obey lives on four surfaces
at once: **the prompt / the schema / the checker / the assembling code.**

⚠️ 2026-09-09 (D-150): **the parameter cap itself is gone.** The tests that
pinned the number across the four surfaces went with it — there is no number.
What is left here is the wiring that still carries something: the objective
block, the hints, and the fitter choice.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _llm(budget: int | None):
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import FeatureRegistry

    os.environ.setdefault("OPENAI_API_KEY", "t")
    return OpenAILLM(LLMConfig(parameters=budget), feature_names=[],
                     shape_values=[], registry=FeatureRegistry("F1"))


# ---------------------------------------------------------------------------
# ★ The caps **attached** to the budget (D-106)
# ---------------------------------------------------------------------------
#
# An 8-term rule has a measured median of 271 AST nodes and a maximum of 383,
# against a cap of 400. Raising only the budget to 16 makes 16-term rules
# **refused at the node cap** — what gets measured is not "a budget of 16 has
# no effect" but "16 terms could not be used".


# ---------------------------------------------------------------------------
# ★ The output schema (D-107) — the fourth place
# ---------------------------------------------------------------------------
#
# `pydantic-ai` **sends `RuleOutput`'s field descriptions to the model as the
# tool schema.** With "★ at most 8 terms" frozen in there, all 29 rules of
# the budget-16 campaign had 8 terms even though the prompt said "a cap of
# 16". Schema refusals were 0 across all 36 rounds — the model never even
# tried.


#: ★ **Every surface** the budget number goes out on. One missed and the
#: condition changes.
#:
#:   D-105  only the checker was reached (the prompt files / the user prompt)
#:   D-106  the attached cap (ast_nodes) did not follow
#:   D-107  the output schema's description was frozen at 8
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
