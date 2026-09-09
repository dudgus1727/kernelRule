"""Do the prompt documents avoid table-derived claims (§12.3b / D-32)?

## Why it is needed

§3 blocks the rule function from seeing the table in four layers. But
**moving that table's conclusions in as sentences** makes those four layers
do nothing.

    "warp_m=128 was never optimal"
      = a sentence that requires knowing the optimum of all 66 shapes
      = the answer summarised into the prompt

Counting on the training split makes no difference (§12.3b). And leaving the
correction history in the prompt means **writing the sentence again while
saying it was removed** (§12.3c) — which is exactly what happened in
`hw/sm_86.md` (deleted 2026-09-08, D-146).

## Limits

It is a string check, so it is not complete. The purpose is to make it
**visible the next time someone adds one**, not to prove anything. The
structural defence is `report/table_facts.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROMPTS = Path(__file__).resolve().parents[1] / "kernelrule/agents/prompts"

#: Traces of a sentence that can only be written by looking at the table.
#:
#: ⚠️ 2026-09-08 (D-146): the prompts became English and the patterns
#: followed. The Korean forms were kept for one day for the frozen
#: `hw/sm_86.md`; that file is deleted and every file this scans is English,
#: so they are gone. What this scans is `prompts/*.md` and the schema
#: descriptions — both live code, both checked for Korean by
#: `test_prompt_layout` with no exception.
_LEAK = (
    (re.compile(r"\d+\s*of\s*\d+\s*shapes"), "M of N shapes — a full tally"),
    (re.compile(r"\d+\s*/\s*66|\d+\s*/\s*61"), "N/66 · N/61 — this table's denominator"),
    (re.compile(r"(was|were)\s+never\s+(?:been\s+)?optimal"), "'was never optimal'"),
    (re.compile(r"optimal\s*0\s*times"), "'optimal 0 times'"),
    (re.compile(r"hypothesis\s+(was\s+)?rejected"), "a hypothesis rejected by the table"),
    (re.compile(r"in this table"), "'in this table' — it points at the table"),
    (re.compile(r"(entered|is in|are in) the answer set"), "an answer-set tally"),
)

#: The hardware specs are known without the table. A number is not a leak.
_HW_OK = re.compile(
    r"SM\s*84|101,?376|65,?536|1,?536|116\.1|729\.7|159\.1|6\s*MB|"
    r"1\.024|154\.8|768\s*GB|1350|7601|99KB")


def _prompt_files() -> list[Path]:
    return sorted(PROMPTS.rglob("*.md"))


def test_prompts_exist():
    assert _prompt_files(), f"no prompt was found: {PROMPTS}"


@pytest.mark.parametrize("path", _prompt_files(), ids=lambda p: p.name)
def test_prompt_has_no_table_derived_claim(path: Path):
    """★ A claim that came from the table must not be in a prompt
    (§12.3b).

    A hit is handled two ways.
      - it really came from the table  -> remove it. The correction history
                                         goes in `docs/` (§12.3c)
      - hardware / execution model     -> add it to `_HW_OK` and leave one
                                         line on **why it is known without
                                         the table**
    """
    hits = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if _HW_OK.search(line):
            continue
        for pat, why in _LEAK:
            if pat.search(line):
                hits.append(f"  {path.name}:{i}  [{why}]\n    {line.strip()}")
                break
    assert not hits, (
        "there is a table-derived claim in a prompt (§12.3b). Blocking the "
        "rule function from seeing the table and then putting that table's "
        "conclusions in as sentences makes §3 do nothing:\n"
        + "\n".join(hits))


# ---------------------------------------------------------------------------
# ★ The schema field descriptions are **part of the prompt** too (D-90)
# ---------------------------------------------------------------------------
#
#   The `description` of a structured-output schema is passed to the model as
#   it is. But when the prompts were edited this was not swept, and it ended
#   up in a state where **the system prompt and the field description said
#   opposite things within the same request**:
#
#     _rules_common.md  "do not give a careless w0 ... reflect the physical
#                        magnitude"
#     schemas.py        "roughly is enough — the numerical optimiser fits it"
#
#   The counterpart of principle 26 — when the checker and the prompt
#   diverge the model believes the prompt, but when the prompt and the schema
#   diverge **the model sees both.**


def _schema_descriptions() -> dict[str, str]:
    from kernelrule.agents.schemas import HAVE_PYDANTIC

    if not HAVE_PYDANTIC:
        pytest.skip("no pydantic")
    import kernelrule.agents.schemas as S

    out: dict[str, str] = {}
    for name in dir(S):
        cls = getattr(S, name)
        fields = getattr(cls, "model_fields", None)
        if not isinstance(fields, dict):
            continue
        for fname, f in fields.items():
            if getattr(f, "description", None):
                out[f"{name}.{fname}"] = f.description
    return out


def test_schema_descriptions_have_no_table_leak():
    """The leak check applied to the prompts is applied **to the schema
    descriptions too**."""
    bad = []
    for where, text in _schema_descriptions().items():
        for rx, why in _LEAK:
            if rx.search(text):
                bad.append(f"  {where}: {why}")
    assert not bad, ("there is a table-derived sentence in a schema field "
                     "description — this goes to the model too:\n"
                     + "\n".join(bad))


def test_shared_schema_does_not_mention_a_parent():
    """★ `RuleOutput` is used **by both** RuleEditor and RuleWriter.

    Putting talk of a parent into the description makes RuleWriter look for a
    parent that does not exist. The same reason `_rules_edit.md` was taken
    out of RuleWriter (§30.10). The replacement instruction is inserted
    dynamically by the RuleEditor prompt's `budget_note`.
    """
    from kernelrule.agents.schemas import HAVE_PYDANTIC

    if not HAVE_PYDANTIC:
        pytest.skip("no pydantic")
    import kernelrule.agents.schemas as S

    for fname in ("code", "w0"):
        d = S.RuleOutput.model_fields[fname].description or ""
        assert "parent" not in d.lower(), (
            f"the RuleOutput.{fname} description mentions a parent — "
            f"RuleWriter has no parent")


def test_w0_description_agrees_with_the_prompt():
    """★ Did the `w0` description follow the §29 correction (D-54, D-90)?

    The prompt says "do not give them carelessly, reflect the physical
    magnitude" while the schema alone was left at "roughly is enough".
    """
    from kernelrule.agents.openai_client import load_prompt
    from kernelrule.agents.schemas import HAVE_PYDANTIC

    if not HAVE_PYDANTIC:
        pytest.skip("no pydantic")
    import kernelrule.agents.schemas as S

    d = S.RuleOutput.model_fields["w0"].description or ""
    assert "roughly is enough" not in d, (
        "it says the opposite of the prompt")
    assert "Do not give them carelessly" in d
    assert "do not give a careless `w0`" in load_prompt("role/_rules_common.md")


# ---------------------------------------------------------------------------
# ★ The narrow path by which something from the table leaks into a rule
# (D-114)
# ---------------------------------------------------------------------------
#
# The static checks block `p.M > 1024`. But **a hypothesis sentence does not
# go through them** — `claim` goes to the RuleEditor whole through
# `json.dumps`, and if "at M=4096" is in there it can be copied across as a
# literal.


@pytest.mark.parametrize("claim", [
    "at M=4096 split-K always loses",
    "on the 4096x4096 shape the tail grows",
    "only shapes with N = 11008 should be handled differently",
])
def test_hypothesis_claim_refuses_shape_sizes(claim):
    import kernelrule.agents.schemas as S

    if not S.HAVE_PYDANTIC:                              # pragma: no cover
        pytest.skip("no pydantic")
    with pytest.raises(Exception, match="shape size"):
        S.HypothesisOut(claim=claim, evidence_cases=[1])


@pytest.mark.parametrize("claim", [
    "on shapes where waves < 1 the tail grows",
    "a config with stages=3 loses on short shapes",
    "at split_k=8 the reduction dominates",
])
def test_hypothesis_claim_allows_regimes_and_config_values(claim):
    """★ It looks at false positives — config values under three digits must
    not be caught."""
    import kernelrule.agents.schemas as S

    if not S.HAVE_PYDANTIC:                              # pragma: no cover
        pytest.skip("no pydantic")
    assert S.HypothesisOut(claim=claim, evidence_cases=[1]).claim == claim


def test_evidence_cases_do_not_reach_the_rule_editor():
    """A case number points at something the RuleEditor cannot see."""
    import os

    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import FeatureRegistry

    os.environ.setdefault("OPENAI_API_KEY", "t")
    llm = OpenAILLM(LLMConfig(), feature_names=[], shape_values=[],
                    registry=FeatureRegistry("F1"))
    hyp = {"claim": "it loses on short shapes",
           "evidence_cases": [17, 23, 41],
           "affected_regime": "waves < 1", "id": "H9",
           "proposed_direction": "strengthen the tail term",
           "measurable_with": ["tail_waste"]}
    txt = llm._user_prompt("rule_editor", "", parent=None, parent_n_terms=0,
                           hypothesis=hyp, analyst=True)
    block = txt.split("## This round's hypothesis")[1][:600]
    # ★ Only the allow list goes (D-117)
    for keep in ("it loses on short shapes", "strengthen the tail term",
                 "tail_waste"):
        assert keep in block, keep
    # The rest is for the Analyst record — it stays in `hypotheses.jsonl`
    for drop in ("evidence_cases", "affected_regime", "waves < 1", "H9"):
        assert drop not in block, drop


def test_prompt_no_longer_names_the_optimizer():
    """The rank path is L-BFGS-B. Writing the name down makes them diverge
    again (pending 12)."""
    from kernelrule.agents.openai_client import load_prompt

    txt = load_prompt("role/_rules_common.md", parameters=8)
    assert "Nelder-Mead" not in txt
    assert "fitted by a numerical optimiser" in txt


def test_n_candidates_is_described_as_enumeration_not_performance():
    from kernelrule.agents.openai_client import load_prompt

    txt = load_prompt("role/_rules_common.md", parameters=8)
    assert "p.n_candidates" in txt and (
        "enumeration information, not performance" in txt)
