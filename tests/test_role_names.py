"""A role name is one thing — no alias is kept (D-93).

## Why no alias is kept

Making a compatibility path that reads the old name means **two names coexist
and they diverge.** It would be the eighth after `is_reference` / `top_k` /
`DEFAULT_MODEL` / `REGISTRY` / `load_generated` / `approx_equal` / the budget
constants (principle 2).

Instead of an alias, **this test blocks it.** Using an old name directly
fails.

## What is an exception

**A historical description** such as "at the time it was called RuleWriter"
in `decisions.md` is left as it is (documentation rule 2 — a wrong value is
not deleted). This test looks only at **the code**.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The role names no longer used. It fails if one appears as a string literal.
#: ⚠️ This tuple holds **the old names**, so a bulk substitution can eat it
#: and silently disable the check — that happened once. The exclusion list
#: below is this file only, for that reason.
_OLD_ROLES = ("architect", "optimize")
#: ★ `critique` is **a role outside the loop**, so `experiments/critic.py`
#: brings it in with `register_role`. What is forbidden is it remaining inside
#: `kernelrule/` — `test_no_critique_role_in_the_agent_package` below looks at
#: that.


def _py_files():
    yield from (ROOT / "kernelrule").rglob("*.py")
    yield from (ROOT / "experiments").glob("*.py")
    yield from (ROOT / "tests").glob("*.py")


def test_no_old_role_string_literals():
    """★ It fails if an old role name is left in the code **as a string**."""
    bad: list[str] = []
    for f in _py_files():
        rel = f.relative_to(ROOT).as_posix()
        if rel == "tests/test_role_names.py":
            continue                      # this test has to know the old names
        try:
            tree = ast.parse(f.read_text(), filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            v = node.value
            if v in _OLD_ROLES or (v == "critique"
                                   and rel.startswith("kernelrule/")):
                bad.append(f"  {rel}:{node.lineno}  {v!r}")
    assert not bad, (
        "an old role name is left in the code. No alias is kept, so this is "
        "the only line of defence (D-93):\n" + "\n".join(bad))


def test_no_critique_role_in_the_agent_package():
    """★ The Critic is **outside the loop** (D-92).

    Leaving a role the loop does not call inside `kernelrule/agents/` reads as
    "something to switch on some day" and keeps getting dragged into the
    condition list and the ablation table. The experiment script brings its
    own prompt and its own schema with `register_role`.
    """
    hits = [f.relative_to(ROOT).as_posix()
            for f in (ROOT / "kernelrule").rglob("*")
            if f.is_file() and f.suffix in (".py", ".md")
            and "critique" in f.read_text(errors="ignore")]
    assert not hits, f"critique is left in kernelrule/: {hits}"
    assert not (ROOT / "kernelrule/agents/prompts/role/critique.md").exists()


def test_loop_roles_are_exactly_four():
    """The loop calls exactly four roles — if the list grows, it is caught
    here."""
    from kernelrule.agents.openai_client import (
        _EDITS_RULES,
        _NEEDS_HW,
        _WRITES_RULES,
    )

    assert set(_NEEDS_HW) == {"rule_writer"}
    assert set(_WRITES_RULES) == {"rule_editor", "rule_writer"}
    assert set(_EDITS_RULES) == {"rule_editor"}


@pytest.mark.parametrize("name", ["rule_writer", "rule_editor", "analyze",
                                  "feature", "categorize"])
def test_register_role_refuses_to_shadow_a_loop_role(name):
    """If a registered role shadows a loop role, **something else runs
    silently**."""
    import os

    _ = os.environ.setdefault("OPENAI_API_KEY", "test-key")
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import REGISTRY

    llm = OpenAILLM(LLMConfig(),
                    feature_names=REGISTRY.names(shape_level=False),
                    shape_values=REGISTRY.names(shape_level=True),
                    registry=REGISTRY, cache=False)
    with pytest.raises(ValueError, match="is a loop role"):
        llm.register_role(name, instructions="x", output_type=dict)


def test_the_old_role_names_are_still_named_here():
    """★ The guard above only works while this tuple holds **the old** names.

    A bulk substitution once rewrote the tuple itself and the check silently
    passed on everything. ⚠️ 2026-09-09 (D-147): the move script
    (`rename_roles.py`, D-93) was deleted — it had 0 files left to move — so
    the identity check that read its `RENAME` went with it. This is what is
    left of it, and it is the part that matters.
    """
    assert _OLD_ROLES == ("architect", "optimize"), (
        "the old-name tuple was rewritten — the check above is now vacuous")
