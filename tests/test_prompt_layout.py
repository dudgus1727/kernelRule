"""The two-axis prompt layout (§30.10).

The original design split along one axis only, "hardware-independent /
dependent". **It did not split along "role-independent / dependent"**, so
things no role needed piled up in the common part — the FeatureWriter
received the regret definition, the weight budget of 8 and the gallery of
refused rules every time.

              hardware-independent   hardware-dependent
    role-indep _base.md              the generated hw facts
    role-dep   role/*.md             (none)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kernelrule.agents.openai_client import (
    _EDITS_RULES,
    _NEEDS_HW,
    load_prompt,
)

ROLES = ("analyze", "rule_editor", "feature", "rule_writer")
PROMPTS = Path(__file__).resolve().parents[1] / "kernelrule/agents/prompts"


#: A stand-in for the hardware facts. The real text is **generated from the
#: bundle** (`hwprompt`, D-113) and `tests/test_hw_prompt.py` is where its
#: content is checked. These tests are about layout, so all they need is a
#: block carrying hardware markers.
HW_TEXT = "GPU  NVIDIA RTX A6000 (sm_86)\nSMs  84\n"


def _instructions(role: str, *, objective: str = "rank") -> str:
    """★ It calls **the same function** as `_agent()` (principle 2).

    The assembly used to be rewritten here. When `{objective_block}`
    appeared, only the test side left it unfilled and they diverged —
    assembly happens in one place only.

    ★ The hardware facts are **passed in explicitly.** The default is gone
    (D-113), and a test leaning on a default cannot see that the default is a
    condition.
    """
    from kernelrule.agents.openai_client import assemble_instructions

    return assemble_instructions(role, objective=objective,
                                 hw_text=HW_TEXT)


# ---------------------------------------------------------------------------
# ★ Does the FeatureWriter avoid receiving rule material
# ---------------------------------------------------------------------------

#: What must not be in the FeatureWriter prompt. All of it is **rule**
#: material.
_RULE_ONLY = ("regret", "8 parameters", "literal", "w[0]", "np.random",
              "Parent rule", "hypothesis", "lookup table")


def test_feature_prompt_has_no_rule_material():
    body = _instructions("feature")
    hit = [t for t in _RULE_ONLY if t in body]
    assert not hit, (
        f"there is rule material in the FeatureWriter prompt: {hit}\n"
        "The FeatureWriter's job is only to find a physical quantity from "
        "raw values — it need not know the downstream pipeline (§30.10).")


def test_feature_and_rule_editor_prompts_have_no_hardware_constants():
    """★ Not seeing hw makes that prompt **GPU-independent** (§16.2)."""
    marks = [m for m in ("RTX A6000", "sm_86") if m in HW_TEXT]
    assert marks, "the hw block carries no marker — this check is moot"
    for role in ("feature", "rule_editor"):
        body = _instructions(role)
        hit = [m for m in ("RTX A6000", "sm_86") if m in body]
        assert not hit, (
            f"a hardware constant leaked into the {role} prompt: {hit}")


def test_rule_writers_get_the_rule_shape():
    """★ 2026-09-09 (D-150): this used to check that the parameter budget
    rendered **as the value given** — `"8" in body` had once passed while the
    prompt said 16 (D-105).

    There is no budget now, so what is pinned is the opposite: the rule
    shape reaches both roles and **no cap sentence does.**
    """
    for role in ("rule_editor", "rule_writer"):
        body = _instructions(role)
        assert "w[0]" in body, f"{role} has no rule shape"
        assert "per execution path" not in body, f"{role}: a cap sentence"
        assert "{parameters}" not in body, f"{role}: an unfilled budget slot"


def test_hw_goes_only_to_roles_that_need_it():
    """★ RuleWriter only. The Analyst receives the same facts in block 1 of
    the report."""
    assert set(_NEEDS_HW) == {"rule_writer"}


def test_architect_does_not_get_the_edit_block():
    """`role/rule_writer.md` says "no scores"; receiving the regret
    definition would contradict that head on (§30.10)."""
    assert "rule_writer" not in _EDITS_RULES
    body = _instructions("rule_writer")
    assert "no scores" in body, "the role file changed — this check is moot"
    assert "regret` = " not in body, (
        "the regret definition leaked into RuleWriter")


def test_optimizer_gets_the_edit_block():
    """Only the RuleEditor receives the edit block (the goal definition +
    the refused cases).

    ★ The goal definition's sentence differs per objective (D-101). The
    default is `rank`, so "regret` = " must not be searched for — it checks
    **whether the section exists**.
    """
    body = _instructions("rule_editor")
    assert "## How you are scored" in body and "actually got rejected" in body
    assert "{objective_block}" not in body, (
        "the placeholder was not filled")
    assert "Lower is better" in body


def test_objective_block_differs_and_only_for_the_editor():
    """★ Changing the objective changes **only the RuleEditor** (D-101).

    RuleWriter has "no scores" and does not receive this section. A failure
    here means changing the objective also changes RuleWriter's condition,
    and then the experiment plan's list of "what changes" is wrong.
    """
    a = {r: _instructions(r, objective="regret")
         for r in ("rule_editor", "rule_writer", "analyze", "feature")}
    b = {r: _instructions(r, objective="rank")
         for r in ("rule_editor", "rule_writer", "analyze", "feature")}
    assert a["rule_editor"] != b["rule_editor"]
    for r in ("rule_writer", "analyze", "feature"):
        assert a[r] == b[r], f"{r} is affected by the objective"


# ---------------------------------------------------------------------------
# ★ The same sentence in two role files diverges (principle 2)
# ---------------------------------------------------------------------------

#: Lines not counted as duplicates. Markdown structure, or too short.
def _meaningful(line: str) -> bool:
    t = line.strip()
    return (len(t) >= 30 and not t.startswith(("#", "```", "|", "-", ">", "<!--"))
            and t not in ("", "---"))


def test_no_duplicate_sentences_between_role_files():
    seen: dict[str, str] = {}
    dupes: list[str] = []
    for f in sorted(PROMPTS.glob("role/*.md")):
        for line in f.read_text().splitlines():
            if not _meaningful(line):
                continue
            t = line.strip()
            if t in seen and seen[t] != f.name:
                dupes.append(f"  {seen[t]} <-> {f.name}: {t[:60]}")
            seen.setdefault(t, f.name)
    assert not dupes, (
        "the same sentence appears in two role files — fixing one of them "
        "makes them diverge (principle 2). If it is genuinely common, lift "
        "it into `_base.md` or `role/_rules_common.md`:\n" + "\n".join(dupes))


def test_base_is_not_duplicated_into_role_files():
    base_lines = {ln.strip() for ln in load_prompt("_base.md").splitlines()
                  if _meaningful(ln)}
    dupes = []
    for f in sorted(PROMPTS.glob("role/*.md")):
        for line in f.read_text().splitlines():
            if _meaningful(line) and line.strip() in base_lines:
                dupes.append(f"  {f.name}: {line.strip()[:60]}")
    assert not dupes, ("a sentence from `_base.md` is copied into a role "
                       "file:\n" + "\n".join(dupes))


# ---------------------------------------------------------------------------
# Does the example avoid handing over the answer (D-35)
# ---------------------------------------------------------------------------
def test_feature_examples_are_from_another_domain():
    """★ **Under F1**, an example that touches a GEMM config axis hands
    over the answer.

    For F2/F3, giving public knowledge is the definition of the condition,
    so they show the real features down to the code (§30.17) — those use
    `examples/known5.md` and are not the subject of this check.
    """
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.features import REGISTRY

    block = load_prompt("examples/other_domain.md")
    leaked = [n for n in REGISTRY._items if n in block]
    assert not leaked, f"the example contains a real feature: {leaked}"
    # Config axis names must not appear either
    axes = ("tile_m", "tile_n", "tile_k", "split_k", "stages", "warp_m",
            "smem", "cp_async")
    hit = [a for a in axes if a in block]
    assert not hit, f"the example touches a GEMM config axis: {hit}"


@pytest.mark.parametrize("role", ROLES)
def test_every_role_gets_the_base(role):
    assert "Measurements are not available at deployment time" in _instructions(role)


def test_rule_writer_is_told_it_gets_no_cases():
    """★ RuleWriter receives no cases, and the role file says so (§30.10).

    ⚠️ 2026-09-08 (D-146): this used to also grep the hw block for a sentence
    pointing at cases. That block was hand-written prose (`hw/sm_86.md`); it
    is generated from the bundle now, so there is nothing to grep.
    """
    arch = _instructions("rule_writer")
    assert "no cases" in arch, "the role file changed — this check is moot"


def test_analyst_gets_hardware_facts_from_the_report_not_a_file():
    """Block 1 of the report and `hw/*.md` are **the same facts**. The
    report is generated from the table every time and the file is fixed, so
    giving both makes them diverge when the bundle changes.
    """
    import warnings

    from kernelrule.core.table import PerfTable
    from kernelrule.report.diagnostic import hardware_block

    body = _instructions("analyze")
    assert "RTX A6000" not in body, (
        "hw is in the Analyst system prompt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = PerfTable.from_bundle("datasets/rtx-a6000-sm_86-c63710df",
                                  env_hash="c63710df", ok_only=False)
    blk = hardware_block(t.hw, t.noise)
    for fact in ("RTX A6000", "84", "ridge point"):
        assert fact in blk, f"block 1 of the report has no {fact!r}"


# ---------------------------------------------------------------------------
# ★ **A real feature name** must not be nailed into any prompt (D-35,
# D-65)
#
#   The magnitude-matching example in `role/rule_writer.md` hardcoded
#   `f.traffic_amplification` / `f.tail_waste`. Under F1 that **hands over
#   the answer** — a name that is not in the registry, yet it points at the
#   physics. The `if p.is_memory_bound:` of `_base.md` was the same.
# ---------------------------------------------------------------------------
def test_no_prompt_hardcodes_a_registry_feature_name():
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.features import REGISTRY

    bad: list[str] = []
    for f in sorted(PROMPTS.rglob("*.md")):
        rel = f.relative_to(PROMPTS).as_posix()
        if rel in _KNOWN_BY_DESIGN:
            continue
        body = f.read_text()
        for n in REGISTRY._items:
            if n in body:
                bad.append(f"  {rel}: {n}")
    assert not bad, (
        "a prompt nailed in a real feature name — under F1 it is a name "
        "not in the registry and it points at the physics (D-35). Use a "
        "placeholder such as `f.<name>`. If it is an example for a condition "
        "that gives public knowledge, put it in `_KNOWN_BY_DESIGN` but "
        "**confirm that the file does not reach F1**:\n"
        + "\n".join(bad))


def test_known_by_design_files_never_reach_f1():
    """★ Does the exempted file really not reach F1 — the premise of the
    exception is checked."""
    from kernelrule.agents.openai_client import _EXAMPLES

    for cond in ("F1",):
        assert f"examples/{_EXAMPLES[cond]}.md" not in _KNOWN_BY_DESIGN, cond


# ---------------------------------------------------------------------------
# ★ F2 — the condition that starts from the five public facts (§30.17)
# ---------------------------------------------------------------------------
def _feature_prompt(condition: str):
    import os

    import kernelrule.features.known5 as K
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import FeatureRegistry

    os.environ.setdefault("OPENAI_API_KEY", "t")
    reg = FeatureRegistry(condition)
    if condition not in ("F0", "F1"):
        for n in sorted(K.KNOWN5._items):
            reg.add(K.KNOWN5[n])
    llm = OpenAILLM(LLMConfig(), feature_names=[], shape_values=[],
                    registry=reg)
    return llm._user_prompt("feature", "", condition=condition, registry=reg)


#: Files that contain real feature names **deliberately**.
#:
#:   examples/known5.md      the feature example for the conditions that give
#:                           public knowledge (§30.17)
#:   examples/rule_known.md  the **rule** example for the same conditions
#:                           (§30.20)
#:
#: Both use "only names already in the registry". That invariant is held by
#: `test_rule_example_never_names_a_feature_outside_the_registry` and
#: `test_known_by_design_files_never_reach_f0_or_f1` — **the exception is
#: made and whether it leaks is checked alongside.**
_KNOWN_BY_DESIGN = {"examples/known5.md", "examples/rule_known.md"}


#: Statements knowable only from the table. One of them in the prompt is a
#: §12.3 violation.
#:
#: ⚠️ The prompts are English (D-146). The Korean list they replaced is in
#: the history, not here.
_MEASURED = ("in this table", "optimal 0 times", "picked as optimal",
             "rel median", "7.4%", "13.6", "37.2", "off by 26%",
             "answer set")


def test_f2_prompt_has_no_measurement():
    """★ The most important point of this work — removing `has_spill`'s
    table observations."""
    body = _feature_prompt("F2")
    hit = [m for m in _MEASURED if m in body]
    assert not hit, (
        f"there is a measured statement in the F2 prompt: {hit}\n"
        "Only what is knowable without the table stays (§12.3, §30.17).")


def test_f2_shows_the_five_with_sources():
    import kernelrule.features.known5 as K

    body = _feature_prompt("F2")
    for n in K.KNOWN5._items:
        assert f"f.{n}" in body or f"p.{n}" in body, f"{n} does not appear"
    assert body.count("Source:") >= 5, "there are fewer than five sources"


def test_f2_does_not_leak_the_other_nineteen():
    """★ The other 19 are the F3 condition."""
    import re

    import kernelrule.features.known5 as K
    from kernelrule.features import REGISTRY

    body = _feature_prompt("F2")
    rest = sorted(set(REGISTRY._items) - set(K.KNOWN5._items))
    leak = [n for n in rest if re.search(rf"\b{re.escape(n)}\b", body)]
    assert not leak, f"the other 19 leaked: {leak}"


def test_examples_differ_by_condition():
    """F1 gets an unrelated domain; the conditions that give public
    knowledge get the real features (D-35)."""
    f1 = _feature_prompt("F1")
    f2 = _feature_prompt("F2")
    assert "branch_divergence_cost" in f1 and "queue_backlog" in f1
    assert "branch_divergence_cost" not in f2
    assert "def tail_waste" in f2 and "Do not rebuild" in f2


def test_areas_are_fixed_and_do_not_name_features():
    """An area is only "a place where something is measured", not "build
    this" (§30.18)."""
    from kernelrule.agents.openai_client import load_prompt

    areas = load_prompt("areas.md")
    body = areas[areas.index("```") + 3:areas.rindex("```")]
    rows = [ln for ln in body.splitlines() if "|" in ln]
    assert len(rows) == 7, f"there are not seven areas: {len(rows)}"
    # There must be no enumeration naming the features to build
    for banned in ("wave quantization", "tile waste"):
        assert banned not in areas, banned


def test_known5_values_are_identical_to_physical(perf_table_for_known5):
    """★ The cleaned-up version must produce **the same values** as the
    original for "the known features were given" to be true."""
    import numpy as np

    import kernelrule.features.known5 as K
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY, FeatureRegistry

    t = perf_table_for_known5
    shapes = list(t.shapes())[:3]
    for n in sorted(K.KNOWN5._items):
        ra, rb = FeatureRegistry("a"), FeatureRegistry("b")
        ra.add(REGISTRY[n])
        rb.add(K.KNOWN5[n])
        ma, mb = FeatureMatrix(t, ra), FeatureMatrix(t, rb)
        sl = REGISTRY[n].shape_level
        for p in shapes:
            fa, ia = ma.for_shape(p)
            fb, ib = mb.for_shape(p)
            a = np.atleast_1d(np.asarray(getattr(ia if sl else fa, n), float))
            b = np.atleast_1d(np.asarray(getattr(ib if sl else fb, n), float))
            assert np.allclose(a, b, rtol=1e-9, atol=1e-12), n


@pytest.fixture(scope="module")
def perf_table_for_known5():
    import warnings

    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PerfTable.from_bundle("datasets/rtx-a6000-sm_86-c63710df",
                                     env_hash="c63710df", ok_only=False)


def test_internal_notes_never_reach_the_model():
    """★ `<!-- ... -->` is a note for humans. It must not go to the model.

    **Internal decision numbers** such as `§30.18`, `D-45` and `D-47` really
    were going out as they were — they cost tokens, leak internal
    references, and under some conditions could hand over the answer.
    """
    from kernelrule.agents.openai_client import load_prompt

    bad = []
    for f in sorted(PROMPTS.rglob("*.md")):
        rel = f.relative_to(PROMPTS).as_posix()
        if "<!--" not in f.read_text():
            continue                       # a file with no notes
        if "<!--" in load_prompt(rel):
            bad.append(f"  {rel}")
    assert not bad, ("the comments are not stripped:\n" + "\n".join(bad))

    # They must be absent from the fully rendered text too
    for cond in ("F1", "F2", "F3"):
        body = _feature_prompt(cond)
        assert "<!--" not in body, cond
        for tag in ("§30.", "D-45", "D-47", "D-63"):
            assert tag not in body, (
                f"the internal reference {tag} is in {cond}")


# ---------------------------------------------------------------------------
# ★ §30.20 — the RuleWriter rule example differs per condition too
#
#   FeatureWriter's example differs per condition while RuleWriter had a
#   single placeholder. Give a good example, but it **must not hand over the
#   answer** (D-35).
#
#   The condition name is not used as the key — RuleWriter's `condition` is
#   A/B (with or without table observations), a different axis from the
#   feature conditions. **It is decided by looking at the registry.**
# ---------------------------------------------------------------------------
def _reg(names):
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.features import REGISTRY, FeatureRegistry

    r = FeatureRegistry("probe")
    for n in names:
        r.add(REGISTRY[n])
    return r


def test_rule_example_is_chosen_by_registry_contents():
    import kernelrule.features.known5 as K
    from kernelrule.agents.openai_client import _rule_example_for
    from kernelrule.features import REGISTRY, FeatureRegistry

    human = _reg(sorted(REGISTRY._items))
    k5 = FeatureRegistry("k5")
    for n in sorted(K.KNOWN5._items):
        k5.add(K.KNOWN5[n])

    assert "f.tail_waste" in _rule_example_for(human)
    assert "f.tail_waste" in _rule_example_for(k5)
    # With a registry lacking the names it falls back to the unrelated
    # domain
    assert "f.tail_waste" not in _rule_example_for(FeatureRegistry("empty"))


def test_rule_example_never_names_a_feature_outside_the_registry():
    """★ This is the real invariant — not the condition name but **whether
    it leaks**."""
    import re

    import kernelrule.features.known5 as K
    from kernelrule.agents.openai_client import _rule_example_for
    from kernelrule.features import REGISTRY, FeatureRegistry

    k5 = FeatureRegistry("k5")
    for n in sorted(K.KNOWN5._items):
        k5.add(K.KNOWN5[n])
    cases = {"human24": _reg(sorted(REGISTRY._items)), "known5": k5,
             "empty": FeatureRegistry("empty"),
             "partial": _reg(["waves", "edge_waste"])}
    for tag, r in cases.items():
        ex = _rule_example_for(r)
        leak = [n for n in REGISTRY._items
                if re.search(rf"[fp]\.{re.escape(n)}\b", ex)
                and n not in r._items]
        assert not leak, (
            f"{tag}: a name outside the registry is in the example {leak}")


def test_rule_examples_keep_placeholders():
    """Given a finished rule it submits it as is and the structural
    comparison collapses (D-35)."""
    from kernelrule.agents.openai_client import load_prompt

    for f in ("examples/rule_known.md", "examples/rule_other_domain.md"):
        body = load_prompt(f)
        assert "<" in body and ">" in body, f"{f} has no placeholder"
        assert "re-weighting" in body and "selection" in body, (
            f"{f} does not show the difference between the two")


def test_no_korean_on_the_llm_path():
    """★ What goes to the LLM is **English only** (D-146).

    Korean is expensive in tokens — measured, the `rule_editor` input was
    1.08 tokens per character (32.8% Korean), while pure English is one
    token per 3.5~4 characters. Over 15 runs x 12 rounds that is about a
    million tokens (20~25% of the total load).

    ⚠️ `docs/` is not on this path. Most of it stays Korean; the artefact
    documents that code reads or writes were translated with it (D-146 §7).
    """
    import json
    import re

    from kernelrule.agents.openai_client import (
        assemble_instructions,
        load_prompt,
    )

    KO = re.compile(r"[가-힣]")
    bad: list[str] = []

    # (1) the four assembled system prompts
    for role in ("analyze", "feature", "rule_writer", "rule_editor"):
        kw: dict = {"objective": "regret", "parameters": 8}
        if role == "rule_writer":
            kw["hw_text"] = "GPU: TEST\n"
        hits = KO.findall(assemble_instructions(role, **kw))
        if hits:
            bad.append(f"system prompt {role}: {''.join(hits[:20])}")

    # (2) the prompt files. ★ The exception for the frozen `hw/sm_86.md` is
    #     gone — that file was deleted on 2026-09-08 (D-146), so every file
    #     left here is on today's LLM path and every one of them is checked.
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "kernelrule/agents/prompts"
    for f in sorted(root.rglob("*.md")):
        hits = KO.findall(load_prompt(str(f.relative_to(root))))
        if hits:
            bad.append(f"{f.relative_to(root)}: {''.join(hits[:20])}")

    # (3) the descriptions of the output schemas
    from kernelrule.agents import schemas as S
    if S.HAVE_PYDANTIC:
        for name in ("AnalysisOutput", "CategoryOutput", "CritiqueOutput"):
            js = json.dumps(getattr(S, name).model_json_schema(),
                            ensure_ascii=False)
            hits = KO.findall(js)
            if hits:
                bad.append(f"schema {name}: {''.join(hits[:20])}")
        js = json.dumps(S.rule_output_for(8).model_json_schema(),
                        ensure_ascii=False)
        if (hits := KO.findall(js)):
            bad.append(f"schema RuleOutput: {''.join(hits[:20])}")

    # (4) the feature block — it goes into the rule prompt whole
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.features import REGISTRY, render_features
    if (hits := KO.findall(render_features(REGISTRY, include_observed=False))):
        bad.append(f"feature block: {''.join(hits[:20])}")

    assert not bad, ("there is Korean on the LLM path:\n  "
                     + "\n  ".join(bad))
