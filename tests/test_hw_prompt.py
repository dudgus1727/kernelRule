"""★ Are the hardware facts **the ones for this table** (D-113)?

The default of `LLMConfig.arch_prompt` was pinned to `"hw/sm_86.md"` and
`f1_pipeline` did not change it. There was only one file in `hw/`, so another
architecture **could not be chosen in the first place.** So the §29.5 (c)
regeneration run on the 5090 table received A6000 facts.

What these tests hold:

```
**generated** from the bundle    writing it by hand diverges again (principle 2)
there is no default              missing means failure (§26.4)
does it catch it in reverse      by putting the A6000 prompt on the 5090 table
```
"""
from __future__ import annotations

import warnings

import pytest

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")


def _table(path, env_hash):
    from pathlib import Path

    from kernelrule.core.table import PerfTable

    if not (Path(path) / "BUNDLE.json").exists():
        pytest.skip(f"no bundle: {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PerfTable.from_bundle(path, env_hash=env_hash, ok_only=False)


# ---------------------------------------------------------------------------
# ★ There is no default
# ---------------------------------------------------------------------------


def test_rule_writer_without_hardware_facts_fails():
    """★ It must not silently fall back to a default — that was D-113."""
    from kernelrule.agents.openai_client import assemble_instructions

    with pytest.raises(ValueError, match="hardware facts"):
        assemble_instructions("rule_writer", objective="rank", parameters=8)


def test_roles_without_hardware_still_assemble():
    """RuleEditor/FeatureWriter do not receive hw (§16.2) — they must not
    be blocked."""
    from kernelrule.agents.openai_client import assemble_instructions

    for role in ("rule_editor", "feature", "analyze"):
        assert assemble_instructions(role, objective="rank", parameters=8)


def test_llm_config_has_no_default_hardware():
    from kernelrule.agents.openai_client import LLMConfig

    c = LLMConfig()
    assert c.arch_prompt is None and c.hw_text is None, (
        "a default is still alive — then it goes silently again (D-113)")


# ---------------------------------------------------------------------------
# ★ Generated from the bundle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("bundle", "env_hash"), [A6000, G5090])
def test_generated_prompt_matches_its_bundle(bundle, env_hash):
    from kernelrule.agents.hwprompt import check_hw_prompt, hw_prompt_from_bundle

    t = _table(bundle, env_hash)
    txt, facts = hw_prompt_from_bundle(bundle, env_hash=env_hash, table=t)
    check_hw_prompt(txt, t.hw, float(t.noise.tick_ms))
    assert facts["arch"] == t.hw.arch
    assert f"SMs        {t.hw.sm_count}" in txt
    assert f"{t.hw.ridge_point:.1f} FLOP/byte" in txt


def test_generated_a6000_prompt_reproduces_the_frozen_numbers():
    """★ The generator reproduces the **numbers** of the old hand-written
    file exactly.

    The body differs (the codename `GA102` is not in `env.json`). What must
    match is the numbers — those are the condition.

    ⚠️ 2026-09-08 (D-146): `hw/sm_86.md` was deleted. The numbers below are
    **the ones that file held** and they stay written here — that is what
    makes the deletion checkable (the file itself is at commit `ee53b4d`).
    """
    from kernelrule.agents.hwprompt import hw_prompt_from_bundle

    txt, _ = hw_prompt_from_bundle(A6000[0], env_hash=A6000[1],
                                   table=_table(*A6000))
    # ★ 2026-09-08 (D-146): the prompt became English. **What has to be
    #   reproduced is the numbers**, so only the numbers are checked.
    for want in ("SMs        84", "101,376 B", "6 MB", "116.1 TFLOP/s",
                 "729.7 GB/s", "159.1 FLOP/byte", "tick (1.024 us)"):
        assert want in txt, f"cannot reproduce {want!r} from the frozen file"


# ---------------------------------------------------------------------------
# ★ Does it catch it in reverse (principle 38 — break it on purpose and
# watch it fail once)
# ---------------------------------------------------------------------------


def test_a6000_prompt_on_a_5090_table_is_refused():
    """★ 2026-09-08 (D-146): this used to load the frozen `hw/sm_86.md`. That
    file was deleted, so the A6000 prompt is **generated** here — what is
    tested is the same thing, one GPU's facts put on another's table."""
    from kernelrule.agents.hwprompt import (
        HwPromptError,
        check_hw_prompt,
        hw_prompt_from_bundle,
    )

    t = _table(*G5090)
    a6000 = _table(*A6000)
    txt, _ = hw_prompt_from_bundle(A6000[0], env_hash=A6000[1], table=a6000)
    with pytest.raises(HwPromptError, match="5090"):
        check_hw_prompt(txt, t.hw, float(t.noise.tick_ms))


def test_same_gpu_but_wrong_tick_is_refused():
    """★ Checking only the name lets **a different tick of the same GPU**
    pass."""
    from kernelrule.agents.hwprompt import (
        HwPromptError,
        check_hw_prompt,
        hw_prompt_from_bundle,
    )

    t = _table(*A6000)
    txt, _ = hw_prompt_from_bundle(A6000[0], env_hash=A6000[1], table=t)
    check_hw_prompt(txt, t.hw, float(t.noise.tick_ms))   # passes when right
    with pytest.raises(HwPromptError, match="tick"):
        check_hw_prompt(txt, t.hw, float(t.noise.tick_ms) * 4)


def test_tick_table_is_computed_not_hardcoded():
    """The 5090's tick is 1/64 of the A6000's — the table must reflect
    that."""
    from kernelrule.agents.hwprompt import hw_prompt_from_bundle

    a, _ = hw_prompt_from_bundle(A6000[0], env_hash=A6000[1],
                                 table=_table(*A6000))
    g, _ = hw_prompt_from_bundle(G5090[0], env_hash=G5090[1],
                                 table=_table(*G5090))
    assert "9.091%" in a, "the A6000's minimum row changed"
    assert "9.091%" not in g, (
        "the 5090 prompt states the A6000's tick ratio — it is nailed in as "
        "a constant")


# ---------------------------------------------------------------------------
# ★ The **conclusion** of the measurement-limit section differs per table
# (D-116)
# ---------------------------------------------------------------------------
#
# The noise floor is `max(statistical term, tick term)`, and which one wins
# differs. Sending the A6000's conclusion ("short shapes are buried inside
# the tick") to the 5090 is a **wrong warning** — on the 5090 the
# statistical term is larger at every length.


def test_tick_advisory_follows_which_term_binds():
    from kernelrule.agents.hwprompt import hw_prompt_from_bundle

    a, fa = hw_prompt_from_bundle(A6000[0], env_hash=A6000[1],
                                  table=_table(*A6000))
    g, fg = hw_prompt_from_bundle(G5090[0], env_hash=G5090[1],
                                  table=_table(*G5090))

    assert fa["tick_binds"] is True and fg["tick_binds"] is False
    # A6000 — the tick binds
    assert "A difference inside one tick may as well not exist" in a
    assert "the tick is not the limit" not in a
    # 5090 — the tick does not bind
    assert "the tick is not the limit" in g
    assert "A difference inside one tick may as well not exist" not in g


def test_both_noise_terms_are_shown():
    """★ Both terms must be shown for the model to know **why**."""
    from kernelrule.agents.hwprompt import hw_prompt_from_bundle

    for bundle, env_hash in (A6000, G5090):
        txt, _ = hw_prompt_from_bundle(bundle, env_hash=env_hash,
                                       table=_table(bundle, env_hash))
        assert "tick " in txt and "statistical " in txt
        assert "The noise floor is the **larger** of two terms" in txt


def test_binding_term_is_recorded_as_a_condition():
    """It is a condition, so it must stay in the artefact (principle 39)."""
    from kernelrule.agents.hwprompt import hw_prompt_from_bundle

    _, f = hw_prompt_from_bundle(G5090[0], env_hash=G5090[1],
                                 table=_table(*G5090))
    for k in ("tick_binds", "tick_pct_at_min", "sigma_at_min", "min_ms"):
        assert k in f


def test_judgement_point_comes_from_the_table_not_a_constant():
    """★ Judging at a length not in the table is a criterion unrelated to
    the table (D-117)."""
    import pytest as _pt

    from kernelrule.agents.hwprompt import HwPromptError, hw_prompt_from_bundle

    with _pt.raises(HwPromptError, match="min_ms"):
        hw_prompt_from_bundle(A6000[0], env_hash=A6000[1])   # neither given
    for bundle, env_hash in (A6000, G5090):
        t = _table(bundle, env_hash)
        _, f = hw_prompt_from_bundle(bundle, env_hash=env_hash, table=t)
        want = float(min(t.best_time(q) for q in t.shapes()))
        assert abs(f["min_ms"] - want) < 1e-12
