"""★ Are the hardware facts **the ones for this table** (D-113)?

The default of `LLMConfig.arch_prompt` was pinned to `"hw/sm_86.md"` (that
file was deleted on 2026-09-08, D-146) and
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
    txt, facts = hw_prompt_from_bundle(bundle, env_hash=env_hash)
    check_hw_prompt(txt, t.hw)
    assert facts["arch"] == t.hw.arch
    assert f"SMs        {t.hw.sm_count}" in txt
    assert f"{t.hw.ridge_point:.1f} FLOP/byte" in txt


def test_generated_a6000_prompt_has_the_bundle_numbers():
    """★ The numbers below are the ones the hand-written `hw/sm_86.md` held.
    They stay written out here — that is what made deleting the file
    (2026-09-08, D-146) checkable rather than a leap of faith.

    The body differs from that file (the codename `GA102` is not in
    `env.json`). What must match is the numbers — those are the condition.
    """
    from kernelrule.agents.hwprompt import hw_prompt_from_bundle

    # ⚠️ 2026-09-11 (D-166): `tick (1.024 us)` left this list with the
    #    measurement-limit section. The rest are the frozen file's numbers
    #    and they stay.
    txt, _ = hw_prompt_from_bundle(A6000[0], env_hash=A6000[1])
    for want in ("SMs        84", "101,376 B", "6 MB", "116.1 TFLOP/s",
                 "729.7 GB/s", "159.1 FLOP/byte"):
        assert want in txt, f"cannot reproduce {want!r} from the frozen file"
    assert "Limits of measurement" not in txt, (
        "the measurement-limit section is back in the RuleWriter prompt "
        "(D-166 §E)")


# ---------------------------------------------------------------------------
# ★ Does it catch it in reverse (principle 38 — break it on purpose and
# watch it fail once)
# ---------------------------------------------------------------------------


def test_a6000_prompt_on_a_5090_table_is_refused():
    """One GPU's facts put on another's table must be refused."""
    from kernelrule.agents.hwprompt import (
        HwPromptError,
        check_hw_prompt,
        hw_prompt_from_bundle,
    )

    t = _table(*G5090)
    txt, _ = hw_prompt_from_bundle(A6000[0], env_hash=A6000[1])
    with pytest.raises(HwPromptError, match="5090"):
        check_hw_prompt(txt, t.hw)


def test_same_gpu_but_different_effective_numbers_is_refused():
    """★ Checking only the name lets **another bundle of the same GPU**
    pass. The second leg used to be the timer tick; the tick section left
    with the measurement-limit section (D-166 §E), so the anchor is now the
    numbers the prompt actually asserts — two bundles of one card whose
    clocks were locked differently differ exactly there.

    ⛔ One leg is not enough. That is what D-113 was defending."""
    from dataclasses import replace

    from kernelrule.agents.hwprompt import (
        HwPromptError,
        check_hw_prompt,
        hw_prompt_from_bundle,
    )

    t = _table(*A6000)
    txt, _ = hw_prompt_from_bundle(A6000[0], env_hash=A6000[1])
    check_hw_prompt(txt, t.hw)                           # passes when right
    other = replace(t.hw, peak_tflops_f16=t.hw.peak_tflops_f16 * 1.1,
                    bandwidth_gbps=t.hw.bandwidth_gbps * 1.1)
    with pytest.raises(HwPromptError, match="effective numbers"):
        check_hw_prompt(txt, other)


# ---------------------------------------------------------------------------
# ★ D-166 §E — the measurement-limit section is **no longer in this prompt**
# ---------------------------------------------------------------------------
#
#   It held a tick table, a per-table verdict, and an advisory ("do not
#   refine below the tick"). It is gone, for two reasons:
#
#     1 RuleWriter cannot refine — no parent, no score, no report. The
#       objective block is kept out of RuleWriter for the same reason
#     2 the verdict was read at `min(table.best_time(q) for q in
#       table.shapes())` — a measured minimum over **every** shape. On the
#       H100 that shape is in the holdout
#
#   ⛔ The lessons of D-113 · D-116 · D-117 are **not** deleted with it. The
#   tick and noise information still reaches the Analyst and the RuleEditor
#   through the diagnostic report, and the tests that guarded those lessons
#   moved to `tests/test_diagnostic.py` against that renderer.


def test_the_measurement_limit_section_is_gone():
    """★ D-166 §E. Both halves: the section, and the argument that fed it."""
    import inspect

    from kernelrule.agents import hwprompt

    for bundle, env_hash in (A6000, G5090):
        txt, facts = hwprompt.hw_prompt_from_bundle(bundle, env_hash=env_hash)
        for gone in ("Limits of measurement", "noise floor", "tick (",
                     "statistical "):
            assert gone not in txt, f"{gone!r} is back in the prompt"
        # the verdict made from the answer is not in the facts either
        for k in ("min_ms", "tick_binds", "tick_pct_at_min", "sigma_at_min"):
            assert k not in facts, f"{k} is back — it came from the answer"
        assert facts["tick_ms"] > 0, "the bundle's tick is still recorded"

    sig = inspect.signature(hwprompt.hw_prompt_from_bundle).parameters
    assert "table" not in sig and "min_ms" not in sig, (
        "an argument nothing reads is exactly the leftover D-166 removed")


def test_the_hardware_facts_themselves_are_untouched():
    """⛔ What was removed is the measurement-limit section — **not** the
    hardware block or the execution model. D-113's defence rests on those."""
    from kernelrule.agents.hwprompt import hw_prompt_from_bundle

    txt, _ = hw_prompt_from_bundle(A6000[0], env_hash=A6000[1])
    for want in ("GPU        ", "SMs        ", "smem       ", "ridge      ",
                 "Execution model", "split-K divides K"):
        assert want in txt, f"{want!r} was removed by mistake"
