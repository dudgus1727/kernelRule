"""★ 하드웨어 사실이 **그 표의 것**인가 (D-113).

`LLMConfig.arch_prompt` 의 기본값이 `"hw/sm_86.md"` 로 고정돼 있었고
`f1_pipeline` 이 그것을 안 바꿨다. `hw/` 에 파일이 하나뿐이라 다른
아키텍처는 **애초에 고를 수 없었다.** 그래서 5090 표로 돌린 §29.5 (c)
재생성이 A6000 사실을 받았다.

여기 시험이 지키는 것:

```
번들에서 **생성**한다        손으로 쓰면 또 갈린다 (원칙 2)
기본값이 없다               없으면 실패다 (§26.4)
되돌려서 잡는가              A6000 프롬프트를 5090 표에 붙여 보고
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
        pytest.skip(f"번들 없음: {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PerfTable.from_bundle(path, env_hash=env_hash, ok_only=False)


# ---------------------------------------------------------------------------
# ★ 기본값이 없다
# ---------------------------------------------------------------------------


def test_rule_writer_without_hardware_facts_fails():
    """★ 조용히 기본값으로 떨어지면 안 된다 — 그것이 D-113 이었다."""
    from kernelrule.agents.openai_client import assemble_instructions

    with pytest.raises(ValueError, match="하드웨어 사실"):
        assemble_instructions("rule_writer", objective="rank", budget=8)


def test_roles_without_hardware_still_assemble():
    """RuleEditor/FeatureWriter 는 hw 를 안 받는다 (§16.2) — 막으면 안 된다."""
    from kernelrule.agents.openai_client import assemble_instructions

    for role in ("rule_editor", "feature", "analyze"):
        assert assemble_instructions(role, objective="rank", budget=8)


def test_llm_config_has_no_default_hardware():
    from kernelrule.agents.openai_client import LLMConfig

    c = LLMConfig()
    assert c.arch_prompt is None and c.hw_text is None, (
        "기본값이 살아 있다 — 그러면 또 조용히 간다 (D-113)")


# ---------------------------------------------------------------------------
# ★ 번들에서 생성한다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("bundle", "env_hash"), [A6000, G5090])
def test_generated_prompt_matches_its_bundle(bundle, env_hash):
    from kernelrule.agents.hwprompt import check_hw_prompt, hw_prompt_from_bundle

    t = _table(bundle, env_hash)
    txt, facts = hw_prompt_from_bundle(bundle, env_hash=env_hash)
    check_hw_prompt(txt, t.hw, float(t.noise.tick_ms))
    assert facts["arch"] == t.hw.arch
    assert f"{t.hw.sm_count}개" in txt
    assert f"{t.hw.ridge_point:.1f} FLOP/byte" in txt


def test_generated_a6000_prompt_reproduces_the_frozen_numbers():
    """★ 손으로 쓴 옛 파일의 **숫자**를 생성기가 그대로 낸다.

    본문은 다르다 (코드명 `GA102` 는 `env.json` 에 없다). 같아야 하는
    것은 숫자다 — 그것이 조건이다.
    """
    from kernelrule.agents.hwprompt import hw_prompt_from_bundle

    txt, _ = hw_prompt_from_bundle(*A6000[:1], env_hash=A6000[1])
    for want in ("84개", "101,376 B", "6 MB", "116.1 TFLOP/s",
                 "729.7 GB/s", "159.1 FLOP/byte", "눈금(1.024 us)"):
        assert want in txt, f"옛 파일의 {want!r} 를 재현 못 한다"


# ---------------------------------------------------------------------------
# ★ 되돌려서 잡는가 (원칙 38 — 일부러 틀리게 만들어 한 번 떨어뜨려 본다)
# ---------------------------------------------------------------------------


def test_a6000_prompt_on_a_5090_table_is_refused():
    from kernelrule.agents.hwprompt import HwPromptError, check_hw_prompt
    from kernelrule.agents.openai_client import load_prompt

    t = _table(*G5090)
    with pytest.raises(HwPromptError, match="5090"):
        check_hw_prompt(load_prompt("hw/sm_86.md"), t.hw,
                        float(t.noise.tick_ms))


def test_same_gpu_but_wrong_tick_is_refused():
    """★ 이름만 보면 **같은 GPU 의 다른 눈금**이 통과한다."""
    from kernelrule.agents.hwprompt import (
        HwPromptError,
        check_hw_prompt,
        hw_prompt_from_bundle,
    )

    t = _table(*A6000)
    txt, _ = hw_prompt_from_bundle(A6000[0], env_hash=A6000[1])
    check_hw_prompt(txt, t.hw, float(t.noise.tick_ms))       # 맞으면 통과
    with pytest.raises(HwPromptError, match="눈금"):
        check_hw_prompt(txt, t.hw, float(t.noise.tick_ms) * 4)


def test_tick_table_is_computed_not_hardcoded():
    """5090 의 눈금은 A6000 의 1/64 다 — 표가 그것을 반영해야 한다."""
    from kernelrule.agents.hwprompt import hw_prompt_from_bundle

    a, _ = hw_prompt_from_bundle(A6000[0], env_hash=A6000[1])
    g, _ = hw_prompt_from_bundle(G5090[0], env_hash=G5090[1])
    assert "7.314%" in a, "A6000 의 14us 행이 바뀌었다"
    assert "7.314%" not in g, (
        "5090 프롬프트가 A6000 의 눈금 비율을 말한다 — 상수로 박혀 있다")
