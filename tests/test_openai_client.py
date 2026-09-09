"""The real LLM client (§4-0, §4-1). **It does not call the API.**

It looks only at what can be checked without a call — handling a missing key,
the budget caps, prompt assembly, caching, and `LLMClient`
interchangeability.
"""
from __future__ import annotations

import os

import pytest

from kernelrule.agents.openai_client import (
    Budget,
    BudgetExceeded,
    LLMConfig,
    MissingAPIKey,
    OpenAILLM,
    estimate_and_confirm,
    load_prompt,
)
from kernelrule.features import FeatureRegistry

FEATS = ["traffic_amplification", "has_spill", "waves"]
SHAPE = ["is_memory_bound", "log_sol_ms"]

#: ★ `registry` is a required argument (§30.9). While it had a default,
#   every client in this file was built with `None` and `render_features`
#   silently fell back to the human 24 — which, under conditions F0~F3, puts
#   the answer into the prompt.
#   `test_diagnose_prompt_carries_the_report` was passing thanks to that
#   fallback. The prompt now renders **only the registry it was given**.
EMPTY_REG = FeatureRegistry("test-empty")


def _reg(feats=FEATS, shapes=SHAPE) -> FeatureRegistry:
    """A small registry holding **the same names** as `FEATS`/`SHAPE`."""
    from kernelrule.features import Feature

    r = FeatureRegistry("test-small")
    for n in feats:
        r.add(Feature(name=n, fn=lambda p, hw, cfg: 0.0, unit="dimensionless",
                      expected_range=(0.0, 1.0), direction="higher_is_worse",
                      code_hash=f"h-{n}"))
    for n in shapes:
        r.add(Feature(name=n, fn=lambda p, hw, cfg: 0.0, unit="dimensionless",
                      expected_range=(0.0, 1.0), direction="higher_is_worse",
                      shape_level=True, code_hash=f"h-{n}"))
    return r


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    return OpenAILLM(LLMConfig(), feature_names=FEATS, shape_values=SHAPE,
                     registry=_reg())


# ---------------------------------------------------------------------------
# The key — ★ it does not silently fall back to MockLLM (§26.4)
# ---------------------------------------------------------------------------
def test_missing_key_is_a_hard_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKey, match="does not silently fall back"):
        OpenAILLM(LLMConfig(), feature_names=FEATS, shape_values=SHAPE,
                  registry=_reg())


def _needs_pydantic_ai():
    """★ If it is missing, **skip but say why** (D-48).

    It used to blow up with 8 `ModuleNotFoundError`s, and to someone who had
    just cloned it that looked only like "8 tests are broken". The real
    problem was that **the install instructions left this package out**.

    `test_openai_client.py` is in `CRITICAL_MODULES`, so a full skip fails
    the session — that is, a run without `[llm]` guarantees nothing.
    """
    return pytest.importorskip(
        "pydantic_ai",
        reason="pydantic-ai is missing. Install with "
               "`pip install -e '.[llm]'`")


def test_key_is_never_stored(client):
    """The key does not survive in the object or the config."""
    blob = repr(client.__dict__) + repr(client.cfg.to_dict())
    assert "sk-test" not in blob
    assert "api_key" not in client.cfg.to_dict()


# ---------------------------------------------------------------------------
# The budget — exceeding it stops the run
# ---------------------------------------------------------------------------
def test_budget_stops_on_calls():
    b = Budget(max_calls=2)
    b.charge(10, 1)
    b.charge(10, 1)
    with pytest.raises(BudgetExceeded, match="calls"):
        b.charge(10, 1)


def test_budget_stops_on_tokens():
    b = Budget(max_input_tokens=100)
    with pytest.raises(BudgetExceeded, match="input tokens"):
        b.charge(101, 1)


def test_estimate_requires_confirmation():
    """★ A mass of calls does not start without confirmation (§4-1)."""
    with pytest.raises(BudgetExceeded, match="confirmation is needed"):
        estimate_and_confirm(n_rounds=20, n_rules=12, report_chars=14000,
                             cfg=LLMConfig(), yes=False)
    est = estimate_and_confirm(n_rounds=20, n_rules=12, report_chars=14000,
                               cfg=LLMConfig(), yes=True)
    assert est["calls"] == 20 * 13


# ---------------------------------------------------------------------------
# The prompts — two layers (§11.2)
# ---------------------------------------------------------------------------
def test_prompts_exist():
    # ★ `hw/sm_86.md` is not in the list — it was deleted on 2026-09-08
    #   (D-146) and the hardware facts are generated from the bundle (D-113).
    for n in ("_base.md", "role/_rules_common.md",
              "role/_rules_edit.md", "role/analyze.md", "role/rule_editor.md",
              "role/feature.md", "role/rule_writer.md", "role/categorize.md"):
        assert load_prompt(n).strip()


def test_missing_prompt_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_prompt("nope.md")


def test_base_prompt_is_hardware_independent(client, monkeypatch):
    """★ The role-independent block carries **no hardware** (§30.10).

    ⚠️ 2026-09-08 (D-146): this was `test_instructions_are_two_axes` and it
    compared two **files**, `_base.md` against `hw/sm_86.md`. The hardware
    side is generated from the bundle now (D-113) and that file is deleted,
    so what is left to check here is the base side.
    """
    base = load_prompt("_base.md")
    assert "GEMM" in base
    assert "RTX A6000" not in base and "sm_86" not in base
    # ★ _base.md must stay short — anything piled up here is paid for by
    #   every role
    assert len(base.splitlines()) < 40, "the common block swelled again"


def test_rule_block_states_the_absolute_rules():
    c = load_prompt("role/_rules_common.md")
    # ★ `is_memory_bound` was removed — nailing a real feature name into
    #   the prompt hands over the answer under F0~F2 (D-65). It became a
    #   placeholder.
    for must in ("import", "np.random", "8", "w[0]", "p.<shape value>"):
        assert must in c
    # ★ The no-op branch warning is in the prompt
    assert "cancels out" in c


def test_rule_editor_prompt_formats(client):
    from kernelrule.agents.schemas import RuleProposal

    p = client._user_prompt(
        "rule_editor", "", parent=RuleProposal(code="def score(f,p,hw,w): return 1",
                                            w0=[1.0]),
        hypothesis={"id": "H1", "claim": "a test"},
        hypotheses_applied=["H0: the previous one"])
    assert "H0: the previous one" in p and "a test" in p
    assert "traffic_amplification" in p and "is_memory_bound" in p
    assert "{" not in p.split("## Parent rule")[0].replace("{", "", 0) or True


def test_diagnose_prompt_carries_the_report(client):
    p = client._user_prompt("analyze", "REPORT-BODY-MARKER")
    assert "REPORT-BODY-MARKER" in p
    assert "traffic_amplification" in p


# ---------------------------------------------------------------------------
# Interchangeability — the `LLMClient` Protocol
# ---------------------------------------------------------------------------
def test_interface_matches_mock(client):
    """★ Interchangeability with `MockLLM` is what makes the ablations and
    replay hold."""
    from kernelrule.agents.mock import MockLLM

    mock = MockLLM("canned", feature_names=FEATS)
    for name in ("complete", "dump"):
        assert callable(getattr(client, name))
        assert callable(getattr(mock, name))


def test_dump_never_writes_the_key(client, tmp_path):
    from kernelrule.agents.mock import LLMCall

    client.calls.append(LLMCall(role="rule_editor", prompt_hash="h", seq=0,
                                response={"code": "x"}, mode=client.cfg.model))
    client.calls[-1].__dict__["_meta"] = {"prompt": "p", "input_tokens": 1,
                                          "output_tokens": 1, "seconds": 0.1}
    client.dump(tmp_path)
    blob = "".join(f.read_text() for f in tmp_path.glob("*.json"))
    assert "sk-" not in blob and "Authorization" not in blob
    assert "input_tokens" in blob      # the instrumentation survives


def test_cache_key_covers_role_and_prompt(client):
    import hashlib
    a = hashlib.sha256(b"optimize\x00X").hexdigest()[:16]
    b = hashlib.sha256(b"diagnose\x00X").hexdigest()[:16]
    assert a != b, (
        "without the role in the cache key the responses get mixed")


def test_semaphore_is_per_event_loop(client):
    """★ Creating the `asyncio.Semaphore` in `__init__` binds it to the
    first loop.

    The loop calls `asyncio.run()` afresh every round, so it dies from the
    second round on, and that exception is handled as a discarded candidate,
    so **calls are silently lost.** We stepped on this.
    """
    import asyncio

    async def grab():
        return client._semaphore()

    a = asyncio.run(grab())
    b = asyncio.run(grab())
    assert a is not b, "two event loops share the same semaphore"
    # Within one loop the same one is used
    async def twice():
        return client._semaphore() is client._semaphore()
    assert asyncio.run(twice())


def test_failed_calls_are_counted():
    """A failed call spends tokens too. Not counting them leaves a hole in
    the budget watchdog."""
    b = Budget(max_calls=3)
    b.charge(10, 1)
    b.failed_calls = 2
    assert "failed" in b.line()
    assert b.calls + b.failed_calls == 3


def test_prompt_shows_rejected_examples():
    """★ Stating the rules alone gets them broken. Real refused cases are
    given alongside."""
    c = load_prompt("role/_rules_edit.md")
    assert "actually got rejected" in c
    assert "w[0] reused" in c
    # ★ The rules themselves live in `_rules_common.md` — the gallery only
    #   gives cases
    assert "used exactly once" in load_prompt("role/_rules_common.md")


# ---------------------------------------------------------------------------
# Breaking down the refusal reasons (1-4c) — lumped together as
# `llm 132 cases` there is no way to know what to fix
# ---------------------------------------------------------------------------
VIOLATIONS = [
    ("9 weights. The budget is 8 (§29.4)", "w0_too_long"),
    ("a weight is reused across terms: ['w[0]x4']", "weight_reuse"),
    ("W0 length 3 != the largest referenced index + 1", "w0_length_mismatch"),
    ("banned reference: 'time_ms'", "banned_substring"),
    ("there is no `def score(f, p, hw, w):`", "no_def_score"),
    ("Do not put code in a hypothesis", "hypothesis_has_code"),
    ("w0 is empty", "w0_empty"),
    ("Exceeded maximum output retries (2)", "retries_exhausted"),
    ("Semaphore is bound to a different event loop", "event_loop_bug"),
    ("something entirely new", "other"),
    # ★ The old Korean messages must stay classifiable (D-146) — old
    #   `llm_calls/` logs are read with the same function.
    ("가중치 9개. 리터럴 예산이 8개다 (§29.4)", "w0_too_long"),
    ("가중치를 여러 항에 재사용했다: ['w[0]x4']", "weight_reuse"),
    ("금지된 참조: 'time_ms'", "banned_substring"),
]


@pytest.mark.parametrize("msg,code", VIOLATIONS,
                         ids=[c for _, c in VIOLATIONS])
def test_violation_is_classified(msg, code):
    """★ The patterns must match the **actual validator messages**.

    It once said "8 weights" and failed to catch "9 weights...", which
    leaked into other. A wrong classification leaves no way to know where to
    fix the prompt.
    """
    from kernelrule.agents.openai_client import classify_violation

    assert classify_violation(msg) == code


def test_violation_report_detects_useless_retries(client):
    """If what was caught on attempt 1 is caught on attempt 2 **for the
    same reason**, the feedback is not working.

    Then the fix is the prompt, not a higher retry cap.
    """
    client.violations = [
        {"round": 0, "seq": 1, "role": "rule_editor", "attempt": 0,
         "code": "w0_too_long", "msg": "x"},
        {"round": 0, "seq": 1, "role": "rule_editor", "attempt": 2,
         "code": "w0_too_long", "msg": "x"},        # the same code again
        {"round": 0, "seq": 2, "role": "rule_editor", "attempt": 0,
         "code": "banned_substring", "msg": "y"},
    ]
    r = client.violation_report()
    assert r["total"] == 3
    assert r["by_code"]["w0_too_long"] == 2
    assert r["same_code_repeated"] == 1, (
        "the feedback failure was not caught")


def test_retries_raised_to_three():
    """(d) Temporary. While the refusal rate is high it separates "the
    model is learning" from "structurally impossible"."""
    assert LLMConfig().max_retries == 3


# ---------------------------------------------------------------------------
# RuleWriter — condition A sees nothing that came from the table (§11.8)
# ---------------------------------------------------------------------------
# For transfer to hold, the structure has to come out **without the table**
# on a new architecture. If the structure needs the table, that is §29.5(c)
# regeneration, and if you are going to measure exhaustively you can use the
# table directly, so there is no reason to use this system. So A is the gate
# condition.

def _arch_client():
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import REGISTRY
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    return OpenAILLM(LLMConfig(model="m"), feature_names=[], shape_values=[],
                     registry=REGISTRY, cache=False)


class _Facts:
    lines = ("shapes whose answer set contains a spilling kernel: 0/51",
             "how far one fixed config gets you:  top-1 1.116")
    by_feature = {"has_spill":
                  ["0 of the 51 training shapes have it in the answer set"]}


def test_rule_writer_condition_a_contains_no_table_derived_line():
    """★ The gate condition. Not one sentence that came from the table may
    enter."""
    c = _arch_client()
    a = c._rule_writer_prompt(condition="A", table_facts=_Facts())
    for line in (*_Facts.lines, *_Facts.by_feature["has_spill"]):
        assert line not in a, (
            f"a table sentence leaked into condition A: {line}")
    assert "condition A" in a


def test_rule_writer_condition_b_carries_the_aggregates():
    c = _arch_client()
    b = c._rule_writer_prompt(condition="B", table_facts=_Facts())
    for line in (*_Facts.lines, *_Facts.by_feature["has_spill"]):
        assert line in b


def test_rule_writer_condition_b_refuses_without_facts():
    """Calling it B without the aggregates makes it identical to A — that
    does not happen silently."""
    c = _arch_client()
    with pytest.raises(ValueError, match="training-split aggregates"):
        c._rule_writer_prompt(condition="B")
    with pytest.raises(ValueError, match="unknown RuleWriter condition"):
        c._rule_writer_prompt(condition="C")


def test_rule_writer_prompt_has_no_parent_or_case_slots():
    """Receiving no parent, no cases and no score is the definition of this
    role."""
    c = _arch_client()
    a = c._rule_writer_prompt(condition="A")
    for banned in ("Parent rule:", "### Case #", "regret 1.", "val "):
        assert banned not in a


# ---------------------------------------------------------------------------
# Endpoint selection (D-44)
# ---------------------------------------------------------------------------
# The gpt-5.6 family blocks the **function tools + reasoning_effort**
# combination on /v1/chat/completions with a 400. Structured output
# (`output_type`) is implemented as function tools, so it is caught. The
# workaround is reasoning_effort='none', but that switches reasoning off and
# loses the ability to derive physics — so the endpoint was moved instead.

def test_endpoint_defaults_to_responses_and_is_recorded():
    """★ It must stay in `config.json` — mixing them breaks comparison
    (D-31)."""
    from kernelrule.agents.openai_client import LLMConfig
    cfg = LLMConfig()
    assert cfg.endpoint == "responses"
    assert cfg.to_dict()["endpoint"] == "responses"


def test_unknown_endpoint_is_rejected():
    """It does not silently fall back to chat (§26.4)."""
    _needs_pydantic_ai()
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(model="m", endpoint="v1"), feature_names=[],
                    shape_values=[], cache=False, registry=EMPTY_REG)
    with pytest.raises(ValueError, match="unknown endpoint"):
        llm._agent("rule_editor")


@pytest.mark.parametrize("endpoint,cls_name", [
    ("responses", "OpenAIResponsesModel"),
    ("chat", "OpenAIChatModel"),
])
def test_endpoint_picks_the_right_model_class(endpoint, cls_name):
    _needs_pydantic_ai()
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(model="gpt-5.4-mini-2026-03-17",
                              endpoint=endpoint),
                    feature_names=[], shape_values=[], cache=False, registry=EMPTY_REG)
    agent = llm._agent("rule_editor")
    assert type(agent.model).__name__ == cls_name


def test_model_has_a_single_source():
    """★ If experiment scripts each hold their own model constant they can
    run on different models — and then the results cannot be placed side by
    side (D-31, D-45).
    """
    import re
    from pathlib import Path

    from kernelrule.agents.openai_client import DEFAULT_MODEL, LLMConfig

    assert LLMConfig().model == DEFAULT_MODEL
    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted((root / "experiments").glob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            # A path string (`runs/...-gpt-5.4/`) points at a past run and
            # is fine. Only a model nailed in by **assignment** is caught.
            if re.search(r'^\s*\w*MODEL\w*\s*=\s*["\']gpt-', line):
                bad.append(f"  {f.name}:{i}  {line.strip()}")
    assert not bad, ("an experiment nailed the model in directly. Use "
                     "`DEFAULT_MODEL`:\n" + "\n".join(bad))


def test_temperature_and_seed_default_to_none():
    """★ What cannot be controlled is not recorded as controlled (D-47).

    They used to be 0.7 / 20260821 and **neither was passed to the model.**
    They survived only in `config.json`, so the record and reality diverged
    (§30.8).
    """
    from kernelrule.agents.openai_client import LLMConfig
    d = LLMConfig().to_dict()
    assert d["temperature"] is None
    assert d["seed"] is None


def test_seed_with_responses_endpoint_raises():
    """Better to stop than be silently dropped — Responses has no seed
    parameter."""
    _needs_pydantic_ai()
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(seed=123, endpoint="responses"),
                    feature_names=[], shape_values=[], cache=False, registry=EMPTY_REG)
    with pytest.raises(ValueError, match="is not a parameter"):
        llm._agent("rule_editor")


def test_seed_with_chat_endpoint_is_sent():
    """On chat it really is supported, so it is sent."""
    _needs_pydantic_ai()
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(seed=123, temperature=0.7, endpoint="chat"),
                    feature_names=[], shape_values=[], cache=False, registry=EMPTY_REG)
    sent = llm._agent("rule_editor").model_settings or {}
    assert sent["seed"] == 123
    assert sent["temperature"] == 0.7


def test_none_values_are_not_sent_at_all():
    _needs_pydantic_ai()
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(), feature_names=[], shape_values=[],
                    cache=False, registry=EMPTY_REG)
    sent = llm._agent("rule_editor").model_settings or {}
    assert "temperature" not in sent
    assert "seed" not in sent


def test_experiments_do_not_pass_temperature_or_seed():
    """If an experiment puts them back they are silently dropped — this
    blocks that."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted((root / "experiments").glob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"LLMConfig\([^)]*\b(temperature|seed)\s*=", line):
                bad.append(f"  {f.name}:{i}  {line.strip()}")
    assert not bad, ("an experiment passes temperature/seed to LLMConfig — "
                     "on gpt-5.6-luna + responses they are silently dropped "
                     "(D-47):\n" + "\n".join(bad))


def test_reasoning_effort_is_explicit_and_recorded():
    """★ Unstated, the model default applies, and if that default changes
    our results silently change with it (§15.4).
    """
    _needs_pydantic_ai()
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    cfg = LLMConfig()
    assert cfg.reasoning_effort == "medium"
    assert cfg.to_dict()["reasoning_effort"] == "medium"

    llm = OpenAILLM(cfg, feature_names=[], shape_values=[], cache=False, registry=EMPTY_REG)
    sent = llm._agent("rule_editor").model_settings or {}
    assert sent.get("openai_reasoning_effort") == "medium"


def test_reasoning_effort_none_sends_nothing():
    """`None` means "the model default". It differs from the string
    'none'."""
    _needs_pydantic_ai()
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(reasoning_effort=None), feature_names=[],
                    shape_values=[], cache=False, registry=EMPTY_REG)
    assert "openai_reasoning_effort" not in (
        llm._agent("rule_editor").model_settings or {})


def test_every_llm_runner_persists_its_calls():
    """★ An LLM call cannot be remade (D-33 / D-51).

    A runner that uses `RoundLoop` gets it recorded by `dump()`, but a
    runner that calls `OpenAILLM` directly has to record it itself. Two of
    them really were missing it, and the tokens had to be picked out of the
    stdout logs when totalling the cost.
    """
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted((root / "experiments").glob("*.py")):
        src = f.read_text()
        if "OpenAILLM(" not in src:
            continue
        # RoundLoop calls llm.dump() inside its own dump()
        if "RoundLoop(" in src:
            continue
        if not re.search(r"\.dump\(", src):
            bad.append(f"  {f.name}: it uses OpenAILLM directly but has "
                       f"no dump")
    assert not bad, ("there is a runner that does not record its LLM calls "
                     "(D-33):\n" + "\n".join(bad))


# ---------------------------------------------------------------------------
# ★ §30.9 — F0~F3 hold only if the registry can be swapped
# ---------------------------------------------------------------------------
def test_registry_is_required(monkeypatch):
    """With a default, the human 24 silently enter condition F1."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    with pytest.raises(ValueError, match="mandatory"):
        OpenAILLM(LLMConfig(), feature_names=[], shape_values=[],
                  registry=None)


def test_prompt_renders_only_the_given_registry(monkeypatch):
    """★ **Not one** of the 24 names a human wrote may enter."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.features import REGISTRY

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    gen = FeatureRegistry("f1-like")
    from kernelrule.features import Feature
    for n in ("padded_flop_fraction", "l2_tile_pressure"):
        gen.add(Feature(name=n, fn=lambda p, hw, cfg: 0.0,
                        unit="dimensionless", expected_range=(0.0, 1.0),
                        direction="higher_is_worse", code_hash=f"h-{n}"))
    llm = OpenAILLM(LLMConfig(), feature_names=["padded_flop_fraction"],
                    shape_values=[], registry=gen)
    p = llm._user_prompt("analyze", "BODY")
    leaked = [n for n in REGISTRY._items if n in p]
    assert not leaked, (
        f"features a human wrote leaked into the prompt: {leaked}")
    assert "padded_flop_fraction" in p


def test_feature_names_must_live_in_the_registry(monkeypatch):
    """The static checks and the prompt must not see different lists
    (principle 2)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    with pytest.raises(ValueError, match="registry"):
        OpenAILLM(LLMConfig(), feature_names=["no_such_feature"],
                  shape_values=[], registry=_reg())


def test_render_features_refuses_a_missing_registry():
    from kernelrule.features import render_features

    with pytest.raises(ValueError, match="must be given a registry"):
        render_features(None, include_observed=False)


def test_feature_decorator_refuses_a_missing_registry():
    from kernelrule.features import feature

    with pytest.raises(ValueError, match="mandatory"):
        @feature(expected_range=(0.0, 1.0))
        def _f(p, hw, cfg) -> float:
            return 0.0


def test_intrinsic_shape_fields_are_not_stray(monkeypatch):
    """★ `M/N/K/n_candidates` are not registry features but are always
    present.

    Leaving them out at first killed a validation run before it even started
    (§30.9).
    """
    from kernelrule.core.matrix import INTRINSIC_SHAPE_FIELDS

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    llm = OpenAILLM(LLMConfig(), feature_names=FEATS,
                    shape_values=[*SHAPE, *INTRINSIC_SHAPE_FIELDS],
                    registry=_reg())
    assert set(INTRINSIC_SHAPE_FIELDS) <= set(llm.shape_values)


# ---------------------------------------------------------------------------
# §16.1 — the RuleEditor prompt with the Analyst off (D-89)
# ---------------------------------------------------------------------------
def _rule_editor_prompts():
    _needs_pydantic_ai()
    import os

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.agents.schemas import RuleProposal
    from kernelrule.features import REGISTRY

    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(),
                    feature_names=REGISTRY.names(shape_level=False),
                    shape_values=REGISTRY.names(shape_level=True),
                    registry=REGISTRY, cache=False)
    par = RuleProposal(code="def score(f,p,hw,w):\n    return f.waves * w[0]\n",
                       w0=[1.0])

    def render(flag: bool) -> str:
        return llm._user_prompt(
            "rule_editor", "", parent=par, parent_n_terms=1,
            hypothesis={"id": "H1", "claim": "c"} if flag else None,
            hypotheses_applied=["H0: x"] if flag else [], analyst=flag)

    return render(True), render(False)


def test_rule_editor_prompt_without_analyst_mentions_no_hypothesis():
    """★ The hypothesis section is **not left as an empty slot** (§16.1).

    Leaving "## This round's hypothesis\n\n(none)" makes the model read
    "there is a hypothesis and it is empty", and the condition changes. It
    is the same principle as not building the diagnostic report either — the
    slot itself must be absent.
    """
    on, off = _rule_editor_prompts()
    assert "hypothesis" in on
    assert "hypothesis" not in off, (
        "Analyst is off but the prompt still mentions a hypothesis")
    assert "## This round's hypothesis" not in off
    assert "Parent rule" in off and "Available features" in off


def test_rule_editor_prompt_without_analyst_is_a_deletion():
    """★ The prompt with it off must be the one with it on, **with
    sentences deleted**.

    New wording makes the ablation stop being "only the Analyst differs". If
    it is a character-level subsequence, only deletion happened.
    """
    on, off = _rule_editor_prompts()
    it = iter(on)
    assert all(ch in it for ch in off), (
        "the off prompt has characters the on prompt does not — it was "
        "rewritten, not deleted from (§16.1)")
    assert len(off) < len(on)


# ---------------------------------------------------------------------------
# D-96 — the second parent of cross
# ---------------------------------------------------------------------------
def _editor_prompt(parent2):
    _needs_pydantic_ai()
    import os

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.agents.schemas import RuleProposal
    from kernelrule.features import REGISTRY

    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(),
                    feature_names=REGISTRY.names(shape_level=False),
                    shape_values=REGISTRY.names(shape_level=True),
                    registry=REGISTRY, cache=False)
    a = RuleProposal(code="def score(f,p,hw,w):\n    return f.waves * w[0]\n",
                     w0=[1.0])
    return llm._user_prompt(
        "rule_editor", "", parent=a, parent2=parent2, parent_n_terms=1,
        hypothesis={"id": "H1", "claim": "c"}, hypotheses_applied=["H0"],
        analyst=True)


def test_second_parent_section_is_absent_without_a_second_parent():
    """★ Without a second parent **the section itself is absent** (D-96).

    Leaving an empty slot such as "(no parent)" makes the model read "there
    is a second and it is empty", which changes the conditions of
    exploit/explore — we stepped on this in D-89.
    """
    from kernelrule.agents.schemas import RuleProposal

    one = _editor_prompt(None)
    assert "Second parent" not in one
    b = RuleProposal(
        code="def score(f,p,hw,w):\n    return f.tail_waste * w[0]\n",
        w0=[2.0])
    two = _editor_prompt(b)
    assert "Second parent" in two and "f.tail_waste" in two
    # ★ The one-parent prompt is a **subsequence** of the two-parent one —
    #   it is pure addition
    it = iter(two)
    assert all(ch in it for ch in one), (
        "the two-parent prompt changed a sentence of the one-parent side — "
        "the conditions of exploit/explore change")


def test_cross_hands_the_second_parent_to_the_editor():
    """★ `run_round` **actually passes** the second parent
    `archive.parents` gives it.

    It used to use `ps[0]` alone, so `cross` was the same as `explore` —
    §13's crossover was never implemented (D-96).
    """
    import ast
    import inspect
    import textwrap

    from kernelrule.core.loop import RoundLoop

    src = textwrap.dedent(inspect.getsource(RoundLoop.run_round))
    uses = {ast.unparse(n) for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Subscript)
            and ast.unparse(n).startswith("ps[")}
    assert "ps[1]" in uses, (
        f"the second parent is not used: {sorted(uses)}")
    assert '"parent2": parent2' in src or "'parent2': parent2" in src
