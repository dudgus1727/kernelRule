"""The real LLM client — Pydantic AI + OpenAI (§4-0).

## It sits behind the `LLMClient` Protocol

It must be **interchangeable** with `MockLLM` for the ablations and `replay`
to hold. `pydantic_ai.Agent` is not exposed to the call sites — the loop
knows only `complete(role, prompt, **kw)`.

## A schema violation is a retry, then a **discard**

Pydantic AI feeds a validator failure back to the model and retries. Beyond
the cap (`retries`) the candidate is thrown away. **There is no partial
acceptance** (§26.4) — patching up a half-right rule makes it impossible to
say what that rule tested.

## The key

It is read only from the `OPENAI_API_KEY` environment variable. **It is not
put in the code and not stored.** Without it, it stops with a clear error —
it does not silently fall back to `MockLLM` (§26.4).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from kernelrule.agents.mock import LLMCall

__all__ = ["OpenAILLM", "LLMConfig", "Budget", "BudgetExceeded",
           "MissingAPIKey", "load_prompt", "classify_violation",
           "DEFAULT_MODEL"]

_PROMPTS = Path(__file__).parent / "prompts"

#: ★ The **single source** for the model. Experiment scripts used to hold
#: their own constants and ran on different models — and then the results
#: cannot be placed side by side (D-31). To change it, change this one place,
#: and **only when the user says so**.
DEFAULT_MODEL = "gpt-5.6-luna"


class MissingAPIKey(RuntimeError):
    """There is no API key. **It does not fall back to `MockLLM`**
    (§26.4)."""


class BudgetExceeded(RuntimeError):
    """The budget cap was exceeded. The run stops."""


#: validator message -> a short reason code. Lumped together as
#: `llm 132 cases`, there is no way to know what was caught, nor where to fix
#: the prompt.
#:
#: ⚠️ 2026-09-08 (D-146): the validator messages became English. **The old
#:    Korean patterns are kept** — old `llm_calls/` logs have to stay
#:    classifiable (the correction-history rule).
_VIOLATION_PATTERNS: tuple[tuple[str, str], ...] = (
    # ⚠️ The patterns must match the **actual validator messages**. It once
    #    said "8 weights" and failed to catch "9 weights...", which leaked
    #    into other.
    ("the budget is", "w0_too_long"),
    ("numeric literals +", "w0_too_long"),
    ("리터럴 예산이", "w0_too_long"),
    ("최대 8개", "w0_too_long"),
    ("reused across terms", "weight_reuse"),
    ("재사용", "weight_reuse"),
    ("largest referenced index", "w0_length_mismatch"),
    ("최대 인덱스", "w0_length_mismatch"),
    ("w0 is empty", "w0_empty"),
    ("w0 가 비었다", "w0_empty"),
    ("abnormally large", "w0_huge"),
    ("비정상적으로 크다", "w0_huge"),
    ("banned reference", "banned_substring"),
    ("금지된 참조", "banned_substring"),
    ("def score", "no_def_score"),
    ("code in a hypothesis", "hypothesis_has_code"),
    ("가설에 코드", "hypothesis_has_code"),
    ("hypotheses", "hypothesis_count"),
    ("가설이", "hypothesis_count"),
    ("Exceeded maximum", "retries_exhausted"),
    ("event loop", "event_loop_bug"),
    ("rate limit", "rate_limit"),
)


def classify_violation(msg: str) -> str:
    """Classifies an exception or validator message into a reason code."""
    for pat, code in _VIOLATION_PATTERNS:
        if pat.lower() in msg.lower():
            return code
    return "other"


#: The three feature conditions (D-128). `F2` starts from the **five
#: public facts** (§30.17). The old names `F0` (no features) and the old
#: `F2` (5 raw values) had 0 runs and were deleted, and the old
#: `F1-K` is today's `F2` (D-128). **There are no aliases.**
_CONDITIONS = frozenset({"F1", "F2", "F3"})

#: Condition -> the **feature** example file. A condition that must not be
#: handed the answer gets an unrelated domain.
#: ★ The condition selects the prompt **example** — moving it wrongly gives
#: the model a different example, and that is a change of condition (the
#: spot most carefully watched during the D-128 rename).
_EXAMPLES = {"F1": "other_domain", "F2": "known5", "F3": "known5"}

#: The features `examples/rule_known.md` **calls by name**.
#: All four must be in the registry for that example to be usable (§30.20).
_RULE_EXAMPLE_NEEDS = ("tail_waste", "has_spill", "occupancy_deficit",
                       "roofline_ratio")


def _rule_example_for(registry, *, parameters: int | None = None) -> str:
    """Selects the rule example **by looking at the registry** (§30.20).

    RuleWriter's `condition` is A/B (with or without table observations), a
    different axis from the feature conditions. So it is decided not by the
    condition but by **whether the names the example uses are in the
    registry**.

    ```
    all present   using the real names is no extra leak — they are already
                  in the list
    absent        ★ an unrelated domain. Giving names that do not exist as
                  an example points at the physics (D-35)
    ```

    Keying on the condition name means the table has to be edited whenever a
    condition is added, and a missed entry leaks silently. **Looking at the
    registry cannot be missed.**
    """
    names = set(getattr(registry, "_items", {}) or {})
    ok = names.issuperset(_RULE_EXAMPLE_NEEDS)
    return load_prompt(
        f"examples/{'rule_known' if ok else 'rule_other_domain'}.md",
        parameters=parameters)

#: The roles that receive the hardware facts (`hw/*.md`). **RuleWriter
#: only.**
#:
#:   RuleEditor / FeatureWriter   the features have already absorbed hw.
#:                                Not seeing it makes that prompt
#:                                GPU-independent (§16.2)
#:   Analyst                      ★ **block 1 of the diagnostic report is
#:                                the same facts.** The report is generated
#:                                from the table every time and `hw/*.md` is
#:                                fixed, so when the bundle changes the two
#:                                diverge and it receives contradictory
#:                                facts. The live one is kept (principle 2)
#:   RuleWriter                   it receives no report, so it must get them
#:                                here
#: ★ The goal definition (D-101). **RuleEditor alone** receives it —
#: RuleWriter has "no score", so it never sees this section (§30.10). That
#: is why changing the objective does not change RuleWriter's condition.
_OBJECTIVE_BLOCKS = {
    "regret": """`regret` = (time of the config the rule ranked first) /
(the exhaustively measured best time for that shape).
1.0 is perfect. Lower is better.""",
    #: ★ `rank` is a function — `k` and `lambda` differ per run. With 100
    #: nailed into the sentence as a constant, giving `--rank-top-k 10`
    #: still made the prompt say "100" (the same spot as D-105/D-107, the
    #: fifth surface).
    "rank": None,
}


def _rank_block(k: int = 100, lam: float = 0.0) -> str:
    extra = "" if not lam else (
        f"\n\n★ On top of that, **getting the true first place right** "
        f"carries an extra weight of {lam:g}.\nLosses on pairs that include "
        "the winner count that much more — keep the top order\n**and do not "
        "miss first place.**")
    return f"""★ You are scored on **the order of the top ranks**, not on
first place alone.

For each shape we take the {k} genuinely fastest configs and check, for every
pair (i, j) among them, whether you gave **the faster one the lower score**.
Each pair is weighted by the difference between the two times — inverting a
pair with a large gap costs a lot.

0 is perfect. Lower is better.

⚠️ Pairs that measurement noise cannot separate are excluded from scoring.
**Do not chase tiny differences; use the physics that creates the order.**{extra}"""


#: ★ Experiment B (D-110) — it states that **two features may be
#: multiplied within one term**. The static checks never blocked products
#: (312 of 531 rules already use them). So this is not "loosening a
#: constraint" but **saying so** — as learned in D-107, what is not said
#: does not get tried.
#:
#: ⚠️ Switched off it is the **empty string**, so the prompt is
#: byte-identical to before.
_PRODUCT_BLOCK = """

## ★ Term shape — you may multiply features

Within one term you **may multiply two features.** It is still one weight.

```python
s = s + (f.<nameA> * f.<nameB>) * w[3]           # ✅ one parameter
s = s + f.<nameA> * np.log2(f.<nameB>) * w[4]    # ✅
```

If you want to penalise only when two quantities are bad **at the same
time**, a product is right — a sum penalises when either one is bad. Do not
worry about it being harder to interpret. **Use the shape that improves
performance; why it works can be looked at later.**"""

_PRODUCT_NOTE = """
★ A term multiplying two features is allowed — it costs one parameter.
"""


#: * Experiment (b) (D-112) — a weight may sit **in the exponent slot**.
#: This is the only form not yet touched: what has been widened so far is
#: the term count / the node count / products, and **that weights sit only
#: in linear positions** stayed as it was.
#:
#: Switched off it is the **empty string**, so the prompt is byte-identical
#: to before.
_POWER_BLOCK = """

## ★ Term shape — a weight may sit in the exponent

So far weights have only **multiplied** a term. **They can also be
exponents.**

```python
s = s + f.<name> * w[3]                     # the usual form (one parameter)
s = s + np.power(f.<name>, w[3])            # ✅ exponent only (**one** parameter)
s = s + np.power(f.<name>, w[3]) * w[4]     # exponent + scale (two parameters)
```

★ **The middle one replaces an existing term as is** — it costs no extra
room. If you are at the cap, do not drop a term; **turn `f.a * w[i]` into
`np.power(f.a, w[i])`.**

What changes: `f.a * w[3]` only sets "how hard to penalise".
`np.power(f.a, w[3])` sets **"how fast it gets worse"** — above 1 it
accelerates as the value grows, below 1 it flattens.

There are two constraints. They exist to **keep the numerics from breaking**,
not to remove expressiveness.

```
1. The base must be a single `f.<name>`
   An expression as the base can go negative, and a real power of a negative
   number is nan
2. The exponent weight is bounded to 0~4
   Below 0 it is inf at f == 0, and the direction flips
   Faster than 4 is not smooth physics but a threshold, and thresholds are
   what np.where is for
```

**Think about which axis "degrades slowly" and which "degrades suddenly",
then write it.**"""

_POWER_NOTE = """
★ You **may replace** `f.<name> * w[i]` with `np.power(f.<name>, w[i])`
   — it costs no extra room, and the optimiser fits the exponent (0~4).
"""


def power_block(on: bool) -> str:
    return _POWER_BLOCK if on else ""


def power_note(on: bool) -> str:
    return _POWER_NOTE if on else ""


#: ★ The hypothesis fields sent to the RuleEditor — an **allow list**
#: (D-117).
#:
#: It was first a deny list (drop `evidence_cases` only). Then every field
#: added to the Analyst schema **leaks automatically.** And the shape-size
#: validator is attached to `claim` alone, so `risk` / `affected_regime` do
#: not go through it.
#:
#: The rest (`evidence_cases`, `risk`, `affected_regime`, `id`) are for the
#: Analyst record and stay in `hypotheses.jsonl` as they are — they are not
#: discarded.
_EDITOR_KEEPS = ("claim", "measurable_with", "proposed_direction")


def _for_editor(hyp: dict) -> dict:
    return {k: hyp[k] for k in _EDITOR_KEEPS if k in hyp}


def product_block(on: bool) -> str:
    return _PRODUCT_BLOCK if on else ""


def product_note(on: bool) -> str:
    return _PRODUCT_NOTE if on else ""


def objective_block(objective: str, *, rank_top_k: int = 100,
                    rank_lambda: float = 0.0) -> str:
    """★ The goal definition, in one place (principle 2). `k` and `lambda`
    are **put into the sentence.**"""
    if objective not in _OBJECTIVE_BLOCKS:
        raise ValueError(f"unknown objective: {objective!r}")
    if objective == "rank":
        return _rank_block(rank_top_k, rank_lambda)
    return _OBJECTIVE_BLOCKS[objective]
def assemble_instructions(role: str, *, objective: str = "rank",
                          hw_file: str | None = None,
                          hw_text: str | None = None,
                          body: str | None = None,
                          parameters: int | None = None,
                          rank_top_k: int = 100,
                          rank_lambda: float = 0.0,
                          product_hint: bool = False,
                          power_hint: bool = False) -> str:
    """★ Assembling the system prompt. **Done in one place only**
    (principle 2).

    `_agent()` and `tests/test_prompt_layout.py` used to assemble it
    separately. Adding `{objective_block}` made them diverge because **only
    the test side left it unfilled** — the eighth instance of "the same
    judgement in several places diverges".
    """
    if objective not in _OBJECTIVE_BLOCKS:
        raise ValueError(f"unknown objective: {objective!r}")
    parts = [load_prompt("_base.md", parameters=parameters)]
    if role in _NEEDS_HW:
        # ★ There is no silent default (D-113). Missing means failure
        #   (§26.4).
        if hw_text is None and hw_file is None:
            raise ValueError(
                f"role {role!r} must be given the hardware facts, but "
                "neither `hw_text` nor `hw_file` is set. Generate it from "
                "the bundle and pass it in "
                "(`kernelrule.agents.hwprompt.hw_prompt_from_bundle`). "
                "Falling back to a default sends another GPU's facts "
                "(D-113).")
        parts.append(hw_text if hw_text is not None
                     else load_prompt(hw_file, parameters=parameters))
    if role in _WRITES_RULES:
        parts.append(load_prompt("role/_rules_common.md", parameters=parameters)
                     .replace("{product_block}", product_block(product_hint))
                     .replace("{power_block}", power_block(power_hint)))
    if role in _EDITS_RULES:
        parts.append(load_prompt("role/_rules_edit.md", parameters=parameters).replace(
            "{objective_block}", objective_block(
                objective, rank_top_k=rank_top_k, rank_lambda=rank_lambda)))
    parts.append(body if body is not None
                 else load_prompt(f"role/{role}.md", parameters=parameters)
                 .replace("{product_note}", product_note(product_hint))
                 .replace("{power_note}", power_note(power_hint)))
    return "\n\n---\n\n".join(parts)


_NEEDS_HW = frozenset({"rule_writer"})

#: The roles that write the rule **function** — they share the shape,
#: budget and vectorisation constraints.
_WRITES_RULES = frozenset({"rule_editor", "rule_writer"})

#: The role that **edits** a parent rule. It receives the regret definition
#: and the gallery of refused cases.
#: ★ RuleWriter does not — it writes from a blank page, so there is no term
#: to replace and no previous score, and giving it would contradict its own
#: role file's "no score" head on.
_EDITS_RULES = frozenset({"rule_editor"})


#: **Internal notes** in the prompt files. They are written for humans and
#: are not sent to the model — internal references such as `§30.18` and
#: `D-45` were going out as they were.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def load_prompt(name: str, *, parameters: int | None = None) -> str:
    """Reads a prompt. **It strips HTML comments and fills in
    `{parameters}`.**

    `<!-- ... -->` is where "why it was written this way" is recorded. If
    that goes to the model it (1) costs tokens, (2) leaks internal decision
    numbers and (3) under some conditions could hand over the answer.

    ★ The parameter count is **not written into the prompt directly.**
    Written as `{parameters}`, it is filled in here from
    `checks.PARAMETERS`. Five prompt files, the schema and the checker each
    wrote their own number, and then a change misses one (it would be the
    sixth after `is_reference` / `top_k` / `DEFAULT_MODEL` / `REGISTRY` /
    `load_generated`).
    """
    from kernelrule.rules.checks import PARAMETERS, limits_for

    p = _PROMPTS / name
    if not p.exists():
        raise FileNotFoundError(f"no such prompt: {p}")
    txt = _HTML_COMMENT.sub("", p.read_text()).strip() + "\n"
    # ★ The caps attached to the parameter count are filled in **too**
    #   (D-106). With `ast_nodes` written as a constant, at 16 parameters
    #   the prompt says "the cap is 400" while the checker uses 800 — the
    #   prompt and the checker diverge.
    lim = limits_for(parameters if parameters is not None else PARAMETERS)
    for k, v in lim.items():
        txt = txt.replace("{" + k + "}", str(v))
    return txt


@dataclass
class LLMConfig:
    """Recorded into `config.json` as is (§15.4 reproducibility)."""

    model: str = DEFAULT_MODEL
    #: ★ It sets the goal definition (D-101). It has to stay in
    #: `config.json` for the condition to be recorded. Only the RuleEditor's
    #: "how it is scored" section changes — RuleWriter does not receive it.
    objective: str = "regret"
    #: ★ The parameter cap (D-104). With `None` it is
    #: `checks.PARAMETERS` (8). The prompt's `{parameters}` is filled from
    #: this value — it must not diverge from the checker, so the loop passes
    #: the same value to `check_rule(limits=...)` too.
    parameters: int | None = None
    #: ★ The **numbers** of the goal definition. The prompt must say the
    #: same thing as the run conditions (D-105/D-107). Used only under the
    #: `rank` objective.
    rank_top_k: int = 100
    rank_lambda: float = 0.0
    #: ★ Experiment B (D-110). It **states** the product term. The checker
    #: never blocked it.
    product_hint: bool = False
    #: * Experiment (b) (D-112). It states that a weight may sit in the
    #: exponent slot.
    power_hint: bool = False
    # ------------------------------------------------------------------
    # ★ temperature / seed — both are `None`. They cannot be controlled
    # (D-47)
    # ------------------------------------------------------------------
    # They used to be 0.7 / 20260821 with "diversity is controlled and
    # reproducibility secured" written next to them. **Neither was being
    # passed to the model.** The values were written into `config.json`
    # while it actually ran on the model defaults (§30.8).
    #
    # Measurement separated the layers (3 models x 2 endpoints):
    #
    #   seed         **the parameter does not exist** on the Responses
    #                endpoint. The SDK raises a `TypeError` — the request
    #                never goes out. It is independent of the model
    #                (gpt-4.1-mini too).
    #   temperature  **reasoning models refuse it.** gpt-5.6-luna gives 400
    #                on both endpoints, while gpt-5.4-mini / gpt-4.1-mini
    #                accept it on both. It is independent of the endpoint.
    #                (A reasoning model goes through several internal rounds
    #                 of reasoning, verification and selection, so it blocks
    #                 sampling. It offers reasoning_effort instead.)
    #
    # pydantic-ai **silently drops** both — which is why nobody knew. It hid
    # the symptom, not the cause.
    #
    # So the defaults are `None`. A value given is sent, but **an
    # unsendable combination raises** — stopping is better than being
    # silently dropped (§26.4).
    temperature: float | None = None
    seed: int | None = None
    #: The retry cap on a schema violation. 2 -> 3 (temporary).
    #: It is unnecessary once the contradictory instructions are resolved,
    #: but while the refusal rate is still high it separates "the model is
    #: learning" from "structurally impossible".
    max_retries: int = 3
    #: The concurrency cap. Lower it when hitting a rate limit, but
    #: **record that in the log.**
    concurrency: int = 6
    #: ★ The hardware facts. **There is no default** (D-113). The default
    #: was pinned to `"hw/sm_86.md"`, so a 5090 run received A6000 facts.
    #: One of the two must be present, and calling RuleWriter without it is
    #: a **failure**.
    #:   `hw_text`     the body generated from the bundle (recommended — the
    #:                 `hwprompt` module)
    #:   `arch_prompt` a file path. Used only when retracing an old run
    arch_prompt: str | None = None
    hw_text: str | None = None
    #: ★ The OpenAI endpoint. **It stays in `config.json`** — mixing them
    #: breaks comparison, so it must be checkable later (D-31, D-44).
    #:
    #:   "responses"  /v1/responses.  structured output + reasoning together
    #:   "chat"       /v1/chat/completions.  the traditional form
    #:
    #: The gpt-5.6 family blocks the **function tools + reasoning_effort**
    #: combination on chat with a 400. Structured output (`output_type`) is
    #: implemented as function tools, so it is caught. The workaround is
    #: `reasoning_effort='none'`, but that switches reasoning off and loses
    #: the ability to derive physics — so the endpoint was moved instead.
    #: gpt-5.4 / 5.4-mini support responses too, so this can be uniform.
    endpoint: str = "responses"
    #: ★ The reasoning effort. **It is stated** — otherwise the model
    #: default applies, and if that default changes our results silently
    #: change with it (§15.4 reproducibility).
    #:
    #: Measured (gpt-5.6-luna, Responses API):
    #:   none    0 reasoning tokens
    #:   low     ~150
    #:   medium  ~130   <- adopted
    #:   high    ~520
    #: On our real prompt (7,177 tokens) the default spent 1,756 reasoning
    #: tokens — a heavier task spends proportionally more.
    #:
    #: With `None` nothing is sent (the model default). Even then, **to
    #: record that it was intended**, None has to be written explicitly.
    reasoning_effort: str | None = "medium"
    #: ★ How the features are shown. **An experimental condition, so it is
    #: recorded** (D-31).
    #:
    #:   "full"   name + range + physical meaning + why it matters (today's
    #:            default)
    #:   "names"  names only — the state before 2026-08-22
    #:
    #: The difference between these two was the only effect in this
    #: repository that exceeded the seed spread, and that measurement came
    #: from **a model that had been changed arbitrarily**, so it is being
    #: remeasured (D-52). It is a flag because reverting the code back and
    #: forth makes it impossible to tell which run was under which
    #: condition.
    feature_detail: str = "full"

    def to_dict(self) -> dict:
        """The form recorded into `config.json`.

        ★ `hw_text` is kept as **a hash and the first line**, not the whole
        body. That is enough to retrace the condition, and the body is
        already there verbatim in `llm_calls/_system-*.md` (added in
        pending_fixes 10).
        """
        import hashlib

        d = dict(self.__dict__)
        t = d.pop("hw_text", None)
        d["hw_text"] = None if t is None else {
            "sha256": hashlib.sha256(t.encode()).hexdigest()[:16],
            "n_chars": len(t),
            "gpu": next((ln.split("GPU")[1].strip()
                         for ln in t.split("\n") if ln.startswith("GPU")),
                        "?")}
        return d


@dataclass
class Budget:
    """The call and token caps. **Exceeding them stops the run**
    (§4-1)."""

    max_calls: int = 400
    max_input_tokens: int = 3_000_000
    max_output_tokens: int = 600_000
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_hits: int = 0
    #: ★ A failed call spends tokens too. Not counting them leaves a hole
    #: in the budget watchdog.
    failed_calls: int = 0

    def charge(self, n_in: int, n_out: int) -> None:
        self.calls += 1
        self.input_tokens += n_in
        self.output_tokens += n_out
        if self.calls > self.max_calls:
            raise BudgetExceeded(
                f"calls {self.calls} > cap {self.max_calls}")
        if self.input_tokens > self.max_input_tokens:
            raise BudgetExceeded(
                f"input tokens {self.input_tokens:,} > cap "
                f"{self.max_input_tokens:,}")
        if self.output_tokens > self.max_output_tokens:
            raise BudgetExceeded(
                f"output tokens {self.output_tokens:,} > cap "
                f"{self.max_output_tokens:,}")

    def line(self) -> str:
        return (f"calls {self.calls}+{self.failed_calls} failed "
                f"(cached {self.cached_hits})  "
                f"in {self.input_tokens:,}  out {self.output_tokens:,}")


class OpenAILLM:
    """The same interface as `MockLLM`. The loop does not distinguish
    between them."""

    def __init__(self, cfg: LLMConfig, *, feature_names, shape_values,
                 registry, budget: Budget | None = None,
                 cache: bool = True) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingAPIKey(
                "there is no OPENAI_API_KEY. The real LLM run stops.\n"
                "  It does not silently fall back to MockLLM — that would "
                "make us falsely believe 'the LLM wrote the rules' "
                "(§26.4).")
        self.cfg = cfg
        self.features = list(feature_names)
        self.shape_values = list(shape_values)
        # ★ RuleWriter must be given the **physical definitions**, not a
        #   list of names. `feature_names` cannot read physical_meaning.
        #   There is no default — which registry goes into the prompt is an
        #   experimental condition, and with `None`, `render_features` fell
        #   back to the human 24 (§30.9). The call site must now state it.
        if registry is None:
            raise ValueError(
                "OpenAILLM(registry=...) is mandatory. Which feature list "
                "goes into the prompt under conditions F1~F3 is the "
                "experiment itself (§26.4).")
        # ★ The same judgement in two places diverges (principle 2).
        #   `feature_names` is used by the static checks and `registry` by
        #   the prompt — diverged, the LLM is either checked against names
        #   it has never seen, or the prompt recommends names the checks do
        #   not have. When neither is empty, containment is enforced.
        # ★ `M/N/K/n_candidates` are always present, independent of the
        #   registry — they are properties of the problem, not features.
        #   Leaving them out at first killed a validation run before it even
        #   started (before any LLM call, so nothing was lost).
        from kernelrule.core.matrix import INTRINSIC_SHAPE_FIELDS

        known = set(registry._items) | set(INTRINSIC_SHAPE_FIELDS)
        if set(registry._items) and (self.features or self.shape_values):
            stray = sorted((set(self.features) | set(self.shape_values))
                           - known)
            if stray:
                raise ValueError(
                    f"feature_names/shape_values contain names absent from "
                    f"registry {registry.name!r}: {stray}. The static checks "
                    f"and the prompt would see different lists "
                    f"(principle 2).")
        self.registry = registry
        self.budget = budget or Budget()
        self.calls: list[LLMCall] = []
        self._seq = 0
        #: Prompt hash -> response. Early on the same prompt repeats often
        #: (§15.4)
        self._cache: dict[str, object] = {} if cache else None
        self._agents: dict[str, object] = {}
        #: ★ Roles outside the loop (D-92). An experiment script registers
        #: one carrying **its own prompt and its own schema**.
        #: `kernelrule/agents/` knows only the four the loop calls (analyze /
        #: rule_writer / rule_editor / feature + categorize) — a role that
        #: is not in the loop, left here, reads as "to be switched on some
        #: day".
        self._extra: dict[str, tuple] = {}
        # ★ The effective term budget. **Decided once here and passed to
        #   every place** (principle 2). It used to be decided separately by
        #   `load_prompt`'s default and a direct import of
        #   `checks.PARAMETERS`, so even with `parameters=16` **the user
        #   prompt and the role files rendered 8** (D-105).
        from kernelrule.rules.checks import PARAMETERS as _CHECK_PARAMETERS
        self._parameters = int(cfg.parameters if cfg.parameters is not None
                           else _CHECK_PARAMETERS)
        self._rank_top_k = int(getattr(cfg, "rank_top_k", 100))
        self._rank_lambda = float(getattr(cfg, "rank_lambda", 0.0))
        self._product = bool(getattr(cfg, "product_hint", False))
        self._power = bool(getattr(cfg, "power_hint", False))
        self._base = load_prompt("_base.md", parameters=self._parameters)
        # ★ Missing hardware facts **do not kill it here** — it dies when
        #   RuleWriter is called. Analyst/RuleEditor/FeatureWriter do not
        #   receive them (§16.2), so there is no reason to block calls
        #   constructed without a table.
        self._hw = (cfg.hw_text if cfg.hw_text is not None
                    else (load_prompt(cfg.arch_prompt, parameters=self._parameters)
                          if cfg.arch_prompt else None))
        #: ★ The objective. It sets the prompt's "how it is scored"
        #: section (D-101).
        self.objective = getattr(cfg, "objective", "regret")
        if self.objective not in _OBJECTIVE_BLOCKS:
            raise ValueError(f"unknown objective: {self.objective!r}")
        self._rules = load_prompt("role/_rules_common.md",
                                  parameters=self._parameters)
        # ★ Assembly happens in `assemble_instructions` alone
        #   (principle 2). The four below are read **only to confirm the
        #   prompts exist** — a missing file must kill it here, not on the
        #   first call.
        self._edit = load_prompt("role/_rules_edit.md",
                                 parameters=self._parameters)
        # ⚠️ Creating the `asyncio.Semaphore` here **binds it to the first
        #    event loop.** The loop calls `asyncio.run()` afresh every round,
        #    so from the second round it dies with "bound to a different
        #    event loop". We stepped on this, and the exception was handled
        #    as a discarded candidate, so **calls were silently lost.** One
        #    is created per loop.
        self._sems: dict[int, asyncio.Semaphore] = {}
        self.rate_limit_events = 0
        #: (round, attempt number, reason code, message). It shows whether
        #: the feedback works — if what was caught on attempt 1 is caught on
        #: attempt 2 **for the same reason**, the fix is the prompt, not a
        #: higher retry cap.
        self.violations: list[dict] = []
        self.round = -1

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = self._sems.get(id(loop))
        if sem is None:
            sem = self._sems[id(loop)] = asyncio.Semaphore(
                self.cfg.concurrency)
        return sem

    # -- Agent construction (§11.2 — fixed roles + injected domain facts) -
    def _agent(self, role: str):
        if role in self._agents:
            return self._agents[role]
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

        from kernelrule.agents.schemas import (
            AnalysisOutput,
            CategoryOutput,
            FeatureOutput,
            rule_output_for,
        )

        # ★ Roles outside the loop are registered by experiment scripts
        #   themselves through `register_role` (D-92).
        #   `kernelrule/agents/` knows only **the four the loop calls**.
        if role in self._extra:
            out, body = self._extra[role]
        else:
            # ★ The rule roles use an output type **carrying the budget**
            #   (D-107). The field descriptions go to the model as the tool
            #   schema — with "at most 8" frozen in there, the model
            #   produces 8 even when the prompt says 16.
            _rule_out = rule_output_for(self._parameters,
                                        product_hint=self._product,
                                        power_hint=self._power)
            out = {"analyze": AnalysisOutput, "rule_editor": _rule_out,
                   "rule_writer": _rule_out, "feature": FeatureOutput,
                   "categorize": CategoryOutput}[role]
            body = load_prompt(f"role/{role}.md", parameters=self._parameters) \
                .replace("{product_note}", product_note(self._product)) \
                .replace("{power_note}", power_note(self._power))
        # ★ It splits along **two axes** (§30.10). Split along one axis
        #   only (hardware-independent / dependent), things no role needed
        #   piled up in the common part — FeatureWriter was receiving the
        #   regret definition and the weight budget every time.
        #
        #                      hardware-independent   hardware-dependent
        #     role-independent  _base.md              the generated hw facts
        #     role-dependent    role/*.md             (none)
        #
        #   `hw` goes to Analyst / RuleWriter only. Not seeing it makes the
        #   RuleEditor and FeatureWriter prompts **GPU-independent**, usable
        #   as they are on a new GPU (§16.2).
        #   The rule block splits in two again — `_rules_common.md` (function
        #   shape, budget, vectorisation) for RuleEditor + RuleWriter, and
        #   `_rules_edit.md` (the regret definition, the gallery of refused
        #   cases) for **RuleEditor only**. RuleWriter writes from a blank
        #   page, so there is no term to replace and no previous score
        #   (§30.10).
        instructions = assemble_instructions(
            role, objective=self.objective, hw_file=self.cfg.arch_prompt,
            hw_text=self.cfg.hw_text,
            body=body, parameters=self.cfg.parameters,
            rank_top_k=self._rank_top_k, rank_lambda=self._rank_lambda,
            product_hint=self._product, power_hint=self._power)
        if self.cfg.endpoint not in ("responses", "chat"):
            raise ValueError(
                f"unknown endpoint: {self.cfg.endpoint!r}. "
                f"'responses' or 'chat'")
        model = (OpenAIResponsesModel(self.cfg.model)
                 if self.cfg.endpoint == "responses"
                 else OpenAIChatModel(self.cfg.model))
        # ★ An unsendable combination **stops here** (D-47). Handed to
        #   pydantic-ai it is silently dropped, while the value stays in
        #   config.json, so the record and reality diverge.
        if self.cfg.seed is not None and self.cfg.endpoint == "responses":
            raise ValueError(
                "seed **is not a parameter** on the Responses endpoint "
                "(the SDK raises a TypeError). pydantic-ai drops it "
                "silently, so it survives only in config.json. Leave "
                "`seed=None` or use `endpoint='chat'` (D-47).")
        settings: dict = {}
        if self.cfg.temperature is not None:
            settings["temperature"] = self.cfg.temperature
        if self.cfg.seed is not None:
            settings["seed"] = self.cfg.seed
        if self.cfg.reasoning_effort is not None:
            # pydantic-ai passes it under a provider-prefixed key.
            settings["openai_reasoning_effort"] = self.cfg.reasoning_effort
        a = Agent(model, output_type=out, instructions=instructions,
                  retries=self.cfg.max_retries, model_settings=settings)
        self._agents[role] = a
        return a

    # -- Prompt assembly --------------------------------------------------
    def _feature_block(self) -> str:
        """★ **Every role uses the same renderer** (§11.2 / D-34).

        Only RuleWriter used to receive the ranges and physical definitions
        through `render_features()`, while RuleEditor and Analyst got **a
        list of names**. The evolution loop's LLM picked terms without
        knowing what `has_spill` measures, and so missed a free pruning
        (`artifacts/spill-term.md`).

        Leaving two paths lets them diverge again. `_feature_block` is
        absorbed into this one so that there is **a single source**.

        ⚠️ The hardware constants (SM 84, smem 99KB, ridge) are not given.
        The features have already absorbed them, and not seeing them makes
        this prompt **GPU-independent**, usable as it is on a new GPU
        (§16.2).
        """
        from kernelrule.features import render_features

        if self.cfg.feature_detail not in ("full", "names"):
            raise ValueError(
                f"unknown feature_detail: {self.cfg.feature_detail!r}. "
                f"'full' or 'names'")
        if self.cfg.feature_detail == "names":
            # ★ It reproduces the state before 2026-08-22 exactly — a list
            #   of names only.
            return ("## Config level (`f.<name>`)\n\n"
                    + "\n".join(f"- `{n}`" for n in self.features)
                    + "\n\n## Shape level (`p.<name>`)\n\n"
                    + "\n".join(f"- `{n}`" for n in self.shape_values))
        if self.registry is None:
            # Without a registry, names only — and **it says so** (§26.4)
            names = "\n".join(f"- `{n}`" for n in self.features)
            svals = "\n".join(f"- `{n}`" for n in self.shape_values)
            return ("⚠️ The physical definitions of the features could not be "
                    f"loaded (no registry). Only the names are "
                    f"here.\n\n{names}\n\n{svals}")
        return render_features(self.registry, include_observed=False)

    def _user_prompt(self, role: str, prompt: str, **kw) -> str:
        fl = self._feature_block()
        if role == "rule_writer":
            return self._rule_writer_prompt(**kw)
        if role == "categorize":
            return self._categorize_prompt(**kw)
        if role == "feature":
            return self._feature_prompt(**kw)
        if role in self._extra:
            # For a registered role the caller builds the user prompt too.
            return prompt
        if role == "analyze":
            return (prompt + "\n\n---\n\n## Registered features\n\n"
                    + fl + "\n")
        parent = kw.get("parent")
        hyp = kw.get("hypothesis") or {}
        applied = kw.get("hypotheses_applied") or []
        # ★ The parent's current term count is injected, and at saturation
        #   it **instructs a replacement**. "You may drop one" is an option;
        #   "drop one and put it in" is an instruction. With the budget only
        #   in `role/_rules.md` (the system prompt) it is diluted in a long
        #   context.
        n_terms = int(kw.get("parent_n_terms") or 0)
        # ★ The budget is **per path** (D-144). The room left is counted on
        #   the heaviest path.
        n_path = int(kw.get("parent_path_params") or 0)
        n_w = len(parent.w0) if parent else 0
        # ★ Reading `checks.PARAMETERS` directly ignores `parameters`
        #   (D-105). The effective budget is `self._parameters` alone.
        if n_path >= self._parameters:
            note = ("\n★ The **heaviest path** is at the cap "
                    f"({n_path}/{self._parameters}).\n"
                    "  Do one of two things:\n"
                    "  (1) **split a branch** with `if p.<shape value>` — "
                    "each branch spends its own room\n"
                    "  (2) **drop the least important term** on that path and "
                    "put the new one there\n"
                    "  Write what you did and why you chose it in `changes`.")
        else:
            note = (f"Room left on the heaviest path: "
                    f"{self._parameters - n_path} ({n_terms} terms in total)")
        # ★ With the Analyst off, **the hypothesis section is not built at
        #   all** (§16.1, D-89). Leaving an empty slot such as
        #   "## This round's hypothesis\n\n(none)" makes the model read "there
        #   is a hypothesis and it is empty", a different condition. It is
        #   the same principle as not building the diagnostic report either.
        # ★ The prompt with the Analyst off must be **the one with it on,
        #   with sentences deleted** (§16.1, D-89). Writing new wording
        #   breaks "only the Analyst differs" —
        #   `test_optimize_prompt_without_analyst_is_a_deletion` pins it.
        if kw.get("analyst", True):
            hyp_block = (
                "## Hypotheses already reflected in the current rule\n\n"
                + ("\n".join(f"- {h}" for h in applied)
                   or "(none yet — this is the first round)")
                + "\n\n## This round's hypothesis\n\n"
                + (json.dumps(_for_editor(hyp), ensure_ascii=False, indent=1)
                   if hyp else
                   "(no hypothesis. Find a direction to improve the parent "
                   "yourself)"))
            inputs_hyp = ("the one hypothesis to reflect this round\n"
                          "the hypotheses already reflected in the rule\n")
            one_change = "The hypothesis being local is **deliberate**. "
            applied_warn = (
                "\n**Do not undo the effect of the existing hypotheses.** The "
                "terms listed below\nare there for a reason. Removing one "
                "requires evidence that this round's\nhypothesis invalidates "
                "that reason.\n")
        else:
            hyp_block = inputs_hyp = one_change = applied_warn = ""
        # ★ There is a second parent only under `cross` (D-96). Without
        #   one **the section is not built at all** — leaving an empty slot
        #   such as "(no parent)" makes the model read "there is a second and
        #   it is empty", which changes the conditions of exploit/explore
        #   (we stepped on this in D-89).
        p2 = kw.get("parent2")
        if p2 is None:
            second = ""
        else:
            second = (
                "\n\n## ★ Second parent — **combine these two**\n\n"
                "```python\n" + p2.code.strip() + "\n```\n\n"
                f"Second parent's weights: {list(p2.w0)}\n\n"
                "**Pick the good terms from each and make one rule.** Do not "
                "copy one side as is — that is not a crossover.\n\n"
                f"⚠️ The cap is {self._parameters} per path, so combining "
                "**forces you to drop something.** Write what you dropped and "
                "why you chose it in `changes`.\n")
        body = load_prompt("role/rule_editor.md", parameters=self._parameters) \
            .replace("{product_note}", product_note(self._product)) \
            .replace("{power_note}", power_note(self._power))
        return body.format(
            second_parent_block=second,
            n_terms=n_terms, n_weights=n_w, parameters_note=note,
            feature_block=fl, hypothesis_block=hyp_block,
            inputs_hyp=inputs_hyp, one_change_hyp=one_change,
            applied_warning=applied_warn,
            parent_code=(parent.code if parent else
                         "(no parent — write from scratch)"),
            parent_w=(list(parent.w0) if parent else "-"))


    # -- RuleWriter (§11.8) — it receives no parent, no cases, no score ---
    def _rule_writer_prompt(self, *, condition: str = "A",
                          table_facts=None, registry=None, **_kw) -> str:
        """★ Condition A has **not one sentence that came from the
        table.**

        Consistency with the transfer scenarios is why this condition
        exists:

            full port    §29.5(a)   0 table      structure + weights as they are
            refit        §29.5(b)   5% sample    structure fixed, weights only
            regenerate   §29.5(c)   exhaustive   from the structure up

        If the **structure** needs the table, that is (c). But if you are
        going to measure exhaustively you can use the table directly, so
        there is no reason to use this system. So A is the gate condition,
        and the gap against B is exactly "what the table is worth".

        The assembly is not done by hand — it goes through
        `render_features` (D-28).
        """
        from kernelrule.features import render_features

        if condition not in ("A", "B"):
            raise ValueError(f"unknown RuleWriter condition: {condition!r}. "
                             f"A (physics only) or B (physics + training-split "
                             f"aggregates)")
        if condition == "B" and table_facts is None:
            raise ValueError(
                "condition B needs the training-split aggregates. Pass "
                "TableFacts.compute(table, splits.train) (§12.3).")

        # ★ It uses the registry it was given. It used to be pinned to
        #   `self.registry`, which could diverge when called with another
        #   library such as F1/F2 (§30.9).
        reg = registry if registry is not None else self.registry
        extra = getattr(table_facts, "by_feature", None) if table_facts else None
        block = render_features(reg, include_observed=condition == "B",
                                extra_observed=extra)
        if condition == "A":
            note = "you do not see this GPU's measurement table"
            agg = ("## Table aggregates\n\n**None.** That is condition A — "
                   "write from physics alone.")
        else:
            lines = "\n".join(table_facts.lines)
            note = ("you see only **aggregates** of the training split, "
                    "never per-shape answers")
            agg = ("## Table aggregates (training split only — §12.3)\n\n"
                   "These are patterns over the whole split, not answers for "
                   "individual shapes. **Nothing here identifies a shape.**"
                   f"\n\n```\n{lines}\n```")
        # ★ The rule example **differs per condition** too (§30.20).
        #   RuleWriter's `condition` is A/B (with or without table
        #   observations), a different axis from the feature conditions — if
        #   the registry is the human 24 the real names may be used, and with
        #   an F0/F1 registry an unrelated domain must be.
        rule_ex = _rule_example_for(reg, parameters=self._parameters)
        return load_prompt("role/rule_writer.md",
                           parameters=self._parameters).format(
            rule_example_block=rule_ex,
            table_note=note, feature_block=block, aggregate_block=agg)


    # -- FeatureWriter (§11.4) — it builds axes that do not exist ---------
    def _categorize_prompt(self, *, n_min: int = 5, n_max: int = 8,
                           **_kw) -> str:
        """★ **The LLM** partitions the areas (§30.10).

        Handing over human-written categories hands over prior knowledge —
        it amounts to saying "memory traffic matters". And the partition
        itself is an object of observation: how does the LLM structure the
        physics of GEMM performance?
        """
        from kernelrule.features.generated import field_block

        return load_prompt("role/categorize.md",
                           parameters=self._parameters).format(
            field_block=field_block(), n_min=n_min, n_max=n_max)

    def _feature_prompt(self, *, condition: str = "F1", task: str = "",
                        registry=None, **_kw) -> str:
        """F1~F3 — the condition is **how many features are given**.
        0 -> 5 -> 24.

            F1  start from 0    can it build derived quantities  ★ the
                                fundamental question
            F2  5 public facts  can it build on top of them
                                (`F1-K` before the D-128 rename)
            F3  the human 24    combination only (= every run so far)

        ⚠️ The shape example is **something unrelated to this problem**. The
        output example in `optimize.md` was an abridged `human_guided`, and a
        seed once entered the "no seed" condition that way (D-35).
        """
        from kernelrule.features import render_features
        from kernelrule.features.generated import field_block

        if condition not in _CONDITIONS:
            raise ValueError(
                f"unknown condition: {condition!r}. {sorted(_CONDITIONS)}")
        reg = registry if registry is not None else self.registry
        if condition == "F1":
            block = ("## Existing features\n\n**None.** Derive the "
                     "physical quantities from the raw values above.\n\n"
                     "★ That is the point of this condition — whether you can "
                     "build derived quantities yourself.")
        else:
            if reg is None:
                raise ValueError(
                    f"condition {condition} needs a registry")
            block = ("## Existing features — **duplicates are discarded**"
                     "\n\n"
                     + render_features(reg, include_observed=False))
        # ★ The example **differs per condition** (§30.17). F1 uses an
        #   unrelated domain so as not to hand over the answer, and the
        #   conditions that do give public knowledge (F2/F3) show the real
        #   features down to the code — that is the definition of the
        #   condition, so D-35's caution does not apply here.
        example = load_prompt(f"examples/{_EXAMPLES[condition]}.md",
                              parameters=self._parameters)
        return load_prompt("role/feature.md", parameters=self._parameters).format(
            field_block=field_block(), feature_block=block,
            example_block=example,
            area_block=load_prompt("areas.md", parameters=self._parameters),
            task_block=task or ("## What to build now\n\nPropose one "
                                "feature."))

    # -- Registering roles outside the loop (D-92) ------------------------
    def register_role(self, name: str, *, instructions: str,
                      output_type) -> None:
        """**The caller** registers a role the loop does not have.

        ★ A role the loop never calls, left in `kernelrule/agents/`, reads
        as "to be switched on some day" and keeps being dragged through the
        condition lists and the ablation table (D-92). **The side that uses
        it brings** the prompt and the schema.

        The budget, retries, tracing and `dump()` are used as they are —
        an LLM call cannot be remade, so there must be exactly one path that
        records it (D-33).
        """
        if name in ("analyze", "rule_writer", "rule_editor", "feature",
                    "categorize"):
            raise ValueError(
                f"{name!r} is a loop role. Overwriting it silently runs "
                f"something else.")
        self._extra[name] = (output_type, instructions)

    # -- Entry points -----------------------------------------------------
    def complete(self, role: str, prompt: str, **kw):
        return asyncio.run(self.acomplete(role, prompt, **kw))

    async def acomplete(self, role: str, prompt: str, **kw):
        user = self._user_prompt(role, prompt, **kw)
        h = hashlib.sha256((role + "\x00" + user).encode()).hexdigest()[:16]
        seq = self._seq
        self._seq += 1
        if self._cache is not None and h in self._cache:
            self.budget.cached_hits += 1
            return self._cache[h]

        agent = self._agent(role)
        async with self._semaphore():
            t0 = time.perf_counter()
            try:
                res = await self._run_traced(agent, user, role, seq)
            except Exception as e:                       # noqa: BLE001
                name = type(e).__name__
                if "RateLimit" in name or "429" in str(e):
                    self.rate_limit_events += 1
                    # ★ It is not quietly reduced. It is logged and it
                    #   backs off once.
                    await asyncio.sleep(20.0)
                    res = await agent.run(user)
                else:
                    # ★ Even on failure the tokens are already spent. At
                    #   least the call count is counted — without it, a
                    #   runaway retry loop never trips the budget watchdog.
                    self.budget.failed_calls += 1
                    if (self.budget.calls + self.budget.failed_calls
                            > self.budget.max_calls):
                        raise BudgetExceeded(
                            f"calls {self.budget.calls}+"
                            f"{self.budget.failed_calls} failed > cap "
                            f"{self.budget.max_calls}") from e
                    raise
            dt = time.perf_counter() - t0

        # In pydantic-ai 2.x it is an attribute, in 1.x a method. Falling
        # silently to 0 would disarm the budget watchdog, so **both are
        # tried and failure is an error**.
        u = res.usage
        if callable(u):
            u = u()
        n_in = (getattr(u, "input_tokens", None)
                or getattr(u, "request_tokens", None))
        n_out = (getattr(u, "output_tokens", None)
                 or getattr(u, "response_tokens", None))
        if n_in is None or n_out is None:
            raise RuntimeError(
                f"the token usage cannot be read: {type(u).__name__} "
                f"{[a for a in dir(u) if 'token' in a]}. "
                f"Falling to 0 would disarm the budget watchdog (§26.4).")
        self.budget.charge(n_in, n_out)
        out = res.output
        payload = out.model_dump() if hasattr(out, "model_dump") else out
        self.calls.append(LLMCall(role=role, prompt_hash=h, response=payload,
                                  seq=seq, mode=self.cfg.model))
        # The original is kept. ★ No keys or auth headers are stored.
        self._last = {"prompt": user, "seconds": dt,
                      "input_tokens": n_in, "output_tokens": n_out}
        self.calls[-1].__dict__["_meta"] = self._last
        if self._cache is not None:
            self._cache[h] = payload
        return payload

    async def _run_traced(self, agent, user: str, role: str, seq: int):
        """Records the **per-attempt violations** of Pydantic AI's
        retries.

        The framework feeds a validator failure back to the model and
        retries, and that history is invisible from outside. It is recovered
        from the result messages and recorded per attempt.
        """
        # ★ It is wrapped in `capture_run_messages`. Without it the
        #   per-attempt messages are invisible **when it fails** — only the
        #   exception remains and there is no `res`. Under RuleWriter
        #   condition A, 8 of 10 died on exhausted retries and there was no
        #   way to know what was caught. Then there is no way to know where
        #   to fix the prompt (§26.4 — a failure must leave information).
        from pydantic_ai import capture_run_messages

        def _harvest(msgs, seq_: int) -> None:
            for i, m in enumerate(msgs or []):
                for part in getattr(m, "parts", []):
                    # ★ `RetryPromptPart.content` is **a list of dicts**.
                    #   Looking only at str misses the whole feedback
                    #   history — the reason for exhausted retries really was
                    #   unreadable.
                    raw = getattr(part, "content", "")
                    content = raw if isinstance(raw, str) else str(raw)
                    if ("validation error" in content.lower()
                            or "Value error" in content):
                        self.violations.append(
                            {"round": self.round, "seq": seq_, "role": role,
                             "attempt": i, "code": classify_violation(content),
                             "msg": content[:300]})

        with capture_run_messages() as msgs:
            try:
                res = await agent.run(user)
            except Exception as e:                        # noqa: BLE001
                _harvest(msgs, seq)
                self.violations.append(
                    {"round": self.round, "seq": seq, "role": role,
                     "attempt": -1,
                     "code": classify_violation(f"{type(e).__name__}: {e}"),
                     "msg": f"{type(e).__name__}: {e}"[:200]})
                raise
            _harvest(msgs, seq)
        return res

    def violation_report(self) -> dict:
        """The reason code x attempt distribution. It shows whether the
        feedback works."""
        from collections import Counter

        by_code = Counter(v["code"] for v in self.violations)
        by_attempt = Counter(v["attempt"] for v in self.violations)
        # Did the same code appear more than once within one call (seq)
        seen: dict[int, list[str]] = {}
        for v in self.violations:
            seen.setdefault(v["seq"], []).append(v["code"])
        repeated = sum(1 for codes in seen.values()
                       if len(codes) > 1 and len(set(codes)) == 1)
        return {"total": len(self.violations), "by_code": dict(by_code),
                "by_attempt": dict(by_attempt),
                "same_code_repeated": repeated,
                "n_calls_with_violation": len(seen)}

    async def many(self, role: str, items: list[dict]):
        """Calls for 12 rules **in parallel** (§4-0)."""
        return await asyncio.gather(
            *(self.acomplete(role, it.pop("prompt", ""), **it)
              for it in items), return_exceptions=True)

    # -- Recording --------------------------------------------------------
    def dump(self, out: str | Path) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for c in self.calls:
            meta = c.__dict__.get("_meta", {})
            (out / f"{c.seq:05d}-{c.role}.json").write_text(json.dumps(
                {"role": c.role, "prompt_hash": c.prompt_hash, "seq": c.seq,
                 "model": c.mode, "response": c.response,
                 "prompt": meta.get("prompt", ""),
                 "input_tokens": meta.get("input_tokens"),
                 "output_tokens": meta.get("output_tokens"),
                 "seconds": meta.get("seconds")},
                ensure_ascii=False, indent=1))
        # ★ **The conditions** are recorded (pending_fixes 10). Recording
        #   the user prompt alone makes contradictions inside the system
        #   prompt and the output schema unfindable from the artefacts —
        #   D-105 was inside the system prompt and D-107 inside the output
        #   schema, and neither was in the logs.
        for role in sorted({c.role for c in self.calls}):
            try:
                sysp, sch = self._condition_of(role)
            except Exception as e:                  # pragma: no cover
                (out / f"_system-{role}.err").write_text(f"{type(e).__name__}: {e}")
                continue
            (out / f"_system-{role}.md").write_text(sysp)
            if sch is not None:
                (out / f"_schema-{role}.json").write_text(
                    json.dumps(sch, ensure_ascii=False, indent=1))

    def _condition_of(self, role: str) -> tuple[str, dict | None]:
        """The system prompt and output schema that role **actually
        received**."""
        from kernelrule.agents.schemas import rule_output_for

        if role in self._extra:
            return self._extra[role][1], None
        body = load_prompt(f"role/{role}.md", parameters=self._parameters)
        sysp = assemble_instructions(
            role, objective=self.objective, hw_file=self.cfg.arch_prompt,
            hw_text=self.cfg.hw_text,
            body=body, parameters=self._parameters,
            rank_top_k=self._rank_top_k, rank_lambda=self._rank_lambda,
            product_hint=self._product, power_hint=self._power)
        sch = None
        if role in ("rule_editor", "rule_writer"):
            sch = rule_output_for(
                self._parameters, product_hint=self._product,
                power_hint=self._power).model_json_schema()
        return sysp, sch


def estimate_and_confirm(*, n_rounds: int, n_rules: int, report_chars: int,
                         cfg: LLMConfig, yes: bool = False) -> dict:
    """Prints the expected call and token counts and asks for confirmation
    (§4-1)."""
    per_round = 1 + n_rules
    calls = per_round * n_rounds
    tok_in = int(report_chars / 3) * n_rounds + int(report_chars / 6) * \
        n_rules * n_rounds
    est = {"model": cfg.model, "calls": calls,
           "est_input_tokens": tok_in, "per_round": per_round}
    print("=" * 62)
    print(f"real LLM run estimate  model {cfg.model}  "
          f"temperature {cfg.temperature}")
    print(f"  {n_rounds} rounds x (1 diagnosis + {n_rules} rules) = "
          f"{calls} calls")
    print(f"  input tokens roughly {tok_in:,}")
    print("=" * 62)
    if not yes:
        raise BudgetExceeded(
            "confirmation is needed. Proceed with `--yes` or adjust the "
            "budget.")
    return est
