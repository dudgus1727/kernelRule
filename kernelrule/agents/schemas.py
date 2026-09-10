"""The schemas at the LLM boundary (§11.7).

**Pydantic is used only here.** The hot scoring path (`core/types.py`) is
frozen dataclasses — they are constructed and hashed millions of times per
round, so a validation layer would make it an order of magnitude slower.

Importing must work without Pydantic (the `[llm]` optional dependency).
Without it, it falls back to thin dataclasses but **states plainly that
there is no validation** — it does not pass silently (§26.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from kernelrule.rules.checks import (
    FITTER_SWITCH_DIM,
    exponent_message,
    literal_parameter_message,
    noop_term_message,
    weight_reuse_message,
)

__all__ = ["Hypothesis", "HypothesisSet", "FeatureProposal", "CritiqueOutput",
           "RuleProposal", "SchemaViolation", "validate_rule_proposal",
           "HAVE_PYDANTIC", "check_banned", "MAX_WEIGHTS", "rule_output_for",
           "N_HYP_MIN", "N_HYP_MAX"]

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    HAVE_PYDANTIC = True
except ImportError:                                # pragma: no cover
    HAVE_PYDANTIC = False


class SchemaViolation(ValueError):
    """The LLM response violated the schema. **Retry, then discard**
    (§26.4).

    There is no partial acceptance — patching up a half-right rule makes it
    impossible to say what that rule tested.
    """


class _NoPydantic:
    """Announces Pydantic's absence **at the moment of use** (§26.4 /
    4-5).

    It used to be `AnalysisOutput = None`. Passing `output_type=None` to
    Pydantic AI raises an `AttributeError` somewhere far below, and from that
    message alone **you cannot read that validation was switched off
    entirely.** It does not roll on silently in a bad state.

    ★ It is defined **even when** Pydantic is present — so that this
    behaviour can be tested.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def _die(self, *_a, **_k):
        raise ImportError(
            f"{self._name} needs Pydantic. Validation at the LLM boundary "
            "is **disabled** — schema violations are not filtered out. "
            "Install it with `pip install -e '.[llm]'` (§26.4)")

    __call__ = _die
    __getattr__ = _die


#: Appearing in rule code means immediate refusal. `rules/checks.py` looks
#: again with the AST.
#: **A string check is bypassable, so it runs alongside the structural
#: defence** (§11.7).
BANNED_SUBSTRINGS = ("time_ms", "cublas_ms", "difficulty", "tflops",
                     "distinct_time_frac", "import ", "open(", "TABLE",
                     "__globals__", "eval(", "exec(", "np.random")


def _code_only(src: str) -> str:
    """Joins the tokens with comments and string literals removed (D-27).

    ★ It stops substring matching from **catching comments**. When the LLM
    wrote "this shape has high difficulty" in a comment, perfectly good code
    was refused — which burns retries and does not even say what was wrong.

    ★ This does **not weaken** the checks. `rules/checks.py` looks again at
    names, calls and imports with the AST, and the sandbox isolates
    execution (§11.7). An `import ` inside a comment does not run, so there
    is no reason to catch it here.

    If tokenisation fails (a syntax error) it **returns the original as is**
    — the check is not skipped (§26.4).
    """
    import io
    import tokenize
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src              # unparsable -> conservatively check the
                                # original
    return " ".join(t.string for t in toks
                    if t.type not in (tokenize.COMMENT, tokenize.STRING))


def check_banned(code: str) -> str | None:
    """Returns the banned string if one is found, else `None`. **Shared by
    both paths.**"""
    probe = _code_only(code)
    for b in BANNED_SUBSTRINGS:
        if b in probe:
            return b
    return None


#: ★ The **single source** for the number of hypotheses (§30.8 / D-26).
#:
#: The description, the validation and the error message were all three
#: different — the description said "3~5", the validation `1 <= n <= 8`, and
#: the error "3~5" again. Producing only 1 passed, and then every rule of
#: that round reflected **the same hypothesis**, collapsing §14.2's
#: diversity.
#:
#: ★ 2026-09-08 (D-144): **fixed at 3**.
#:
#: Measured from the traces — of 77 exploit duplicates, **67 (87%) were "the
#: same parent + the same hypothesis"**. With 12 proposals (6 exploit) and
#: 3~5 hypotheses, there were more exploit slots than hypotheses, so overlap
#: was **inevitable**. 6 proposals (3 exploit) + 3 hypotheses fits exactly.
#:
#: Old value history: description "3~5" / validation `1<=n<=8` / error "3~5",
#: all three different (D-26 reconciled them), and after that it was
#: `2, 8`.
N_HYP_MIN, N_HYP_MAX = 3, 3

#: ★ The **shape sizes** that must not enter a hypothesis sentence (D-114).
#: It catches multiplication forms such as `4096x4096` and namings such as
#: `M = 4096`. It does not catch fewer than three digits — that would
#: false-positive on config values like `stages=3`.
_SHAPE_SIZE = re.compile(r"\d{3,6}\s*[x*×]\s*\d{3,6}"
                         r"|\b[MNK]\s*=\s*\d{3,6}\b")

#: ⚠️ 2026-09-10 (D-152): **there is no weight cap** and nothing here counts
#: weights any more. The name survives only because it is exported; it is the
#: dimension at which the fitter switches, nothing else.
MAX_WEIGHTS = FITTER_SWITCH_DIM


#: ★ Experiment B (D-110). The schema must say the same thing as the
#: prompt (D-107).
_PRODUCT_DESC = (" ★ Within one term you **may multiply two features** — "
                 "`(f.a * f.b) * w[i]` costs one parameter.")

#: ★ Experiment (b) (D-112). A weight may sit **in the exponent**.
_POWER_DESC = (" ★ A weight may sit **in the exponent** — replacing "
               "`f.a * w[i]` with `np.power(f.a, w[i])` costs no extra room "
               "and fits the exponent. The base must be a single "
               "`f.<name>`, and the exponent weight is bounded to 0~4.")


def _desc_code(product: bool = False, power: bool = False) -> str:
    return ("The full function, starting at `def score(f, p, hw, w):`. "
            "No prose, no markdown fences. "
            "★ Each w[i] may be used exactly once — reusing one weight "
            "across terms is rejected — and `len(w0)` must equal the largest "
            "index used + 1, with no gaps. "
            "Use as many terms as the physics needs; when you split on a "
            "shape value, give each branch its own weights."
            + (_PRODUCT_DESC if product else "")
            + (_POWER_DESC if power else ""))


def _desc_w0() -> str:
    """⚠️ 2026-09-10 (D-151): this sentence used to end with "★ At most 8
    **per execution path**". D-150 took the cap out of the checker and the
    prompts — **and missed this one.** `pydantic-ai` hands the field
    description to the model as the tool schema, so the model kept reading a
    cap that no longer existed and, asked point-blank, quoted it back: "the
    limit comes from the output-schema requirement"."""
    return ("Initial weights. ★ Do not give them carelessly — the objective "
            "is a step function and the optimiser can get stuck on a plateau "
            "near the starting point. Give a **starting point that reflects "
            "the physical magnitude of each term**. The length must equal the "
            "largest index the code references + 1, with no gaps")


@dataclass
class Hypothesis:
    """A natural-language sentence. **Not executable.** Code is not asked
    for alongside (§11.3)."""

    claim: str
    evidence_cases: list[int] = field(default_factory=list)
    affected_regime: str = ""
    measurable_with: list[str] = field(default_factory=list)
    #: ★ The slot that asks for an axis that does not exist. Only this is
    #: passed to the FeatureWriter — the diagnostic report is not (D-75).
    #:
    #: ⚠️ On 2026-08-28 it was renamed to `physical_requirement` and then
    #: **reverted** (D-81). The 17.9% baseline was measured with this name
    #: and this description, and comparing with the rename in place changes
    #: two variables. `loop._requirement_of` reads both names, so runs made
    #: in between are still read correctly.
    needs_new_feature: str | None = None
    proposed_direction: str = ""
    risk: str = ""
    id: str = ""


@dataclass
class HypothesisSet:
    hypotheses: list[Hypothesis] = field(default_factory=list)


@dataclass
class FeatureProposal:
    name: str
    code: str
    rationale: str
    unit: str = "dimensionless"
    expected_range: tuple[float, float] = (0.0, 1.0)
    direction: str = "higher_is_worse"


@dataclass
class CritiqueOutput:
    """★ If no defect is found, it must write **the physical quantity in
    one sentence** (§11.5).

    Being unable to write the description is itself a rejection signal.
    """

    has_defect: bool
    defects: list[str] = field(default_factory=list)
    measures_what: str = ""
    confidence: float = 0.5


@dataclass
class RuleProposal:
    """★ It takes the **full code**, not a diff (§11.6).

    A diff fails to apply often and retries are expensive.
    """

    code: str
    w0: list[float]
    changes: str = ""
    hypothesis_id: str = ""
    parent_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def validate_rule_proposal(obj: Any, *, parameters: int | None = None
                           ) -> RuleProposal:
    """LLM response -> `RuleProposal`. **A violation raises. It is not
    patched up and used.**

    ⚠️ 2026-09-09 (D-150): `parameters` is accepted and ignored. There is no
    cap — a proposal is refused for reusing a weight index or leaving a hole
    in `w0`, not for how many it uses.
    """
    del parameters
    if isinstance(obj, RuleProposal):
        d = {"code": obj.code, "w0": obj.w0, "changes": obj.changes,
             "hypothesis_id": obj.hypothesis_id, "parent_ids": obj.parent_ids,
             "meta": obj.meta}
    elif isinstance(obj, dict):
        d = dict(obj)
    else:
        raise SchemaViolation(
            f"the rule proposal is not a dict: {type(obj)}")

    code = d.get("code")
    if not isinstance(code, str) or "def score" not in code:
        raise SchemaViolation(
            "code has no `def score(f, p, hw, w)`")
    if (b := check_banned(code)) is not None:
        raise SchemaViolation(f"banned reference: {b!r}")
    w0 = d.get("w0")
    if not isinstance(w0, (list, tuple)) or not w0:
        raise SchemaViolation("w0 is empty or not a list")
    try:
        w0 = [float(x) for x in w0]
    except (TypeError, ValueError) as e:
        raise SchemaViolation(f"a non-numeric value in w0: {e}") from None
    # ★ It must be **the same condition** as the Pydantic validator (§24 /
    #   D-26). Without it here, a budget overrun passes on the MockLLM path
    #   alone and the ablation breaks.
    if not all(abs(x) < 1e6 for x in w0):
        raise SchemaViolation("the w0 values are abnormally large")
    return RuleProposal(code=code, w0=w0, changes=str(d.get("changes", "")),
                        hypothesis_id=str(d.get("hypothesis_id", "")),
                        parent_ids=list(d.get("parent_ids", [])),
                        meta=dict(d.get("meta", {})))


# ---------------------------------------------------------------------------
# The Pydantic output schemas — used directly as Pydantic AI's
# `output_type`
# ---------------------------------------------------------------------------
# ★ There is no free-text parsing. The framework retries on a schema
#   violation, and once the cap is exceeded the candidate is **discarded**
#   (§26.4 — no partial acceptance).
#
# ⚠️ These models are **for the LLM boundary only**. The hot scoring path is
#    frozen dataclasses (§11.7) — they are constructed millions of times per
#    round, so no validation layer may sit there.

if HAVE_PYDANTIC:                                   # pragma: no branch

    class HypothesisOut(BaseModel):
        """One hypothesis. **No code** (§11.3).

        Asking for code alongside makes the root-cause analysis shallow and
        jumps straight to adding an `if`.
        """

        claim: str = Field(
            description="What is wrong and why. One or two sentences. No code")
        evidence_cases: list[int] = Field(
            default_factory=list,
            description="Case numbers this rests on. Do not leave empty — "
                        "it is the device that blocks unfounded generalities")
        affected_regime: str = Field(
            default="", description="Which regime (e.g. 'waves < 1')")
        measurable_with: list[str] = Field(
            default_factory=list,
            description="Names of existing features. Registered ones only")
        needs_new_feature: str | None = Field(
            default=None,
            description="If existing features cannot measure it, the name of that quantity. Otherwise null")
        proposed_direction: str = Field(
            default="", description="How to fix it. A direction, not code")
        risk: str = Field(
            default="",
            description="Which regime this fix could break. Always fill this in")

        @field_validator("claim")
        @classmethod
        def _no_code(cls, v: str) -> str:
            if "def " in v or "return " in v or "w[" in v:
                raise ValueError(
                    "Do not put code in a hypothesis. It must be prose "
                    "(§11.3)")
            # ★ Do not put a shape size in the sentence (D-114). `claim`
            #   goes to the RuleEditor whole through `json.dumps` — if "at
            #   M=4096" is in there, it can be copied straight across as a
            #   literal. `p.M > 1024` is blocked by the static checks, but
            #   **a hypothesis sentence does not go through them.**
            if _SHAPE_SIZE.search(v):
                raise ValueError(
                    "Do not put a shape size in a hypothesis (e.g. "
                    "'M=4096', '4096x4096'). Speak in regimes — 'shapes "
                    "where waves < 1', say. Copying a size across produces a "
                    "rule aimed at that one shape (§29.4). ★ An inequality "
                    "range such as 'small M (< 128)' is fine.")
            return v

        # ★ Checking `claim` alone leaves a place to leak from (D-117).
        #   The same check is applied to every field that actually goes to
        #   the RuleEditor.
        @field_validator("proposed_direction")
        @classmethod
        def _direction_no_shape_size(cls, v: str) -> str:
            if v and _SHAPE_SIZE.search(v):
                raise ValueError(
                    "Do not write a shape size in the proposed direction "
                    "(e.g. 'M=4096'). Speak in regimes — 'shapes where "
                    "waves < 1', say.")
            return v

    class AnalysisOutput(BaseModel):
        hypotheses: list[HypothesisOut] = Field(
            description=(f"★ Exactly {N_HYP_MIN}. Cover different failure "
                         "modes — one is assigned to each of this round's 3 "
                         "exploit proposals"
                         if N_HYP_MIN == N_HYP_MAX else
                         f"{N_HYP_MIN}~{N_HYP_MAX}. Cover different failure "
                         "modes"))

        @field_validator("hypotheses")
        @classmethod
        def _count(cls, v: list) -> list:
            if not N_HYP_MIN <= len(v) <= N_HYP_MAX:
                raise ValueError(
                    f"You gave {len(v)} hypotheses. "
                    + (f"Give exactly {N_HYP_MIN}"
                       if N_HYP_MIN == N_HYP_MAX
                       else f"Give {N_HYP_MIN}~{N_HYP_MAX}"))
            return v

    class RuleOutput(BaseModel):
        """One rule. ★ Not a diff — the **full code** (§11.6)."""

        # ⚠️ This schema is **shared by RuleEditor and RuleWriter.** Put
        #   talk of a parent into the description and RuleWriter goes looking
        #   for a parent that does not exist — the same reason
        #   `_rules_edit.md` was taken out of RuleWriter (§30.10). The
        #   replacement instruction is inserted per round by the RuleEditor
        #   prompt's `{parameters_note}`.
        code: str = Field(description=_desc_code())
        # ⚠️ It used to say "roughly is enough". After the §29 correction
        #   the prompt (`_rules_common.md`) says "give a starting point that
        #   reflects the physical magnitude of each term", and this
        #   description alone did not follow, so **the same request said the
        #   opposite of itself.** The objective is a step function, so it
        #   cannot escape the plateau near the starting point (D-54).
        w0: list[float] = Field(description=_desc_w0())
        # ★ For lineage tracking. **A rule is not thrown away for leaving
        #   it empty** — more required fields only raise the chance of
        #   burning the retries. If empty, a warning is recorded.
        changes: str = Field(
            default="", description="What changed from the parent. One sentence")
        hypothesis_id: str = Field(
            default="", description="Id of the hypothesis this reflects")

        @model_validator(mode="after")
        def _budget(self):
            """★ Literals and weights must be looked at **together**.

            Checked separately, "8 weights" and "1 literal" each pass and the
            sum is 9. Three RuleWriter proposals in a row really were
            discarded here, and the model never heard why.
            """
            if (m := literal_parameter_message(self.code, len(self.w0))):
                raise ValueError(m)
            return self

        @field_validator("code")
        @classmethod
        def _clean(cls, v: str) -> str:
            v = v.strip()
            if v.startswith("```"):
                v = "\n".join(ln for ln in v.split("\n")
                              if not ln.strip().startswith("```"))
            if "def score" not in v:
                raise ValueError("there is no `def score(f, p, hw, w):`")
            if (b := check_banned(v)) is not None:
                raise ValueError(
                    f"banned reference: {b!r}. A rule cannot see the table "
                    f"and cannot import (§3)")
            # ★ Reuse lived only in the static checks, so **no retry was
            #   triggered** — the proposal was silently discarded and the
            #   model never heard what was wrong. Lifted here, Pydantic AI
            #   feeds the message back and it gets fixed.
            if (m := weight_reuse_message(v)) is not None:
                raise ValueError(m)
            # ★ A term that silently does nothing — no exception, and it
            #   runs. Without blocking it here, one unit of budget is simply
            #   thrown away (§26.4).
            if (m := noop_term_message(v)) is not None:
                raise ValueError(m)
            # ★ The exponent-slot guard (D-112). **Always applied,
            #   regardless of the hint** — it is numerical safety, not a
            #   condition. Applying it here is what lets the model hear the
            #   reason and fix it.
            if (m := exponent_message(v)) is not None:
                raise ValueError(m)
            return v

        @field_validator("w0")
        @classmethod
        def _w0(cls, v: list[float]) -> list[float]:
            # ⚠️ 2026-09-10 (D-152): a `len(v) > 8` refusal lived here.
            #   D-150 removed the cap from the checker and D-151 from the
            #   field description, and **this branch survived both** — so
            #   every "no cap" measurement so far was taken with a live cap
            #   of 8. What is left is numerical safety only.
            if not v:
                raise ValueError("w0 is empty")
            if not all(abs(x) < 1e6 for x in v):
                raise ValueError("the w0 values are abnormally large")
            return v

    class FeatureOutput(BaseModel):
        name: str
        code: str
        rationale: str
        unit: str = "dimensionless"
        expected_range: tuple[float, float] = (0.0, 1.0)
        direction: str = "higher_is_worse"

    class Category(BaseModel):
        name: str = Field(description="lower case + underscores")
        description: str = Field(description="One sentence. What is wasted "
                                             "or constrained, by how much")

    class CategoryOutput(BaseModel):
        """★ How the LLM structures the physics (§30.10).

        Possibly a more interesting observation than the rediscovery count —
        it is material for comparing against a human partition. Recorded in
        `stage1-features/categories.json`.
        """

        categories: list[Category]
        notes: str = Field(default="", description="What you left out while partitioning")

    class CritiqueOutput(BaseModel):
        has_defect: bool
        defects: list[str] = Field(default_factory=list)
        measures_what: str = Field(
            description="If you found no defect, one sentence on the "
                        "physical quantity this function measures. **Being "
                        "unable to write it is itself a rejection signal** "
                        "(§11.5)")
        confidence: float = 0.5

else:                                               # pragma: no cover
    AnalysisOutput = _NoPydantic("AnalysisOutput")
    RuleOutput = _NoPydantic("RuleOutput")
    FeatureOutput = _NoPydantic("FeatureOutput")
    CritiqueOutput = _NoPydantic("CritiqueOutput")
    Category = _NoPydantic("Category")
    CategoryOutput = _NoPydantic("CategoryOutput")
    HypothesisOut = _NoPydantic("HypothesisOut")


@lru_cache(maxsize=16)
def rule_output_for(parameters: int | None = None, *,
                    product_hint: bool = False, power_hint: bool = False):
    """★ An output type with the budget baked into **the schema
    description and validation too** (D-107).

    `RuleOutput`'s field descriptions go straight to the model —
    `pydantic-ai` hands them over as the tool schema. That sentence was
    frozen at "★ at most 8 terms", so even when the prompt said "a cap of
    16", **the model produced 8.** All 29 rules across 3 seeds of the
    budget-16 campaign had 8 terms, and schema refusals were 0 across all 36
    rounds — the model never even tried.

    The **fourth** instance of the same spot: the checker (D-105) / the
    attached cap (D-106) / the prompt file / **the output schema**.
    """
    if not HAVE_PYDANTIC:                           # pragma: no cover
        return RuleOutput
    # ⚠️ 2026-09-10 (D-152): `parameters` decided a budget here. It decides
    #   nothing now — only whether a hint block is added — so the plain type
    #   is returned unless a hint is on. It stays in the signature because
    #   callers pass it and it is part of this function's cache key.
    del parameters
    if not product_hint and not power_hint:
        return RuleOutput

    class _BudgetedRuleOutput(RuleOutput):          # type: ignore[misc]
        code: str = Field(description=_desc_code(product_hint, power_hint))
        w0: list[float] = Field(description=_desc_w0())

        # ⚠️ 2026-09-10 (D-152): the subclass used to re-declare `_budget`
        #   and `_w0` to override the parent's cap of 8. The parent has no
        #   cap now, so the overrides are gone — the subclass differs from
        #   the parent **only in the hint sentences**.

    #: Keeps the type name the model sees — it is not a condition.
    _BudgetedRuleOutput.__name__ = "RuleOutput"
    _BudgetedRuleOutput.__qualname__ = "RuleOutput"
    return _BudgetedRuleOutput
