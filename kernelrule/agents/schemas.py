"""LLM 경계의 스키마 (§11.7).

**Pydantic 은 여기서만 쓴다.** 채점 뜨거운 경로(`core/types.py`)는 frozen
dataclass 다 — 라운드당 수백만 번 생성·해시되므로 검증 계층을 두면 한 자릿수
느려진다.

Pydantic 이 없어도 import 는 돼야 한다 (`[llm]` 선택 의존성). 없으면 얇은
dataclass 로 떨어지되 **검증이 없다는 사실을 명시**한다 — 조용히 통과하지
않는다 (§26.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from kernelrule.rules.checks import (
    LIMITS,
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
    """LLM 응답이 스키마를 위반했다. **재시도 후 폐기**다 (§26.4).

    부분 수용하지 않는다 — 반쯤 맞는 규칙을 고쳐서 쓰면 그 규칙이 무엇을
    시험한 것인지 알 수 없어진다.
    """


class _NoPydantic:
    """Pydantic 부재를 **쓰려는 순간** 알린다 (§26.4 / 4-5).

    전에는 `AnalysisOutput = None` 이었다. `output_type=None` 을 Pydantic AI
    에 넘기면 저 아래에서 `AttributeError` 가 나고, 그 메시지만 보고는
    **검증이 통째로 꺼졌다는 사실을 못 읽는다.** 조용히 나쁜 상태로 굴러가지
    않는다.

    ★ Pydantic 이 **있어도** 정의된다 — 그래야 이 동작을 시험할 수 있다.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def _die(self, *_a, **_k):
        raise ImportError(
            f"{self._name} 를 쓰려면 Pydantic 이 필요하다. LLM 경계의 검증이 "
            "**비활성화된 상태**다 — 스키마 위반이 걸러지지 않는다. "
            "`pip install -e '.[llm]'` 로 설치하라 (§26.4)")

    __call__ = _die
    __getattr__ = _die


#: 규칙 코드에 나타나면 즉시 거부. `rules/checks.py` 가 AST 로 다시 본다.
#: **문자열 검사는 우회 가능하므로 구조적 방어와 병행한다** (§11.7).
BANNED_SUBSTRINGS = ("time_ms", "cublas_ms", "difficulty", "tflops",
                     "distinct_time_frac", "import ", "open(", "TABLE",
                     "__globals__", "eval(", "exec(", "np.random")


def _code_only(src: str) -> str:
    """주석과 문자열 리터럴을 뺀 토큰만 잇는다 (D-27).

    ★ 부분 문자열 매칭이 **주석을 잡는 것**을 막는다. LLM 이 "이 형상은
    난이도(difficulty)가 높으니" 라고 주석에 쓰면 코드가 멀쩡한데도
    거부됐다 — 그러면 재시도만 소진하고 무엇이 틀렸는지도 알려주지
    못한다.

    ★ 검사를 **약화시키는 것이 아니다**. `rules/checks.py` 가 AST 로
    이름·호출·import 를 다시 보고, 샌드박스가 실행을 격리한다 (§11.7).
    주석 안의 `import ` 는 실행되지 않으므로 여기서 잡을 이유가 없다.

    토큰화가 실패하면(문법 오류) **원본을 그대로 돌려준다** — 검사를
    건너뛰지 않는다 (§26.4).
    """
    import io
    import tokenize
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src                      # 파싱 불가 -> 보수적으로 원본 검사
    return " ".join(t.string for t in toks
                    if t.type not in (tokenize.COMMENT, tokenize.STRING))


def check_banned(code: str) -> str | None:
    """금지어를 찾으면 그 문자열을, 없으면 `None`. **두 경로가 공유한다.**"""
    probe = _code_only(code)
    for b in BANNED_SUBSTRINGS:
        if b in probe:
            return b
    return None


#: ★ 가설 개수의 **유일한 출처** (§30.8 / D-26).
#:
#: 설명·검증·에러 메시지가 셋 다 달랐다 — 설명은 "3~5", 검증은 `1 <= n <= 8`,
#: 에러는 다시 "3~5". 1개만 내도 통과했고, 그러면 그 라운드의 규칙이 **전부
#: 같은 가설**을 반영해 §14.2 의 다양성이 무너진다.
#:
#: ★ 2026-09-08 (D-144): **3 으로 고정**한다.
#:
#: 트레이스 실측 — exploit 중복 77건 중 **67건(87%)이 "같은 부모 + 같은
#: 가설"** 이었다. 제안 12(exploit 6)에 가설 3~5 였으므로 exploit 자리가
#: 가설보다 많아 **반드시 겹쳤다.** 제안 6(exploit 3) + 가설 3 이면 딱 맞는다.
#:
#: 옛 값 이력: 설명 "3~5" / 검증 `1<=n<=8` / 에러 "3~5" 로 셋이 달랐고
#: (D-26 이 정리), 그 뒤 `2, 8` 이었다.
N_HYP_MIN, N_HYP_MAX = 3, 3

#: 가중치 상한. **`rules.checks.LIMITS` 가 유일한 출처다** (D-26) —
#: 스키마와 정적 검사가 어긋나면 한쪽만 통과하는 규칙이 생긴다.
#: ★ 가설 문장에 들어가면 안 되는 **형상 크기** (D-114).
#: `4096x4096` 같은 곱셈 표기와 `M = 4096` 같은 지목을 잡는다. 세 자리
#: 미만은 안 잡는다 — `stages=3` 같은 config 값을 오탐한다.
_SHAPE_SIZE = re.compile(r"\d{3,6}\s*[x*×]\s*\d{3,6}"
                         r"|\b[MNK]\s*=\s*\d{3,6}\b")

MAX_WEIGHTS = LIMITS["parameters"]


#: ★ 실험 B (D-110). 스키마도 프롬프트와 같은 말을 해야 한다 (D-107).
_PRODUCT_DESC = (" ★ Within one term you **may multiply two features** — "
                 "`(f.a * f.b) * w[i]` costs one parameter.")

#: ★ 실험 (b) (D-112). 가중치를 **지수 자리**에 둘 수 있다.
_POWER_DESC = (" ★ A weight may sit **in the exponent** — replacing "
               "`f.a * w[i]` with `np.power(f.a, w[i])` costs no extra room "
               "and fits the exponent. The base must be a single "
               "`f.<name>`, and the exponent weight is bounded to 0~4.")


def _desc_code(b: int, product: bool = False,
               power: bool = False) -> str:
    return ("The full function, starting at `def score(f, p, hw, w):`. "
            "No prose, no markdown fences. "
            f"★ At most {b} parameters **per execution path**, and each w[i] "
            "may be used exactly once — reusing one weight across terms to "
            "add terms is rejected. "
            "★ Comparison constants in branch conditions "
            "(`p.roofline_ratio < 1`) do not count — write physical "
            "boundaries as plain numbers"
            + (_PRODUCT_DESC if product else "")
            + (_POWER_DESC if power else ""))


def _desc_w0(b: int) -> str:
    return ("Initial weights. ★ Do not give them carelessly — the objective "
            "is a step function and the optimiser can get stuck on a plateau "
            "near the starting point. Give a **starting point that reflects "
            "the physical magnitude of each term**. The length must equal the "
            f"largest index the code references + 1. ★ At most {b} **per "
            "execution path**, summed with numeric literals, so literals "
            "reduce it — except comparison constants in branch conditions")


def _w0_message(n: int, b: int) -> str:
    return (f"가중치 {n}개. 예산이 {b}개다 — 숫자 리터럴과 합산된다. "
            "단 **분기 조건의 비교 상수는 빠진다** (§29.4 / D-78)")


@dataclass
class Hypothesis:
    """자연어 문장. **실행 불가.** 코드를 같이 시키지 않는다 (§11.3)."""

    claim: str
    evidence_cases: list[int] = field(default_factory=list)
    affected_regime: str = ""
    measurable_with: list[str] = field(default_factory=list)
    #: ★ 없는 축을 요구하는 자리. 이것만 FeatureWriter 에게 전달된다 —
    #: 진단 리포트는 안 간다 (D-75).
    #:
    #: ⚠️ 2026-08-28 에 `physical_requirement` 로 바꿨다가 **되돌렸다**
    #: (D-81). 기준선 17.9% 가 이 이름과 이 설명으로 측정됐고, 바꾼 채로
    #: 비교하면 두 변수가 다르다. `loop._requirement_of` 는 두 이름을 다
    #: 읽으므로 그 사이에 만들어진 실행도 그대로 읽힌다.
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
    """★ 결함을 못 찾으면 **물리량을 한 문장으로** 쓰게 한다 (§11.5).

    설명을 못 쓰면 그 자체가 거부 신호다.
    """

    has_defect: bool
    defects: list[str] = field(default_factory=list)
    measures_what: str = ""
    confidence: float = 0.5


@dataclass
class RuleProposal:
    """★ diff 가 아니라 **전체 코드**를 받는다 (§11.6).

    diff 는 적용 실패가 잦고 재시도 비용이 크다.
    """

    code: str
    w0: list[float]
    changes: str = ""
    hypothesis_id: str = ""
    parent_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def validate_rule_proposal(obj: Any, *, parameters: int | None = None
                           ) -> RuleProposal:
    """LLM 응답 -> `RuleProposal`. **위반은 예외다. 고쳐서 쓰지 않는다.**

    ★ `parameters` 를 안 주면 `LIMITS["parameters"]`(8) 이다. `--parameters` 를
    쓰는 경로는 **반드시 넘겨야 한다** — 안 넘기면 16항 제안이 여기서
    조용히 거부되고, 실험은 "예산 16 이 효과 없다" 를 재게 된다 (D-107).
    """
    _b = int(parameters if parameters is not None else MAX_WEIGHTS)
    if isinstance(obj, RuleProposal):
        d = {"code": obj.code, "w0": obj.w0, "changes": obj.changes,
             "hypothesis_id": obj.hypothesis_id, "parent_ids": obj.parent_ids,
             "meta": obj.meta}
    elif isinstance(obj, dict):
        d = dict(obj)
    else:
        raise SchemaViolation(f"규칙 제안이 dict 가 아니다: {type(obj)}")

    code = d.get("code")
    if not isinstance(code, str) or "def score" not in code:
        raise SchemaViolation("code 에 `def score(f, p, hw, w)` 가 없다")
    if (b := check_banned(code)) is not None:
        raise SchemaViolation(f"금지된 참조: {b!r}")
    w0 = d.get("w0")
    if not isinstance(w0, (list, tuple)) or not w0:
        raise SchemaViolation("w0 가 비어 있거나 리스트가 아니다")
    try:
        w0 = [float(x) for x in w0]
    except (TypeError, ValueError) as e:
        raise SchemaViolation(f"w0 에 숫자가 아닌 값: {e}") from None
    # ★ Pydantic validator 와 **같은 조건**이어야 한다 (§24 / D-26). 여기에
    #   없으면 MockLLM 경로에서만 예산 초과가 통과해 ablation 이 깨진다.
    if len(w0) > _b:
        raise SchemaViolation(_w0_message(len(w0), _b))
    if not all(abs(x) < 1e6 for x in w0):
        raise SchemaViolation("w0 값이 비정상적으로 크다")
    return RuleProposal(code=code, w0=w0, changes=str(d.get("changes", "")),
                        hypothesis_id=str(d.get("hypothesis_id", "")),
                        parent_ids=list(d.get("parent_ids", [])),
                        meta=dict(d.get("meta", {})))


# ---------------------------------------------------------------------------
# Pydantic 출력 스키마 — Pydantic AI 의 `output_type` 으로 그대로 쓴다
# ---------------------------------------------------------------------------
# ★ 자유 텍스트 파싱을 하지 않는다. 스키마 위반은 프레임워크가 재시도시키고,
#   상한을 넘으면 그 후보를 **폐기**한다 (§26.4 — 부분 수용 금지).
#
# ⚠️ 이 모델들은 **LLM 경계 전용**이다. 채점 뜨거운 경로는 frozen dataclass
#    다 (§11.7) — 라운드당 수백만 번 생성되므로 검증 계층을 두면 안 된다.

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
            # ★ 형상 크기를 문장에 담지 마라 (D-114). `claim` 은
            #   `json.dumps` 로 RuleEditor 에 통째로 간다 — "M=4096 에서"
            #   가 거기 있으면 그것을 그대로 리터럴로 옮겨 적을 수 있다.
            #   `p.M > 1024` 는 정적 검사가 막지만 **가설 문장은 안 거친다.**
            if _SHAPE_SIZE.search(v):
                raise ValueError(
                    "Do not put a shape size in a hypothesis (e.g. "
                    "'M=4096', '4096x4096'). Speak in regimes — 'shapes "
                    "where waves < 1', say. Copying a size across produces a "
                    "rule aimed at that one shape (§29.4). ★ An inequality "
                    "range such as 'small M (< 128)' is fine.")
            return v

        # ★ `claim` 만 검사하면 새는 자리가 남는다 (D-117). RuleEditor 에
        #   실제로 가는 필드 전부에 같은 검사를 건다.
        @field_validator("proposed_direction")
        @classmethod
        def _direction_no_shape_size(cls, v: str) -> str:
            if v and _SHAPE_SIZE.search(v):
                raise ValueError(
                    "제안 방향에 형상 크기를 쓰지 마라 (예: 'M=4096'). "
                    "체제로 말하라 — 'waves < 1 인 형상' 처럼.")
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

        # ⚠️ 이 스키마는 **RuleEditor 와 RuleWriter 가 함께 쓴다.** 설명에
        #   부모 이야기를 넣으면 RuleWriter 가 없는 부모를 찾는다 —
        #   `_rules_edit.md` 를 RuleWriter 에서 뺀 이유와 같다 (§30.10).
        #   교체 지시는 RuleEditor 프롬프트의 `{parameters_note}` 가 라운드마다
        #   동적으로 넣는다.
        code: str = Field(description=_desc_code(MAX_WEIGHTS))
        # ⚠️ "대략적이면 충분하다" 였다. 프롬프트(`_rules_common.md`)는
        #   §29 정정 뒤 "각 항의 물리적 크기를 반영한 출발점을 주라" 인데
        #   이 설명만 안 따라와서 **같은 요청 안에서 반대를 말하고 있었다.**
        #   목적함수가 계단이라 출발점 근처 평지에서 못 빠져나온다 (D-54).
        w0: list[float] = Field(description=_desc_w0(MAX_WEIGHTS))
        # ★ 계보 추적용이다. **비었다고 규칙을 버리지 않는다** — 필수
        #   필드가 많을수록 재시도 소진 확률만 올라간다. 비면 경고를 남긴다.
        changes: str = Field(
            default="", description="What changed from the parent. One sentence")
        hypothesis_id: str = Field(
            default="", description="Id of the hypothesis this reflects")

        @model_validator(mode="after")
        def _budget(self):
            """★ 리터럴과 가중치를 **함께** 봐야 한다.

            둘을 따로 검사하면 "가중치 8개" 와 "리터럴 1개" 가 각각
            통과하고 합이 9가 된다. 실제로 RuleWriter 제안 3개가 연속으로
            여기서 폐기됐고 모델은 이유를 듣지 못했다.
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
                raise ValueError("`def score(f, p, hw, w):` 가 없다")
            if (b := check_banned(v)) is not None:
                raise ValueError(
                    f"금지된 참조: {b!r}. 규칙은 표를 볼 수 없고 "
                    "import 도 못 한다 (§3)")
            # ★ 재사용은 정적 검사에만 있어서 **재시도가 안 걸렸다** —
            #   제안이 조용히 폐기되고 모델은 무엇이 틀렸는지 못 들었다.
            #   여기로 올리면 Pydantic AI 가 메시지를 되먹여 고치게 한다.
            if (m := weight_reuse_message(v)) is not None:
                raise ValueError(m)
            # ★ 조용히 아무 일도 하지 않는 항 — 예외도 안 나고 실행도 된다.
            #   여기서 막지 않으면 예산 하나가 그냥 버려진다 (§26.4).
            if (m := noop_term_message(v)) is not None:
                raise ValueError(m)
            # ★ 지수 자리 가드 (D-112). **힌트와 무관하게 항상 건다** —
            #   조건이 아니라 수치 안전이다. 여기서 걸어야 모델이 이유를
            #   듣고 고쳐 낸다.
            if (m := exponent_message(v)) is not None:
                raise ValueError(m)
            return v

        @field_validator("w0")
        @classmethod
        def _w0(cls, v: list[float]) -> list[float]:
            if not v:
                raise ValueError("w0 가 비었다")
            if len(v) > MAX_WEIGHTS:
                raise ValueError(_w0_message(len(v), MAX_WEIGHTS))
            if not all(abs(x) < 1e6 for x in v):
                raise ValueError("w0 값이 비정상적으로 크다")
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


def rule_output_to_proposal(out, *, parameters: int | None = None
                            ) -> RuleProposal:
    """`RuleOutput` -> `RuleProposal`. 경계에서 한 번만 변환한다."""
    return validate_rule_proposal({"code": out.code, "w0": list(out.w0),
                                   "changes": out.changes,
                                   "hypothesis_id": out.hypothesis_id},
                                  parameters=parameters)


@lru_cache(maxsize=16)
def rule_output_for(parameters: int | None = None, *,
                    product_hint: bool = False, power_hint: bool = False):
    """★ 예산이 **스키마 설명과 검증에도** 들어간 출력 타입 (D-107).

    `RuleOutput` 의 필드 설명은 모델에게 그대로 간다 — `pydantic-ai` 가
    도구 스키마로 넘긴다. 그 문장이 "★ 항은 최대 8개" 로 굳어 있어서,
    프롬프트가 "상한 16개" 라고 말해도 **모델은 8개를 냈다.** 예산 16
    캠페인 3시드 29개 규칙이 전부 8항이었고 스키마 거부는 36라운드
    내내 0 이었다 — 모델은 시도조차 하지 않았다.

    같은 자리 **네 번째**다: 검사기(D-105) / 딸린 상한(D-106) /
    프롬프트 파일 / **출력 스키마**.
    """
    if not HAVE_PYDANTIC:                           # pragma: no cover
        return RuleOutput
    b = int(parameters if parameters is not None else MAX_WEIGHTS)
    if b == MAX_WEIGHTS and not product_hint and not power_hint:
        return RuleOutput

    class _BudgetedRuleOutput(RuleOutput):          # type: ignore[misc]
        code: str = Field(description=_desc_code(b, product_hint, power_hint))
        w0: list[float] = Field(description=_desc_w0(b))

        # ★ 이름을 부모와 **같게** 둔다. pydantic 은 데코레이터를 이름으로
        #   모으므로 같은 이름이어야 부모 것을 **대체**한다. 다른 이름을
        #   쓰면 부모의 8 검사가 그대로 남아 둘 다 돈다.
        @model_validator(mode="after")
        def _budget(self):
            if (m := literal_parameter_message(self.code, len(self.w0),
                                            parameters=b)):
                raise ValueError(m)
            return self

        @field_validator("w0")
        @classmethod
        def _w0(cls, v: list[float]) -> list[float]:
            if not v:
                raise ValueError("w0 가 비었다")
            if len(v) > b:
                raise ValueError(_w0_message(len(v), b))
            if not all(abs(x) < 1e6 for x in v):
                raise ValueError("w0 값이 비정상적으로 크다")
            return v

    #: 모델에 보이는 타입 이름을 유지한다 — 조건이 아니다.
    _BudgetedRuleOutput.__name__ = "RuleOutput"
    _BudgetedRuleOutput.__qualname__ = "RuleOutput"
    return _BudgetedRuleOutput
