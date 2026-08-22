"""LLM 경계의 스키마 (§11.7).

**Pydantic 은 여기서만 쓴다.** 채점 뜨거운 경로(`core/types.py`)는 frozen
dataclass 다 — 라운드당 수백만 번 생성·해시되므로 검증 계층을 두면 한 자릿수
느려진다.

Pydantic 이 없어도 import 는 돼야 한다 (`[llm]` 선택 의존성). 없으면 얇은
dataclass 로 떨어지되 **검증이 없다는 사실을 명시**한다 — 조용히 통과하지
않는다 (§26.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kernelrule.rules.checks import (
    LIMITS,
    literal_budget_message,
    noop_term_message,
    weight_reuse_message,
)

__all__ = ["Hypothesis", "HypothesisSet", "FeatureProposal", "CritiqueOutput",
           "RuleProposal", "SchemaViolation", "validate_rule_proposal",
           "HAVE_PYDANTIC", "check_banned", "MAX_WEIGHTS",
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
#: 하한 2 — 1개면 다양성이 없고, 3개를 강제하면 억지 가설이 나온다.
N_HYP_MIN, N_HYP_MAX = 2, 8

#: 가중치 상한. **`rules.checks.LIMITS` 가 유일한 출처다** (D-26) —
#: 스키마와 정적 검사가 어긋나면 한쪽만 통과하는 규칙이 생긴다.
MAX_WEIGHTS = LIMITS["literal_budget"]


@dataclass
class Hypothesis:
    """자연어 문장. **실행 불가.** 코드를 같이 시키지 않는다 (§11.3)."""

    claim: str
    evidence_cases: list[int] = field(default_factory=list)
    affected_regime: str = ""
    measurable_with: list[str] = field(default_factory=list)
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


def validate_rule_proposal(obj: Any) -> RuleProposal:
    """LLM 응답 -> `RuleProposal`. **위반은 예외다. 고쳐서 쓰지 않는다.**"""
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
    if len(w0) > MAX_WEIGHTS:
        raise SchemaViolation(
            f"가중치 {len(w0)}개. 리터럴 예산이 {MAX_WEIGHTS}개다 — 숫자 "
            "리터럴과 합산된다 (§29.4)")
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
        """가설 하나. **코드를 쓰지 않는다** (§11.3).

        같이 시키면 원인 분석을 대충 하고 바로 `if` 를 추가한다.
        """

        claim: str = Field(
            description="무엇이 왜 잘못됐는가. 한두 문장. 코드 금지")
        evidence_cases: list[int] = Field(
            default_factory=list,
            description="근거가 된 사례 번호. 비우지 마라 — 근거 없는 "
                        "일반론을 막는 장치다")
        affected_regime: str = Field(
            default="", description="어느 체제인가 (예: 'waves < 1')")
        measurable_with: list[str] = Field(
            default_factory=list,
            description="기존 피처 이름들. 등록된 것만 쓴다")
        needs_new_feature: str | None = Field(
            default=None,
            description="기존 피처로 못 재면 그 물리량의 이름. 아니면 null")
        proposed_direction: str = Field(
            default="", description="어떻게 고칠지. 코드가 아니라 방향")
        risk: str = Field(
            default="",
            description="이 수정이 망가뜨릴 수 있는 구간. 반드시 채워라")

        @field_validator("claim")
        @classmethod
        def _no_code(cls, v: str) -> str:
            if "def " in v or "return " in v or "w[" in v:
                raise ValueError(
                    "가설에 코드를 쓰지 마라. 자연어 문장이어야 한다 (§11.3)")
            return v

    class AnalysisOutput(BaseModel):
        hypotheses: list[HypothesisOut] = Field(
            description=f"{N_HYP_MIN}~{N_HYP_MAX}개. 서로 다른 실패 모드를 "
                        "다뤄라")

        @field_validator("hypotheses")
        @classmethod
        def _count(cls, v: list) -> list:
            if not N_HYP_MIN <= len(v) <= N_HYP_MAX:
                raise ValueError(
                    f"가설이 {len(v)}개다. {N_HYP_MIN}~{N_HYP_MAX}개를 내라")
            return v

    class RuleOutput(BaseModel):
        """규칙 하나. ★ diff 가 아니라 **전체 코드**다 (§11.6)."""

        code: str = Field(
            description="`def score(f, p, hw, w):` 로 시작하는 함수 전문. "
                        "설명이나 마크다운 펜스를 넣지 마라. "
                        "★ 항은 최대 8개이고 각 w[i] 는 정확히 한 번만 쓸 수 "
                        "있다 — 하나의 가중치를 여러 항에 재사용해 항을 늘리면 "
                        "거부된다. 부모가 이미 8항이면 항을 추가하지 말고 "
                        "가장 덜 중요한 항 하나를 지우고 그 자리에 넣어라")
        w0: list[float] = Field(
            description="가중치 초기값. 대략적이면 충분하다 — 수치 "
                        "최적화기가 맞춘다. 길이는 코드가 참조하는 최대 "
                        "인덱스 + 1 이어야 한다. ★ 최대 8개. 숫자 리터럴과 "
                        "합산되므로 리터럴을 쓰면 그만큼 줄어든다")
        # ★ 계보 추적용이다. **비었다고 규칙을 버리지 않는다** — 필수
        #   필드가 많을수록 재시도 소진 확률만 올라간다. 비면 경고를 남긴다.
        changes: str = Field(
            default="", description="부모에서 무엇을 바꿨는가. 한 문장")
        hypothesis_id: str = Field(
            default="", description="반영한 가설 id")

        @model_validator(mode="after")
        def _budget(self):
            """★ 리터럴과 가중치를 **함께** 봐야 한다.

            둘을 따로 검사하면 "가중치 8개" 와 "리터럴 1개" 가 각각
            통과하고 합이 9가 된다. 실제로 Architect 제안 3개가 연속으로
            여기서 폐기됐고 모델은 이유를 듣지 못했다.
            """
            if (m := literal_budget_message(self.code, len(self.w0))):
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
            return v

        @field_validator("w0")
        @classmethod
        def _w0(cls, v: list[float]) -> list[float]:
            if not v:
                raise ValueError("w0 가 비었다")
            if len(v) > MAX_WEIGHTS:
                raise ValueError(
                    f"가중치 {len(v)}개. 리터럴 예산이 {MAX_WEIGHTS}개다 — "
                    "숫자 리터럴과 합산된다 (§29.4)")
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

    class CritiqueOutput(BaseModel):
        has_defect: bool
        defects: list[str] = Field(default_factory=list)
        measures_what: str = Field(
            description="결함을 못 찾았으면 이 함수가 재는 물리량을 한 "
                        "문장으로. **못 쓰면 그 자체가 거부 신호다** (§11.5)")
        confidence: float = 0.5

else:                                               # pragma: no cover
    AnalysisOutput = _NoPydantic("AnalysisOutput")
    RuleOutput = _NoPydantic("RuleOutput")
    FeatureOutput = _NoPydantic("FeatureOutput")
    CritiqueOutput = _NoPydantic("CritiqueOutput")
    HypothesisOut = _NoPydantic("HypothesisOut")


def rule_output_to_proposal(out) -> RuleProposal:
    """`RuleOutput` -> `RuleProposal`. 경계에서 한 번만 변환한다."""
    return validate_rule_proposal({"code": out.code, "w0": list(out.w0),
                                   "changes": out.changes,
                                   "hypothesis_id": out.hypothesis_id})
