"""실제 LLM 클라이언트 — Pydantic AI + OpenAI (§4-0).

## `LLMClient` Protocol 뒤에 둔다

`MockLLM` 과 **교체 가능**해야 ablation 과 `replay` 가 성립한다.
`pydantic_ai.Agent` 를 호출부에 노출하지 않는다 — 루프는 `complete(role,
prompt, **kw)` 만 안다.

## 스키마 위반은 재시도 후 **폐기**다

Pydantic AI 가 validator 실패를 모델에 되먹여 재시도한다. 상한(`retries`)
을 넘으면 그 후보를 버린다. **부분 수용하지 않는다** (§26.4) — 반쯤 맞는
규칙을 고쳐서 쓰면 그 규칙이 무엇을 시험한 것인지 알 수 없어진다.

## 키

`OPENAI_API_KEY` 환경변수에서만 읽는다. **코드에 넣지 않고, 저장하지도
않는다.** 없으면 명확한 에러로 중단한다 — `MockLLM` 으로 조용히
폴백하지 않는다 (§26.4).
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

#: ★ 모델의 **유일한 출처**. 실험 스크립트가 각자 상수를 들고 있다가
#: 서로 다른 모델로 도는 일이 있었다 — 그러면 결과를 나란히 놓을 수 없다
#: (D-31). 바꿀 때는 여기 하나만 고치고, **사용자가 지시했을 때만** 바꾼다.
DEFAULT_MODEL = "gpt-5.6-luna"


class MissingAPIKey(RuntimeError):
    """API 키가 없다. **`MockLLM` 으로 폴백하지 않는다** (§26.4)."""


class BudgetExceeded(RuntimeError):
    """예산 상한을 넘었다. 실행을 멈춘다."""


#: validator 메시지 -> 짧은 사유 코드. `llm 132건` 으로 뭉뚱그리면
#: 무엇이 걸렸는지 모르고, 프롬프트를 어디를 고쳐야 할지도 모른다.
_VIOLATION_PATTERNS: tuple[tuple[str, str], ...] = (
    # ⚠️ 패턴은 **실제 validator 메시지**와 맞춰야 한다.
    #    "가중치 8개" 로 뒀다가 "가중치 9개..." 를 못 잡아 other 로 샜다.
    ("리터럴 예산이", "w0_too_long"),
    ("최대 8개", "w0_too_long"),
    ("재사용", "weight_reuse"),
    ("최대 인덱스", "w0_length_mismatch"),
    ("w0 가 비었다", "w0_empty"),
    ("비정상적으로 크다", "w0_huge"),
    ("금지된 참조", "banned_substring"),
    ("def score", "no_def_score"),
    ("가설에 코드", "hypothesis_has_code"),
    ("가설이", "hypothesis_count"),
    ("Exceeded maximum", "retries_exhausted"),
    ("event loop", "event_loop_bug"),
    ("rate limit", "rate_limit"),
)


def classify_violation(msg: str) -> str:
    """예외/validator 메시지를 사유 코드로 분류한다."""
    for pat, code in _VIOLATION_PATTERNS:
        if pat.lower() in msg.lower():
            return code
    return "other"


#: 피처 조건. `F1-K` 는 **공개 지식 다섯**으로 시작한다 (§30.17).
_CONDITIONS = frozenset({"F0", "F1", "F1-K", "F2", "F3"})

#: 조건 -> **피처** 예시 파일. 답을 건네지 않아야 하는 조건은 무관 도메인.
_EXAMPLES = {"F0": "other_domain", "F1": "other_domain",
             "F1-K": "known5", "F2": "known5", "F3": "known5"}

#: `examples/rule_known.md` 가 **이름으로 부르는** 피처들.
#: 이 넷이 레지스트리에 다 있어야 그 예시를 쓸 수 있다 (§30.20).
_RULE_EXAMPLE_NEEDS = ("tail_waste", "has_spill", "occupancy_deficit",
                       "roofline_ratio")


def _rule_example_for(registry, *, budget: int | None = None) -> str:
    """규칙 예시를 **레지스트리를 보고** 고른다 (§30.20).

    RuleWriter 의 `condition` 은 A/B(표 관측 유무)라 피처 조건과 축이
    다르다. 그래서 조건이 아니라 **예시가 쓰는 이름이 레지스트리에
    있는가**로 정한다.

    ```
    다 있다   실제 이름을 써도 추가 누출이 아니다 — 이미 목록에 있다
    없다      ★ 무관 도메인. 없는 이름을 예시로 주면 물리를 지목한다 (D-35)
    ```

    조건 이름을 키로 쓰면 새 조건이 생길 때마다 표를 고쳐야 하고,
    빠뜨리면 조용히 누출된다. **레지스트리를 보면 빠뜨릴 수 없다.**
    """
    names = set(getattr(registry, "_items", {}) or {})
    ok = names.issuperset(_RULE_EXAMPLE_NEEDS)
    return load_prompt(
        f"examples/{'rule_known' if ok else 'rule_other_domain'}.md",
        budget=budget)

#: 하드웨어 사실(`hw/*.md`)을 받는 역할. **RuleWriter 뿐이다.**
#:
#:   RuleEditor / FeatureWriter   피처가 hw 를 이미 흡수했다. 안 보면 그
#:                               프롬프트는 GPU 무관해진다 (§16.2)
#:   Analyst                     ★ 진단 리포트 **블록 1 이 같은 사실**이다.
#:                               리포트는 표에서 매번 생성되고 `hw/*.md` 는
#:                               고정이라, 번들이 바뀌면 둘이 갈려 모순된
#:                               사실을 받는다. 살아 있는 쪽을 남긴다 (원칙 2)
#:   RuleWriter                   리포트를 안 받으므로 여기서 받아야 한다
#: ★ 목표 정의 (D-101). **RuleEditor 만** 받는다 — RuleWriter 는 "점수
#: 없음" 이므로 이 절 자체를 안 본다 (§30.10). 그래서 목적함수를 바꿔도
#: RuleWriter 의 조건은 안 바뀐다.
_OBJECTIVE_BLOCKS = {
    "regret": """`regret` = (규칙이 1등으로 고른 config 의 시간) / (그 형상의 전수 최적 시간).
1.0 이 완벽입니다. 낮을수록 좋습니다.""",
    "rank": """★ **1등 하나가 아니라 상위권의 순서**로 채점합니다.

각 형상에서 실제로 가장 빠른 config 100개를 놓고, 그 안의 모든 쌍 (i, j) 에
대해 **더 빠른 쪽에 더 낮은 점수**를 주었는지 봅니다. 쌍마다 두 시간의
차이만큼 무게가 붙습니다 — 차이가 큰 쌍을 뒤집으면 손해가 큽니다.

0 이 완벽입니다. 낮을수록 좋습니다.

⚠️ 측정 노이즈로 구별할 수 없는 쌍은 채점에서 빠집니다. **미세한 차이를
맞추려 하지 말고 순서를 만드는 물리를 쓰세요.**""",
}
def assemble_instructions(role: str, *, objective: str = "rank",
                          hw_file: str = "hw/sm_86.md",
                          body: str | None = None,
                          budget: int | None = None) -> str:
    """★ 시스템 프롬프트 조립. **한 곳에서만 한다** (원칙 2).

    전에는 `_agent()` 와 `tests/test_prompt_layout.py` 가 각자 조립했다.
    `{objective_block}` 을 넣자 **시험 쪽만 안 채워져서** 갈렸다 —
    "같은 판정이 여러 곳에 있으면 갈린다" 의 여덟 번째다.
    """
    if objective not in _OBJECTIVE_BLOCKS:
        raise ValueError(f"알 수 없는 목적함수: {objective!r}")
    parts = [load_prompt("_base.md", budget=budget)]
    if role in _NEEDS_HW:
        parts.append(load_prompt(hw_file, budget=budget))
    if role in _WRITES_RULES:
        parts.append(load_prompt("role/_rules_common.md", budget=budget))
    if role in _EDITS_RULES:
        parts.append(load_prompt("role/_rules_edit.md", budget=budget).replace(
            "{objective_block}", _OBJECTIVE_BLOCKS[objective]))
    parts.append(body if body is not None
                 else load_prompt(f"role/{role}.md", budget=budget))
    return "\n\n---\n\n".join(parts)


_NEEDS_HW = frozenset({"rule_writer"})

#: 규칙 **함수**를 쓰는 역할 — 형태·예산·벡터화 제약을 공유한다.
_WRITES_RULES = frozenset({"rule_editor", "rule_writer"})

#: 부모 규칙을 **고치는** 역할. regret 정의와 거부 사례 갤러리를 받는다.
#: ★ RuleWriter 는 안 받는다 — 백지에서 쓰므로 교체할 항도 이전 점수도
#: 없고, 주면 자기 역할 파일의 "점수 없음" 과 정면으로 모순된다.
_EDITS_RULES = frozenset({"rule_editor"})


#: 프롬프트 파일의 **내부 메모**. 사람이 읽으라고 쓴 것이고 모델에 보내지
#: 않는다 — `§30.18`, `D-45` 같은 내부 참조가 그대로 나가고 있었다.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def load_prompt(name: str, *, budget: int | None = None) -> str:
    """프롬프트를 읽는다. **HTML 주석은 걷어내고 `{budget}` 를 채운다.**

    `<!-- ... -->` 는 "왜 이렇게 썼나" 를 남기는 자리다. 그것이 모델에
    가면 (1) 토큰을 쓰고 (2) 내부 결정 번호가 새고 (3) 조건에 따라서는
    답을 건네줄 수도 있다.

    ★ 예산 숫자는 **프롬프트에 직접 쓰지 않는다.** `{budget}` 로 쓰면
    여기서 `checks.BUDGET` 으로 채워진다. 프롬프트 다섯 파일과 스키마와
    검사기가 각자 숫자를 적고 있었고, 그러면 바꿀 때 하나를 빠뜨린다
    (`is_reference` / `top_k` / `DEFAULT_MODEL` / `REGISTRY` /
    `load_generated` 에 이은 여섯 번째가 된다).
    """
    from kernelrule.rules.checks import BUDGET  # 호출 시점에 읽는다

    p = _PROMPTS / name
    if not p.exists():
        raise FileNotFoundError(f"프롬프트가 없다: {p}")
    txt = _HTML_COMMENT.sub("", p.read_text()).strip() + "\n"
    return txt.replace("{budget}", str(budget if budget is not None
                                       else BUDGET))


@dataclass
class LLMConfig:
    """`config.json` 에 그대로 기록된다 (§15.4 재현성)."""

    model: str = DEFAULT_MODEL
    #: ★ 목표 정의를 정한다 (D-101). `config.json` 에 남아야 조건이 기록된다.
    #: RuleEditor 의 "채점 방식" 절만 바뀐다 — RuleWriter 는 안 받는다.
    objective: str = "regret"
    #: ★ 항 예산 (D-104). `None` 이면 `checks.BUDGET`(8). 프롬프트의
    #: `{budget}` 이 이 값으로 채워진다 — 검사기와 갈리면 안 되므로
    #: 루프가 같은 값을 `check_rule(limits=...)` 에도 넘긴다.
    rule_budget: int | None = None
    # ------------------------------------------------------------------
    # ★ temperature / seed — 둘 다 `None` 이다. 통제할 수 없다 (D-47)
    # ------------------------------------------------------------------
    # 전에는 0.7 / 20260821 로 두고 "다양성을 통제하고 재현성을 확보한다" 고
    # 적었다. **둘 다 모델에 전달되지 않고 있었다.** `config.json` 에는
    # 값이 적히는데 실제로는 모델 기본값으로 돌았다 (§30.8).
    #
    # 실측으로 층을 갈랐다 (3모델 x 2엔드포인트):
    #
    #   seed         Responses 엔드포인트에 **파라미터 자체가 없다.**
    #                SDK 가 `TypeError` 를 낸다 — 요청이 나가지도 않는다.
    #                모델과 무관하다 (gpt-4.1-mini 도 마찬가지).
    #   temperature  **추론 모델이 거부한다.** gpt-5.6-luna 는 두 엔드포인트
    #                모두 400 이고, gpt-5.4-mini / gpt-4.1-mini 는 둘 다 된다.
    #                엔드포인트와 무관하다.
    #                (추론 모델은 내부적으로 여러 차례 추론·검증·선택을
    #                 거치므로 샘플링을 막는다. 대신 reasoning_effort 를 준다.)
    #
    # pydantic-ai 는 둘 다 **조용히 버린다** — 그래서 몰랐다. 원인이 아니라
    # 증상을 감춘 쪽이다.
    #
    # 따라서 기본을 `None` 으로 둔다. 값을 넣으면 보내되, **보낼 수 없는
    # 조합이면 예외를 낸다** — 조용히 버려지느니 멈추는 편이 낫다 (§26.4).
    temperature: float | None = None
    seed: int | None = None
    #: 스키마 위반 시 재시도 상한. 2 -> 3 (임시).
    #: 지시 모순이 해소되면 필요 없지만, 거부율이 여전히 높을 때
    #: "모델이 배우는 중" 과 "구조적으로 불가능" 을 구분해 준다.
    max_retries: int = 3
    #: 동시 호출 상한. rate limit 에 걸리면 줄이되 **로그에 남긴다.**
    concurrency: int = 6
    arch_prompt: str = "hw/sm_86.md"
    #: ★ OpenAI 엔드포인트. **`config.json` 에 남는다** — 섞이면 비교가
    #: 깨지므로 나중에 확인할 수 있어야 한다 (D-31, D-44).
    #:
    #:   "responses"  /v1/responses.  구조화 출력 + 추론을 함께 쓸 수 있다
    #:   "chat"       /v1/chat/completions.  전통적 형식
    #:
    #: gpt-5.6 계열은 chat 에서 **함수 도구 + reasoning_effort** 조합을
    #: 400 으로 막는다. 구조화 출력(`output_type`)이 함수 도구로 구현되므로
    #: 그대로 걸린다. 우회는 `reasoning_effort='none'` 인데 그러면 추론이
    #: 꺼져 물리 유도 능력을 잃는다 — 그래서 엔드포인트를 옮겼다.
    #: gpt-5.4 / 5.4-mini 도 responses 를 지원하므로 통일이 가능하다.
    endpoint: str = "responses"
    #: ★ 추론 강도. **명시한다** — 안 하면 모델 기본값이 적용되고, 그 기본이
    #: 바뀌면 우리 결과가 조용히 달라진다 (§15.4 재현성).
    #:
    #: 실측 (gpt-5.6-luna, Responses API):
    #:   none    추론 0 토큰
    #:   low     ~150
    #:   medium  ~130   ← 채택
    #:   high    ~520
    #: 우리 실제 프롬프트(7,177토큰)에서는 기본값이 1,756 추론토큰을 썼다 —
    #: 과제가 무거우면 그만큼 더 쓴다.
    #:
    #: `None` 이면 보내지 않는다(모델 기본값). 그 경우도 **의도한 것임을
    #: 기록으로 남기려면** 명시적으로 None 을 적어야 한다.
    reasoning_effort: str | None = "medium"
    #: ★ 피처를 어떻게 보여주는가. **실험 조건이므로 기록한다** (D-31).
    #:
    #:   "full"   이름 + 범위 + 물리적 의미 + 왜 중요한가  (지금 기본)
    #:   "names"  이름만 — 2026-08-22 이전 상태
    #:
    #: 이 둘의 차이가 이 저장소에서 시드 폭을 넘은 유일한 효과였는데,
    #: 그 측정이 **임의로 바꾼 모델**에서 나온 것이라 다시 잰다 (D-52).
    #: 플래그로 둔 이유는 코드를 되돌렸다 돌렸다 하면 어느 실행이 어느
    #: 조건이었는지 알 수 없게 되기 때문이다.
    feature_detail: str = "full"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Budget:
    """호출 수와 토큰 상한. **넘으면 멈춘다** (§4-1)."""

    max_calls: int = 400
    max_input_tokens: int = 3_000_000
    max_output_tokens: int = 600_000
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_hits: int = 0
    #: ★ 실패한 호출도 토큰을 쓴다. 안 세면 예산 감시에 구멍이 생긴다.
    failed_calls: int = 0

    def charge(self, n_in: int, n_out: int) -> None:
        self.calls += 1
        self.input_tokens += n_in
        self.output_tokens += n_out
        if self.calls > self.max_calls:
            raise BudgetExceeded(
                f"호출 {self.calls} > 상한 {self.max_calls}")
        if self.input_tokens > self.max_input_tokens:
            raise BudgetExceeded(
                f"입력 토큰 {self.input_tokens:,} > 상한 "
                f"{self.max_input_tokens:,}")
        if self.output_tokens > self.max_output_tokens:
            raise BudgetExceeded(
                f"출력 토큰 {self.output_tokens:,} > 상한 "
                f"{self.max_output_tokens:,}")

    def line(self) -> str:
        return (f"호출 {self.calls}+{self.failed_calls}실패 "
                f"(캐시 {self.cached_hits})  "
                f"입력 {self.input_tokens:,}  출력 {self.output_tokens:,}")


class OpenAILLM:
    """`MockLLM` 과 같은 인터페이스. 루프는 둘을 구분하지 않는다."""

    def __init__(self, cfg: LLMConfig, *, feature_names, shape_values,
                 registry, budget: Budget | None = None,
                 cache: bool = True) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingAPIKey(
                "OPENAI_API_KEY 가 없다. 실제 LLM 실행을 중단한다.\n"
                "  MockLLM 으로 조용히 폴백하지 않는다 — 그러면 'LLM 이 "
                "규칙을 만들었다' 를 거짓으로 믿게 된다 (§26.4).")
        self.cfg = cfg
        self.features = list(feature_names)
        self.shape_values = list(shape_values)
        # ★ RuleWriter 는 이름 목록이 아니라 **물리적 정의**를 넣어야 한다.
        #   `feature_names` 로는 physical_meaning 을 못 읽는다.
        #   기본값 없음 — 어느 레지스트리가 프롬프트에 들어가는지가 실험
        #   조건이고, `None` 이면 `render_features` 가 사람 24개로 떨어졌다
        #   (§30.9). 이제는 호출부가 반드시 명시한다.
        if registry is None:
            raise ValueError(
                "OpenAILLM(registry=...) 는 필수다. F0~F3 조건에서 어느 "
                "피처 목록이 프롬프트에 들어가는지가 실험 자체다 (§26.4).")
        # ★ 같은 판정이 두 곳에 있으면 갈린다 (원칙 2). `feature_names` 는
        #   정적 검사가 쓰고 `registry` 는 프롬프트가 쓴다 — 어긋나면 LLM 이
        #   본 적 없는 이름으로 검사받거나, 검사에 없는 이름을 프롬프트가
        #   권한다. 둘 다 비어 있지 않으면 포함 관계를 강제한다.
        # ★ `M/N/K/n_candidates` 는 레지스트리와 무관하게 항상 있다 —
        #   문제 자체의 성질이지 피처가 아니다. 처음에 이걸 빼먹어서 검증
        #   실행이 시작도 못 하고 죽었다 (LLM 호출 전이라 손해는 없었다).
        from kernelrule.core.matrix import INTRINSIC_SHAPE_FIELDS

        known = set(registry._items) | set(INTRINSIC_SHAPE_FIELDS)
        if set(registry._items) and (self.features or self.shape_values):
            stray = sorted((set(self.features) | set(self.shape_values))
                           - known)
            if stray:
                raise ValueError(
                    f"feature_names/shape_values 에 레지스트리 "
                    f"{registry.name!r} 에 없는 이름이 있다: {stray}. "
                    "정적 검사와 프롬프트가 서로 다른 목록을 보게 된다 "
                    "(원칙 2).")
        self.registry = registry
        self.budget = budget or Budget()
        self.calls: list[LLMCall] = []
        self._seq = 0
        #: 프롬프트 해시 -> 응답. 초반에 같은 프롬프트가 자주 반복된다 (§15.4)
        self._cache: dict[str, object] = {} if cache else None
        self._agents: dict[str, object] = {}
        #: ★ 루프 밖 역할 (D-92). 실험 스크립트가 **자기 프롬프트와 자기
        #: 스키마**를 들고 등록한다. `kernelrule/agents/` 는 루프가 부르는
        #: 넷(analyze / rule_writer / rule_editor / feature + categorize)만
        #: 안다 — 루프에 없는 역할이 여기 남으면 "언젠가 켤 것" 으로 읽힌다.
        self._extra: dict[str, tuple] = {}
        # ★ 유효 항 예산. **여기서 한 번 정하고 모든 자리에 넘긴다**
        #   (원칙 2). 전에는 `load_prompt` 의 기본값과 `checks.BUDGET`
        #   직접 import 가 각자 정했고, `rule_budget=16` 을 줘도
        #   **사용자 프롬프트와 역할 파일은 8 로 렌더링됐다** (D-105).
        from kernelrule.rules.checks import BUDGET as _CHECK_BUDGET
        self._budget = int(cfg.rule_budget if cfg.rule_budget is not None
                           else _CHECK_BUDGET)
        self._base = load_prompt("_base.md", budget=self._budget)
        self._hw = load_prompt(cfg.arch_prompt, budget=self._budget)
        #: ★ 목적함수. 프롬프트의 "채점 방식" 절을 정한다 (D-101).
        self.objective = getattr(cfg, "objective", "regret")
        if self.objective not in _OBJECTIVE_BLOCKS:
            raise ValueError(f"알 수 없는 목적함수: {self.objective!r}")
        self._rules = load_prompt("role/_rules_common.md",
                                  budget=self._budget)
        # ★ 조립은 `assemble_instructions` 한 곳에서 한다 (원칙 2).
        #   아래 넷은 **프롬프트 존재 확인용**으로만 읽는다 — 파일이
        #   없으면 첫 호출이 아니라 여기서 죽어야 한다.
        self._edit = load_prompt("role/_rules_edit.md",
                                 budget=self._budget)
        # ⚠️ `asyncio.Semaphore` 를 여기서 만들면 **첫 이벤트 루프에
        #    바인딩된다.** 루프는 라운드마다 `asyncio.run()` 을 새로 부르므로
        #    두 번째 라운드부터 "bound to a different event loop" 로 죽는다.
        #    실제로 밟았고, 그 예외가 후보 폐기로 처리돼 **조용히 호출을
        #    잃었다.** 루프마다 새로 만든다.
        self._sems: dict[int, asyncio.Semaphore] = {}
        self.rate_limit_events = 0
        #: (라운드, 시도 회차, 사유 코드, 메시지). 되먹임이 작동하는지 본다 —
        #: 1회차에 걸린 것이 2회차에도 **같은 이유**로 걸리면 재시도 상한을
        #: 올릴 것이 아니라 프롬프트를 고쳐야 한다.
        self.violations: list[dict] = []
        self.round = -1

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = self._sems.get(id(loop))
        if sem is None:
            sem = self._sems[id(loop)] = asyncio.Semaphore(
                self.cfg.concurrency)
        return sem

    # -- Agent 구성 (§11.2 — 고정 역할 + 주입 도메인 사실) ------------------
    def _agent(self, role: str):
        if role in self._agents:
            return self._agents[role]
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

        from kernelrule.agents.schemas import (
            AnalysisOutput,
            CategoryOutput,
            FeatureOutput,
            RuleOutput,
        )

        # ★ 루프 밖 역할은 `register_role` 로 실험 스크립트가 직접 등록한다
        #   (D-92). `kernelrule/agents/` 는 **루프가 부르는 넷**만 안다.
        if role in self._extra:
            out, body = self._extra[role]
        else:
            out = {"analyze": AnalysisOutput, "rule_editor": RuleOutput,
                   "rule_writer": RuleOutput, "feature": FeatureOutput,
                   "categorize": CategoryOutput}[role]
            body = load_prompt(f"role/{role}.md", budget=self._budget)
        # ★ **두 축**으로 나뉜다 (§30.10). 한 축(하드웨어 무관/의존)만으로
        #   나눴더니 역할별로 필요 없는 것이 공용에 쌓였다 — FeatureWriter 가
        #   regret 정의와 가중치 예산을 매번 받고 있었다.
        #
        #                 하드웨어 무관        하드웨어 의존
        #     역할 무관   _base.md            hw/sm_86.md
        #     역할 의존   role/*.md           (없음)
        #
        #   `hw` 는 Analyst / RuleWriter 만 받는다. RuleEditor 와
        #   FeatureWriter 가 안 보면 그 프롬프트는 **GPU 무관**해져서 새
        #   GPU 에 그대로 쓸 수 있다 (§16.2).
        #   규칙 블록은 다시 둘로 쪼갠다 — `_rules_common.md`(함수 형태·
        #   예산·벡터화)는 RuleEditor + RuleWriter, `_rules_edit.md`(regret
        #   정의·거부 사례 갤러리)는 **RuleEditor 만**. RuleWriter 는 백지
        #   에서 쓰므로 교체할 항도 이전 점수도 없다 (§30.10).
        instructions = assemble_instructions(
            role, objective=self.objective, hw_file=self.cfg.arch_prompt,
            body=body, budget=self.cfg.rule_budget)
        if self.cfg.endpoint not in ("responses", "chat"):
            raise ValueError(
                f"알 수 없는 엔드포인트: {self.cfg.endpoint!r}. "
                "'responses' 또는 'chat'")
        model = (OpenAIResponsesModel(self.cfg.model)
                 if self.cfg.endpoint == "responses"
                 else OpenAIChatModel(self.cfg.model))
        # ★ 보낼 수 없는 조합은 **여기서 멈춘다** (D-47). pydantic-ai 에
        #   넘기면 조용히 버려지고, config.json 에는 값이 남아 기록과 실제가
        #   어긋난다.
        if self.cfg.seed is not None and self.cfg.endpoint == "responses":
            raise ValueError(
                "seed 는 Responses 엔드포인트에 **파라미터가 없다** (SDK 가 "
                "TypeError 를 낸다). pydantic-ai 는 조용히 버리므로 "
                "config.json 에만 남는다. `seed=None` 으로 두거나 "
                "`endpoint='chat'` 을 써라 (D-47).")
        settings: dict = {}
        if self.cfg.temperature is not None:
            settings["temperature"] = self.cfg.temperature
        if self.cfg.seed is not None:
            settings["seed"] = self.cfg.seed
        if self.cfg.reasoning_effort is not None:
            # pydantic-ai 는 공급자 접두사를 붙인 키로 전달한다.
            settings["openai_reasoning_effort"] = self.cfg.reasoning_effort
        a = Agent(model, output_type=out, instructions=instructions,
                  retries=self.cfg.max_retries, model_settings=settings)
        self._agents[role] = a
        return a

    # -- 프롬프트 조립 ----------------------------------------------------
    def _feature_block(self) -> str:
        """★ **모든 역할이 같은 렌더러를 쓴다** (§11.2 / D-34).

        전에는 RuleWriter 만 `render_features()` 로 범위와 물리적 정의를 받고,
        RuleEditor 와 Analyst 는 **이름 목록**만 받았다. 진화 루프의 LLM 이
        `has_spill` 이 무엇을 재는지 모르는 채로 항을 골랐고, 그래서 비용
        없는 가지치기를 놓쳤다 (`artifacts/spill-term.md`).

        두 경로를 남겨두면 다시 갈린다. `_feature_block` 을 이쪽으로 흡수해
        **출처를 하나로** 만든다.

        ⚠️ 하드웨어 상수(SM 84, smem 99KB, ridge)는 주지 않는다. 피처가 이미
        흡수했고, 안 보면 이 프롬프트가 **GPU 무관**해져 새 GPU 에 그대로
        쓸 수 있다 (§16.2).
        """
        from kernelrule.features import render_features

        if self.cfg.feature_detail not in ("full", "names"):
            raise ValueError(
                f"알 수 없는 feature_detail: {self.cfg.feature_detail!r}. "
                "'full' 또는 'names'")
        if self.cfg.feature_detail == "names":
            # ★ 2026-08-22 이전 상태를 그대로 재현한다 — 이름 목록만.
            return ("## config 수준 (`f.<이름>`)\n\n"
                    + "\n".join(f"- `{n}`" for n in self.features)
                    + "\n\n## 형상 수준 (`p.<이름>`)\n\n"
                    + "\n".join(f"- `{n}`" for n in self.shape_values))
        if self.registry is None:
            # 레지스트리가 없으면 이름만 — 그리고 **그 사실을 말한다** (§26.4)
            names = "\n".join(f"- `{n}`" for n in self.features)
            svals = "\n".join(f"- `{n}`" for n in self.shape_values)
            return ("⚠️ 피처의 물리적 정의를 불러오지 못했다 (registry 없음). "
                    f"이름만 있다.\n\n{names}\n\n{svals}")
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
            # 등록 역할은 사용자 프롬프트도 호출자가 만든다.
            return prompt
        if role == "analyze":
            return prompt + "\n\n---\n\n## 등록된 피처\n\n" + fl + "\n"
        parent = kw.get("parent")
        hyp = kw.get("hypothesis") or {}
        applied = kw.get("hypotheses_applied") or []
        # ★ 부모의 현재 항 수를 주입하고, 포화 시 **교체를 지시**한다.
        #   "버려도 된다" 는 선택지이고 "버리고 넣어라" 는 지시다.
        #   예산이 `role/_rules.md`(시스템)에만 있으면 긴 컨텍스트에서 희석된다.
        n_terms = int(kw.get("parent_n_terms") or 0)
        n_w = len(parent.w0) if parent else 0
        # ★ `checks.BUDGET` 을 직접 읽으면 `rule_budget` 을 무시한다
        #   (D-105). 유효 예산은 `self._budget` 하나뿐이다.
        if n_terms >= self._budget:
            note = ("\n★ 예산이 찼습니다. 항을 추가하지 마세요.\n"
                    "  이번 가설을 반영하려면 **가장 덜 중요한 항 하나를 "
                    "지우고**\n  그 자리에 넣으세요. 무엇을 지웠고 왜 그것을 "
                    "골랐는지\n  `changes` 에 쓰세요.")
        else:
            note = f"남은 예산: {self._budget - n_terms}항"
        # ★ Analyst 가 꺼져 있으면 **가설 절 자체를 안 만든다** (§16.1, D-89).
        #   "## 이번 가설\n\n(가설 없음)" 처럼 빈 자리를 남기면 모델이
        #   "가설이 있는데 비어 있다" 로 읽어 다른 조건이 된다. 진단
        #   리포트를 만들지도 않는 것과 같은 원칙이다.
        # ★ Analyst 를 끈 프롬프트는 켠 것에서 **문장을 지운 것**이어야 한다
        #   (§16.1, D-89). 새 문구를 쓰면 "Analyst 만 다르다" 가 깨진다 —
        #   `test_optimize_prompt_without_analyst_is_a_deletion` 이 고정한다.
        if kw.get("analyst", True):
            hyp_block = (
                "## 현재 규칙에 반영된 가설\n\n"
                + ("\n".join(f"- {h}" for h in applied)
                   or "(아직 없음 — 첫 라운드다)")
                + "\n\n## 이번 가설\n\n"
                + (json.dumps(hyp, ensure_ascii=False, indent=1) if hyp else
                   "(가설 없음. 부모를 개선할 방향을 스스로 찾아라)"))
            inputs_hyp = ("이번에 반영할 가설 하나\n"
                          "현재 규칙에 이미 반영된 가설들\n")
            one_change = "가설이 국소적인 것은 **의도**입니다. "
            applied_warn = (
                "\n**기존 가설들의 효과를 훼손하지 마세요.** 아래 목록의 "
                "항들은 이유가 있어\n들어간 것입니다. 그것을 지우려면 이번 "
                "가설이 그 이유를 무효화한다는 근거가\n있어야 합니다.\n")
        else:
            hyp_block = inputs_hyp = one_change = applied_warn = ""
        # ★ 두 번째 부모는 `cross` 일 때만 있다 (D-96). 없으면 **절 자체를
        #   안 만든다** — "(부모 없음)" 같은 빈 자리를 남기면 모델이 "둘째가
        #   있는데 비어 있다" 로 읽고, 그러면 exploit/explore 의 조건이
        #   달라진다 (D-89 에서 밟았다).
        p2 = kw.get("parent2")
        if p2 is None:
            second = ""
        else:
            second = (
                "\n\n## ★ 두 번째 부모 — 이 둘을 **합치세요**\n\n"
                "```python\n" + p2.code.strip() + "\n```\n\n"
                f"두 번째 부모의 가중치: {list(p2.w0)}\n\n"
                "**각각의 좋은 항을 골라 하나로 만드세요.** 한쪽을 그대로 "
                "베끼지 마세요 — 그러면 교차가 아닙니다.\n\n"
                f"⚠️ 예산이 {self._budget}항이므로 합치면 **반드시 버려야 "
                "합니다.** 무엇을 버렸고 왜 그것을 골랐는지 `changes` 에 "
                "쓰세요.\n")
        body = load_prompt("role/rule_editor.md", budget=self._budget)
        return body.format(
            second_parent_block=second,
            n_terms=n_terms, n_weights=n_w, budget_note=note,
            feature_block=fl, hypothesis_block=hyp_block,
            inputs_hyp=inputs_hyp, one_change_hyp=one_change,
            applied_warning=applied_warn,
            parent_code=(parent.code if parent else
                         "(부모 없음 — 처음부터 만들어라)"),
            parent_w=(list(parent.w0) if parent else "-"))


    # -- RuleWriter (§11.8) — 부모도 사례도 점수도 받지 않는다 ---------------
    def _rule_writer_prompt(self, *, condition: str = "A",
                          table_facts=None, registry=None, **_kw) -> str:
        """★ 조건 A 는 **표에서 나온 문장이 하나도 없다.**

        전이 시나리오와의 정합성이 이 조건의 존재 이유다:

            완전 이식 §29.5(a)   표 0      구조+가중치 그대로
            재적합   §29.5(b)   표본 5%   구조 고정, 가중치만
            재생성   §29.5(c)   전수      구조부터 새로

        표를 봐야 **구조**가 나오면 그것은 (c) 다. 그런데 전수를 잴 거면
        표를 직접 쓰면 되므로 이 시스템을 쓸 이유가 없다. 그래서 A 가
        관문이고, B 와의 격차가 곧 "표의 값어치" 다.

        조립을 손으로 하지 않는다 — `render_features` 를 통과시킨다 (D-28).
        """
        from kernelrule.features import render_features

        if condition not in ("A", "B"):
            raise ValueError(f"알 수 없는 RuleWriter 조건: {condition!r}. "
                             "A(물리만) 또는 B(물리+학습분할 집계)")
        if condition == "B" and table_facts is None:
            raise ValueError(
                "조건 B 는 학습 분할 집계가 필요하다. "
                "TableFacts.compute(table, splits.train) 을 넘겨라 (§12.3).")

        # ★ 넘겨받은 레지스트리를 쓴다. 전에는 `self.registry` 고정이라
        #   F1/F1-K 처럼 다른 라이브러리로 부를 때 어긋날 수 있었다 (§30.9).
        reg = registry if registry is not None else self.registry
        extra = getattr(table_facts, "by_feature", None) if table_facts else None
        block = render_features(reg, include_observed=condition == "B",
                                extra_observed=extra)
        if condition == "A":
            note = "당신은 이 GPU 의 측정 표를 보지 않습니다"
            agg = ("## 표 집계\n\n**없습니다.** 이것이 조건 A 입니다 — "
                   "물리만 보고 쓰세요.")
        else:
            lines = "\n".join(table_facts.lines)
            note = "학습 분할의 **집계**만 봅니다. 형상별 답은 보지 않습니다"
            agg = ("## 표 집계 (학습 분할에서만 — §12.3)\n\n"
                   "개별 형상의 답이 아니라 전체에서 나온 패턴입니다. "
                   "**형상을 식별할 수 있는 것은 없습니다.**\n\n"
                   f"```\n{lines}\n```")
        # ★ 규칙 예시도 **조건마다 다르다** (§30.20). RuleWriter 는
        #   `condition` 이 A/B(표 관측 유무)라 피처 조건과 축이 다르다 —
        #   레지스트리가 사람 24개면 실제 이름을 써도 되고, F0/F1
        #   레지스트리면 무관 도메인을 써야 한다.
        rule_ex = _rule_example_for(reg, budget=self._budget)
        return load_prompt("role/rule_writer.md",
                           budget=self._budget).format(
            rule_example_block=rule_ex,
            table_note=note, feature_block=block, aggregate_block=agg)


    # -- FeatureWriter (§11.4) — 없는 축을 만든다 ---------------------------
    def _categorize_prompt(self, *, n_min: int = 5, n_max: int = 8,
                           **_kw) -> str:
        """★ 영역을 **LLM 이** 나눈다 (§30.10).

        사람이 카테고리를 주면 사전 지식을 건네는 것이다 — "메모리
        트래픽이 중요하다" 를 알려주는 셈이다. 그리고 나눈 결과 자체가
        관찰 대상이다: LLM 이 GEMM 성능의 물리를 어떻게 구조화하는가.
        """
        from kernelrule.features.generated import field_block

        return load_prompt("role/categorize.md",
                           budget=self._budget).format(
            field_block=field_block(), n_min=n_min, n_max=n_max)

    def _feature_prompt(self, *, condition: str = "F1", task: str = "",
                        registry=None, **_kw) -> str:
        """F0~F3 — **피처를 얼마나 주느냐**가 조건이다.

            F0  없음        물리를 처음부터 코드로 옮길 수 있나
            F1  원시 값만    파생 물리량을 만들 수 있나   ★ 근본 질문
            F2  기초 5개     그 위에 쌓을 수 있나
            F3  전부        조합만 (= 지금까지의 모든 실행)

        ⚠️ 형태 예시는 **이 문제와 무관한 것**을 쓴다. `optimize.md` 의
        출력 예시가 `physics_seeded` 축약판이라 "씨앗 없음" 조건에 씨앗이
        들어간 일이 있다 (D-35).
        """
        from kernelrule.features import render_features
        from kernelrule.features.generated import field_block

        if condition not in _CONDITIONS:
            raise ValueError(
                f"알 수 없는 조건: {condition!r}. {sorted(_CONDITIONS)}")
        reg = registry if registry is not None else self.registry
        if condition == "F0":
            block = ("## 이미 있는 피처\n\n**없습니다.** 처음부터 만드세요.")
        elif condition == "F1":
            block = ("## 이미 있는 피처\n\n**없습니다.** 위 원시 값만으로 "
                     "물리량을 유도하세요.\n\n★ 이것이 이 조건의 요점입니다 "
                     "— 파생량을 스스로 만들 수 있는지를 봅니다.")
        else:
            if reg is None:
                raise ValueError(f"조건 {condition} 은 레지스트리가 필요하다")
            block = ("## 이미 있는 피처 — **중복되면 폐기됩니다**\n\n"
                     + render_features(reg, include_observed=False))
        # ★ 예시는 **조건마다 다르다** (§30.17). F0/F1 은 답을 건네지
        #   않으려 무관 도메인을 쓰고, 공개 지식을 주는 조건(F1-K/F2/F3)은
        #   실제 피처를 코드까지 보여준다 — 그것이 조건의 정의이므로
        #   D-35 의 조심이 여기서는 불필요하다.
        example = load_prompt(f"examples/{_EXAMPLES[condition]}.md",
                              budget=self._budget)
        return load_prompt("role/feature.md", budget=self._budget).format(
            field_block=field_block(), feature_block=block,
            example_block=example,
            area_block=load_prompt("areas.md", budget=self._budget),
            task_block=task or ("## 이번에 만들 것\n\n피처 하나를 제안하세요."))

    # -- 루프 밖 역할 등록 (D-92) -----------------------------------------
    def register_role(self, name: str, *, instructions: str,
                      output_type) -> None:
        """루프에 없는 역할을 **호출자가** 등록한다.

        ★ 루프가 부르지 않는 역할을 `kernelrule/agents/` 에 남겨 두면
        "언젠가 켤 것" 으로 읽히고, 조건 목록과 ablation 표에 계속 끌려
        다닌다 (D-92). 프롬프트와 스키마를 **쓰는 쪽이 들고 온다.**

        예산·재시도·추적·`dump()` 는 그대로 쓴다 — LLM 호출은 다시 만들
        수 없으므로 남기는 경로가 하나여야 한다 (D-33).
        """
        if name in ("analyze", "rule_writer", "rule_editor", "feature",
                    "categorize"):
            raise ValueError(
                f"{name!r} 은 루프 역할이다. 덮어쓰면 조용히 다른 것이 돈다.")
        self._extra[name] = (output_type, instructions)

    # -- 진입점 -----------------------------------------------------------
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
                    # ★ 조용히 줄이지 않는다. 로그에 남기고 한 번 물러선다.
                    await asyncio.sleep(20.0)
                    res = await agent.run(user)
                else:
                    # ★ 실패해도 토큰은 이미 소모됐다. 호출 수만이라도 센다 —
                    #   안 세면 재시도가 폭주해도 예산 감시가 안 걸린다.
                    self.budget.failed_calls += 1
                    if (self.budget.calls + self.budget.failed_calls
                            > self.budget.max_calls):
                        raise BudgetExceeded(
                            f"호출 {self.budget.calls}+"
                            f"{self.budget.failed_calls}실패 > 상한 "
                            f"{self.budget.max_calls}") from e
                    raise
            dt = time.perf_counter() - t0

        # pydantic-ai 2.x 는 속성, 1.x 는 메서드다. 조용히 0 으로 떨어지면
        # 예산 감시가 무력해지므로 **둘 다 시도하고 실패하면 에러**다.
        u = res.usage
        if callable(u):
            u = u()
        n_in = (getattr(u, "input_tokens", None)
                or getattr(u, "request_tokens", None))
        n_out = (getattr(u, "output_tokens", None)
                 or getattr(u, "response_tokens", None))
        if n_in is None or n_out is None:
            raise RuntimeError(
                f"토큰 사용량을 읽을 수 없다: {type(u).__name__} "
                f"{[a for a in dir(u) if 'token' in a]}. "
                "0 으로 떨어지면 예산 감시가 무력해진다 (§26.4).")
        self.budget.charge(n_in, n_out)
        out = res.output
        payload = out.model_dump() if hasattr(out, "model_dump") else out
        self.calls.append(LLMCall(role=role, prompt_hash=h, response=payload,
                                  seq=seq, mode=self.cfg.model))
        # 원본을 남긴다. ★ 키나 인증 헤더는 저장하지 않는다.
        self._last = {"prompt": user, "seconds": dt,
                      "input_tokens": n_in, "output_tokens": n_out}
        self.calls[-1].__dict__["_meta"] = self._last
        if self._cache is not None:
            self._cache[h] = payload
        return payload

    async def _run_traced(self, agent, user: str, role: str, seq: int):
        """Pydantic AI 재시도의 **회차별 위반**을 기록한다.

        프레임워크가 validator 실패를 모델에 되먹여 재시도하는데, 그 내역이
        밖에서 안 보인다. 결과 메시지에서 되짚어 회차별로 남긴다.
        """
        # ★ `capture_run_messages` 로 감싼다. 그러지 않으면 **실패했을 때**
        #   회차별 메시지를 볼 수 없다 — 예외만 남고 `res` 가 없다.
        #   RuleWriter A 조건에서 10회 중 8회가 재시도 소진으로 죽었는데
        #   무엇이 걸렸는지 알 수 없었다. 그러면 프롬프트를 어디를 고칠지
        #   모른다 (§26.4 — 실패가 정보를 남겨야 한다).
        from pydantic_ai import capture_run_messages

        def _harvest(msgs, seq_: int) -> None:
            for i, m in enumerate(msgs or []):
                for part in getattr(m, "parts", []):
                    # ★ `RetryPromptPart.content` 는 **dict 리스트**다.
                    #   str 만 보면 되먹임 내역이 통째로 안 잡힌다 —
                    #   실제로 재시도 소진의 이유를 못 읽고 있었다.
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
        """사유 코드 x 회차 분포. 되먹임이 작동하는지 본다."""
        from collections import Counter

        by_code = Counter(v["code"] for v in self.violations)
        by_attempt = Counter(v["attempt"] for v in self.violations)
        # 같은 호출(seq)에서 같은 코드가 두 번 이상 나왔는가
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
        """규칙 12개를 **병렬로** 부른다 (§4-0)."""
        return await asyncio.gather(
            *(self.acomplete(role, it.pop("prompt", ""), **it)
              for it in items), return_exceptions=True)

    # -- 기록 -------------------------------------------------------------
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


def estimate_and_confirm(*, n_rounds: int, n_rules: int, report_chars: int,
                         cfg: LLMConfig, yes: bool = False) -> dict:
    """예상 호출 수와 토큰을 출력하고 확인을 요구한다 (§4-1)."""
    per_round = 1 + n_rules
    calls = per_round * n_rounds
    tok_in = int(report_chars / 3) * n_rounds + int(report_chars / 6) * \
        n_rules * n_rounds
    est = {"model": cfg.model, "calls": calls,
           "est_input_tokens": tok_in, "per_round": per_round}
    print("=" * 62)
    print(f"실제 LLM 실행 예상  모델 {cfg.model}  온도 {cfg.temperature}")
    print(f"  라운드 {n_rounds} x (진단 1 + 규칙 {n_rules}) = 호출 {calls}")
    print(f"  입력 토큰 대략 {tok_in:,}")
    print("=" * 62)
    if not yes:
        raise BudgetExceeded(
            "확인이 필요하다. `--yes` 로 진행하거나 예산을 조정하라.")
    return est
