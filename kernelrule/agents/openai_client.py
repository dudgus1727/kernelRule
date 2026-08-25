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


def load_prompt(name: str) -> str:
    p = _PROMPTS / name
    if not p.exists():
        raise FileNotFoundError(f"프롬프트가 없다: {p}")
    return p.read_text()


@dataclass
class LLMConfig:
    """`config.json` 에 그대로 기록된다 (§15.4 재현성)."""

    model: str = DEFAULT_MODEL
    #: ★ 규칙 생성은 다양성이 필요하므로 0 으로 두지 않는다.
    #:   값을 기록하고 run 간 고정한다.
    temperature: float = 0.7
    seed: int | None = 20260821
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
                 budget: Budget | None = None, cache: bool = True,
                 registry=None) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingAPIKey(
                "OPENAI_API_KEY 가 없다. 실제 LLM 실행을 중단한다.\n"
                "  MockLLM 으로 조용히 폴백하지 않는다 — 그러면 'LLM 이 "
                "규칙을 만들었다' 를 거짓으로 믿게 된다 (§26.4).")
        self.cfg = cfg
        self.features = list(feature_names)
        self.shape_values = list(shape_values)
        # ★ Architect 는 이름 목록이 아니라 **물리적 정의**를 넣어야 한다.
        #   `feature_names` 로는 physical_meaning 을 못 읽는다.
        self.registry = registry
        self.budget = budget or Budget()
        self.calls: list[LLMCall] = []
        self._seq = 0
        #: 프롬프트 해시 -> 응답. 초반에 같은 프롬프트가 자주 반복된다 (§15.4)
        self._cache: dict[str, object] = {} if cache else None
        self._agents: dict[str, object] = {}
        self._common = load_prompt("_common.md")
        self._hw = load_prompt(cfg.arch_prompt)
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

        from kernelrule.agents.schemas import AnalysisOutput, FeatureOutput, RuleOutput

        out = {"analyze": AnalysisOutput, "optimize": RuleOutput,
               "architect": RuleOutput, "feature": FeatureOutput}[role]
        role_md = load_prompt(f"{role}.md")
        # ★ 두 층으로 나뉜다: [고정] 역할·제약  +  [주입] 하드웨어 사실
        instructions = f"{self._common}\n\n---\n\n{self._hw}\n\n---\n\n{role_md}"
        if self.cfg.endpoint not in ("responses", "chat"):
            raise ValueError(
                f"알 수 없는 엔드포인트: {self.cfg.endpoint!r}. "
                "'responses' 또는 'chat'")
        model = (OpenAIResponsesModel(self.cfg.model)
                 if self.cfg.endpoint == "responses"
                 else OpenAIChatModel(self.cfg.model))
        a = Agent(model, output_type=out, instructions=instructions,
                  retries=self.cfg.max_retries,
                  model_settings={"temperature": self.cfg.temperature,
                                  **({"seed": self.cfg.seed}
                                     if self.cfg.seed is not None else {})})
        self._agents[role] = a
        return a

    # -- 프롬프트 조립 ----------------------------------------------------
    def _feature_block(self) -> str:
        """★ **모든 역할이 같은 렌더러를 쓴다** (§11.2 / D-34).

        전에는 Architect 만 `render_features()` 로 범위와 물리적 정의를 받고,
        Optimizer 와 Analyst 는 **이름 목록**만 받았다. 진화 루프의 LLM 이
        `has_spill` 이 무엇을 재는지 모르는 채로 항을 골랐고, 그래서 비용
        없는 가지치기를 놓쳤다 (`artifacts/spill-term.md`).

        두 경로를 남겨두면 다시 갈린다. `_feature_block` 을 이쪽으로 흡수해
        **출처를 하나로** 만든다.

        ⚠️ 하드웨어 상수(SM 84, smem 99KB, ridge)는 주지 않는다. 피처가 이미
        흡수했고, 안 보면 이 프롬프트가 **GPU 무관**해져 새 GPU 에 그대로
        쓸 수 있다 (§16.2).
        """
        from kernelrule.features import render_features

        if self.registry is None:
            # 레지스트리가 없으면 이름만 — 그리고 **그 사실을 말한다** (§26.4)
            names = "\n".join(f"- `{n}`" for n in self.features)
            svals = "\n".join(f"- `{n}`" for n in self.shape_values)
            return ("⚠️ 피처의 물리적 정의를 불러오지 못했다 (registry 없음). "
                    f"이름만 있다.\n\n{names}\n\n{svals}")
        return render_features(self.registry, include_observed=False)

    def _user_prompt(self, role: str, prompt: str, **kw) -> str:
        fl = self._feature_block()
        if role == "architect":
            return self._architect_prompt(**kw)
        if role == "feature":
            return self._feature_prompt(**kw)
        if role == "analyze":
            return prompt + "\n\n---\n\n## 등록된 피처\n\n" + fl + "\n"
        parent = kw.get("parent")
        hyp = kw.get("hypothesis") or {}
        applied = kw.get("hypotheses_applied") or []
        # ★ 부모의 현재 항 수를 주입하고, 포화 시 **교체를 지시**한다.
        #   "버려도 된다" 는 선택지이고 "버리고 넣어라" 는 지시다.
        #   예산이 `_common.md`(시스템)에만 있으면 긴 컨텍스트에서 희석된다.
        n_terms = int(kw.get("parent_n_terms") or 0)
        n_w = len(parent.w0) if parent else 0
        if n_terms >= 8:
            note = ("\n★ 예산이 찼습니다. 항을 추가하지 마세요.\n"
                    "  이번 가설을 반영하려면 **가장 덜 중요한 항 하나를 "
                    "지우고**\n  그 자리에 넣으세요. 무엇을 지웠고 왜 그것을 "
                    "골랐는지\n  `changes` 에 쓰세요.")
        else:
            note = f"남은 예산: {8 - n_terms}항"
        body = load_prompt("optimize.md")
        return body.format(
            n_terms=n_terms, n_weights=n_w, budget_note=note,
            feature_block=fl,
            hypotheses_applied=("\n".join(f"- {h}" for h in applied)
                                or "(아직 없음 — 첫 라운드다)"),
            hypothesis=(json.dumps(hyp, ensure_ascii=False, indent=1)
                        if hyp else "(가설 없음. 부모를 개선할 방향을 "
                                    "스스로 찾아라)"),
            parent_code=(parent.code if parent else
                         "(부모 없음 — 처음부터 만들어라)"),
            parent_w=(list(parent.w0) if parent else "-"))


    # -- Architect (§11.8) — 부모도 사례도 점수도 받지 않는다 ---------------
    def _architect_prompt(self, *, condition: str = "A",
                          table_facts=None, **_kw) -> str:
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
            raise ValueError(f"알 수 없는 Architect 조건: {condition!r}. "
                             "A(물리만) 또는 B(물리+학습분할 집계)")
        if condition == "B" and table_facts is None:
            raise ValueError(
                "조건 B 는 학습 분할 집계가 필요하다. "
                "TableFacts.compute(table, splits.train) 을 넘겨라 (§12.3).")

        extra = getattr(table_facts, "by_feature", None) if table_facts else None
        block = render_features(self.registry, include_observed=condition == "B",
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
        return load_prompt("architect.md").format(
            table_note=note, feature_block=block, aggregate_block=agg)


    # -- FeatureWriter (§11.4) — 없는 축을 만든다 ---------------------------
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

        if condition not in ("F0", "F1", "F2", "F3"):
            raise ValueError(f"알 수 없는 조건: {condition!r}. F0~F3")
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
        return load_prompt("feature.md").format(
            field_block=field_block(), feature_block=block,
            task_block=task or ("## 이번에 만들 것\n\n피처 하나를 제안하세요."))

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
        #   Architect A 조건에서 10회 중 8회가 재시도 소진으로 죽었는데
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
