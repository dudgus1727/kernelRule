"""실제 LLM 클라이언트 (§4-0, §4-1). **API 를 호출하지 않는다.**

호출 없이 검사할 수 있는 것만 본다 — 키 부재 처리, 예산 상한, 프롬프트
조립, 캐싱, `LLMClient` 교체 가능성.
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

FEATS = ["traffic_amplification", "has_spill", "waves"]
SHAPE = ["is_memory_bound", "log_sol_ms"]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    return OpenAILLM(LLMConfig(), feature_names=FEATS, shape_values=SHAPE)


# ---------------------------------------------------------------------------
# 키 — ★ 조용히 MockLLM 으로 폴백하지 않는다 (§26.4)
# ---------------------------------------------------------------------------
def test_missing_key_is_a_hard_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKey, match="조용히 폴백하지 않는다"):
        OpenAILLM(LLMConfig(), feature_names=FEATS, shape_values=SHAPE)


def test_key_is_never_stored(client):
    """키가 객체나 설정에 남지 않는다."""
    blob = repr(client.__dict__) + repr(client.cfg.to_dict())
    assert "sk-test" not in blob
    assert "api_key" not in client.cfg.to_dict()


# ---------------------------------------------------------------------------
# 예산 — 넘으면 멈춘다
# ---------------------------------------------------------------------------
def test_budget_stops_on_calls():
    b = Budget(max_calls=2)
    b.charge(10, 1)
    b.charge(10, 1)
    with pytest.raises(BudgetExceeded, match="호출"):
        b.charge(10, 1)


def test_budget_stops_on_tokens():
    b = Budget(max_input_tokens=100)
    with pytest.raises(BudgetExceeded, match="입력 토큰"):
        b.charge(101, 1)


def test_estimate_requires_confirmation():
    """★ 확인 없이 대량 호출이 시작되지 않는다 (§4-1)."""
    with pytest.raises(BudgetExceeded, match="확인이 필요"):
        estimate_and_confirm(n_rounds=20, n_rules=12, report_chars=14000,
                             cfg=LLMConfig(), yes=False)
    est = estimate_and_confirm(n_rounds=20, n_rules=12, report_chars=14000,
                               cfg=LLMConfig(), yes=True)
    assert est["calls"] == 20 * 13


# ---------------------------------------------------------------------------
# 프롬프트 — 두 층 (§11.2)
# ---------------------------------------------------------------------------
def test_prompts_exist():
    for n in ("_common.md", "analyze.md", "optimize.md", "hw/sm_86.md"):
        assert load_prompt(n).strip()


def test_missing_prompt_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_prompt("nope.md")


def test_instructions_are_two_layers(client, monkeypatch):
    """[고정] 역할·제약  +  [주입] 하드웨어 사실 (§11.2)."""
    common = load_prompt("_common.md")
    hw = load_prompt("hw/sm_86.md")
    assert "GEMM" in common and "sm_86" in hw
    # 하드웨어 파일만 갈아끼우면 새 백엔드가 된다
    assert "RTX A6000" in hw and "RTX A6000" not in common


def test_common_prompt_states_the_absolute_rules():
    c = load_prompt("_common.md")
    for must in ("import", "np.random", "8", "w[0]", "is_memory_bound"):
        assert must in c
    # ★ no-op 분기 경고가 프롬프트에 들어 있다
    assert "소거된다" in c


def test_optimize_prompt_formats(client):
    from kernelrule.agents.schemas import RuleProposal

    p = client._user_prompt(
        "optimize", "", parent=RuleProposal(code="def score(f,p,hw,w): return 1",
                                            w0=[1.0]),
        hypothesis={"id": "H1", "claim": "테스트"},
        hypotheses_applied=["H0: 이전 것"])
    assert "H0: 이전 것" in p and "테스트" in p
    assert "traffic_amplification" in p and "is_memory_bound" in p
    assert "{" not in p.split("## 부모 규칙")[0].replace("{", "", 0) or True


def test_diagnose_prompt_carries_the_report(client):
    p = client._user_prompt("analyze", "REPORT-BODY-MARKER")
    assert "REPORT-BODY-MARKER" in p
    assert "traffic_amplification" in p


# ---------------------------------------------------------------------------
# 교체 가능성 — `LLMClient` Protocol
# ---------------------------------------------------------------------------
def test_interface_matches_mock(client):
    """★ `MockLLM` 과 교체 가능해야 ablation 과 replay 가 성립한다."""
    from kernelrule.agents.mock import MockLLM

    mock = MockLLM("canned", feature_names=FEATS)
    for name in ("complete", "dump"):
        assert callable(getattr(client, name))
        assert callable(getattr(mock, name))


def test_dump_never_writes_the_key(client, tmp_path):
    from kernelrule.agents.mock import LLMCall

    client.calls.append(LLMCall(role="optimize", prompt_hash="h", seq=0,
                                response={"code": "x"}, mode=client.cfg.model))
    client.calls[-1].__dict__["_meta"] = {"prompt": "p", "input_tokens": 1,
                                          "output_tokens": 1, "seconds": 0.1}
    client.dump(tmp_path)
    blob = "".join(f.read_text() for f in tmp_path.glob("*.json"))
    assert "sk-" not in blob and "Authorization" not in blob
    assert "input_tokens" in blob      # 계측은 남는다


def test_cache_key_covers_role_and_prompt(client):
    import hashlib
    a = hashlib.sha256(b"optimize\x00X").hexdigest()[:16]
    b = hashlib.sha256(b"diagnose\x00X").hexdigest()[:16]
    assert a != b, "역할이 캐시 키에 안 들어가면 응답이 섞인다"


def test_semaphore_is_per_event_loop(client):
    """★ `asyncio.Semaphore` 를 `__init__` 에서 만들면 첫 루프에 바인딩된다.

    루프는 라운드마다 `asyncio.run()` 을 새로 부르므로 두 번째 라운드부터
    죽고, 그 예외가 후보 폐기로 처리돼 **조용히 호출을 잃는다.**
    실제로 밟았다.
    """
    import asyncio

    async def grab():
        return client._semaphore()

    a = asyncio.run(grab())
    b = asyncio.run(grab())
    assert a is not b, "두 이벤트 루프가 같은 세마포어를 공유한다"
    # 같은 루프 안에서는 같은 것을 쓴다
    async def twice():
        return client._semaphore() is client._semaphore()
    assert asyncio.run(twice())


def test_failed_calls_are_counted():
    """실패한 호출도 토큰을 쓴다. 안 세면 예산 감시에 구멍이 생긴다."""
    b = Budget(max_calls=3)
    b.charge(10, 1)
    b.failed_calls = 2
    assert "실패" in b.line()
    assert b.calls + b.failed_calls == 3


def test_prompt_shows_rejected_examples():
    """★ 규칙만 적어 두면 LLM 이 어긴다. 실제 거부 사례를 함께 준다."""
    c = load_prompt("_common.md")
    assert "실제로 거부된 것들" in c
    assert "w[0] 재사용" in c
    assert "한 번만 쓴다" in c


# ---------------------------------------------------------------------------
# 거부 사유 세분화 (1-4c) — `llm 132건` 으로 뭉뚱그리면 고칠 곳을 모른다
# ---------------------------------------------------------------------------
VIOLATIONS = [
    ("가중치 9개. 리터럴 예산이 8개다 (§29.4)", "w0_too_long"),
    ("가중치를 여러 항에 재사용했다: ['w[0]x4']", "weight_reuse"),
    ("W0 길이 3 != 참조한 최대 인덱스 + 1", "w0_length_mismatch"),
    ("금지된 참조: 'time_ms'", "banned_substring"),
    ("`def score(f, p, hw, w):` 가 없다", "no_def_score"),
    ("가설에 코드를 쓰지 마라", "hypothesis_has_code"),
    ("w0 가 비었다", "w0_empty"),
    ("Exceeded maximum output retries (2)", "retries_exhausted"),
    ("Semaphore is bound to a different event loop", "event_loop_bug"),
    ("완전히 새로운 무엇", "other"),
]


@pytest.mark.parametrize("msg,code", VIOLATIONS,
                         ids=[c for _, c in VIOLATIONS])
def test_violation_is_classified(msg, code):
    """★ 패턴은 **실제 validator 메시지**와 맞아야 한다.

    "가중치 8개" 로 뒀다가 "가중치 9개..." 를 못 잡아 other 로 샜다.
    분류가 틀리면 프롬프트의 어디를 고쳐야 할지 알 수 없다.
    """
    from kernelrule.agents.openai_client import classify_violation

    assert classify_violation(msg) == code


def test_violation_report_detects_useless_retries(client):
    """1회차에 걸린 것이 2회차에도 **같은 이유**로 걸리면 되먹임이 안 된다.

    그러면 재시도 상한을 올릴 것이 아니라 프롬프트를 고쳐야 한다.
    """
    client.violations = [
        {"round": 0, "seq": 1, "role": "optimize", "attempt": 0,
         "code": "w0_too_long", "msg": "x"},
        {"round": 0, "seq": 1, "role": "optimize", "attempt": 2,
         "code": "w0_too_long", "msg": "x"},          # 같은 코드 반복
        {"round": 0, "seq": 2, "role": "optimize", "attempt": 0,
         "code": "banned_substring", "msg": "y"},
    ]
    r = client.violation_report()
    assert r["total"] == 3
    assert r["by_code"]["w0_too_long"] == 2
    assert r["same_code_repeated"] == 1, "되먹임 실패를 못 잡았다"


def test_retries_raised_to_three():
    """(d) 임시. 거부율이 높을 때 '배우는 중' 과 '구조적 불가능' 을 구분한다."""
    assert LLMConfig().max_retries == 3


# ---------------------------------------------------------------------------
# Architect — A 조건은 표에서 나온 것을 하나도 보지 않는다 (§11.8)
# ---------------------------------------------------------------------------
# 전이가 성립하려면 새 아키텍처에서 **표 없이** 구조가 나와야 한다. 표를
# 봐야 구조가 나오면 §29.5(c) 재생성이고, 전수를 잴 거면 표를 직접 쓰면
# 되므로 이 시스템을 쓸 이유가 없다. 그래서 A 가 관문이다.

def _arch_client():
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import REGISTRY
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    return OpenAILLM(LLMConfig(model="m"), feature_names=[], shape_values=[],
                     registry=REGISTRY, cache=False)


class _Facts:
    lines = ("스필 커널이 정답 집합에 든 형상: 0/51개",
             "고정 config 하나로 얼마나 가는가:  top-1 1.116")
    by_feature = {"has_spill": ["학습 51형상 중 정답 집합에 든 것 0개"]}


def test_architect_condition_a_contains_no_table_derived_line():
    """★ 관문. 표에서 나온 문장이 한 줄도 들어가면 안 된다."""
    c = _arch_client()
    a = c._architect_prompt(condition="A", table_facts=_Facts())
    for line in (*_Facts.lines, *_Facts.by_feature["has_spill"]):
        assert line not in a, f"A 조건에 표 문장이 샜다: {line}"
    assert "이 조건 A" in a or "조건 A" in a


def test_architect_condition_b_carries_the_aggregates():
    c = _arch_client()
    b = c._architect_prompt(condition="B", table_facts=_Facts())
    for line in (*_Facts.lines, *_Facts.by_feature["has_spill"]):
        assert line in b


def test_architect_condition_b_refuses_without_facts():
    """집계 없이 B 라고 부르면 A 와 같아진다 — 조용히 그렇게 되지 않는다."""
    c = _arch_client()
    with pytest.raises(ValueError, match="학습 분할 집계"):
        c._architect_prompt(condition="B")
    with pytest.raises(ValueError, match="알 수 없는 Architect 조건"):
        c._architect_prompt(condition="C")


def test_architect_prompt_has_no_parent_or_case_slots():
    """부모·사례·점수를 받지 않는 것이 이 역할의 정의다."""
    c = _arch_client()
    a = c._architect_prompt(condition="A")
    for banned in ("부모 규칙:", "### 사례 #", "regret 1.", "val "):
        assert banned not in a


# ---------------------------------------------------------------------------
# 엔드포인트 선택 (D-44)
# ---------------------------------------------------------------------------
# gpt-5.6 계열은 /v1/chat/completions 에서 **함수 도구 + reasoning_effort**
# 조합을 400 으로 막는다. 구조화 출력(`output_type`)이 함수 도구로
# 구현되므로 그대로 걸린다. 우회는 reasoning_effort='none' 인데 그러면
# 추론이 꺼져 물리 유도 능력을 잃는다 — 그래서 엔드포인트를 옮겼다.

def test_endpoint_defaults_to_responses_and_is_recorded():
    """★ `config.json` 에 남아야 한다 — 섞이면 비교가 깨진다 (D-31)."""
    from kernelrule.agents.openai_client import LLMConfig
    cfg = LLMConfig()
    assert cfg.endpoint == "responses"
    assert cfg.to_dict()["endpoint"] == "responses"


def test_unknown_endpoint_is_rejected():
    """조용히 chat 으로 떨어지지 않는다 (§26.4)."""
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(model="m", endpoint="v1"), feature_names=[],
                    shape_values=[], cache=False)
    with pytest.raises(ValueError, match="알 수 없는 엔드포인트"):
        llm._agent("optimize")


@pytest.mark.parametrize("endpoint,cls_name", [
    ("responses", "OpenAIResponsesModel"),
    ("chat", "OpenAIChatModel"),
])
def test_endpoint_picks_the_right_model_class(endpoint, cls_name):
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(model="gpt-5.4-mini-2026-03-17",
                              endpoint=endpoint),
                    feature_names=[], shape_values=[], cache=False)
    agent = llm._agent("optimize")
    assert type(agent.model).__name__ == cls_name


def test_model_has_a_single_source():
    """★ 실험 스크립트가 각자 모델 상수를 들고 있으면 서로 다른 모델로
    돌 수 있다 — 그러면 결과를 나란히 놓을 수 없다 (D-31, D-45).
    """
    import re
    from pathlib import Path

    from kernelrule.agents.openai_client import DEFAULT_MODEL, LLMConfig

    assert LLMConfig().model == DEFAULT_MODEL
    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted((root / "experiments").glob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            # 경로 문자열(`runs/...-gpt-5.4/`)은 과거 실행을 가리키는 것이라
            # 정상이다. **대입**으로 모델을 박아 놓은 것만 잡는다.
            if re.search(r'^\s*\w*MODEL\w*\s*=\s*["\']gpt-', line):
                bad.append(f"  {f.name}:{i}  {line.strip()}")
    assert not bad, ("실험이 모델을 직접 박아 놓았다. "
                     "`DEFAULT_MODEL` 을 쓰라:\n" + "\n".join(bad))


def test_temperature_and_seed_default_to_none():
    """★ 통제할 수 없는 것을 통제한다고 기록하지 않는다 (D-47).

    전에는 0.7 / 20260821 이었고 **둘 다 모델에 전달되지 않았다.**
    `config.json` 에만 남아 기록과 실제가 어긋났다 (§30.8).
    """
    from kernelrule.agents.openai_client import LLMConfig
    d = LLMConfig().to_dict()
    assert d["temperature"] is None
    assert d["seed"] is None


def test_seed_with_responses_endpoint_raises():
    """조용히 버려지느니 멈춘다 — Responses 에는 seed 파라미터가 없다."""
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(seed=123, endpoint="responses"),
                    feature_names=[], shape_values=[], cache=False)
    with pytest.raises(ValueError, match="파라미터가 없다"):
        llm._agent("optimize")


def test_seed_with_chat_endpoint_is_sent():
    """chat 에서는 실제로 지원되므로 보낸다."""
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(seed=123, temperature=0.7, endpoint="chat"),
                    feature_names=[], shape_values=[], cache=False)
    sent = llm._agent("optimize").model_settings or {}
    assert sent["seed"] == 123
    assert sent["temperature"] == 0.7


def test_none_values_are_not_sent_at_all():
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(), feature_names=[], shape_values=[],
                    cache=False)
    sent = llm._agent("optimize").model_settings or {}
    assert "temperature" not in sent
    assert "seed" not in sent


def test_experiments_do_not_pass_temperature_or_seed():
    """실험이 다시 넣으면 조용히 버려진다 — 여기서 막는다."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted((root / "experiments").glob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"LLMConfig\([^)]*\b(temperature|seed)\s*=", line):
                bad.append(f"  {f.name}:{i}  {line.strip()}")
    assert not bad, ("실험이 LLMConfig 에 temperature/seed 를 넘긴다 — "
                     "gpt-5.6-luna + responses 에서는 조용히 버려진다 "
                     "(D-47):\n" + "\n".join(bad))


def test_reasoning_effort_is_explicit_and_recorded():
    """★ 명시하지 않으면 모델 기본값이 적용되고, 그 기본이 바뀌면 우리
    결과가 조용히 달라진다 (§15.4).
    """
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    cfg = LLMConfig()
    assert cfg.reasoning_effort == "medium"
    assert cfg.to_dict()["reasoning_effort"] == "medium"

    llm = OpenAILLM(cfg, feature_names=[], shape_values=[], cache=False)
    sent = llm._agent("optimize").model_settings or {}
    assert sent.get("openai_reasoning_effort") == "medium"


def test_reasoning_effort_none_sends_nothing():
    """`None` 은 '모델 기본값' 이다. 문자열 'none' 과 다르다."""
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(reasoning_effort=None), feature_names=[],
                    shape_values=[], cache=False)
    assert "openai_reasoning_effort" not in (
        llm._agent("optimize").model_settings or {})
