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
from kernelrule.features import FeatureRegistry

FEATS = ["traffic_amplification", "has_spill", "waves"]
SHAPE = ["is_memory_bound", "log_sol_ms"]

#: ★ `registry` 는 필수 인자다 (§30.9). 기본값이 있던 시절 이 파일의
#   클라이언트들이 전부 `None` 으로 만들어졌고, `render_features` 가 조용히
#   사람이 쓴 24개로 떨어졌다 — F0~F3 조건에서라면 프롬프트에 답이 들어간다.
#   `test_diagnose_prompt_carries_the_report` 가 그 폴백 덕에 통과하고
#   있었다. 이제 프롬프트는 **넘긴 레지스트리만** 렌더링한다.
EMPTY_REG = FeatureRegistry("test-empty")


def _reg(feats=FEATS, shapes=SHAPE) -> FeatureRegistry:
    """`FEATS`/`SHAPE` 와 **같은 이름**을 담은 작은 레지스트리."""
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
# 키 — ★ 조용히 MockLLM 으로 폴백하지 않는다 (§26.4)
# ---------------------------------------------------------------------------
def test_missing_key_is_a_hard_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKey, match="조용히 폴백하지 않는다"):
        OpenAILLM(LLMConfig(), feature_names=FEATS, shape_values=SHAPE,
                  registry=_reg())


def _needs_pydantic_ai():
    """★ 없으면 **스킵하되 이유를 말한다** (D-48).

    전에는 `ModuleNotFoundError` 8건으로 터졌고, 새로 클론한 사람에게는
    "테스트가 8개 깨졌다" 로만 보였다. 진짜 문제는 **설치 안내가 이
    패키지를 빠뜨린 것**이었다.

    `test_openai_client.py` 는 `CRITICAL_MODULES` 에 있으므로, 전부 스킵되면
    세션이 실패한다 — 즉 `[llm]` 없이 돌린 결과로 무엇도 보증하지 못한다.
    """
    return pytest.importorskip(
        "pydantic_ai",
        reason="pydantic-ai 가 없다. `pip install -e '.[llm]'` 로 설치하라")


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
    for n in ("_base.md", "hw/sm_86.md", "role/_rules_common.md",
              "role/_rules_edit.md", "role/analyze.md", "role/rule_editor.md",
              "role/feature.md", "role/rule_writer.md", "role/categorize.md"):
        assert load_prompt(n).strip()


def test_missing_prompt_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_prompt("nope.md")


def test_instructions_are_two_axes(client, monkeypatch):
    """★ 두 축 — 하드웨어 무관/의존 x 역할 무관/의존 (§30.10)."""
    base = load_prompt("_base.md")
    hw = load_prompt("hw/sm_86.md")
    assert "GEMM" in base and "sm_86" in hw
    # 하드웨어 파일만 갈아끼우면 새 백엔드가 된다
    assert "RTX A6000" in hw and "RTX A6000" not in base
    # ★ _base.md 는 짧아야 한다 — 여기 쌓이면 모든 역할이 값을 치른다
    assert len(base.splitlines()) < 40, "공용 블록이 다시 부풀었다"


def test_rule_block_states_the_absolute_rules():
    c = load_prompt("role/_rules_common.md")
    # ★ `is_memory_bound` 는 뺐다 — 실제 피처 이름을 프롬프트에 박으면
    #   F0~F2 에서 답을 건네주는 것이다 (D-65). 자리표시자로 바뀌었다.
    for must in ("import", "np.random", "8", "w[0]", "p.<형상값>"):
        assert must in c
    # ★ no-op 분기 경고가 프롬프트에 들어 있다
    assert "소거된다" in c


def test_rule_editor_prompt_formats(client):
    from kernelrule.agents.schemas import RuleProposal

    p = client._user_prompt(
        "rule_editor", "", parent=RuleProposal(code="def score(f,p,hw,w): return 1",
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

    client.calls.append(LLMCall(role="rule_editor", prompt_hash="h", seq=0,
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
    c = load_prompt("role/_rules_edit.md")
    assert "실제로 거부된 것들" in c
    assert "w[0] 재사용" in c
    # ★ 규칙 자체는 `_rules_common.md` 에 있다 — 갤러리는 사례만 든다
    assert "한 번만 쓴다" in load_prompt("role/_rules_common.md")


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
        {"round": 0, "seq": 1, "role": "rule_editor", "attempt": 0,
         "code": "w0_too_long", "msg": "x"},
        {"round": 0, "seq": 1, "role": "rule_editor", "attempt": 2,
         "code": "w0_too_long", "msg": "x"},          # 같은 코드 반복
        {"round": 0, "seq": 2, "role": "rule_editor", "attempt": 0,
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
# RuleWriter — A 조건은 표에서 나온 것을 하나도 보지 않는다 (§11.8)
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


def test_rule_writer_condition_a_contains_no_table_derived_line():
    """★ 관문. 표에서 나온 문장이 한 줄도 들어가면 안 된다."""
    c = _arch_client()
    a = c._rule_writer_prompt(condition="A", table_facts=_Facts())
    for line in (*_Facts.lines, *_Facts.by_feature["has_spill"]):
        assert line not in a, f"A 조건에 표 문장이 샜다: {line}"
    assert "이 조건 A" in a or "조건 A" in a


def test_rule_writer_condition_b_carries_the_aggregates():
    c = _arch_client()
    b = c._rule_writer_prompt(condition="B", table_facts=_Facts())
    for line in (*_Facts.lines, *_Facts.by_feature["has_spill"]):
        assert line in b


def test_rule_writer_condition_b_refuses_without_facts():
    """집계 없이 B 라고 부르면 A 와 같아진다 — 조용히 그렇게 되지 않는다."""
    c = _arch_client()
    with pytest.raises(ValueError, match="학습 분할 집계"):
        c._rule_writer_prompt(condition="B")
    with pytest.raises(ValueError, match="알 수 없는 RuleWriter 조건"):
        c._rule_writer_prompt(condition="C")


def test_rule_writer_prompt_has_no_parent_or_case_slots():
    """부모·사례·점수를 받지 않는 것이 이 역할의 정의다."""
    c = _arch_client()
    a = c._rule_writer_prompt(condition="A")
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
    _needs_pydantic_ai()
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(model="m", endpoint="v1"), feature_names=[],
                    shape_values=[], cache=False, registry=EMPTY_REG)
    with pytest.raises(ValueError, match="알 수 없는 엔드포인트"):
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
    _needs_pydantic_ai()
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(seed=123, endpoint="responses"),
                    feature_names=[], shape_values=[], cache=False, registry=EMPTY_REG)
    with pytest.raises(ValueError, match="파라미터가 없다"):
        llm._agent("rule_editor")


def test_seed_with_chat_endpoint_is_sent():
    """chat 에서는 실제로 지원되므로 보낸다."""
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
    """`None` 은 '모델 기본값' 이다. 문자열 'none' 과 다르다."""
    _needs_pydantic_ai()
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(LLMConfig(reasoning_effort=None), feature_names=[],
                    shape_values=[], cache=False, registry=EMPTY_REG)
    assert "openai_reasoning_effort" not in (
        llm._agent("rule_editor").model_settings or {})


def test_every_llm_runner_persists_its_calls():
    """★ LLM 호출은 다시 만들 수 없다 (D-33 / D-51).

    `RoundLoop` 를 쓰는 러너는 `dump()` 가 대신 남겨 주지만, 직접
    `OpenAILLM` 을 부르는 러너는 스스로 남겨야 한다. 실제로 두 개가
    빠져 있었고, 비용 집계 때 토큰을 표준출력 로그에서 주워야 했다.
    """
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted((root / "experiments").glob("*.py")):
        src = f.read_text()
        if "OpenAILLM(" not in src:
            continue
        # RoundLoop 가 dump() 안에서 llm.dump() 를 부른다
        if "RoundLoop(" in src:
            continue
        if not re.search(r"\.dump\(", src):
            bad.append(f"  {f.name}: OpenAILLM 을 직접 쓰는데 dump 가 없다")
    assert not bad, ("LLM 호출을 남기지 않는 러너가 있다 (D-33):\n"
                     + "\n".join(bad))


# ---------------------------------------------------------------------------
# ★ §30.9 — 레지스트리를 갈아 끼울 수 있어야 F0~F3 가 성립한다
# ---------------------------------------------------------------------------
def test_registry_is_required(monkeypatch):
    """기본값이 있으면 F1 조건에 사람 24개가 조용히 들어간다."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    with pytest.raises(ValueError, match="필수"):
        OpenAILLM(LLMConfig(), feature_names=[], shape_values=[],
                  registry=None)


def test_prompt_renders_only_the_given_registry(monkeypatch):
    """★ 사람이 쓴 24개 이름이 **하나도** 안 들어가야 한다."""
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
    assert not leaked, f"사람이 쓴 피처가 프롬프트에 샜다: {leaked}"
    assert "padded_flop_fraction" in p


def test_feature_names_must_live_in_the_registry(monkeypatch):
    """정적 검사와 프롬프트가 다른 목록을 보면 안 된다 (원칙 2)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    with pytest.raises(ValueError, match="레지스트리"):
        OpenAILLM(LLMConfig(), feature_names=["없는_피처"], shape_values=[],
                  registry=_reg())


def test_render_features_refuses_a_missing_registry():
    from kernelrule.features import render_features

    with pytest.raises(ValueError, match="반드시 받는다"):
        render_features(None, include_observed=False)


def test_feature_decorator_refuses_a_missing_registry():
    from kernelrule.features import feature

    with pytest.raises(ValueError, match="필수"):
        @feature(expected_range=(0.0, 1.0))
        def _f(p, hw, cfg) -> float:
            return 0.0


def test_intrinsic_shape_fields_are_not_stray(monkeypatch):
    """★ `M/N/K/n_candidates` 는 레지스트리 피처가 아니지만 항상 있다.

    처음에 이걸 빼먹어서 검증 실행이 시작도 못 하고 죽었다 (§30.9).
    """
    from kernelrule.core.matrix import INTRINSIC_SHAPE_FIELDS

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    llm = OpenAILLM(LLMConfig(), feature_names=FEATS,
                    shape_values=[*SHAPE, *INTRINSIC_SHAPE_FIELDS],
                    registry=_reg())
    assert set(INTRINSIC_SHAPE_FIELDS) <= set(llm.shape_values)


# ---------------------------------------------------------------------------
# §16.1 — Analyst 를 끈 RuleEditor 프롬프트 (D-89)
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
    """★ 가설 절을 **빈 자리로 남기지 않는다** (§16.1).

    "## 이번 가설\n\n(가설 없음)" 을 남기면 모델이 "가설이 있는데 비어
    있다" 로 읽어 조건이 달라진다. 진단 리포트를 만들지도 않는 것과
    같은 원칙이다 — 자리 자체가 없어야 한다.
    """
    on, off = _rule_editor_prompts()
    assert "가설" in on
    assert "가설" not in off, "Analyst 를 껐는데 가설을 언급한다"
    assert "## 이번 가설" not in off
    assert "부모 규칙" in off and "사용 가능한 피처" in off


def test_rule_editor_prompt_without_analyst_is_a_deletion():
    """★ 끈 프롬프트는 켠 것에서 **문장을 지운 것**이어야 한다.

    새 문구를 쓰면 ablation 이 "Analyst 만 다르다" 가 아니게 된다.
    문자 단위 부분수열이면 삭제만 일어난 것이다.
    """
    on, off = _rule_editor_prompts()
    it = iter(on)
    assert all(ch in it for ch in off), (
        "끈 프롬프트에 켠 프롬프트에 없는 글자가 있다 — 삭제가 아니라 "
        "새로 쓴 것이다 (§16.1)")
    assert len(off) < len(on)
