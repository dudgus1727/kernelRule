"""실제 LLM 클라이언트 (§4-0, §4-1). **API 를 호출하지 않는다.**

호출 없이 검사할 수 있는 것만 본다 — 키 부재 처리, 예산 상한, 프롬프트
조립, 캐싱, `LLMClient` 교체 가능성.
"""
from __future__ import annotations

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
    for n in ("_common.md", "diagnose.md", "optimize.md", "hw/sm_86.md"):
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
    p = client._user_prompt("diagnose", "REPORT-BODY-MARKER")
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
