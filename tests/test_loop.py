"""라운드 루프와 아카이브 (§13, §14)."""
from __future__ import annotations

import numpy as np
import pytest

from kernelrule.agents.mock import MockLLM
from kernelrule.core.archive import CELL_AXES, Archive, Elite
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.splits import Split, SplitSet


def _elite(regret=1.2, short=1.2, long=1.2, n=100, rnd=0, rid="r1"):
    return Elite(rule_id=rid, code="x", w=[1.0], regret=regret,
                 short_regret=short, long_regret=long, code_len=n, round=rnd)


# ---------------------------------------------------------------------------
# 아카이브
# ---------------------------------------------------------------------------
def test_cell_axes_are_64_cells():
    n = 1
    for edges in CELL_AXES.values():
        n *= len(edges) - 1
    assert n == 64


def test_specialist_survives_even_with_bad_overall():
    """★ 전체 점수가 낮아도 **특정 영역 최고면 살려둔다** (§13.1)."""
    a = Archive()
    a.consider(_elite(regret=1.12, short=1.20, long=1.10, rid="all"))
    won = a.consider(_elite(regret=1.24, short=1.02, long=1.40, rid="shortspec"))
    assert "new_cell" in won, "짧은 형상 전문가가 버려졌다"
    assert a.best.rule_id == "all"
    assert any(e.rule_id == "shortspec" for e in a.cells.values())


def test_noise_tolerance_blocks_meaningless_updates():
    """"조금 좋아졌다" 로 갱신하면 아카이브가 노이즈를 축적한다 (§13.4)."""
    a = Archive(noise_tol=0.01)
    a.consider(_elite(regret=1.20, rid="a"))
    assert a.consider(_elite(regret=1.195, rid="b")) == []
    assert "best" in a.consider(_elite(regret=1.15, rid="c"))


def test_parent_mix_is_exploit_explore_cross():
    """6 착실 / 3 탐색 / 3 교차 (§13.3)."""
    a = Archive()
    for i in range(6):
        a.consider(_elite(regret=1.3 - 0.01 * i, short=1.0 + 0.05 * i,
                          long=1.0 + 0.02 * i, n=50 + 30 * i, rid=f"r{i}"))
    kinds = [k for k, _ in a.parents(12, np.random.default_rng(0))]
    assert kinds.count("exploit") == 6
    assert kinds.count("explore") == 3
    assert kinds.count("cross") == 3


def test_new_cell_round_is_tracked():
    a = Archive()
    a.consider(_elite(rnd=0, rid="a"))
    a.consider(_elite(short=1.02, rnd=4, rid="b"))
    assert a.last_new_cell_round == 4


# ---------------------------------------------------------------------------
# 루프
# ---------------------------------------------------------------------------
@pytest.fixture
def loop(synth_table, tmp_path):
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY

    fm = FeatureMatrix(synth_table, REGISTRY)
    sh = synth_table.shapes()
    splits = SplitSet(train=Split("train", tuple(sh[:-2])),
                      val=Split("val", tuple(sh[-2:])))
    llm = MockLLM("mutate", seed=1, feature_names=fm.feature_names(),
                  shape_values=["is_memory_bound"])
    cfg = LoopConfig(run_id="test", n_rules_per_round=4, max_rounds=3,
                     max_evals=40, seed=0, sandbox_first_seen=False,
                     out_dir=str(tmp_path))
    return RoundLoop(cfg=cfg, table=synth_table, matrix=fm, splits=splits,
                     llm=llm)


def test_loop_runs_and_fills_the_archive(loop):
    loop.run(3, verbose=False)
    assert len(loop.rounds) == 3
    assert loop.archive.best is not None
    assert loop.archive.n_cells >= 1
    assert sum(r.n_scored for r in loop.rounds) > 0


def test_llm_call_budget_matches_the_design(loop):
    """라운드당 진단 1회 + 규칙 n회 (§11.1 — 호출의 약 89%가 Optimizer)."""
    loop.run(3, verbose=False)
    for i, r in enumerate(loop.rounds):
        assert r.llm_calls["optimize"] == loop.cfg.n_rules_per_round
        # 1라운드는 아카이브가 비어 진단을 건너뛴다
        assert r.llm_calls["analyze"] == (0 if i == 0 else 1)


def test_adversarial_mode_scores_nothing(synth_table, tmp_path):
    """★ adversarial 모드에서는 **하나도 채점되면 안 된다** (§24.3)."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY

    fm = FeatureMatrix(synth_table, REGISTRY)
    sh = synth_table.shapes()
    splits = SplitSet(train=Split("train", tuple(sh[:-2])),
                      val=Split("val", tuple(sh[-2:])))
    llm = MockLLM("adversarial", seed=0, feature_names=fm.feature_names())
    cfg = LoopConfig(run_id="adv", n_rules_per_round=12, max_rounds=1,
                     max_evals=20, seed=0, out_dir=str(tmp_path))
    lp = RoundLoop(cfg=cfg, table=synth_table, matrix=fm, splits=splits,
                   llm=llm)
    r = lp.run_round()
    assert r.n_proposed == 12
    assert r.n_scored == 0, f"적대적 코드가 {r.n_scored}개 채점됐다"
    assert lp.archive.best is None


def test_loop_dump_writes_everything(loop, tmp_path):
    loop.run(2, verbose=False)
    d = loop.dump()
    for name in ("archive.jsonl", "rounds.jsonl", "failures.jsonl",
                 "hypotheses.jsonl"):
        assert (d / name).exists(), name
    assert (d / "llm_calls").exists()


def test_replay_reproduces_the_run(synth_table, tmp_path):
    """★ 같은 LLM 응답으로 결과가 재현된다 (§24.4)."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY

    fm = FeatureMatrix(synth_table, REGISTRY)
    sh = synth_table.shapes()
    splits = SplitSet(train=Split("train", tuple(sh[:-2])),
                      val=Split("val", tuple(sh[-2:])))

    def build(llm, run_id):
        cfg = LoopConfig(run_id=run_id, n_rules_per_round=3, max_rounds=2,
                         max_evals=30, seed=0, sandbox_first_seen=False,
                         out_dir=str(tmp_path))
        return RoundLoop(cfg=cfg, table=synth_table, matrix=fm,
                         splits=splits, llm=llm)

    a = build(MockLLM("mutate", seed=4, feature_names=fm.feature_names()), "a")
    a.run(2, verbose=False)
    a.llm.dump(tmp_path / "calls")
    b = build(MockLLM("replay", replay_dir=tmp_path / "calls"), "b")
    b.run(2, verbose=False)
    assert [r.best_regret for r in a.rounds] == [r.best_regret
                                                 for r in b.rounds]


def test_early_stop_uses_the_validation_split(loop):
    """★ 조기 종료는 **검증 분할**로 판정한다 (§10.2, §14.3)."""
    import inspect
    src = inspect.getsource(RoundLoop.should_stop)
    assert "self.splits.val" in src
    assert "self.splits.train" not in src


def test_early_stop_needs_both_conditions(loop):
    """점수가 멈춰도 새 셀이 나오면 계속 돈다 (§14.3)."""
    loop.cfg.patience = 2
    loop.run(3, verbose=False)
    # 새 셀이 최근에 나왔으면 멈추지 않는다
    loop.archive.last_new_cell_round = len(loop.rounds) - 1
    for r in loop.rounds:
        r.best_val_regret = 1.2
    stop, _ = loop.should_stop()
    assert not stop, "새 셀이 나왔는데 멈췄다"
    loop.archive.last_new_cell_round = -1
    stop, why = loop.should_stop()
    assert stop and "새 셀도 없다" in why


def test_duplicate_code_is_not_rescored(loop):
    """같은 코드가 나오면 재채점하지 않는다 (§15.4)."""
    loop.run(2, verbose=False)
    assert len(loop._seen_code) <= sum(r.n_scored for r in loop.rounds)


def test_seed_puts_the_baseline_in_the_archive(loop):
    """★ 씨앗이 없으면 루프가 **다른 규칙의 리포트**를 본다.

    손규칙을 기준선으로 "리포트를 읽고 그것을 고칠 수 있는가" 를 시험하려면
    거기서 출발해야 한다. 처음 20라운드를 씨앗 없이 돌려서 실험 자체가
    엉뚱한 것을 재고 있었다.
    """
    code = ("def score(f, p, hw, w):\n"
            "    return f.traffic_amplification * w[0]\n")
    e = loop.seed(code, [1.0], changes="baseline")
    assert loop.archive.best is not None
    assert loop.archive.best.rule_id == e.rule_id
    assert e.round == -1
    # 같은 코드를 다시 채점하지 않는다
    assert code.strip() in loop._seen_code


def test_seed_rejects_a_bad_rule(loop):
    with pytest.raises(ValueError, match="초기 규칙이 거부됐다"):
        loop.seed("def score(f, p, hw, w):\n    return f.nope * w[0]\n", [1.0])


def test_val_blowup_is_reported_not_hidden(loop):
    """★ 아카이브는 **학습** 점수로 고른다 — 검증에서 무너지는 규칙이
    "최고" 가 될 수 있다. 실제로 났다 (train 1.164 / val 6.085).

    선택 규칙은 그대로 두되(검증을 쓰면 홀드아웃이 오염된다) **경보를 낸다.**
    """
    from kernelrule.core.archive import Elite
    from kernelrule.core.loop import VAL_GAP_ALARM

    loop.run(1, verbose=False)
    # 진단 리포트가 최고 규칙을 컴파일하므로 유효한 코드여야 한다
    bad = Elite(rule_id="bad",
                code="def score(f, p, hw, w):\n"
                     "    return f.waves * w[0]\n", w=[1.0], regret=1.0,
                short_regret=1.0, long_regret=1.0, code_len=10, round=0,
                val_regret=1.0 + VAL_GAP_ALARM * 10)
    loop.archive.consider(bad)
    r = loop.run_round()
    assert r.n_val_blowups >= 1, "검증 폭발이 보고되지 않았다"
    assert "폭발" in r.line()


# ---------------------------------------------------------------------------
# ★ 체제 균형 (§10.1) — 학습이 소수 체제를 희생하는 것을 막는다
# ---------------------------------------------------------------------------
def test_regime_balance_flags_a_lopsided_train_split(real_bundle_path):
    """★ `M > 2048` 이 학습을 82%/18% 로 가른다 — 경고해야 한다.

    실측: 그 구성에서 진화가 전체 regret 을 1.177 -> 1.390 으로 악화시키면서
    학습 점수는 1.201 -> 1.118 로 좋아졌다.
    """
    import warnings

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.splits import MIN_REGIME_FRAC, check_balance, split_by_M_range
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                   ok_only=False)
        sh = [p for p in tb.shapes()
              if bool((tb.frame_for(p).align_a == 8).all()
                      and (tb.frame_for(p).align_b == 8).all()
                      and (tb.frame_for(p).align_c == 8).all())]
    sp = split_by_M_range(sh)
    with pytest.warns(UserWarning, match="소수 체제"):
        bal = check_balance(sp.train, tb.hw)
    assert not bal.ok
    assert bal.minority()[1] < MIN_REGIME_FRAC
    assert bal.counts["long"] == 9 and bal.counts["short"] == 41


def test_regime_balance_accepts_a_crossing_split(real_bundle_path):
    """체제를 가로지르는 분할은 통과한다 (실측 69%/31%)."""
    import warnings

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.splits import by_predicate, check_balance
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                   ok_only=False)
        sh = [p for p in tb.shapes()
              if bool((tb.frame_for(p).align_a == 8).all()
                      and (tb.frame_for(p).align_b == 8).all()
                      and (tb.frame_for(p).align_c == 8).all())]
    sp = by_predicate(sh, lambda p: (p.N, p.K) == (11008, 4096), name="nk")
    bal = check_balance(sp.train, tb.hw)      # 경고가 나면 안 된다
    assert bal.ok and bal.counts["long"] == 16


def test_balance_check_is_strictable():
    """`strict=True` 면 에러다. 조용히 통과시키지 않는다 (§26.4)."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.splits import Split, SplitError, check_balance
    from kernelrule.core.types import Hardware, Problem

    hw = Hardware(name="t", arch="sm_86", sm_count=84, smem_per_block=101376,
                  max_threads_per_sm=1536, regs_per_sm=65536,
                  peak_tflops_f16=116.1, bandwidth_gbps=729.7,
                  l2_bytes=6291456)
    tiny = [Problem(1, 4096, 4096)] * 9 + [Problem(8192, 8192, 8192)]
    with pytest.raises(SplitError, match="소수 체제"):
        check_balance(Split("train", tuple(tiny)), hw, strict=True)


def test_cell_axes_use_size_regimes():
    """★ 셀 축이 크기 체제다 — mem/comp 가 아니다 (§10.1, §30.5)."""
    from kernelrule.core.archive import CELL_AXES

    assert set(CELL_AXES) == {"code_len", "short_regret", "long_regret"}


def test_regime_gap_is_exposed():
    e = _elite(regret=1.1, short=1.05, long=1.40)
    assert e.regime_gap == pytest.approx(0.35)


def test_loop_warns_when_train_has_one_regime(synth_table, tmp_path):
    """학습이 한 체제만 담으면 셀 축이 무의미해진다 — 경고한다."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import Split, SplitSet
    from kernelrule.features import REGISTRY

    fm = FeatureMatrix(synth_table, REGISTRY)
    sh = list(synth_table.shapes())
    import math

    from kernelrule.core.splits import _DUMMY_CFG
    from kernelrule.features.physical import log_sol_ms
    short = [p for p in sh if log_sol_ms(p, synth_table.hw, _DUMMY_CFG)
             < math.log2(0.5)]
    if len(short) < 2 or len(short) == len(sh):
        pytest.skip("합성 격자에 두 체제가 다 있어야 이 시험이 성립한다")
    splits = SplitSet(train=Split("train", tuple(short)),
                      val=Split("val", tuple(p for p in sh
                                              if p not in short)))
    llm = MockLLM("mutate", seed=0, feature_names=fm.feature_names())
    cfg = LoopConfig(run_id="one", n_rules_per_round=2, max_rounds=1,
                     max_evals=20, sandbox_first_seen=False,
                     out_dir=str(tmp_path))
    with pytest.warns(UserWarning, match="한 크기 체제"):
        RoundLoop(cfg=cfg, table=synth_table, matrix=fm, splits=splits,
                  llm=llm)


# ---------------------------------------------------------------------------
# LLM 전송 실패를 스키마 거부와 가른다 (D-43)
# ---------------------------------------------------------------------------
# HTTP 429(크레딧 소진)가 "거부 스키마 144건" 으로 집계됐다. 인프라 실패가
# 로그에서 **모델의 실패로 보인다** — D-39 와 같은 부류다.

@pytest.mark.parametrize("exc,transport", [
    (RuntimeError("status_code: 429, body: You have no credits remaining"), True),
    (RuntimeError("invalid_api_key"), True),
    (ConnectionError("connection reset"), False),
    (ValueError("Exceeded maximum output retries (3)"), False),
    (ValueError("가중치를 여러 항에 재사용했다"), False),
])
def test_transport_errors_are_told_apart(exc, transport):
    from kernelrule.core.loop import _is_transport_error
    assert _is_transport_error(exc) is transport


def test_named_transport_exceptions_are_caught():
    from kernelrule.core.loop import _is_transport_error

    class ModelHTTPError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    for cls in (ModelHTTPError, APIConnectionError):
        assert _is_transport_error(cls("무관한 본문"))


def test_round_of_total_transport_failure_stops_the_run():
    """★ 크레딧 문제는 저절로 낫지 않는다. 남은 라운드를 태우지 않는다."""
    from kernelrule.core.loop import LLMUnreachable, RoundResult

    res = RoundResult(round=0, n_proposed=12, n_llm_error=12)
    res.rejections.append(("llm-transport", "429 no credits"))
    assert res.n_llm_error == res.n_proposed
    assert "★LLM오류 12" in res.line()
    assert issubclass(LLMUnreachable, RuntimeError)


def test_dump_records_what_it_ran_with(loop, tmp_path):
    """★ 무엇으로 돌렸는지 없으면 나중에 나란히 놓을 수 없다 (D-51).

    30개 실행 중 2개만 `config.json` 이 있었다 — `dump()` 가 안 썼고,
    그 둘은 다른 스크립트가 쓴 것이었다.
    """
    import json

    cfg = json.loads((loop.dump(tmp_path / "d") / "config.json").read_text())
    assert cfg["loop"]["run_id"] == "test"
    assert cfg["split"]["n_train"] == len(loop.splits.train.shapes)
    assert cfg["split"]["n_val"] == len(loop.splits.val.shapes)
    assert cfg["n_features"] > 0
    # MockLLM 이면 클래스 이름이라도 남아야 한다 — 빈칸이면 안 된다
    assert cfg["llm"]


# ---------------------------------------------------------------------------
# D-75 — Analyst -> FeatureWriter 경로
# ---------------------------------------------------------------------------
#: 씨앗 규칙. 미사용 피처가 남아야 목 Analyst 가 가설을 낸다.
_SEED_RULE = (("def score(f, p, hw, w):\n"
               "    return np.log2(f.traffic_amplification) * w[0]\n"), [1.0])


def _d75_loop(synth_table, tmp_path, *, cap: int):
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY, FeatureRegistry

    # ★ 레지스트리를 **복제**한다. 루프가 여기에 축을 더하므로, 전역
    #   `REGISTRY` 를 그대로 쓰면 다른 시험으로 새 나간다.
    reg = FeatureRegistry("d75")
    for name in REGISTRY.names():
        reg.add(REGISTRY[name])
    fm = FeatureMatrix(synth_table, reg)
    sh = synth_table.shapes()
    splits = SplitSet(train=Split("train", tuple(sh[:-2])),
                      val=Split("val", tuple(sh[-2:])))
    cfg = LoopConfig(run_id="d75", n_rules_per_round=2, max_rounds=1,
                     max_evals=30, seed=0, sandbox_first_seen=False,
                     out_dir=str(tmp_path),
                     max_new_features_per_round=cap)
    llm = MockLLM("mutate", seed=1, feature_names=fm.feature_names())
    return RoundLoop(cfg=cfg, table=synth_table, matrix=fm, splits=splits,
                     llm=llm), reg


def test_feature_path_is_off_by_default(synth_table, tmp_path):
    """★ 기본값은 **꺼짐**이다 — 지금까지의 실행과 같은 조건이어야 한다."""
    loop, _reg = _d75_loop(synth_table, tmp_path, cap=0)
    assert loop.cfg.max_new_features_per_round == 0
    loop.seed(*_SEED_RULE)
    r = loop.run_round()
    assert r.n_feature_requests == 0 and r.n_features_made == 0
    assert loop.features_made == []
    assert r.llm_calls.get("feature", 0) == 0


def test_analyst_request_reaches_the_feature_writer(synth_table, tmp_path):
    """★ 요구가 **버려지지 않는다** (D-75).

    33실행에서 303번 채워진 필드를 `loop.py` 가 안 읽고 있었다. 경로가
    생겼는지는 "요구가 있었다" 와 "피처 호출이 있었다" 로 본다.
    """
    loop, reg = _d75_loop(synth_table, tmp_path, cap=1)
    n_before = len(reg._items)
    loop.seed(*_SEED_RULE)
    r = loop.run_round()
    assert r.n_feature_requests >= 1, "요구를 못 읽었다"
    assert r.llm_calls.get("feature", 0) >= 1, "FeatureWriter 를 안 불렀다"
    assert loop.features_made, "시도 기록이 없다"
    row = loop.features_made[0]
    assert row["requirement"], "요구 문장이 안 실렸다"
    print("D-75 시도:", row)          # -s 로 보면 무엇이 만들어졌는지 나온다
    if row.get("accepted"):
        assert len(reg._items) == n_before + 1
        assert row["name"] in loop.matrix.feature_names() \
            or row.get("shape_level"), "열이 안 만들어졌다"


def test_feature_writer_never_sees_the_diagnostic_report():
    """★ 조건 1 — 진단 리포트를 주지 않는다 (D-75).

    루프 안에서 만든 피처가 학습 형상에 맞춰지는 통로를 막는다. 넘어가는
    것은 **요구 문장 하나**뿐이어야 한다.
    """
    import ast
    import inspect

    from kernelrule.core import loop as loop_mod
    from kernelrule.core.loop import _feature_task

    # (1) 요구 문장 **말고는 아무것도 변하지 않는다.** 문자열 포함 검사로
    #     쓰면 안내문 자체("표도 사례도 보지 않고")에 걸린다 — 검사기가
    #     자기가 허용한 문구를 금지하는 D-73 과 같은 실수다.
    a = _feature_task("split-K 가 만드는 CTA 병렬성 이득")
    b = _feature_task("L2 재사용 이득")
    assert "split-K 가 만드는 CTA 병렬성 이득" in a
    assert a.replace("split-K 가 만드는 CTA 병렬성 이득", "<X>") \
        == b.replace("L2 재사용 이득", "<X>"), "요구 말고 다른 것이 변한다"

    # (2) 호출부가 **요구 문장 하나만** 넘긴다. 프롬프트 자리는 빈 문자열이다.
    import textwrap
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(loop_mod.RoundLoop._write_features)))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "complete"]
    assert len(calls) == 1, "FeatureWriter 호출이 하나가 아니다"
    c = calls[0]
    assert c.args[0].value == "feature"
    assert c.args[1].value == "", "프롬프트 자리에 리포트가 들어간다"
    kw = {k.arg for k in c.keywords}
    assert kw == {"condition", "registry", "task"}, f"넘기는 것이 늘었다: {kw}"


def test_requirement_reads_the_old_field_name():
    """두 이름을 다 읽는다 — 어느 쪽으로 만든 실행도 조용히 0건이 되면 안 된다."""
    from kernelrule.core.loop import _requirement_of

    assert _requirement_of({"needs_new_feature": "L2 재사용 이득"}) \
        == "L2 재사용 이득"
    # ★ 2026-08-28 에 잠깐 쓴 이름. 그때 만든 실행 3개를 계속 읽어야 한다
    assert _requirement_of({"physical_requirement": "잠깐 쓴 이름"}) \
        == "잠깐 쓴 이름"
    assert _requirement_of({"needs_new_feature": None}) == ""


def test_over_cap_requests_are_recorded_not_dropped(synth_table, tmp_path):
    """★ 상한에 걸린 요구를 **조용히 버리지 않는다**.

    상한은 §21 캐시 때문에 필요하지만, 넘친 요구를 안 남기면 "얼마나
    요구했나" 를 못 잰다 — 그것이 D-75 의 주 관찰이다.
    """
    from kernelrule.core.loop import RoundResult

    loop, _reg = _d75_loop(synth_table, tmp_path, cap=1)
    res = RoundResult(round=0)
    hyps = [{"id": "H0", "needs_new_feature": "L2 재사용 이득"},
            {"id": "H1", "needs_new_feature": "CTA 절대 개수"},
            {"id": "H2", "needs_new_feature": "split-K 병렬성 이득"}]
    loop._write_features(hyps, 0, res)
    assert res.n_feature_requests == 3
    assert res.n_feature_over_cap == 2
    over = [x for x in loop.features_made if x.get("over_cap")]
    assert len(over) == 2, "상한 초과 요구가 기록되지 않았다"
    assert {x["hypothesis_id"] for x in over} == {"H1", "H2"}
    assert all(x["requirement"] for x in over), "요구 문장이 안 남았다"


def test_analyze_prompt_matches_the_measured_baseline():
    """★ 요구 필드 안내는 **기준선이 측정된 문구 그대로**여야 한다 (D-81).

    17.9%(옛 6실행)는 아래 세 줄로 측정됐다. 여기에 무엇을 더하거나 빼면
    비교 대상이 달라진다 — 실제로 안내를 늘렸다가 0~5.9% 로 눌렸다
    (D-80). 억제 문구("대부분의 라운드에서는 null")도 **기준선의 일부**라
    그대로 둔다.

    바꿔야 할 이유가 생기면 이 시험을 같이 고치고, **그 실행부터 새
    계열**로 다뤄라.
    """
    from kernelrule.agents.openai_client import load_prompt

    baseline = (
        "`measurable_with` 에는 **아래 목록에 있는 이름만** 쓰세요.\n"
        "목록에 없는 물리량이 필요하면 `needs_new_feature` 에 그 이름을 쓰세요\n"
        "(대부분의 라운드에서는 `null` 입니다 — 물리량이 그렇게 많지 "
        "않습니다).")
    txt = load_prompt("role/analyze.md")
    assert baseline in txt, (
        "요구 필드 안내가 기준선 문구와 다르다. 그대로 두거나, 바꿀 거면 "
        "이 시험을 고치고 새 계열로 다뤄라 (D-81)")
    # 2026-08-28 에 넣었다가 되돌린 것들이 다시 들어오지 않았는가
    for gone in ("physical_requirement", "전달되지 않고 버려집니다",
                 "measurable_with 를 쓰는 편이 낫습니다"):
        assert gone not in txt, f"되돌린 문구가 다시 들어왔다: {gone!r}"


# ---------------------------------------------------------------------------
# §16.1 — Analyst ablation
# ---------------------------------------------------------------------------
def test_analyst_off_makes_no_analyze_call(synth_table, tmp_path):
    """★ 끄면 진단 리포트를 **만들지도 않는다** (§16.1, D-89).

    만들어 놓고 안 주면 "진단이 있는데 안 쓴다" 가 되어 조건이 달라진다.
    호출 수와 가설 수 둘 다 0 이어야 한다.
    """
    loop, _reg = _d75_loop(synth_table, tmp_path, cap=0)
    loop.cfg.use_analyst = False
    loop.seed(*_SEED_RULE)
    r = loop.run_round()
    assert r.llm_calls.get("analyze", 0) == 0, "Analyst 를 껐는데 불렀다"
    assert loop.hypotheses == []
    assert r.n_proposed > 0, "Optimizer 는 그대로 돌아야 한다"


def test_analyst_on_is_the_default(synth_table, tmp_path):
    """기본은 켬이다 — 지금까지의 모든 실행이 그 조건이다."""
    from kernelrule.core.loop import LoopConfig

    assert LoopConfig(run_id="x").use_analyst is True
    loop, _reg = _d75_loop(synth_table, tmp_path, cap=0)
    loop.seed(*_SEED_RULE)
    r = loop.run_round()
    assert r.llm_calls.get("analyze", 0) == 1


def test_borrowed_hypotheses_skip_the_same_seed_index(synth_table, tmp_path):
    """★ 대조군 C — `abl-B-s1` 의 가설을 `-s1` 에 주면 '다른 실행' 이 아니다.

    그리고 풀이 비면 **조용히 가설 없이 돌지 않는다** (§26.4).
    """
    import json

    pool = tmp_path / "f1pipe-x-s0" / "hypotheses.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text("\n".join(json.dumps(
        {"id": f"H{i}", "round": i // 2, "claim": f"c{i}", "analyst_pass": 1},
        ensure_ascii=False) for i in range(6)))

    loop, _reg = _d75_loop(synth_table, tmp_path, cap=0)
    loop.cfg.use_analyst = False
    loop.cfg.hypothesis_pool = (str(pool),)
    loop.cfg.run_id = "f1pipe-y-s1"          # 시드 번호가 다르다 -> 쓴다
    got = loop._pool_round(0)
    assert got and all("borrowed_from" in h for h in got)
    assert all("analyst_pass" not in h for h in got)

    loop2, _r2 = _d75_loop(synth_table, tmp_path, cap=0)
    loop2.cfg.use_analyst = False
    loop2.cfg.hypothesis_pool = (str(pool),)
    loop2.cfg.run_id = "f1pipe-y-s0"          # ★ 같은 시드 번호 -> 뺀다
    with pytest.raises(ValueError, match="가설 풀이 비었다"):
        loop2._pool_round(0)


def test_borrowed_arm_calls_no_analyst_but_renders_the_section(synth_table,
                                                               tmp_path):
    """★ C 는 Analyst 를 안 부르지만 **가설 절은 있어야 한다**.

    A(가설 없음)와 C(남의 가설)를 프롬프트 구조까지 같게 만들면 무엇이
    다른지 못 가른다 — C 의 차이는 **문장의 출처**뿐이어야 한다.
    """
    import json

    pool = tmp_path / "f1pipe-x-s2" / "hypotheses.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text("\n".join(json.dumps(
        {"id": f"H{i}", "round": 0, "claim": f"c{i}", "analyst_pass": 1},
        ensure_ascii=False) for i in range(3)))

    loop, _reg = _d75_loop(synth_table, tmp_path, cap=0)
    loop.cfg.use_analyst = False
    loop.cfg.hypothesis_pool = (str(pool),)
    loop.seed(*_SEED_RULE)
    r = loop.run_round()
    assert r.llm_calls.get("analyze", 0) == 0, "C 인데 Analyst 를 불렀다"
    assert loop.hypotheses, "빌려온 가설이 기록되지 않았다"
    assert all(h.get("analyst_pass") == 0 for h in loop.hypotheses)
