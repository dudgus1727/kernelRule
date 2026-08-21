"""라운드 루프와 아카이브 (§13, §14)."""
from __future__ import annotations

import numpy as np
import pytest

from kernelrule.agents.mock import MockLLM
from kernelrule.core.archive import CELL_AXES, Archive, Elite, cell_of
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
        assert r.llm_calls["diagnose"] == (0 if i == 0 else 1)


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
    from kernelrule.core.loop import VAL_GAP_ALARM
    from kernelrule.core.archive import Elite

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
    from kernelrule.core.splits import (MIN_REGIME_FRAC, check_balance,
                                        split_by_M_range)
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
