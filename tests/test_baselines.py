"""베이스라인 (§9, §30.5b). **세 절차를 병기한다.**"""
from __future__ import annotations

import warnings

import pytest
from toy import make_table

from kernelrule.baselines.static_topk import PROCEDURES, StaticTopK
from kernelrule.core.splits import (
    split_by_alignment,
    split_by_M_range,
    split_by_size,
    split_by_waves,
)


def test_greedy_finds_known_optimum():
    """정적 top-k 가 명백한 경우에 최적 집합을 찾는다 (§26.2).

    config 0 은 형상 A 에서만, config 1 은 형상 B 에서만 좋다. k=2 면
    둘을 합쳐 완벽해야 한다.
    """
    t = make_table({
        (1024, 4096, 4096): [1.0, 4.0, 8.0],
        (2048, 4096, 4096): [4.0, 1.0, 8.0],
    })
    r = StaticTopK(t, coverage="union").run(ks=(1, 2, 3))
    assert r.by_k[1]["all"] == pytest.approx(2.0)      # geomean(1, 4)
    assert r.by_k[2]["all"] == pytest.approx(1.0)      # 둘을 합치면 완벽
    assert r.coverage[2] == 1.0


def test_union_coverage_beats_individual_when_configs_are_partial():
    """★ 합집합 덮개가 필요한 이유 (§30.5b).

    `split_k=3` 은 K 가 3의 배수인 형상에서만 유효하다. 개별 전덮개를
    요구하면 그런 config 가 통째로 배제된다.
    """
    t = make_table({(1024, 4096, 4096): [1.0, 2.0],
                    (2048, 4096, 4096): [2.0, 1.0]})
    a = StaticTopK(t, coverage="union").run(ks=(2,))
    b = StaticTopK(t, coverage="individual").run(ks=(2,))
    assert a.by_k[2]["all"] <= b.by_k[2]["all"] + 1e-12


def test_procedures_are_three_and_canonical_is_last():
    names = [p[0] for p in PROCEDURES]
    assert names == ["ok_individual", "ok_union", "canonical"]
    assert "정본" in PROCEDURES[-1][2]


def test_coverage_is_always_reported():
    """덮개율을 병기하지 않으면 완화 변형이 23% 로 도망간 것을 못 본다."""
    t = make_table({(1024, 4096, 4096): [1.0, 2.0]})
    r = StaticTopK(t).run(ks=(1,))
    assert 0.0 <= r.coverage[1] <= 1.0


@pytest.mark.needs_bundle
def test_canonical_reproduces_documented_values(real_bundle_path):
    """★ 정본 절차가 문서 값(§30.5)을 재현한다. 회귀로 고정한다."""
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                   ok_only=False)
        sh = [p for p in tb.shapes()
              if bool((tb.frame_for(p).align_a == 8).all()
                      and (tb.frame_for(p).align_b == 8).all()
                      and (tb.frame_for(p).align_c == 8).all())]
        r = StaticTopK(tb, sh, coverage="union").run(ks=(1, 3, 8))
    assert len(sh) == 61
    assert r.by_k[1]["all"] == pytest.approx(1.115, abs=0.005)
    assert r.by_k[1]["large(>=0.5ms)"] == pytest.approx(1.021, abs=0.005)
    assert r.by_k[1]["small(<0.5ms)"] == pytest.approx(1.164, abs=0.005)
    assert r.by_k[3]["all"] == pytest.approx(1.031, abs=0.005)
    assert r.by_k[8]["all"] == pytest.approx(1.006, abs=0.005)
    assert r.coverage[1] == 1.0


@pytest.mark.needs_bundle
def test_ok_only_individual_reproduces_the_artifact(real_bundle_path):
    """★ 옛 값 1.394 가 **덮개 인공물**임을 고정한다 (§30.5b).

    후보가 3개로 줄어드는 것이 원인이다. 그 사실이 회귀로 남아야 나중에
    같은 숫자를 다시 정본으로 착각하지 않는다.
    """
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                   ok_only=True)
        sh = [p for p in tb.shapes()
              if bool((tb.frame_for(p).align_a == 8).all()
                      and (tb.frame_for(p).align_b == 8).all()
                      and (tb.frame_for(p).align_c == 8).all())]
        r = StaticTopK(tb, sh, coverage="individual").run(ks=(1, 3, 8))
    assert r.n_configs_considered == 3, \
        f"후보가 {r.n_configs_considered}개다 — 1.394 의 원인이 사라졌다"
    assert r.by_k[1]["all"] == pytest.approx(1.394, abs=0.005)
    # ★ k>=3 이 포화한다. 문서의 1.060 / 1.009 는 이 절차의 값이 아니다.
    assert r.by_k[3]["all"] == pytest.approx(r.by_k[8]["all"], abs=1e-9)


# ---------------------------------------------------------------------------
# 블록 분할 (§10.1)
# ---------------------------------------------------------------------------
@pytest.mark.needs_bundle
def test_block_splits_match_documented_sizes(real_bundle_path):
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                   ok_only=False)
    sh = tb.shapes()
    assert len(split_by_M_range(sh).val) == 11        # GBDT 주 지표의 홀드아웃
    assert len(split_by_waves(sh, tb.hw).val) == 15   # §2 의 waves<1 형상
    assert len(split_by_alignment(sh).val) == 5       # 층 D


@pytest.mark.needs_bundle
def test_size_split_does_not_use_answers(real_bundle_path):
    """★ 크기 분할 경계를 `best_ms`(정답)가 아니라 roofline 으로 잡는다.

    그런데도 실측 짧은 형상 45개를 **전부** 포함해야 한다.
    """
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                   ok_only=False)
    sp = split_by_size(tb.shapes(), tb.hw)
    held = {p.key for p in sp.val} | ({p.key for p in sp.test}
                                      if sp.test else set())
    real_small = {s.key for s in tb.all_stats() if s.is_small}
    assert real_small <= held, \
        f"실측 짧은 형상 {len(real_small - held)}개가 홀드아웃에서 빠졌다"


def test_gbdt_module_imports_without_lightgbm():
    """lightgbm 이 없어도 모듈은 import 된다 (별도 venv 에서 돌린다)."""
    import kernelrule.baselines.gbdt as g
    assert callable(g.build_xy) and "objective" in g.GBDT_PARAMS
