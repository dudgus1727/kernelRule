"""The baselines (§9, §30.5b). **The three procedures are reported side by
side.**"""
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
    """The static top-k finds the optimal set in an obvious case (§26.2).

    config 0 is good only on shape A, config 1 only on shape B. At k=2 the
    two together have to be perfect.
    """
    t = make_table({
        (1024, 4096, 4096): [1.0, 4.0, 8.0],
        (2048, 4096, 4096): [4.0, 1.0, 8.0],
    })
    r = StaticTopK(t, coverage="union").run(ks=(1, 2, 3))
    assert r.by_k[1]["all"] == pytest.approx(2.0)      # geomean(1, 4)
    assert r.by_k[2]["all"] == pytest.approx(1.0)      # the two together are perfect
    assert r.coverage[2] == 1.0


def test_union_coverage_beats_individual_when_configs_are_partial():
    """★ Why the union cover is needed (§30.5b).

    `split_k=3` is valid only on shapes whose K is a multiple of 3. Requiring
    a full individual cover excludes such a config entirely.
    """
    t = make_table({(1024, 4096, 4096): [1.0, 2.0],
                    (2048, 4096, 4096): [2.0, 1.0]})
    a = StaticTopK(t, coverage="union").run(ks=(2,))
    b = StaticTopK(t, coverage="individual").run(ks=(2,))
    assert a.by_k[2]["all"] <= b.by_k[2]["all"] + 1e-12


def test_procedures_are_three_and_canonical_is_last():
    names = [p[0] for p in PROCEDURES]
    assert names == ["ok_individual", "ok_union", "canonical"]
    assert "representative" in PROCEDURES[-1][2]


def test_coverage_is_always_reported():
    """Without reporting the cover rate alongside, a relaxed variant escaping
    to 23% goes unseen."""
    t = make_table({(1024, 4096, 4096): [1.0, 2.0]})
    r = StaticTopK(t).run(ks=(1,))
    assert 0.0 <= r.coverage[1] <= 1.0


@pytest.mark.needs_bundle
def test_canonical_reproduces_documented_values(real_bundle_path):
    """★ The representative procedure reproduces the documented value
    (§30.5). It is pinned as a regression."""
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
    """★ It pins that the old value 1.394 is **a cover artefact** (§30.5b).

    The cause is the candidates shrinking to 3. That fact has to stay as a
    regression so the same number is not mistaken for a representative value
    later.
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
        (f"there are {r.n_configs_considered} candidates — the cause of "
         f"1.394 is gone")
    assert r.by_k[1]["all"] == pytest.approx(1.394, abs=0.005)
    # ★ k>=3 saturates. The documented 1.060 / 1.009 are not values of this
    #   procedure.
    assert r.by_k[3]["all"] == pytest.approx(r.by_k[8]["all"], abs=1e-9)


# ---------------------------------------------------------------------------
# Block splits (§10.1)
# ---------------------------------------------------------------------------
@pytest.mark.needs_bundle
def test_block_splits_match_documented_sizes(real_bundle_path):
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                   ok_only=False)
    sh = tb.shapes()
    assert len(split_by_M_range(sh).val) == 11        # the GBDT main metric's holdout
    assert len(split_by_waves(sh, tb.hw).val) == 15   # the waves<1 shapes of §2
    assert len(split_by_alignment(sh).val) == 5       # layer D


@pytest.mark.needs_bundle
def test_size_split_does_not_use_answers(real_bundle_path):
    """★ The size-split boundary is taken from the roofline, not from
    `best_ms` (the answer).

    Even so it has to include **all** 45 of the really short shapes.
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
        f"{len(real_small - held)} really short shapes fell out of the holdout"


def test_gbdt_module_imports_without_lightgbm():
    """The module imports even without lightgbm (it runs in a separate
    venv)."""
    import kernelrule.baselines.gbdt as g
    assert callable(g.build_xy) and "objective" in g.GBDT_PARAMS
