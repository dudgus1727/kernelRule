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


# ---------------------------------------------------------------------------
# ★ D-166 — the vendor path had no test at all
# ---------------------------------------------------------------------------
def test_gpu_presets_are_one_table():
    """★ The preset table was copied in two files and D-158 fixed **one** of
    them. A re-extraction through the unfixed one would have put an
    `H100_SXM` preset on an H100 NVL bundle — no error, no warning, a
    recommendation from a different card.

    The check is identity, not equality: two equal copies drift apart again.
    """
    import experiments.vendor_extract as ve
    from kernelrule.baselines import vendor as v

    assert ve.GPU_PRESETS is v.GPU_PRESETS
    assert ve.preset_for is v.preset_for
    assert ve.PAT is v.KERNEL_PAT
    assert ve.CLUSTER_PAT is v.CLUSTER_PAT
    # and the job has one implementation
    assert not hasattr(v, "extract"), (
        "`vendor.extract` is back. The recorded extractions all went through "
        "`experiments/vendor_extract.py` (D-166)")


@pytest.mark.parametrize("name,want", [
    ("NVIDIA H100 NVL", "H100_NVL"),
    ("NVIDIA H100 PCIe", "H100_PCIE"),
    ("NVIDIA H100 80GB HBM3", "H100_SXM"),
    ("NVIDIA RTX A6000", "RTX_A6000"),
])
def test_h100_variants_map_to_their_own_preset(name, want):
    """★ D-158's regression guard. Our bundle is an **H100 NVL** and the bare
    `h100` key used to swallow it."""
    from kernelrule.baselines.vendor import preset_for

    assert preset_for(name) == want


@pytest.mark.needs_bundle
def test_committed_vendor_json_reproduces():
    """★ D-166 — the committed extraction can be reproduced.

    Three shapes only: the point is that the procedure written down
    (0.1.0.27 · CUTLASS · TN_ROW_MAJOR · HSS · count 8 · the preset from the
    bundle) still produces what is in the file, not to re-extract it.
    ⚠️ The file itself is **not** rewritten (D-166).
    """
    import json
    from pathlib import Path

    nv = pytest.importorskip("nvMatmulHeuristics",
                             reason="nvMatmulHeuristics is not installed")
    b = Path("datasets/rtx-a6000-sm_86-c63710df")
    f = Path("datasets/baselines/vendor-a6000-c63710df.json")
    if not b.exists() or not f.exists():
        pytest.skip("the a6000 bundle or its vendor json is not here")

    from kernelrule.baselines.vendor import preset_for

    committed = json.loads(f.read_text())
    info = json.loads((b / "BUNDLE.json").read_text())
    preset = preset_for(info["gpu_name"])
    shapes = sorted({tuple(int(x) for x in s)
                     for rows in info["shape_layers"].values() for s in rows})
    h = nv.NvMatmulHeuristicsInterface(nv.NvMatmulHeuristicsTarget.CUTLASS,
                                       precision="HSS")
    hd = h.createHardwareDescriptor()
    h.setHardwarePredefinedGpu(hd, getattr(nv.NvMatmulHeuristicsNvidiaGpu,
                                           preset))
    layout = nv.NvMatmulHeuristicsMatmulLayout.TN_ROW_MAJOR
    for (M, N, K) in shapes[:3]:
        got = h.get_with_mnk(M, N, K, layout, 8, hd)
        want = committed[f"{M}x{N}x{K}"]
        assert len(got) == len(want), f"{M}x{N}x{K}: candidate count"
        for c, w in zip(got, want, strict=True):
            k = c["kernel"]
            assert [k.cta_tile_m, k.cta_tile_n, k.cta_tile_k] == w["cta"]
            assert k.stages == w["stages"] and k.split_k == w["split_k"]
            assert abs((c.get("runtime") or 0) * 1000.0
                       - w["pred_ms"]) < 1e-9
