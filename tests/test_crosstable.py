"""The common subset of two tables (different GPUs) (D-88).

"It transferred" must not be **a result of sample selection**. Count what was
discarded.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from kernelrule.core.crosstable import (
    AXIS_FIELDS,
    axis_key,
    bound_flipped,
    common_axis_keys,
    common_shapes,
    cross_report,
)
from kernelrule.core.types import Problem


class _FakeTable:
    """A stand-in that has only `hw` and `shapes()`. Bound flips are decided
    by those two."""

    def __init__(self, hw, shapes):
        self.hw = hw
        self._s = list(shapes)

    def shapes(self):
        return list(self._s)


def test_axis_key_excludes_build_outputs():
    """★ `kernel_id` / registers / smem are **build results**, so they
    differ when the GPU changes.

    Putting them in the join key empties the intersection entirely, and that
    reads as "it does not transfer".
    """
    for bad in ("kernel_id", "regs_per_thread", "smem_bytes", "spill_bytes",
                "arch", "inst_total", "max_blocks_per_sm"):
        assert bad not in AXIS_FIELDS, (
            f"{bad} is not architecture-independent")
    row = dict.fromkeys(AXIS_FIELDS, 1) | {"kernel_id": "x", "arch": "sm_86"}
    other = row | {"kernel_id": "y", "arch": "sm_120"}
    assert axis_key(row) == axis_key(other), "the kernel id leaked into the key"


def test_common_shapes_keeps_a_order(synth_table):
    got = common_shapes(synth_table, synth_table)
    assert got == synth_table.shapes()


def test_common_shapes_is_an_intersection(synth_table, null_table):
    a, b = synth_table, null_table
    ka = {(p.M, p.N, p.K, p.dtype) for p in a.shapes()}
    kb = {(p.M, p.N, p.K, p.dtype) for p in b.shapes()}
    got = {(p.M, p.N, p.K, p.dtype) for p in common_shapes(a, b)}
    assert got == ka & kb


def test_common_axis_keys(synth_table):
    p = synth_table.shapes()[0]
    got = common_axis_keys(synth_table, synth_table, p)
    all_keys = {axis_key(r)
                for r in synth_table.frame_for(p).to_dict("records")}
    assert got == all_keys


def test_bound_flip_is_detected_when_ridge_moves(synth_table):
    """★ When the ridge moves, the regime of the same shape flips.

    Weights are fitted per regime (§10), so if that verdict differs per table
    **the two tables measure different things.** Letting it pass silently
    reads as "transfer failed".
    """
    hw = synth_table.hw
    shapes = [Problem(1024, 4096, 4096), Problem(128, 4096, 4096)]
    a = _FakeTable(hw, shapes)
    # Raising the bandwidth a lot lowers the ridge, so more shapes become
    # compute-bound
    b = _FakeTable(replace(hw, bandwidth_gbps=hw.bandwidth_gbps * 8), shapes)
    flipped = bound_flipped(a, b, shapes)
    assert flipped, "the ridge moved 8x yet nothing flipped"
    for _p, ma, mb in flipped:
        assert ma != mb


def test_no_flip_when_hardware_is_identical(synth_table):
    assert bound_flipped(synth_table, synth_table) == []


def test_report_counts_what_was_dropped(synth_table):
    r = cross_report(synth_table, synth_table)
    assert r.n_shapes_common == r.n_shapes_a == r.n_shapes_b
    assert r.n_bound_flipped == 0
    body = r.render()
    assert "dropped from A" in body and "bound flips" in body


def test_report_is_not_symmetric_in_labels(synth_table, null_table):
    """Swapping A/B changes the "dropped" ratio — not writing the direction
    down invites a misreading."""
    ab = cross_report(synth_table, null_table)
    ba = cross_report(null_table, synth_table)
    assert (ab.n_shapes_a, ab.n_shapes_b) == (ba.n_shapes_b, ba.n_shapes_a)


@pytest.mark.parametrize("field", AXIS_FIELDS)
def test_axis_key_uses_every_declared_field(field, synth_table):
    """Is every declared axis actually used — if one is missing, different
    configs get merged."""
    row = dict.fromkeys(AXIS_FIELDS, 1)
    other = row | {field: 99}
    assert axis_key(row) != axis_key(other), f"{field} did not enter the key"


# ---------------------------------------------------------------------------
# ★ The definition is not written a third time (2026-08-31)
# ---------------------------------------------------------------------------
def test_arith_intensity_matches_the_table_column():
    """★ Our arithmetic intensity **must equal the table's
    `arith_intensity` column.**

    `crosstable` once had a definition of its own that multiplied the output
    term by `acc_bytes_per_element` (f32). That was wrong, because the
    accumulator lives in registers and the C that goes out to DRAM is f16.

    ```
    128x4096x4096   old definition 117.03   table 120.471   5090 ridge 117.855
    ```

    **The boundary sits between them, so it counted 4 flips as 3.** It is not
    that the value differs a little — **the classification flipped.** It
    differed on all 53 common shapes.
    """
    from pathlib import Path

    import pytest

    b = Path("datasets/rtx-5090-sm_120-5bb6f403")
    if not b.exists():
        pytest.skip("no 5090 bundle")
    import warnings

    from kernelrule.core.crosstable import _arith_intensity
    from kernelrule.core.numerics import approx_equal
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = PerfTable.from_bundle(str(b), env_hash="5bb6f403")
    bad = []
    for p in t.shapes():
        col = float(t.frame_for(p)["arith_intensity"].iloc[0])
        if not approx_equal(_arith_intensity(p), col, tol=1e-3):
            bad.append((p.M, p.N, p.K, _arith_intensity(p), col))
    assert not bad, (
        f"{len(bad)} shapes differ from the table column: {bad[:3]}")


def test_ridge_comes_from_hardware_not_recomputed():
    """`_ridge` must equal the bundle's `ridge_point`.

    Whether effective or spec values are used differs by 26% (§6.2). If that
    judgement lives in two places, the regime of boundary shapes changes
    silently.
    """
    import json
    import warnings
    from pathlib import Path

    import pytest

    from kernelrule.core.crosstable import _ridge
    from kernelrule.core.numerics import approx_equal
    from kernelrule.core.table import PerfTable

    for d, h in (("datasets/rtx-a6000-sm_86-c63710df", "c63710df"),
                 ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")):
        if not Path(d).exists():
            pytest.skip(f"{d} is missing")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = PerfTable.from_bundle(d, env_hash=h)
        want = json.loads(Path(d, "BUNDLE.json").read_text())["ridge_point"]
        assert approx_equal(_ridge(t.hw), float(want), tol=1e-2), d
