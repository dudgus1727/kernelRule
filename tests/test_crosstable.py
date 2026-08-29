"""두 표(다른 GPU)의 공통 부분집합 (D-88).

"전이가 됐다" 가 **표본 선택의 결과**이면 안 된다. 버린 것을 센다.
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
    """`hw` 와 `shapes()` 만 있는 대역. 바운드 뒤집힘은 그 둘로 정해진다."""

    def __init__(self, hw, shapes):
        self.hw = hw
        self._s = list(shapes)

    def shapes(self):
        return list(self._s)


def test_axis_key_excludes_build_outputs():
    """★ `kernel_id` / 레지스터 / smem 은 **빌드 결과**라 GPU 가 바뀌면 다르다.

    조인 키에 넣으면 교집합이 통째로 비고, 그것이 "전이가 안 된다" 로
    읽힌다.
    """
    for bad in ("kernel_id", "regs_per_thread", "smem_bytes", "spill_bytes",
                "arch", "inst_total", "max_blocks_per_sm"):
        assert bad not in AXIS_FIELDS, f"{bad} 는 아키텍처 독립이 아니다"
    row = dict.fromkeys(AXIS_FIELDS, 1) | {"kernel_id": "x", "arch": "sm_86"}
    other = row | {"kernel_id": "y", "arch": "sm_120"}
    assert axis_key(row) == axis_key(other), "커널 id 가 키에 섞였다"


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
    """★ ridge 가 움직이면 같은 형상의 체제가 뒤집힌다.

    체제별로 가중치를 따로 적합하는데(§10) 그 판정이 표마다 다르면
    **두 표에서 다른 것을 잰다.** 조용히 넘기면 "전이 실패" 로 읽힌다.
    """
    hw = synth_table.hw
    shapes = [Problem(1024, 4096, 4096), Problem(128, 4096, 4096)]
    a = _FakeTable(hw, shapes)
    # 대역폭을 크게 올리면 ridge 가 내려가 컴퓨트 바운드가 늘어난다
    b = _FakeTable(replace(hw, bandwidth_gbps=hw.bandwidth_gbps * 8), shapes)
    flipped = bound_flipped(a, b, shapes)
    assert flipped, "ridge 를 8배 움직였는데 아무것도 안 뒤집혔다"
    for _p, ma, mb in flipped:
        assert ma != mb


def test_no_flip_when_hardware_is_identical(synth_table):
    assert bound_flipped(synth_table, synth_table) == []


def test_report_counts_what_was_dropped(synth_table):
    r = cross_report(synth_table, synth_table)
    assert r.n_shapes_common == r.n_shapes_a == r.n_shapes_b
    assert r.n_bound_flipped == 0
    body = r.render()
    assert "버림" in body and "바운드 뒤집힘" in body


def test_report_is_not_symmetric_in_labels(synth_table, null_table):
    """A/B 를 바꾸면 '버림' 비율이 달라진다 — 방향을 안 적으면 오해가 난다."""
    ab = cross_report(synth_table, null_table)
    ba = cross_report(null_table, synth_table)
    assert (ab.n_shapes_a, ab.n_shapes_b) == (ba.n_shapes_b, ba.n_shapes_a)


@pytest.mark.parametrize("field", AXIS_FIELDS)
def test_axis_key_uses_every_declared_field(field, synth_table):
    """선언한 축을 실제로 쓰는가 — 하나라도 빠지면 서로 다른 config 가 합쳐진다."""
    row = dict.fromkeys(AXIS_FIELDS, 1)
    other = row | {field: 99}
    assert axis_key(row) != axis_key(other), f"{field} 가 키에 안 들어갔다"
