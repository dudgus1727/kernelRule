"""노이즈 바닥 (§30). **고정 1% 로 되돌아가면 여기서 걸린다.**"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from kernelrule.core.noise import NoiseMismatchError, NoiseModel


def test_15us_kernel_cannot_resolve_one_percent(noise_a6000):
    """15µs 커널에서 노이즈 바닥이 1% 를 넘는다 (실측 6.8%).

    고정 1% 허용치로 되돌아가면 이 테스트가 걸린다 (§30.3).
    """
    assert noise_a6000.floor(0.015) > 0.01
    assert noise_a6000.floor(0.014) > 0.07     # 눈금 하나가 7.3%


def test_tick_dominates_below_1_5ms(noise_a6000):
    """1.5ms 가 경계다. 그보다 짧으면 **분해능이 통계를 넘는다** (§30.2)."""
    m = noise_a6000
    assert m.tick_pct(0.014) > m.sigma(0.014)
    assert m.tick_pct(0.5) > m.sigma(0.5)
    assert m.tick_pct(4.0) < m.sigma(4.0)
    # 경계가 대략 1.5ms 근처인지
    lo, hi = 1.0, 2.5
    assert m.tick_pct(lo) > m.sigma(lo) * 0.9
    assert m.tick_pct(hi) < m.sigma(hi)


def test_floor_is_max_not_sum(noise_a6000):
    """`max` 이지 합이 아니다. 두 항의 성격이 다르기 때문이다 (§30.2)."""
    m = noise_a6000
    for t in (0.011, 0.1, 1.0, 5.0):
        assert m.floor(t) == pytest.approx(max(m.sigma(t), m.tick_pct(t)))


def test_floor_scales_35x_across_shape_sizes(noise_a6000):
    """형상 크기에 따라 바닥이 크게 달라진다. 단일 임계값이 왜 틀리는지."""
    small = noise_a6000.floor(0.0113)     # 이 표의 최소 최적시간
    large = noise_a6000.floor(9.74)       # 최대
    assert small / large > 30


def test_floor_fails_conservatively(noise_a6000):
    """★ 계산할 수 없으면 **보수적으로 큰 값**이다. 0 이 아니다 (§26.4).

    0 을 내면 "모든 차이가 유의하다" 가 되어 노이즈를 전부 신호로 배운다.
    """
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        assert noise_a6000.floor(bad) == 1.0
    arr = noise_a6000.floor(np.array([0.0, np.nan, 1.0]))
    assert arr[0] == 1.0 and arr[1] == 1.0 and arr[2] < 0.01


def test_answer_tol_is_two_sigma(noise_a6000):
    """정답 허용치는 2σ 다. 과대평가가 과소평가보다 안전하다 (§30.3)."""
    assert noise_a6000.answer_tol(0.5) == pytest.approx(2.0 * noise_a6000.floor(0.5))


def test_from_bundle_records_tick_fallback(real_bundle_path):
    """schema_version 1 번들은 `tick_ms` 가 없다. **그 사실이 기록된다** (§30.3b)."""
    from kerneltab.core.bundle import load_bundle

    b = load_bundle(real_bundle_path)
    assert b.schema_version == 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = NoiseModel.from_bundle(b)
    assert m.tick_is_fallback is True
    assert any("tick_ms" in str(x.message) for x in w), \
        "tick_ms 대체를 조용히 넘어갔다 — 다른 GPU 번들에서 틀린 눈금을 쓴다"
    assert "tick_ms=대체값" in m.source


def test_from_bundle_records_coefficient_mismatch(real_bundle_path, monkeypatch):
    """계수 불일치는 **기록**한다. 막지는 않는다.

    ## ★ 정정 이력 (2026-08-31)

    이 시험은 원래 **에러**를 요구했다. 근거는 "`answer_set()` 이 모듈
    전역을 쓰므로 불일치 상태에서 채점하면 정답 집합이 틀린 계수로
    만들어진다" 였다.

    **kernelTab 이 그 구멍을 막았다** — `answer_set`/`answer_tolerance`
    는 이제 계수 주입을 요구하고 없으면 `NoiseCoefRequired` 를 던진다.
    그러면 값 대조는 **다른 GPU 를 영구히 막는 통행 금지**가 된다
    (5090 눈금은 A6000 의 1/64 이라 정의상 다르다).

    위험을 지키는 것은 `_require_injected_noise` 로 옮겼다 — 아래
    `test_guard_catches_a_revived_default_tolerance`.
    """
    from kerneltab.core import noise as kt
    from kerneltab.core.bundle import load_bundle

    monkeypatch.setattr(kt, "SIGMA_ABS_MS", 0.999)
    b = load_bundle(real_bundle_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = NoiseModel.from_bundle(b)
    assert m.sigma_abs_ms != 0.999, "번들 계수를 써야 한다"
    assert any("A6000 참조값과 다르다" in str(x.message) for x in caught), (
        "조용히 지나가면 안 된다 — 사실로 기록해야 한다")
    assert pytest is not None


def test_from_bundle_requires_coefficients():
    """계수가 없으면 기본값으로 때우지 않고 에러다 (§26.4)."""
    class FakeBundle:
        info = {"bundle_id": "X"}
        tick_ms = 0.001
    with pytest.raises(NoiseMismatchError):
        NoiseModel.from_bundle(FakeBundle())


# ---------------------------------------------------------------------------
# 능력 검사 — 값 대조가 아니라 "위험한 경로가 막혀 있는가" (2026-08-31)
# ---------------------------------------------------------------------------
def test_guard_catches_a_revived_default_tolerance(monkeypatch):
    """★ `answer_tolerance` 가 주입 없이 값을 돌려주면 **막는다**.

    옛 검사(번들 계수 == 모듈 전역)는 이것을 못 잡았다 — 개발용 A6000
    번들에서는 값이 같아 통과했다. 그리고 정작 위험이 없는 다른 GPU 는
    영구히 막았다. 방향이 둘 다 틀렸다.
    """
    import pytest

    from kernelrule.core.noise import NoiseMismatchError, _require_injected_noise

    _require_injected_noise()           # 지금은 통과해야 한다
    import kerneltab.core.table as ktt
    monkeypatch.setattr(ktt, "answer_tolerance", lambda *a, **k: 0.01)
    with pytest.raises(NoiseMismatchError, match="기본값이 되살아났다"):
        _require_injected_noise()


def test_foreign_gpu_bundle_is_not_blocked_by_coefficient_drift():
    """다른 GPU 번들은 **계수가 다른 것이 정상**이다. 막지 않고 기록한다.

    5090 은 A6000 대비 눈금이 1/64 이다. 값이 같기를 요구하면 A6000 이
    아닌 모든 번들이 영구히 막힌다 — 안전장치가 아니라 통행 금지다.
    """
    import warnings
    from pathlib import Path

    import pytest

    b = Path("datasets/rtx-5090-sm_120-5bb6f403")
    if not b.exists():
        pytest.skip("5090 번들이 없다")
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = PerfTable.from_bundle(str(b), env_hash="5bb6f403")
    assert t.noise.tick_ms < 1e-4 and not t.noise.tick_is_fallback
    msgs = [str(x.message) for x in caught]
    assert any("A6000 참조값과 다르다" in m for m in msgs), (
        "불일치를 **사실로 기록**해야 한다 — 조용히 지나가면 안 된다")
