"""합성 표 생성기 (§22, §28)."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from kernelrule.tools import synth
from kernelrule.tools.synth import self_check, synth_times


# ---------------------------------------------------------------------------
# §28 — 동어반복 방지
# ---------------------------------------------------------------------------
def test_generator_does_not_import_features():
    """★ 생성기가 `features/` 를 쓰면 파이프라인 검증이 동어반복이 된다 (§28)."""
    tree = ast.parse(Path(synth.__file__).read_text())
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        for mod in mods:
            assert "kernelrule.features" not in mod and mod != "features", \
                f"생성기가 {mod} 를 import 한다 — 물리는 같되 구현을 분리해야 한다"


def test_generator_never_sees_measured_times(real_bundle_path):
    """★ 격자를 실제 번들에서 가져올 때도 `ranking` 로더만 쓴다."""
    src = inspect.getsource(synth.Grid.from_bundle)
    assert ".ranking(" in src
    assert ".scoring(" not in src and "load_for_scoring" not in src, \
        "생성기가 정답 로더에 손을 댔다 — 합성 표가 실측을 베낄 수 있게 된다"


# ---------------------------------------------------------------------------
# 물리 — 심은 구조가 실제로 들어갔는가
# ---------------------------------------------------------------------------
def test_timer_quantization_is_applied(tiny_grid):
    """★ 양자화를 빼면 합성 표에 실제로는 없는 순위가 생긴다 (§22.3)."""
    t = synth_times(tiny_grid, "normal", seed=0)
    q = t / synth._TICK_MS
    assert np.allclose(q, np.round(q)), "시간이 눈금의 정수배가 아니다"
    # 짧은 형상에서 동점이 실제로 대량 발생하는가
    short = t < 0.05
    if short.sum() > 50:
        vals, cnt = np.unique(t[short], return_counts=True)
        assert cnt.max() > 1, "짧은 형상에 동점이 하나도 없다 — 눈금이 안 먹었다"


def test_noise_depends_on_kernel_time(tiny_grid):
    """★ 고정 노이즈면 작은 형상의 어려움이 사라진다 (§22.3, §30.2)."""
    a = synth_times(tiny_grid, "normal", seed=1)
    b = synth_times(tiny_grid, "normal", seed=2)
    rel = np.abs(a - b) / np.maximum(a, 1e-9)
    small = a < np.quantile(a, 0.2)
    large = a > np.quantile(a, 0.8)
    assert rel[small].mean() > rel[large].mean() * 3, \
        "짧은 커널의 상대 변동이 긴 커널과 비슷하다 — 노이즈가 시간 의존이 아니다"


def test_null_preset_has_no_config_structure(tiny_grid):
    """★ `struct=0` 이면 penalty 가 정확히 1 — 시간이 config 와 무관하다."""
    _, parts = synth_times(tiny_grid, "null", seed=0, return_parts=True)
    assert np.allclose(parts["penalty"], 1.0)


def test_presets_are_ordered_by_difficulty(tiny_grid):
    """easy > normal > hard 순으로 구조가 강해야 한다 (난이도가 높다)."""
    import pandas as pd
    key = tiny_grid.df[["M", "N", "K"]].apply(tuple, axis=1).to_numpy()
    d = {}
    for name in ("easy", "normal", "hard", "null"):
        t = synth_times(tiny_grid, name, seed=0)
        g = pd.DataFrame({"k": key, "t": t}).groupby("k").t
        d[name] = float((g.median() / g.min()).median())
    assert d["easy"] > d["normal"] > d["hard"] > d["null"]
    assert d["null"] < 1.15


def test_edge_penalty_makes_small_M_prefer_small_tiles(tiny_grid):
    """★ 형상 x config 상호작용이 실재하는가 (구조 #11).

    M=1 형상에서 128행 타일은 일의 99% 를 버린다. 이 상호작용이 없으면
    고정 config 하나가 모든 형상에서 최적에 가까워져 배울 것이 없어진다.
    """
    df = tiny_grid.df
    t = synth_times(tiny_grid, "normal", seed=0)
    m1 = (df.M == 1).to_numpy()
    if m1.sum() < 50:
        pytest.skip("격자에 M=1 형상이 없다")
    tm = df.tile_m.to_numpy()[m1]
    tt = t[m1]
    small_tile = tt[tm <= 64].min()
    big_tile = tt[tm >= 256].min()
    assert small_tile < big_tile, \
        "M=1 에서 큰 타일이 더 빠르다 — 부분 타일 낭비가 안 들어갔다"


def test_unknown_preset_is_an_error(tiny_grid):
    with pytest.raises(ValueError, match="알 수 없는 프리셋"):
        synth_times(tiny_grid, "medium")


# ---------------------------------------------------------------------------
# 번들 형식과 자기 검사
# ---------------------------------------------------------------------------
def test_bundle_is_watermarked(synth_bundles):
    """★ 합성 산출물이 실제로 오인되는 경로를 이름 수준에서 막는다 (§22.6)."""
    for preset, path in synth_bundles.items():
        assert "SYNTHETIC" in Path(path).name
        info = json.loads((Path(path) / "BUNDLE.json").read_text())
        assert "SYNTHETIC" in info["bundle_id"]
        assert info["synthetic"]["preset"] == preset
        assert "성능 수치를 보고하지 마라" in info["synthetic"]["warning"]


def test_synthetic_env_hash_cannot_collide_with_real(synth_bundles):
    """합성 `env_hash` 가 실제 조건과 섞이지 않는다 (§3.4)."""
    for path in synth_bundles.values():
        info = json.loads((Path(path) / "BUNDLE.json").read_text())
        assert info["env_hash"].startswith("5y47he71c")


def test_bundle_loads_through_the_real_loaders(synth_bundles):
    """합성 표도 진짜 표와 같은 형식이어야 로더/어댑터가 검증된다 (§22.3)."""
    import warnings

    from kerneltab.core.bundle import load_bundle
    from kerneltab.core.table import assert_no_answers

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = load_bundle(synth_bundles["normal"], verify=True)
        assert b.schema_version == 2          # tick_ms 를 실제로 싣는다
        assert b.noise_floor(0.014) > 0.07
        X = b.ranking(ok_only=False, unknown_columns="ignore")
        assert_no_answers(X)
        y = b.scoring(ok_only=False)
    assert len(X) == len(y)


def test_self_check_targets(synth_bundles):
    """★ 생성기가 목표 통계를 벗어나면 그 위의 모든 개발이 현실과 멀어진다."""
    r = self_check(synth_bundles["normal"])
    assert 1.35 <= r["difficulty_median"] <= 1.95, r
    r_null = self_check(synth_bundles["null"])
    assert r_null["difficulty_median"] < 1.05, r_null
    r_easy = self_check(synth_bundles["easy"])
    assert r_easy["difficulty_median"] > r["difficulty_median"]


def test_generation_is_deterministic(tiny_grid):
    a = synth_times(tiny_grid, "normal", seed=42)
    b = synth_times(tiny_grid, "normal", seed=42)
    assert np.array_equal(a, b)
    c = synth_times(tiny_grid, "normal", seed=43)
    assert not np.array_equal(a, c)


def test_bundle_id_must_be_watermarked(tiny_grid, tmp_path):
    from kernelrule.tools.synth import generate
    with pytest.raises(ValueError, match="SYNTHETIC"):
        generate("normal", 0, tmp_path, tiny_grid, bundle_id="looks-real")
