"""The synthetic table generator (§22, §28)."""
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
# §28 — blocking a tautology
# ---------------------------------------------------------------------------
def test_generator_does_not_import_features():
    """★ If the generator uses `features/`, verifying the pipeline becomes a
    tautology (§28)."""
    tree = ast.parse(Path(synth.__file__).read_text())
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        for mod in mods:
            assert "kernelrule.features" not in mod and mod != "features", \
                (f"the generator imports {mod} — the physics may be the "
                 f"same but the implementation has to be separate")


def test_generator_never_sees_measured_times(real_bundle_path):
    """★ Even when the grid comes from a real bundle, only the `ranking`
    loader is used."""
    src = inspect.getsource(synth.Grid.from_bundle)
    assert ".ranking(" in src
    assert ".scoring(" not in src and "load_for_scoring" not in src, \
        ("the generator touched the answer loader — the synthetic table "
         "could then copy the measurement")


# ---------------------------------------------------------------------------
# The physics — did the planted structure really go in
# ---------------------------------------------------------------------------
def test_timer_quantization_is_applied(tiny_grid):
    """★ Without the quantisation, a ranking appears in the synthetic table
    that is not really there (§22.3)."""
    t = synth_times(tiny_grid, "normal", seed=0)
    q = t / synth._TICK_MS
    assert np.allclose(q, np.round(q)), "the times are not integer multiples of the tick"
    # Do ties really appear in bulk on the short shapes
    short = t < 0.05
    if short.sum() > 50:
        vals, cnt = np.unique(t[short], return_counts=True)
        assert cnt.max() > 1, "no tie at all on a short shape — the tick did not take"


def test_noise_depends_on_kernel_time(tiny_grid):
    """★ With fixed noise the difficulty of a small shape disappears
    (§22.3, §30.2)."""
    a = synth_times(tiny_grid, "normal", seed=1)
    b = synth_times(tiny_grid, "normal", seed=2)
    rel = np.abs(a - b) / np.maximum(a, 1e-9)
    small = a < np.quantile(a, 0.2)
    large = a > np.quantile(a, 0.8)
    assert rel[small].mean() > rel[large].mean() * 3, \
        ("a short kernel's relative variation is similar to a long one's — "
         "the noise is not time dependent")


def test_null_preset_has_no_config_structure(tiny_grid):
    """★ With `struct=0` the penalty is exactly 1 — the time is independent of
    the config."""
    _, parts = synth_times(tiny_grid, "null", seed=0, return_parts=True)
    assert np.allclose(parts["penalty"], 1.0)


def test_presets_are_ordered_by_difficulty(tiny_grid):
    """The structure has to get stronger in the order easy > normal > hard
    (the difficulty rises)."""
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
    """★ Does the shape x config interaction really exist (structure #11).

    On an M=1 shape a 128-row tile throws away 99% of the work. Without this
    interaction one fixed config gets close to optimal on every shape and
    there is nothing to learn.
    """
    df = tiny_grid.df
    t = synth_times(tiny_grid, "normal", seed=0)
    m1 = (df.M == 1).to_numpy()
    if m1.sum() < 50:
        pytest.skip("there is no M=1 shape in the grid")
    tm = df.tile_m.to_numpy()[m1]
    tt = t[m1]
    small_tile = tt[tm <= 64].min()
    big_tile = tt[tm >= 256].min()
    assert small_tile < big_tile, \
        "a large tile is faster at M=1 — the partial-tile waste did not go in"


def test_unknown_preset_is_an_error(tiny_grid):
    with pytest.raises(ValueError, match="unknown preset"):
        synth_times(tiny_grid, "medium")


# ---------------------------------------------------------------------------
# The bundle format and the self-check
# ---------------------------------------------------------------------------
def test_bundle_is_watermarked(synth_bundles):
    """★ It blocks, at the level of names, the path by which a synthetic
    artefact is actually mistaken for a real one (§22.6)."""
    for preset, path in synth_bundles.items():
        assert "SYNTHETIC" in Path(path).name
        info = json.loads((Path(path) / "BUNDLE.json").read_text())
        assert "SYNTHETIC" in info["bundle_id"]
        assert info["synthetic"]["preset"] == preset
        assert "Do not report" in info["synthetic"]["warning"]


def test_synthetic_env_hash_cannot_collide_with_real(synth_bundles):
    """A synthetic `env_hash` does not get mixed with a real condition
    (§3.4)."""
    for path in synth_bundles.values():
        info = json.loads((Path(path) / "BUNDLE.json").read_text())
        assert info["env_hash"].startswith("5y47he71c")


def test_bundle_loads_through_the_real_loaders(synth_bundles):
    """A synthetic table has to have the same format as a real one for the
    loader/adapter to be verified (§22.3)."""
    import warnings

    from kerneltab.core.bundle import load_bundle
    from kerneltab.core.table import assert_no_answers

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = load_bundle(synth_bundles["normal"], verify=True)
        assert b.schema_version == 2          # it really carries tick_ms
        assert b.noise_floor(0.014) > 0.07
        X = b.ranking(ok_only=False, unknown_columns="ignore")
        assert_no_answers(X)
        y = b.scoring(ok_only=False)
    assert len(X) == len(y)


def test_self_check_targets(synth_bundles):
    """★ If the generator drifts off its target statistics, everything built
    on top of it drifts away from reality."""
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
