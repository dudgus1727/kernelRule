"""The feature library and automatic validation (§8.2, §8.3)."""
from __future__ import annotations

import inspect
import warnings

import pytest

import kernelrule.features.physical  # noqa: F401  registration
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.features import REGISTRY
from kernelrule.features.validate import validate_registry


@pytest.fixture(scope="module")
def matrix(synth_table):
    return FeatureMatrix(synth_table, REGISTRY)


def test_library_has_enough_features():
    """§8.2 — 10~15 are written by hand before starting."""
    assert len(REGISTRY.names(shape_level=False)) >= 15
    assert len(REGISTRY.names(shape_level=True)) >= 4


def test_no_feature_touches_ext():
    """★ No reference to `cfg.ext` — the architecture-transfer premise
    (§4.3, §8.2).

    `ext_*` has no counterpart on SM90. A rule aiming at transfer that uses
    it collapses on the main metric (the architecture holdout).
    """
    bad = []
    for f in REGISTRY.items(active_only=False):
        try:
            src = inspect.getsource(f.fn)
        except (OSError, TypeError):      # pragma: no cover
            continue
        if "cfg.ext" in src or ".ext[" in src:
            bad.append(f.name)
    assert not bad, f"features that reference `cfg.ext`: {bad}"


def test_no_feature_references_answers():
    """No answer-column name is referenced **as an identifier**.

    A plain substring check will not do — `hw.peak_tflops_f16` contains
    `tflops` and gives a false positive. It looks at token boundaries.
    """
    import re

    from kerneltab.core.table import ANSWER_COLS

    bad = []
    for f in REGISTRY.items(active_only=False):
        try:
            src = inspect.getsource(f.fn)
        except (OSError, TypeError):      # pragma: no cover
            continue
        code = "\n".join(ln for ln in src.split("\n")
                          if not ln.strip().startswith("#"))
        for col in ANSWER_COLS:
            if re.search(rf"(?<![\w.]){re.escape(col)}\b", code):
                bad.append((f.name, col))
    assert not bad, f"features that reference an answer column: {bad}"


def test_all_features_are_short():
    """§8.2 — at most 10 lines. Longer means combination, not physics."""
    long = []
    for f in REGISTRY.items(active_only=False):
        try:
            src = inspect.getsource(f.fn)
        except (OSError, TypeError):      # pragma: no cover
            continue
        body = [ln for ln in src.split("\n")
                if ln.strip() and not ln.strip().startswith("#")]
        # The effective line count, excluding docstring and decorators
        n = len([ln for ln in body if not ln.lstrip().startswith(("@", '"""'))])
        if n > 22:
            long.append((f.name, n))
    assert not long, f"features that are too long: {long}"


#: ★ Axes of the human registry that the duplication check calls duplicates
#: of each other (D-162). **They are not deleted** — they are what every
#: number so far was measured with, and deleting them would silently change
#: the condition. What is pinned here is that the list does not grow.
#:
#: Measured on the whole A6000 table (66 shapes, 980,915 rows):
#:   has_spill ~ spill_magnitude          Spearman 1.000 / Pearson 0.976
#:   arith_intensity ~ reuse_ratio        1.000 / 1.000
#:   arith_intensity ~ roofline_ratio     1.000 / 1.000
#:   reuse_ratio ~ roofline_ratio         1.000 / 1.000
#: On the synthetic table only the first pair is above the line.
_KNOWN_DUPLICATE_AXES = {"has_spill", "spill_magnitude"}


def test_registry_validates_clean(synth_table, matrix, hw_other):
    """★ Every feature passes the automatic validation, **except the
    duplicate pairs that are written down** (D-162).

    ⚠️ Before D-162 the duplication check never ran — the candidate was
    measured on one set of shapes and the comparison set on another, and
    every mismatched pair was skipped. Now that it runs, it finds pairs
    inside the human registry too. They stay; the test pins that no *new*
    axis joins them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reps = validate_registry(REGISTRY, synth_table, matrix,
                                 hw_alt=hw_other, n_shapes=4)
    failed = {n: [str(c) for c in r.fails()]
              for n, r in reps.items() if r.failed}
    unexpected = {n: c for n, c in failed.items()
                  if not (n in _KNOWN_DUPLICATE_AXES
                          and all("duplication" in x for x in c))}
    assert not unexpected, f"validation rejections: {unexpected}"
    assert set(failed) <= _KNOWN_DUPLICATE_AXES


def test_scale_invariance_catches_a_hardcoded_constant(synth_table, matrix,
                                                       hw_other):
    """★ Is a feature with a hardcoded hardware constant **caught** (§8.3
    item 6)?

    It checks that the watchdog actually works. The mere existence of a check
    guarantees nothing (§30.8).
    """
    from kernelrule.features import Feature, FeatureRegistry
    from kernelrule.features.validate import validate_feature

    r = FeatureRegistry("bad")

    def fake_waves(p, hw, cfg) -> float:
        """It pretends to read hw.sm_count but has 84 nailed in."""
        import math
        return math.ceil(p.M / cfg.tile_m) * math.ceil(p.N / cfg.tile_n) / 84.0

    f = Feature(name="fake_waves", fn=fake_waves, unit="dimensionless",
                expected_range=(0.0, 1e6), direction="neutral",
                vec=None, code_hash="x")
    r.add(f)
    m2 = FeatureMatrix(synth_table, r)
    rep = validate_feature(f, synth_table, m2, hw_alt=hw_other, n_shapes=3)
    # The string `hw.` appears only in the docstring, so it looks like it
    # uses the hardware
    assert rep.failed, "the hardcoded 84 was not caught"
    assert any("scale" in c.name for c in rep.fails())


def test_vectorized_matches_scalar_everywhere(synth_table, matrix):
    """★ Do training (the matrix) and deployment (the scalar) use the same
    function?"""
    from kernelrule.features import verify_vectorized

    p = synth_table.shapes()[0]
    df = synth_table.frame_for(p)
    _, info = matrix.for_shape(p)
    for f in REGISTRY.items():
        if f.vec is None or f.shape_level:
            continue
        verify_vectorized(f, df, matrix.hw, info, n=96)


def test_directions_are_declared():
    for f in REGISTRY.items(active_only=False):
        assert f.direction in ("higher_is_worse", "higher_is_better",
                               "neutral"), f.name


def test_shape_features_ignore_config(synth_table, matrix):
    """A shape-level feature must not look at `cfg` — that is its
    definition."""
    p = synth_table.shapes()[0]
    cfgs = synth_table.configs(p)
    for f in REGISTRY.items(shape_level=True):
        vals = {float(f.fn(p, matrix.hw, c)) for c in cfgs[:40]}
        assert len(vals) == 1, (
            f"{f.name} produces different values per config: {vals}")


# ---------------------------------------------------------------------------
# The hardware-usage verdict for generated features (D-37)
# ---------------------------------------------------------------------------
# For a function built with `exec`, `inspect.getsource` raises OSError.
# Falling back to "it uses hw" there makes the scale-invariance check
# **reject every perfectly good hardware-independent feature.** In the first
# F1 run features really were thrown away that way.

_NO_HW = ("def tile_aspect(p, hw, cfg) -> float:\n"
          "    a = float(cfg.tile_m)\n"
          "    b = max(1.0, float(cfg.tile_n))\n"
          "    return abs(a - b) / (a + b)\n")
_USES_HW = ("def per_sm(p, hw, cfg) -> float:\n"
            "    return float(p.M) / max(1.0, float(hw.sm_count))\n")


def _gen(code):
    from kernelrule.features import Feature
    from kernelrule.features.generated import compile_feature
    name, fn = compile_feature(code, known=frozenset())
    return Feature(name=name, fn=fn, unit="dimensionless",
                   expected_range=(0.0, 1.0), direction="neutral",
                   source=code)


@pytest.mark.parametrize("code,expected", [(_NO_HW, False), (_USES_HW, True)])
def test_generated_feature_hardware_usage_is_read_from_source(code, expected):
    from kernelrule.features.validate import _uses_hardware
    assert _uses_hardware(_gen(code)) is expected


def test_unreadable_source_raises_instead_of_guessing():
    """★ If the source cannot be read, **no verdict is made** (§26.4).

    Falling back to `True` is the claim "it uses hw", and if that claim is
    wrong a perfectly good feature is rejected.
    """
    from kernelrule.features import Feature
    from kernelrule.features.generated import compile_feature
    from kernelrule.features.validate import _uses_hardware

    name, fn = compile_feature(_NO_HW, known=frozenset())
    f = Feature(name=name, fn=fn, unit="dimensionless",
                expected_range=(0.0, 1.0), direction="neutral")  # no source
    with pytest.raises(ValueError, match="the source cannot be read"):
        _uses_hardware(f)


# ---------------------------------------------------------------------------
# ★ §30.9 — the library never references the global registry directly
#
# F0~F3 exist by **swapping the registry**. If anywhere in the library does
# `from kernelrule.features import REGISTRY`, then even when the condition
# changes that one place keeps seeing the 24 a human wrote — silently, with
# no error. The same class of accident as `is_reference()` / `top_k` /
# `DEFAULT_MODEL` (principle 2).
# ---------------------------------------------------------------------------

#: Where referencing the global registry is allowed. **Only itself and the
#: registration file.**
_MAY_TOUCH_GLOBAL = {"kernelrule/features/__init__.py",
                     "kernelrule/features/physical.py"}


def test_library_never_imports_the_global_registry():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bad: list[str] = []
    for f in sorted((root / "kernelrule").rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        if rel in _MAY_TOUCH_GLOBAL:
            continue
        tree = ast.parse(f.read_text(), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                    a.name == "REGISTRY" for a in node.names):
                bad.append(f"  {rel}:{node.lineno} from ... import REGISTRY")
            elif (isinstance(node, ast.Attribute) and node.attr == "REGISTRY"
                  and isinstance(node.value, ast.Name)):
                bad.append(f"  {rel}:{node.lineno} {node.value.id}.REGISTRY")
    assert not bad, (
        "the library looks at the global registry directly — even when the "
        "condition changes, this one place keeps using the 24 a human wrote "
        "(§30.9):\n" + "\n".join(bad))


def test_no_function_defaults_to_the_global_registry():
    """★ Type (c) — a default that silently uses the 24 when the caller
    does not pass one."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bad: list[str] = []
    for f in sorted((root / "kernelrule").rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        tree = ast.parse(f.read_text(), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            a = node.args
            for d in [*a.defaults, *a.kw_defaults]:
                if isinstance(d, ast.Name) and d.id == "REGISTRY":
                    bad.append(f"  {rel}:{node.lineno} def {node.name}(...="
                               "REGISTRY)")
    assert not bad, (
        "there is a function that defaults to the global registry. When "
        "the caller does not pass one it silently uses the human's 24 "
        "(§26.4, §30.9):\n" + "\n".join(bad))


@pytest.fixture(scope="module")
def perf_table():
    """The measured bundle. The shape-level verdict needs **a real config
    set**."""
    import warnings

    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PerfTable.from_bundle("datasets/rtx-a6000-sm_86-c63710df",
                                     env_hash="c63710df", ok_only=False)


# ---------------------------------------------------------------------------
# ★ §30.12 — the automatic shape-level verdict
#
#   The generation path has no `shape_feature` decorator, so **every
#   generated feature was being registered as config level.** Then a rule
#   cannot branch with `if p.<x>:`, and a term constant within a shape
#   cannot change the ranking at all. 5 of F1's 21 were in that state
#   (D-65).
# ---------------------------------------------------------------------------
def test_detection_matches_the_hand_written_labels(perf_table):
    """★ Validating the verdict logic — does it agree with the labels a
    human attached by hand?

    Agreement means the logic is right; disagreement means the verdict has a
    defect.
    """
    import inspect
    from dataclasses import replace

    from kernelrule.features import REGISTRY
    from kernelrule.features.generated import detect_shape_level

    bad = []
    for n in sorted(REGISTRY._items):
        f = REGISTRY[n]
        try:
            src = inspect.getsource(f.fn)
        except (OSError, TypeError):
            src = ""
        got, why = detect_shape_level(
            replace(f, source=src, shape_level=False), perf_table)
        if got != f.shape_level:
            bad.append(f"  {n}: human {f.shape_level} vs automatic {got} "
                       f"({why})")
    assert not bad, ("the shape-level verdict disagrees with the human "
                     "labels:\n" + "\n".join(bad))


def test_detection_is_two_tiered():
    """The AST layer is what makes "constant only in this table"
    distinguishable."""
    from kernelrule.features.generated import uses_cfg

    assert uses_cfg("def f(p, hw, cfg) -> float:\n    return float(cfg.tile_m)")
    assert not uses_cfg("def f(p, hw, cfg) -> float:\n    return float(p.M)")


def test_recheck_warning_is_recorded(perf_table):
    """★ One that references cfg yet is constant **must be re-judged when
    the bundle changes**."""
    from dataclasses import replace

    from kernelrule.features import Feature
    from kernelrule.features.generated import detect_shape_level

    # A function that references cfg but whose value is constant
    code = ("def probe_const(p, hw, cfg) -> float:\n"
            "    return float(cfg.tile_m) * 0.0 + float(p.M)\n")
    env: dict = {}
    exec(compile(code, "<t>", "exec"), env)  # noqa: S102
    f = Feature(name="probe_const", fn=env["probe_const"],
                unit="dimensionless",
                expected_range=(0.0, 1e9), direction="neutral",
                code_hash="h", source=code)
    is_shape, why = detect_shape_level(replace(f, shape_level=False),
                                       perf_table)
    assert is_shape
    assert "Re-judgement" in why, why


def test_the_shape_level_verdict_looks_at_every_shape(perf_table):
    """★ D-160 — it used to look at the first 8 shapes only.

    `k_loop_padding_fraction` (the F1 run's own feature, D-159) is constant
    on those 8 and config-dependent on 3 of the 66 (K = 4097 · 4098 · 4100),
    so it was registered as shape level and its value was then taken from
    one representative config. **The verdict is checked by behaviour**: the
    same function must come out config level now, and still come out shape
    level under the old 8-shape window.
    """
    from kernelrule.features import Feature
    from kernelrule.features.generated import detect_shape_level

    code = ("def k_loop_padding_fraction(p, hw, cfg) -> float:\n"
            "    tk = max(cfg.tile_k, 1)\n"
            "    iters = (max(p.K, 1) + tk - 1) // tk\n"
            "    covered = iters * tk\n"
            "    return 1.0 - max(p.K, 1) / max(covered, 1)\n")
    env: dict = {}
    exec(compile(code, "<t>", "exec"), env)  # noqa: S102
    f = Feature(name="k_loop_padding_fraction",
                fn=env["k_loop_padding_fraction"], unit="ratio",
                expected_range=(0.0, 1.0), direction="higher_is_worse",
                code_hash="h", source=code)

    got, why = detect_shape_level(f, perf_table)
    assert got is False, f"it must be config level now: {why}"

    old, _why = detect_shape_level(f, perf_table, n_shapes=8)
    assert old is True, ("the 8-shape window is what produced the wrong "
                         "verdict — if this changes, the test no longer "
                         "shows what D-160 fixed")


def test_load_generated_requires_a_table():
    """★ Giving `table` a default makes callers leave it out (D-67).

    With one, two call sites left it out, and one of those was the stage-2
    path, so RuleWriter ran with **0 shape-level features**. If re-judgement
    really is unnecessary, `table=None` must be **stated explicitly** — that
    is distinguishable from leaving it out.
    """
    import inspect

    from kernelrule.features.loader import load_generated

    prm = inspect.signature(load_generated).parameters["table"]
    assert prm.default is inspect.Parameter.empty, "table has a default"
    assert prm.kind is inspect.Parameter.KEYWORD_ONLY


def test_every_load_generated_call_passes_table():
    """An exhaustive sweep of the call sites — principle 23 (one kind, not
    one place)."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted([*(root / "kernelrule").rglob("*.py"),
                     *(root / "experiments").glob("*.py")]):
        tree = ast.parse(f.read_text(), filename=f.name)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "load_generated"
                    and not any(k.arg == "table" for k in node.keywords)):
                bad.append(f"  {f.name}:{node.lineno}")
    assert not bad, ("call sites that do not pass table to "
                     "load_generated:\n" + "\n".join(bad))


# ---------------------------------------------------------------------------
# ★ 4-4 — `expected_range` must be **what the LLM declared** (D-71)
#
#   If the pipeline overwrites it with measurements, table information gets
#   into the prompt. The LLM cannot see the table, so any leak was made by
#   the pipeline.
# ---------------------------------------------------------------------------
def test_register_generated_keeps_the_declared_range(perf_table):
    from dataclasses import replace

    from kernelrule.features import FeatureRegistry
    from kernelrule.features.generated import register_generated

    # ⚠️ 2026-09-11 (D-162): this was `cfg.tile_m / p.M`, and once the
    #   duplication check started running it was rejected — the comparison
    #   set is the table's first 4 shapes and all four have M=1, where that
    #   formula is `tile_m` and lines up with `edge_waste` at 1.000. The
    #   subject here is the **declared range**, so the probe is an axis
    #   nothing else measures.
    code = ("def probe_range(p, hw, cfg) -> float:\n"
            "    return float(cfg.threads) / max(1.0, float(cfg.tile_k))\n")
    reg = FeatureRegistry("probe")
    meta = {"name": "probe_range", "unit": "dimensionless",
            "direction": "higher_is_worse", "expected_range": [0.0, 7.0]}
    hw_alt = replace(perf_table.hw, sm_count=perf_table.hw.sm_count * 2 + 3,
                     smem_per_block=int(perf_table.hw.smem_per_block * 0.7),
                     max_threads_per_sm=int(perf_table.hw.max_threads_per_sm * 1.33),
                     peak_tflops_f16=perf_table.hw.peak_tflops_f16 * 1.6,
                     bandwidth_gbps=perf_table.hw.bandwidth_gbps * 0.8,
                     regs_per_sm=int(perf_table.hw.regs_per_sm * 1.5),
                     l2_bytes=int(perf_table.hw.l2_bytes * 2))
    f = register_generated(code, registry=reg, meta=meta, table=perf_table,
                           matrix=FeatureMatrix(perf_table, REGISTRY),
                           hw_alt=hw_alt)
    # The observed range is far narrower than [0, 7]. The declaration must
    # stay as it is anyway.
    assert f.expected_range == (0.0, 7.0), (
        "the pipeline overwrote the declared range — table information gets "
        "into the prompt")


def test_range_warning_never_reaches_a_rejection_message():
    """The range warning contains the **observed min/max**. If that leaks
    into a rejection message it can be fed back to the LLM (the old
    `feature_writer.py`).
    """
    import inspect

    from kernelrule.features import validate as V

    src = inspect.getsource(V)
    assert "observed" in src, (
        "the range warning changed — this check is meaningless")
    # The range check must be "warn". As "fail" it leaks out through
    # `fails()`.
    i = src.index("observed")
    ctx = src[max(0, i - 400):i]
    assert '"range", "warn"' in ctx, (
        "the range check is not a warn — the observed values leak into "
        "FeatureRejected")


# ---------------------------------------------------------------------------
# ★ D-73 — the checker was banning **a field it had itself allowed**
# ---------------------------------------------------------------------------
def test_allowed_fields_are_not_caught_by_banned_words():
    """`hw.peak_tflops_f16` matched the banned word `"tflops"` as a
    substring.

    A proposal trying to build a roofline was refused that way — a defect of
    the checker looks like a failure of the LLM (principle 8, the same class
    as D-37).
    """
    from kernelrule.features.generated import (
        _BANNED,
        RAW_FIELDS,
        check_feature_code,
    )

    # There really are allowed fields that contain a banned word as a
    # substring
    risky = [f"{b}.{n}" for b, ns in RAW_FIELDS.items() for n in ns
             if any(x in f"{b}.{n}" for x in _BANNED)]
    assert risky, "there is no risky field — this check is meaningless"

    for ref in risky:
        code = (f"def probe_ok(p, hw, cfg) -> float:\n"
                f"    return float({ref}) * 1.0\n")
        assert check_feature_code(code, known=frozenset()) == "probe_ok", ref


def test_banned_words_still_catch_real_leaks():
    """The masking must not let a real leak through."""
    from kernelrule.features.generated import FeatureRejected, check_feature_code

    for leak in ("best_ms", "time_ms", "difficulty"):
        code = (f"def probe_leak(p, hw, cfg) -> float:\n"
                f"    {leak} = 1.0\n"
                f"    return {leak}\n")
        with pytest.raises(FeatureRejected):
            check_feature_code(code, known=frozenset())


def test_every_raw_field_has_a_meaning():
    """★ A field listed with only its name is either unused or used wrongly
    (D-159).

    `cfg.pipeline_kind` · `cfg.split_k_mode` · `cfg.max_blocks_per_sm` sat in
    the prompt as bare names for months.
    """
    from kernelrule.features.generated import (
        FIELD_MEANING,
        RAW_FIELDS,
        field_block,
    )

    for base, names in RAW_FIELDS.items():
        for n in names:
            key = f"{base}.{n}"
            assert key in FIELD_MEANING, f"{key} has no meaning line"
            assert len(FIELD_MEANING[key]) > 10, key
    txt = field_block()
    for base, names in RAW_FIELDS.items():
        for n in names:
            assert f"`{base}.{n}`" in txt, f"{base}.{n} is not rendered"


def test_the_feature_writer_is_told_no_hardware_value():
    """⛔ The FeatureWriter must not learn which GPU this is, or any of its
    numbers (D-159).

    A feature has to be a formula that holds on any GPU. Knowing "84 SMs"
    invites `/ 84`, and then it does not transfer — which is the claim this
    project rests on.

    ⚠️ The prompt used to say "Writing 84 or 101376 is rejected", which
    **handed over both numbers** while forbidding them.
    """
    from kernelrule.agents.openai_client import assemble_instructions
    from kernelrule.features.generated import field_block

    txt = assemble_instructions("feature", objective="regret") + field_block()
    for banned in ("A6000", "RTX", "sm_86", "101376", "116.1", "729.7",
                   "159.1", "6291456"):
        assert banned not in txt, f"the feature path names {banned!r}"


def test_the_stage_count_is_a_config_field_and_it_varies(perf_table):
    """★ D-161 — `cfg.stages` is exposed, and it is not `pipeline_kind`.

    The Analyst asked for "explicit pipeline stage count" six rounds running
    under F1, and under F2 the FeatureWriter rebuilt it out of
    `smem_bytes / stage_bytes` (|pearson| 0.999 with the real value). The
    depth was already exposed as `pipeline_kind`, flattened to two levels.

    Checked by behaviour on the measured table: more than two distinct
    values, and `pipeline_kind` is exactly the 2 / 3-and-above split.
    """
    from kernelrule.core.types import config_from_row
    from kernelrule.features.generated import FIELD_MEANING, RAW_FIELDS

    assert "stages" in RAW_FIELDS["cfg"]
    assert "cfg.stages" in FIELD_MEANING

    seen, pairs = set(), set()
    for p in list(perf_table.shapes())[:4]:
        for row in perf_table.frame_for(p).to_dict("records"):
            c = config_from_row(row)
            seen.add(c.stages)
            pairs.add((c.pipeline_kind, c.stages == 2))
    assert len(seen) > 2, f"stages is flat on this table: {sorted(seen)}"
    assert pairs <= {("pipelined", True), ("multistage", False)}, (
        f"pipeline_kind is not the 2 / 3+ split of stages: {sorted(pairs)}")


def test_swizzle_stays_out_of_the_raw_fields():
    """⛔ D-75 — `identity`/`horizontal` are SM80 words and do not transfer.
    Lifting `stages` out of `ext` is not a licence to lift the rest."""
    from kernelrule.features.generated import RAW_FIELDS, field_block

    flat = {f"{b}.{n}" for b, ns in RAW_FIELDS.items() for n in ns}
    for banned in ("cfg.swizzle_type", "cfg.swizzle_n", "cfg.warp_m",
                   "cfg.warp_n", "cfg.warp_k", "cfg.ext"):
        assert banned not in flat
    text = field_block()
    for word in ("swizzle", "warp_m", "identity", "horizontal"):
        assert word not in text, f"{word} is in the prompt"


# ---------------------------------------------------------------------------
# ★ D-162 — the duplication check has to actually run
# ---------------------------------------------------------------------------
_WAVES_COPY = ("def waves_copy(p, hw, cfg) -> float:\n"
               "    tiles_m = (p.M + cfg.tile_m - 1) // max(cfg.tile_m, 1)\n"
               "    tiles_n = (p.N + cfg.tile_n - 1) // max(cfg.tile_n, 1)\n"
               "    ctas = tiles_m * tiles_n * max(cfg.split_k, 1)\n"
               "    return ctas / max(hw.sm_count * cfg.max_blocks_per_sm, 1)\n")


def _small(table, names=("waves", "edge_waste", "reg_pressure")):
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY, FeatureRegistry

    reg = FeatureRegistry("dup")
    for n in names:
        reg.add(REGISTRY[n])
    return reg, FeatureMatrix(table, reg)


def test_a_copy_of_an_existing_axis_is_rejected(perf_table):
    """★ D-162 — before this, `waves` under another name passed with a
    maximum correlation of **0.000**: the candidate was measured on 6
    sampled shapes and the comparison set on the first 4, and every pair
    whose lengths did not line up was skipped. On this table they never do.
    """
    import pytest

    from kernelrule.features.generated import FeatureRejected, register_generated
    from kernelrule.features.validate import alt_hw

    reg, m = _small(perf_table)
    meta = {"unit": "count", "expected_range": [0.0, 1e7],
            "direction": "higher_is_worse", "rationale": "a copy"}
    with pytest.raises(FeatureRejected, match="duplication"):
        register_generated(_WAVES_COPY, registry=reg, meta=meta,
                           table=perf_table, matrix=m,
                           hw_alt=alt_hw(perf_table.hw))


def test_a_mismatched_comparison_set_is_an_error_not_a_skip(perf_table):
    """⛔ Skipping is how the check came to be there without ever running.
    A pool that cannot say which shapes it holds must **stop** the
    verdict."""
    import pytest

    from kernelrule.features import Feature
    from kernelrule.features.generated import compile_feature
    from kernelrule.features.validate import alt_hw, validate_feature

    reg, m = _small(perf_table)
    name, fn = compile_feature(_WAVES_COPY, known=frozenset(reg._items))
    f = Feature(name=name, fn=fn, unit="count", expected_range=(0.0, 1e7),
                direction="higher_is_worse", code_hash="h",
                source=_WAVES_COPY)
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import FeatureRegistry

    one = FeatureRegistry("probe-dup")
    one.add(f)
    probe = FeatureMatrix(perf_table, one)
    with pytest.raises(ValueError, match="which shapes"):
        validate_feature(f, perf_table, probe, hw_alt=alt_hw(perf_table.hw),
                         others={"waves": m.column("waves")})


# ---------------------------------------------------------------------------
# ★ D-163 — the banned-word scan must not read prose
# ---------------------------------------------------------------------------
def test_a_docstring_may_say_execute_important_randomly():
    """★ D-163 — `'exec'` matched "execute" in a docstring and an axis was
    refused for it (D-162 run, r6). The same shape as D-73: **the checker
    banned what it had itself allowed.** "import" is inside "important" and
    "random" inside "randomly" — prose is full of them.
    """
    from kernelrule.features.generated import check_feature_code

    code = ('def probe_prose(p, hw, cfg) -> float:\n'
            '    """CTAs needed to execute the tiled grid. This is important\n'
            '    for randomly shaped problems; nothing is imported here."""\n'
            '    return float(cfg.tile_m) / max(1.0, float(p.M))\n')
    assert check_feature_code(code, known=frozenset()) == "probe_prose"


@pytest.mark.parametrize("code,word", [
    ("def b(p, hw, cfg) -> float:\n    return float(np.random.rand())\n",
     "random"),
    ("def b(p, hw, cfg) -> float:\n    import os\n    return 1.0\n", "import"),
    ("def b(p, hw, cfg) -> float:\n    return float(cfg.__class__ is int)\n",
     "__"),
])
def test_the_banned_words_still_catch_real_code(code, word):
    """⚠️ Reading prose is what was wrong, not the ban. An identifier that
    really contains the word is still refused."""
    from kernelrule.features.generated import FeatureRejected, check_feature_code

    with pytest.raises(FeatureRejected):
        check_feature_code(code, known=frozenset())


def test_the_comparison_shapes_are_not_all_one_m(perf_table):
    """★ D-163 — the comparison set was the table's first 4 shapes and all
    four are `M=1`, so a shape-level axis had four values in the whole set.
    Measured against the whole table's verdict: 4 shapes gives 5 false
    positives and misses one, 12 spread by M gives 1 and misses none.
    """
    from kernelrule.features.generated import DUP_SHAPES, _spread_shapes

    sh = _spread_shapes(perf_table, DUP_SHAPES)
    assert len(sh) == DUP_SHAPES
    assert len({p.M for p in sh}) >= 8, [p.M for p in sh]
    # and it is reproducible
    assert [p.key for p in sh] == [
        p.key for p in _spread_shapes(perf_table, DUP_SHAPES)]


# ---------------------------------------------------------------------------
# ★ D-166 §M — the axis list of a finished run
#
#   A rule from an F2 run uses axes the FeatureWriter built, in three
#   places: the condition's base, `stage1-features/proposals.jsonl`, and
#   **`features.jsonl` of the loop directory**. The third is the one that
#   gets forgotten — `transfer_generated` was fixed for it in D-165 §1 and
#   `export_rules` / `verify_rules` were not looked at (principle 23), so a
#   campaign rule loaded as `AttributeError: unregistered feature` and
#   landed in the "cannot be re-scored" bucket next to a green line.
# ---------------------------------------------------------------------------
@pytest.mark.needs_bundle
def test_a_rebuilt_run_registry_rescores_that_run_s_archive(perf_table):
    """★ The reproduction: `runs/cap1`'s best rule, re-scored through a
    registry rebuilt from the run's artefacts, equals the `regret` the loop
    recorded.

    It is an exact-equality check on purpose. Both sides are the same
    deterministic formula over the same shapes — a tolerance here would
    hide the axis list being subtly different.
    """
    import json
    from pathlib import Path

    import numpy as np

    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.scoring import evaluate_scores
    from kernelrule.core.weights import make_score_of
    from kernelrule.features.loader import run_registry

    run, seed = "cap1", 0
    arc = Path(f"runs/{run}-s{seed}/archive.jsonl")
    if not arc.exists():
        pytest.skip(f"{arc} is not here — `runs/` is .gitignore'd")

    reg, origin = run_registry(run, table=perf_table, seed=seed)
    layers = {v.split()[0] for v in origin.values()}
    assert "loop" in layers, (
        "the loop's own axes are missing from the rebuilt registry — that "
        f"is the D-165 §1 trap. layers: {sorted(layers)}")

    from experiments.f1_pipeline import _splits
    cfg = json.loads(Path(f"runs/{run}/config.json").read_text())
    assert cfg["split_kind"] == "kfold0-seed12345", cfg["split_kind"]
    splits = _splits(perf_table, fold=0, split_seed=12345, k=3)
    matrix = FeatureMatrix(perf_table, reg)

    rows = [json.loads(ln) for ln in arc.open() if ln.strip()]
    best = min(rows, key=lambda e: e["regret"])
    fn = compile_rule(best["code"])
    ev = evaluate_scores(make_score_of(fn, matrix, np.asarray(best["w"],
                                                              float)),
                         perf_table, list(splits.train.shapes), ks=(1,))
    assert ev.at(1) == pytest.approx(best["regret"], abs=1e-9), (
        f"recorded {best['regret']:.6f} != rescored {ev.at(1):.6f}")
