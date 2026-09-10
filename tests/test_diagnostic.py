"""The diagnostic report (§12). **It stops the template writing its
conclusions in advance.**"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import numpy as np
import pytest
from toy import make_table

import kernelrule.report.diagnostic as D
from kernelrule.core.splits import Split, SplitError

#: Comparative words that must not be written without a computation.
#: Nailing **the conclusion of other data** into the template, rather than
#: this data's, is the trap this project stepped into.
#:
#: ⚠️ The report is English (D-146). The Korean list it replaced is in the
#: history, not here.
COMPARATIVES = ("larger than", "smaller than", "greater than", "dominant",
                "dominates", "mostly", "much more", "much larger",
                "in general", "generally", "typically")

#: The exception — physical constants, not observations of this split.
EXEMPT_FUNCS = {"hardware_block"}


def _rendered_literals(fn_name: str, node: ast.AST):
    """Extracts the string literals rendered through `add(...)` /
    `L.append(...)`.

    Returns: (text, does it contain a computed value)
    """
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        f = sub.func
        name = (f.id if isinstance(f, ast.Name) else
                f.attr if isinstance(f, ast.Attribute) else "")
        if name not in ("add", "append"):
            continue
        for arg in sub.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append((arg.value, False))
            elif isinstance(arg, ast.JoinedStr):
                lit = "".join(v.value for v in arg.values
                              if isinstance(v, ast.Constant))
                computed = any(isinstance(v, ast.FormattedValue)
                               for v in arg.values)
                out.append((lit, computed))
            elif isinstance(arg, ast.BinOp):
                for side in (arg.left, arg.right):
                    if isinstance(side, ast.Constant) and isinstance(
                            side.value, str):
                        out.append((side.value, False))
                    elif isinstance(side, ast.JoinedStr):
                        lit = "".join(v.value for v in side.values
                                      if isinstance(v, ast.Constant))
                        computed = any(isinstance(v, ast.FormattedValue)
                                       for v in side.values)
                        out.append((lit, computed))
    return out


def test_template_never_states_an_uncomputed_comparison():
    """★ It blocks in code the trap this project actually stepped into.

    Block 3 had "size stratification moves more than difficulty
    stratification" written in advance, and the actual numbers on the
    training split were the opposite (0.0007 vs 0.1651). **When the report
    contradicts its own data, the LLM believes the sentence, not the data.**

    The rule: if a rendered string contains a comparative word, that string
    **must contain a computed value.** A pure literal comparison is
    forbidden.
    """
    tree = ast.parse(Path(D.__file__).read_text())
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name in EXEMPT_FUNCS:
            continue
        for text, computed in _rendered_literals(node.name, node):
            if computed:
                continue
            for w in COMPARATIVES:
                if w in text:
                    bad.append(f"{node.name}(): {w!r} in {text[:70]!r}")
    assert not bad, (
        "the template contains an uncomputed comparison:\n  "
        + "\n  ".join(bad)
        + "\n  Build the sentence from the computed result (§12).")


def _toy(regret_by_shape):
    """A toy table with a per-shape regret specified + an identity
    rule."""
    times = {}
    for (m, n, k), rel in regret_by_shape.items():
        times[(m, n, k)] = [1.0, float(rel)]
    return make_table(times)


def test_the_report_names_no_regime_axis():
    """★ The report must not tell the model which axis to look along
    (D-156).

    It used to print eight bands (`t_sol < 0.5ms`, `waves 1~4`, ...), label
    every case with its band, and give per-band baselines. Every one of those
    is **our** axis: the SOL gap was the biggest number in the report, the
    Analyst read it every round, and the rules branched there.

    ⚠️ This replaces `test_stratification_sentence_follows_the_data`, which
    checked that the size-gap sentence followed the data. There is no such
    sentence now.
    """
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.noise import NoiseModel
    from kernelrule.features import Feature, FeatureRegistry

    reg = FeatureRegistry("toy")
    reg.add(Feature(name="idx", fn=lambda p, hw, c: 0.0, unit="dimensionless",
                    expected_range=(0.0, 10.0), direction="neutral",
                    vec=lambda df, hw, p: np.arange(len(df), dtype=float),
                    code_hash="x"))
    reg.add(Feature(name="is_memory_bound", fn=lambda p, hw, c: 0.0,
                    expected_range=(0.0, 1.0), direction="neutral",
                    unit="dimensionless", shape_level=True, code_hash="z"))

    code = "def score(f, p, hw, w):\n    return f.idx * w[0]\n"

    def score(f, p, hw, w):
        return f.idx * w[0]

    short_bad = {(64, 64, 64): [0.02, 0.10], (4096, 4096, 4096): [2.0, 2.02]}
    long_bad = {(64, 64, 64): [0.02, 0.0202], (4096, 4096, 4096): [2.0, 10.0]}
    texts = []
    for times in (short_bad, long_bad):
        t = make_table(times, noise=NoiseModel.a6000_reference())
        m = FeatureMatrix(t, reg)
        rep = D.build_report(run_id="toy", table=t, matrix=m, score_fn=score,
                             weights=[-1.0], code=code,
                             train=Split("train", tuple(t.shapes())))
        texts.append(rep.render())
    a, b = texts
    assert a != b, "the data is opposite yet the reports are identical"
    for txt in (a, b):
        low = txt.lower()
        for banned in ("t_sol", "0.5ms", "regime", "memory-bound",
                       "compute-bound", "waves <", "short mainloop",
                       "stratification", "size_gap"):
            assert banned not in low, f"the report names an axis: {banned!r}"


def test_report_refuses_non_train_split():
    """★ There is no path by which a holdout score enters the prompt
    (§12.3)."""
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import Feature, FeatureRegistry

    reg = FeatureRegistry("t2")
    reg.add(Feature(name="idx", fn=lambda p, hw, c: 0.0, unit="dimensionless",
                    expected_range=(0.0, 10.0), direction="neutral",
                    vec=lambda df, hw, p: np.arange(len(df), dtype=float),
                    code_hash="x"))
    t = make_table({(1024, 4096, 4096): [1.0, 2.0]})
    m = FeatureMatrix(t, reg)
    for role in ("val", "test"):
        with pytest.raises(SplitError,
                           match="takes the training split only"):
            D.build_report(run_id="x", table=t, matrix=m,
                           score_fn=lambda f, p, hw, w: f.idx * w[0],
                           weights=[1.0], code="def score(f, p, hw, w):\n"
                                                "    return f.idx * w[0]\n",
                           train=Split(role, tuple(t.shapes())))


def test_sigma_is_relative_to_noise_floor():
    from kernelrule.core.noise import NoiseModel
    m = NoiseModel.a6000_reference()
    # 4ms shape: floor 0.053%. A 1% difference is about 19 sigma
    assert D._sigma(4.04, 4.0, m) == pytest.approx(0.01 / m.floor(4.0), rel=1e-6)
    # 11us shape: floor 9.1%. The same 1% is 0.11 sigma — inside the noise
    assert D._sigma(0.0113 * 1.01, 0.0113, m) < 1.0


@pytest.mark.needs_bundle
def test_real_report_has_no_holdout_and_fits_budget(real_bundle_path):
    """A report built from the real table stays within budget and contains
    no holdout shape."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import split_by_M_range
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY
    from kernelrule.rules.human_guided import CODE, W0, score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                   ok_only=False)
        fm = FeatureMatrix(tb, REGISTRY)
        sp = split_by_M_range(tb.shapes())
        rep = D.build_report(run_id="t", table=tb, matrix=fm, score_fn=score,
                             weights=W0, code=CODE, train=sp.train)
    txt = rep.render()
    assert rep.token_estimate() < 9000, rep.token_estimate()
    held = {f"{p.M}x{p.N}x{p.K}" for p in sp.val}
    for h in held:
        assert h not in txt, f"the holdout shape {h} entered the report"
    # Each case has its σ and its usage flag
    assert "the noise floor" in txt and "unused" in txt


# ---------------------------------------------------------------------------
# Aggregates do not cross the holdout either (§12.3 / D-28)
# ---------------------------------------------------------------------------
# `build_report` takes only the train split, but `table_facts` was a list of
# free strings and **bypassed that check entirely**. Block 3.5 of the first
# real run was computed on the full table. The bypass path itself was
# removed, and that is pinned here.

def _facts_table():
    return make_table({(1024, 4096, 4096): [1.0, 2.0, 3.0],
                       (512, 4096, 4096): [1.0, 1.5],
                       (2048, 4096, 4096): [2.0, 2.2, 9.0]})


def test_table_facts_rejects_holdout_splits():
    from kernelrule.report.table_facts import TableFacts
    t = _facts_table()
    shapes = tuple(t.shapes())
    TableFacts.compute(t, Split("train", shapes))       # train passes
    for role in ("val", "test"):
        with pytest.raises(SplitError, match="12.3"):
            TableFacts.compute(t, Split(role, shapes))


def test_table_facts_only_sees_the_train_shapes():
    """★ A shape absent from the training split must not mix into the
    aggregates."""
    from kernelrule.report.table_facts import TableFacts
    t = _facts_table()
    shapes = list(t.shapes())
    part = TableFacts.compute(t, Split("train", tuple(shapes[:2])))
    whole = TableFacts.compute(t, Split("train", tuple(shapes)))
    assert part.n_shapes == 2
    assert whole.n_shapes == len(shapes)
    assert "training split only, 2 shapes" in part.lines[0]
    assert part.lines != whole.lines


def test_build_report_refuses_raw_strings():
    """Accepting free strings makes the split check do nothing."""
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import Feature, FeatureRegistry

    reg = FeatureRegistry("t3")
    reg.add(Feature(name="idx", fn=lambda p, hw, c: 0.0, unit="dimensionless",
                    expected_range=(0.0, 10.0), direction="neutral",
                    vec=lambda df, hw, p: np.arange(len(df), dtype=float),
                    code_hash="x"))
    t = make_table({(1024, 4096, 4096): [1.0, 2.0]})
    m = FeatureMatrix(t, reg)
    with pytest.raises(SplitError, match="TableFacts.compute"):
        D.build_report(run_id="x", table=t, matrix=m,
                       score_fn=lambda f, p, hw, w: f.idx * w[0],
                       weights=[1.0],
                       code="def score(f, p, hw, w):\n    return f.idx * w[0]\n",
                       train=Split("train", tuple(t.shapes())),
                       table_facts=[("a sentence computed on the full "
                                     "table")])


# ---------------------------------------------------------------------------
# ★ §12.3b — block 3.5 **must not name an axis**
#
#   "shapes whose answer set contains a spilling kernel: 0/61" is an answer
#   summary. With that line, the LLM did not learn from the table that it
#   need not look at `has_spill` — it was **given** it. GBDT feature
#   importances were removed for the same reason.
#
#   It is worse under F0~F3 — the features the LLM builds all have different
#   names, so the axis-name mapping does not match at all and yet it stays
#   in the prompt.
# ---------------------------------------------------------------------------

#: What must not appear in block 3.5 — the config axis column names and
#: their values.
_FORBIDDEN_AXES = ("has_spill", "spill", "ext_stages", "stages=2", "pipelined",
                   "ext_warp_m", "warp_m", "split_k_mode", "parallel",
                   "mainloop_iters", "workspace_bytes", "waves_occ",
                   "grid_tiles", "inst_total", "tile_m",
                   "regs_total_per_block", "GBDT")


def test_block_3_5_never_names_an_axis():
    import warnings

    from kernelrule.core.splits import Split
    from kernelrule.core.table import PerfTable
    from kernelrule.report.table_facts import TableFacts

    bundle = "datasets/rtx-a6000-sm_86-c63710df"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = PerfTable.from_bundle(bundle, env_hash="c63710df", ok_only=False)
    train = Split("train", tuple(p for p in t.shapes()
                                 if 11008 not in (p.N, p.K)))
    facts = TableFacts.compute(t, train)

    body = "\n".join(facts.lines)
    hit = [a for a in _FORBIDDEN_AXES if a in body]
    assert not hit, (
        f"block 3.5 names an axis: {hit}\n"
        "That is an answer summary (§12.3b). What may stay is only 'the size "
        "of the room' — the fixed-config top-k, the per-regime breakdown, "
        "the tie width.\n" + body)


def test_block_3_5_never_annotates_individual_features():
    """`by_feature` attaches table observations **to each feature's
    description** — that is naming an axis."""
    import warnings

    from kernelrule.core.splits import Split
    from kernelrule.core.table import PerfTable
    from kernelrule.report.table_facts import TableFacts

    bundle = "datasets/rtx-a6000-sm_86-c63710df"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = PerfTable.from_bundle(bundle, env_hash="c63710df", ok_only=False)
    train = Split("train", tuple(p for p in t.shapes()
                                 if 11008 not in (p.N, p.K)))
    assert not TableFacts.compute(t, train).by_feature, (
        "the per-feature table observations are still alive. The feature "
        "descriptions in the prompt would carry 'this axis enters the answer "
        "0 times' (§12.3b).")
