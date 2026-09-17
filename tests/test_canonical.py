"""Does final scoring use the loop's split as is (§10.2 / D-36)?

The scorer and the loop each decided the split, and 11 of the 19 holdout
shapes ended up being the loop's training shapes. The structure had evolved
while looking at them, so it was not a holdout.
**The arbitrary-split path itself was removed, and that is pinned here.**
"""

from __future__ import annotations

import numpy as np
import pytest

from kernelrule.core.canonical import canonical_score
from kernelrule.core.splits import Split, SplitError, SplitSet
from tests.toy import make_table

_CODE = "def score(f, p, hw, w):\n    return f.idx * w[0]\n"


def _setup():
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import Feature, FeatureRegistry

    reg = FeatureRegistry("canon")
    reg.add(Feature(name="idx", fn=lambda p, hw, c: 0.0, unit="dimensionless",
                    expected_range=(0.0, 10.0), direction="neutral",
                    vec=lambda df, hw, p: np.arange(len(df), dtype=float),
                    code_hash="x"))
    t = make_table({(1024, 4096, 4096): [1.0, 2.0],
                    (512, 4096, 4096): [1.0, 1.5],
                    (2048, 4096, 4096): [2.0, 2.2],
                    (4096, 4096, 4096): [1.0, 3.0]})
    return t, FeatureMatrix(t, reg)


def test_canonical_requires_the_loop_splitset():
    """★ There is no path that picks shapes separately."""
    t, m = _setup()
    shapes = list(t.shapes())
    with pytest.raises(SplitError, match="SplitSet"):
        canonical_score(_CODE, [1.0], table=t, matrix=m, splits=shapes)
    with pytest.raises(SplitError, match="SplitSet"):
        canonical_score(_CODE, [1.0], table=t, matrix=m,
                        splits=Split("train", tuple(shapes)))


def test_holdout_never_overlaps_the_training_shapes():
    """Overlap is already blocked when the SplitSet is built — pin that."""
    t, _ = _setup()
    shapes = list(t.shapes())
    with pytest.raises(SplitError):
        SplitSet(train=Split("train", tuple(shapes)),
                 val=Split("val", (shapes[0],)))


def test_canonical_scores_only_the_val_shapes():
    """★ The holdout score comes only from `splits.val` (D-36)."""
    t, m = _setup()
    shapes = list(t.shapes())
    splits = SplitSet(train=Split("train", tuple(shapes[:3])),
                      val=Split("val", tuple(shapes[3:])))
    r = canonical_score(_CODE, [1.0], table=t, matrix=m, splits=splits)
    assert r.n_holdout == len(shapes) - 3
    assert tuple(r.evaluation.shapes) == tuple(shapes[3:])
    # In-sample comes from the training shapes — the two must not be the
    # same set
    assert set(r.evaluation.shapes).isdisjoint(splits.train.shapes)


def test_scoring_does_not_fit():
    """⛔ 2026-09-18 (D-182) — the scorer must **not** call `fit_weights`.

    Scoring is where a finished rule is measured. Refitting there measures
    something the loop never produced: the loop had already fitted on those
    very shapes, and its answer was being used only as a starting point for
    another 300 evaluations.

    ★ Two ways of checking, because one alone is weak:

    ```
    1  ★ the **AST** of `canonical_score` contains no call to `fit_weights`
       (grepping the text would hit this very docstring)
    2  the weights come back ★ exactly as handed in — a refit would move them
    ```

    ⛔ The old test here (`test_thin_training_split_warns_instead_of_pretending`)
    pinned a "too few training shapes" warning. That warning belonged to the
    refit; the scorer no longer looks at the training split to fit anything,
    so whether the weights are trustworthy is the loop's verdict to make.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(canonical_score)))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "fit_weights" not in called | imported, \
        "the scorer is fitting again (D-182)"

    t, m = _setup()
    shapes = list(t.shapes())
    splits = SplitSet(train=Split("train", tuple(shapes[:3])),
                      val=Split("val", tuple(shapes[3:])))
    good = canonical_score(_CODE, [1.0], table=t, matrix=m, splits=splits)
    bad = canonical_score(_CODE, [-50.0], table=t, matrix=m, splits=splits)
    assert good.weights == {"all": [1.0]}
    assert bad.weights == {"all": [-50.0]}, \
        "the weights came back changed — something refitted them"


def test_scoring_fits_one_weight_vector():
    """★ D-179 — one fit, one vector, read on the whole holdout.

    The loop optimises **one** weight vector. While scoring fitted two (one
    per SOL regime), the optimised objective and the reported number were
    different functions.
    """
    t, m = _setup()
    shapes = list(t.shapes())
    splits = SplitSet(train=Split("train", tuple(shapes[:3])),
                      val=Split("val", tuple(shapes[3:])))
    r = canonical_score(_CODE, [1.0], table=t, matrix=m, splits=splits)
    assert list(r.weights) == ["all"], r.weights
    assert r.n_holdout == len(shapes) - 3
    # ⚠️ by_regime is a **read-only** roofline breakdown — never short/long
    assert set(r.by_regime) <= {"mem", "comp"}


# ---------------------------------------------------------------------------
# Do the committed rules and the recorded scores agree?
# ---------------------------------------------------------------------------
# `runs/` is gitignored, so there was no way to check the documented
# numbers. Committing the rule and the **fitted** weights makes scoring
# deterministic, so it can be verified.

def test_exported_rules_match_their_index():
    """★ Do `rules/*.py` and `index.json` match?

    A full rescore is what `experiments/verify_rules.py` does (it needs a
    bundle and takes minutes). Here we only check that **the files and the
    record do not disagree** — forgetting to export lets the documentation go
    stale silently.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "docs/artifacts/rules"
    idx = root / "index.json"
    if not idx.exists():
        pytest.skip("no exported rules — experiments/export_rules.py")
    index = json.loads(idx.read_text())
    assert index, "index.json is empty"
    for row in index:
        f = root / f"{row['run']}.py"
        assert f.exists(), f"no rule file for {row['run']}"
        src = f.read_text()
        assert "def score(" in src and "W_FITTED" in src
        w = row["weights"]
        assert set(w) == {"short", "long"}, (
            f"{row['run']}: a regime is missing")
        assert all(len(v) > 0 for v in w.values())
        # ⚠️ The two regimes **may have identical weights.** Nelder-Mead
        #   really can fail to take a single step on a step objective (D-54)
        #   — several of the 12 are like that. It is not an export bug, so it
        #   does not fail here. The watchdog is `w_moved` in `index.json`.
