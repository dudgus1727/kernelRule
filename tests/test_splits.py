"""Splits (§10.2) — the role living in the type, and the seal on the final
split."""
from __future__ import annotations

import pytest

from kernelrule.core.types import Problem


# ---------------------------------------------------------------------------
# ★ 4-2 — the seal on the final split (§30.15)
#
#   "exactly once at the end" was **an intention, not enforcement.** One
#   could simply read `splits.test.shapes`. It is blocked in code, and a run
#   that opened it stays in the record.
# ---------------------------------------------------------------------------
def test_test_split_is_sealed(monkeypatch):
    from kernelrule.core.splits import UNSEAL_ENV, Split, SplitError

    monkeypatch.delenv(UNSEAL_ENV, raising=False)
    p = Problem(M=128, N=128, K=128)
    te = Split("test", (p,))
    with pytest.raises(SplitError, match="sealed"):
        _ = te.shapes


def test_train_and_val_are_not_sealed(monkeypatch):
    from kernelrule.core.splits import UNSEAL_ENV, Split

    monkeypatch.delenv(UNSEAL_ENV, raising=False)
    p = Problem(M=128, N=128, K=128)
    assert len(Split("train", (p,)).shapes) == 1
    assert len(Split("val", (p,)).shapes) == 1


def test_size_is_visible_without_unsealing(monkeypatch):
    """★ That a split **exists** and looking inside it are different
    things."""
    from kernelrule.core.splits import UNSEAL_ENV, Split

    monkeypatch.delenv(UNSEAL_ENV, raising=False)
    te = Split("test", (Problem(M=1, N=1, K=1),))
    assert len(te) == 1          # unaffected by the seal
    assert te.role == "test"


def test_unseal_flag_opens_it_and_is_reportable(monkeypatch):
    from kernelrule.core.splits import UNSEAL_ENV, Split, is_unsealed

    p = Problem(M=128, N=128, K=128)
    te = Split("test", (p,))
    monkeypatch.setenv(UNSEAL_ENV, "1")
    assert is_unsealed()
    assert len(te.shapes) == 1
    # An empty value or "0" keeps the seal — so it cannot open by accident
    for off in ("", "0", "false"):
        monkeypatch.setenv(UNSEAL_ENV, off)
        assert not is_unsealed(), off


def test_nothing_reads_the_private_field(monkeypatch):
    """Reading `_shapes` directly bypasses the seal — it must not appear in
    the library."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted([*(root / "kernelrule").rglob("*.py"),
                     *(root / "experiments").glob("*.py")]):
        rel = f.relative_to(root).as_posix()
        if rel == "kernelrule/core/splits.py":
            continue            # the implementation itself
        for node in ast.walk(ast.parse(f.read_text(), filename=rel)):
            if isinstance(node, ast.Attribute) and node.attr == "_shapes":
                bad.append(f"  {rel}:{node.lineno}")
    assert not bad, ("`_shapes` is read directly, bypassing the seal "
                     "(§30.15):\n" + "\n".join(bad))


# ---------------------------------------------------------------------------
# ★ D-167 §R — the shape population is one function now
# ---------------------------------------------------------------------------
@pytest.mark.needs_bundle
def test_experiment_shapes_is_61_on_the_a6000_table():
    """★ The count is pinned because **every number in the repository has
    it as a denominator.**

    The predicate used to be copied into 36 places across 35 files. Merging
    them must not move the population by one shape, and a future change to
    `ALIGNMENT_REQUIRED` has to trip this rather than quietly re-baseline
    the whole repository.
    """
    import warnings

    from kernelrule.core.splits import ALIGNMENT_REQUIRED, experiment_shapes
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = PerfTable.from_bundle("datasets/rtx-a6000-sm_86-c63710df",
                                      env_hash="c63710df", ok_only=False)
    assert ALIGNMENT_REQUIRED == 8
    shapes = experiment_shapes(table)
    assert len(table.shapes()) == 66
    assert len(shapes) == 61, "the shape population moved"
    dropped = {f"{p.M}x{p.N}x{p.K}"
               for p in table.shapes()} - {f"{p.M}x{p.N}x{p.K}"
                                           for p in shapes}
    assert dropped == {"1024x4096x4097", "1024x4096x4098", "1024x4096x4100",
                       "1024x4098x4096", "1024x4100x4096"}, dropped


def test_no_experiment_copies_the_alignment_predicate():
    """★ Principle 2, counted. 36 copies is what this replaced."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bad = [f"{f.relative_to(root)}:{i + 1}"
           for f in sorted((root / "experiments").glob("*.py"))
           for i, ln in enumerate(f.read_text().splitlines())
           if "align_a == 8" in ln]
    assert not bad, (
        "the alignment predicate was copied again instead of calling "
        "`kernelrule.core.splits.experiment_shapes` (D-167 §R):\n"
        + "\n".join(bad))
