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
