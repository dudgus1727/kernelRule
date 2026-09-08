"""Term ablation (D-85). The tool that verifies the Critic's qualitative
verdict quantitatively."""
from __future__ import annotations

import pytest

from kernelrule.rules.ablate import AblateError, drop_terms, term_indices

SIMPLE = """def score(f, p, hw, w):
    s = f.a * w[0]
    s = s + f.b * w[1]
    s = s + f.c * w[2]
    return s
"""


def test_drop_middle_term_and_renumber():
    out = drop_terms(SIMPLE, {1})
    assert "f.b" not in out
    assert term_indices(out) == [0, 1], "the weights were not renumbered"


def test_drop_first_term_keeps_s_defined():
    """★ Removing the first term makes `s = s + …` the first assignment —
    and `s` does not exist."""
    out = drop_terms(SIMPLE, {0})
    assert "s = f.b * w[0]" in out, out
    ns: dict = {}
    exec(compile(out, "<t>", "exec"), ns)          # noqa: S102
    from types import SimpleNamespace
    f = SimpleNamespace(a=1.0, b=2.0, c=3.0)
    assert ns["score"](f, None, None, [1.0, 1.0]) == 5.0


def test_drop_term_in_return_line():
    code = ("def score(f, p, hw, w):\n"
            "    s = f.a * w[0]\n"
            "    return s + f.b * w[1]\n")
    out = drop_terms(code, {1})
    assert out.strip().endswith("return s")
    assert term_indices(out) == [0]


def test_refuses_when_two_terms_share_a_line():
    """★ Not skipped silently — "could not remove" differs from "removed
    with no effect"."""
    code = "def score(f, p, hw, w):\n    return f.a * w[0] + f.b * w[1]\n"
    with pytest.raises(AblateError, match="several terms on one line"):
        drop_terms(code, {1})


def test_refuses_when_branch_would_be_empty():
    code = ("def score(f, p, hw, w):\n"
            "    s = f.a * w[0]\n"
            "    if p.is_memory_bound:\n"
            "        s = s + f.b * w[1]\n"
            "    return s\n")
    with pytest.raises(AblateError, match="branch body is empty"):
        drop_terms(code, {1})


def test_refuses_to_drop_everything():
    code = "def score(f, p, hw, w):\n    return f.a * w[0]\n"
    with pytest.raises(AblateError, match="removing every term"):
        drop_terms(code, {0})


def test_dropping_nothing_is_identity_modulo_renumber():
    out = drop_terms(SIMPLE, set())
    assert term_indices(out) == [0, 1, 2]
    assert "f.a" in out and "f.b" in out and "f.c" in out
