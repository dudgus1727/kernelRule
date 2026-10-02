"""Removing one weight's term(s) from a rule (D-190 §7). **A removal must
equal `w[i] = 0`** — nothing else is allowed to change."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from kernelrule.rules.prune import drop_weight, prune_dead, removable_indices


def _run(code: str, w, f, p=None):
    g = {"np": np}
    exec(code, g)                                          # noqa: S102
    return np.asarray(g["score"](f, p or SimpleNamespace(x=1.0), None,
                                 np.asarray(w, float)), float)


F = SimpleNamespace(a=np.array([1.0, 2.0, 3.0]), b=np.array([3.0, 1.0, 2.0]),
                    c=np.array([0.5, 0.25, 4.0]))

RULE = ("def score(f, p, hw, w):\n"
        "    # the first term\n"
        "    s = f.a * w[0]\n"
        "    # the second term\n"
        "    s = s + f.b * w[1]\n"
        "    # the third term\n"
        "    s = s + np.power(f.c, w[2]) * w[3]  # an exponent pair\n"
        "    return s\n")


def test_removal_equals_zeroing_the_weight():
    w = [1.5, -0.7, 0.5, 2.0]
    for i in removable_indices(RULE):
        code, w2, removed = drop_weight(RULE, w, i)
        wz = np.asarray(w, float)
        wz[removed] = 0.0
        assert np.allclose(_run(code, w2, F), _run(RULE, wz, F))


def test_exponent_and_threshold_weights_are_not_removable():
    assert removable_indices(RULE) == {0, 1, 3}
    th = ("def score(f, p, hw, w):\n"
          "    return np.where(f.a < w[0], f.b, 0.0) * w[1]\n")
    assert removable_indices(th) == {1}
    assert drop_weight(th, [1.0, 1.0], 0) is None


def test_the_exponent_weight_goes_with_its_term_and_indices_are_renumbered():
    code, w2, removed = drop_weight(RULE, [1.0, 2.0, 3.0, 4.0], 3)
    assert removed == [2, 3]
    assert w2 == [1.0, 2.0]
    assert "w[2]" not in code and "w[3]" not in code
    assert "an exponent pair" not in code
    assert "the third term" not in code          # `return s` is not a term


def test_renumbering_keeps_the_values_with_their_terms():
    code, w2, removed = drop_weight(RULE, [1.0, 2.0, 3.0, 4.0], 0)
    assert removed == [0]
    assert w2 == [2.0, 3.0, 4.0]
    assert "s = s + f.b * w[0]" in code
    assert "np.power(f.c, w[1]) * w[2]" in code


def test_the_comment_of_a_deleted_statement_goes_with_it():
    code, _w, _r = drop_weight(RULE, [1.0, 2.0, 3.0, 4.0], 1)
    assert "the second term" not in code
    assert "the first term" in code


def test_a_comment_heading_a_block_stays():
    code = ("def score(f, p, hw, w):\n"
            "    # two memory terms\n"
            "    s = f.a * w[0]\n"
            "    s = s + f.b * w[1]\n"
            "    return s\n")
    out, _w, _r = drop_weight(code, [1.0, 1.0], 0)
    assert "# two memory terms" in out


def test_a_root_term_becomes_zero_and_an_emptied_block_gets_pass():
    code = ("def score(f, p, hw, w):\n"
            "    s = f.a * w[0]\n"
            "    if p.x < 2:\n"
            "        s = s + f.b * w[1]\n"
            "    return s\n")
    out, w2, _r = drop_weight(code, [1.0, 2.0], 1)
    assert "pass" in out
    assert np.allclose(_run(out, w2, F), _run(code, [1.0, 0.0], F))
    out0, w0, _r = drop_weight(code, [1.0, 2.0], 0)
    assert "s = 0.0" in out0
    assert np.allclose(_run(out0, w0, F), _run(code, [0.0, 2.0], F))


def test_a_shared_line_is_not_touched():
    code = ("def score(f, p, hw, w):\n"
            "    s = f.a * w[0]; s = s + f.b * w[1]\n"
            "    return s\n")
    assert drop_weight(code, [1.0, 1.0], 1) is None


def test_a_trailing_comment_stays_on_a_rewritten_line():
    code = ("def score(f, p, hw, w):\n"
            "    s = f.a * w[0] + f.b * w[1]  # both\n"
            "    return s\n")
    out, _w, _r = drop_weight(code, [1.0, 1.0], 1)
    assert "s = f.a * w[0]  # both" in out


def test_prune_keeps_a_removal_only_when_train_regret_does_not_get_worse():
    calls = []

    def regret_of(code, w):
        calls.append(len(w))
        return 1.10 if "f.b" in code else 1.20       # removing w[1] hurts

    code, w2, log = prune_dead(RULE, [1.0, 2.0, 3.0, 4.0], [3, 1, 0],
                               regret_of, base=1.10)
    kept = [x["i"] for x in log if x["kept"]]
    assert kept == [3, 0]
    assert [x.get("why") for x in log if not x["kept"]] == ["worse"]
    assert "f.b" in code and len(w2) == 1
