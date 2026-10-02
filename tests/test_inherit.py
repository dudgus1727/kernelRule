"""Which weights of a child are its parent's (D-190 §6). Matched by the
statement a weight sits in — not by its index, not by the LLM's numbers."""
from __future__ import annotations

from kernelrule.rules.inherit import inherit_weights

PARENT = ("def score(f, p, hw, w):\n"
          "    s = f.a * w[0]\n"
          "    if p.x < 1:\n"
          "        s = s + f.b * w[1]\n"
          "    else:\n"
          "        s = s + f.b * w[2]\n"
          "    return s\n")
PW = [10.0, 200.0, -3.0]


def test_renumbered_terms_still_inherit_and_new_terms_are_marked():
    child = ("def score(f, p, hw, w):\n"
             "    # a new first term\n"
             "    s = f.c * w[0]\n"
             "    s = s + f.a * w[1]\n"
             "    if p.x < 1:\n"
             "        s = s + f.b * w[2]\n"
             "    else:\n"
             "        s = s + f.b * w[3]\n"
             "    return s\n")
    start, new, src = inherit_weights(child, [1.0, 1.0, 1.0, 1.0],
                                      [(PARENT, PW)])
    # `s = f.c * w[0]` is new; the old first term is now `s = s + ...`,
    # a different statement, so it is new too.
    assert new == [True, True, False, False]
    assert start[2:] == [200.0, -3.0]
    assert src[2:] == [1, 2]


def test_a_term_under_another_branch_is_not_the_same_term():
    child = ("def score(f, p, hw, w):\n"
             "    s = f.a * w[0]\n"
             "    if p.x < 2:\n"
             "        s = s + f.b * w[1]\n"
             "    else:\n"
             "        s = s + f.b * w[2]\n"
             "    return s\n")
    start, new, _src = inherit_weights(child, [1.0, 1.0, 1.0],
                                       [(PARENT, PW)])
    assert new == [False, True, True]
    assert start[0] == 10.0


def test_the_second_parent_fills_what_the_first_lacks():
    other = ("def score(f, p, hw, w):\n"
             "    s = f.d * w[0]\n"
             "    return s\n")
    child = ("def score(f, p, hw, w):\n"
             "    s = f.a * w[0]\n"
             "    s = s + f.d * w[1]\n"
             "    return s\n")
    # `s = s + f.d * w[1]` is not the other parent's `s = f.d * w[0]`
    start, new, _src = inherit_weights(child, [1.0, 1.0],
                                       [(PARENT, PW), (other, [7.0])])
    assert new == [False, True] and start[0] == 10.0
    child2 = ("def score(f, p, hw, w):\n"
              "    s = f.d * w[0]\n"
              "    return s\n")
    start2, new2, _ = inherit_weights(child2, [1.0],
                                      [(PARENT, PW), (other, [7.0])])
    assert new2 == [False] and start2 == [7.0]


def test_comments_do_not_matter_and_bad_code_inherits_nothing():
    child = PARENT.replace("    s = f.a * w[0]\n",
                           "    # comment\n    s = f.a * w[0]  # x\n")
    start, new, _ = inherit_weights(child, [0.0, 0.0, 0.0], [(PARENT, PW)])
    assert new == [False, False, False] and start == PW
    s2, n2, _ = inherit_weights("def (", [1.0], [(PARENT, PW)])
    assert n2 == [True] and s2 == [1.0]
