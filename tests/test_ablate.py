"""항 절제 (D-85). Critic 의 정성 판정을 정량으로 검증하는 도구."""
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
    assert term_indices(out) == [0, 1], "가중치를 다시 번호 매기지 않았다"


def test_drop_first_term_keeps_s_defined():
    """★ 첫 항을 지우면 `s = s + …` 가 첫 대입이 된다 — `s` 가 없다."""
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
    """★ 조용히 건너뛰지 않는다 — 못 지운 것과 '지웠는데 영향 없음' 은 다르다."""
    code = "def score(f, p, hw, w):\n    return f.a * w[0] + f.b * w[1]\n"
    with pytest.raises(AblateError, match="한 줄에 여러 항"):
        drop_terms(code, {1})


def test_refuses_when_branch_would_be_empty():
    code = ("def score(f, p, hw, w):\n"
            "    s = f.a * w[0]\n"
            "    if p.is_memory_bound:\n"
            "        s = s + f.b * w[1]\n"
            "    return s\n")
    with pytest.raises(AblateError, match="분기 안이 비었다"):
        drop_terms(code, {1})


def test_refuses_to_drop_everything():
    code = "def score(f, p, hw, w):\n    return f.a * w[0]\n"
    with pytest.raises(AblateError, match="다 지우면"):
        drop_terms(code, {0})


def test_dropping_nothing_is_identity_modulo_renumber():
    out = drop_terms(SIMPLE, set())
    assert term_indices(out) == [0, 1, 2]
    assert "f.a" in out and "f.b" in out and "f.c" in out
