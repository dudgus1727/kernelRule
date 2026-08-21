"""격리 실행 (§15.3). **정적 검사를 뚫었을 때의 2차 방어.**"""
from __future__ import annotations

import numpy as np
import pytest

from kernelrule.core.matrix import Feats, ShapeInfo
from kernelrule.core.sandbox import (
    SandboxError,
    compile_rule,
    run_isolated,
    safe_namespace,
)


@pytest.fixture
def args():
    return (Feats({"waves": np.arange(5.0), "has_spill": np.zeros(5)}),
            ShapeInfo({"is_memory_bound": 0.0}), None, np.array([1.0]))


GOOD = "def score(f, p, hw, w):\n    return f.waves * w[0]\n"


def test_good_rule_runs(args):
    r = run_isolated(GOOD, args, timeout=10.0)
    assert r.ok
    assert np.allclose(r.value, np.arange(5.0))


ESCAPES = [
    ("무한 루프", "def score(f, p, hw, w):\n    while True:\n        pass\n",
     "timeout"),
    ("import", """def score(f, p, hw, w):
    import os
    os.system('echo PWNED')
    return f.waves * w[0]
""",
     "error"),
    ("파일 열기", ("def score(f, p, hw, w):\n"
                  "    open('/etc/passwd').read()\n    return f.waves * w[0]\n"),
     "error"),
    ("eval", """def score(f, p, hw, w):
    return eval('1') * f.waves * w[0]
""", "error"),
    ("np.random", "def score(f, p, hw, w):\n    return np.random.rand(5)\n",
     "error"),
    ("nan", """def score(f, p, hw, w):
    return f.waves * w[0] + np.log(-1.0)
""", "error"),
    ("inf", "def score(f, p, hw, w):\n    return f.waves * w[0] / 0.0\n",
     "error"),
    ("오타 크래시", ("def score(f, p, hw, w):\n"
                    "    return f.nope * w[0]\n"), "error"),
]


@pytest.mark.parametrize("name,code,kind", ESCAPES,
                         ids=[c[0] for c in ESCAPES])
def test_escape_attempt_is_contained(name, code, kind, args):
    r = run_isolated(code, args, timeout=6.0)
    assert not r.ok, f"{name} 가 통과했다"
    if kind == "timeout":
        assert r.timed_out, "무한 루프가 타임아웃으로 처리되지 않았다"


def test_timeout_is_a_failure_not_a_slow_pass(args):
    """★ 타임아웃은 **실패**다. '느리지만 통과' 가 아니다 (§26.4)."""
    code = "def score(f, p, hw, w):\n    while True:\n        pass\n"
    r = run_isolated(code, args, timeout=2.0)
    assert r.timed_out and not r.ok


def test_safe_namespace_has_no_import_or_open():
    ns = safe_namespace()
    b = ns["__builtins__"]
    for name in ("__import__", "open", "eval", "exec", "compile", "getattr"):
        assert name not in b, f"{name} 가 노출돼 있다"


def test_np_proxy_blocks_nondeterminism():
    ns = safe_namespace()
    with pytest.raises(AttributeError, match="쓸 수 없다"):
        _ = ns["np"].random
    assert callable(ns["np"].where)


def test_compile_rule_rejects_missing_function():
    with pytest.raises(SandboxError, match="score"):
        compile_rule("x = 1")


def test_compile_rule_rejects_syntax_error():
    with pytest.raises(SandboxError, match="컴파일 실패"):
        compile_rule("def score(:")


def test_feats_survive_pickle():
    """★ 샌드박스가 자식 프로세스로 넘기려면 pickle 이 돌아야 한다.

    `__slots__` + `__setattr__` 금지 조합이라 기본 경로가 무한 재귀에
    빠졌었다. 회귀로 고정한다.
    """
    import pickle
    f = Feats({"waves": np.arange(3.0)})
    g = pickle.loads(pickle.dumps(f))
    assert np.allclose(g.waves, np.arange(3.0))
    with pytest.raises(AttributeError, match="등록되지 않은"):
        _ = g.nope
