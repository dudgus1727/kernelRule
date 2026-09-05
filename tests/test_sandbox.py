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


# ---------------------------------------------------------------------------
# ★ numpy 경고가 규칙을 죽이면 안 된다 (D-135)
# ---------------------------------------------------------------------------
def test_numpy_warning_does_not_kill_the_rule():
    """★ `log(음수)` 가 `KeyError: '__import__'` 를 내면 안 된다.

    numpy 는 경고를 내려고 `warnings` 를 import 하는데 제한된 builtins 에
    `__import__` 가 없다. 대표값 6실행에서 **제안 1,728개 중 34개(2.0%)**
    가 이것으로 버려졌다 — 규칙이 나빠서가 아니라 우리 결함으로.

    비유한 값이 나온 뒤는 이미 정해져 있다 (적합기는 inf, `top_k` 는 거부).
    """
    import numpy as np

    from kernelrule.core.sandbox import compile_rule

    fn = compile_rule("def score(f, p, hw, w):\n"
                      "    return np.log(f.waves - 5.0) * w[0]\n")

    class F:
        waves = np.array([1.0, 2.0, 3.0])

    out = fn(F(), None, None, np.array([1.0]))
    assert np.isnan(out).all(), out          # 죽지 않고 nan 이 나온다


def test_the_same_guard_is_on_both_paths():
    """★ 같은 방어가 **양쪽에** 있는가 (원칙 2).

    전에는 샌드박스 자식에만 있었다. 한쪽에만 있으면 어느 쪽으로 부르느냐가
    결과를 바꾼다 — 그것이 D-135 였다.
    """
    import inspect

    from kernelrule.core import sandbox

    assert "errstate" in inspect.getsource(sandbox.compile_rule)
    assert "seterr" in inspect.getsource(sandbox._child)


def test_errstate_does_not_leak_globally():
    """★ 전역 `np.seterr` 를 바꾸지 않는다 — 다른 계산까지 조용히 달라진다."""
    import numpy as np

    from kernelrule.core.sandbox import compile_rule

    before = np.geterr()
    fn = compile_rule("def score(f, p, hw, w):\n    return f.waves * w[0]\n")

    class F:
        waves = np.array([1.0])

    fn(F(), None, None, np.array([1.0]))
    assert np.geterr() == before
