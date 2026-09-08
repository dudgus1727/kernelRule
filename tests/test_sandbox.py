"""Isolated execution (§15.3). **The second line of defence for what got
through the static checks.**"""
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
    ("infinite loop", "def score(f, p, hw, w):\n    while True:\n        pass\n",
     "timeout"),
    ("import", """def score(f, p, hw, w):
    import os
    os.system('echo PWNED')
    return f.waves * w[0]
""",
     "error"),
    ("open a file", ("def score(f, p, hw, w):\n"
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
    ("typo crash", ("def score(f, p, hw, w):\n"
                    "    return f.nope * w[0]\n"), "error"),
]


@pytest.mark.parametrize("name,code,kind", ESCAPES,
                         ids=[c[0] for c in ESCAPES])
def test_escape_attempt_is_contained(name, code, kind, args):
    r = run_isolated(code, args, timeout=6.0)
    assert not r.ok, f"{name} passed"
    if kind == "timeout":
        assert r.timed_out, "the infinite loop was not treated as a timeout"


def test_timeout_is_a_failure_not_a_slow_pass(args):
    """★ A timeout is a **failure**, not "slow but passing" (§26.4)."""
    code = "def score(f, p, hw, w):\n    while True:\n        pass\n"
    r = run_isolated(code, args, timeout=2.0)
    assert r.timed_out and not r.ok


def test_safe_namespace_has_no_import_or_open():
    ns = safe_namespace()
    b = ns["__builtins__"]
    for name in ("__import__", "open", "eval", "exec", "compile", "getattr"):
        assert name not in b, f"{name} is exposed"


def test_np_proxy_blocks_nondeterminism():
    ns = safe_namespace()
    with pytest.raises(AttributeError, match="cannot be used in a rule"):
        _ = ns["np"].random
    assert callable(ns["np"].where)


def test_compile_rule_rejects_missing_function():
    with pytest.raises(SandboxError, match="score"):
        compile_rule("x = 1")


def test_compile_rule_rejects_syntax_error():
    with pytest.raises(SandboxError, match="compilation failed"):
        compile_rule("def score(:")


def test_feats_survive_pickle():
    """★ For the sandbox to hand this to a child process, pickle has to
    work.

    The `__slots__` + forbidden `__setattr__` combination once made the
    default path fall into infinite recursion. Pinned here as a regression.
    """
    import pickle
    f = Feats({"waves": np.arange(3.0)})
    g = pickle.loads(pickle.dumps(f))
    assert np.allclose(g.waves, np.arange(3.0))
    with pytest.raises(AttributeError, match="unregistered"):
        _ = g.nope


# ---------------------------------------------------------------------------
# ★ A numpy warning must not kill the rule (D-135)
# ---------------------------------------------------------------------------
def test_numpy_warning_does_not_kill_the_rule():
    """★ `log(negative)` must not raise `KeyError: '__import__'`.

    numpy imports `warnings` in order to raise one, and the restricted
    builtins have no `__import__`. **34 of 1,728 proposals (2.0%)** across
    the six reference runs were thrown away by this — not because the rules
    were bad but because of a defect of ours.

    What happens after a non-finite value is already settled (the fitter
    gives inf, `top_k` refuses).
    """
    import numpy as np

    from kernelrule.core.sandbox import compile_rule

    fn = compile_rule("def score(f, p, hw, w):\n"
                      "    return np.log(f.waves - 5.0) * w[0]\n")

    class F:
        waves = np.array([1.0, 2.0, 3.0])

    out = fn(F(), None, None, np.array([1.0]))
    assert np.isnan(out).all(), out          # nan comes out; it does not die


def test_the_same_guard_is_on_both_paths():
    """★ Is the same defence on **both** paths (principle 2)?

    It used to be on the sandbox child only. With it on one side only, which
    way you call changes the result — that was D-135.
    """
    import inspect

    from kernelrule.core import sandbox

    assert "errstate" in inspect.getsource(sandbox.compile_rule)
    assert "seterr" in inspect.getsource(sandbox._child)


def test_errstate_does_not_leak_globally():
    """★ It does not change the global `np.seterr` — that would silently
    change other computations too."""
    import numpy as np

    from kernelrule.core.sandbox import compile_rule

    before = np.geterr()
    fn = compile_rule("def score(f, p, hw, w):\n    return f.waves * w[0]\n")

    class F:
        waves = np.array([1.0])

    fn(F(), None, None, np.array([1.0]))
    assert np.geterr() == before
