"""Isolated execution of LLM-generated code (§15.3).

Plain `exec` will not do. The LLM **really does** produce infinite loops.

## Two layers

    1. static checks (`rules/checks.py`)   filtered by AST **before** running
    2. this file                           what still got through is run in
                                           isolation

The static checks are the first line of defence and this is the second. The
order must not be swapped — handing code that does not even parse to a
separate process only costs.

## What it blocks

    infinite loop     a separate process + timeout -> killed
    memory blowup     RLIMIT_AS
    files/network     restricted builtins + an import hook
    crashes           a separate process, so the parent does not die

## Why compilation and execution are separated

`compile_rule()` does an `exec` **in the parent process** with a restricted
namespace. Code that passed the static checks can neither import nor touch
files, so it is safe here, and the scoring loop need not cross a process
boundary every time (12 rules x 66 shapes per round).

`run_isolated()` is for trial-running **code seen for the first time**.
Infinite loops and crashes are filtered out here before handing it to
`compile_rule()`.
"""

from __future__ import annotations

import functools
import multiprocessing as mp
import queue as _queue
import resource
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["SandboxError", "SandboxResult", "compile_rule", "run_isolated",
           "safe_namespace"]

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MEM_MB = 2048

#: The `np` given to a rule is **not the whole module.** `np.random` makes
#: things non-deterministic and `np.load` opens files. The static checks
#: already filter the names, but it is better for them to be absent at
#: runtime too — the two layers must fail in the same direction.
_NP_ALLOWED = (
    "where", "clip", "minimum", "maximum", "log", "log2", "log10", "sqrt",
    "abs", "exp", "power", "sign", "floor", "ceil", "round", "isfinite",
    "nan_to_num", "square", "reciprocal", "logical_and", "logical_or",
    "logical_not", "greater", "less", "equal", "asarray", "zeros_like",
    "ones_like", "full_like", "fmin", "fmax", "hypot", "cbrt",
    "float64", "inf", "pi", "e",
)

_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "float": float, "int": int, "bool": bool, "round": round,
    "range": range, "enumerate": enumerate, "zip": zip, "sorted": sorted,
    "True": True, "False": False, "None": None,
}


class SandboxError(RuntimeError):
    """The isolated run failed. **The rule is discarded.**"""


@dataclass
class SandboxResult:
    ok: bool
    value: Any = None
    error: str = ""
    seconds: float = 0.0
    timed_out: bool = False

    def __str__(self) -> str:
        if self.timed_out:
            return f"[timed out] {self.seconds:.1f}s"
        return f"[{'ok' if self.ok else 'failed'}] {self.error[:200]}"


class _NpProxy:
    """A numpy stand-in exposing only the allowed functions. Everything
    else is an `AttributeError`."""

    __slots__ = ()

    def __getattr__(self, name: str):
        if name not in _NP_ALLOWED:
            raise AttributeError(
                f"np.{name} cannot be used in a rule. "
                f"allowed: {', '.join(sorted(_NP_ALLOWED)[:10])} ...")
        return getattr(np, name)


def safe_namespace() -> dict:
    """The globals a rule sees. **No import, no file access.**"""
    return {"__builtins__": dict(_BUILTINS), "np": _NpProxy()}


def compile_rule(code: str, *, name: str = "score"):
    """`exec`s in a restricted namespace and pulls the function out.

    ⚠️ **Pass the static checks first.** This function does not look at the
    AST.

    ## ★ It is called with numpy warnings off (D-135)

    numpy imports `warnings` in order to raise one, and the restricted
    builtins have no `__import__`, so it **dies with
    `KeyError: '__import__'`.** A rule that produced `log(negative)`, a
    division by zero, or an overflow even once was thrown away without even
    getting a score — **34 of 1,728 proposals (2.0%)** across the six
    reference runs.

    ```
    sandbox child      np.seterr(all="ignore") **was there** (below)
    scoring/fit path   ★ was not   -> the same defence on one side only
                       (principle 2)
    ```

    ★ Instead of a global `np.seterr`, **a per-call `np.errstate`** is used
    — changing global state would silently change other computations in this
    process too.

    What happens after a non-finite value **is already settled**: the fitter
    treats those weights as unrunnable (`inf`, not a structural rejection),
    and scoring's `top_k` refuses that rule with a `ValueError`.
    """
    ns = safe_namespace()
    try:
        exec(compile(code, "<rule>", "exec"), ns)      # noqa: S102
    except Exception as e:                             # noqa: BLE001
        raise SandboxError(
            f"rule compilation failed: {type(e).__name__}: {e}") from e
    fn = ns.get(name)
    if not callable(fn):
        raise SandboxError(f"the rule has no `{name}` function")

    @functools.wraps(fn)
    def guarded(*a, **kw):
        with np.errstate(all="ignore"):
            return fn(*a, **kw)

    return guarded


def _child(code: str, name: str, args_pickle: bytes, mem_mb: int, q) -> None:
    """The child process. Sets resource limits and runs the rule once."""
    try:
        soft = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
        # ⚠️ RLIMIT_NPROC must not be set to 0 — on Linux **threads count
        #    as processes too.** The feeder thread of
        #    `multiprocessing.Queue` cannot start, so no result comes back
        #    and every case looks like a "timeout". (We stepped on this.
        #    Even correct code came out as a timeout.) Process creation is
        #    blocked by not being able to import `os`/`subprocess`.
    except (ValueError, OSError):        # pragma: no cover - platform diff
        pass
    try:
        import pickle
        # numpy's warning path imports `warnings`, and the restricted
        # builtins have no `__import__`. Then `np.log(-1)` never reaches the
        # nan check and dies with `KeyError: __import__`, **hiding the
        # cause.** Warnings are turned off — nan/inf are caught explicitly
        # below anyway.
        np.seterr(all="ignore")
        fn = compile_rule(code, name=name)
        args = pickle.loads(args_pickle)     # noqa: S301 - we made it
        out = fn(*args)
        arr = np.asarray(out, dtype=np.float64)
        q.put(("ok", (arr.shape, arr.tobytes(),
                      bool(np.all(np.isfinite(arr))))))
    except BaseException as e:                          # noqa: BLE001
        q.put(("err", f"{type(e).__name__}: {e}"))


def _context():
    """How the child process is started.

    `fork` inherits the whole parent state, so it is not isolation.
    `spawn` makes **the child re-import `__main__`** — if the calling script
    is not wrapped in `if __name__ == "__main__":` it breaks with infinite
    recursion. That guard cannot be demanded of every script that runs the
    loop (we stepped on this).

    `forkserver` forks from a clean server process, so it does not inherit
    the parent state. On Linux this is the right one.

    ⚠️ **Both `forkserver` and `spawn` make the child re-import `__main__`**
    (`multiprocessing.spawn.get_preparation_data`). If the calling script is
    not wrapped in `if __name__ == "__main__":` it breaks with infinite
    recursion, and Python shows that as a `BrokenPipeError` — the cause is
    completely invisible. `_preflight()` below turns it into **a readable
    error**.
    """
    try:
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload(["numpy", "kernelrule.core.sandbox"])
        return ctx
    except (ValueError, AttributeError, RuntimeError):   # pragma: no cover
        return mp.get_context("spawn")


def _preflight() -> None:
    """Checks **once** whether a child process can be started.

    On failure it says what to fix instead of a `BrokenPipeError`.
    """
    if getattr(_preflight, "done", False):
        return
    ctx = _context()
    q = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_ping, args=(q,))
    try:
        proc.start()
    except (BrokenPipeError, OSError, RuntimeError) as e:
        raise SandboxError(
            f"cannot start the sandbox child process: "
            f"{type(e).__name__}: {e}\n"
            "  The most common cause: the calling script has no\n"
            "  `if __name__ == \"__main__\":` guard. multiprocessing's\n"
            "  forkserver/spawn make the child re-import `__main__`, so\n"
            "  without the guard it is infinite recursion.\n"
            "\n"
            "  How to fix it:\n"
            "      def main():\n"
            "          ...\n"
            "      if __name__ == \"__main__\":\n"
            "          main()\n"
            "\n"
            "  Turning the sandbox off is not the answer — the LLM really\n"
            "  does produce infinite loops (§15.3)."
        ) from e
    try:
        proc.join(20.0)
    finally:
        if proc.is_alive():                          # pragma: no cover
            proc.kill()
            proc.join(1.0)
    _preflight.done = True


def _ping(q) -> None:                                # pragma: no cover
    """The preflight child. Even on failure the parent judges it a timeout,
    so it may be swallowed."""
    import contextlib

    with contextlib.suppress(Exception):
        q.put("ok")


def run_isolated(code: str, args: tuple, *, name: str = "score",
                 timeout: float = DEFAULT_TIMEOUT_S,
                 mem_mb: int = DEFAULT_MEM_MB) -> SandboxResult:
    """Runs once in a separate process. Infinite loops and crashes are
    caught here.

    ⚠️ A timeout is a **failure**. Not "slow but passing" (§26.4).
    """
    import pickle
    import time

    _preflight()
    ctx = _context()
    q = ctx.Queue(maxsize=1)
    pickled = pickle.dumps(args)
    proc = ctx.Process(target=_child, args=(code, name, pickled, mem_mb, q))
    t0 = time.perf_counter()
    proc.start()
    try:
        kind, payload = q.get(timeout=timeout)
    except _queue.Empty:
        proc.terminate()
        proc.join(1.0)
        if proc.is_alive():                             # pragma: no cover
            proc.kill()
            proc.join(1.0)
        return SandboxResult(ok=False, timed_out=True,
                             seconds=time.perf_counter() - t0,
                             error=f"did not finish within {timeout}s. "
                                   f"Discarded")
    finally:
        if proc.is_alive():
            proc.join(1.0)
        if proc.is_alive():                             # pragma: no cover
            proc.kill()

    dt = time.perf_counter() - t0
    if kind == "err":
        return SandboxResult(ok=False, error=payload, seconds=dt)
    shape, raw, finite = payload
    arr = np.frombuffer(raw, dtype=np.float64).reshape(shape)
    if not finite:
        return SandboxResult(ok=False, seconds=dt,
                             error="the scores contain nan/inf. Discarded")
    return SandboxResult(ok=True, value=arr, seconds=dt)
