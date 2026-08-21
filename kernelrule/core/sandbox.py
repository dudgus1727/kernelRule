"""LLM 생성 코드의 격리 실행 (§15.3).

`exec` 를 그대로 쓰면 안 된다. LLM 은 무한 루프를 **실제로** 만들어낸다.

## 두 겹

    1. 정적 검사 (`rules/checks.py`)   실행 **전에** AST 로 거른다
    2. 이 파일                          그래도 통과한 것을 격리해서 돌린다

정적 검사가 1차 방어이고 여기는 2차다. 순서를 바꾸면 안 된다 — 파싱조차
안 되는 코드를 별도 프로세스에 넘기는 것은 비용만 든다.

## 무엇을 막는가

    무한 루프      별도 프로세스 + 타임아웃 -> 죽인다
    메모리 폭주    RLIMIT_AS
    파일/네트워크   builtins 제한 + import 훅
    크래시         별도 프로세스라 부모가 안 죽는다

## 왜 컴파일과 실행을 나누는가

`compile_rule()` 은 **부모 프로세스에서** 제한된 네임스페이스로 `exec` 한다.
정적 검사를 통과한 코드는 import 도 파일 접근도 못 하므로 여기서는 안전하고,
채점 루프가 프로세스 경계를 매번 넘지 않아도 된다 (라운드당 12규칙 x 66형상).

`run_isolated()` 는 **처음 보는 코드**를 시험 실행할 때 쓴다. 무한 루프와
크래시를 여기서 걸러낸 뒤 `compile_rule()` 로 넘긴다.
"""

from __future__ import annotations

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

#: 규칙에 주는 `np` 는 **모듈 전체가 아니다.** `np.random` 이 비결정론을
#: 만들고 `np.load` 가 파일을 연다. 정적 검사가 이미 이름을 거르지만
#: 실행 시점에도 없는 편이 낫다 — 두 겹이 같은 방향으로 실패해야 한다.
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
    """격리 실행이 실패했다. **규칙을 폐기한다.**"""


@dataclass
class SandboxResult:
    ok: bool
    value: Any = None
    error: str = ""
    seconds: float = 0.0
    timed_out: bool = False

    def __str__(self) -> str:
        if self.timed_out:
            return f"[시간 초과] {self.seconds:.1f}s"
        return f"[{'성공' if self.ok else '실패'}] {self.error[:200]}"


class _NpProxy:
    """허용된 함수만 노출하는 numpy 대역. 나머지는 `AttributeError`."""

    __slots__ = ()

    def __getattr__(self, name: str):
        if name not in _NP_ALLOWED:
            raise AttributeError(
                f"np.{name} 는 규칙에서 쓸 수 없다. "
                f"허용: {', '.join(sorted(_NP_ALLOWED)[:10])} ...")
        return getattr(np, name)


def safe_namespace() -> dict:
    """규칙이 보는 전역. **import 도 파일 접근도 없다.**"""
    return {"__builtins__": dict(_BUILTINS), "np": _NpProxy()}


def compile_rule(code: str, *, name: str = "score"):
    """제한된 네임스페이스에서 `exec` 하고 함수를 꺼낸다.

    ⚠️ **정적 검사를 먼저 통과시켜라.** 이 함수는 AST 를 안 본다.
    """
    ns = safe_namespace()
    try:
        exec(compile(code, "<rule>", "exec"), ns)      # noqa: S102
    except Exception as e:                             # noqa: BLE001
        raise SandboxError(f"규칙 컴파일 실패: {type(e).__name__}: {e}") from e
    fn = ns.get(name)
    if not callable(fn):
        raise SandboxError(f"규칙에 `{name}` 함수가 없다")
    return fn


def _child(code: str, name: str, args_pickle: bytes, mem_mb: int, q) -> None:
    """자식 프로세스. 자원 한도를 걸고 규칙을 한 번 실행한다."""
    try:
        soft = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
        # ⚠️ RLIMIT_NPROC 을 0 으로 두면 안 된다 — 리눅스에서 **스레드도
        #    프로세스로 센다.** `multiprocessing.Queue` 의 피더 스레드가 못
        #    떠서 결과를 못 돌려주고, 모든 케이스가 "시간 초과" 로 보인다.
        #    (실제로 밟았다. 정상 코드까지 타임아웃으로 나왔다.)
        #    프로세스 생성은 `os`/`subprocess` 를 못 import 하는 것으로 막힌다.
    except (ValueError, OSError):        # pragma: no cover - 플랫폼 차이
        pass
    try:
        import pickle
        # numpy 의 경고 경로는 `warnings` 를 import 하는데 제한된 builtins 에는
        # `__import__` 가 없다. 그러면 `np.log(-1)` 이 nan 검사에 도달하지 못하고
        # `KeyError: __import__` 로 죽어서 **원인이 가려진다.** 경고를 끈다 —
        # 어차피 nan/inf 는 아래에서 명시적으로 잡는다.
        np.seterr(all="ignore")
        fn = compile_rule(code, name=name)
        args = pickle.loads(args_pickle)     # noqa: S301 - 우리가 만든 것
        out = fn(*args)
        arr = np.asarray(out, dtype=np.float64)
        q.put(("ok", (arr.shape, arr.tobytes(),
                      bool(np.all(np.isfinite(arr))))))
    except BaseException as e:                          # noqa: BLE001
        q.put(("err", f"{type(e).__name__}: {e}"))


def _context():
    """자식 프로세스 시작 방식.

    `fork` 는 부모 상태를 통째로 물려받아 격리가 아니다.
    `spawn` 은 **자식이 `__main__` 을 다시 import 한다** — 호출 스크립트가
    `if __name__ == "__main__":` 로 감싸여 있지 않으면 무한 재귀로 깨진다.
    루프를 돌리는 스크립트마다 그 가드를 요구할 수는 없다 (실제로 밟았다).

    `forkserver` 는 깨끗한 서버 프로세스에서 fork 하므로 부모 상태를 안
    물려받는다. 리눅스에서는 이쪽이 맞다.

    ⚠️ **`forkserver` 도 `spawn` 도 자식이 `__main__` 을 다시 import 한다**
    (`multiprocessing.spawn.get_preparation_data`). 호출 스크립트가
    `if __name__ == "__main__":` 로 감싸여 있지 않으면 무한 재귀로 깨지고,
    파이썬은 그것을 `BrokenPipeError` 로 보여준다 — 원인이 전혀 안 보인다.
    아래 `_preflight()` 가 그것을 **읽을 수 있는 에러**로 바꾼다.
    """
    try:
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload(["numpy", "kernelrule.core.sandbox"])
        return ctx
    except (ValueError, AttributeError, RuntimeError):   # pragma: no cover
        return mp.get_context("spawn")


_PREFLIGHT: bool | None = None


def _preflight() -> None:
    """자식 프로세스를 띄울 수 있는지 **한 번** 확인한다.

    실패하면 `BrokenPipeError` 대신 무엇을 고쳐야 하는지 말한다.
    """
    global _PREFLIGHT
    if _PREFLIGHT:
        return
    ctx = _context()
    q = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_ping, args=(q,))
    try:
        proc.start()
    except (BrokenPipeError, OSError, RuntimeError) as e:
        raise SandboxError(
            f"샌드박스 자식 프로세스를 띄울 수 없다: {type(e).__name__}: {e}\n"
            "  가장 흔한 원인: 호출 스크립트에 `if __name__ == \"__main__\":`\n"
            "  가드가 없다. multiprocessing 의 forkserver/spawn 은 자식이\n"
            "  `__main__` 을 다시 import 하므로 가드가 없으면 무한 재귀다.\n"
            "\n"
            "  고치는 법:\n"
            "      def main():\n"
            "          ...\n"
            "      if __name__ == \"__main__\":\n"
            "          main()\n"
            "\n"
            "  샌드박스를 끄는 것은 답이 아니다 — LLM 은 무한 루프를 실제로\n"
            "  만들어낸다 (§15.3)."
        ) from e
    try:
        proc.join(20.0)
    finally:
        if proc.is_alive():                          # pragma: no cover
            proc.kill()
            proc.join(1.0)
    _PREFLIGHT = True


def _ping(q) -> None:                                # pragma: no cover
    try:
        q.put("ok")
    except Exception:                                # noqa: BLE001, S110
        pass


def run_isolated(code: str, args: tuple, *, name: str = "score",
                 timeout: float = DEFAULT_TIMEOUT_S,
                 mem_mb: int = DEFAULT_MEM_MB) -> SandboxResult:
    """별도 프로세스에서 한 번 실행한다. 무한 루프와 크래시를 여기서 잡는다.

    ⚠️ 타임아웃이면 **실패**다. "느리지만 통과" 가 아니다 (§26.4).
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
                             error=f"{timeout}s 안에 끝나지 않았다. 폐기한다")
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
                             error="점수에 nan/inf 가 있다. 폐기한다")
    return SandboxResult(ok=True, value=arr, seconds=dt)
