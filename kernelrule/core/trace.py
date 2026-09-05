"""★ 실행 트레이스 — **한 파일에 시간순으로** (D-133).

지금 기록이 여섯 군데에 흩어져 있다 (`llm_calls/` · `rounds.jsonl` ·
`archive.jsonl` · `bests.jsonl` · `failures.jsonl` · `hypotheses.jsonl`).
각자 맞으나 **"무엇 다음에 무엇이 왜"** 가 사후에 시각·id 로 맞춰야만
보인다.

`trace.jsonl` 은 한 줄이 한 사건이고 **덧붙이기만** 한다.

## ★ 부수 효과가 없어야 한다

```
꺼져 있으면    아무것도 안 한다 (`ev` 가 즉시 돌아온다)
켜져 있으면    파일에 한 줄 쓰고 flush 한다. **계산 경로를 안 건드린다**
★ 확인        MockLLM 으로 켜고/끄고 돌려 산출물이 **같은지** 본다
              (`tests/test_trace.py`)
```

## 왜 flush 하나

중간에 죽어도 **거기까지는 남아야** 한다 — D-33 이 78분 1400호출을
잃은 자리다. 한 줄이 수 KB 이고 라운드당 수십 줄이라 비용이 무시할
만하다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

__all__ = ["Tracer"]


class Tracer:
    """`trace.jsonl` 에 사건을 덧붙인다. `path=None` 이면 **아무것도 안 한다**."""

    __slots__ = ("_f", "n", "_seen_calls")

    def __init__(self, path: str | Path | None = None) -> None:
        self._f = None
        self.n = 0
        #: 이미 트레이스에 넣은 LLM 호출 수 (`llm.calls` 의 앞에서부터).
        self._seen_calls = 0
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._f = p.open("a", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._f is not None

    def ev(self, ev: str, **kw) -> None:
        """사건 하나. ★ 꺼져 있으면 인자를 만지지도 않는다."""
        if self._f is None:
            return
        rec = {"ev": ev, "t": round(time.time(), 3)}
        rec.update(kw)
        self._f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._f.flush()
        self.n += 1

    def llm_calls(self, llm, **kw) -> None:
        """`llm.calls` 에 새로 쌓인 것을 전문으로 옮긴다.

        ★ LLM 쪽 코드를 **안 건드린다** — 이미 있는 목록을 읽기만 한다.
        프롬프트·응답 전문을 담는다 (24라운드 실행이 2.7 MB 다 — 실측).
        """
        if self._f is None:
            return
        calls = getattr(llm, "calls", None) or []
        for c in calls[self._seen_calls:]:
            meta = getattr(c, "__dict__", {}).get("_meta", {}) or {}
            self.ev("llm_call", role=c.role, seq=c.seq,
                    prompt_hash=c.prompt_hash,
                    user_prompt=meta.get("prompt", ""),
                    response=c.response,
                    n_in=meta.get("input_tokens"),
                    n_out=meta.get("output_tokens"),
                    ms=(None if meta.get("seconds") is None
                        else round(meta["seconds"] * 1000)),
                    **kw)
        self._seen_calls = len(calls)

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
