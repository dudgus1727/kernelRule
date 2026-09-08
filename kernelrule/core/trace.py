"""★ Execution trace — **one file, in time order** (D-133).

The record is currently scattered across six places (`llm_calls/` ·
`rounds.jsonl` · `archive.jsonl` · `bests.jsonl` · `failures.jsonl` ·
`hypotheses.jsonl`). Each is correct, but **"what followed what, and why"**
only becomes visible by matching timestamps and ids after the fact.

In `trace.jsonl` one line is one event, and it is **append-only**.

## ★ It must have no side effects

```
disabled   does nothing (`ev` returns immediately)
enabled    writes one line and flushes. **It does not touch the compute path**
★ checked  run with MockLLM on and off and see that the outputs are
           **identical** (`tests/test_trace.py`)
```

## Why flush

Even if it dies midway, **what got that far must survive** — D-33 is where
78 minutes and 1,400 calls were lost. A line is a few KB and there are tens
per round, so the cost is negligible.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

__all__ = ["Tracer"]


class Tracer:
    """Appends events to `trace.jsonl`. With `path=None` it **does
    nothing**."""

    __slots__ = ("_f", "n", "_seen_calls")

    def __init__(self, path: str | Path | None = None) -> None:
        self._f = None
        self.n = 0
        #: How many LLM calls are already traced (from the front of
        #: `llm.calls`).
        self._seen_calls = 0
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._f = p.open("a", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._f is not None

    def ev(self, ev: str, **kw) -> None:
        """One event. ★ When disabled it does not even touch the
        arguments."""
        if self._f is None:
            return
        rec = {"ev": ev, "t": round(time.time(), 3)}
        rec.update(kw)
        self._f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._f.flush()
        self.n += 1

    def llm_calls(self, llm, **kw) -> None:
        """Copy whatever accumulated in `llm.calls`, in full.

        ★ It **does not touch** the LLM-side code — it only reads a list that
        already exists. Prompts and responses are stored in full (a 24-round
        run measures 2.7 MB).
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
