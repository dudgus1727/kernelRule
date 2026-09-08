"""★ It **generates** the index at the head of `decisions.md`. 0 LLM calls.

    python3 experiments/decisions_index.py           # update
    python3 experiments/decisions_index.py --check    # fail if it diverged

5,900 lines hold D-1~D-114 in time order and there was no index. For a new
session to find "what was D-77" it would have to read all of it.

★ It is not written by hand. Writing it by hand makes it diverge every time
one D is added (principle 2). The titles are taken from the `## D-N  ...`
lines as they are.

Only `_SUPERSEDED` is written by a human — "what corrected what" cannot be
read automatically from the titles.

⚠️ 2026-09-08 (D-146): **the strings that go into `docs/decisions.md`** (the
BEGIN marker, the index heading and note, the `_SUPERSEDED` statuses) **stay
in Korean** — `docs/` is not translated. Only what this file prints to the
terminal is English.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs/decisions.md"
BEGIN = "<!-- INDEX:BEGIN — experiments/decisions_index.py 가 만든다 -->"
END = "<!-- INDEX:END -->"

#: ★ The corrected decisions. `D number -> (status, the D that corrected it)`.
#: **Written by a human.**
_SUPERSEDED: dict[int, tuple[str, str]] = {
    77: ("부분 철회", "D-103 — 순위 경로에서는 도달률 100%"),
    92: ("정정됨", "D-92 안의 정정 — 표본 단위가 틀렸다"),
    102: ("철회", "D-103 — 2x2 를 채우니 상호작용이었다"),
    105: ("조건 오류", "D-108 — 예산 실험은 네 번째에 유효했다"),
    106: ("조건 오류", "D-108"),
    107: ("조건 오류", "D-108"),
}


def _entries(text: str) -> list[tuple[int, str, str]]:
    """`(number, title, the raw header)`. The anchor is made from **the raw
    header**."""
    out = []
    # ★ The format is `## D-N  title`, **one** format (unified 2026-09-03).
    #   Taking the old `## D-1.` format in the same regex too missed 25 of
    #   them — **making the format one thing** is more right than growing the
    #   regex (principle 2).
    for m in re.finditer(r"^## D-(\d+)\s+(.+)$", text, re.M):
        out.append((int(m.group(1)), m.group(2).strip(),
                    m.group(0)[3:].strip()))
    return out


def _slug(header: str) -> str:
    """The GitHub anchor. Korean stays as it is, spaces become `-`, and the
    other symbols are dropped."""
    s = header.lower()
    s = re.sub(r"[^0-9a-z가-힣\s\-_]", "", s)
    return re.sub(r"\s+", "-", s.strip())


def build(text: str) -> str:
    rows = ["", "## 색인", "",
            "★ 이 블록은 **생성물이다** — `experiments/decisions_index.py`.",
            "손으로 고치지 마라. D 를 추가하고 그 스크립트를 돌려라.", ""]
    for n, title, header in _entries(text):
        mark = ""
        if n in _SUPERSEDED:
            st, by = _SUPERSEDED[n]
            mark = f"  ⚠️ **{st}** ({by})"
        rows.append(f"- [D-{n}](#{_slug(header)})  {title}{mark}")
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = DOC.read_text()
    if BEGIN in text:
        i, j = text.index(BEGIN), text.index(END) + len(END)
        body = text[:i] + text[j:]
    else:
        # It is inserted before the first `## `
        m = re.search(r"^## ", text, re.M)
        body = text
        i = m.start() if m else len(text)
    block = BEGIN + "\n" + build(body) + END + "\n\n"
    new = body[:i] + block + body[i:] if BEGIN not in text else \
        text[:text.index(BEGIN)] + block + text[text.index(END) + len(END):].lstrip("\n")
    # ★ **It counts them** (principle 38). If a new format the regex cannot
    #   catch appears, "it diverged" does not catch it — because both sides
    #   miss it equally.
    body_only = text[text.index(END) + len(END):] if END in text else text
    n_head = len(re.findall(r"^## D-", body_only, re.M))
    n_idx = len(_entries(body_only))
    if n_head != n_idx:
        sys.exit(f"only {n_idx} of the {n_head} `## D-` headers in the body "
                 f"get into the index. {n_head - n_idx} of them are not in "
                 f"the `## D-N  title` format — fix the format (D-116).")
    if a.check:
        if new != text:
            sys.exit("the decisions.md index diverged. Run "
                     "`python3 experiments/decisions_index.py`.")
        print(f"the index is up to date ({n_idx} entries, matching the body "
              f"headers)")
        return
    DOC.write_text(new)
    print(f"the index was updated, {len(_entries(body))} lines")


if __name__ == "__main__":
    main()
