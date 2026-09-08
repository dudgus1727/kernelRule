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

⚠️ 2026-09-08 (D-146): the strings this file **writes into
`docs/decisions.md`** (the BEGIN marker, the index heading and note, the
`_SUPERSEDED` statuses) were translated into English. The entry titles
themselves are copied from the `## D-N` headers of the body, so they **stay
Korean** — `docs/decisions.md` is the running record and it is not translated.
The Korean originals of the generated strings are at commit `ee53b4d`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs/decisions.md"
BEGIN = "<!-- INDEX:BEGIN — experiments/decisions_index.py builds it -->"
END = "<!-- INDEX:END -->"

#: ★ The corrected decisions. `D number -> (status, the D that corrected it)`.
#: **Written by a human.**
_SUPERSEDED: dict[int, tuple[str, str]] = {
    77: ("partly withdrawn", "D-103 — on the rank path the reach is 100%"),
    92: ("corrected", ("the correction inside D-92 — the sample unit was "
                       "wrong")),
    102: ("withdrawn", "D-103 — filling in the 2x2 made it an interaction"),
    105: ("condition error", ("D-108 — the budget experiment was valid on "
                              "the fourth try")),
    106: ("condition error", "D-108"),
    107: ("condition error", "D-108"),
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
    rows = ["", "## Index", "",
            "★ This block is **generated** — `experiments/decisions_index.py`.",
            "Do not edit it by hand. Add the D and run that script.", ""]
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
