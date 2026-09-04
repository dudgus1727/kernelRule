"""★ `decisions.md` 머리의 색인을 **생성한다**. LLM 0회.

    python3 experiments/decisions_index.py          # 갱신
    python3 experiments/decisions_index.py --check   # 달라졌으면 실패

5,900줄에 D-1~D-114 가 시간순으로 쌓여 있고 색인이 없었다. 새 세션이
"D-77 이 무엇이었나" 를 찾으려면 전부 훑어야 한다.

★ 손으로 쓰지 않는다. 손으로 쓰면 D 하나 추가할 때마다 달라진다 (원칙 2).
제목은 `## D-N  ...` 줄에서 그대로 가져온다.

`_SUPERSEDED` 만 사람이 적는다 — "무엇이 무엇을 정정했나" 는 제목에서
자동으로 못 읽는다.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs/decisions.md"
BEGIN = "<!-- INDEX:BEGIN — experiments/decisions_index.py 가 만든다 -->"
END = "<!-- INDEX:END -->"

#: ★ 정정된 결정. `D번호 -> (상태, 정정한 D)`. **사람이 적는다.**
_SUPERSEDED: dict[int, tuple[str, str]] = {
    77: ("부분 철회", "D-103 — 순위 경로에서는 도달률 100%"),
    92: ("정정됨", "D-92 안의 정정 — 표본 단위가 틀렸다"),
    102: ("철회", "D-103 — 2x2 를 채우니 상호작용이었다"),
    105: ("조건 오류", "D-108 — 예산 실험은 네 번째에 유효했다"),
    106: ("조건 오류", "D-108"),
    107: ("조건 오류", "D-108"),
}


def _entries(text: str) -> list[tuple[int, str, str]]:
    """`(번호, 제목, 헤더 원문)`. 앵커는 **헤더 원문**에서 만든다."""
    out = []
    # ★ 형식은 `## D-N  제목` **하나**다 (2026-09-03 통일). 옛 `## D-1.`
    #   형식을 정규식으로 같이 받다가 25개를 빠뜨렸다 — 정규식을 늘리는
    #   것보다 **형식을 하나로 만드는 것**이 맞다 (원칙 2).
    for m in re.finditer(r"^## D-(\d+)\s+(.+)$", text, re.M):
        out.append((int(m.group(1)), m.group(2).strip(),
                    m.group(0)[3:].strip()))
    return out


def _slug(header: str) -> str:
    """GitHub 앵커. 한글은 그대로, 공백은 `-`, 나머지 기호는 뺀다."""
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
        # 첫 `## ` 앞에 끼운다
        m = re.search(r"^## ", text, re.M)
        body = text
        i = m.start() if m else len(text)
    block = BEGIN + "\n" + build(body) + END + "\n\n"
    new = body[:i] + block + body[i:] if BEGIN not in text else \
        text[:text.index(BEGIN)] + block + text[text.index(END) + len(END):].lstrip("\n")
    # ★ **개수를 센다** (원칙 38). 정규식이 못 잡는 형식이 새로 생기면
    #   "달라졌다" 로는 안 잡힌다 — 양쪽이 똑같이 빠지기 때문이다.
    body_only = text[text.index(END) + len(END):] if END in text else text
    n_head = len(re.findall(r"^## D-", body_only, re.M))
    n_idx = len(_entries(body_only))
    if n_head != n_idx:
        sys.exit(f"본문의 `## D-` 헤더 {n_head}개 중 {n_idx}개만 색인에 "
                 f"들어간다. 헤더 형식이 `## D-N  제목` 이 아닌 것이 "
                 f"{n_head - n_idx}개 있다 — 형식을 고쳐라 (D-116).")
    if a.check:
        if new != text:
            sys.exit("decisions.md 색인이 달라졌다. "
                     "`python3 experiments/decisions_index.py` 를 돌려라.")
        print(f"색인 최신 ({n_idx}개, 본문 헤더와 일치)")
        return
    DOC.write_text(new)
    print(f"색인 {len(_entries(body))}줄 갱신")


if __name__ == "__main__":
    main()
