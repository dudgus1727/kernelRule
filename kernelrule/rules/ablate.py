"""규칙에서 **항 하나를 빼는** 변환 (D-85).

Critic 이 "이 항은 물리로 설명 못 하겠다" 고 한 것을 **정량으로 검증**하려면
그 항을 빼고 다시 적합해 봐야 한다.

```
빼도 regret 이 안 나빠진다   ★ Critic 판정이 맞았다
크게 나빠진다               Critic 이 틀렸거나, 설명 못 해도 유용한 항이다
```

⚠️ **가중치를 0 으로 두는 것과 다르다.** 0 으로 두면 적합기가 도로 키운다.
항을 실제로 지우고 남은 가중치를 다시 번호 매겨야 `len(w0)` 규약이 산다.

⚠️ **모든 규칙에 되는 것은 아니다.** 한 줄에 여러 항이 있거나 분기 안에
항이 하나뿐이면 지울 수 없다 — 그때는 `AblateError` 다. **조용히 건너뛰지
않는다** (§26.4): 못 지운 것은 "지웠는데 영향 없음" 과 완전히 다르다.
"""

from __future__ import annotations

import ast
import re

__all__ = ["AblateError", "drop_terms", "term_exprs", "reorder_terms",
           "term_indices"]

_W = re.compile(r"\bw\[(\d+)\]")


class AblateError(RuntimeError):
    """이 규칙에서 그 항을 지울 수 없다."""


def term_indices(code: str) -> list[int]:
    """이 규칙이 쓰는 `w` 인덱스들."""
    return sorted({int(i) for i in _W.findall(code)})


def _line_indices(line: str) -> set[int]:
    return {int(i) for i in _W.findall(line)}


def drop_terms(code: str, drop: set[int]) -> str:
    """`w[i] (i in drop)` 가 곱해진 항을 지우고 가중치를 다시 번호 매긴다."""
    keep_idx = [i for i in term_indices(code) if i not in drop]
    if not keep_idx:
        raise AblateError("항을 다 지우면 규칙이 남지 않는다")

    lines = code.split("\n")
    out: list[str] = []
    for ln in lines:
        got = _line_indices(ln)
        hit = got & drop
        if not hit:
            out.append(ln)
            continue
        if got - drop:
            raise AblateError(
                f"한 줄에 여러 항이 있다: w{sorted(got)} — 이 줄만 지우면 "
                "남은 항까지 사라진다")
        stripped = ln.strip()
        if stripped.startswith("return"):
            # `return s + expr * w[i]` -> `return s`
            indent = ln[:len(ln) - len(ln.lstrip())]
            out.append(f"{indent}return s")
        # 그 외에는 줄을 통째로 뺀다
    new = "\n".join(out)

    # ★ 첫 항을 지웠으면 `s = s + …` 가 첫 대입이 된다 — `s` 가 없다.
    new = _fix_first_assignment(new)
    # 남은 가중치를 0..n-1 로 다시 번호 매긴다 (len(w0) 규약, §29.4)
    remap = {old: k for k, old in enumerate(keep_idx)}
    new = _W.sub(lambda m: f"w[{remap[int(m.group(1))]}]", new)

    # ★ 빈 블록을 **파싱 전에** 잡는다. 파싱은 IndentationError 를 내는데
    #   그 메시지로는 "왜 못 지웠나" 를 못 읽는다.
    _refuse_if_empty_block(new)
    try:
        ast.parse(new)
    except SyntaxError as e:
        raise AblateError(f"지운 뒤 문법이 깨진다: {e}") from None
    return new


def _refuse_if_empty_block(code: str) -> None:
    """`if …:` 뒤에 들여쓴 줄이 안 남았으면 그 항이 유일했던 것이다."""
    lines = [ln for ln in code.split("\n") if ln.strip()]
    for k, ln in enumerate(lines[:-1]):
        if not ln.rstrip().endswith(":"):
            continue
        indent = len(ln) - len(ln.lstrip())
        nxt = lines[k + 1]
        if len(nxt) - len(nxt.lstrip()) <= indent:
            raise AblateError("분기 안이 비었다 — 그 항이 유일했다")
    if lines and lines[-1].rstrip().endswith(":"):
        raise AblateError("분기 안이 비었다 — 그 항이 유일했다")


def _fix_first_assignment(code: str) -> str:
    """첫 `s` 대입이 `s = s + …` 이면 `s = …` 로 바꾼다.

    항을 지우기 전의 첫 줄이 `s = <식> * w[0]` 이었고 그것을 지웠으면,
    다음 줄이 `s = s + <식> * w[1]` 이라 **`s` 가 정의되기 전에 읽힌다.**
    """
    lines = code.split("\n")
    for k, ln in enumerate(lines):
        st = ln.strip()
        if not st.startswith("s = "):
            continue
        if st.startswith("s = s "):
            indent = ln[:len(ln) - len(ln.lstrip())]
            rest = st[len("s = s "):].lstrip()
            op, _, tail = rest.partition(" ")
            if op not in ("+", "-"):
                raise AblateError(
                    f"첫 항을 지웠는데 다음 줄이 `s = s {op} …` 다 — "
                    "덧셈/뺄셈이 아니라 되살릴 수 없다")
            if op == "-":
                tail = f"-({tail})"
            lines[k] = f"{indent}s = {tail}"
        break
    return "\n".join(lines)


_TERM = re.compile(r"^(\s*)s = (?:s \+ )?(.*?)\s*\*\s*w\[(\d+)\]\s*$")


def term_exprs(code: str) -> dict[int, str]:
    """`w[i]` -> 그 항의 식. **한 줄에 항 하나**인 규칙만 받는다.

    형태가 다르면 `AblateError` 다 — 조용히 일부만 읽으면 순서를 섞었을 때
    항이 사라진다 (§26.4).
    """
    out: dict[int, str] = {}
    for ln in code.split("\n"):
        if "w[" not in ln:
            continue
        m = _TERM.match(ln)
        if m is None:
            raise AblateError(f"이 줄은 `s = <식> * w[i]` 형태가 아니다: "
                              f"{ln.strip()[:60]}")
        out[int(m.group(3))] = m.group(2)
    if not out:
        raise AblateError("항을 하나도 못 읽었다")
    return out


def reorder_terms(code: str, order: list[int]) -> str:
    """항 **순서를 바꾸고** `w` 인덱스를 새 순서로 다시 매긴다 (D-86).

    Critic 이 "마지막 항" 을 지목하는 것이 위치 편향인지 보려면, 같은
    식들을 다른 순서로 보여 줘야 한다. **인덱스도 다시 매긴다** — 안 하면
    모델이 원래 순서를 그대로 읽는다.

    `order` 는 원래 인덱스를 새 순서로 늘어놓은 것이다.
    """
    exprs = term_exprs(code)
    if sorted(order) != sorted(exprs):
        raise AblateError(f"순서가 항 집합과 다르다: {order} vs {sorted(exprs)}")
    head = code.split("\n", maxsplit=1)[0]
    if not head.startswith("def score"):
        raise AblateError("첫 줄이 `def score(...)` 가 아니다")
    lines = [head]
    for k, old in enumerate(order):
        op = "s = " if k == 0 else "s = s + "
        lines.append(f"    {op}{exprs[old]} * w[{k}]")
    lines.append("    return s")
    return "\n".join(lines) + "\n"
