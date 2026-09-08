"""The transform that **removes one term** from a rule (D-85).

To verify **quantitatively** what the Critic called "a term I cannot explain
physically", the term has to be removed and the weights refitted.

    regret does not get worse    ★ the Critic's verdict was right
    it gets much worse           the Critic was wrong, or the term is useful
                                 even though it cannot be explained

⚠️ **This is not the same as setting the weight to 0.** Set to 0, the fitter
grows it right back. The term must actually be deleted and the remaining
weights renumbered for the `len(w0)` convention to hold.

⚠️ **It does not work on every rule.** If a line holds several terms, or a
branch holds only one, the term cannot be removed — then it is an
`AblateError`. **It is not skipped silently** (§26.4): "could not remove" is
completely different from "removed and it made no difference".
"""

from __future__ import annotations

import ast
import re

__all__ = ["AblateError", "drop_terms", "term_exprs", "reorder_terms",
           "term_indices"]

_W = re.compile(r"\bw\[(\d+)\]")


class AblateError(RuntimeError):
    """That term cannot be removed from this rule."""


def term_indices(code: str) -> list[int]:
    """The `w` indices this rule uses."""
    return sorted({int(i) for i in _W.findall(code)})


def _line_indices(line: str) -> set[int]:
    return {int(i) for i in _W.findall(line)}


def drop_terms(code: str, drop: set[int]) -> str:
    """Remove terms multiplied by `w[i] (i in drop)` and renumber the
    weights."""
    keep_idx = [i for i in term_indices(code) if i not in drop]
    if not keep_idx:
        raise AblateError("removing every term leaves no rule")

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
                f"several terms on one line: w{sorted(got)} — removing this "
                "line would take the other terms with it")
        stripped = ln.strip()
        if stripped.startswith("return"):
            # `return s + expr * w[i]` -> `return s`
            indent = ln[:len(ln) - len(ln.lstrip())]
            out.append(f"{indent}return s")
        # Otherwise drop the whole line
    new = "\n".join(out)

    # ★ If the first term went, `s = s + …` becomes the first assignment —
    #   and `s` does not exist.
    new = _fix_first_assignment(new)
    # Renumber the remaining weights to 0..n-1 (the len(w0) convention, §29.4)
    remap = {old: k for k, old in enumerate(keep_idx)}
    new = _W.sub(lambda m: f"w[{remap[int(m.group(1))]}]", new)

    # ★ Catch an empty block **before parsing**. Parsing raises
    #   IndentationError, and that message does not say "why it failed".
    _refuse_if_empty_block(new)
    try:
        ast.parse(new)
    except SyntaxError as e:
        raise AblateError(f"the syntax breaks after removal: {e}") from None
    return new


def _refuse_if_empty_block(code: str) -> None:
    """No indented line left after `if …:` means that term was the only
    one."""
    lines = [ln for ln in code.split("\n") if ln.strip()]
    for k, ln in enumerate(lines[:-1]):
        if not ln.rstrip().endswith(":"):
            continue
        indent = len(ln) - len(ln.lstrip())
        nxt = lines[k + 1]
        if len(nxt) - len(nxt.lstrip()) <= indent:
            raise AblateError("the branch body is empty — that term was the "
                              "only one")
    if lines and lines[-1].rstrip().endswith(":"):
        raise AblateError("the branch body is empty — that term was the "
                          "only one")


def _fix_first_assignment(code: str) -> str:
    """If the first `s` assignment is `s = s + …`, turn it into `s = …`.

    If the first line before removal was `s = <expr> * w[0]` and that is what
    was removed, the next line is `s = s + <expr> * w[1]` — so **`s` is read
    before it is defined.**
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
                    f"the first term was removed and the next line is "
                    f"`s = s {op} …` — not an addition or subtraction, so it "
                    "cannot be repaired")
            if op == "-":
                tail = f"-({tail})"
            lines[k] = f"{indent}s = {tail}"
        break
    return "\n".join(lines)


_TERM = re.compile(r"^(\s*)s = (?:s \+ )?(.*?)\s*\*\s*w\[(\d+)\]\s*$")


def term_exprs(code: str) -> dict[int, str]:
    """`w[i]` -> that term's expression. Accepts only rules with **one term
    per line**.

    A different shape is an `AblateError` — silently reading only part of it
    would make terms vanish when the order is shuffled (§26.4).
    """
    out: dict[int, str] = {}
    for ln in code.split("\n"):
        if "w[" not in ln:
            continue
        m = _TERM.match(ln)
        if m is None:
            raise AblateError("this line is not of the form "
                              "`s = <expr> * w[i]`: "
                              f"{ln.strip()[:60]}")
        out[int(m.group(3))] = m.group(2)
    if not out:
        raise AblateError("no term could be read")
    return out


def reorder_terms(code: str, order: list[int]) -> str:
    """**Reorder** the terms and renumber the `w` indices to match (D-86).

    To see whether the Critic pointing at "the last term" is a position bias,
    the same expressions must be shown in a different order. **The indices are
    renumbered too** — otherwise the model just reads the original order.

    `order` lists the original indices in their new order.
    """
    exprs = term_exprs(code)
    if sorted(order) != sorted(exprs):
        raise AblateError(f"the order differs from the term set: {order} vs "
                          f"{sorted(exprs)}")
    head = code.split("\n", maxsplit=1)[0]
    if not head.startswith("def score"):
        raise AblateError("the first line is not `def score(...)`")
    lines = [head]
    for k, old in enumerate(order):
        op = "s = " if k == 0 else "s = s + "
        lines.append(f"    {op}{exprs[old]} * w[{k}]")
    lines.append("    return s")
    return "\n".join(lines) + "\n"
