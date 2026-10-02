"""★ Which weights of a child rule are its parent's (D-190 §6).

## Why the loop does this and not the LLM

The RuleEditor is shown the parent's fitted weights and asked to carry them
over. On the 1,152 c2 a6000 RuleEditor calls it carried the parent's fitted
value over for **73%** of the inherited terms — 46% when |w| >= 1e4 — and of
the values it did not carry, 47% were reset to exactly 1.0 and 49% flipped
sign. 24.9% of the calls renumbered at least one inherited term, so matching
by index would be wrong a quarter of the time.

## How a weight is matched — by the statement it sits in

Each `w[i]` gets a signature per statement it appears in:

```
(the if-path down to the statement,  the statement with w[i] -> W and every
 each test unparsed)                  other w[j] -> w_)
```

Comments and the numbering do not enter it. A child weight is the parent's
`w[j]` when every one of its statements is one of `w[j]`'s and exactly one
`j` fits. Anything else is a **new** term — its start is left to the fitter
(`fit_weights(new_terms=...)`), which sets it from the score's scale.

⚠️ Pure AST. It reads no table, no score and no time.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

__all__ = ["weight_signatures", "inherit_weights"]


class _Mask(ast.NodeTransformer):
    def __init__(self, i: int) -> None:
        self.i = i

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        if (isinstance(node.value, ast.Name) and node.value.id == "w"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)):
            return ast.Name(id="W" if node.slice.value == self.i else "w_",
                            ctx=ast.Load())
        return node


def _indices(node: ast.AST) -> set[int]:
    return {n.slice.value for n in ast.walk(node)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
            and n.value.id == "w" and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, int)}


def weight_signatures(code: str) -> dict[int, set[tuple]]:
    """`w` index -> the signatures of the statements it appears in."""
    import copy

    tree = ast.parse(code)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    out: dict[int, set[tuple]] = {}

    def walk(body: list, path: tuple) -> None:
        for st in body:
            if isinstance(st, ast.If):
                t = ast.unparse(st.test)
                walk(st.body, (*path, ("if", t)))
                walk(st.orelse, (*path, ("else", t)))
                continue
            for i in sorted(_indices(st)):
                masked = _Mask(i).visit(copy.deepcopy(st))
                out.setdefault(i, set()).add((path, ast.unparse(masked)))

    if fn is not None:
        walk(fn.body, ())
    return out


def inherit_weights(child_code: str, child_w0: Sequence[float],
                    parents: Sequence[tuple[str, Sequence[float]]]
                    ) -> tuple[list[float], list[bool], list[int | None]]:
    """`(start, new, source)` for the child's weights.

    `start[i]` is the parent's fitted value where the term is inherited and
    the LLM's `w0[i]` otherwise; `new[i]` marks the latter; `source[i]` is
    the parent index (into the parent that matched — the first parent is
    tried first, then the second for what is still unmatched).
    """
    start = [float(x) for x in child_w0]
    new = [True] * len(start)
    src: list[int | None] = [None] * len(start)
    try:
        cs = weight_signatures(child_code)
    except SyntaxError:
        return start, new, src
    for p_code, p_w in parents:
        try:
            ps = weight_signatures(p_code)
        except SyntaxError:
            continue
        by_sig: dict[tuple, set[int]] = {}
        for j, sigs in ps.items():
            for sg in sigs:
                by_sig.setdefault(sg, set()).add(j)
        for i, sigs in cs.items():
            if i >= len(start) or not new[i]:
                continue
            cand: set[int] | None = None
            for sg in sigs:
                js = by_sig.get(sg, set())
                cand = set(js) if cand is None else cand & js
            if cand and len(cand) == 1:
                j = next(iter(cand))
                if j < len(p_w):
                    start[i], new[i], src[i] = float(p_w[j]), False, j
    return start, new, src
