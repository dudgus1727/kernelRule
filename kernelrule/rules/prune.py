"""★ Removing one weight's term(s) from a rule (D-190 §7).

★ 2026-10-07 (D-198): **the loop uses this when `LoopConfig.prune_dead`**
(`--prune-dead`) — user decision, after the record below and D-190 §7's
re-measurement on d190 rules (inner-CV +0.0012 against a +0.001 bar, one
unseen shape 1.23 -> 8.27). **On by default** (user decision, the same day);
every run before D-198 is unpruned (`runset._OLD_DEFAULTS`).

History — why it stayed out until then. Pruning the dead weights of the c2 a6000
elites before they enter the archive was measured first
(`experiments/d190_offline.py prune`, `docs/artifacts/d190/prune.json`): 42% of
the weights went with the training regret unchanged, the holdout of all 134
elites moved 1.1339 -> 1.1333, but the rule each run would report moved
1.1217 -> 1.1263 — one run went 1.1186 -> 1.2304, because a weight that moves
no training pick can still move holdout picks. The criterion set before the
measurement (no worse than +0.001 on both) was not met, so it stayed out.

## What "removing a weight" means here

A weight can only be removed when every use of `w[i]` is a **factor of a
term** — a pure product chain (`*`, and `/` with the chain on the
numerator side) whose top is an operand of `+`/`-`, or the whole value of an
assignment, an augmented assignment or a `return`:

```
s = s + np.log2(1.0 + f.edge_waste) * w[0]     ✔ the term goes
s = np.where(p.x < 1024, f.a, 0.0) * w[26]     ✔ s = 0.0
s = s + np.power(f.a, w[3]) * w[4]             ✔ for w[4] — w[3] goes with it
s = s + np.power(f.a, w[3]) * w[4]             ✗ for w[3] (an exponent)
s = np.where(f.a < w[5], ...)                   ✗ (a threshold)
```

For those uses, deleting the term **is** `w[i] = 0` — the scores are the
same (up to `0 * inf`, which deleting avoids). Anything else is not
removable and `drop_weight` returns `None`; nothing is guessed.

## The text is edited, not regenerated

`ast.unparse` would drop every comment, and the comments are the
RuleEditor's own explanation of each term — the next round reads them. So
only the line range of a touched statement is replaced (or deleted), and the
weight indices are renumbered in the text. A statement that shares its lines
with another one (`a = 1; b = 2`) is not touched — `None`.

⚠️ It knows nothing about scores. Whether a removal is **kept** is decided
by the caller (`prune_dead`) with the training regret.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Sequence

__all__ = ["drop_weight", "removable_indices", "prune_dead", "PruneStep"]

_W_INDEX = re.compile(r"(?<![\w.])w\[(\d+)\]")


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    out: dict[int, ast.AST] = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            out[id(c)] = n
    return out


def _is_w(n: ast.AST, i: int | None = None) -> bool:
    return (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
            and n.value.id == "w" and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, int)
            and (i is None or n.slice.value == i))


def _term_of(node: ast.AST, par: dict[int, ast.AST]) -> ast.AST:
    """The top of the product chain `node` is a factor of."""
    cur = node
    while True:
        up = par.get(id(cur))
        product = isinstance(up, ast.BinOp) and (
            isinstance(up.op, ast.Mult)
            or (isinstance(up.op, ast.Div) and up.left is cur))
        sign = isinstance(up, ast.UnaryOp) and isinstance(
            up.op, (ast.USub, ast.UAdd))
        if not (product or sign):
            return cur
        cur = up


_SIMPLE = (ast.Assign, ast.AugAssign, ast.Return, ast.AnnAssign)


def _site(term: ast.AST, par: dict[int, ast.AST]):
    """Where the term sits: `("binop", binop)` or `("stmt", stmt)`, or
    `None` when it is not a removable position."""
    up = par.get(id(term))
    if (isinstance(up, ast.BinOp) and isinstance(up.op, (ast.Add, ast.Sub))
            and (up.left is term or up.right is term)):
        return ("binop", up)
    if isinstance(up, _SIMPLE) and getattr(up, "value", None) is term:
        return ("stmt", up)
    return None


def _stmt_of(node: ast.AST, par: dict[int, ast.AST]) -> ast.stmt | None:
    cur = node
    while cur is not None and not isinstance(cur, ast.stmt):
        cur = par.get(id(cur))
    return cur if isinstance(cur, _SIMPLE) else None


def removable_indices(code: str) -> frozenset[int]:
    """The `w[i]` whose every use is a factor of a term (see the module
    docstring)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return frozenset()
    par = _parents(tree)
    ok: dict[int, bool] = {}
    for n in ast.walk(tree):
        if _is_w(n):
            i = n.slice.value
            good = (_stmt_of(n, par) is not None
                    and _site(_term_of(n, par), par) is not None)
            ok[i] = ok.get(i, True) and good
    return frozenset(i for i, g in ok.items() if g)


class _Remove(ast.NodeTransformer):
    """Removes the given term nodes from a statement (works on a copy)."""

    def __init__(self, terms: set[int]) -> None:
        self.terms = terms
        self.root_gone = False

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if not isinstance(node.op, (ast.Add, ast.Sub)):
            return node
        left_gone = id(node.left) in self.terms or _gone(node.left)
        right_gone = id(node.right) in self.terms or _gone(node.right)
        if left_gone and right_gone:
            return _GONE()
        if right_gone:
            return node.left
        if left_gone:
            return (node.right if isinstance(node.op, ast.Add)
                    else ast.UnaryOp(op=ast.USub(), operand=node.right))
        return node


class _GONE(ast.Constant):
    """A marker for "this whole expression was a removed term"."""

    def __init__(self) -> None:
        super().__init__(value=0.0)


def _gone(n: ast.AST) -> bool:
    return isinstance(n, _GONE)


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def drop_weight(code: str, w: Sequence[float],                 # noqa: PLR0911
                i: int) -> tuple[str, list[float], list[int]] | None:
    """Removes every term `w[i]` is a factor of, renumbers the remaining
    weights and returns `(code, w, removed_old_indices)`.

    `removed_old_indices` holds `i` and every other index that was used only
    inside the removed terms (an exponent weight, say). `None` when `i` is
    not removable or the edit would not be clean.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if i not in removable_indices(code):
        return None
    par = _parents(tree)
    lines = code.split("\n")

    # The statements to rewrite and, per statement, the term nodes to drop.
    by_stmt: dict[int, tuple[ast.stmt, set[int]]] = {}
    for n in ast.walk(tree):
        if _is_w(n, i):
            st = _stmt_of(n, par)
            term = _term_of(n, par)
            if st is None:
                return None
            by_stmt.setdefault(id(st), (st, set()))[1].add(id(term))

    edits: list[tuple[int, int, list[str]]] = []      # (first, last, new)
    for st, terms in by_stmt.values():
        a, b = st.lineno, st.end_lineno or st.lineno
        ind = _indent(lines[a - 1])
        # A statement must own its lines — `a = 1; b = 2`, `if x: s = ...`
        # and `else: s = ...` are not touched. ⚠️ AST offsets are UTF-8
        # bytes.
        if st.col_offset != len(ind.encode()):
            return None
        tail = lines[b - 1].encode()[st.end_col_offset or 0:].decode(
            errors="ignore")
        if tail.strip() and not tail.strip().startswith("#"):
            return None
        new = _rewrite(st, terms)
        # A trailing comment stays on a rewritten line; it goes with a
        # deleted one.
        keep = ("  " + tail.strip()) if tail.strip() else ""
        edits.append((a, b, [] if new is None else [ind + new + keep]))

    # A block emptied by the deletions gets a `pass`.
    deleted = {a for a, _b, new in edits if not new}
    for n in ast.walk(tree):
        for field in ("body", "orelse"):
            blk = getattr(n, field, None)
            if not isinstance(blk, list) or not blk:
                continue
            if all(isinstance(s, ast.stmt) and s.lineno in deleted
                   for s in blk):
                last = blk[-1]
                for k, (a, b, _new) in enumerate(edits):
                    if a == last.lineno:
                        edits[k] = (a, b, [_indent(lines[a - 1]) + "pass"])

    out = list(lines)
    for a, b, new in sorted(edits, key=lambda e: -e[0]):
        lo = a - 1
        if not new:
            # ★ The comment lines right above a deleted statement go with it
            #   — unless the next line is another **term** (a statement at
            #   the same indent that uses `w`) with no comment of its own,
            #   in which case the comment may head the whole run of terms.
            nxt = out[b] if b < len(out) else ""
            heads_block = (nxt.strip() and not nxt.strip().startswith("#")
                           and _indent(nxt) == _indent(out[a - 1])
                           and _W_INDEX.search(nxt) is not None)
            if not heads_block:
                while (lo - 1 >= 0 and out[lo - 1].strip().startswith("#")
                       and _indent(out[lo - 1]) == _indent(out[a - 1])):
                    lo -= 1
        out[lo:b] = new
    text = "\n".join(out)
    try:
        ast.parse(text)
    except SyntaxError:
        return None

    used = sorted({int(m) for m in _W_INDEX.findall(text)})
    removed = [k for k in range(len(w)) if k not in used]
    if i not in removed or not used:
        return None
    remap = {old: new for new, old in enumerate(used)}
    text = _W_INDEX.sub(lambda m: f"w[{remap[int(m.group(1))]}]", text)
    return text, [float(w[old]) for old in used], removed


def _rewrite(st: ast.stmt, terms: set[int]) -> str | None:
    """The statement after its terms are removed. `None` = delete it."""
    import copy

    # `deepcopy` keeps no node identity, so the terms are marked first.
    for n in ast.walk(st):
        if id(n) in terms:
            n._kr_drop = True                             # noqa: SLF001
    st2 = copy.deepcopy(st)
    for n in ast.walk(st):
        if hasattr(n, "_kr_drop"):
            del n._kr_drop                                # noqa: SLF001
    marked = {id(n) for n in ast.walk(st2) if hasattr(n, "_kr_drop")}
    rm = _Remove(marked)
    if getattr(st2, "value", None) is not None and id(st2.value) in marked:
        st2.value = _GONE()
    else:
        st2 = rm.visit(st2)
    val = getattr(st2, "value", None)
    if isinstance(st2, ast.AugAssign) and _gone(val):
        return None
    if isinstance(st2, (ast.Assign, ast.Return)) and _gone(val):
        st2.value = ast.Constant(value=0.0)
    if (isinstance(st2, ast.Assign) and len(st2.targets) == 1
            and isinstance(st2.targets[0], ast.Name)
            and isinstance(st2.value, ast.Name)
            and st2.value.id == st2.targets[0].id):
        return None                        # `s = s` — nothing left
    # A marker left inside a call (`np.log2(a*w[1] + b*w[1])`) prints as the
    # constant it stands for — the same as `w[i] = 0`.
    st2 = _Plain().visit(st2)
    return ast.unparse(ast.fix_missing_locations(st2))


class _Plain(ast.NodeTransformer):
    # ⚠️ The visitor dispatches on the class name, so the marker needs its
    #   own method — `visit_Constant` never sees it.
    def visit__GONE(self, node: ast.Constant):
        return ast.Constant(value=0.0)


class PruneStep(dict):
    """One attempted removal, for the trace."""


def prune_dead(code: str, w: Sequence[float], order: Sequence[int],
               regret_of: Callable[[str, list[float]], float],
               base: float) -> tuple[str, list[float], list[PruneStep]]:
    """★ Removes the weights in `order` one at a time and **keeps a removal
    only if the training regret did not get worse** (D-190 §7).

    `order` holds indices of the **original** rule. `regret_of(code, w)`
    scores a candidate on the training split (no refit) and returns `inf`
    when it cannot be run. One at a time, because "one weight alone does not
    move a pick" does not mean "two of them together do not" — two terms
    that stand in for each other each look dead.
    """
    cur_code, cur_w, cur = code, [float(x) for x in w], float(base)
    alive = list(range(len(w)))          # current position -> original index
    log: list[PruneStep] = []
    for orig in order:
        if orig not in alive:
            log.append(PruneStep(i=orig, kept=False, why="already_gone"))
            continue
        pos = alive.index(orig)
        got = drop_weight(cur_code, cur_w, pos)
        if got is None:
            log.append(PruneStep(i=orig, kept=False, why="not_removable"))
            continue
        new_code, new_w, removed = got
        r = regret_of(new_code, new_w)
        if r <= cur + 1e-12:
            gone = {alive[k] for k in removed}
            alive = [a for a in alive if a not in gone]
            log.append(PruneStep(i=orig, kept=True, regret=r,
                                 also=sorted(gone - {orig})))
            cur_code, cur_w, cur = new_code, new_w, r
        else:
            log.append(PruneStep(i=orig, kept=False, why="worse", regret=r))
    return cur_code, cur_w, log
