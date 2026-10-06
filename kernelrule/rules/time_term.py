"""★ The time main term of a rule — which part of the score is the time
estimate (D-193, D-195). Pure AST: it reads no table and no time.

```
path-time features   PATH — each a time over the ideal time of the problem
time main term       one expression that combines two or more of them inside
                     max / sqrt / log / ... (helper variables followed:
                     `sm = np.maximum(f.tm_crit_ratio, f.tm_l2_ratio)` then
                     `t = np.sqrt(np.square(sm) + ...)` is one expression)
main weights         the w[i] that multiply such an expression — when the
                     rule weights its time estimate instead of fixing it
```

`analyse` is the pre-registered structure test of D-193
(`experiments/seed_structure.py`); `main_weight_indices` is what the term
cap (`rules/caps.py`) keeps out of the corrections.
"""

from __future__ import annotations

import ast

__all__ = ["PATH", "COMBINE", "analyse", "main_weight_indices"]

#: The path-time features of condition F4 — each a time over the ideal time.
PATH = ("tm_crit_ratio", "tm_l2_ratio", "tm_dram_ratio", "dram_w1_id_ratio",
        "dram_w1_hz_ratio")
#: The calls that combine paths into one time (and their log).
COMBINE = ("maximum", "fmax", "sqrt", "hypot", "log2", "log", "minimum",
           "fmin", "power")


def _feats(node: ast.AST) -> set[str]:
    return {n.attr for n in ast.walk(node)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "f"}


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _weights(node: ast.AST) -> set[int]:
    return {n.slice.value for n in ast.walk(node)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
            and n.value.id == "w" and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, int)}


def _fn(code: str) -> ast.FunctionDef:
    return next(n for n in ast.parse(code).body
                if isinstance(n, ast.FunctionDef))


def _helpers(fn: ast.FunctionDef) -> tuple[dict, dict, dict]:
    """variable -> (path features it carries, does a weight sit in it),
    variable -> does it hold a path **combination** (>= 2 paths in one
    COMBINE call), and variable -> every feature it reads. `s` is the
    score, not a helper."""
    var: dict[str, tuple[set[str], bool]] = {}
    combo: dict[str, bool] = {}
    allf: dict[str, set[str]] = {}

    def closure(node: ast.AST) -> tuple[set[str], bool]:
        fs, w = set(_feats(node)) & set(PATH), bool(_weights(node))
        for nm in _names(node):
            if nm in var and nm != "s":
                fs |= var[nm][0]
                w = w or var[nm][1]
        return fs, w

    def has_combo(node: ast.AST) -> bool:
        if any(combo.get(nm) for nm in _names(node) if nm != "s"):
            return True
        for c in ast.walk(node):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) \
                    and c.func.attr in COMBINE:
                fs: set[str] = set()
                for a in c.args:
                    fs |= closure(a)[0]
                if len(fs) >= 2:
                    return True
        return False

    for st in ast.walk(fn):
        if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                and isinstance(st.targets[0], ast.Name) \
                and st.targets[0].id != "s":
            t = st.targets[0].id
            fs, w = closure(st.value)
            old = var.get(t, (set(), False))
            var[t] = (old[0] | fs, old[1] or w)
            combo[t] = combo.get(t, False) or has_combo(st.value)
            fs_all = set(_feats(st.value))
            for nm in _names(st.value):
                if nm in allf and nm != "s":
                    fs_all |= allf[nm]
            allf[t] = allf.get(t, set()) | fs_all
    return var, combo, allf


def analyse(code: str) -> dict:
    """Does the rule carry a time main term, and what else is in it."""
    fn = _fn(code)
    var, _combo, _allf = _helpers(fn)

    def closure(node: ast.AST) -> tuple[set[str], bool]:
        fs, w = set(_feats(node)) & set(PATH), bool(_weights(node))
        for nm in _names(node):
            if nm in var and nm != "s":
                fs |= var[nm][0]
                w = w or var[nm][1]
        return fs, w

    found: list[dict] = []
    for st in ast.walk(fn):
        if isinstance(st, ast.Call) and isinstance(st.func, ast.Attribute) \
                and st.func.attr in COMBINE:
            fs, w = set(), False
            for a in st.args:
                f2, w2 = closure(a)
                fs |= f2
                w = w or w2
            if len(fs) >= 2:
                found.append({"call": st.func.attr, "paths": sorted(fs),
                              "weighted_inside": bool(w),
                              "expr": ast.unparse(st)[:160]})
    branches = [ast.unparse(n.test) for n in ast.walk(fn)
                if isinstance(n, ast.If)]
    return {"time_main_term": bool(found),
            "main_term_paths": sorted({p for x in found for p in x["paths"]}),
            "combinations": found[:4], "n_weights": len(_weights(fn)),
            "branches": branches,
            "path_features_used": sorted(_feats(fn) & set(PATH)),
            "features_used": sorted(_feats(fn))}


def main_weight_indices(code: str) -> set[int]:
    """The weights that multiply the time main term — an additive piece of
    the score that uses a path combination (directly or through a helper
    variable) and reads **nothing but path-time features**. A piece that
    mixes a path time with other axes (a DRAM time times a pressure
    factor) is a correction. Empty when the time term carries no weight
    (the D-193 template `s = 10.0 * np.log2(t)`)."""
    fn = _fn(code)
    _var, combo, allf = _helpers(fn)
    out: set[int] = set()
    for st in ast.walk(fn):
        if not (isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and st.targets[0].id == "s"):
            continue
        # the score statement: each additive piece is judged on its own
        pieces = _addends(st.value)
        for piece in pieces:
            uses_combo = any(combo.get(nm) for nm in _names(piece)
                             if nm != "s")
            if not uses_combo:
                for c in ast.walk(piece):
                    if isinstance(c, ast.Call) \
                            and isinstance(c.func, ast.Attribute) \
                            and c.func.attr in COMBINE:
                        fs = set(_feats(c)) & set(PATH)
                        if len(fs) >= 2:
                            uses_combo = True
            reads = set(_feats(piece))
            for nm in _names(piece):
                if nm in allf and nm != "s":
                    reads |= allf[nm]
            if uses_combo and reads <= set(PATH):
                out |= _weights(piece)
    return out


def _addends(node: ast.AST) -> list[ast.AST]:
    """`a + b - c` -> [a, b, c] (the score is a sum of terms)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add,
                                                            ast.Sub)):
        return _addends(node.left) + _addends(node.right)
    return [node]
