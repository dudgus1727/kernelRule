"""Checks and registers the features the LLM wrote (§11.4 —
FeatureWriter).

## Why a separate registry

`REGISTRY` is the 24 a human wrote. Mixing generated features into it makes
**the F0~F3 conditions indistinguishable** — "did it merely combine the 24"
versus "did it build them itself" is the fundamental question of this
project.

    F0   no features       can it put physics into code from scratch
    F1   raw values only   can it build derived physical quantities
                                                        ★ the most interesting
    F2   the basic 5       can it build on top of them
    F3   all 24            combination only (= every run so far)

## The order of checks — one failure means it is not registered (§26.4)

    1. AST      banned names / import / cfg.ext / hardcoded constants
    2. sandbox  isolated execution. Infinite loops and exceptions are caught
                here
    3. §8.3     runs, range, vectorisation, constancy, duplication, scale
                invariance, usefulness

**Item 3's "scale invariance" is the core.** If changing `hw` does not change
the value, a hardware constant was hardcoded, and then architecture transfer
collapses.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import replace

import numpy as np

from kernelrule.core.matrix import FeatureMatrix
from kernelrule.features import Feature, FeatureRegistry

__all__ = ["FeatureRejected", "check_feature_code", "compile_feature",
           "register_generated", "RAW_FIELDS", "field_block"]


class FeatureRejected(ValueError):
    """The feature did not pass the checks. **It is not patched up and
    used** (§26.4)."""


#: The raw fields exposed in the prompt. `Config.ext` is **deliberately
#: left out** (§4.3).
RAW_FIELDS: dict[str, tuple[str, ...]] = {
    # ★ `bytes_per_element` / `acc_bytes_per_element` are derived from
    #   `dtype`, so they are **not new information.** They are exposed
    #   because of §30.11 — `p.dtype` is a string and the sandbox has no
    #   `np.dtype(...).itemsize`, so an LLM trying to build a roofline was
    #   refused three times in a row (D-63).
    "p": ("M", "N", "K", "dtype", "acc_dtype",
          "bytes_per_element", "acc_bytes_per_element",
          "layout_a", "layout_b", "layout_c"),
    "hw": ("sm_count", "smem_per_block", "max_threads_per_sm", "regs_per_sm",
           "peak_tflops_f16", "bandwidth_gbps", "l2_bytes", "ridge_point"),
    "cfg": ("tile_m", "tile_n", "tile_k", "align_a", "align_b", "align_c",
            "split_k", "split_k_mode", "regs_per_thread", "threads",
            "smem_bytes", "spill_bytes", "max_blocks_per_sm",
            "pipeline_kind", "inst_total"),
}

_ALLOWED_NP = frozenset({
    "where", "clip", "minimum", "maximum", "log", "log2", "sqrt", "abs",
    "exp", "power", "sign", "floor", "ceil", "round", "isfinite",
    "nan_to_num", "square", "fmin", "fmax"})
_ALLOWED_MATH = frozenset({
    "ceil", "floor", "log", "log2", "sqrt", "exp", "fabs", "pow", "inf"})

#: If one of these names appears, it is trying to look at the answer
#: (§3).
_BANNED = ("time_ms", "cublas_ms", "difficulty", "tflops", "best_ms",
           "distinct_time_frac", "TABLE", "__", "open", "eval", "exec",
           "import", "random")

#: Literals that look like hardware constants. They must be read from
#: `hw.*`.
_HW_LITERALS = re.compile(r"\b(84|101376|99328|65536|1536|116\.1|729\.7|"
                          r"159\.1|154\.8|768|6291456)\b")

MAX_LINES = 12


def field_block() -> str:
    """The list of raw fields to put in the prompt."""
    out = []
    for base, names in RAW_FIELDS.items():
        out.append(f"### `{base}` — "
                   + {"p": "the GEMM shape", "hw": "the hardware",
                      "cfg": "the kernel config"}[base])
        out.append("  " + "  ".join(f"`{base}.{n}`" for n in names))
    out.append("")
    out.append("`hw.ridge_point` is `peak_flops / bandwidth` — the knee of "
               "the roofline, already computed for you.")
    out.append("")
    out.append("★ `p.dtype` / `p.acc_dtype` are **strings** (like "
               "`\"f16\"`). If you need bytes, use "
               "`p.bytes_per_element` / `p.acc_bytes_per_element` — floats "
               "derived from the dtype.")
    return "\n".join(out)


def uses_cfg(code: str) -> bool:
    """Does this feature function reference `cfg.*` (§30.12)?

    The **AST layer** of the `shape_level` verdict. If it does not, the code
    itself settles that the value is config-independent — a verdict that
    does not depend on the table.
    """
    tree = ast.parse(code.strip())
    return any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
               and n.value.id == "cfg" for n in ast.walk(tree))


#: A relative variance within a shape below this counts as "constant". It
#: is a floating-point comparison, so demanding exact equality would trip on
#: last-digit differences.
_CONST_RTOL = 1e-12

#: Feature name -> why it was judged shape-level. ★ One that "references
#: cfg but is constant in this table" **must be re-judged when the bundle
#: changes** (§30.12).
SHAPE_LEVEL_REASON: dict[str, str] = {}


def detect_shape_level(f: Feature, table, *, n_shapes: int = 8
                       ) -> tuple[bool, str]:
    """★ A two-layer verdict — `(is it shape level, why)` (§30.12).

    ```
    1. data   is the relative variance across configs 0 on every shape
    2. AST    does the function reference cfg.*
              it does not      -> certainly shape level (the code guarantees it)
              it does, yet     -> ★ that may hold only in this table. A
              it is constant      warning is attached
    ```

    Why the second is needed: `alignment_guarantee_deficit` may be constant
    because alignment is config-independent across this bundle's 61 shapes.
    **In another table it may be config-dependent.** Such a feature is
    registered as shape level, but the fact is recorded so it gets re-judged
    when the bundle changes.
    """
    from kernelrule.core.matrix import FeatureMatrix

    one = FeatureRegistry(f"probe-shape-{f.name}")
    one.add(f)
    mat = FeatureMatrix(table, one)
    for p in list(table.shapes())[:n_shapes]:
        fe, _ = mat.for_shape(p)
        v = np.asarray(getattr(fe, f.name), dtype=np.float64)
        scale = max(float(np.nanmax(np.abs(v))), 1.0)
        if float(np.nanstd(v)) > _CONST_RTOL * scale:
            return False, "the value differs per config"
    if f.source and not uses_cfg(f.source):
        return True, "it does not reference cfg — the code guarantees it"
    return True, ("★ it references cfg yet is constant in this table — it "
                  "may be config-dependent in another bundle. Re-judgement "
                  "needed")


def check_feature_code(code: str, *, known: frozenset[str]) -> str:
    """The static AST check. Returns the function name if it passes, else
    raises.

    ★ It is an AST check, not a string check — refusing because a comment
    contains `import` only burns retries (D-27).
    """
    try:
        tree = ast.parse(code.strip())
    except SyntaxError as e:
        raise FeatureRejected(f"syntax error: {e}") from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise FeatureRejected("it must be **one** function definition")
    fn = tree.body[0]
    args = [a.arg for a in fn.args.args]
    if args != ["p", "hw", "cfg"]:
        raise FeatureRejected(
            f"the signature must be (p, hw, cfg). Got: {args}")
    if fn.name in known:
        raise FeatureRejected(f"that name already exists: {fn.name}")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", fn.name):
        raise FeatureRejected(
            f"the name must be lower case + underscores: {fn.name}")
    n_lines = len(code.strip().splitlines())
    if n_lines > MAX_LINES:
        raise FeatureRejected(
            f"{n_lines} lines > {MAX_LINES}. Make it shorter")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise FeatureRejected(
                "no import. np and math are already there")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base, attr = node.value.id, node.attr
            if base in RAW_FIELDS and attr not in RAW_FIELDS[base]:
                raise FeatureRejected(
                    f"unusable field: {base}.{attr}. "
                    f"allowed: {sorted(RAW_FIELDS[base])}")
            if base == "np" and attr not in _ALLOWED_NP:
                raise FeatureRejected(f"numpy function not allowed: np.{attr}")
            if base == "math" and attr not in _ALLOWED_MATH:
                raise FeatureRejected(
                    f"math function not allowed: math.{attr}")
            if base == "cfg" and attr == "ext":
                raise FeatureRejected(
                    "cfg.ext is architecture-specific (§4.3)")
        if isinstance(node, ast.Name) and node.id not in (
                "p", "hw", "cfg", "np", "math", "float", "int", "max", "min",
                "abs", "sum", "len", "bool", "round", "pow", *(
                    t.id for t in ast.walk(tree)
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store))):
            raise FeatureRejected(f"unknown name: {node.id}")

    src = ast.unparse(tree)
    # ★ The banned words are searched **after masking the allowed fields**
    #   (D-73).
    #
    #   `hw.peak_tflops_f16` is a field `RAW_FIELDS` explicitly allows, yet
    #   the banned word `"tflops"` matched it as a substring. A proposal
    #   trying to build a roofline was refused that way — **the checker
    #   banned what it had itself allowed.** The same class as D-37 (failing
    #   `inspect.getsource` was turned into "it uses hw"): a defect of the
    #   checker looks like a failure of the LLM (principle 8).
    masked = src
    for base, names in RAW_FIELDS.items():
        for n in names:
            masked = masked.replace(f"{base}.{n}", f"{base}.<ok>")
    for b in _BANNED:
        if b in masked:
            raise FeatureRejected(f"banned reference: {b!r} (§3)")
    if (m := _HW_LITERALS.search(src)):
        raise FeatureRejected(
            f"a hardware constant is hardcoded: {m.group()}. Read it from "
            f"`hw.*` — otherwise it becomes a wrong value on another GPU "
            f"(§4.3)")
    return fn.name


def compile_feature(code: str, *, known: frozenset[str]):
    """Checks, then turns it into a callable. **Only `np`/`math` are
    given.**"""
    name = check_feature_code(code, known=known)
    env: dict = {"np": np, "math": math, "__builtins__": {
        "max": max, "min": min, "abs": abs, "float": float, "int": int,
        "sum": sum, "len": len, "bool": bool, "round": round, "pow": pow}}
    exec(compile(code.strip(), f"<feature:{name}>", "exec"), env)  # noqa: S102
    return name, env[name]


def _reference_columns(table, matrix, extra: FeatureRegistry,
                       n_shapes: int = 4) -> dict[str, np.ndarray]:
    """The reference columns for the duplication verdict. It looks at
    **both what a human wrote and what has already been built.**

    Asking for a new axis and then not checking duplication against what was
    already built leads to the same thing being repeated under another name.
    """
    out: dict[str, list] = {}
    shapes = list(table.shapes())[:n_shapes]
    for reg, mat in ((matrix.registry, matrix),
                     (extra, FeatureMatrix(table, extra) if extra._items
                      else None)):
        if mat is None:
            continue
        for p in shapes:
            fe, info = mat.for_shape(p)
            for n in reg._items:
                f = reg[n]
                v = (np.full(int(info.n_candidates), float(getattr(info, n)))
                     if f.shape_level else np.asarray(getattr(fe, n), float))
                out.setdefault(n, []).append(v)
    return {n: np.concatenate(v) for n, v in out.items()}


def register_generated(code: str, *, registry: FeatureRegistry, meta: dict,
                       table, matrix, hw_alt,
                       others: dict | None = None) -> Feature:
    """Check -> sandbox -> §8.3 validation -> registration. One failure
    raises.

    `others` are the reference columns for the duplication verdict. Without
    them they are recomputed every time; the 24 a human wrote do not change,
    so it is better for the caller to build them once and pass them in (at 20
    proposals that is 30 seconds x 20).
    """
    from kernelrule.features.validate import validate_feature

    name, fn = compile_feature(code, known=frozenset(registry._items))
    rng = tuple(meta.get("expected_range", (0.0, 1.0)))
    f = Feature(name=name, fn=fn, unit=str(meta.get("unit", "dimensionless")),
                expected_range=(float(rng[0]), float(rng[1])),
                direction=str(meta.get("direction", "higher_is_worse")),
                doc=str(meta.get("rationale", ""))[:200],
                physical_meaning=str(meta.get("rationale", "")),
                code_hash=str(abs(hash(code.strip()))),
                source=code.strip())
    # ★ Validation runs on a **temporary matrix** holding the candidate.
    #   `validate_feature` reads values from `matrix.for_shape()`, so a
    #   feature not yet registered is absent from the caller's matrix. Only
    #   one column is computed, so the cost is small (§21.2).
    tmp = FeatureRegistry(f"probe-{name}")
    tmp.add(f)
    probe = FeatureMatrix(table, tmp)
    if others is None:
        others = _reference_columns(table, matrix, registry)
    rep = validate_feature(f, table, probe, hw_alt=hw_alt, others=others)
    if rep.failed:
        raise FeatureRejected(
            f"{name}: §8.3 validation failed — "
            + "; ".join(f"{c.name}: {c.detail}" for c in rep.fails()))
    # ★ The shape-level verdict (§30.12). The generation path has no
    #   `shape_feature` decorator, so **everything was being registered as
    #   config level.** Then a rule cannot branch with `if p.<x>:`, and a
    #   term that is constant within a shape cannot change the ranking at
    #   all (absolute rule 2). 5 of F1's 21 were in that state (D-65).
    is_shape, why = detect_shape_level(f, table)
    if is_shape:
        f = replace(f, shape_level=True)
        # ★ The reason is recorded. In particular one that "references cfg
        #   yet is constant in this table" **must be re-judged when the
        #   bundle changes.** It is a frozen dataclass, so it goes into a
        #   module-level table — the caller writes it into summary.json.
        SHAPE_LEVEL_REASON[name] = why
    registry.add(f)
    return f
