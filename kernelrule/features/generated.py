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
from kernelrule.features.validate import ReferenceColumns

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
            "pipeline_kind", "stages", "inst_total"),
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


#: What each raw field **means**, one line each (D-159).
#:
#: ⚠️ Names alone are not self-explanatory. `cfg.pipeline_kind` ·
#: `cfg.split_k_mode` · `cfg.max_blocks_per_sm` were listed with nothing but
#: their names, and a field you do not understand is either unused or used
#: wrongly.
#:
#: ⛔ **No values, and no GPU name.** The FeatureWriter is told what a field
#: is, never what it holds on this machine. A feature has to be a
#: hardware-independent formula —
#: `waves = tiles / (hw.sm_count * cfg.max_blocks_per_sm)` is right whatever
#: the SM count is — and knowing "84" invites `/ 84`, which does not
#: transfer. (`RuleWriter` is different: it fits weights **for** one GPU and
#: needs the magnitudes to match its terms.)
FIELD_MEANING: dict[str, str] = {
    # -- the shape ---------------------------------------------------------
    "p.M": "rows of A and of the output (int)",
    "p.N": "columns of B and of the output (int)",
    "p.K": "the reduction length — the mainloop runs over it (int)",
    "p.dtype": "element type of A/B, e.g. \"f16\" (string)",
    "p.acc_dtype": "type the accumulator keeps, e.g. \"f32\" (string)",
    "p.bytes_per_element": "bytes of one A/B/C element, from dtype (float)",
    "p.acc_bytes_per_element":
        "bytes of one accumulator element — the size of a parallel split-K "
        "partial sum (float)",
    "p.layout_a": "\"row\" or \"col\" for A (string)",
    "p.layout_b": "\"row\" or \"col\" for B (string)",
    "p.layout_c": "\"row\" or \"col\" for the output (string)",
    # -- the hardware ------------------------------------------------------
    "hw.sm_count": "how many SMs the GPU has (int)",
    "hw.smem_per_block": "shared memory one block may use, in bytes (int)",
    "hw.max_threads_per_sm": "thread slots on one SM (int)",
    "hw.regs_per_sm": "size of one SM's register file, in registers (int)",
    "hw.peak_tflops_f16":
        "effective f16 tensor-core throughput, TFLOP/s — measured with the "
        "clocks locked, not the spec number (float)",
    "hw.bandwidth_gbps": "effective DRAM bandwidth, GB/s (float)",
    "hw.l2_bytes": "L2 cache size in bytes (int)",
    "hw.ridge_point":
        "peak_flops / bandwidth [FLOP/byte] — the roofline knee, already "
        "computed (float)",
    # -- the config --------------------------------------------------------
    "cfg.tile_m": "rows of the output tile one CTA computes (int)",
    "cfg.tile_n": "columns of that tile (int)",
    "cfg.tile_k": "how much of K one mainloop iteration consumes (int)",
    "cfg.align_a":
        "alignment of A in elements — 8 means 16-byte access is possible, "
        "which is what cp.async needs (int)",
    "cfg.align_b": "the same for B (int)",
    "cfg.align_c": "the same for the output (int)",
    "cfg.split_k":
        "how many pieces K is cut into. 1 means no split (int)",
    "cfg.split_k_mode":
        "\"serial\" (partials reduced in place, in the accumulator type) or "
        "\"parallel\" (partials written to DRAM and read back) (string)",
    "cfg.regs_per_thread": "registers one thread uses (int)",
    "cfg.threads": "threads in one CTA (int)",
    "cfg.smem_bytes": "shared memory one CTA takes, in bytes (int)",
    "cfg.spill_bytes":
        "bytes spilled to local memory per thread. 0 means no spill (int)",
    "cfg.max_blocks_per_sm":
        "how many CTAs fit on one SM at once, from the resource limits "
        "(int)",
    "cfg.pipeline_kind":
        "\"pipelined\" (2 stages) or \"multistage\" (3+ stages, cp.async) — "
        "**different kernel families**, not a knob on one (string)",
    "cfg.stages":
        "operand-buffer stages the mainloop keeps in flight. 2 means the "
        "pipelined family; 3 and above is multistage (int)",
    "cfg.inst_total":
        "estimated SASS instruction count of the kernel (int)",
}


def field_block() -> str:
    """The list of raw fields to put in the prompt, **with one line of
    meaning each** (D-159).

    ⚠️ Every field in `RAW_FIELDS` must be in `FIELD_MEANING` — a field
    listed without one is what this fixes, so it fails loudly rather than
    printing a bare name (§26.4).
    """
    out = []
    for base, names in RAW_FIELDS.items():
        out.append(f"### `{base}` — "
                   + {"p": "the GEMM shape", "hw": "the hardware",
                      "cfg": "the kernel config"}[base])
        for n in names:
            key = f"{base}.{n}"
            try:
                mean = FIELD_MEANING[key]
            except KeyError:
                raise KeyError(
                    f"{key} is in RAW_FIELDS with no line in FIELD_MEANING. "
                    f"A field with only a name is either unused or used "
                    f"wrongly (D-159).") from None
            out.append(f"  `{key}`".ljust(31) + f" {mean}")
        out.append("")
    out.append("★ A dtype field is a **string** — the sandbox has no "
               "`np.dtype(...).itemsize`, so use `p.bytes_per_element` when "
               "you need bytes.")
    out.append("")
    out.append("★ The values are **not** given — not the GPU's name either. "
               "A feature must be a formula that is right on any GPU: "
               "`tiles / (hw.sm_count * cfg.max_blocks_per_sm)` holds "
               "whatever the SM count is.")
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


def detect_shape_level(f: Feature, table, *, n_shapes: int | None = None,
                       matrix=None) -> tuple[bool, str]:
    """★ A two-layer verdict — `(is it shape level, why)` (§30.12).

    ⚠️ 2026-09-10 (D-160): it used to look at **the first 8 shapes**
    (`n_shapes=8`). `k_loop_padding_fraction` is config-dependent on 3 of
    the 66 shapes (K = 4097 · 4098 · 4100) and all three fall outside those
    8, so it was registered as shape level and its value was then read off
    one representative config (D-159). It now looks at **every shape**, and
    that costs nothing: `FeatureMatrix(table, one)` below already computes
    the column for every shape — the slice only threw the rest away.

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

    # ★ 2026-09-11 (D-161): `matrix` is an already-built matrix holding this
    #   column over every shape. The caller passes it so the column is not
    #   computed a second time; without one it is built here as before.
    if matrix is not None and matrix.has_column(f.name) \
            and len(matrix.shapes()) == len(list(table.shapes())):
        mat = matrix
    else:
        one = FeatureRegistry(f"probe-shape-{f.name}")
        one.add(f)
        mat = FeatureMatrix(table, one)
    shapes = list(table.shapes())
    for p in (shapes if n_shapes is None else shapes[:n_shapes]):
        fe, _ = mat.for_shape(p)
        v = np.asarray(getattr(fe, f.name), dtype=np.float64)
        scale = max(float(np.nanmax(np.abs(v))), 1.0)
        if float(np.nanstd(v)) > _CONST_RTOL * scale:
            return False, "the value differs per config"
    if f.source and not uses_cfg(f.source):
        return True, "it does not reference cfg — the code guarantees it"
    return True, (f"★ it references cfg yet is constant on all "
                  f"{len(shapes)} shapes of this table — it may be "
                  f"config-dependent in another bundle. Re-judgement needed")


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
    # ★ The banned words are searched **in identifiers only** (D-73 · D-163).
    #
    #   D-73: `hw.peak_tflops_f16` is a field `RAW_FIELDS` explicitly allows,
    #   yet the banned word `"tflops"` matched it as a substring. A proposal
    #   trying to build a roofline was refused that way — **the checker
    #   banned what it had itself allowed.** The same class as D-37 (failing
    #   `inspect.getsource` was turned into "it uses hw"): a defect of the
    #   checker looks like a failure of the LLM (principle 8).
    #
    #   ⚠️ D-163: it came back. The scan ran over `ast.unparse(tree)`, which
    #   keeps **the docstring**, and an axis whose docstring said "needed to
    #   execute the tiled grid" was refused for `'exec'`. `"import"` is in
    #   "important" and `"random"` is in "randomly" — prose is full of them.
    #   So the scan now looks at **names the code actually uses**: `Name`
    #   ids, attribute names, argument names. Prose cannot trip it, and
    #   `np.random.rand` still does (the attribute is `random`).
    names_used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names_used.add(node.id)
        elif isinstance(node, ast.Attribute):
            names_used.add(node.attr)
        elif isinstance(node, ast.arg) or isinstance(node, ast.keyword) and node.arg:
            names_used.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            raise FeatureRejected("banned reference: 'import' (§3)")
    allowed = {n for names in RAW_FIELDS.values() for n in names}
    for b in _BANNED:
        hit = next((u for u in sorted(names_used)
                    if u not in allowed and b in u), None)
        if hit is not None:
            raise FeatureRejected(
                f"banned reference: {b!r}"
                + (f" in {hit!r}" if hit != b else "") + " (§3)")
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


#: ★ How many shapes the duplication comparison is measured on (D-163).
#:
#: It was **4**, and the table's first four are all `M=1`. A shape-level axis
#: then has four distinct values across the whole comparison set, and
#: unrelated axes hit 1.000 by accident: measured on the human registry,
#: 6 pairs cross the 0.95 line at 4 shapes, 7 at 8, 5 at 12 and **4 at both
#: 24 and 66**. `log_flops` and `is_memory_bound` appear only at 4~8 and are
#: gone by 24 — that is the resolution, not a duplicate.
#:
#: 12 is the smallest count in that sweep that already gives the answer 24
#: and 66 give for the config-level axes, and `_spread_shapes` makes sure the
#: 12 are not all one M. ⚠️ It is **not** the whole table: a partial matrix
#: over 12 shapes is what makes the check cheap enough to run on every
#: candidate (D-161).
DUP_SHAPES = 12


def _spread_shapes(table, n: int) -> list:
    """`n` shapes with **M spread out** (D-163).

    The first `n` of the table are all `M=1` — the sweep above is what that
    produced. They are taken one per distinct M, largest group first, so the
    set covers the range instead of one corner.
    """
    shapes = list(table.shapes())
    by_m: dict = {}
    for p in shapes:
        by_m.setdefault(p.M, []).append(p)
    out: list = []
    while len(out) < n and any(by_m.values()):
        for m in sorted(by_m):
            if by_m[m] and len(out) < n:
                out.append(by_m[m].pop(0))
    # ★ The order follows the table so the set is reproducible.
    return sorted(out, key=shapes.index)


def _reference_columns(table, matrix, extra: FeatureRegistry,
                       n_shapes: int = DUP_SHAPES) -> ReferenceColumns:
    """The reference columns for the duplication verdict. It looks at
    **both what a human wrote and what has already been built.**

    Asking for a new axis and then not checking duplication against what was
    already built leads to the same thing being repeated under another name.

    ⚠️ 2026-09-11 (D-161): this was **the cost of adding an axis.** It built
    a second `FeatureMatrix` over the whole registry and **every shape in
    the table** while only ever reading `n_shapes` of them — and in the loop
    `extra` *is* `matrix.registry`, so the entire matrix was recomputed to
    obtain columns the caller already held. Two changes: a registry already
    covered by `matrix` is skipped, and what must be built is built for
    those `n_shapes` alone.
    """
    out: dict[str, list] = {}
    shapes = _spread_shapes(table, n_shapes)
    extra_mat = None
    if extra._items and extra is not matrix.registry:
        # Only what the caller's matrix does not already hold.
        missing = FeatureRegistry(f"{extra.name}-missing")
        for n in extra._items:
            if not matrix.has_column(n):
                missing.add(extra[n])
        if missing._items:
            extra_mat = FeatureMatrix(table, missing, shapes=shapes)
            extra = missing
    for reg, mat in ((matrix.registry, matrix), (extra, extra_mat)):
        if mat is None:
            continue
        for p in shapes:
            fe, info = mat.for_shape(p)
            for n in reg._items:
                f = reg[n]
                v = (np.full(int(info.n_candidates), float(getattr(info, n)))
                     if f.shape_level else np.asarray(getattr(fe, n), float))
                out.setdefault(n, []).append(v)
    ref = ReferenceColumns({n: np.concatenate(v) for n, v in out.items()})
    ref.shapes = tuple(shapes)
    return ref


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
    # ★ D-163: the registry goes in so a duplication refusal can name what
    #   the candidate collides with, in that axis's own words.
    rep = validate_feature(f, table, probe, hw_alt=hw_alt, others=others,
                           registry=registry)
    if rep.failed:
        raise FeatureRejected(
            f"{name}: §8.3 validation failed — "
            + "; ".join(f"{c.name}: {c.detail}" for c in rep.fails()))
    # ★ The shape-level verdict (§30.12). The generation path has no
    #   `shape_feature` decorator, so **everything was being registered as
    #   config level.** Then a rule cannot branch with `if p.<x>:`, and a
    #   term that is constant within a shape cannot change the ranking at
    #   all (absolute rule 2). 5 of F1's 21 were in that state (D-65).
    # ★ The probe already holds this column over every shape — the
    #   verdict reads it instead of computing it a second time (D-161).
    is_shape, why = detect_shape_level(f, table, matrix=probe)
    if is_shape:
        f = replace(f, shape_level=True)
        # ★ The reason is recorded. In particular one that "references cfg
        #   yet is constant in this table" **must be re-judged when the
        #   bundle changes.** It is a frozen dataclass, so it goes into a
        #   module-level table — the caller writes it into summary.json.
        SHAPE_LEVEL_REASON[name] = why
    registry.add(f)
    # ★ The caller's matrix takes the column the probe already computed
    #   (D-161). Without this the same values are computed a third time by
    #   `invalidate`.
    if matrix is not None and matrix.registry is registry \
            and matrix.table is table:
        if f.shape_level:
            # The probe computed it as a column; the caller's matrix wants
            # the scalar. One shape-level pass over the probe, not a third
            # full column.
            probe.registry.add(f, replace=True)
            probe.invalidate(name)
        matrix.adopt(probe, name)
    return f
