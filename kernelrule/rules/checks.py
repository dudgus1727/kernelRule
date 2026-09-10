"""Static checks on LLM-generated rules (§8.3 + the updated appendix
§8.1).

**A parse failure is a refusal, not a pass** (§26.4). Every path in this file
leans towards failure.

## What it blocks

    memorisation             `if problem.M == 4096: ...`
    conditional specialising `if f.waves < 1: s += 5.0`  <- a config-level branch
    answer references        `time_ms`, `difficulty`, `TABLE`
    escapes                  import, open, exec, attribute detours
                             (`__globals__`)
    overfitting              numeric literals + weight count <= 8

## What it allows

    shape-level branches  `if p.is_memory_bound:`   <- a scalar, so it generalises
    np operations         where/clip/minimum/log/sqrt/... and arithmetic
    weights               `w[0]`, `w[1]` — **index access only**

`f.*` is an array, so an `if` raises `ValueError` at runtime, but by then the
rule has already run. The AST catches it **before execution**.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

__all__ = ["FITTER_SWITCH_DIM", "MAX_PATHS", "fitter_for", "CheckReport", "RuleCheckError", "check_rule", "LIMITS",
           "weight_reuse_message", "literal_parameter_message",
           "exponent_message", "exponent_indices", "weight_bounds",
           "EXPONENT_BOUNDS",
           "identity_transform_message", "noop_term_message"]


def noop_term_message(code: str) -> str | None:
    """Finds terms that only add a shape constant to the running score.
    They do not change the ranking.

    ★ A rule is **sorted independently per shape**, so adding or multiplying
    a shape-level scalar into the whole score is a monotone transform and
    changes no ordering at all. Such a term simply throws away one unit of
    budget.

    It is syntactically legal, so it runs and raises nothing — it **silently
    does nothing.** That is why this check is needed (§26.4). The first
    successful rule under RuleWriter condition A threw away one term as
    `p.log_sol_ms * w[0]`.

    A shape-level value **multiplied with `f.*`** does mean something — then
    it changes the weight of a config-level term by shape, so the ranking
    changes.
    """
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None
    bad_terms: list[str] = []

    def walk(body):
        for st in body:
            if isinstance(st, ast.If):
                walk(st.body)
                walk(st.orelse)
                continue
            if not isinstance(st, (ast.Assign, ast.AugAssign)):
                continue
            val = st.value
            # Only the <expr> of `s = s + <expr>` or `s += <expr>`
            if isinstance(st, ast.Assign) and isinstance(val, ast.BinOp) \
                    and isinstance(val.op, ast.Add):
                expr = val.right
            elif isinstance(st, ast.AugAssign) and isinstance(st.op, ast.Add):
                expr = val
            else:
                continue
            if not _uses_weight(expr):
                continue
            if not _touches_features(expr):
                bad_terms.append(ast.unparse(expr))

    walk(tree.body[0].body)
    if not bad_terms:
        return None
    return (f"there is a term with no effect on the ranking: {bad_terms}. "
            "A rule is sorted independently per shape, so **adding a "
            "shape-level value to the whole score changes no ordering at "
            "all** — it throws away one weight. Use a shape-level value to "
            "branch with `if p.<name>:` and change the weight of a "
            "config-level term, or multiply it with `f.*`")


def _uses_weight(node) -> bool:
    return any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
               and n.value.id == "w" for n in ast.walk(node))


def _numeric_literals(tree: ast.AST) -> tuple[list[ast.Constant],
                                              list[ast.Constant]]:
    """Splits numeric literals into **those that count against the budget
    and those that do not** (D-78).

    Returns `(counted, comparison constants)`.

    ## Why split them

    §29.4 tied two different things to one budget.

    ```
    the weight limit    caps the parameter count — for structural comparison
    the literal limit   was meant to block "hardcoding constants to fit this
                        table"
    ```

    `p.roofline_ratio < 1` is not the latter — it is a **physical constant**,
    the knee of the roofline. Yet it hit the combined budget and evolution
    avoided using `1`, going around it instead:

    ```
    np.square(x) < x        x < np.sqrt(x)        x < np.sign(x)
    np.isfinite(x)          <- 9 times in one rule. Purely to use the
                               constant 1
    ```

    **All four equal `x < 1`, and they are hard for a human to read.**
    "Interpretable rules" is this project's claim, and the budget was eating
    that claim.

    ## What is exempt

    Only numeric literals that are **direct operands** of an `ast.Compare`. A
    constant inside a nested expression is not exempt — in `(f.x - 3) < 1`,
    the `3` counts.

    ## What is still forbidden

    A direct comparison against a shape size (`p.M > 1024`) is **refused
    regardless of the exemption**. That check is done separately by
    `check_rule`. The dividing line is **whether that constant comes from
    the hardware/physics or from this table's shape distribution**, and that
    cannot be decided statically, so **every exempted constant is recorded**
    for a human to look at (`CheckReport.branch_constants`).
    """
    skip = {id(n.slice) for n in ast.walk(tree)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
            and n.value.id == "w"}
    exempt: set[int] = set()
    def _direct_const(side):
        """A numeric constant that is a **direct operand** of a comparison.
        ★ Negatives are caught too (D-144).

        `-1.0` is a `UnaryOp(USub, Constant)`, so looking only for
        `Constant` misses it, and then the `-1.0` of
        `p.log_sol_ms < -1.0` enters the budget.
        """
        if isinstance(side, ast.UnaryOp) and isinstance(
                side.op, (ast.USub, ast.UAdd)):
            side = side.operand
        if (isinstance(side, ast.Constant)
                and isinstance(side.value, (int, float))
                and not isinstance(side.value, bool)):
            return side
        return None

    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            for side in [n.left, *n.comparators]:
                c = _direct_const(side)
                if c is not None:
                    exempt.add(id(c))
    counted: list[ast.Constant] = []
    branch: list[ast.Constant] = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Constant)
                and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool)):
            continue
        if id(n) in skip:
            continue          # the 0 of w[0] is counted on the weight side
        (branch if id(n) in exempt else counted).append(n)
    return counted, branch


def _branch_depth(node: ast.AST) -> int:
    """The maximum nesting depth of `if` statements. `IfExp` (the ternary)
    is not counted.

    ⚠️ **It is not used for the verdict** (D-145). Depth does not bound the
    path count — `if/elif/elif/else` is depth 3 with 4 paths, while 4
    sequential `if`s are depth 1 with 16 paths. It is kept for reporting
    only.
    """
    if isinstance(node, ast.If):
        return 1 + max((_branch_depth(x)
                        for x in [*node.body, *node.orelse]), default=0)
    return max((_branch_depth(c) for c in ast.iter_child_nodes(node)),
               default=0)


def _path_use(node: ast.AST, counted_ids: set[int]) -> tuple[set, set]:
    """The (counted literal ids, weight indices) this fragment uses."""
    lits: set[int] = set()
    ws: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and id(n) in counted_ids:
            lits.add(id(n))
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id == "w" and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, int)):
            ws.add(n.slice.value)
    return lits, ws


def _paths(stmts, counted_ids: set[int], *,
           cap: int | None = None) -> list[tuple[set, set]]:
    """★ Per execution path, the (set of literal ids, set of weight
    indices) (D-144).

    ```
    a common term outside if/else   ★ belongs to every path
    np.where                        ★ not a branch — both sides are computed,
                                       so it is one path
    ```

    ★ With `cap`, it **stops building** the moment the paths exceed that
    number (D-145). Building the full Cartesian product and then refusing
    produces 16,384 entries for 14 sequential `if`s — 228ms spent on
    something that will be refused. Only the fact that the cap was exceeded
    matters, so it returns holding at most `cap + 1`.
    """
    paths: list[tuple[set, set]] = [(set(), set())]
    for st in stmts:
        if isinstance(st, ast.If):
            hl, hw = _path_use(st.test, counted_ids)
            subs = _paths(st.body, counted_ids, cap=cap) + (
                _paths(st.orelse, counted_ids, cap=cap) if st.orelse
                else [(set(), set())])
            out: list[tuple[set, set]] = []
            for pl, pw in paths:
                for sl, sw in subs:
                    out.append((pl | hl | sl, pw | hw | sw))
                    if cap is not None and len(out) > cap:
                        return out          # ★ early stop
            paths = out
        else:
            al, aw = _path_use(st, counted_ids)
            paths = [(pl | al, pw | aw) for pl, pw in paths]
        if cap is not None and len(paths) > cap:
            return paths
    return paths


#: ★ The cap on the number of execution paths (D-145). 4 paths -> at most
#: 32 effective parameters.
#:
#: ⚠️ The old value was `MAX_BRANCH_DEPTH = 2` (D-144). **Depth does not
#: bound the path count** — in the Python AST an `elif` is an `If` inside
#: `orelse`, so it eats depth:
#:
#: ```
#: nested if/else, 2 levels   depth 2  4 paths    passed
#: if/elif/elif/else          depth 3  4 paths    ⛔ refused, same 4 paths
#: 4 sequential ifs           depth 1  16 paths   ⛔ depth did not catch it
#: ```
#:
#: **The syntax split identical path counts.** The paths are counted
#: directly.
MAX_PATHS = 4

#: Calls that are **always 1** when the argument is finite. They can only
#: be used to manufacture a constant.
#: ⚠️ "it is finite in this table" is a **table-dependent fact** (confirmed
#: across all 19 f features, D-78). On another table it may genuinely
#: discriminate, so the message says **"risks being treated as a
#: constant"** rather than "always 1".
_CONST_CALLS = ("isfinite",)


def identity_transform_message(code: str) -> str | None:
    """★ Is a constant being manufactured by an identity transform (D-92)?

    ```
    np.isfinite(x)                      always 1 when finite
    np.sign(x)                          always 1 when x > 0
    x < np.sqrt(x) / np.square(x) < x   both equal x < 1
    ```

    **This is a defect, not explainability.** All four equal a constant or a
    simple comparison mathematically, yet they are hard for a human to read,
    and they eat away at the claim of "interpretable rules".

    ## Why block it now

    While the combined budget was blocking literals, there **was a reason to
    go around** — using 8 terms required 0 literals. D-78 removed branch
    comparison constants from the budget and so removed that reason, and
    literal comparisons then appeared in 6/6 runs (D-84). **It is blocked
    after the reason is gone** — in the other order, another detour is found.

    The refusal message states **the alternative** too. Saying only what is
    forbidden makes the model invent yet another detour.
    """
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None
    bad: list[str] = []
    for n in ast.walk(tree):
        # np.isfinite(x) — a call that manufactures the constant 1
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "np"
                and n.func.attr in _CONST_CALLS):
            bad.append(f"np.{n.func.attr}(...) — when the argument is "
                       "finite this risks being treated as the constant 1")
        # np.sign(x) on one side of a comparison is effectively the literal 1
        if isinstance(n, ast.Compare):
            for side in [n.left, *n.comparators]:
                if (isinstance(side, ast.Call)
                        and isinstance(side.func, ast.Attribute)
                        and isinstance(side.func.value, ast.Name)
                        and side.func.value.id == "np"
                        and side.func.attr == "sign"):
                    bad.append("np.sign(...) is used in a comparison — on "
                               "positive input it equals `< 1`")
            # x < np.sqrt(x) / np.square(x) < x  — both equal x < 1
            for lo, hi in zip([n.left, *n.comparators][:-1],
                              [n.left, *n.comparators][1:], strict=False):
                if _same_arg_transform(lo, hi, "sqrt") \
                        or _same_arg_transform(hi, lo, "square"):
                    bad.append("`x < np.sqrt(x)` / `np.square(x) < x` — "
                               "both equal `x < 1`")
    if not bad:
        return None
    uniq = sorted(set(bad))
    return ("a constant is being manufactured by an identity transform "
            "(D-92): " + " / ".join(uniq)
            + ". ★ There is no need — **comparison constants in branch "
              "conditions are exempt from the budget** (D-78). Write the "
              "number plainly, as in `p.<shape value> < 1`. The reader has "
              "to be able to see the physical boundary")


def _same_arg_transform(a, b, fn: str) -> bool:
    """Are `a` and `np.<fn>(a)` the same expression?"""
    if not (isinstance(b, ast.Call) and isinstance(b.func, ast.Attribute)
            and isinstance(b.func.value, ast.Name) and b.func.value.id == "np"
            and b.func.attr == fn and len(b.args) == 1):
        return False
    try:
        return ast.unparse(a) == ast.unparse(b.args[0])
    except Exception:                                       # noqa: BLE001
        return False


def literal_parameter_message(code: str, n_weights: int,
                           *, parameters: int | None = None) -> str | None:
    """A message when the rule breaks a **counting** rule, else `None`.

    ⚠️ 2026-09-09 (D-150): the parameter cap is gone, so what is left here is
    the execution-path count. `parameters` is accepted and ignored — the
    callers pass it and removing it from all four surfaces at once is how
    D-105/D-107 happened. It is dropped in one place, here.

    ★ Split out for the same reason as `weight_reuse_message` — the retry
    has to happen at the LLM boundary for the model to hear what was wrong.
    While this check lived only in the static stage, three RuleWriter
    proposals in a row were discarded for the same reason and the model never
    learned "with 8 weights you cannot use a literal".

    ★ The counting is done by `_numeric_literals` alone — if `check_rule`
    and this counted **separately they would diverge** (the D-37 family).
    Then the LLM boundary lets it through and the static check silently
    throws it away.
    """
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None
    counted, _branch = _numeric_literals(tree)
    fnode = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fnode is None:
        return None
    # ★ The same `_paths` as `check_rule` — counted separately they would
    #   diverge (the D-37 family).
    paths = _paths(fnode.body, {id(n) for n in counted}, cap=MAX_PATHS)
    if len(paths) > MAX_PATHS:
        return (f"there are more than {MAX_PATHS} execution paths. "
                f"**At most {MAX_PATHS}** are allowed — beyond that the "
                f"paths explode and the rule becomes a lookup table. Two "
                f"levels of nesting, `if/elif/elif/else`, and two sequential "
                f"`if`s are all 4 paths")
    return None


#: ★ The bounds on an exponent-slot weight (D-112). **Normalisation, not a
#: hyperparameter** — not a value put there to be tuned but a physical cap
#: that prevents divergence and overflow.
#:
#:   lower 0   A negative exponent is `inf` at `f.* == 0`. And every feature
#:             is "larger is worse", so a negative exponent flips the
#:             direction (§8.2).
#:   upper 4   A penalty growing faster than the 4th power of a normalised
#:             feature is not a smooth physical response but effectively a
#:             threshold, and thresholds are already done by `np.where`. So
#:             **no expressiveness is taken away.**
#:
#: The values are not swept. Sweeping them makes it a hyperparameter from
#: that moment on.
EXPONENT_BOUNDS = (0.0, 4.0)


def _exponent_sites(tree) -> list[tuple[int, object]]:
    """`(weight index, base node)` — `np.power(base, w[i])` and
    `base ** w[i]`."""
    out = []
    for n in ast.walk(tree):
        base = expo = None
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            base, expo = n.left, n.right
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "power" and len(n.args) == 2):
            base, expo = n.args
        if expo is None:
            continue
        if (isinstance(expo, ast.Subscript)
                and isinstance(expo.value, ast.Name) and expo.value.id == "w"
                and isinstance(expo.slice, ast.Constant)
                and isinstance(expo.slice.value, int)):
            out.append((int(expo.slice.value), base))
    return out


def exponent_indices(code: str) -> frozenset[int]:
    """The indices of the `w[i]` used in an exponent slot."""
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return frozenset()
    return frozenset(i for i, _ in _exponent_sites(tree))


def weight_bounds(code: str, n_weights: int) -> list[tuple] | None:
    """Per-weight bounds. **`None` when there is no exponent slot** —
    existing rules are unchanged.

    ★ Returning `None` matters. Always attaching bounds would silently make
    even old runs that use no exponent a different condition (principle 36).
    """
    idx = exponent_indices(code)
    if not idx:
        return None
    lo, hi = EXPONENT_BOUNDS
    inf = float("inf")
    return [(lo, hi) if i in idx else (-inf, inf)
            for i in range(n_weights)]


def exponent_message(code: str,
                     feature_mins: dict | None = None) -> str | None:
    """The **numerical guard** on the exponent slot. A message on
    violation, else `None`.

    ```
    in np.power(x, w)
      x < 0 and w not an integer   nan
      x = 0 and w < 0              inf
      x large and w large          overflow
    ```

    The first two are blocked **by constraining the base** — the base must
    be a **single** `f.<name>` whose declared range minimum is 0 or above.
    The third is blocked by the exponent cap `EXPONENT_BOUNDS`.

    ★ Split out for the same reason as `weight_reuse_message` — the retry
    has to happen at the LLM boundary for the model to hear what was wrong.
    """
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            e = n.right
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "power" and len(n.args) == 2):
            e = n.args[1]
        else:
            continue
        ok_w = (isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name)
                and e.value.id == "w")
        if not ok_w and _touches_weight(e):
            return ("only a **single** `w[i]` may sit in the exponent "
                    "slot. Raising an expression to the exponent leaves no "
                    "way to bound it and it diverges.")
    for _i, base in _exponent_sites(tree):
        if not (isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name)
                and base.value.id == "f"):
            return ("the **base of `np.power` must be a single "
                    "`f.<name>`**. An expression as the base can go "
                    "negative, and then it is `nan` — which silently makes "
                    "the rule useless.")
        if feature_mins is not None:
            m = feature_mins.get(base.attr)
            if m is not None and m < 0:
                return (f"`f.{base.attr}` can be negative (range minimum "
                        f"{m}). As the base of an exponent it becomes "
                        f"`nan`.")
    return None


def _touches_weight(node) -> bool:
    return any(isinstance(m, ast.Name) and m.id == "w"
               for m in ast.walk(node))


def weight_reuse_message(code: str) -> str | None:
    """A message if `w[i]` is reused across terms, else `None`.

    ★ This one was split out **in order to trigger a retry at the LLM
    boundary**. The full check walks the AST heavily and needs the list of
    registered features, whereas reuse can be seen from the code alone. When
    the schema validator calls this, Pydantic AI feeds the message back to
    the model and **has it fixed and resubmitted**.

    While this check lived only in `checks.py`, RuleWriter proposals were
    silently discarded — the model never heard what was wrong and repeated
    the same mistake.
    """
    uses: dict[int, int] = {}
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None       # a syntax error is caught by another check
    for n in ast.walk(tree):
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id == "w" and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, int)):
            uses[n.slice.value] = uses.get(n.slice.value, 0) + 1
    dup = sorted(i for i, c in uses.items() if c > 1)
    if not dup:
        return None
    return (f"a weight is reused across terms: "
            f"{[f'w[{i}]x{uses[i]}' for i in dup]}. You built "
            f"{sum(uses.values())} terms out of {len(uses)} weights — one "
            f"fitted coefficient cannot carry two different physical "
            f"quantities. Use a **different** weight for each term")

#: The dimension at which `fitter_for` switches from Nelder-Mead to CMA-ES.
#:
#: ⚠️ It **was** the parameter budget, and 8 was never validated as one
#: (§29.4; the 8 vs 16 attempt stopped at D-77 because the fitter could not
#: cope in 16 dimensions). D-150 removed the cap; the number survives here
#: only as the fitter boundary, which D-125 and D-123 did measure.
#:
#: ⚠️ 2026-09-10 (D-152): renamed from `PARAMETERS`. It stopped being a cap
#: at D-150 and the old name kept reading like one — this is **the number of
#: fitted dimensions at which the fitter switches**, and nothing else.
#: `docs/glossary.md` carries the mapping.
FITTER_SWITCH_DIM = 8

LIMITS = {
    #: ★ A safety valve against runaway code, **not a size limit**
    #: (D-150 · D-154).
    #:
    #: ⚠️ 800 became the cap the moment the parameter budget went: the first
    #: full 12 rounds refused **17 proposals out of 72** on it, all of them
    #: the large rules (D-153). A parameter costs about 30 nodes, so 800 was
    #: a cap of roughly 27 parameters.
    #:
    #: ⚠️ **3000 is arbitrary too.** It is about 100 parameters at 30 nodes
    #: each. Whether the evolution ever goes that far is exactly what is
    #: being measured — **if this number starts refusing proposals again it
    #: has to be raised again**, and that is a finding, not a nuisance.
    #: Parsing 3000 nodes is milliseconds; real runaway code still stops.
    "ast_nodes": 3000,
    #: ⚠️ Swept at D-154 together with `ast_nodes`: the largest rule of the
    #: 12-round run was 61 lines against this 120, so it is not binding yet.
    #: It is raised in the same proportion so it does not become the next
    #: wall while nobody is looking.
    "max_lines": 400,
}


def fitter_for(n_weights: int | None) -> dict:
    """★ **`len(W0)`** decides the fitter (D-144). Decided in one place
    only.

    ⚠️ **The meaning of the argument changed on 2026-09-08.** It used to be
    the campaign's `parameters` (the per-path budget cap) and is now **that
    rule's `len(W0)`**. With a per-path budget one rule can reach up to 32
    dimensions, so deciding per campaign would fit a 16-dimensional rule with
    Nelder-Mead (a reach rate of 42~58%, D-77).

    ```
    <= 8   nelder-mead / 4 restarts / 200 evals   every run so far
    >  8   cma         / 1 restart  / 300 evals   the arm §2's gate chose (D-123)
    ```

    Rationale: in 8 dimensions the two fitters are indistinguishable (D-125),
    and in 16 dimensions Nelder-Mead's reach rate is 92%, short of the mark
    (D-77 · D-123). The polish budget of 600 is the same on both sides.

    ⚠️ **Some old runs differ from this rule** — `rb08`/`rprod`/`rpow` ran
    with CMA at 8 parameters (D-124). On re-measurement they run under this
    rule.
    """
    b = int(n_weights if n_weights is not None else FITTER_SWITCH_DIM)
    # ★ The key names are exactly `LoopConfig`'s fields — being splattable
    #   as `**fitter_for(n)` is what leaves no place for divergence
    #   (principle 2).
    if b <= FITTER_SWITCH_DIM:
        return {"fit_method": "nelder-mead", "fit_restarts": 4,
                "max_evals": 200}
    return {"fit_method": "cma", "fit_restarts": 1, "max_evals": 300}


def limits_for(parameters: int | None = None) -> dict:
    """The size caps. **They no longer move with anything** (D-150).

    They used to scale with the parameter budget, because leaving the node
    cap at 400 while raising the budget to 16 would have refused 16-term
    rules at the node cap — measuring "16 terms could not be used" instead of
    "a budget of 16 has no effect" (the same spot as D-105).

    ★ With the budget gone the same trap is here: if the node cap stayed at
    400 it would silently **become** the budget. So the standing cap is the
    value the 16-parameter arm ran under.

    `parameters` is accepted and ignored — the callers are unpicked one at a
    time, not all four surfaces at once (that is how D-105/D-107 happened).
    """
    return dict(LIMITS)

#: Names that must not appear in rule source. They are answers or escape
#: routes.
_BANNED_NAMES = frozenset({
    # answers (ANSWER_COLS)
    "time_ms", "time_std_ms", "time_min_ms", "time_max_ms", "n_reps",
    "outlier_frac", "cublas_ms", "tflops", "frac_of_peak", "vs_cublas",
    "difficulty", "distinct_time_frac", "n_distinct_times",
    # values knowable only by measuring (OUTCOME_COLS)
    "status", "max_rel_error", "actual_split_k", "drift_ratio",
    "sm_clock_mhz", "mem_clock_mhz", "gpu_temp_c", "power_w", "timestamp",
    # table access
    "TABLE", "PerfTable", "table", "times_of", "best_time", "answer_mask",
    "load_for_scoring", "load_for_ranking", "load_bundle", "read_parquet",
    # escapes
    "eval", "exec", "compile", "open", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "breakpoint",
})

#: What must not appear in an attribute name (detours such as
#: `x.__globals__`).
_BANNED_ATTR_PREFIX = ("__",)

#: The allowed numpy names. Anything else is refused — `np.random` makes
#: things non-deterministic.
#:
#: ★ 2026-09-09 (D-147): this is **the one list** (principle 2).
#: `core/sandbox.py` imports it for the runtime proxy. It used to keep its
#: own copy and the two had already drifted apart — the runtime allowed
#: `e`/`pi`/`inf`/`float64`, which this gate rejects, so no rule could ever
#: reach them. The gate is unchanged; the runtime is now exactly as wide.
ALLOWED_NP = frozenset({
    "where", "clip", "minimum", "maximum", "log", "log2", "log10", "sqrt",
    "abs", "exp", "power", "sign", "floor", "ceil", "round", "isfinite",
    "nan_to_num", "square", "reciprocal", "logical_and", "logical_or",
    "logical_not", "greater", "less", "equal", "asarray", "zeros_like",
    "ones_like", "full_like", "fmin", "fmax", "hypot", "cbrt",
})

class RuleCheckError(RuntimeError):
    """The rule did not pass the static checks."""


@dataclass
class CheckReport:
    ok: bool
    violations: list[str] = field(default_factory=list)
    #: Not a refusal, but something a human should look at. **It does not
    #: affect `ok`.**
    warnings: list[str] = field(default_factory=list)
    n_literals: int = 0
    n_weights: int = 0
    n_nodes: int = 0
    max_w_index: int = -1
    #: The number of terms a weight is multiplied into. Exceeding
    #: `n_weights` means there is reuse.
    n_terms: int = 0
    features_used: set[str] = field(default_factory=set)
    shape_values_used: set[str] = field(default_factory=set)
    #: ★ The branch comparison constants **exempted** from the budget
    #: (D-78). They are not refused but they are recorded — "a physical
    #: constant or this table's shape distribution" cannot be decided
    #: statically, so a human looks.
    branch_constants: list[float] = field(default_factory=list)
    #: ★ The (literals + weights) count per execution path (D-144). The
    #: budget is counted **per path**.
    path_parameters: list[int] = field(default_factory=list)
    #: The maximum nesting depth of `if`. ⚠️ For reporting — the verdict is
    #: made by `n_paths`.
    branch_depth: int = 0
    #: ★ The number of execution paths (D-145). Beyond the cap it stops at
    #: `MAX_PATHS + 1`.
    n_paths: int = 0

    @property
    def parameters_used(self) -> int:
        """The parameter count of the heaviest path (the global sum when
        there are no paths). ★ 2026-09-09 (D-150): **reported, not
        enforced** — it is now one of the things being measured."""
        return (max(self.path_parameters) if self.path_parameters
                else self.n_literals + self.n_weights)

    def raise_if_bad(self) -> CheckReport:
        if not self.ok:
            raise RuleCheckError(
                "the rule did not pass the static checks:\n  "
                + "\n  ".join(self.violations))
        return self

    def __str__(self) -> str:
        head = "passed" if self.ok else "refused"
        bc = (f", branch constants {self.branch_constants} (exempt)"
              if self.branch_constants else "")
        pp = (f", {self.n_paths} paths {self.path_parameters} "
              f"(depth {self.branch_depth})"
              if len(self.path_parameters) > 1 else "")
        return (f"[{head}] literals {self.n_literals} + weights "
                f"{self.n_weights} "
                f"= heaviest path {self.parameters_used}"
                f"{bc}{pp}, "
                f"terms {self.n_terms}, "
                f"nodes {self.n_nodes}/{LIMITS['ast_nodes']}, "
                f"features {sorted(self.features_used)}"
                + "".join("\n    ✗ " + v for v in self.violations)
                + "".join("\n    ! " + v for v in self.warnings))


def check_rule(code: str, *, feature_names, shape_value_names,
               n_weights: int, limits: dict | None = None,
               feature_mins: dict | None = None) -> CheckReport:
    """Checks the source of `score(f, p, hw, w)`.

    `n_weights` is the length of the `W0` the LLM gave. ⚠️ 2026-09-09
    (D-150): it is no longer summed into a budget — it is used for the
    reuse and index-hole checks and reported.
    """
    lim = {**LIMITS, **(limits or {})}
    rep = CheckReport(ok=True, n_weights=int(n_weights))
    feature_names = set(feature_names)
    shape_value_names = set(shape_value_names)

    def bad(msg: str) -> None:
        rep.ok = False
        rep.violations.append(msg)

    def warn(msg: str) -> None:
        """Not a refusal, but something a human should look at."""
        rep.warnings.append(msg)

    # -- Parsing. A failure is a **refusal** (§26.4) -----------------------
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        rep.ok = False
        rep.violations.append(
            f"parse failure: {e}. That is a refusal, not a pass.")
        return rep

    if code.count("\n") + 1 > lim["max_lines"]:
        bad(f"line count {code.count(chr(10)) + 1} > {lim['max_lines']}")

    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fns) != 1 or fns[0].name != "score":
        bad("there must be exactly one top-level `def score(f, p, hw, w)` "
            f"(found: {[getattr(n, 'name', type(n).__name__) for n in tree.body]})")
        return rep
    fn = fns[0]
    args = [a.arg for a in fn.args.args]
    if args[:4] != ["f", "p", "hw", "w"]:
        bad(f"the signature must be `score(f, p, hw, w)` (got: {args})")

    rep.n_nodes = sum(1 for _ in ast.walk(tree))
    if rep.n_nodes > lim["ast_nodes"]:
        bad(f"AST nodes {rep.n_nodes} > {lim['ast_nodes']}")

    # -- The walk ---------------------------------------------------------
    array_names: set[str] = set()   # local variables derived from f.*
    #: How many times each w[i] was used. Reuse goes around the budget.
    w_index_uses: dict[int, int] = {}
    #: Signatures of the expressions a weight multiplies. For detecting
    #: duplicate terms.
    term_sigs: list[str] = []

    # ★ Literals are counted by `_numeric_literals` **alone** — if the LLM
    #   boundary (`literal_parameter_message`) and this counted separately
    #   they would diverge (the D-37 family). The `0` of `w[0]` is excluded
    #   there: weights already enter the budget through `n_weights`, so
    #   without excluding it each term would be counted twice and the budget
    #   would halve.
    _counted, _branch = _numeric_literals(tree)
    rep.n_literals = len(_counted)
    rep.branch_constants = [n.value for n in _branch]

    for node in ast.walk(tree):
        # no import
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bad("no import. `np` is already given")

        # banned names
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            bad(f"banned name reference: {node.id!r}")
        if isinstance(node, ast.Attribute):
            if node.attr in _BANNED_NAMES:
                bad(f"banned attribute reference: .{node.attr}")
            if node.attr.startswith(_BANNED_ATTR_PREFIX):
                bad(f"no dunder attribute access: .{node.attr}")

        # f.<name> / p.<name>
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base, attr = node.value.id, node.attr
            if base == "f":
                rep.features_used.add(attr)
                if attr not in feature_names:
                    bad(f"unregistered feature: f.{attr}. "
                        "A typo is not waved through silently")
            elif base == "p":
                rep.shape_values_used.add(attr)
                # ★ M/N/K are not registered features but **values of the
                #   shape itself**. Only inequality comparisons are allowed —
                #   equality is refused below (D-144).
                if attr not in shape_value_names and attr not in ("M", "N",
                                                                  "K"):
                    bad(f"unregistered shape-level value: p.{attr}")
            elif base == "hw":
                pass
            elif base == "np":
                if attr not in ALLOWED_NP:
                    bad(f"numpy function not allowed: np.{attr}. "
                        f"allowed: {sorted(ALLOWED_NP)[:8]} ...")

        # A direct comparison against problem.M / p.M — ★ only equality is
        # forbidden (D-144)
        #
        #   p.M == 4096   ⛔ memorises a single point
        #   p.M < 128     ★ cuts a band — a form that generalises
        #
        # The old rule blocked both together. That left the model unable to
        # express the notion of "a small M" at all.
        if isinstance(node, ast.Compare):
            eq_ops = any(isinstance(o, (ast.Eq, ast.NotEq, ast.In, ast.NotIn))
                         for o in node.ops)
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id in ("p", "problem")
                        and sub.attr in ("M", "N", "K") and eq_ops):
                    bad(f"no branching on a shape size compared **by "
                        f"equality**: {sub.value.id}.{sub.attr}. That is the "
                        f"form that memorises a single point — cut a band "
                        f"with an inequality (<, <=, >, >=)")

        # w is accessed by index only
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id == "w":
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, int):
                rep.max_w_index = max(rep.max_w_index, sl.value)
                w_index_uses[sl.value] = w_index_uses.get(sl.value, 0) + 1
            else:
                bad("w is accessed only by a constant index (w[0], w[1] "
                    "...). No slicing, no variable index")
        # Local-variable tracking — if f.* is assigned, that variable is an
        # array too
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            if _touches_features(node.value):
                array_names.add(node.targets[0].id)

    # Collect signatures of the expressions weights multiply (duplicate
    # term detection)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for a, b_ in ((node.left, node.right), (node.right, node.left)):
                if (isinstance(a, ast.Subscript)
                        and isinstance(a.value, ast.Name) and a.value.id == "w"):
                    import contextlib

                    with contextlib.suppress(Exception):
                        term_sigs.append(ast.dump(b_))

    # Using `w` whole, without an index, is forbidden (w.sum(), w * f, ...)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "w":
            if not _is_subscript_base(tree, node):
                bad("do not use w whole. Only index access is allowed "
                    "(§8.1)")
                break

    # -- ★ The exponent-slot guard (D-112) --------------------------------
    #    A base that can go negative is `nan`, and that is not a refusal but
    #    **a silent neutering** — the whole score becomes unusable.
    if (m := exponent_message(code, feature_mins)) is not None:
        bad(m)

    # -- ★ No weight reuse — it blocks going around the literal budget ----
    #
    #   In a real run the LLM broke through like this:
    #
    #       s = s + f.log_workspace_bytes * w[0]
    #       s = s + f.log_dram_traffic    * w[0]      <- w[0] reused
    #       s = s + f.is_two_stage        * w[0]      <- again
    #
    #   It built **17 terms** out of 8 weights. The budget's purpose (§29.4)
    #   is to block "with many parameters any structure reaches a similar
    #   regret and structural comparison becomes meaningless", and growing
    #   the term count without limit destroys that purpose. The
    #   `len(W0) == max_index + 1` check alone cannot catch it.
    dup_w = sorted(i for i, c in w_index_uses.items() if c > 1)
    if dup_w:
        bad(f"a weight is reused across terms: "
            f"{[f'w[{i}]x{w_index_uses[i]}' for i in dup_w]}. "
            f"{sum(w_index_uses.values())} terms on {rep.n_weights} weights "
            f"— that goes around the literal budget (§29.4). Use a different "
            f"weight for each term")
    rep.n_terms = sum(w_index_uses.values())

    # ⚠️ "refuse when the same expression appears twice" is not added. It
    #    **false-positives on legitimate reweighting** — giving the same
    #    feature a different weight under a shape-level branch is exactly the
    #    pattern we want (§A-1):
    #
    #        s = f.traffic * w[0]
    #        if p.is_memory_bound:
    #            s = s + f.traffic * w[6]      # a reweight. Legitimate
    #
    #    The pathology actually observed (the same expression + **the same**
    #    weight twice) is already caught by the reuse check above.

    if rep.max_w_index >= 0 and rep.max_w_index + 1 != rep.n_weights:
        bad(f"W0 length {rep.n_weights} != the largest referenced index + 1 "
            f"({rep.max_w_index + 1}). An unused weight is wasted budget")
    # ★ **There must be no hole in the indices** (D-144).
    #
    #   Under the old combined budget, `len(W0)` itself entered the budget so
    #   holes were blocked automatically. Counting per path, **an unused
    #   index is caught by no path yet the fitter still fits it** — it
    #   becomes a free parameter. A test caught it: a rule using only `w[0]`
    #   and `w[8]` with `len(W0)=9` passed.
    if rep.max_w_index >= 0:
        holes = [i for i in range(rep.max_w_index + 1)
                 if i not in w_index_uses]
        if holes:
            bad(f"the unused weight indices {holes} are in W0. The fitter "
                f"fits those too, so they become **free parameters** — use "
                f"the indices from 0 with no gaps")

    if (m := identity_transform_message(code)):
        bad(m)

    # -- ★ The path count. **There is no parameter cap any more** (D-150) --
    #   The cap existed to keep structures comparable (§29.4), but it made
    #   the intended behaviour unreachable: a rule with 8 common terms had to
    #   throw a term away before it could grow a single branch, so 12 of 12
    #   proposals stayed at `len(w0) = 8` and never split their weights
    #   (D-149 §0). D-108 had already measured 8 vs 16 as indistinguishable
    #   with a **negative** train-holdout gap on 6/6 seeds.
    #   ★ The path count stays — it is a safety valve against path explosion
    #     and check cost, not a budget.
    fnode = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fnode is not None:
        rep.branch_depth = _branch_depth(fnode)
        paths = _paths(fnode.body, {id(n) for n in _counted}, cap=MAX_PATHS)
        rep.n_paths = len(paths)
        #: Reported, not enforced — "how many parameters did it settle on"
        #: is now a **result** (D-150 §1-5).
        rep.path_parameters = sorted(len(a) + len(b) for a, b in paths)
        if rep.n_paths > MAX_PATHS:
            bad(f"there are more than {MAX_PATHS} execution paths. At most "
                f"{MAX_PATHS} are allowed — beyond that the paths explode "
                f"and the rule becomes a lookup table (two levels of "
                f"nesting, if/elif/elif/else, and two sequential ifs are "
                f"all 4 paths)")

    # -- ★ No config-level branching --------------------------------------
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            if _touches_features(node.test, array_names):
                bad("do not branch on a config-level feature — `f.*` is an "
                    "array. Use `np.where(...)` (the 'bad fix' of §2.3)")
        if isinstance(node, ast.comprehension):
            bad("no comprehensions. Write it as a vector operation")

    # -- ★ When a shape-level branch cannot change the ranking (a warning) -
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and not _touches_features(node.test,
                                                              array_names):
            noop = _noop_shape_branch(node, array_names)
            if noop:
                warn(f"a shape-level branch cannot change the ranking: "
                     f"{noop}. Multiplying or adding a shape constant into "
                     f"the whole score is a **monotone transform** within "
                     f"that shape, so the sort comes out the same. It only "
                     f"means something if it **changes the weight** of a "
                     f"config-level term")

    return rep


def _noop_shape_branch(node: ast.If, array_names: set) -> str | None:
    """Does a shape-level `if` block have no effect on the ranking (common
    forms only)?

    ## Why it is needed — we stepped on it

        if p.is_memory_bound:
            s = s * w[2]        # ⛔ the ranking **does not change at all**

    A rule is sorted independently per shape, so a **shape constant
    multiplied or added into the running score cancels out.** It is entirely
    legal syntactically, so no other check catches it. The first version of
    the hand rule carried a dead term because of this.

    ## What it catches

    If **every** statement in the block is of the form
    `s = s <op> <shape constant>` and `op` is `*` or `+` (or `-`, `/`), it is
    a no-op. A shape constant is `p.*` / `hw.*` / `w[i]` / a numeric literal
    and combinations of those.

    ## What it cannot catch

    The general case (going through an intermediate variable, or redefining
    a term conditionally). **It catches only the most common mistake, and it
    is a warning, not a refusal.** The real verdict is made by the scorer —
    "does a shape-level branch change the ranking" in the §12 diagnostic
    report.
    """
    stmts = list(node.body)
    if not stmts:
        return None
    target = None
    for st in stmts:
        if not isinstance(st, (ast.Assign, ast.AugAssign)):
            return None   # no verdict on return / nested if and the like
        if isinstance(st, ast.Assign):
            if len(st.targets) != 1 or not isinstance(st.targets[0], ast.Name):
                return None
            name = st.targets[0].id
            val = st.value
            # It must be of the form `s = s <op> <scalar>`
            if not (isinstance(val, ast.BinOp)
                    and isinstance(val.op, (ast.Mult, ast.Add, ast.Sub,
                                            ast.Div))):
                return None
            left, right = val.left, val.right
            if isinstance(left, ast.Name) and left.id == name:
                other = right
            elif isinstance(right, ast.Name) and right.id == name:
                other = left
            else:
                return None
        else:                     # AugAssign: s *= ...
            if not isinstance(st.target, ast.Name):
                return None
            if not isinstance(st.op, (ast.Mult, ast.Add, ast.Sub, ast.Div)):
                return None
            name, other = st.target.id, st.value
        if target is None:
            target = name
        elif target != name:
            return None
        if not _is_shape_scalar(other, array_names):
            return None
    return (f"only shape constants applied to `{target}` {len(stmts)} times"
            if target else None)


def _is_shape_scalar(node, array_names: set) -> bool:
    """Is this expression only shape-level scalars (`p.*` / `hw.*` /
    `w[i]` / literals)?"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            if isinstance(sub.value, ast.Name) and sub.value.id in ("p", "hw"):
                continue
            return False
        if isinstance(sub, ast.Name):
            if sub.id in ("p", "hw", "w", "np"):
                continue
            if sub.id in array_names:
                return False
            return False
        if isinstance(sub, (ast.Constant, ast.BinOp, ast.UnaryOp,
                            ast.Subscript, ast.Load, ast.Mult, ast.Add,
                            ast.Sub, ast.Div, ast.Pow, ast.USub, ast.Index)):
            continue
        if isinstance(sub, ast.Call):
            return False
    return True


def _touches_features(node, extra: set | None = None) -> bool:
    """Does this expression touch a config-level feature (an array)?"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
                and sub.value.id == "f":
            return True
        if extra and isinstance(sub, ast.Name) and sub.id in extra:
            return True
    return False


def _is_subscript_base(tree, target: ast.Name) -> bool:
    """Is this `Name` node the base of a `w[...]`?"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and node.value is target:
            return True
    return False
