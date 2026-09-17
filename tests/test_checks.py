"""The static checks (§8.3 + appendix §8.1). **Every adversarial case must
be caught** (§24.3)."""
from __future__ import annotations

import pytest

from kernelrule.rules.checks import CheckReport, RuleCheckError, check_rule

FEAT = {"tail_waste", "smem_pressure", "waves", "has_spill", "edge_waste",
        "traffic_amplification", "sm_idle_cost", "split_k_cost",
        "pipeline_warmup_frac"}
SHAPE = {"is_memory_bound", "arith_intensity", "roofline_ratio",
         "M", "N", "K"}


def chk(code: str, n_weights: int = 1) -> CheckReport:
    return check_rule(code, feature_names=FEAT, shape_value_names=SHAPE,
                      n_weights=n_weights)


def test_well_formed_rule_passes():
    code = ("def score(f, p, hw, w):\n"
            "    s = f.tail_waste * w[0]\n"
            "    s = s + f.smem_pressure * w[1]\n"
            "    if p.is_memory_bound:\n"
            "        s = s + f.edge_waste * w[2]\n"
            "    return s + np.where(f.has_spill > 0, w[3], 0.0)\n")
    r = chk(code, 4)
    assert r.ok, r.violations
    assert r.features_used == {"tail_waste", "smem_pressure", "edge_waste",
                               "has_spill"}


def test_weight_indices_do_not_eat_the_literal_budget():
    """★ The `0` of `w[0]` is not a literal — weights are already counted
    through `n_weights`."""
    code = ("def score(f, p, hw, w):\n"
            "    return (f.tail_waste * w[0] + f.waves * w[1]"
            " + f.has_spill * w[2] + f.edge_waste * w[3])\n")
    r = chk(code, 4)
    assert r.n_literals == 0 and r.n_weights == 4 and r.ok


# ---------------------------------------------------------------------------
# The adversarial cases of §24.3 — one of them passing means a hole in the
# defence
# ---------------------------------------------------------------------------
ADVERSARIAL = [
    # ★ The reason changed from "a direct comparison" to "by equality"
    #   (D-144). Inequalities are allowed; equality is still the
    #   memorisation route.
    ("memorisation", ("def score(f, p, hw, w):\n"
             "    if p.M == 4096:\n"
             "        return f.waves * w[0]\n"
             "    return f.waves * w[1]\n"), 2, "by equality"),
    ("answer leak", ("def score(f, p, hw, w):\n"
                  "    return f.waves * w[0] + time_ms\n"), 1, "banned name"),
    ("difficulty reference", ("def score(f, p, hw, w):\n"
                    "    return f.waves * w[0] * difficulty\n"), 1,
     "banned name"),
    ("table access", ("def score(f, p, hw, w):\n"
                "    return TABLE[0] * w[0]\n"), 1, "banned name"),
    ("infinite loop source", ("def score(f, p, hw, w):\n"
                       "    while f.waves > 0:\n"
                       "        pass\n"
                       "    return f.waves * w[0]\n"), 1, "config-level"),
    ("sandbox escape", ("def score(f, p, hw, w):\n"
                      "    import os\n"
                      "    return f.waves * w[0]\n"), 1, "import"),
    ("typo", ("def score(f, p, hw, w):\n"
             "    return f.tail_wast * w[0]\n"), 1, "unregistered feature"),
    ("if on an array", ("def score(f, p, hw, w):\n"
                  "    if f.waves < 1:\n"
                  "        return f.waves * w[0]\n"
                  "    return f.waves * w[0]\n"), 1, "config-level"),
    ("ternary on an array", ("def score(f, p, hw, w):\n"
                    "    return (w[0] if f.waves < 1 else w[0]) * f.waves\n"),
     1, "config-level"),
    ("non-determinism", ("def score(f, p, hw, w):\n"
                 "    return np.random.rand(3) * w[0]\n"), 1, "numpy"),
    ("dunder detour", ("def score(f, p, hw, w):\n"
                  "    return f.waves * w[0] + score.__globals__['x']\n"),
     1, "dunder"),
    ("w slicing", ("def score(f, p, hw, w):\n"
                   "    return f.waves * w[0] + sum(w[1:])\n"), 2,
     "constant index"),
    ("w whole", ("def score(f, p, hw, w):\n"
                 "    return f.waves * len(w)\n"), 1, "w whole"),
    ("syntax error", "def score(f, p, hw, w)\n    return 1\n", 1,
     "parse failure"),
    ("several functions", ("def helper():\n    return 1\n"
                     "def score(f, p, hw, w):\n"
                     "    return f.waves * w[0]\n"), 1, "exactly one"),
    ("changed signature", ("def score(problem, hw, candidates):\n"
                      "    return 0.0\n"), 1, "signature"),
    ("comprehension", ("def score(f, p, hw, w):\n"
                   "    return sum([x for x in f.waves]) * w[0]\n"),
     1, "comprehension"),
    ("unregistered shape value", ("def score(f, p, hw, w):\n"
                      "    if p.secret_difficulty:\n"
                      "        return f.waves * w[0]\n"
                      "    return f.waves * w[0]\n"), 1, "shape-level"),
]


@pytest.mark.parametrize("name,code,nw,expect",
                         ADVERSARIAL, ids=[c[0] for c in ADVERSARIAL])
def test_adversarial_case_is_rejected(name, code, nw, expect):
    r = chk(code, nw)
    assert not r.ok, f"{name} passed — there is a hole in the defence"
    assert any(expect in v for v in r.violations), \
        f"{name}: not the expected reason ({expect}) but {r.violations}"


def test_a_long_flat_rule_is_accepted():
    """★ 2026-09-09 (D-150): 9 parameters on one path used to be refused.

    The cap is gone, so the same rule passes. What still refuses is the path
    count, weight reuse and index holes — those have their own tests.
    """
    code = ("def score(f, p, hw, w):\n"
            "    return f.waves * w[0] + f.tail_waste * w[1]\n")
    assert chk(code, 2).ok
    long_one = ("def score(f, p, hw, w):\n"
                "    return (f.waves*w[0] + f.tail_waste*w[1]"
                " + f.traffic_amplification*w[2]"
                " + f.has_spill*w[3] + f.sm_idle_cost*w[4] + f.split_k_cost*w[5]"
                " + f.edge_waste*w[6] + f.smem_pressure*w[7]"
                " + f.pipeline_warmup_frac*w[8])\n")
    r = check_rule(long_one, feature_names=FEAT, shape_value_names=SHAPE,
                   n_weights=9)
    assert r.ok, r.violations
    assert r.parameters_used == 9      # reported, not enforced


def test_sparse_weight_indices_are_rejected():
    """★ A hole in the indices is a **free parameter** (D-144).

    It is a hole opened by the move to a per-path budget — under the old
    combined budget, `len(W0)` itself entered the budget and blocked it
    automatically. The fitter fits unused indices too.
    """
    code = ("def score(f, p, hw, w):\n"
            "    return f.waves * w[0] + f.tail_waste * w[8]\n")
    r = check_rule(code, feature_names=FEAT, shape_value_names=SHAPE,
                   n_weights=9)
    assert not r.ok
    assert any("unused weight indices" in v for v in r.violations)


def test_unused_weights_are_rejected():
    code = "def score(f, p, hw, w):\n    return f.waves * w[0]\n"
    r = chk(code, 3)
    assert not r.ok and any("largest referenced index" in v
                            for v in r.violations)


def test_parse_failure_is_rejection_not_pass():
    """★ A parse failure is a **refusal**, not a pass (§26.4)."""
    r = chk("this is not python", 1)
    assert not r.ok
    with pytest.raises(RuleCheckError):
        r.raise_if_bad()


def test_shape_level_if_is_allowed():
    """A shape-level branch is **allowed**. That is the kind that
    generalises."""
    code = ("def score(f, p, hw, w):\n"
            "    s = f.waves * w[0]\n"
            "    if p.is_memory_bound:\n"
            "        s = s + f.edge_waste * w[1]\n"
            "    return s\n")
    assert chk(code, 2).ok


def test_human_guided_rule_obeys_the_same_constraints():
    """★ The human baseline is under **the same constraints** as a rule
    (§9.4).

    The comparison is only fair under the same conditions. ⚠️ 2026-09-09
    (D-150): the parameter-count assertion went with the cap; what is left is
    every constraint that still refuses.
    """
    import kernelrule.features.physical  # noqa: F401  registration
    from kernelrule.features import REGISTRY
    from kernelrule.rules.human_guided import CODE, W0

    r = check_rule(CODE, feature_names=REGISTRY.names(shape_level=False),
                   shape_value_names=REGISTRY.names(shape_level=True),
                   n_weights=len(W0))
    assert r.ok, r.violations


# ---------------------------------------------------------------------------
# A-1 — the no-op warning on shape-level branches (not a refusal)
# ---------------------------------------------------------------------------
NOOP = [
    ("scalar multiply", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                  "    if p.is_memory_bound:\n        s = s * w[1]\n"
                  "    return s\n"), 2),
    ("scalar add", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                    "    if p.is_memory_bound:\n        s = s + w[1]\n"
                    "    return s\n"), 2),
    ("literal multiply", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                  "    if p.is_memory_bound:\n        s = s * 2.0\n"
                  "    return s\n"), 1),
    ("hw constant", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                "    if p.is_memory_bound:\n        s = s / hw.sm_count\n"
                "    return s\n"), 1),
    ("AugAssign", """def score(f, p, hw, w):
    s = f.waves * w[0]
    if p.is_memory_bound:
        s *= w[1]
    return s
""", 2),
]


@pytest.mark.parametrize("name,code,nw", NOOP, ids=[c[0] for c in NOOP])
def test_noop_shape_branch_warns_but_passes(name, code, nw):
    """★ Multiplying or adding a shape constant into the running score does
    not change the ranking within that shape.

    It is **a warning, not a refusal** — it is syntactically legal and the
    general case cannot be caught. The real verdict is made by the scorer
    and the §12 diagnostic report.
    """
    r = chk(code, nw)
    assert r.ok, f"{name} must not be refused (it should be a warning)"
    assert r.warnings, f"{name} is a no-op yet there is no warning"
    assert "cannot change the ranking" in r.warnings[0]


VALID_BRANCH = [
    ("term reweight", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                  "    if p.is_memory_bound:\n"
                  "        s = s + f.edge_waste * w[1]\n    return s\n"), 2),
    ("AugAssign term", ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
                     "    if p.is_memory_bound:\n"
                     "        s += f.edge_waste * w[1]\n    return s\n"), 2),
    ("no branch", "def score(f, p, hw, w):\n    return f.waves * w[0]\n", 1),
]


@pytest.mark.parametrize("name,code,nw", VALID_BRANCH,
                         ids=[c[0] for c in VALID_BRANCH])
def test_meaningful_shape_branch_is_not_warned(name, code, nw):
    """A branch that **changes the weight** of a config-level term is not
    warned about."""
    r = chk(code, nw)
    assert r.ok and not r.warnings, r.warnings


def test_warnings_do_not_affect_ok():
    """A warning does not affect `ok`. That is what keeps evolution from
    stalling."""
    code = ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
            "    if p.is_memory_bound:\n        s = s * w[1]\n    return s\n")
    r = chk(code, 2)
    assert r.ok and len(r.warnings) == 1
    r.raise_if_bad()      # it must not raise


# ---------------------------------------------------------------------------
# ★ Going around the literal budget — the hole a real LLM broke through
# (§29.4)
# ---------------------------------------------------------------------------
def test_weight_reuse_is_allowed_and_counted():
    """★ 2026-09-10 (D-156): reusing `w[i]` across terms **is allowed**.

    The refusal existed to stop a rule going around the parameter budget
    (a rule really did build 19 terms out of 8 weights). There is no budget
    since D-150, so there is nothing to go around, and `f.a * w[0] + f.b *
    w[0]` is a structural claim — "these two carry the same weight".

    ⚠️ What must not break: `n_terms` counts **uses**, because it is the
    archive's size axis now (`len(w0)` would read 2 for this rule).
    """
    code = ("def score(f, p, hw, w):\n"
            "    s = f.waves * w[0]\n"
            "    s = s + f.tail_waste * w[0]\n"
            "    s = s + f.has_spill * w[1]\n"
            "    return s\n")
    r = chk(code, 2)
    assert r.ok, r.violations
    assert r.n_terms == 3 and r.n_weights == 2


def test_term_count_is_reported():
    code = ("def score(f, p, hw, w):\n"
            "    s = f.waves * w[0]\n"
            "    return s + f.tail_waste * w[1]\n")
    r = chk(code, 2)
    assert r.ok and r.n_terms == 2


def test_shape_branch_reweighting_is_still_allowed():
    """★ Reweighting, giving the same feature a **different** weight, is
    legitimate.

    Adding "refuse when the same expression appears twice" false-positived
    on this pattern. This is exactly the form §A-1 recommends.
    """
    code = ("def score(f, p, hw, w):\n"
            "    s = f.traffic_amplification * w[0]\n"
            "    if p.is_memory_bound:\n"
            "        s = s + f.traffic_amplification * w[1]\n"
            "    return s\n")
    r = chk(code, 2)
    assert r.ok, r.violations


def test_interaction_term_is_allowed():
    """A product of features is real physics (an interaction). It is not
    blocked."""
    code = ("def score(f, p, hw, w):\n"
            "    s = f.waves * w[0]\n"
            "    return s + f.has_spill * f.smem_pressure * w[1]\n")
    assert chk(code, 2).ok


def test_the_rule_that_reused_weights_now_passes():
    """The rule that came out of a real run and was refused for reuse.

    ⚠️ 2026-09-10 (D-156): kept as a regression the other way round — it
    passes now, and its six terms are counted as six.
    """
    code = ("def score(f, p, hw, w):\n"
            "    s = np.log2(f.traffic_amplification) * w[0]\n"
            "    s = s + f.sm_idle_cost * w[1]\n"
            "    s = s + f.smem_pressure * w[2]\n"
            "    s = s + f.has_spill * w[3]\n"
            "    s = s + f.edge_waste * w[0]\n"
            "    s = s + f.waves * w[1]\n"
            "    return s\n")
    r = chk(code, 4)
    assert r.ok, r.violations
    assert r.n_terms == 6 and r.n_weights == 4


def test_human_guided_rule_uses_one_weight_per_term():
    """Pins as a regression that the human baseline satisfies the new
    rules."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.features import REGISTRY
    from kernelrule.rules.human_guided import CODE, W0

    r = check_rule(CODE, feature_names=REGISTRY.names(shape_level=False),
                   shape_value_names=REGISTRY.names(shape_level=True),
                   n_weights=len(W0))
    assert r.ok, r.violations
    assert r.n_terms == len(W0), (
        f"terms {r.n_terms} != weights {len(W0)}")


# ---------------------------------------------------------------------------
# D-78 — branch comparison constants are taken out of the budget
# ---------------------------------------------------------------------------

def test_branch_comparison_constant_is_free():
    """★ The `1` of `p.roofline_ratio < 1` does not count against the
    budget (D-78).

    The combined budget blocked even physical constants, so evolution
    avoided using `1` and went around — `np.square(x) < x`,
    `x < np.sqrt(x)`, `x < np.sign(x)`, `np.isfinite(x)`. All four equal
    `x < 1` and are hard for a human to read.
    """
    code = ("def score(f, p, hw, w):\n"
            "    s = np.where(p.roofline_ratio < 1, f.waves, f.tail_waste) * w[0]\n"
            "    s = s + f.edge_waste * w[1]\n"
            "    return s\n")
    r = check_rule(code, feature_names=FEAT,
                   shape_value_names=SHAPE | {"roofline_ratio"}, n_weights=2)
    assert r.ok, r.violations
    assert r.n_literals == 0, "a comparison constant entered the budget"
    assert r.branch_constants == [1], (
        "the exempted constant was not recorded")


def test_non_comparison_constant_still_costs():
    """A constant inside a nested expression still costs — only comparison
    operands are exempt."""
    code = ("def score(f, p, hw, w):\n"
            "    s = np.where((f.waves - 3.0) < 1, f.waves, f.tail_waste) * w[0]\n"
            "    return s\n")
    r = check_rule(code, feature_names=FEAT,
                   shape_value_names=SHAPE | {"roofline_ratio"}, n_weights=1)
    assert r.n_literals == 1, f"the 3.0 was not counted: {r}"
    assert r.branch_constants == [1]


def test_shape_size_equality_is_banned_but_inequality_is_not():
    """★ 2026-09-08 (D-144): **only equality** is blocked.

    ```
    p.M == 4096   ⛔ memorises a single point
    p.M < 128     ★ cuts a band — a form that generalises
    ```

    The old rule blocked both together ("a direct comparison against a shape
    size"), which left the model unable to express the notion of "a small M"
    at all.
    """
    eq = ("def score(f, p, hw, w):\n"
          "    s = np.where(p.M == 4096, f.waves, f.tail_waste) * w[0]\n"
          "    return s\n")
    r = check_rule(eq, feature_names=FEAT, shape_value_names=SHAPE,
                   n_weights=1)
    assert not r.ok
    assert any("by equality" in v for v in r.violations), r.violations

    lt = eq.replace("p.M == 4096", "p.M < 1024")
    assert check_rule(lt, feature_names=FEAT, shape_value_names=SHAPE,
                      n_weights=1).ok


def test_the_path_counter_agrees_at_the_llm_boundary():
    """★ The two counters must not diverge (the D-37 family).

    ⚠️ 2026-09-10 (D-152): this compared "static count > cap" with "the LLM
    boundary refuses". There is no cap, so what is left to agree on is the
    **path count** — the one thing both still refuse.
    """
    from kernelrule.rules.checks import MAX_PATHS, literal_parameter_message

    ok = "def score(f, p, hw, w):\n    return f.waves * w[0]\n"
    many = ("def score(f, p, hw, w):\n    s = f.waves * w[0]\n"
            + "".join(f"    if p.{g}:\n        s = s + f.tail_waste * w[{i}]\n"
                      for i, g in enumerate(
                          ("is_memory_bound", "can_use_cp_async",
                           "roofline_ratio"), start=1))
            + "    return s\n")
    for code, nw in ((ok, 1), (many, 4)):
        r = check_rule(code, feature_names=FEAT,
                       shape_value_names=SHAPE | {"roofline_ratio",
                                                  "can_use_cp_async"},
                       n_weights=nw)
        over_static = r.n_paths > MAX_PATHS
        over_llm = literal_parameter_message(code, nw) is not None
        assert over_static == over_llm, (
            f"the two counters diverged: paths {r.n_paths}/{MAX_PATHS} vs "
            f"boundary {over_llm}\n{code}")


# ---------------------------------------------------------------------------
# D-92 — manufacturing a constant by an identity transform is blocked
# ---------------------------------------------------------------------------
IDENTITY = [
    ("isfinite",
     ("def score(f, p, hw, w):\n"
     "    return np.nan_to_num(np.isfinite(f.tail_waste)\n"
     "                         / (np.isfinite(f.tail_waste) - f.tail_waste)) * w[0]\n")),
    ("sign in a comparison",
     ("def score(f, p, hw, w):\n"
     "    return np.where(p.roofline_ratio < np.sign(p.roofline_ratio),\n"
     "                    f.waves, f.tail_waste) * w[0]\n")),
    ("x < sqrt(x)",
     ("def score(f, p, hw, w):\n"
     "    return np.where(p.roofline_ratio < np.sqrt(p.roofline_ratio),\n"
     "                    f.waves, f.tail_waste) * w[0]\n")),
    ("square(x) < x",
     ("def score(f, p, hw, w):\n"
     "    return np.where(np.square(p.roofline_ratio) < p.roofline_ratio,\n"
     "                    f.waves, f.tail_waste) * w[0]\n")),
]


@pytest.mark.parametrize(("name", "code"), IDENTITY,
                         ids=[c[0] for c in IDENTITY])
def test_identity_transform_is_rejected(name, code):
    """★ An identity transform that manufactures a constant is a **defect**
    (D-92).

    All four equal a constant or a simple comparison mathematically, yet
    they are hard for a human to read. "Interpretable rules" is this
    project's claim, so they eat away at it.
    """
    from kernelrule.rules.checks import identity_transform_message

    msg = identity_transform_message(code)
    assert msg, f"{name} is not caught"
    # ★ The alternative must be stated too. Saying only what is forbidden
    #   makes it invent yet another detour.
    assert "exempt" in msg and "D-78" in msg
    r = check_rule(code, feature_names=FEAT,
                   shape_value_names=SHAPE | {"roofline_ratio"}, n_weights=1)
    assert not r.ok, f"{name}: the message appears but it is not refused"


LEGIT = [
    ("literal comparison",
     ("def score(f, p, hw, w):\n"
     "    return np.where(p.roofline_ratio < 1, f.waves, f.tail_waste) * w[0]\n")),
    ("legitimate sqrt",
     "def score(f, p, hw, w):\n    return np.sqrt(f.waves) * w[0]\n"),
    ("legitimate square",
     ("def score(f, p, hw, w):\n    return np.square(f.reg_pressure) * w[0]\n")),
]


@pytest.mark.parametrize(("name", "code"), LEGIT, ids=[c[0] for c in LEGIT])
def test_legitimate_uses_are_not_rejected(name, code):
    """★ `sqrt` / `square` themselves are legitimate — it is a defect only
    **when compared against the same argument**.

    Blocking them wholesale would forbid non-linear transforms (§30.10's
    "1/(1-x) and log2(x) are different forms of the same physical
    quantity").
    """
    from kernelrule.rules.checks import identity_transform_message

    assert identity_transform_message(code) is None, (
        f"{name} is a false positive")


def test_identity_check_catches_the_real_archive():
    """★ Check that it catches things in reverse (the D-39 family).

    Synthetic cases alone only show that "the checker we wrote catches the
    cases we wrote". It has to catch **rules evolution actually produced**.
    """
    import json
    from pathlib import Path

    from kernelrule.rules.checks import identity_transform_message

    root = Path(__file__).resolve().parents[1] / "runs"
    codes = []
    for f in sorted(root.glob("*/archive.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                codes.append(json.loads(ln)["code"])
    if not codes:
        pytest.skip("no runs/ (gitignored) — it cannot run right after a "
                    "clone")
    hit = sum(1 for c in codes if identity_transform_message(c))
    assert hit > 0, ("it catches nothing in the archive — meaning the "
                     "checker does not see the real detours")


# ---------------------------------------------------------------------------
# ★ D-163 — a refusal has to say what to do instead
# ---------------------------------------------------------------------------
def test_the_wrong_prefix_is_named_as_such():
    """★ 7 of 10 RuleWriter tries and one loop proposal of the D-162 run
    died on `f.X` where X is a shape value, and the retry message said only
    "unregistered" — three times, and the model repeated it three times.
    The checker holds both lists; it can say which side the name is on."""
    from kernelrule.rules.checks import check_rule, limits_for

    r = check_rule("def score(f, p, hw, w):\n    return f.roof * w[0]\n",
                   feature_names=["waves"], shape_value_names=["roof"],
                   n_weights=1, limits=limits_for())
    assert not r.ok
    assert "p.roof" in r.violations[0] and "shape-level" in r.violations[0]

    r = check_rule("def score(f, p, hw, w):\n    s = 0.0\n"
                   "    if p.waves > 1.0:\n        s = s + w[0]\n    return s\n",
                   feature_names=["waves"], shape_value_names=["roof"],
                   n_weights=1, limits=limits_for())
    assert not r.ok
    assert "f.waves" in r.violations[0] and "config-level" in r.violations[0]


def test_the_w0_length_message_says_the_number():
    """★ Five proposals of the D-162 run missed it in **both** directions
    (53 vs 52, 54 vs 55, 57 vs 58, 50 vs 51). Nobody counts fifty weights by
    hand. ⚠️ It is not corrected for the model — only stated."""
    from kernelrule.rules.checks import check_rule, limits_for

    r = check_rule("def score(f, p, hw, w):\n"
                   "    return f.waves * w[0] + f.waves * w[3]\n",
                   feature_names=["waves"], shape_value_names=[],
                   n_weights=3, limits=limits_for())
    v = next(x for x in r.violations if "W0 length" in x)
    assert "must hold exactly 4" in v and "you sent 3" in v
