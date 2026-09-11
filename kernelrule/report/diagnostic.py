"""The diagnostic report — **the engine of this system** (§12).

It is the only place an LLM can beat an existing optimisation algorithm.
Given only a scalar score it becomes a poor numerical optimiser.

## Five blocks

    1    hardware facts       injected fixed every time. Recalled from
                              memory it comes out wrong
    2    the current rule code in full
    3    per-regime regret breakdown  ★ size stratification first (§30.5)
    3.5  table-structure observations  patterns individual cases never show
    4    10~15 cases          ★ the core. The pick and the optimum side by side
    5    the failure history  without it the same idea returns every 3 rounds

## ★ How it is validated (§12.4)

**A human reads it before the prompt is finished.** If these five blocks
alone make "what would I fix" come to mind, the report is good. If not,
**fix the report, not the prompt.**

## ★ The report does not write its conclusions in advance. It computes them

    forbidden  **nailing** "X differs greatly from Y" or "A dominates" into
               the template
    allowed    rendering a computed result as prose (measuring which is
               larger and building the sentence)

Why this rule is needed: we stepped on it. Block 3 had "size stratification
moves more than difficulty stratification" written in advance, and **the
actual numbers on the training split (M<=2048) were the opposite** (0.0007 vs
0.1651). That split has only 9 long shapes, so the size gap disappears.
**When the report contradicts its own data, the LLM believes the sentence,
not the data.**

★ This rule **does not apply to the hardware-facts block.** Those are
physical constants, not observations of this split. `hardware_block()` is the
only exception.

`tests/test_diagnostic.py` checks the rendered text for uncomputed
comparative words.

## Never put in (§12.3)

    the whole table          it does not fit in tokens, and it memorises the
                             moment it goes in
    holdout scores           put them in and it optimises against the holdout
    the optimum of every     given all of them it transcribes them into
    shape                    conditionals

**Structural enforcement:** `DiagnosticReport` takes only `train_shapes`.
There is no path by which the validation or final split can come in (§10.2).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

import numpy as np

from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.numerics import approx_equal
from kernelrule.core.scoring import Evaluation, evaluate_scores, geomean
from kernelrule.core.splits import Split, SplitError
from kernelrule.core.table import PerfTable
from kernelrule.core.types import Problem
from kernelrule.report.table_facts import TableFacts

__all__ = ["DiagnosticReport", "Case", "Regime", "build_report"]

#: ⚠️ 2026-09-10 (D-156): **empty on purpose.** It used to hold eight bands
#: — memory/compute, `t_sol` 0.5 ms, wave counts, `K<=1024` — and Block 3
#: printed the regret of each. Every one of them is **an axis we chose**, and
#: the biggest number in the report was the SOL gap (+0.4307 at D-155). The
#: Analyst read it every round, the hypotheses pointed there, and the rules
#: branched there: the best rule of D-155 starts at `p.log_sol_ms < -5` and
#: then drifts to `log_flops > 20 / 40 / 30`.
#:
#: The report now gives **cases** and lets the model find the pattern.
#: ⚠️ The price is real: "the short shapes are +0.43 worse" was one line and
#: is now something to infer from nine cases. It may not be seen at all.
#: That is what D-156's run is measuring.
REGIMES: tuple[tuple[str, str], ...] = ()


@dataclass
class Regime:
    name: str
    n_shapes: int
    regret: float
    worst_shape: str = ""
    worst_regret: float = float("nan")


@dataclass
class Case:
    """One case. **Showing the pick and the optimum side by side is the
    core** (§12.1)."""

    shape: tuple
    regime: str
    kind: str                 # "worst" | "best"
    regret: float
    difficulty: float
    best_ms: float
    noise_floor: float
    n_answers: int
    n_candidates: int
    picked: dict              # config axes + feature values
    optimum: dict
    #: The measured top 5 (config summary, ms, how many σ from the optimum)
    neighbors: list[tuple]
    #: (feature name, picked value, optimum value, is the rule using it)
    feature_rows: list[tuple]
    #: ★ How many times the noise floor the gap between the pick and the
    #: optimum is. Below 1, the ranking does not exist as a measurement on
    #: that shape (§30.2).
    gap_sigma: float = float("nan")


@dataclass
class DiagnosticReport:
    run_id: str
    hw_block: str
    rule_code: str
    rule_weights: list[float]
    overall: dict
    regimes: list[Regime]
    table_facts: TableFacts | None
    cases: list[Case]
    failures: list[dict] = field(default_factory=list)
    hypotheses_applied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        return _render(self)

    def token_estimate(self) -> int:
        """A rough token count. §12.2's budget is ~5,500."""
        return len(self.render()) // 3


# ---------------------------------------------------------------------------
# Block 1 — the hardware facts (§12.1). **Recalled from memory it comes out
# wrong**
# ---------------------------------------------------------------------------
def hardware_block(hw, noise) -> str:
    return textwrap.dedent(f"""\
        GPU: {hw.name} ({hw.arch})
          {hw.sm_count} SMs / smem {hw.smem_per_block}B per block
          up to {hw.max_threads_per_sm} threads / {hw.regs_per_sm} registers per SM
          L2 {hw.l2_bytes // (1 << 20)}MB
          effective {hw.peak_tflops_f16} TFLOP/s / {hw.bandwidth_gbps} GB/s
          ridge point {hw.ridge_point:.1f} FLOP/byte

        Execution model:
          CTAs are distributed across SMs; on the last wave some SMs idle.
          A tile computes everything it covers, even outside the shape — a
            128-row tile on M=1 wastes 99.2% of the work.
          split-K divides K to create more tiles, adding a reduction cost.
          serial split-K round-trips D in fp16 per partition (precision loss).
          parallel split-K writes M*N*sk partials to DRAM and reads them back.
          stages=2 (MmaPipelined) and stages>=3 (multistage) are different
            kernel families.
          If alignment does not reach 16 bytes, cp.async is unavailable and
            only 2 stages are possible.

        Limits of measurement:
          ★ 2026-09-11 (D-166): the two percentages used to be **written
            into the string** (7.3% / 0.08%) while the tick above them came
            from the bundle. They are the A6000's — on the 5090 the first is
            wrong by 64x and on the H100 by 32x, and the §29.5 (c) 5090 run
            received exactly that. The same sentence D-117 fixed in
            `hwprompt.py`; this copy was not swept (principles 2 · 23).
            ⛔ The example lengths stay fixed at 14us / 1.3ms — replacing
            them with this table's minimum would put an answer-derived
            number into the Analyst prompt, which is what D-166 §E removed
            from the RuleWriter prompt.
          Time is only recorded in units of the CUDA event timer's tick
            ({noise.tick_ms * 1000:.3f}us). Smaller differences **cannot be
            distinguished by measurement.**
          The shorter the kernel, the larger that tick is relatively —
            one tick is {noise.tick_pct(0.014):.2%} at 14us, \
{noise.tick_pct(1.3):.3%} at 1.3ms.""")


# ---------------------------------------------------------------------------
# Block 3 — the per-regime breakdown. ★ Size comes first
# ---------------------------------------------------------------------------
def _regime_masks(table: PerfTable, matrix: FeatureMatrix,
                  shapes) -> dict[str, np.ndarray]:
    import math

    # ★ A regime is a property of (shape, hardware), **not a property of
    #   the feature list.** This used to read `info.log_sol_ms` /
    #   `info.is_memory_bound`, which assumed those two features were in the
    #   registry. The F0/F1 registries do not have them, so the whole report
    #   dies (§30.9). `regime_of` calls the **functions** in `physical.py`
    #   directly, so it is condition-independent and the verdict lives in
    #   one place (principle 2).
    from kernelrule.core.splits import regime_of

    small, mem, waves = [], [], []
    for p in shapes:
        small.append(regime_of(p, table.hw, axis="size") == "short")
        mem.append(regime_of(p, table.hw, axis="roofline") == "mem")
        gm = math.ceil(p.M / 128) * math.ceil(p.N / 128)
        waves.append(gm / table.hw.sm_count)
    small = np.asarray(small)
    mem = np.asarray(mem)
    w = np.asarray(waves, dtype=np.float64)
    k = np.asarray([p.K for p in shapes])
    return {"small": small, "large": ~small, "mem": mem, "comp": ~mem,
            "wlt1": w < 1.0, "w14": (w >= 1.0) & (w < 4.0), "wgt8": w > 8.0,
            "smallk": k <= 1024}


def _regimes(ev: Evaluation, masks: dict, shapes) -> list[Regime]:
    out = []
    r1 = ev.regret[:, 0]
    for name, key in REGIMES:
        m = masks[key]
        if not m.any():
            continue
        i = int(np.argmax(np.where(m, r1, -np.inf)))
        p = shapes[i]
        out.append(Regime(name=name, n_shapes=int(m.sum()),
                          regret=geomean(r1[m]),
                          worst_shape=f"{p.M}x{p.N}x{p.K}",
                          worst_regret=float(r1[i])))
    return sorted(out, key=lambda r: -r.regret)


# ---------------------------------------------------------------------------
# Block 4 — the cases. **Diversity is enforced** (§12.1)
# ---------------------------------------------------------------------------
_CASE_AXES = ("tile_m", "tile_n", "tile_k", "split_k", "split_k_mode",
              "ext_warp_m", "ext_warp_n", "ext_stages", "ext_swizzle_n",
              "ext_swizzle_type", "kernel_id")


def _cfg_summary(row: dict) -> str:
    sw = ("id" if row.get("ext_swizzle_type") == "identity" else "hz") + \
        str(row.get("ext_swizzle_n", "?"))
    return (f"tb{row['tile_m']}x{row['tile_n']}x{row['tile_k']} "
            f"w{row.get('ext_warp_m','?')}x{row.get('ext_warp_n','?')} "
            f"st{row.get('ext_stages','?')} sw{sw} "
            f"sk{row['split_k']}{row['split_k_mode'][:3]}")


def _sigma(t_pick: float, t_opt: float, noise) -> float:
    """How many times the noise floor the gap is.

    One number solves both problems — tick ties and general resolution
    alike. The LLM can judge for itself whether to care about a case.

    ⚠️ **It is not used to select cases.** Short shapes tend to give a small
    σ, and that is exactly where the room is concentrated (§30.5). It is
    only displayed.
    """
    denom = t_opt * float(noise.floor(t_opt))
    return (t_pick - t_opt) / denom if denom > 0 else float("inf")


def _make_case(table: PerfTable, matrix: FeatureMatrix, p: Problem,
               order: np.ndarray, regret: float, regime: str,
               kind: str, used_features: frozenset = frozenset(),
               n_feats: int = 8) -> Case:
    df = table.frame_for(p).reset_index(drop=True)
    t = np.asarray(table.times_of(p))
    st = table.stats(p)
    pick = int(order[0])
    # ★ "the optimum" depends on the tie-break, so a **deterministic
    #   representative among the ties** is used, and the neighbouring configs
    #   are shown alongside so it is clear whether it is sharp or broad
    #   (§12.1).
    cand = table.candidates(p)
    best_mask = t <= st.best_ms * (1.0 + 1e-12)
    opt = int(np.flatnonzero(best_mask)[
        np.argmin(cand.tiebreak[best_mask])])

    f, _ = matrix.for_shape(p)
    rows = []
    for name in matrix.feature_names():
        v = getattr(f, name)
        if not approx_equal(float(v[pick]), float(v[opt])):
            rows.append((name, float(v[pick]), float(v[opt])))
    rows.sort(key=lambda r: -abs(r[1] - r[2]) / (abs(r[2]) + 1e-9))
    # ★ "the term is missing" and "the term is there but the weight is
    #   wrong" are **different fixes** — an addition versus an adjustment.
    #   The verdict is made from the AST of the rule source.
    rows = [(n, a, b, n in used_features) for n, a, b in rows[:n_feats]]

    top5 = np.argsort(t, kind="mergesort")[:5]
    seen, neigh = set(), []
    for i in top5:
        r = df.iloc[int(i)].to_dict()
        cs = _cfg_summary(r)
        if cs in seen:
            continue
        seen.add(cs)
        neigh.append((cs, float(t[i]),
                      _sigma(float(t[i]), st.best_ms, table.noise)))

    return Case(
        shape=(p.M, p.N, p.K), regime=regime, kind=kind, regret=regret,
        difficulty=st.difficulty, best_ms=st.best_ms,
        noise_floor=st.noise_floor, n_answers=st.n_answers,
        n_candidates=st.n_candidates,
        picked={**{k: df.iloc[pick][k] for k in _CASE_AXES if k in df},
                "ms": float(t[pick])},
        optimum={**{k: df.iloc[opt][k] for k in _CASE_AXES if k in df},
                 "ms": float(t[opt])},
        neighbors=neigh, feature_rows=rows,
        gap_sigma=_sigma(float(t[pick]), float(t[opt]), table.noise))


def _select_cases(table, matrix, order_of, ev, masks, shapes,
                  used_features: frozenset = frozenset(),
                  n_worst: int = 8, n_best: int = 2) -> list[Case]:
    """The worst n by regret + 2 **well-matched cases** (§12.1).

    "the top 15 by regret" is a bad choice — the same failure mode repeated
    15 times carries only one piece of information. And **showing failures
    only breaks what was working.**

    ⚠️ 2026-09-10 (D-156): it used to take the worst **two per regime** over
    eight bands, and each case was labelled with its band. Both the selection
    and the label told the model which axis to look along. What is left is
    the diversity rule that does not name an axis: **no two cases with the
    same (picked, optimum) pair.**
    """
    del masks
    r1 = ev.regret[:, 0]
    cases, used = [], set()
    # ★ If the same (pick, optimum) pair repeats, several cases carry only
    #   one piece of information. §12.1's "enforce diversity" applies to
    #   **failure modes**.
    seen_modes: set[tuple[str, str]] = set()

    def _mode(c: Case) -> tuple[str, str]:
        return (_cfg_summary(c.picked), _cfg_summary(c.optimum))

    for raw_i in np.argsort(-r1):
        if len(cases) >= n_worst:
            break
        i = int(raw_i)
        if i in used or r1[i] <= 1.0 + 1e-9:
            continue
        used.add(i)
        c = _make_case(table, matrix, shapes[i], order_of(shapes[i]),
                       float(r1[i]), "", "worst", used_features)
        sig = _mode(c)
        if sig in seen_modes:
            continue   # the same failure mode. It adds no information
        seen_modes.add(sig)
        cases.append(c)
    n_added = 0
    for raw_i in np.argsort(r1):
        if n_added >= n_best:
            break
        i = int(raw_i)
        if i in used:
            continue
        c = _make_case(table, matrix, shapes[i], order_of(shapes[i]),
                       float(r1[i]), "", "best", used_features)
        sig = _mode(c)
        if sig in seen_modes:
            continue
        used.add(i)
        seen_modes.add(sig)
        cases.append(c)
        n_added += 1
    return cases


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_report(*, run_id: str, table: PerfTable, matrix: FeatureMatrix,
                 score_fn, weights, code: str, train: Split,
                 table_facts: TableFacts | None = None,
                 failures: list[dict] | None = None,
                 hypotheses_applied: list[str] | None = None,
                 notes: list[str] | None = None) -> DiagnosticReport:
    """★ `train` takes **the training split only** (§10.2, §12.3).

    There is no path by which the validation or final split can enter the
    report. If a holdout score can go into the prompt, it ends up being
    tuned against.
    """
    if not isinstance(train, Split) or train.role != "train":
        raise SplitError(
            "the diagnostic report takes the training split only (§10.2). "
            "There is no path by which a holdout score enters the prompt.")
    # ★ This used to be a list of free strings and **bypassed the check
    #   above entirely** — block 3.5 of the first real run was computed on
    #   the full table. §12.3 blocked only "scores" and aggregates slipped
    #   through (D-28).
    if table_facts is not None and not isinstance(table_facts, TableFacts):
        raise SplitError(
            "table_facts is built only through "
            "TableFacts.compute(table, train) (§12.3). Accepting free "
            "strings bypasses the training-split check — aggregates do not "
            "cross the holdout either.")

    shapes = list(train.shapes)
    # ★ The `f.<name>`s are extracted from the rule source with the AST
    #   (reusing A-1's checker). It saves the LLM from having to infer "the
    #   term is missing" versus "the weight is wrong".
    from kernelrule.rules.checks import check_rule

    used = frozenset(check_rule(
        code, feature_names=matrix.feature_names(),
        shape_value_names=matrix.shape_value_names(),
        n_weights=len(weights)).features_used)

    from kernelrule.core.weights import make_score_of

    so = make_score_of(score_fn, matrix, weights)
    ev = evaluate_scores(so, table, shapes, ks=(1, 3, 5, 10), label=run_id)
    masks = _regime_masks(table, matrix, shapes)

    def order_of(p):
        cand = table.candidates(p)
        return cand.top_k(so(p, cand), 5)

    overall = {f"regret@{k}": ev.at(k) for k in ev.ks}
    overall.update({f"hit@{k}": ev.hit_rate(k) for k in ev.ks})
    overall.update(ev.stratified(1))
    overall["size_gap@1"] = ev.size_gap(1)
    overall["difficulty_gap@1"] = ev.difficulty_gap(1)
    # ★ Aggregates on the archive axis (the roofline) (D-145). The summary
    #   shows these first.
    import numpy as _np
    _m = _np.asarray(masks["mem"], dtype=bool)
    if _m.any() and not _m.all():
        overall["mem"] = ev.at(1, mask=_m)
        overall["comp"] = ev.at(1, mask=~_m)
        overall["n_mem"] = int(_m.sum())

    return DiagnosticReport(
        run_id=run_id,
        hw_block=hardware_block(table.hw, table.noise),
        rule_code=code.strip(), rule_weights=[float(x) for x in weights],
        overall=overall, regimes=_regimes(ev, masks, shapes),
        table_facts=table_facts,
        cases=_select_cases(table, matrix, order_of, ev, masks, shapes,
                            used_features=used),
        failures=list(failures or []),
        hypotheses_applied=list(hypotheses_applied or []),
        notes=list(notes or []))


def _render(r: DiagnosticReport) -> str:
    L: list[str] = []
    add = L.append
    add(f"# Diagnostic report — {r.run_id}")
    if r.notes:
        add("")
        for n in r.notes:
            add(f"> {n}")
    add("")
    add("## Block 1 — hardware facts")
    add("```")
    add(r.hw_block)
    add("```")

    add("")
    add("## Block 2 — current rule")
    add(f"The weights were fitted by the numerical optimiser: "
        f"{np.round(r.rule_weights, 3).tolist()}")
    add("```python")
    add(r.rule_code)
    add("```")
    if r.hypotheses_applied:
        add("")
        add("The current rule already reflects these hypotheses:")
        for h in r.hypotheses_applied:
            add(f"  {h}")

    add("")
    add("## Block 3 — how the rule is doing overall")
    o = r.overall
    add("```")
    add(f"overall regret@1 {o['regret@1']:.4f}  (@3 {o['regret@3']:.4f}  "
        f"@5 {o['regret@5']:.4f}  @10 {o['regret@10']:.4f})")
    # ★ Look at regret and hit side by side. When they disagree the nature of
    #   the error is different.
    add(f"hit rate hit@1 {o['hit@1']:.3f}  hit@3 {o['hit@3']:.3f}  "
        f"(a hit = within 2 sigma of the noise floor from the optimum)")
    add("  Low regret with hit 0 means **not a near miss but structurally")
    add("  pointing elsewhere** — that needs a term, not a weight change.")
    add(f"  over {int(o['n_shapes'])} shapes")
    add("```")
    # ⚠️ 2026-09-10 (D-156): the per-band breakdown that stood here is gone.
    #   Eight bands, all of them axes we chose; the model read them and
    #   branched on them. The cases below are what is left, and finding the
    #   pattern in them is the model's job.

    if r.table_facts is not None:
        add("")
        add("## Block 3.5 — table structure observations")
        add("Patterns that a single case never shows. "
            "**Computed on the training split only** (§12.3).")
        add("```")
        for f in r.table_facts.lines:
            add(f)
        add("```")

    add("")
    add("## Block 4 — cases")
    add("**Picked vs optimal, side by side.** The neighbouring configs tell")
    add("you whether the optimum is sharp or wide — wide does not mean you")
    add("must hit it exactly.")
    for i, c in enumerate(r.cases, 1):
        add("")
        # ⚠️ 2026-09-10 (D-156): the band label that stood here
        #   (`[t_sol >= 0.5ms (long)]`) is gone — it named our axis on every
        #   single case.
        add(f"### Case #{i}  {c.shape[0]}x{c.shape[1]}x{c.shape[2]}  "
            f"{'★ good match' if c.kind == 'best' else ''}")
        add("```")
        add(f"rule picked: {_cfg_summary(c.picked):46s} -> "
            f"{c.picked['ms']*1000:9.2f}us  (regret {c.regret:.3f})")
        add(f"actual best: {_cfg_summary(c.optimum):46s} -> "
            f"{c.optimum['ms']*1000:9.2f}us")
        add("")
        add(f"gap = **{c.gap_sigma:.1f}x** the noise floor"
            + ("   <- inside the noise. This shape has no ordering"
               if c.gap_sigma < 1.0 else ""))
        add("")
        if c.feature_rows:
            add(f"{'feature (largest diff first)':28s} {'picked':>12} "
                f"{'best':>12}  in rule")
            for name, a, b, in_rule in c.feature_rows:
                add(f"{name:28s} {a:12.4f} {b:12.4f}  "
                    + ("in use" if in_rule else "★ unused"))
            add("")
        add("top 5 measured for this shape (multiples of the noise floor "
            "from the optimum):")
        for cs, ms, sg in c.neighbors:
            tag = "(optimal)" if sg <= 1e-9 else f"+{sg:.1f} sigma"
            add(f"  {ms*1000:9.2f}us  {tag:>12s}  {cs}")
        add("")
        add(f"difficulty {c.difficulty:.2f}   noise floor "
            f"{c.noise_floor*100:.3f}%   "
            f"indistinguishable answers {c.n_answers}/{c.n_candidates}")
        add("```")

    if r.failures:
        add("")
        add("## Block 5 — failure history")
        add("**Do not repeat the same idea.**")
        add("```")
        for f in r.failures:
            add(f"r{f.get('round','?'):<4} {f.get('verdict','?'):12s} "
                f"{f.get('regret_before','?')} -> {f.get('regret_after','?')}"
                f"   {f.get('idea','')}")
        add("```")
    return "\n".join(L)
