"""Automatic feature validation (§8.3). **It does not swallow exceptions
and approve** (§26.4).

It looks at seven things.

    1. does it run           a finite float on a sample
    2. is the range as declared
    3. does the vectorisation match the scalar   ★ where training (matrix)
                                                 and deployment (scalar)
                                                 diverge
    4. is it constant        std < 1e-9 -> rejected (zero explanatory power)
    5. is it a duplicate     Spearman > 0.95 against an existing feature ->
                             a candidate for deprecation
    6. scale invariance      ★ if hw changes and the value does not, it does
                             not use the hardware
    7. is it useful          the standalone ranking AUC — **not a rejection,
                             only a mark**

**Item 6 is the core.** It is the only automatic check that catches a feature
with a hardcoded hardware constant, and architecture transfer is this
project's main metric, so a leak here collapses the conclusion.

What happens with a result:

    fail   -> rejected. It does not enter the registry
    warn   -> passes but is marked. A human looks (3~5 of 30)
    info   -> recorded only
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kernelrule.core.types import Hardware, config_from_row
from kernelrule.features import Feature, FeatureRegistry

__all__ = ["Check", "ValidationReport", "validate_feature", "validate_registry"]

#: A Spearman correlation above this makes it a duplication candidate
#: (§8.4)
DUP_RHO = 0.95
#: A standard deviation below this counts as constant
CONST_STD = 1e-9
#: If changing `hw` moves the value less than this, it does not use the
#: hardware
SCALE_MIN_CHANGE = 1e-9


@dataclass
class Check:
    name: str
    level: str          # "fail" | "warn" | "info" | "ok"
    detail: str = ""

    def __str__(self) -> str:
        mark = {"fail": "✗", "warn": "!", "info": "·", "ok": "✓"}[self.level]
        return f"{mark} {self.name}" + (f" — {self.detail}" if self.detail else "")


@dataclass
class ValidationReport:
    feature: str
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(c.level == "fail" for c in self.checks)

    @property
    def warned(self) -> bool:
        return any(c.level == "warn" for c in self.checks)

    def fails(self) -> list[Check]:
        return [c for c in self.checks if c.level == "fail"]

    def __str__(self) -> str:
        head = f"[{'rejected' if self.failed else 'passed'}] {self.feature}"
        body = "\n".join("    " + str(c) for c in self.checks
                         if c.level != "ok")
        return head + ("\n" + body if body else "")


def alt_hw(hw: Hardware) -> Hardware:
    """Fake hardware for the scale-invariance check. **It changes every
    numeric field** (D-38).

    ★ This function was **copied separately into three experiment scripts**
    (`f1_pipeline` / `revalidate` / `feature_writer`). The loop needed it too
    and it nearly became a fourth copy — fix one and the rest diverge
    (principle 2).
    """
    from dataclasses import replace

    return replace(hw, sm_count=hw.sm_count * 2 + 3,
                   smem_per_block=int(hw.smem_per_block * 0.7),
                   max_threads_per_sm=int(hw.max_threads_per_sm * 1.33),
                   peak_tflops_f16=hw.peak_tflops_f16 * 1.6,
                   bandwidth_gbps=hw.bandwidth_gbps * 0.8,
                   regs_per_sm=int(hw.regs_per_sm * 1.5),
                   l2_bytes=int(hw.l2_bytes * 2))


def _what_it_is(registry, name: str) -> str:
    """The axis it collides with, **as it already stands in the registry**
    (D-163). Empty when the registry was not handed in."""
    if registry is None or name not in getattr(registry, "_items", {}):
        return ""
    f = registry[name]
    doc = (f.physical_meaning or f.doc or "").strip()
    src = (f.source or "").strip()
    out = f"\n{name} already measures:"
    if doc:
        out += f"\n  {doc[:300]}"
    if src:
        out += "\n" + "\n".join(f"  {ln}" for ln in src.splitlines()[:14])
    return out + ("\nMake something this does not already say, or say why "
                  "the difference matters.")


class ReferenceColumns(dict):
    """The duplication check's comparison set, **with the shapes it was
    measured on** (D-162).

    A plain dict lost that, and `validate_feature` then compared a candidate
    measured on 6 sampled shapes against columns measured on the first 4 —
    different lengths, and the code skipped every pair it could not line up.
    The check was there and never fired once. Carrying the shape list makes
    "measured on the same shapes" checkable instead of assumed.
    """

    #: The shapes every column in here was measured on, in order.
    shapes: tuple = ()


def _sample(table, n_shapes: int, rng) -> list:
    shapes = table.shapes()
    idx = rng.choice(len(shapes), size=min(n_shapes, len(shapes)),
                     replace=False)
    return [shapes[int(i)] for i in sorted(idx)]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation. It catches monotone relations, so it finds
    duplicates even at a different scale."""
    if a.size < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _rank_auc(vals: np.ndarray, good: np.ndarray) -> float:
    """How well this one feature separates out the answer set (§8.3 item
    7).

    The convention is that lower is better, so if good configs take small
    values then AUC > 0.5.
    """
    n_pos = int(good.sum())
    n_neg = int((~good).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # ★ Ties are handled with average ranks. `argsort(argsort())` splits
    #   ties arbitrarily, smearing the AUC of a binary feature (has_spill and
    #   the like) to around 0.5.
    order = np.argsort(-vals, kind="mergesort")
    r = np.empty(vals.size, dtype=np.float64)
    sv = -vals[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return float((r[good].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def validate_feature(f: Feature, table, matrix, *, hw_alt: Hardware,
                     others: dict[str, np.ndarray] | None = None,
                     registry: FeatureRegistry | None = None,
                     n_shapes: int = 6, n_rows: int = 512,
                     seed: int = 0) -> ValidationReport:
    """Validates one feature. **Exceptions are caught and turned into
    `fail`.**"""
    rep = ValidationReport(f.name)
    rng = np.random.default_rng(seed)
    # ★ A shape-level feature is looked at over **every shape**. With a
    #   sample, a binary feature such as `is_memory_bound` can happen to draw
    #   only one class and be rejected as "constant" — a sampling problem,
    #   not a feature problem. There are only dozens of shapes, so it costs
    #   nothing.
    shapes = (list(table.shapes()) if f.shape_level
              else _sample(table, n_shapes, rng))

    # -- 1. does it run ---------------------------------------------------
    vals: list[np.ndarray] = []
    try:
        for p in shapes:
            fe, info = matrix.for_shape(p)
            if f.shape_level:
                # Shape level is a scalar. Broadcast it over the candidate
                # count so the statistics line up.
                n = int(info.n_candidates)
                v = np.full(n, float(getattr(info, f.name)))
            else:
                v = np.asarray(getattr(fe, f.name), dtype=np.float64)
            vals.append(v)
    except Exception as e:                       # noqa: BLE001
        # ⚠️ It is not swallowed. An exception is a rejection, not an
        # approval (§26.4).
        rep.checks.append(Check("runs", "fail", f"{type(e).__name__}: {e}"))
        return rep
    all_v = np.concatenate(vals)
    if not np.all(np.isfinite(all_v)):
        rep.checks.append(Check(
            "finite", "fail",
            f"{int((~np.isfinite(all_v)).sum())} non-finite"))
        return rep
    rep.checks.append(Check("runs", "ok"))

    # -- 2. range ---------------------------------------------------------
    lo, hi = f.expected_range
    out = int(((all_v < lo - 1e-9) | (all_v > hi + 1e-9)).sum())
    if out:
        rep.checks.append(Check(
            "range", "warn",
            f"{out}/{all_v.size} outside the declared [{lo}, {hi}] "
            f"(observed [{all_v.min():.4g}, {all_v.max():.4g}])"))
    else:
        rep.checks.append(Check("range", "ok"))

    # -- 3. vectorised == scalar ------------------------------------------
    if f.vec is None:
        rep.checks.append(Check("vectorised", "info",
                                "scalar only (slow)"))
    else:
        try:
            from kernelrule.features import verify_vectorized
            for p in shapes[:2]:
                _, info = matrix.for_shape(p)
                verify_vectorized(f, table.frame_for(p), matrix.hw, info,
                                  n=min(n_rows, 128))
            rep.checks.append(Check("vectorised", "ok"))
        except Exception as e:                   # noqa: BLE001
            rep.checks.append(Check("vectorised", "fail", str(e)))
            return rep

    # -- 4. constant ------------------------------------------------------
    if float(all_v.std()) < CONST_STD:
        rep.checks.append(Check(
            "constant", "fail",
            f"std={all_v.std():.3g} — zero explanatory power. It may have "
            "fallen to 0 because a needed column is absent from the table"))
        return rep
    rep.checks.append(Check("constant", "ok"))

    # -- 5. duplication ---------------------------------------------------
    if others:
        # ★ 2026-09-11 (D-162): **the candidate is re-measured on the
        #   comparison set's own shapes.**
        #
        #   Before, `all_v` came from `_sample`'s 6 shapes and `others` from
        #   the first 4, and the loop skipped any pair whose lengths did not
        #   match — which on this table is every pair (a shape is 15,015 or
        #   17,325 rows, so 6 of them never sum to 4 of them). A copy of
        #   `waves` under another name passed with a maximum correlation of
        #   **0.000**: nothing had been compared.
        #
        #   ⚠️ Lining the lengths up would be the wrong fix. Two columns of
        #   equal length taken from different shapes are still not
        #   comparable — they must be **the same rows**.
        dup_shapes = getattr(others, "shapes", None)
        if not dup_shapes:
            raise ValueError(
                "the comparison columns do not say which shapes they were "
                "measured on. Build them with `_reference_columns` — a "
                "duplication verdict on unknown rows is not a verdict "
                "(D-162)")
        dup_vals = []
        for q in dup_shapes:
            fe, info = matrix.for_shape(q)
            dup_vals.append(
                np.full(int(info.n_candidates), float(getattr(info, f.name)))
                if f.shape_level
                else np.asarray(getattr(fe, f.name), dtype=np.float64))
        dup_v = np.concatenate(dup_vals)
        # ★ Duplication must not be judged on Spearman alone.
        #
        #   `sm_idle_cost = 1/(1-tail_waste) - 1` is a **monotone transform**
        #   of `tail_waste`, so Spearman is 0.999. But the rule is a **linear
        #   weighted sum**, so the two have completely different effects — on
        #   512³ the non-linear term produces a 5.3x penalty a linear term
        #   cannot. That is exactly why the hand rule went from 1.221 to
        #   1.192 (kernelTab baselines.md).
        #
        #   In other words **a monotone transform is not a duplicate.** Only
        #   when linear duplication overlaps too is it a deprecation
        #   candidate.
        worst = ("", 0.0, 0.0)
        for name, ov in others.items():
            if name == f.name:
                continue
            if ov.size != dup_v.size:
                # ⛔ **Not skipped.** Skipping is how the check came to be
                #   there without ever running (D-162).
                raise ValueError(
                    f"{f.name}: the comparison column {name!r} has "
                    f"{ov.size} values and the candidate has {dup_v.size} "
                    f"on the same {len(dup_shapes)} shapes. They are not "
                    f"the same rows, so no duplication verdict is possible")
            rho = abs(_spearman(dup_v, ov))
            r = abs(_pearson(dup_v, ov))
            if min(rho, r) > min(worst[1], worst[2]):
                worst = (name, rho, r)
        if worst[1] > DUP_RHO and worst[2] > DUP_RHO:
            # ★ D-162: a **rejection**, not a note. It was a `warn`, and a
            #   warn is not read by anything — F2 built three internally
            #   duplicated pairs and every one was registered.
            # ★ 2026-09-11 (D-163): **what it overlaps with goes in the
            #   message.** "it overlaps" alone does not say what to make
            #   different, and in the D-162 run r6 and r7 built a wave axis
            #   one after the other. The source is already in the registry —
            #   no model is asked, nothing is computed.
            rep.checks.append(Check(
                "duplication", "fail",
                f"Spearman {worst[1]:.3f} / Pearson {worst[2]:.3f} against "
                f"{worst[0]} — both > {DUP_RHO}. It is the same axis under "
                f"another name (§8.4)." + _what_it_is(registry, worst[0])))
            return rep
        if worst[1] > DUP_RHO:
            rep.checks.append(Check(
                "duplication", "info",
                f"a **monotone transform** of {worst[0]} (Spearman "
                f"{worst[1]:.3f}, Pearson {worst[2]:.3f}). In a linear "
                f"weighted sum it is a different term"))
        else:
            rep.checks.append(Check(
                "duplication", "ok",
                f"max min(rho,r) {min(worst[1], worst[2]):.3f}"))

    # -- 6. ★ scale invariance --------------------------------------------
    if f.shape_level:
        alt = np.asarray([float(f.fn(p, hw_alt, table.configs(p)[0]))
                          for p in shapes], dtype=np.float64)
        base = np.asarray([float(f.fn(p, matrix.hw, table.configs(p)[0]))
                           for p in shapes], dtype=np.float64)
    else:
        alt_list, base_list = [], []
        for p in shapes[:2]:
            df = table.frame_for(p).iloc[:n_rows]
            cfgs = [config_from_row(r) for r in df.to_dict("records")]
            alt_list.append([float(f.fn(p, hw_alt, c)) for c in cfgs])
            base_list.append([float(f.fn(p, matrix.hw, c)) for c in cfgs])
        alt = np.concatenate([np.asarray(x) for x in alt_list])
        base = np.concatenate([np.asarray(x) for x in base_list])
    changed = float(np.max(np.abs(alt - base))) if alt.size else 0.0
    uses_hw = _uses_hardware(f)
    if uses_hw and changed <= SCALE_MIN_CHANGE:
        rep.checks.append(Check(
            "scale invariance", "fail",
            "hw changed and the value did not — it looks like it reads hw.*, "
            "but a hardware constant may actually be hardcoded (§8.3 item "
            "6)"))
    elif not uses_hw:
        rep.checks.append(Check(
            "scale invariance", "info",
            "a feature that does not use hw (a function of shape/config "
            "alone). This may be normal"))
    else:
        rep.checks.append(Check("scale invariance", "ok",
                                f"largest change {changed:.4g}"))

    # -- 7. is it useful (marked only) ------------------------------------
    aucs = []
    for p, v in zip(shapes, vals, strict=True):
        good = table.answer_mask(p)
        a = _rank_auc(v, good)
        if np.isfinite(a):
            aucs.append(a)
    if f.shape_level:
        # A shape-level feature is constant within a shape, so it cannot
        # separate candidates. An AUC of 0.5 is **normal** — these features
        # are used not for ranking but for regime branching such as
        # `if p.is_memory_bound:` (the §8.1 replacement).
        rep.checks.append(Check("standalone AUC", "info",
                                "not applicable at shape level (a branching "
                                "feature)"))
    elif aucs:
        m = float(np.mean(aucs))
        if m < 0.45:
            # ★ The direction is opposite to the declaration. It is not
            #   rejected — combined with other terms it may still mean
            #   something. `tail_waste` really does trip this, and minimising
            #   it alone selects small tiles, which made the hand rule as bad
            #   as 1.776 (kernelTab baselines.md). It only means something
            #   once the confounder (tile size) is included too.
            rep.checks.append(Check(
                "standalone AUC", "warn",
                f"{m:.3f} < 0.5 — it predicts **opposite** to the declared "
                f"direction ({f.direction}). It may be confounded with "
                f"another term"))
        else:
            rep.checks.append(Check(
                "standalone AUC", "info",
                f"{m:.3f} ({'meaningful' if abs(m - 0.5) > 0.05 else 'almost uninformative'})"))
    return rep


def _uses_hardware(f: Feature) -> bool:
    """Is there an `hw.` reference in the source? Used to interpret the
    scale check.

    ★ `f.source` is looked at first. For a feature built with `exec`,
    `inspect.getsource` raises `OSError`, and falling back to `True` there
    **rejects every perfectly good feature that does not use the hardware** —
    because the scale check only fails when `uses_hw`. In the first F1 run
    features really were thrown away that way (D-37).
    """
    if f.source:
        return "hw." in f.source
    import inspect
    try:
        src = inspect.getsource(f.fn)
    except (OSError, TypeError):
        # If the source cannot be read, **no verdict is made**. `True` is
        # the claim "it uses hw", and if that claim is wrong a perfectly good
        # feature is rejected (§26.4).
        raise ValueError(
            f"{f.name}: the source cannot be read, so whether it uses the "
            f"hardware cannot be judged. Put the code into "
            f"`Feature(source=...)` — guessing rejects hardware-independent "
            f"features (D-37)") from None
    return "hw." in src


def validate_registry(reg: FeatureRegistry, table, matrix, *,
                      hw_alt: Hardware, **kw) -> dict[str, ValidationReport]:
    """Validates a whole registry. It collects the values for the
    duplication check."""
    rng = np.random.default_rng(kw.get("seed", 0))
    shapes = _sample(table, kw.get("n_shapes", 6), rng)
    # ★ D-162: the pool says which shapes it was measured on. Without that
    #   the duplication check cannot line the candidate up against it, and
    #   it refuses rather than comparing rows that are not the same rows.
    pool = ReferenceColumns()
    pool.shapes = tuple(shapes)
    for name in reg.names(shape_level=False):
        try:
            pool[name] = np.concatenate(
                [np.asarray(getattr(matrix.for_shape(p)[0], name))
                 for p in shapes])
        except Exception:                        # noqa: BLE001, S112
            continue      # a run failure is caught as a fail in the
                          # individual validation
    return {f.name: validate_feature(f, table, matrix, hw_alt=hw_alt,
                                     others=pool, **kw)
            for f in reg.items()}
