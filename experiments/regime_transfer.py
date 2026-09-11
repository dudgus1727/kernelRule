"""★ Regime transfer — the structure is held fixed and refitted to another
regime. 0 LLM calls.

    python3 experiments/regime_transfer.py

## What this experiment separates

The claim came up that "the LLM structure does not transfer and only the hand
rule does", and "the LLM structure has regime-specific terms" was named as
the cause. **That is an observation, not a cause.** An ablation separates
them.

    1-A  remove the regime-specific terms from the LLM structure and refit on
         the long shapes
         if it recovers -> those terms are the cause
    1-B  add the regime-specific terms to the hand rule, fit on short ->
         refit on long
         if it gets worse -> the causation is confirmed in both directions

Block 0 first counts "who actually uses those terms". Without checking the
premise first, there is no way to know what the ablation measured.

## The result (2026-08-21)

**Both directions denied the causation.**
`docs/artifacts/structure-transfer.md`.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401  — it fills REGISTRY
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import compare, evaluate, evaluate_scores, geomean
from kernelrule.core.splits import _DUMMY_CFG, Split
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.features.physical import log_sol_ms
from kernelrule.rules.human_guided import CODE as HW
from kernelrule.rules.human_guided import W0 as HW_W0

#: The three terms that were named. The ground of the claim was that they only
#: mean something when `K/tile_k` is small.
REGIME_SPECIFIC = frozenset({"is_two_stage", "pipeline_warmup_frac",
                             "log_mainloop_iters"})

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
RUN_REAL = Path("runs/real-gpt-5.4-mini-2026-03-17/archive.jsonl")
RUN_SMOKE = Path("runs/smoke2-gpt-5.4-mini-2026-03-17/archive.jsonl")

#: The long/short boundary (§30). Below 0.5 ms the tick resolution dominates.
SIZE_THRESHOLD_MS = 0.5


# ---------------------------------------------------------------------------
# AST — count the terms, delete them, and renumber the weight indices
# ---------------------------------------------------------------------------

def features_used(node: ast.AST) -> set[str]:
    """The feature names referenced as `f.<name>`."""
    return {n.attr for n in ast.walk(node)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "f"}


def weight_indices(node: ast.AST) -> set[int]:
    """The constant indices of `w[i]`."""
    return {n.slice.value for n in ast.walk(node)
            if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name) and n.value.id == "w"
            and isinstance(n.slice, ast.Constant)}


def _drop_terms(body: list[ast.stmt], banned: frozenset[str]) -> list[ast.stmt]:
    """Deletes the statements that reference a banned feature. An empty `if`
    disappears whole."""
    keep: list[ast.stmt] = []
    for st in body:
        if isinstance(st, ast.If):
            st.body = _drop_terms(st.body, banned)
            if st.body:
                keep.append(st)
        elif isinstance(st, ast.Return) or not (features_used(st) & banned):
            keep.append(st)
    return keep


def ablate(code: str, w0: list[float],
           banned: frozenset[str] = REGIME_SPECIFIC) -> tuple[str, list[float]]:
    """Deletes the terms and renumbers the remaining `w` indices to 0..n-1.

    ★ Without the renumbering they go out of step with `w0` and it becomes
    **measuring a different rule**. That is why this function exists.
    """
    tree = ast.parse(code.strip())
    fn = tree.body[0]
    fn.body = _drop_terms(fn.body, banned)
    used = sorted(weight_indices(fn))
    remap = {old: i for i, old in enumerate(used)}

    class _Renumber(ast.NodeTransformer):
        def visit_Subscript(self, n):                       # noqa: N802
            self.generic_visit(n)
            if (isinstance(n.value, ast.Name) and n.value.id == "w"
                    and isinstance(n.slice, ast.Constant)
                    and n.slice.value in remap):
                n.slice = ast.Constant(remap[n.slice.value])
            return n

    tree = _Renumber().visit(tree)
    ast.fix_missing_locations(tree)
    new_w0 = [float(w0[o]) if o < len(w0) else 1.0 for o in used]
    return ast.unparse(tree), new_w0


# ---------------------------------------------------------------------------

def _load_archive(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


#: The hand rule with the two named terms added. `pipeline_warmup_frac` is
#: **already there**. With 9 weights it goes over the §29.4 budget (8), but
#: this is not a candidate — it is **a diagnostic construct**. Keeping the
#: budget would mean taking another term out, which muddies what was
#: measured.
HW_PLUS = """
def score(f, p, hw, w):
    s  = np.log2(f.traffic_amplification) * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + f.smem_pressure * w[2]
    s = s + f.has_spill * w[3]
    s = s + f.split_k_cost * w[4]
    s = s + f.pipeline_warmup_frac * w[5]
    s = s + f.is_two_stage * w[7]
    s = s + f.log_mainloop_iters * w[8]
    if p.is_memory_bound:
        s = s + np.log2(f.traffic_amplification) * w[6]
    return s
"""


def main() -> None:                                          # noqa: PLR0915
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    all_shapes = [p for p in table.shapes() if aligned(p)]
    thr = math.log2(SIZE_THRESHOLD_MS)
    short = [p for p in all_shapes
             if log_sol_ms(p, table.hw, _DUMMY_CFG) < thr]
    long_ = [p for p in all_shapes if p not in short]
    vendor = load_vendor(VENDOR)

    real = _load_archive(RUN_REAL)
    llm_train = min(real, key=lambda e: e["regret"])
    # ⚠️ 2026-09-11 (D-166): **this arm picks the rule on the holdout.**
    #   It exists as an upper bound — "how well would the ablation read if
    #   the best rule for these shapes had been chosen" — and is only ever
    #   comparable inside this ablation, never as a result. The label below
    #   says so, because a label that does not is how a number walks out of
    #   the badge that fences it (`docs/artifacts/structure-transfer.md`).
    #   ⛔ Do not move this number into `conclusion.md`.
    llm_val = min((e for e in real if np.isfinite(e["val_regret"])),
                  key=lambda e: e["val_regret"])
    llm_smoke = min(_load_archive(RUN_SMOKE), key=lambda e: e["regret"])

    def refit(code, w0, shapes):
        return fit_weights(compile_rule(code), matrix, table,
                           Split("train", tuple(shapes)), w0, max_evals=300,
                          objective="regret")

    def score(code, w, shapes, label=""):
        return evaluate_scores(make_score_of(compile_rule(code), matrix, w),
                               table, shapes, ks=(1,), label=label)

    def vend(shapes):
        return evaluate(vendor_order_fn(table, vendor, mapping="nearest"),
                        table, shapes, ks=(1,), label="vendor")

    v_long, v_short, v_all = vend(long_), vend(short), vend(all_shapes)

    cands = [("hand rule (7 terms)", HW, HW_W0),
             ("LLM smoke (8 terms)", llm_smoke["code"], llm_smoke["w"]),
             (("LLM val-selected (16 terms) ★ picked ON the holdout — an "
               "upper bound, not a result"),
              llm_val["code"], llm_val["w"]),
             ("LLM train best (19 terms)", llm_train["code"], llm_train["w"])]

    # -- 0. checking the premise -------------------------------------------
    print("=" * 76)
    print("0. which rule actually uses the regime-specific terms")
    print("=" * 76)
    for name, code, _ in cands:
        used = features_used(ast.parse(code.strip())) & REGIME_SPECIFIC
        print(f"  {name:26s} regime-specific {sorted(used)}")

    # -- 1-A. does removal restore it? -------------------------------------
    print(f"\n{'=' * 76}")
    print("1-A. remove the regime-specific terms -> refit on the long 20   "
          "(does removal restore it)")
    print("=" * 76)
    print(f"  {'structure':26s} {'orig long20':>12} {'after':>10} "
          f"{'terms':>9}")
    for name, code, w0 in cands:
        fit = refit(code, w0, long_)
        base = score(code, fit.w, long_).at(1)
        abl_code, abl_w0 = ablate(code, list(fit.w))
        if not abl_w0:
            print(f"  {name:26s} {base:12.4f}   (everything was removed and "
                  f"the rule is empty)")
            continue
        fit2 = refit(abl_code, abl_w0, long_)
        after = score(abl_code, fit2.w, long_).at(1)
        print(f"  {name:26s} {base:12.4f} {after:10.4f} "
              f"{len(w0):3d}→{len(abl_w0):<3d}")
    print(f"  {'vendor':26s} {v_long.at(1):12.4f}")

    # -- 1-B. does adding break it? (the opposite direction of the
    #         causation) ----------------------------------------------------
    print(f"\n{'=' * 76}")
    print("1-B. add the regime-specific terms to the hand rule -> fit on "
          "short -> refit on long")
    print("=" * 76)
    # ★ After fitting on the short shapes it refits on the long shapes
    #   **starting from those weights**. That is exactly the procedure §29.5
    #   (b) describes — the structure as it is, only the weights to the new
    #   regime.
    fit_s = refit(HW, HW_W0, short)
    hw_s = score(HW, fit_s.w, short)
    hw_l = score(HW, refit(HW, list(fit_s.w), long_).w, long_, "hand rule")
    plus_fit_s = refit(HW_PLUS, [*HW_W0, 0.3, 0.3], short)
    hp_s = score(HW_PLUS, plus_fit_s.w, short)
    hp_l = score(HW_PLUS, refit(HW_PLUS, list(plus_fit_s.w), long_).w,
                 long_, "hand rule+3")
    print(f"  {'hand rule (7 terms, orig)':28s} short41 {hw_s.at(1):.4f}"
          f"   →  long20 refit {hw_l.at(1):.4f}")
    print(f"  {'hand rule+regime (9 terms)':28s} short41 {hp_s.at(1):.4f}"
          f"   →  long20 refit {hp_l.at(1):.4f}")
    print(f"  {'vendor':28s} short41 {v_short.at(1):.4f}"
          f"   →  long20      {v_long.at(1):.4f}")

    # -- 2. the whole-61 performance of the per-regime refit ---------------
    print(f"\n{'=' * 76}")
    print("2. the whole-61 performance of the hand rule's per-regime refit")
    print("=" * 76)
    i_s = {p: i for i, p in enumerate(hw_s.shapes)}
    i_l = {p: i for i, p in enumerate(hw_l.shapes)}
    per_shape = np.array([hw_s.regret[i_s[p], 0] if p in i_s
                          else hw_l.regret[i_l[p], 0] for p in all_shapes])
    tol = np.array([hw_s.tol[i_s[p]] if p in i_s else hw_l.tol[i_l[p]]
                    for p in all_shapes])
    is_short = np.array([p in i_s for p in all_shapes])
    hw_one = score(HW, refit(HW, HW_W0, all_shapes).w, all_shapes,
                   "hand rule single")
    print(f"  {'':30s} {'short41':>8} {'long20':>8} {'all61':>8}")
    print(f"  {'hand rule (per-regime refit)':30s} {hw_s.at(1):8.4f} "
          f"{hw_l.at(1):8.4f} {geomean(per_shape):8.4f}")
    print(f"  {'hand rule (single fit on all)':30s} "
          f"{hw_one.at(1, mask=is_short):8.4f} "
          f"{hw_one.at(1, mask=~is_short):8.4f} {hw_one.at(1):8.4f}")
    print(f"  {'vendor':30s} {v_short.at(1):8.4f} {v_long.at(1):8.4f} "
          f"{v_all.at(1):8.4f}")
    print("  ★ the pass condition 1.080")

    # -- 3. the significance judgement (§30.6) -----------------------------
    print(f"\n{'=' * 76}")
    print("3. the significance judgement — per shape "
          "(t_A - t_B) / (t_best x noise_floor)")
    print("=" * 76)
    from dataclasses import replace
    hw_mix = replace(hw_one, regret=per_shape.reshape(-1, 1), tol=tol,
                     label="hand rule (per regime)")
    pairs = [
        ("hand rule (per-regime refit) vs vendor  all61", hw_mix, v_all),
        ("hand rule (long refit)       vs vendor  long20", hw_l, v_long),
        ("hand rule (short fit)        vs vendor  short41", hw_s, v_short),
        ("LLM train best               vs vendor  short41",
         score(llm_train["code"], llm_train["w"], short), v_short),
        ("LLM train best               vs vendor  all61",
         score(llm_train["code"], llm_train["w"], all_shapes), v_all),
    ]
    for label, a, b in pairs:
        c = compare(a, b, table, name_a="A", name_b="vendor")
        print(f"  {label:44s} {c.geo_a:.4f} vs {c.geo_b:.4f}  "
              f"wins {int(c.a_wins.sum()):2d} / losses "
              f"{int(c.a_loses.sum()):2d}"
              f" / indistinguishable {int(c.tied.sum()):2d}")


if __name__ == "__main__":
    main()
