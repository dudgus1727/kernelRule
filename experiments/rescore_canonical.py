"""★ Rescoring by the final scoring procedure — only the same procedure /
the same denominator / the same aggregation are compared.

    python3 experiments/rescore_canonical.py

## Why it is needed

RuleWriter A/B reported train41/val20 on the `nk11008` structural split,
while the earlier reports came from a **per-regime refit** over 61 shapes.
Putting the two numbers side by side and writing "it falls short of the pass
condition" is the pattern §30.8 has caught again and again — it compared two
values from different procedures.

Here **every candidate is measured again by the same procedure.**

    the regime judgement   the SOL 2-way split (fast 41 / slow 20).
                           ★ `t_best` is not used
    the weights            fitted separately per regime
    the aggregation        evaluated per regime, then combined over 61 shapes
    the significance       per shape (t_A - t_B) / (t_best x noise_floor), 2σ

★ And **the in-sample and the holdout are reported together**. The per-regime
fit has twice the free parameters, so the in-sample value is bound to improve
(confirmed in 8-2). Reporting the in-sample alone mistakes that gain for
skill.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import compare, evaluate, evaluate_scores, geomean
from kernelrule.core.splits import Split, experiment_shapes, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.rules.human_guided import CODE as PS
from kernelrule.rules.human_guided import W0 as PS_W0

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
RUNS = Path("runs")


def _best(path: Path, key: str = "train"):
    with path.open() as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    ok = [r for r in rows if "code" in r]
    return min(ok, key=lambda r: r[key]) if ok else None


def main() -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    shapes = experiment_shapes(table)
    fast = [p for p in shapes if regime_of(p, table.hw) == "short"]
    slow = [p for p in shapes if regime_of(p, table.hw) == "long"]

    # ★ The holdout — 1 in every 3 in SOL order within each regime. It spreads
    #   evenly inside the regime.
    def thirds(g):
        g = sorted(g, key=lambda p: table.frame_for(p).index[0])
        return ([p for i, p in enumerate(g) if i % 3 != 2],
                [p for i, p in enumerate(g) if i % 3 == 2])

    f_fit, f_ho = thirds(fast)
    s_fit, s_ho = thirds(slow)
    holdout = f_ho + s_ho

    def per_regime(code, w0, fit_groups, eval_groups):
        """Fits separately per regime and scores the given shapes with that
        regime's weights."""
        reg, tol = {}, {}
        for grp, ev_grp in zip(fit_groups, eval_groups, strict=True):
            fit = fit_weights(compile_rule(code), matrix, table,
                              Split("train", tuple(grp)), w0, max_evals=300,
                          objective="regret")
            so = make_score_of(compile_rule(code), matrix, fit.w)
            e = evaluate_scores(so, table, ev_grp, ks=(1,))
            for i, p in enumerate(e.shapes):
                reg[p], tol[p] = e.regret[i, 0], e.tol[i]
        return reg, tol

    cands: list[tuple[str, str, list]] = [("human_guided", PS, PS_W0)]
    for cond, model in (("A", "gpt-5.4"), ("B", "gpt-5.4"),
                        ("A", "gpt-5.4-mini-2026-03-17")):
        r = _best(RUNS / f"architect-{cond}-{model}" / "tries.jsonl")
        if r:
            tag = "mini" if "mini" in model else "5.4"
            cands.append((f"RuleWriter {cond} ({tag})", r["code"], r["w"]))
    ev_run = RUNS / "real-gpt-5.4-mini-2026-03-17" / "archive.jsonl"
    if ev_run.exists():
        with ev_run.open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        e = min(arc, key=lambda x: x["regret"])
        cands.append(("evolved (the first run)", e["code"], e["w"]))

    vendor = load_vendor(VENDOR)
    v_all = evaluate(vendor_order_fn(table, vendor, mapping="nearest"),
                     table, shapes, ks=(1,), label="vendor")
    v_ho = evaluate(vendor_order_fn(table, vendor, mapping="nearest"),
                    table, holdout, ks=(1,), label="vendor")

    print("=" * 78)
    print("rescoring by the final scoring procedure — per-regime (SOL 2-way) "
          "refit, combined over 61 shapes")
    print("=" * 78)
    print(f"  fast {len(fast)} / slow {len(slow)}   "
          f"holdout {len(holdout)} (1 in every 3 within a regime)")
    print(f"\n  {'':24s} {'in-sample61':>12} {'★holdout':>11} "
          f"{'fast':>8} {'slow':>8}")

    results = {}
    for name, code, w0 in cands:
        try:
            r_in, t_in = per_regime(code, w0, [fast, slow], [fast, slow])
            r_ho, t_ho = per_regime(code, w0, [f_fit, s_fit], [f_ho, s_ho])
        except Exception as exc:                            # noqa: BLE001
            print(f"  {name:24s} failed {type(exc).__name__}: "
                  f"{str(exc)[:40]}")
            continue
        g_in = geomean(np.array([r_in[p] for p in shapes]))
        g_ho = geomean(np.array([r_ho[p] for p in holdout]))
        g_f = geomean(np.array([r_in[p] for p in fast]))
        g_s = geomean(np.array([r_in[p] for p in slow]))
        results[name] = (r_ho, t_ho)
        print(f"  {name:24s} {g_in:12.4f} {g_ho:11.4f} {g_f:8.4f} {g_s:8.4f}")
    print(f"  {'vendor nearest ★pass cond':24s} {v_all.at(1):12.4f} "
          f"{v_ho.at(1):11.4f}")

    # -- the significance. ★ judged on the holdout only --------------------
    print(f"\n{'=' * 78}")
    print("significance — ★ against the vendor on the holdout (it is not "
          "judged in-sample)")
    print("=" * 78)
    from dataclasses import replace
    base = evaluate_scores(make_score_of(compile_rule(PS), matrix,
                                         np.ones(len(PS_W0))),
                           table, holdout, ks=(1,))
    for name, (r_ho, t_ho) in results.items():
        ev = replace(base,
                     regret=np.array([r_ho[p] for p in holdout]).reshape(-1, 1),
                     tol=np.array([t_ho[p] for p in holdout]), label=name)
        c = compare(ev, v_ho, table, name_a="A", name_b="vendor")
        print(f"  {name:24s} {c.geo_a:.4f} vs {c.geo_b:.4f}   "
              f"wins {int(c.a_wins.sum()):2d} / losses "
              f"{int(c.a_loses.sum()):2d}"
              f" / indistinguishable {int(c.tied.sum()):2d}")


if __name__ == "__main__":
    main()
