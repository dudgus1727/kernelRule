"""★ The (c) regrow ladder — old / middle / new. 0 LLM calls.

    python3 experiments/c_ladder.py

The pre-registration is `docs/artifacts/c-rerun3-prereg.md`.

```
old (c)     numbers A6000 · warnings A6000    5090sigma-s{0,1,2}
middle (c)  numbers 5090  · warnings A6000    5090sigma-hw-s{0,1,2}
new (c)     numbers 5090  · warnings 5090     5090sigma-hw2-s{0,1,2}
```

Each rung differs in **exactly one** thing. The regret is produced by **the
same procedure** as `sigma_5090.py` (the final scoring) — the old (c)'s value
came out of that procedure (principle 4).

## ⚠️ The recorded (c) 1.0485 is **a value mixing two seeds**

`transfer_29_5.TABLES["5090"]["runs"]` bundles six as (c), and the last three
(`-b-`) are the `human_guided` **hand seed**. That is not "from scratch on
the 5090 table". The ladder's first rung uses the 3 RuleWriter-seeded seeds
(1.0416).

⚠️ 2026-09-08 (D-146): **the two step names stay in Korean.** They are the
keys of `steps` in `c-ladder.json`, and `docs/` is not translated.
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from pathlib import Path

import numpy as np
from regret_at_k import _noise_ranks
from scipy.stats import kendalltau
from sigma_5090 import _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.runset import assert_same_condition
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

BUNDLE, ENV = "datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403"
DELTA = 0.0516        # ★ the σ upper-bound decision line §29.5 already used
TOP_N, KS = 100, (1, 10, 100)

ARMS = [
    ("old (c)     numbers A6000 · warnings A6000", "5090sigma"),
    ("middle (c)  numbers 5090  · warnings A6000", "5090sigma-hw"),
    ("new (c)     numbers 5090  · warnings 5090", "5090sigma-hw2"),
]
#: ★ The hand-seed arm that was mixed into the recorded (c). It is kept out of
#: the ladder and reported **separately**.
MIXED = ("hand seed (mixed into the recorded (c))", "5090sigma-b")


def _arc_best(run: str) -> dict:
    f = Path("runs") / run / "archive.jsonl"
    arc = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return sorted(arc, key=lambda e: e["regret"])[0]


def _taus(code, w0, table, matrix, sp) -> tuple[float, float, float]:
    """The top-100 tau / the noise-aware tau / the all-range tau — on the
    holdout."""
    fn = compile_rule(code)
    ws = {}
    for nm in ("short", "long"):
        g = [q for q in sp.train.shapes if regime_of(q, table.hw) == nm]
        ws[nm] = fit_weights(fn, matrix, table, Split("train", tuple(g)), w0,
                             max_evals=300, objective="regret").w
    rng = np.random.default_rng(12345)
    raw, noi, allr = [], [], []
    for p in sp.val.shapes:
        cand = table.candidates(p)
        s = np.asarray(make_score_of(fn, matrix, ws[regime_of(p, table.hw)])(
            p, cand), dtype=np.float64)
        t = np.asarray(table.times_of(p), dtype=np.float64)
        top = np.argsort(t, kind="stable")[:TOP_N]
        if len(np.unique(t[top])) > 1:
            v = kendalltau(s[top], t[top], variant="b").statistic
            if np.isfinite(v):
                raw.append(float(v))
        nr = _noise_ranks(t, table.noise)[top]
        if len(np.unique(nr)) > 1:
            v = kendalltau(s[top], nr, variant="b").statistic
            if np.isfinite(v):
                noi.append(float(v))
        idx = rng.choice(len(t), size=min(4000, len(t)), replace=False)
        allr.append(float(kendalltau(s[idx], t[idx], variant="b").statistic))
    return (float(np.median(raw)), float(np.median(noi)),
            float(np.median(allr)))


def _seed_shape(tag: str) -> dict:
    d = Path("runs") / f"f1pipe-F3-{tag}" / "stage2-rule-writer"
    cs = {f.name: f.read_text() for f in sorted(d.glob("candidates/*.py"))}
    ch = json.loads((d / "chosen.json").read_text())
    sol = [n for n, c in cs.items() if "log_sol_ms" in c]
    feats = [len({m.attr for m in ast.walk(ast.parse(c))
                  if isinstance(m, ast.Attribute)
                  and isinstance(m.value, ast.Name) and m.value.id == "f"})
             for c in cs.values()]
    return {"n": len(cs), "sol": sol, "chosen_sol": "log_sol_ms" in ch["code"],
            "feats": sorted(feats), "source": ch.get("source"),
            "fit_regret": ch.get("fit_regret")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/c-ladder.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(BUNDLE, env_hash=ENV, ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    out: dict = {"delta": DELTA, "n_holdout": len(sp.val.shapes)}

    print("=" * 84)
    print("§1  the regret ladder — the final scoring, the 5090 holdout of "
          f"{len(sp.val.shapes)} shapes, 3 seeds")
    print("=" * 84)
    print(f"  {'':44s} {'median':>8} {'range':>19}")
    med: dict[str, float] = {}
    for label, tag in [*ARMS, MIXED]:
        # ★ Within each rung the condition has to be single (D-120)
        assert_same_condition([f"f1pipe-F3-{tag}-s{i}" for i in range(3)],
                              label=label)
        h = []
        for i in range(3):
            e = _arc_best(f"f1pipe-F3-{tag}-s{i}")
            h.append(canonical_score(e["code"], e["w"], table=T, matrix=M,
                                     splits=sp).holdout)
        h = np.array(h)
        med[tag] = float(np.median(h))
        mark = "   ← outside the ladder" if tag == MIXED[1] else ""
        print(f"  {label:44s} {np.median(h):8.4f} "
              f"{h.min():8.4f}~{h.max():<8.4f}{mark}")
        out.setdefault("regret", {})[tag] = h.tolist()

    print(f"\n  ★ the recorded (c) 1.0485 = the median of the six above "
          f"({np.median([*out['regret']['5090sigma'], *out['regret']['5090sigma-b']]):.4f}) "
          "— **it is a value mixing two seeds**")

    print(f"\n  the decision line delta = {DELTA} (the σ upper bound, the "
          f"value §29.5 already used)")
    for a_, b_, name in (("5090sigma", "5090sigma-hw", "옛 -> 중간 (숫자)"),
                         ("5090sigma-hw", "5090sigma-hw2",
                          "중간 -> 새 (경고 절)")):
        d = med[a_] - med[b_]          # positive means it got better
        verdict = ("★ it is used" if d >= DELTA else
                   "★ indistinguishable" if abs(d) < DELTA
                   else "indistinguishable (it got worse)")
        print(f"    {name:22s} {med[a_]:.4f} -> {med[b_]:.4f}  "
              f"difference {d:+.4f}   {verdict}")
        out.setdefault("steps", {})[name] = d

    print("\n" + "=" * 84)
    print("§2  tau — do the top ranks rise when the warnings go (the holdout)")
    print("=" * 84)
    print(f"  {'':44s} {'top-100 tau':>12} {'noise-aware':>12} {'all':>9}")
    for label, tag in ARMS:
        v = np.array([_taus(_arc_best(f"f1pipe-F3-{tag}-s{i}")["code"],
                            _arc_best(f"f1pipe-F3-{tag}-s{i}")["w"], T, M, sp)
                      for i in range(3)])
        print(f"  {label:44s} {np.median(v[:, 0]):12.3f} "
              f"{np.median(v[:, 1]):12.3f} {np.median(v[:, 2]):9.3f}")
        out.setdefault("tau", {})[tag] = v.tolist()

    print("\n" + "=" * 84)
    print("§3  the shape of the 10 seeds — does `p.log_sol_ms` split the "
          "shapes by length")
    print("=" * 84)
    for label, tag in ARMS:
        s = _seed_shape(tag)
        print(f"  {label:44s} {len(s['sol'])}/{s['n']}  "
              f"{'present' if s['chosen_sol'] else 'absent'} in the chosen "
              f"seed   axis counts {s['feats']}")
        out.setdefault("seed", {})[tag] = s

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3 seeds cannot give significance — it is read by the decision "
          "line and the range (principle 27)")


if __name__ == "__main__":
    main()
