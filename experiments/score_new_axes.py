"""Scoring item 2 — condition A (the 24) vs B (the 24 + 10 new axes).

    python3 experiments/score_new_axes.py

★ The judgement is made first on the in-sample value and on "is it used", and
**the structural holdout is looked at once, at the end** (§12.3d). If that
number becomes the ground for the next fix, the holdout is used up.

The scoring uses `core.canonical.canonical_score` alone — it runs only if
given the loop's `SplitSet` and there is no path that picks the shapes
separately (D-36).

⚠️ Condition B's rules refer to generated features, so they have to be scored
with **a matrix built from the extended registry**. Scoring with the base
registry raises `AttributeError`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.scoring import compare, evaluate
from kernelrule.core.splits import Split, SplitSet, experiment_shapes
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import extended_registry
from kernelrule.rules.human_guided import CODE as PS
from kernelrule.rules.human_guided import W0 as PS_W0

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"


def main() -> None:
    # ★ Run as a script directly, `experiments` is not seen as a package.
    #   An `__init__.py` is kept and the repository root is put on the path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.new_axes import novel_axes, used_features

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    shapes = experiment_shapes(table)
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")

    novel = novel_axes(table, list(table.shapes())[:12])
    ext = extended_registry(REGISTRY, novel)
    new_names = {f.name for f in novel}
    mats = {"A-base": FeatureMatrix(table, REGISTRY),
            "B-extended": FeatureMatrix(table, ext)}

    v = evaluate(vendor_order_fn(table, load_vendor(VENDOR),
                                 mapping="nearest"),
                 table, list(splits.val.shapes), ks=(1,), label="vendor")

    print("=" * 78)
    print("item 2 — the usefulness of a new axis")
    print("=" * 78)
    print(f"  {'condition':14s} {'seed':>5} {'in-sample':>10} "
          f"{'★struct HO':>11} {'new axes used':>14}")
    rows: dict[str, list] = {}
    for cond in ("A-base", "B-extended"):
        for s in range(3):
            d = Path("runs") / f"newaxes-{cond}-s{s}" / "archive.jsonl"
            if not d.exists():
                continue
            with d.open() as fh:
                arc = [json.loads(ln) for ln in fh if ln.strip()]
            best = min(arc, key=lambda e: e["regret"])
            try:
                r = canonical_score(best["code"], best["w"], table=table,
                                    matrix=mats[cond], splits=splits)
            except Exception as exc:                        # noqa: BLE001
                print(f"  {cond:14s} {s:5d}  failed {type(exc).__name__}: "
                      f"{str(exc)[:50]}")
                continue
            rows.setdefault(cond, []).append(r)
            used = sorted(used_features(best["code"]) & new_names)
            print(f"  {cond:14s} {s:5d} {r.in_sample:10.4f} "
                  f"{r.holdout:11.4f} "
                  f"{len(used):14d}  {used if used else ''}")

    print(f"\n  {'condition':14s} {'in-sample median':>17} {'worst':>8} "
          f"{'★struct HO median':>19} {'worst':>8}")
    for cond, rs in rows.items():
        ins = sorted(x.in_sample for x in rs)
        hos = sorted(x.holdout for x in rs)
        print(f"  {cond:14s} {ins[len(ins) // 2]:17.4f} {ins[-1]:8.4f} "
              f"{hos[len(hos) // 2]:19.4f} {hos[-1]:8.4f}")
    print(f"  {'vendor ★pass cond':18s} {'':13s} {'':8s} {v.at(1):19.4f}")
    ps = canonical_score(PS, PS_W0, table=table, matrix=mats["A-base"],
                         splits=splits)
    print(f"  {'human_guided':14s} {ps.in_sample:17.4f} {'':8s} "
          f"{ps.holdout:19.4f}")

    print(f"\n{'=' * 78}")
    print("significance — the structural holdout, the median seed per "
          "condition")
    print("=" * 78)
    for cond, rs in rows.items():
        mid = sorted(rs, key=lambda x: x.holdout)[len(rs) // 2]
        c = compare(mid.evaluation, v, table, name_a="A", name_b="vendor")
        print(f"  {cond:14s} {c.geo_a:.4f} vs {c.geo_b:.4f}   "
              f"wins {int(c.a_wins.sum()):2d} / "
              f"losses {int(c.a_loses.sum()):2d}"
              f" / indistinguishable {int(c.tied.sum()):2d}")

    print(f"\n{'=' * 78}")
    print(f"how much the {len(new_names)} new axes were used in the archive")
    print("=" * 78)
    tally: dict[str, int] = dict.fromkeys(sorted(new_names), 0)
    total = 0
    for s in range(3):
        d = Path("runs") / f"newaxes-B-extended-s{s}" / "archive.jsonl"
        if not d.exists():
            continue
        with d.open() as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                total += 1
                for n in used_features(json.loads(ln)["code"]) & new_names:
                    tally[n] += 1
    for n, c in sorted(tally.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(20 * c / max(total, 1))
        print(f"  {n:30s} {c:3d}/{total:3d}  {bar}")
    print(f"\n  axes never used once: "
          f"{[n for n, c in tally.items() if c == 0] or 'none'}")


if __name__ == "__main__":
    main()
