"""★ D-166 §C-2 — does narrowing feature validation to the training split
change any acceptance? 0 LLM calls.

    python3 -m experiments.validate_split_ab

## What is asked

`validate_feature` draws its sample (`_sample`) and its duplication
comparison set (`_spread_shapes`) from **the whole table**, holdout shapes
included. None of the checks that produce a rejection reads a measured time
(that is the AUC check alone, and it is `info`/`warn`), so this is not a §3
answer leak. It is the §1.3 question: shapes the rule never saw took part in
deciding which axes exist.

**Before narrowing it, the verdicts are compared.** Every axis recorded in
`stage1-features/proposals.jsonl` is validated twice — as it was, and with
the sample and the comparison set restricted to the training split — and the
accept/reject outcomes are put side by side.

⛔ This changes nothing. It reports.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import FeatureRegistry
from kernelrule.features.generated import (
    FeatureRejected,
    _reference_columns,
    register_generated,
)
from kernelrule.features.known7 import KNOWN7
from kernelrule.features.validate import alt_hw

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
ENV = "c63710df"
RUNS = ("cap1", "dupfix", "F2new")
OUT = "docs/artifacts/validate-split-ab.json"


def _splits_of(run: str, table):
    """The run's own split — read from its `config.json`, not guessed."""
    from experiments.f1_pipeline import _splits

    cfg = json.loads(Path(f"runs/{run}-s0/config.json").read_text())
    kind = cfg["split"]["kind"]
    if kind.startswith("kfold"):
        fold = int(kind[5])
        seed = int(kind.split("-seed")[1])
        return _splits(table, fold=fold, split_seed=seed, k=3), kind
    raise SystemExit(f"unknown split kind {kind!r} for {run}")


def _verdict(code: str, meta: dict, table, matrix, base_names, train_shapes,
             others) -> tuple[bool, str]:
    reg = FeatureRegistry("ab")
    for n in base_names:
        reg.add(KNOWN7[n])
    try:
        register_generated(code, registry=reg, meta=meta, table=table,
                           matrix=matrix, hw_alt=alt_hw(table.hw),
                           others=others, train_shapes=train_shapes,
                           sample_from_train=train_shapes is not None)
        return True, ""
    except FeatureRejected as e:
        return False, str(e)[:120]
    except Exception as e:                                  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"[:120]


def main() -> None:
    warnings.simplefilter("ignore")
    table = PerfTable.from_bundle(BUNDLE, env_hash=ENV, ok_only=False)
    base = FeatureRegistry("known7")
    for n in sorted(KNOWN7._items):
        base.add(KNOWN7[n])
    m0 = FeatureMatrix(table, base)
    res: dict = {"bundle": BUNDLE, "runs": {}}
    print("=" * 84)
    print("D-166 §C-2 — whole table vs training split, on every recorded axis")
    print("=" * 84)
    total_diff = 0
    for run in RUNS:
        path = Path(f"runs/{run}/stage1-features/proposals.jsonl")
        if not path.exists():
            print(f"  {run}: no proposals.jsonl — skipped")
            continue
        sp, kind = _splits_of(run, table)
        train = list(sp.train.shapes)
        rows = [json.loads(x) for x in path.read_text().splitlines()
                if x.strip()]
        print(f"\n  {run}  split {kind}  train {len(train)} / "
              f"val {len(sp.val.shapes)}  ·  {len(rows)} proposals",
              flush=True)
        # the comparison set, both ways — built once each
        ref_all = _reference_columns(table, m0, base)
        ref_tr = _reference_columns(table, m0, base, train_shapes=train)
        out = []
        for r in rows:
            if not r.get("code"):
                continue
            meta = {k: r.get(k) for k in
                    ("unit", "expected_range", "direction", "rationale")}
            ok_all, why_all = _verdict(r["code"], meta, table, m0,
                                       sorted(KNOWN7._items), None, ref_all)
            ok_tr, why_tr = _verdict(r["code"], meta, table, m0,
                                     sorted(KNOWN7._items), train, ref_tr)
            same = ok_all == ok_tr
            total_diff += not same
            out.append({"name": r.get("name"), "recorded": r.get("accepted"),
                        "whole_table": ok_all, "train_only": ok_tr,
                        "same": same, "why_whole": why_all,
                        "why_train": why_tr})
            mark = "  " if same else "★⛔"
            print(f"   {mark} {str(r.get('name')):48s} "
                  f"whole {'✓' if ok_all else '✗'}  "
                  f"train {'✓' if ok_tr else '✗'}"
                  + ("" if same else f"\n        whole: {why_all}"
                                     f"\n        train: {why_tr}"),
                  flush=True)
        res["runs"][run] = {"split": kind, "n_train": len(train),
                            "n_val": len(sp.val.shapes), "axes": out}
    res["n_differing"] = total_diff
    Path(OUT).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n★ verdicts that differ: {total_diff}")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
