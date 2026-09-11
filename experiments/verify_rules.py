"""★ It reproduces the documents' numbers from the committed rules. It runs
without `runs/`.

    python3 experiments/verify_rules.py

## What is verified

```
docs/artifacts/rules/<run>.py            score() + W_FITTED
docs/artifacts/rules/<run>.registry.json the axis list the rule needs
docs/artifacts/rules/index.json          the scores recorded at the time
```

★ 2026-09-11 (D-166 §M): the axis list is **read from the record**, not
assumed to be the 24 human-written ones. A rule built on generated axes used
to fail to load and fall into the D-156 bucket below — and that bucket is
printed next to "all N match". The two are now separate:

```
verified        re-scored and equal to the recorded number
✗ failed        it **should** be verifiable and was not  -> exit 1
⚠️ unverifiable  it uses a value the library retired (D-156). The recorded
                 number stands; re-scoring needs that commit's registry
```

**This script executes the `.py`, recomputes the score and checks it against
`index.json`.** If they diverge it fails.

## Why this matters

An LLM run cannot be reproduced (the randomness is not controllable —
§24.4b). But **the scoring is completely deterministic**. With the rule files
committed, **half of the performance claim becomes verifiable** — anyone can
confirm it in seconds.

`runs/` is in `.gitignore`, so it is not in the repository. This script does
not read it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.numerics import approx_equal
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY, loader

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
RULES = Path("docs/artifacts/rules")
TOL = 5e-4


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score, mod.W_FITTED


def _registry_of(row, table, mats):
    """`(matrix, note)` for one recorded rule. ★ Raises
    `RegistryUnavailable` rather than quietly using the human axes."""
    rec = row.get("registry") or {}
    if rec.get("kind") != "run":
        if "human" not in mats:
            mats["human"] = FeatureMatrix(table, REGISTRY)
        return mats["human"], "human"
    f = RULES / (rec.get("file") or "")
    if not f.exists():
        raise loader.RegistryUnavailable(
            f"{f} is missing — the rule needs {rec.get('n')} axes from run "
            f"{rec.get('run')!r} and they are not in the repository. "
            f"Re-run `python3 experiments/export_rules.py`.")
    spec = json.loads(f.read_text())
    if spec.get("hash") != rec.get("hash"):
        raise loader.RegistryUnavailable(
            f"{f.name} hash {spec.get('hash')} != index.json "
            f"{rec.get('hash')} — the axis list changed after the export")
    key = spec["hash"]
    if key not in mats:
        mats[key] = FeatureMatrix(
            table, loader.registry_from_spec(spec, table=table,
                                             human=REGISTRY))
    return mats[key], f"{rec.get('run')} {rec.get('n')} axes"


def main() -> None:
    import numpy as np

    # ★ Run as a script directly, `experiments` is not seen as a package.
    #   An `__init__.py` is kept and the repository root is put on the path
    #   (the same as `score_new_axes.py`).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.export_rules import _splits_of

    idx_path = RULES / "index.json"
    if not idx_path.exists():
        raise SystemExit(f"{idx_path} does not exist. "
                         "Run `python3 experiments/export_rules.py` first.")
    index = json.loads(idx_path.read_text())

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    mats: dict = {}

    print("=" * 78)
    print("rescoring the committed rules — checked against index.json")
    print("=" * 78)
    print(f"  {'run':16s} {'recorded':>10} {'recomputed':>12}  verdict")
    bad = []
    #: ★ 2026-09-10 (D-156): `log_sol_ms` stopped being a registered
    #: shape-level value, so a rule that branches on `p.log_sol_ms` **cannot
    #: be re-scored under today's registry**. That is the expected
    #: consequence, not a failure: those runs' numbers stand as recorded, and
    #: the rule file plus its weights are in the repository — re-scoring them
    #: needs the registry of their own commit.
    #: ⚠️ 2026-09-11 (D-166 §M): this bucket is **only** for that. A rule
    #: whose own axis list should have been loadable and was not is a
    #: failure — putting the two in one bucket is what hid an entire
    #: campaign behind a green line.
    unverifiable = []
    n_checked = 0
    for row in index:
        run = row["run"]
        f = RULES / f"{run}.py"
        if not f.exists():
            bad.append(f"{run}: the rule file does not exist")
            continue
        fn, W = _load(f)
        try:
            matrix, note = _registry_of(row, table, mats)
        except loader.RegistryUnavailable as exc:
            bad.append(f"{run}: {exc}")
            print(f"  {run:16s} {row['holdout']:10.4f} {'-':>12}  "
                  f"❌ the axis list could not be loaded")
            continue
        splits = _splits_of(table, row.get("split_kind", "nk11008"))
        reg = {}
        try:
            for name in ("short", "long"):
                g = [p for p in splits.val.shapes
                     if regime_of(p, table.hw) == name]
                if not g or name not in W:
                    continue
                e = evaluate_scores(
                    make_score_of(fn, matrix, np.asarray(W[name])),
                    table, g, ks=(1,))
                for i, p in enumerate(e.shapes):
                    reg[p] = e.regret[i, 0]
        except AttributeError as exc:
            if "unregistered" not in str(exc):
                raise
            why = str(exc).split(".")[0]
            if (row.get("registry") or {}).get("kind") == "run":
                # ★ Its own axis list loaded and the rule still asks for a
                #   name that is not in it. That is a failure, not D-156.
                bad.append(f"{run}: {why} — but its axis list "
                           f"({note}) loaded")
                print(f"  {run:16s} {row['holdout']:10.4f} {'-':>12}  "
                      f"❌ {why}")
                continue
            unverifiable.append((run, why))
            print(f"  {run:16s} {row['holdout']:10.4f} {'-':>12}  "
                  f"⚠️ unverifiable — {why}")
            continue
        got = geomean(np.array([reg[p] for p in splits.val.shapes if p in reg]))
        want = row["holdout"]
        ok = approx_equal(got, want, TOL)
        n_checked += 1
        if not ok:
            bad.append(f"{run}: recorded {want:.4f} != recomputed {got:.4f}")
        print(f"  {run:16s} {want:10.4f} {got:12.4f}  {'✅' if ok else '❌'}"
              f"   {note}")

    print()
    if bad:
        print("★ what diverged:")
        for b in bad:
            print(f"    {b}")
        raise SystemExit(1)
    # ★ D-166 §M④: "all matched" is not printed when nothing was checked.
    #   That line, with 0 in front of it, is how a whole campaign could sit
    #   in the unverifiable bucket and still read as verified (principle 1).
    if n_checked:
        print(f"  ★ {n_checked} of {len(index)} verified "
              f"(tolerance {TOL})")
    else:
        print(f"  ★ **0 of {len(index)} verified.** Nothing was re-scored — "
              f"this is not a pass.")
    if unverifiable:
        print(f"  ⚠️ {len(unverifiable)} unverifiable — they use a value "
              f"that is no longer registered (D-156):")
        for run, why in unverifiable:
            print(f"      {run:16s} {why}")
        print("     Their recorded numbers stand; re-scoring them needs the "
              "registry of their own commit.")
    if not n_checked:
        raise SystemExit(1)
    print("  The documents' structural holdout numbers are verified by these "
          "values.")


if __name__ == "__main__":
    main()
