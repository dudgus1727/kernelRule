"""It exports the final rules and their **fitted** weights into the
repository.

    python3 experiments/export_rules.py

## Why it is needed

`runs/` is in `.gitignore`. So **the rule files are not in the repository**
and there was no way to check the numbers in the documents — not against
deliberate manipulation, but **nobody would know if a transcription error or
a rounding slipped in.**

```
an LLM run        not reproducible (the randomness is not controlled —
                  §24.4b)
scoring/rescoring ★ completely deterministic. A few seconds
```

**The latter is verifiable.** With the rules and the weights committed,
anyone can check the numbers in the documents in seconds with
`experiments/verify_rules.py`.

## What it writes

```
docs/artifacts/rules/<run>.py            score() + W_FITTED (per regime)
docs/artifacts/rules/<run>.registry.json ★ the axis list the rule needs
docs/artifacts/rules/index.json          the machine-written scores. The
                                         documents refer to it
```

★ 2026-09-11 (D-166 §M): **the axis list goes with the rule.** Until now
this script loaded the 24 human-written axes for every run. The rules
exported so far are all F3 (human axes) so nothing was wrong — but an F2
rule uses axes the FeatureWriter built, and those live in `runs/`, which is
`.gitignore`d. Such a rule was refused at load time and landed in
`verify_rules`' D-156 "cannot be re-scored" bucket, **next to a green "all
matched" line**. The generated axes' source is now written out beside the
weights, for the same reason the weights are written out at all.

⚠️ `W_FITTED` has to be the **fitted** value. Writing the initial value makes
the file lie — that is exactly what was written at first, and it was caught.

⚠️ 2026-09-11 (D-166 §M): **this script is not idempotent.** `canonical_score`
**refits** per regime, so re-running it today rewrites the committed numbers
(measured: `luna-s5` holdout recorded 1.1378, re-exported 1.0895 — the fitter
conditions changed in D-150/D-152). Re-export a run only when you mean to
replace its record. What checks the existing numbers is
`experiments/verify_rules.py`, which re-scores the committed `W_FITTED` and
does not fit anything.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet, experiment_shapes
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import registry_spec, run_registry

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
OUT = Path("docs/artifacts/rules")
#: The runs to export. **Only those of the instructed model** go in (D-52).
PREFIXES = ("luna-", "lunaNAMES-")


def setup():
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    return table, FeatureMatrix(table, REGISTRY), _splits_of(table, "nk11008")


def _splits_of(table, kind: str) -> SplitSet:
    """The split **the run itself used**. `nk11008` is the structural one.

    ★ The rule has to be scored on its own holdout. Exporting an F2 rule
    under `nk11008` when it evolved on `kfold0-seed12345` would put a
    number in `index.json` that `verify_rules` then reproduces exactly —
    both wrong in the same way (principle 38).
    """
    if kind == "nk11008":
        shapes = experiment_shapes(table)
        held = [p for p in shapes if 11008 in (p.N, p.K)]
        return SplitSet(
            train=Split("train", tuple(p for p in shapes if p not in held)),
            val=Split("val", tuple(held)), kind="nk11008")
    m = re.fullmatch(r"kfold(\d+)-seed(\d+)", kind)
    if not m:
        raise SystemExit(f"unknown split_kind {kind!r}. It is not guessed "
                         f"— the holdout would silently be the wrong set.")
    from experiments.f1_pipeline import _splits
    return _splits(table, fold=int(m.group(1)), split_seed=int(m.group(2)))


def _pipeline_dir(run: str) -> Path | None:
    """`runs/cap1-s0` -> `runs/cap1` when that is a pipeline run.

    The three-layer axis list lives in the pipeline directory; the older
    runs (`luna-*`) have no such directory and used the human registry.
    """
    m = re.fullmatch(r"(.+)-s(\d+)", run)
    if not m:
        return None
    d = Path("runs") / m.group(1)
    return d if (d / "config.json").exists() else None


def _registry_for(run: str, table):
    """`(registry, spec, splits)` for one run. ★ Never a silent default."""
    d = _pipeline_dir(run)
    if d is None:
        # The human axes, as before. `origin` says so for every axis.
        spec = registry_spec(
            REGISTRY, dict.fromkeys(REGISTRY._items, "human"),
            run=None, condition=None)
        return REGISTRY, spec, _splits_of(table, "nk11008")
    cfg = json.loads((d / "config.json").read_text())
    seed = int(re.fullmatch(r"(.+)-s(\d+)", run).group(2))
    reg, origin = run_registry(d.name, table=table, seed=seed,
                               human=REGISTRY)
    spec = registry_spec(reg, origin, run=d.name,
                         condition=cfg.get("condition"))
    return reg, spec, _splits_of(table, cfg.get("split_kind", "nk11008"))


def main() -> None:
    # ★ Run as a script directly, `experiments` is not seen as a package
    #   — `_splits_of` reaches for `f1_pipeline` on the k-fold branch.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(d.name for d in Path("runs").iterdir()
                  if d.is_dir() and d.name.startswith(PREFIXES))
    if not runs:
        raise SystemExit(f"there is no run to export (prefixes {PREFIXES}).")

    index = []
    #: one matrix per distinct axis list — building it is the expensive part
    mats: dict[str, FeatureMatrix] = {}
    #: ★ What is already recorded. A rule that **cannot be re-scored today**
    #:   keeps its entry — re-exporting must not delete a recorded number
    #:   (documentation rule 2). Three `luna-*` rules branch on
    #:   `p.log_sol_ms`, retired in D-156, and re-running this script has
    #:   raised on them ever since.
    prev = {}
    if (OUT / "index.json").exists():
        prev = {e["run"]: e
                for e in json.loads((OUT / "index.json").read_text())}
    for run in runs:
        d = Path("runs") / run
        with (d / "archive.jsonl").open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        best = min(arc, key=lambda e: e["regret"])
        w0 = [round(float(x), 6) for x in best["w"]]
        reg, spec, splits = _registry_for(run, table)
        if spec["hash"] not in mats:
            mats[spec["hash"]] = FeatureMatrix(table, reg)
        matrix = mats[spec["hash"]]
        try:
            r = canonical_score(best["code"], best["w"], table=table,
                                matrix=matrix, splits=splits)
        except AttributeError as exc:
            if "unregistered" not in str(exc) or run not in prev:
                raise
            why = str(exc).split(".")[0]
            kept = dict(prev[run])
            kept["unverifiable"] = {"why": why, "since": "D-156"}
            index.append(kept)
            print(f"  {run:16s} ⚠️ kept as recorded — {why}")
            continue
        if any(a["source"] for a in spec["axes"]):
            # ★ Only a generated list needs writing out — the human axes
            #   are in the repository already.
            (OUT / f"{run}.registry.json").write_text(
                json.dumps(spec, ensure_ascii=False, indent=1) + "\n")
        cfg = {}
        if (d / "config.json").exists():
            cfg = json.loads((d / "config.json").read_text()).get("llm", {})

        w = "\n".join(f"    {k!r}: {[round(x, 6) for x in v]},"
                      for k, v in sorted(r.weights.items()))
        (OUT / f"{run}.py").write_text(
            f'"""The final rule of {run} — the **minimum training score** in '
            f'the archive.\n\n'
            f'model      {cfg.get("model", "?")} / '
            f'{cfg.get("endpoint", "?")}\n'
            f'reasoning  {cfg.get("reasoning_effort", "?")}\n'
            f'features   {cfg.get("feature_detail", "?")}\n\n'
            f'structural holdout {r.holdout:.4f}  '
            f'(in-sample {r.in_sample:.4f})\n\n'
            f'★ `W_FITTED` is the value **fitted per regime**. It is not the '
            f'initial value.\n'
            f'To reproduce:  python3 experiments/verify_rules.py\n"""\n\n'
            "import numpy as np  # noqa: F401\n\n"
            + best["code"].strip() + "\n\n\n"
            + "W_FITTED = {\n" + w + "\n}\n")

        index.append({
            "run": run, "in_sample": round(r.in_sample, 6),
            "holdout": round(r.holdout, 6),
            # ★ D-166 §M: which axis list re-scores this rule, and on which
            #   holdout. Without it `verify_rules` guesses, and guessing
            #   wrong is invisible.
            "registry": {
                "kind": "run" if any(a["source"] for a in spec["axes"])
                        else "human",
                "n": spec["n"], "hash": spec["hash"], "run": spec["run"],
                "condition": spec["condition"],
                "file": (f"{run}.registry.json"
                         if any(a["source"] for a in spec["axes"]) else None)},
            "split_kind": splits.kind,
            "by_regime": {k: round(v, 6) for k, v in r.by_regime.items()},
            "n_holdout": r.n_holdout, "llm": cfg,
            "weights": {k: [round(x, 6) for x in v]
                        for k, v in r.weights.items()},
            # ★ Did the fitter actually move (D-54). If it did not, §29's
            #   "structures are compared fairly" does not hold.
            "w_moved": {k: [round(x, 6) for x in v] != w0
                        for k, v in r.weights.items()}})
        moved = sum(1 for k, v in r.weights.items()
                    if [round(x, 6) for x in v] != w0)
        print(f"  {run:16s} in-sample {r.in_sample:.4f}  "
              f"holdout {r.holdout:.4f} ({splits.kind})"
              f"  fitter moved {moved}/2"
              f"  axes {spec['n']} {spec['hash']}")

    (OUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    print(f"\n  {len(index)} -> {OUT}/  (+ index.json)")
    print("  to verify:  python3 experiments/verify_rules.py")


if __name__ == "__main__":
    main()
