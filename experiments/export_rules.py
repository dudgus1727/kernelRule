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
docs/artifacts/rules/<run>.py      score() + W_FITTED (per regime)
docs/artifacts/rules/index.json    the machine-written scores. The documents
                                   refer to it
```

⚠️ `W_FITTED` has to be the **fitted** value. Writing the initial value makes
the file lie — that is exactly what was written at first, and it was caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
OUT = Path("docs/artifacts/rules")
#: The runs to export. **Only those of the instructed model** go in (D-52).
PREFIXES = ("luna-", "lunaNAMES-")


def setup():
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")
    return table, FeatureMatrix(table, REGISTRY), splits


def main() -> None:
    table, matrix, splits = setup()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(d.name for d in Path("runs").iterdir()
                  if d.is_dir() and d.name.startswith(PREFIXES))
    if not runs:
        raise SystemExit(f"there is no run to export (prefixes {PREFIXES}).")

    index = []
    for run in runs:
        d = Path("runs") / run
        with (d / "archive.jsonl").open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        best = min(arc, key=lambda e: e["regret"])
        w0 = [round(float(x), 6) for x in best["w"]]
        r = canonical_score(best["code"], best["w"], table=table,
                            matrix=matrix, splits=splits)
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
              f"struct HO {r.holdout:.4f}"
              f"  fitter moved {moved}/2")

    (OUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    print(f"\n  {len(index)} -> {OUT}/  (+ index.json)")
    print("  to verify:  python3 experiments/verify_rules.py")


if __name__ == "__main__":
    main()
