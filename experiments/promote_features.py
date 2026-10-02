"""★ Promotes axes a campaign's loop built into a new **library**. 0 LLM calls.

    python3 -m experiments.promote_features \\
        --library runs/k7-1/stage1-features \\
        --runs 'runs/d190-a6000-f*-s*' \\
        --out runs/k7-1+d190/stage1-features

A later campaign loads the result with the existing
`f1_pipeline.py --import-featwriter <out>`.

## Why (D-190 §2)

An axis the loop builds lives in that run's `features.jsonl` and dies with
it. The axes that moved a6000 the most in D-189 were time estimates built
once and reused everywhere; nothing in the pipeline carried such an axis
from one run to the next.

## Which axes

```
candidate   an accepted loop axis that the run's FINAL best rule reads
            (bests.jsonl[-1].code mentions f.<name>) — chosen on the
            training split by the loop itself
⛔ never    holdout (val_regret) — a promotion chosen on the holdout makes
            the next campaign's holdout a training set
order       fixed: run directory name, then the round the axis was built
check       register_generated against base + library + what was promoted
            before it — §8.3 in full, so the same quantity built under two
            names in two runs goes in once
```

## What it writes

`<out>/proposals.jsonl` — the library's lines **verbatim**, then the
promoted rows with their provenance — and `<out>/summary.json` with the
sha256 of every input. Refused candidates are listed there with the reason;
nothing is dropped silently.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def candidates(run_dirs: list[Path]) -> list[dict]:
    """The loop axes the final best rule of each run reads. Pure file
    reading — no table, no score."""
    out = []
    for d in sorted(run_dirs, key=lambda x: x.name):
        fp, bp = d / "features.jsonl", d / "bests.jsonl"
        if not fp.exists() or not bp.exists():
            continue
        lines = [ln for ln in bp.read_text().splitlines() if ln.strip()]
        if not lines:
            continue
        code = json.loads(lines[-1]).get("code") or ""
        used = set(re.findall(r"\bf\.([a-z][a-z0-9_]*)", code))
        rows = [json.loads(ln) for ln in fp.read_text().splitlines()
                if ln.strip()]
        for r in sorted(rows, key=lambda r: r.get("round", 0)):
            if r.get("accepted") and r.get("name") in used:
                out.append({**r, "promoted_from": d.name})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, required=True,
                    help="a stage1-features directory (proposals.jsonl)")
    ap.add_argument("--runs", required=True,
                    help="glob of loop run directories")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--condition", default="F2")
    ap.add_argument("--bundle", default="datasets/rtx-a6000-sm_86-c63710df")
    ap.add_argument("--env-hash", default="c63710df")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the candidates only — no table is read")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    run_dirs = [Path(p) for p in glob.glob(a.runs) if Path(p).is_dir()]
    cand = candidates(run_dirs)
    print(f"runs {len(run_dirs)} · candidates {len(cand)}: "
          f"{[c['name'] for c in cand]}")
    if a.dry_run:
        return

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY
    from kernelrule.features.generated import (
        FeatureRejected,
        register_generated,
    )
    from kernelrule.features.loader import base_registry, load_generated
    from kernelrule.features.validate import alt_hw

    lib = a.library / "proposals.jsonl"
    table = PerfTable.from_bundle(a.bundle, env_hash=a.env_hash,
                                  ok_only=False)
    reg = base_registry(a.condition, human=REGISTRY)
    for f in load_generated(lib, exclude=set(reg._items), table=table):
        reg.add(f)
    matrix = FeatureMatrix(table, reg)
    hw_alt = alt_hw(table.hw)

    promoted, refused = [], []
    for c in cand:
        meta = {k: c.get(k) for k in ("name", "unit", "direction",
                                      "expected_range", "rationale")}
        if meta.get("expected_range") is None:
            meta["expected_range"] = ((c.get("range") or {})
                                      .get("declared"))
        try:
            f = register_generated(c["code"], registry=reg, meta=meta,
                                   table=table, matrix=matrix,
                                   hw_alt=hw_alt)
            promoted.append({**c, "name": f.name, "accepted": True,
                             "unit": f.unit, "direction": f.direction,
                             "expected_range": list(f.expected_range)})
            print(f"  ★ promoted  {f.name}  ({c['promoted_from']})")
        except FeatureRejected as e:
            refused.append({"name": c.get("name"),
                            "from": c["promoted_from"], "why": str(e)[:200]})
            print(f"     refused  {c.get('name')}  {str(e)[:80]}")

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "proposals.jsonl").open("w") as fh:
        for ln in lib.read_text().splitlines():
            if ln.strip():
                fh.write(ln + "\n")
        for r in promoted:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (a.out / "summary.json").write_text(json.dumps({
        "procedure": "D-190 §2 — loop axes the final best rule reads, "
                     "re-checked with register_generated; holdout not read",
        "library": str(lib), "library_sha256": _sha(lib),
        "runs": sorted(str(d) for d in run_dirs),
        "features_sha256": {str(d): _sha(d / "features.jsonl")
                            for d in sorted(run_dirs)
                            if (d / "features.jsonl").exists()},
        "bundle": a.bundle, "condition": a.condition,
        "promoted": [r["name"] for r in promoted], "refused": refused,
    }, ensure_ascii=False, indent=1))
    print(f"  -> {a.out}  (+{len(promoted)} · refused {len(refused)})")


if __name__ == "__main__":
    main()
