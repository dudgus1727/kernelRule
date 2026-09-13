"""★ The shape-judgement table — **which shapes each table contributes, and
why the others are out** (D-170 §1). 0 LLM calls · 0 GPU.

    python3 -m experiments.population65

Until 2026-09-13 the population was "every candidate aligned to 8 on A, B and
C" (61 of the A6000 table's 66). The grounds for that were never written down
anywhere, and D-167 §R measured that 4 of the 5 it excluded have the same
candidate-space structure as the ones it kept. The criterion is now **"more
than one kernel family in the candidate space"**, which is what the old one
was standing in for.

This script writes the judgement per table so that a reader can check it
rather than take the count on trust:

```
per table    table shapes · new population · old population · what is dropped
per dropped  candidates · stage range · kernel families · alignment
             ★ and the vendor's regret on it — a shape the vendor finds hard
             is not what we are excluding
```

⚠️ It reads the per-shape vendor regret out of `vendor-baselines.json`
(`experiments/vendor_baselines.py` writes it). It does not recompute it — two
procedures for one number is how the D-158 vendor-preset split happened.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from experiments.transfer_29_5 import TABLES
from kernelrule.core.splits import (
    ALIGNMENT_REQUIRED,
    KERNEL_FAMILY_COLUMN,
    MIN_KERNEL_FAMILIES,
    aligned_shapes,
    experiment_shapes,
    kernel_families,
)
from kernelrule.core.table import PerfTable

OUT = "docs/artifacts/population-65.json"
OUT_MD = "docs/artifacts/population-65.md"
VENDOR = "docs/artifacts/vendor-baselines.json"


def _row(table, p, vend_per_shape) -> dict:
    d = table.frame_for(p)
    st = sorted({int(x) for x in d["ext_stages"]})
    key = f"{p.M}x{p.N}x{p.K}"
    return {
        "shape": key,
        "candidates": int(len(d)),
        "stages": [st[0], st[-1]],
        "families": kernel_families(table, p),
        "align": [int(d["align_a"].min()), int(d["align_b"].min()),
                  int(d["align_c"].min())],
        # `null` when the vendor artefact does not carry this shape — it is
        # not filled in with a guess (§26.4).
        "vendor_regret": vend_per_shape.get(key)}


def main() -> None:
    warnings.simplefilter("ignore")
    vend = json.loads(Path(VENDOR).read_text()) if Path(VENDOR).exists() else {}
    res: dict = {
        "criterion": {
            "current": (f"at least {MIN_KERNEL_FAMILIES} distinct "
                        f"{KERNEL_FAMILY_COLUMN} in the candidate space"),
            "previous": (f"align_a/b/c == {ALIGNMENT_REQUIRED} on every "
                         f"candidate (until 2026-09-13)"),
            "decision": "D-170 §1"},
        "tables": {}}
    print("=" * 92)
    print("★ the shape judgement per table (D-170 §1). 0 LLM calls")
    print("=" * 92)
    print(f"  {'table':7s} {'shapes':>6} {'★ new':>6} {'old':>5}  dropped now")
    for name in ("a6000", "5090", "4090", "h100"):
        T = TABLES[name]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        vps = (vend.get(name, {}) or {}).get("per_shape") or \
            (vend.get("per_shape", {}) or {}).get(name, {})
        new = experiment_shapes(table)
        old = aligned_shapes(table)
        nk = {p.key for p in new}
        ok = {p.key for p in old}
        dropped_now = [p for p in table.shapes() if p.key not in nk]
        readmitted = [p for p in table.shapes()
                      if p.key in nk and p.key not in ok]
        res["tables"][name] = {
            "bundle": T["bundle"], "env_hash": T["env_hash"],
            "n_table": len(table.shapes()),
            "n_population": len(new), "n_population_previous": len(old),
            "dropped": [_row(table, p, vps) for p in dropped_now],
            "readmitted": [_row(table, p, vps) for p in readmitted],
            "kept_candidates": [int(len(table.frame_for(p)))
                                for p in new]}
        r = res["tables"][name]
        print(f"  {name:7s} {r['n_table']:6d} {r['n_population']:6d} "
              f"{r['n_population_previous']:5d}  "
              f"{[x['shape'] for x in r['dropped']]}")
    print()
    for name, r in res["tables"].items():
        print(f"  ── {name}")
        for tag, rows in (("dropped", r["dropped"]),
                          ("readmitted", r["readmitted"])):
            for x in rows:
                v = ("    —" if x["vendor_regret"] is None
                     else f"{x['vendor_regret']:6.4f}")
                print(f"     {tag:10s} {x['shape']:18s} n={x['candidates']:<6d} "
                      f"stages {x['stages'][0]}..{x['stages'][1]}  "
                      f"align {x['align']}  vendor {v}  "
                      f"{'+'.join(x['families'])}")
    Path(OUT).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    Path(OUT_MD).write_text(_md(res))
    print(f"\n  -> {OUT}\n  -> {OUT_MD}")


def _md(res: dict) -> str:
    c = res["criterion"]
    out = ["# The shape population per table (D-170 §1)", "",
           f"- current criterion — **{c['current']}**",
           f"- previous criterion — {c['previous']}", "",
           "| table | table shapes | ★ population | previous | dropped |",
           "|---|---:|---:|---:|---|"]
    for n, r in res["tables"].items():
        out.append(f"| {n} | {r['n_table']} | **{r['n_population']}** | "
                   f"{r['n_population_previous']} | "
                   f"{', '.join(x['shape'] for x in r['dropped'])} |")
    out += ["", "## Every shape either criterion treats differently", "",
            ("| table | verdict | shape | candidates | stages | families | "
             "align | vendor regret |"),
            "|---|---|---|---:|---|---|---|---:|"]
    for n, r in res["tables"].items():
        for tag, rows in (("dropped", r["dropped"]),
                          ("readmitted ★", r["readmitted"])):
            for x in rows:
                v = "—" if x["vendor_regret"] is None \
                    else f"{x['vendor_regret']:.4f}"
                out.append(
                    f"| {n} | {tag} | `{x['shape']}` | {x['candidates']:,} | "
                    f"{x['stages'][0]}..{x['stages'][1]} | "
                    f"{'+'.join(x['families'])} | {x['align']} | {v} |")
    out += ["", ("★ `readmitted` are the shapes the alignment criterion "
                 "excluded and the kernel-family criterion keeps."), ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
