"""★ Does the seed actually carry the (b) score? (D-175 §3, 4th condition)
**0 LLM calls.**

    python3 -m experiments.porting_check

⛔ If a seed's holdout differs from the transfer table's `b_refit`, the seed
did not go in — the run would measure something else and its curve would
start from the wrong place. That is a stop condition, not a warning.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.f1_pipeline import _load_stage1, _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry

TOL = 1e-4
OUT = Path("docs/artifacts/porting-seed-check.json")


def main() -> None:
    warnings.simplefilter("ignore")
    rows = []
    bad = 0
    print("=" * 84)
    print("★ seed identity — holdout(seed) == transfer (b)? (D-175 §3)")
    print("=" * 84)
    for d in sorted(Path("runs").glob("pc-*-f0")):
        ch = json.loads((d / "stage2-rule-writer" / "chosen.json").read_text())
        dst = d.name.split("2")[1].split("-f")[0]
        T = TABLES[dst]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        splits = _splits(table, fold=0, k=4, design="nkband")
        reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2",
                           table)
        m = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
        cs = canonical_score(ch["code"], np.asarray(ch["w0"], float),
                             table=table, matrix=m, splits=splits)
        diff = abs(cs.holdout - ch["b_refit"])
        ok = diff < TOL
        bad += (not ok)
        rows.append({"dir": d.name, "seed_holdout": round(cs.holdout, 6),
                     "b_refit": ch["b_refit"], "diff": diff, "ok": ok,
                     "n_axes": len(reg._items)})
        print(f"  {d.name:26s} 씨앗 {cs.holdout:.6f}  (b) {ch['b_refit']:.6f}"
              f"  차 {diff:.2e}  {'ok' if ok else '★ MISMATCH'}  "
              f"축 {len(reg._items)}")
    OUT.write_text(json.dumps({"tolerance": TOL, "n_mismatch": bad,
                               "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n  ★ 불일치 {bad}/{len(rows)}  -> {OUT}")
    if bad:
        raise SystemExit("⛔ D-175 §3 4th — a seed does not carry (b).")


if __name__ == "__main__":
    main()
