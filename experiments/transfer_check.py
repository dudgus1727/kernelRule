"""★ When the 5090 table arrives this is run **first** — what overlaps.
0 LLM calls.

    python3 experiments/transfer_check.py datasets/<5090-bundle> \
        --env-hash <hash>

It fills in "what is not known yet" in
`docs/artifacts/transfer-conditions.md`. The overlap is looked at **before
any transfer number is produced** — a number produced without knowing the
overlap can be the result of the sample selection.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

A6000 = "datasets/rtx-a6000-sm_86-c63710df"
A6000_HASH = "c63710df"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--env-hash", required=True)
    ap.add_argument("--base", default=A6000)
    ap.add_argument("--base-hash", default=A6000_HASH)
    ap.add_argument("--out", default="docs/artifacts/transfer-overlap.json")
    a = ap.parse_args()

    warnings.simplefilter("ignore")
    from kernelrule.core.crosstable import (
        AXIS_FIELDS,
        bound_flipped,
        cross_report,
    )
    from kernelrule.core.table import PerfTable

    A = PerfTable.from_bundle(a.base, env_hash=a.base_hash, ok_only=False)
    B = PerfTable.from_bundle(a.bundle, env_hash=a.env_hash, ok_only=False)

    print("=" * 72)
    print(f"the overlap report  A={Path(a.base).name}  "
          f"B={Path(a.bundle).name}")
    print("=" * 72)
    rep = cross_report(A, B)
    print(rep.render())

    flip = bound_flipped(A, B)
    if flip:
        print("\n  ★ shapes whose bound flips — the per-regime fit measures "
              "something different")
        for p, ma, mb in flip:
            print(f"     {p.M}x{p.N}x{p.K}  A {'memory' if ma else 'compute'}"
                  f" -> B {'memory' if mb else 'compute'}")

    # ★ It fills in "what is not known yet"
    print("\n  the hardware")
    for name, t in (("A", A), ("B", B)):
        hw = t.hw
        print(f"    {name}  SM {hw.sm_count}  bandwidth {hw.bandwidth_gbps} "
              f"GB/s"
              f"  f16 {hw.peak_tflops_f16} TFLOP/s")
    print("\n  dtype")
    for name, t in (("A", A), ("B", B)):
        print(f"    {name}  {sorted({p.dtype for p in t.shapes()})}")
    print("\n  the axis value sets — values present on one side only")
    for f in (*AXIS_FIELDS, "pipeline_kind"):
        va: set = set()
        vb: set = set()
        for t, v in ((A, va), (B, vb)):
            for p in t.shapes():
                fr = t.frame_for(p)
                if f in fr.columns:
                    v |= set(fr[f].unique().tolist())
        only_a, only_b = sorted(va - vb, key=str), sorted(vb - va, key=str)
        mark = "★" if (only_a or only_b) else " "
        print(f"    {mark} {f:16s} A only {only_a}  B only {only_b}")

    Path(a.out).write_text(json.dumps(
        {"_note": "it counts the overlap only. The transfer numbers are not "
                  "here",
         "base": a.base, "other": a.bundle,
         "report": rep.__dict__ if hasattr(rep, "__dict__") else
                   {k: getattr(rep, k) for k in rep.__slots__},
         "bound_flipped": [{"M": p.M, "N": p.N, "K": p.K,
                            "a_memory_bound": ma, "b_memory_bound": mb}
                           for p, ma, mb in flip]},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
