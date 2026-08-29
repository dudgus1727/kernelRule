"""★ 5090 표가 오면 **가장 먼저** 돌린다 — 무엇이 겹치나. LLM 0회.

    python3 experiments/transfer_check.py datasets/<5090-번들> --env-hash <해시>

`docs/artifacts/transfer-conditions.md` 의 "아직 모르는 것" 을 채운다.
**전이 수치를 내기 전에** 겹침을 먼저 본다 — 겹침을 모르고 낸 수치는
표본 선택의 결과일 수 있다.
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
    print(f"겹침 보고  A={Path(a.base).name}  B={Path(a.bundle).name}")
    print("=" * 72)
    rep = cross_report(A, B)
    print(rep.render())

    flip = bound_flipped(A, B)
    if flip:
        print("\n  ★ 바운드가 뒤집히는 형상 — 체제별 적합이 다른 것을 잰다")
        for p, ma, mb in flip:
            print(f"     {p.M}x{p.N}x{p.K}  A {'메모리' if ma else '컴퓨트'}"
                  f" -> B {'메모리' if mb else '컴퓨트'}")

    # ★ "아직 모르는 것" 을 채운다
    print("\n  하드웨어")
    for name, t in (("A", A), ("B", B)):
        hw = t.hw
        print(f"    {name}  SM {hw.sm_count}  대역폭 {hw.bandwidth_gbps} GB/s"
              f"  f16 {hw.peak_tflops_f16} TFLOP/s")
    print("\n  dtype")
    for name, t in (("A", A), ("B", B)):
        print(f"    {name}  {sorted({p.dtype for p in t.shapes()})}")
    print("\n  축 값 집합 — 한쪽에만 있는 값을 본다")
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
        print(f"    {mark} {f:16s} A만 {only_a}  B만 {only_b}")

    Path(a.out).write_text(json.dumps(
        {"_note": "겹침만 센다. 전이 수치는 여기 없다",
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
