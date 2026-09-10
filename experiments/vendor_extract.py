"""★ It extracts the vendor's (nvMatmulHeuristics) recommendations for the
bundle's shapes. 0 GPU.

    python3 experiments/vendor_extract.py datasets/rtx-5090-sm_120-5bb6f403 \
        --env-hash 5bb6f403 --out datasets/baselines/vendor-5090-5bb6f403.json

The library **only computes, from a preset** — it does not need a GPU.

## ★ The conditions are matched to kernelTab

```
version    0.1.0.27           a different one changes the recommendation
target     ★ CUTLASS          CUTLASS3 gives cluster(1,4) — our table is the
                              2.x space
layout     TN_ROW_MAJOR       the same for every shape
precision  HSS
preset     derived from the hw name  (RTX_A6000 / RTX_5090 ...)
```

The output format is the same as
`datasets/baselines/vendor-a6000-c63710df.json` —
`kernelrule.baselines.vendor.load_vendor` reads it as it is.

## ★ It is watched after extraction

```
cluster == (1,1)      is it the 2.x space
instr   == (16,8,16)  is it f16 HMMA
```

If either differs, it is **a recommendation from a different kernel space**
and the join with our table does not hold. They are counted and printed, and
if there are any, it exits with code 1.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: GPU name -> preset. ★ The same table as kernelTab's
#: `scripts/baseline_vendor.py`.
GPU_PRESETS = {
    "rtx a6000": "RTX_A6000", "rtx 4090": "RTX_4090", "rtx 3090": "RTX_3090",
    "rtx 5090": "RTX_5090", "rtx 6000 ada": "RTX_6000_ADA",
    "a100": "A100_SXM_80GB", "a40": "A40_PCIE", "a30": "A30_PCIE",
    "a10": "A10_PCIE",
    # ★ 2026-09-10 (D-158): the H100 comes in variants and this table mapped
    #   every one of them to `H100_SXM`. Our bundle is an **H100 NVL** — a
    #   different card (different clocks and bandwidth), and the preset for
    #   it exists. The longer key wins in `preset_for`, so the variants are
    #   listed before the bare "h100".
    "h100 nvl": "H100_NVL", "h100 pcie": "H100_PCIE", "h100": "H100_SXM",
    "h200": "H200_SXM",
    "l40s": "L40S", "l40": "L40", "l4": "L4", "b200": "B200",
}

PAT = re.compile(
    r"stages\((\d+)\)\s+cta\((\d+) (\d+) (\d+)\)\s+warp\((\d+) (\d+) (\d+)\)"
    r"\s+instr\((\d+) (\d+) (\d+)\)\s+splitK\((\d+)\)\s+swizz\((\d+)\)"
    r"\s+ctaOrder\((\d+)\)")
CLUSTER_PAT = re.compile(r"cluster\((\d+) (\d+)(?: (\d+))?\)")


def preset_for(name: str) -> str:
    low = name.lower().replace("nvidia", "").strip()
    for k, v in sorted(GPU_PRESETS.items(), key=lambda kv: -len(kv[0])):
        if k in low:
            return v
    raise SystemExit(
        f"the preset for '{name}' is unknown. Add it to GPU_PRESETS.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--env-hash", required=True)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import nvMatmulHeuristics as nv

    info = json.loads(Path(a.bundle, "BUNDLE.json").read_text())
    if not str(info["env_hash"]).startswith(a.env_hash):
        raise SystemExit(f"env_hash mismatch: {str(info['env_hash'])[:16]}")
    preset = preset_for(info["gpu_name"])
    print(f"{info['gpu_name']}  ->  preset {preset}   "
          f"(target=CUTLASS, layout=TN_ROW_MAJOR, precision=HSS)")

    # ★ The shapes are read **from the bundle**. Calling kernelTab's
    #   all_shapes reads that side's env and can go out of step with the
    #   bundle.
    shapes = sorted({tuple(s) for layer in
                     (info.get("shape_layers") or {}).values() for s in layer})
    print(f"{len(shapes)} bundle shapes")

    h = nv.NvMatmulHeuristicsInterface(nv.NvMatmulHeuristicsTarget.CUTLASS,
                                       precision="HSS")
    hd = h.createHardwareDescriptor()
    h.setHardwarePredefinedGpu(hd, getattr(nv.NvMatmulHeuristicsNvidiaGpu,
                                           preset))
    layout = nv.NvMatmulHeuristicsMatmulLayout.TN_ROW_MAJOR

    out: dict = {"_meta": {
        "gpu": info["gpu_name"], "preset": preset,
        "env_hash": info["env_hash"], "count": a.count,
        "bundle_id": info["bundle_id"], "layout": "TN_ROW_MAJOR",
        "precision": "HSS", "target": "CUTLASS",
        "lib_version": "0.1.0.27"}}
    bad_cluster, bad_instr, n_fail = 0, 0, 0
    for (M, N, K) in shapes:
        lst = []
        for c in h.get_with_mnk(M, N, K, layout, a.count, hd):
            kern, rt = c["kernel"], c.get("runtime")
            raw = str(kern)
            mc = CLUSTER_PAT.search(raw)
            if mc and tuple(int(x) for x in mc.groups() if x is not None)[:2] \
                    != (1, 1):
                bad_cluster += 1
            if not isinstance(kern, str):
                g = kern
                lst.append({"stages": g.stages,
                            "cta": [g.cta_tile_m, g.cta_tile_n, g.cta_tile_k],
                            "warp": [g.warp_tile_m, g.warp_tile_n,
                                     g.warp_tile_k],
                            "split_k": g.split_k, "swizzle": g.swizzle_factor,
                            "cta_order": g.cta_order,
                            "pred_ms": (rt or 0) * 1000.0})
                continue
            mo = PAT.search(raw)
            if not mo:
                n_fail += 1
                lst.append({"raw": raw, "parse_fail": True})
                continue
            g = [int(x) for x in mo.groups()]
            if tuple(g[7:10]) != (16, 8, 16):
                bad_instr += 1
            lst.append({"stages": g[0], "cta": g[1:4], "warp": g[4:7],
                        "split_k": g[10], "swizzle": g[11], "cta_order": g[12],
                        "pred_ms": (rt or 0) * 1000.0})
        out[f"{M}x{N}x{K}"] = lst

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"  -> {a.out}   ({len(out) - 1} shapes x {a.count})")
    print(f"  ★ watch: cluster != (1,1) {bad_cluster}   "
          f"instr != (16,8,16) {bad_instr}   parse failures {n_fail}")
    if bad_cluster or bad_instr:
        raise SystemExit(
            "★ recommendations from a different kernel space got mixed in. "
            "The join with our table (CUTLASS 2.x, f16 HMMA) does not hold. "
            "Check the target.")


if __name__ == "__main__":
    main()
