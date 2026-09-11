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
from pathlib import Path

# ★ 2026-09-11 (D-166): the preset table, the kernel pattern and `preset_for`
#   **used to be copied here.** D-158 fixed the H100 variants in this copy and
#   the library's went unfixed — one job, two definitions (principle 2). They
#   are imported now, and the fixed content is what moved into the library.
from kernelrule.baselines.vendor import (
    CLUSTER_PAT,
    GPU_PRESETS,  # noqa: F401  — re-exported so callers of this module see it
    preset_for,
)
from kernelrule.baselines.vendor import (
    KERNEL_PAT as PAT,
)


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
                # ★ 2026-09-11 (D-166): **the instr check used to sit below
                #   this branch and never ran.** 0.1.0.27 returns a
                #   `GemmConfig` object, not a string, so every candidate took
                #   this path and `continue`d past the check — the printed
                #   `instr != (16,8,16) 0` meant "not counted", not "none".
                #   The cluster check above is fine: it runs on `str(kern)`,
                #   and the object's `__str__` carries the whole line.
                instr = (g.instr_tile_m, g.instr_tile_n, g.instr_tile_k)
                if instr != (16, 8, 16):
                    bad_instr += 1
                lst.append({"stages": g.stages,
                            "cta": [g.cta_tile_m, g.cta_tile_n, g.cta_tile_k],
                            "warp": [g.warp_tile_m, g.warp_tile_n,
                                     g.warp_tile_k],
                            # ★ recorded so the watch can be checked after the
                            #   fact instead of trusted
                            "instr": list(instr),
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
            instr = tuple(g[7:10])
            if instr != (16, 8, 16):
                bad_instr += 1
            lst.append({"stages": g[0], "cta": g[1:4], "warp": g[4:7],
                        "instr": list(instr),
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
