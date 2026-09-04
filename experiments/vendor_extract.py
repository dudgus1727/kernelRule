"""★ 벤더(nvMatmulHeuristics) 추천을 번들 형상에 대해 뽑는다. GPU 0회.

    python3 experiments/vendor_extract.py datasets/rtx-5090-sm_120-5bb6f403 \
        --env-hash 5bb6f403 --out datasets/baselines/vendor-5090-5bb6f403.json

라이브러리는 **프리셋으로 계산만** 한다 — GPU 가 필요 없다.

## ★ kernelTab 과 조건을 맞춘다

```
버전     0.1.0.27          다르면 추천이 달라진다
target   ★ CUTLASS         CUTLASS3 를 쓰면 cluster(1,4) 가 나온다 —
                           우리 표는 2.x 공간이다
layout   TN_ROW_MAJOR      전 형상 공통
precision HSS
프리셋    hw 이름에서 유도  (RTX_A6000 / RTX_5090 ...)
```

출력 형식은 `datasets/baselines/vendor-a6000-c63710df.json` 과 같다 —
`kernelrule.baselines.vendor.load_vendor` 가 그대로 읽는다.

## ★ 뽑은 뒤 감시한다

```
cluster == (1,1)      2.x 공간인가
instr   == (16,8,16)  f16 HMMA 인가
```

둘 중 하나라도 다르면 **다른 커널 공간의 추천**이고, 우리 표와 조인이
성립하지 않는다. 세어서 찍고, 있으면 종료 코드를 1로 낸다.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: GPU 이름 -> 프리셋. ★ kernelTab `scripts/baseline_vendor.py` 와 같은 표다.
GPU_PRESETS = {
    "rtx a6000": "RTX_A6000", "rtx 4090": "RTX_4090", "rtx 3090": "RTX_3090",
    "rtx 5090": "RTX_5090", "rtx 6000 ada": "RTX_6000_ADA",
    "a100": "A100_SXM_80GB", "a40": "A40_PCIE", "a30": "A30_PCIE",
    "a10": "A10_PCIE", "h100": "H100_SXM", "h200": "H200_SXM",
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
        f"'{name}' 에 대응하는 프리셋을 모른다. GPU_PRESETS 에 추가하라.")


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
        raise SystemExit(f"env_hash 불일치: {str(info['env_hash'])[:16]}")
    preset = preset_for(info["gpu_name"])
    print(f"{info['gpu_name']}  ->  프리셋 {preset}   "
          f"(target=CUTLASS, layout=TN_ROW_MAJOR, precision=HSS)")

    # ★ 형상은 **번들에서** 읽는다. kernelTab 의 all_shapes 를 부르면
    #   그쪽 env 를 읽게 되고 번들과 어긋날 수 있다.
    shapes = sorted({tuple(s) for layer in
                     (info.get("shape_layers") or {}).values() for s in layer})
    print(f"번들 형상 {len(shapes)}개")

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
    print(f"  -> {a.out}   ({len(out) - 1} 형상 x {a.count})")
    print(f"  ★ 감시: cluster != (1,1) {bad_cluster}건   "
          f"instr != (16,8,16) {bad_instr}건   파싱 실패 {n_fail}건")
    if bad_cluster or bad_instr:
        raise SystemExit(
            "★ 다른 커널 공간의 추천이 섞였다. 우리 표(CUTLASS 2.x, f16 "
            "HMMA)와 조인이 성립하지 않는다. target 을 확인하라.")


if __name__ == "__main__":
    main()
