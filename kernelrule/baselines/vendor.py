"""벤더 휴리스틱 베이스라인 — **이것이 관문이다** (§30.6b).

정적 top-1 은 실무에서 아무도 안 쓰므로 베이스라인으로 약하다. 측정 없이
config 하나를 고르는 **런타임 디스패치** 시나리오의 실질적 상대는
nvMatmulHeuristics(CUTLASS 타깃)다.

`C/A`(cuBLAS 대비)와 다르다. C/A 는 cuBLAS 의 **다른 커널 계열** 대비라
구현 차이가 섞인다. 여기서는 **같은 CUTLASS 커널 공간 안에서** 휴리스틱의
순위 품질만 본다.

## 두 단계로 나뉜다

    extract   nvMatmulHeuristics 에서 형상별 top-k 를 뽑는다.
              **별도 venv 에서 돌린다** — 이 저장소 환경을 오염시키지 않는다.
              ★ GPU 를 쓰지 않는다. GPU **프리셋**만 쓴다 (CPU 예측 모델).
    score     그 산출물을 우리 표에 매핑해 채점한다. 이쪽은 본 환경.

`extract` 산출물(`vendor.json`)을 저장해 두면 이후 재채점이 표만으로 된다.
kernelTab 은 이것을 커밋하지 않아서 재계산에 네트워크가 필요했다 (C-2 보고).

## ★ `status` 필터

kernelTab 의 원 계산은 `status == "ok"` 만 썼다 (`baseline_vendor.py:143`).
정본은 **전체 status + 합집합 덮개**이므로 여기서 다시 계산한다 (§30.5b).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

__all__ = ["GPU_PRESETS", "extract", "load_vendor", "vendor_order_fn",
           "match_report"]

#: GPU 이름 -> nvMatmulHeuristics 프리셋. `hw.name` 에서 유도한다.
GPU_PRESETS = {
    "rtx a6000": "RTX_A6000", "rtx 4090": "RTX_4090", "rtx 3090": "RTX_3090",
    "rtx 5090": "RTX_5090", "rtx 6000 ada": "RTX_6000_ADA",
    "a100": "A100_SXM_80GB", "a40": "A40_PCIE", "a30": "A30_PCIE",
    "a10": "A10_PCIE", "h100": "H100_SXM", "h200": "H200_SXM",
    "l40s": "L40S", "l40": "L40", "l4": "L4", "b200": "B200",
}

_PAT = re.compile(
    r"stages\((\d+)\)\s+cta\((\d+) (\d+) (\d+)\)\s+warp\((\d+) (\d+) (\d+)\)"
    r"\s+instr\((\d+) (\d+) (\d+)\)\s+splitK\((\d+)\)\s+swizz\((\d+)\)"
    r"\s+ctaOrder\((\d+)\)")

#: 최근접 매핑의 축별 가중치. cta tile 을 가장 중시한다.
#: (tm, tn, tk, wm, wn, wk, stages, swizzle, split_k, mode)
_NEAR_W = (3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 0.5, 2.0, 0.5)


def preset_for(name: str) -> str:
    low = name.lower().replace("nvidia", "").strip()
    for k, v in sorted(GPU_PRESETS.items(), key=lambda kv: -len(kv[0])):
        if k in low:
            return v
    raise SystemExit(
        f"{name!r} 에 대응하는 nvMatmulHeuristics 프리셋을 모른다. "
        "GPU_PRESETS 에 추가하라.")


# ---------------------------------------------------------------------------
# 1단계 — 추출 (별도 venv. stdlib + nvMatmulHeuristics 만 쓴다)
# ---------------------------------------------------------------------------
def extract(bundle_dir: str | Path, out_path: str | Path,
            count: int = 8) -> int:
    """번들의 형상 목록에 대해 휴리스틱 top-`count` 를 뽑는다.

    ★ 이 함수는 `kernelrule` / `kerneltab` 을 import 하지 않는다 — 격리된
    venv 에서 돌아야 하기 때문이다. 형상과 하드웨어는 번들 파일에서 직접 읽는다.
    """
    import nvMatmulHeuristics as nv

    bundle_dir = Path(bundle_dir)
    info = json.loads((bundle_dir / "BUNDLE.json").read_text())
    gpu_name = info["gpu_name"]
    preset = preset_for(gpu_name)
    shapes = sorted({tuple(int(x) for x in s)
                     for rows in info["shape_layers"].values() for s in rows})
    print(f"{gpu_name} -> 프리셋 {preset}, 형상 {len(shapes)}개, top-{count}")

    h = nv.NvMatmulHeuristicsInterface(nv.NvMatmulHeuristicsTarget.CUTLASS,
                                       precision="HSS")
    hd = h.createHardwareDescriptor()
    h.setHardwarePredefinedGpu(hd, getattr(nv.NvMatmulHeuristicsNvidiaGpu,
                                           preset))
    layout = nv.NvMatmulHeuristicsMatmulLayout.TN_ROW_MAJOR

    out = {"_meta": {"gpu": gpu_name, "preset": preset,
                     "env_hash": info["env_hash"], "count": count,
                     "bundle_id": info["bundle_id"],
                     "layout": "TN_ROW_MAJOR", "precision": "HSS",
                     "target": "CUTLASS"}}
    for (M, N, K) in shapes:
        cfgs = h.get_with_mnk(M, N, K, layout, count, hd)
        lst = []
        for c in cfgs:
            kern, rt = c["kernel"], c.get("runtime")
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
            mo = _PAT.search(kern)
            if not mo:
                lst.append({"raw": kern, "parse_fail": True})
                continue
            g = [int(x) for x in mo.groups()]
            lst.append({"stages": g[0], "cta": g[1:4], "warp": g[4:7],
                        "split_k": g[10], "swizzle": g[11], "cta_order": g[12],
                        "pred_ms": (rt or 0) * 1000.0})
        out[f"{M}x{N}x{K}"] = lst
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(f"{len(out) - 1} 형상 -> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# 2단계 — 채점 (본 환경)
# ---------------------------------------------------------------------------
def load_vendor(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    meta = d.pop("_meta", {})
    return {"meta": meta, "by_shape": d}


def _cand_keys(table, p):
    """후보의 축 벡터. 벤더 config 와 같은 좌표계로 만든다."""
    import numpy as np

    df = table.frame_for(p)
    swz = df["ext_swizzle_n"].to_numpy(np.float64) if "ext_swizzle_n" in df \
        else np.ones(len(df))
    ident = (df["ext_swizzle_type"].astype(str).to_numpy() == "identity") \
        if "ext_swizzle_type" in df else np.ones(len(df), dtype=bool)
    return np.stack([
        df["tile_m"].to_numpy(np.float64), df["tile_n"].to_numpy(np.float64),
        df["tile_k"].to_numpy(np.float64),
        df["ext_warp_m"].to_numpy(np.float64),
        df["ext_warp_n"].to_numpy(np.float64),
        df["ext_warp_k"].to_numpy(np.float64),
        df["ext_stages"].to_numpy(np.float64),
        np.where(ident, swz, 0.0),
        df["split_k"].to_numpy(np.float64),
        (df["split_k_mode"].astype(str).to_numpy() != "serial").astype(float),
    ], axis=1)


def _vendor_vec(cf) -> tuple:
    """벤더 config -> 같은 좌표계. CUTLASS 2.x 의 split_k 기본은 serial 이다."""
    return (cf["cta"][0], cf["cta"][1], cf["cta"][2],
            cf["warp"][0], cf["warp"][1], cf["warp"][2], cf["stages"],
            cf["swizzle"] if cf.get("cta_order", 0) == 0 else 0,
            cf["split_k"], 0)


def vendor_order_fn(table, vendor: dict, *, mapping: str = "nearest"):
    """벤더의 형상별 추천을 `order_fn` 으로 만든다.

    `mapping`:
        "nearest" — 우리 공간에 없는 조합은 **축 로그거리로 가장 가까운**
                    측정치로 대체한다. 덮개 100%.
        "strict"  — 정확히 일치하는 것만. 없으면 그 형상은 뒤로 민다.

    kernelTab 실측에서 두 값이 1.081 / 1.088 로 일치했으므로 매핑 방식이
    결론을 바꾸지 않는다. 둘 다 계산해 병기한다.
    """
    import numpy as np

    by_shape = vendor["by_shape"]
    cache: dict = {}

    def order_fn(p, cand):
        key = f"{p.M}x{p.N}x{p.K}"
        got = cache.get((key, mapping))
        if got is None:
            lst = [c for c in by_shape.get(key, []) if not c.get("parse_fail")]
            keys = _cand_keys(table, p)
            lk = np.log2(np.maximum(keys, 1.0) + 1.0)
            picks: list[int] = []
            for cf in lst:
                v = np.asarray(_vendor_vec(cf), dtype=np.float64)
                exact = np.flatnonzero((keys == v).all(axis=1))
                if exact.size:
                    idx = int(exact[0])
                elif mapping == "nearest":
                    lv = np.log2(np.maximum(v, 1.0) + 1.0)
                    d = (np.asarray(_NEAR_W) * (lk - lv) ** 2).sum(axis=1)
                    idx = int(np.argmin(d))
                else:
                    continue
                if idx not in picks:
                    picks.append(idx)
            rest = [i for i in np.argsort(cand.tiebreak) if i not in set(picks)]
            got = np.asarray(picks + rest, dtype=np.int64)
            cache[(key, mapping)] = got
        return got

    return order_fn


def match_report(table, vendor: dict) -> dict:
    """엄격 매핑이 얼마나 되는가. 결론을 바꿀 수 있는 지점이라 보고한다."""
    import numpy as np

    n_exact = n_tot = 0
    missing_shapes = []
    for p in table.shapes():
        key = f"{p.M}x{p.N}x{p.K}"
        lst = [c for c in vendor["by_shape"].get(key, [])
               if not c.get("parse_fail")]
        if not lst:
            missing_shapes.append(key)
            continue
        keys = _cand_keys(table, p)
        for cf in lst:
            n_tot += 1
            v = np.asarray(_vendor_vec(cf), dtype=np.float64)
            if (keys == v).all(axis=1).any():
                n_exact += 1
    return {"exact": n_exact, "total": n_tot,
            "frac": n_exact / max(1, n_tot),
            "shapes_without_vendor": missing_shapes}
