"""Vendor heuristic baseline — **this is the gate** (§30.6b).

Nobody uses a static top-1 in practice, so it is a weak baseline. The real
competitor in the **runtime dispatch** scenario — pick one config without
measuring — is nvMatmulHeuristics (targeting CUTLASS).

This differs from `C/A` (against cuBLAS). C/A compares against a **different
kernel family** in cuBLAS, so implementation differences mix in. Here we look
only at heuristic ranking quality **inside the same CUTLASS kernel space**.

## It has two stages

    extract   ★ `experiments/vendor_extract.py` — pull per-shape top-k out of
              nvMatmulHeuristics. ★ It uses no GPU. Only a GPU **preset** (a
              CPU prediction model).
    score     this module — map that output onto our table and score it.

⚠️ 2026-09-11 (D-166): `extract()` **used to live here too**, and nothing
called it: every recorded extraction went through `experiments/`. Two
implementations of one job is how `GPU_PRESETS` came to be fixed in one of
them and not the other (D-158's H100 variants). The copy here is gone and the
preset table below is now **the single one** — `experiments/vendor_extract.py`
imports it.

Saving the `extract` output (`vendor.json`) makes later rescoring a
table-only operation. kernelTab did not commit it, so recomputation needed
the network (report C-2).

## ★ The `status` filter

kernelTab's original computation used only `status == "ok"`
(`baseline_vendor.py:143`). Our representative value is **all statuses + the
union cover**, so it is recomputed here (§30.5b).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = ["GPU_PRESETS", "preset_for", "KERNEL_PAT", "CLUSTER_PAT",
           "load_vendor", "vendor_order_fn", "match_report"]

#: GPU name -> nvMatmulHeuristics preset. Derived from `hw.name`.
#:
#: ★ **This is the only copy** (D-166). `experiments/vendor_extract.py` had
#: its own, D-158 fixed the H100 variants in that one, and this one kept
#: mapping every H100 to `H100_SXM` — so a re-extraction through this module
#: would have put an SXM preset on an NVL bundle with no error and no
#: warning. Nothing re-extracted through here (the four committed files came
#: from `experiments/`), so no number was wrong; what existed was the trap.
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

#: The kernel string nvMatmulHeuristics prints. ★ One copy (D-166) — the
#: extractor imports it.
KERNEL_PAT = re.compile(
    r"stages\((\d+)\)\s+cta\((\d+) (\d+) (\d+)\)\s+warp\((\d+) (\d+) (\d+)\)"
    r"\s+instr\((\d+) (\d+) (\d+)\)\s+splitK\((\d+)\)\s+swizz\((\d+)\)"
    r"\s+ctaOrder\((\d+)\)")
CLUSTER_PAT = re.compile(r"cluster\((\d+) (\d+)(?: (\d+))?\)")

#: Per-axis weights for the nearest mapping. The cta tile weighs most.
#: (tm, tn, tk, wm, wn, wk, stages, swizzle, split_k, mode)
_NEAR_W = (3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 0.5, 2.0, 0.5)


def preset_for(name: str) -> str:
    low = name.lower().replace("nvidia", "").strip()
    for k, v in sorted(GPU_PRESETS.items(), key=lambda kv: -len(kv[0])):
        if k in low:
            return v
    raise SystemExit(
        f"no nvMatmulHeuristics preset is known for {name!r}. "
        "Add it to GPU_PRESETS.")


# ---------------------------------------------------------------------------
# Stage 2 — scoring (main environment)
# ---------------------------------------------------------------------------
def load_vendor(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    meta = d.pop("_meta", {})
    return {"meta": meta, "by_shape": d}


def _cand_keys(table, p):
    """Axis vector of a candidate, in the same coordinates as a vendor
    config."""
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
    """Vendor config -> the same coordinates. In CUTLASS 2.x split_k
    defaults to serial."""
    return (cf["cta"][0], cf["cta"][1], cf["cta"][2],
            cf["warp"][0], cf["warp"][1], cf["warp"][2], cf["stages"],
            cf["swizzle"] if cf.get("cta_order", 0) == 0 else 0,
            cf["split_k"], 0)


def vendor_order_fn(table, vendor: dict, *, mapping: str = "nearest"):
    """Turn the vendor's per-shape recommendation into an `order_fn`.

    `mapping`:
        "nearest" — a combination absent from our space is replaced by the
                    **nearest measured one in log-axis distance**. 100% cover.
        "strict"  — exact matches only. Without one, that shape is pushed back.

    kernelTab measured the two at 1.081 / 1.088, so the mapping method does
    not change the conclusion. Both are computed and reported side by side.
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
    """How much of the strict mapping lands. Reported because it is a point
    that could change the conclusion."""
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
