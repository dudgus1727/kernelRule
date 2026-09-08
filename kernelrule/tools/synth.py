"""The synthetic performance-table generator (§22, appendix C).

A known physical structure is planted, and validation is whether the pipeline
**recovers** that structure. Because the answer is known, scorer bugs (a
flipped sign, an answer leak) can be caught — on a random table an answer
leak does not make regret look good, so it cannot be caught.

★ **This file does not import `kernelrule.features`** (§28).
  Using the same functions would make the pipeline validation a tautology.
  The physics is the same but the implementations are separate — here the
  wave term is written as `ceil(w)/w` while the feature side writes the waste
  ratio `tail_waste`. Algebraically the same physics, but different code and
  different coefficients.

★ **The generator cannot see the measured times.** Even when the grid comes
  from a real bundle it uses only `load_for_ranking` (answers removed). It is
  structurally impossible for it to copy the measurements.

## The structures planted

    #1 wave quantization        ceil(waves)/waves
    #2 occupancy (smem pressure) 1 + c * smem/smem_per_block
    #3 spilling                  1 + c * (spill > 0)
    #4 memory-bound damping      penalty -> 1 + damp*(penalty-1)
    #5 split-K reduction cost    1 + c * (split_k - 1)
    #6 mainloop warm-up          deep stages with a short K costs
    #7 ★ tile arithmetic intensity  a small tile reuses A/B less, so the
                                    mainloop is slower
    #8 two-stage pipeline        stages=2 is a different kernel family and
                                 hides latency less well
    #9 instruction count         the real work of one mainloop pass
   #10 register pressure         it cuts the resident blocks per SM
   #11 ★ partial-tile waste      a 128-row tile on M=1 throws away 99% of the
                                 work

## ★ The penalties attach **only to the compute side** of the roofline

Building `t = max(t_compute, t_memory)` first and multiplying the penalties
onto that makes **the shape x config interaction disappear** — which config
is good on which shape becomes almost shape-independent, and then one fixed
config is near-optimal on every shape and the static top-1 falls to 1.03.
There is nothing for a rule to learn from such a table.

The right order is this.

    t_compute *= (all the config penalties)   # a matter of work and compute
                                              # efficiency
    t_memory  *= (almost config-independent)  # bandwidth does not know the
                                              # tile shape
    t = max(...)

Then **the weakening of the config effect on memory-bound shapes (structure
#4) emerges naturally instead of being imposed.** An M=1 shape is easy not
"because a damping coefficient was multiplied in" but "because it hit the
bandwidth floor" — the same physics the measurements state.

**#7 is the main driver of difficulty.** The measured median difficulty of
1.671 means "a random config is 67% slower than the optimum", and neither the
wave term nor spilling produces that magnitude (spilling is on few kernels,
so it cannot move the median). Shrinking a tile from 128x128 to 32x64 cuts
FLOP/byte per tile threefold, and that is the loss of a typical config.

## Putting in both noise and quantisation matters (§22.3)

Fixing the noise to a constant removes the difficulty of small shapes and
makes the synthetic table fundamentally different from reality. Leaving out
the quantisation creates **an ordering that does not actually exist**, so the
scorer's tick handling cannot be validated.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["PRESETS", "Grid", "generate", "self_check", "synth_times"]

#: Difficulty presets (§22.5).
#:
#: With `struct` at 0 the time becomes **completely independent** of the
#: config — that is `null`, and it is the **answer-leak detector**. If any
#: rule's regret on that table falls well below 1.0, the answer is leaking
#: somewhere.
PRESETS: dict[str, dict] = {
    "easy":   dict(struct=1.5, noise_scale=0.4, mem_damp=0.05),
    "normal": dict(struct=1.0, noise_scale=1.0, mem_damp=0.15),
    "hard":   dict(struct=0.5, noise_scale=6.0, mem_damp=0.40),
    "null":   dict(struct=0.0, noise_scale=1.0, mem_damp=1.00),
}

# -- Coefficients of the generative model ------------------------------------
# ★ Not shared with the feature library (§28). The values are deliberately
#   different too.
_C_SMEM = 0.37          # the occupancy loss
_C_SPILL = 2.60         # the penalty on a spilling kernel
_C_SPLITK = 0.023       # the reduction cost per split-K partition
_C_WARMUP = 0.055       # the stages warm-up (when the mainloop is short)
_C_TILE_EXP = 0.34      # the tile arithmetic-intensity exponent (structure
                        # #7 — the main driver of difficulty)
_REF_TILE_EFF = 128.0   # 2*tm*tn/(tm+tn) of a 128x128 tile
_C_WARP_EXP = 0.20      # the warp-tile arithmetic-intensity exponent
                        # (structure #7b)
_REF_WARP_EFF = 64.0    # 2*wm*wn/(wm+wn) of a 64x64 warp tile
_C_PIPELINED = 0.21     # the stages=2 (MmaPipelined) penalty
_C_INST = 0.70          # the instruction count (structure #9). The axis GBDT
                        # ranked highly (§30.6)
_C_REG = 0.30           # register pressure (structure #10)
_LAUNCH_MS = 0.0132     # the launch overhead (§18.2's 13µs)

# Noise / tick — measured on the A6000 (§30.2). The preset's noise_scale is
# multiplied in.
_NOISE_A = 0.000374
_NOISE_B = 0.00044
_TICK_MS = 0.001024

_DTYPE_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "f8": 1}


@dataclass
class Grid:
    """The (shape, config) grid + the hardware. **There are no times.**

    `df` has the same shape as `load_for_ranking` — no answer columns.
    """

    df: pd.DataFrame
    env: dict
    hw: object
    source: str = ""
    extra_columns: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.df)

    # -- Taking only the grid from a real bundle --------------------------
    @classmethod
    def from_bundle(cls, ref: str | Path, *, env_hash: str,
                    shapes: list[tuple[int, int, int]] | None = None,
                    max_configs_per_shape: int | None = None,
                    seed: int = 0) -> Grid:
        """Uses a real bundle's (shape, config, static kernel attributes)
        grid as it is.

        ★ It calls only the `ranking` loader — **it cannot see the measured
        times.** That makes it structurally impossible for the synthetic
        table to copy the measurements.

        The real alignment structure (an a888 shape and an a448 shape have
        different candidates) comes along for free (§22.3).
        """
        from kerneltab.core.bundle import load_bundle
        from kerneltab.core.hardware import hardware_from_env

        b = load_bundle(ref, verify=True)
        if not str(b.env_hash).startswith(str(env_hash)):
            raise ValueError(
                f"env_hash mismatch: {env_hash!r} vs {b.env_hash[:16]!r}")
        df = b.ranking(ok_only=False, unknown_columns="ignore")
        env = b.env()
        if shapes is not None:
            want = {tuple(int(x) for x in s) for s in shapes}
            keep = [tuple(r) in want for r in
                    df[["M", "N", "K"]].to_numpy().tolist()]
            df = df[keep]
        if max_configs_per_shape:
            rng = np.random.default_rng(seed)
            parts = []
            for _, grp in df.groupby(["M", "N", "K"], sort=False):
                if len(grp) > max_configs_per_shape:
                    idx = np.sort(rng.choice(len(grp), max_configs_per_shape,
                                             replace=False))
                    grp = grp.iloc[idx]  # noqa: PLW2901 — deliberate cut
                parts.append(grp)
            df = pd.concat(parts, ignore_index=True)
        df = df.reset_index(drop=True)
        if df.empty:
            raise ValueError(
                "the grid is empty. Check the shapes filter.")
        return cls(df=df, env=env, hw=hardware_from_env(env),
                   source=f"bundle:{b.info.get('bundle_id')}")

    # -- Enumeration without a bundle -------------------------------------
    @classmethod
    def enumerate(cls, env: dict, *,
                  shapes: list[tuple[int, int, int]] | None = None,
                  max_kernels: int | None = None, seed: int = 0) -> Grid:
        """Enumerates directly through kernelTab's `shapes.py` /
        `backends/sm80.py`.

        Used when there is no bundle. The static kernel attributes
        (regs/spill/occupancy) are knowable only from a real build, so they
        are **modelled analytically** — and that fact is recorded in
        `BUNDLE.json`'s `synthetic.build_attrs`.
        """
        from kerneltab.backends.sm80 import Sm80Backend
        from kerneltab.core.config import alignments_for
        from kerneltab.core.hardware import hardware_from_env
        from kerneltab.core.shapes import all_shapes
        from kerneltab.core.types import KernelConfig, Problem

        hw = hardware_from_env(env)
        be = Sm80Backend()
        probs = ([Problem(*s) for s in shapes] if shapes is not None
                 else all_shapes(hw))
        ext_pairs = be.enumerate_ext(hw)
        rng = np.random.default_rng(seed)
        if max_kernels and len(ext_pairs) > max_kernels:
            idx = np.sort(rng.choice(len(ext_pairs), max_kernels,
                                     replace=False))
            ext_pairs = [ext_pairs[i] for i in idx]

        rows: list[dict] = []
        for p in probs:
            aa, ab, ac = alignments_for(p)
            for (tm, tn, tk), ext in ext_pairs:
                cfg = KernelConfig(tile_m=tm, tile_n=tn, tile_k=tk,
                                   align_a=aa, align_b=ab, align_c=ac,
                                   arch=hw.arch, ext=ext)
                if not be.is_valid_kernel(cfg, hw, 2):
                    continue
                attrs = _build_attrs(be, cfg, hw, ext)
                kid = be.kernel_id(cfg)
                for rc in be.enumerate_runtime(p, cfg):
                    rows.append({
                        "M": p.M, "N": p.N, "K": p.K, "dtype": p.dtype,
                        "acc_dtype": p.acc_dtype, "layout_a": p.layout_a,
                        "layout_b": p.layout_b, "layout_c": p.layout_c,
                        "tile_m": tm, "tile_n": tn, "tile_k": tk,
                        "align_a": aa, "align_b": ab, "align_c": ac,
                        "split_k": rc.split_k, "split_k_mode": rc.split_k_mode,
                        "kernel_id": kid, "arch": hw.arch,
                        "ext_warp_m": ext.warp_m, "ext_warp_n": ext.warp_n,
                        "ext_warp_k": ext.warp_k, "ext_stages": ext.stages,
                        "ext_swizzle_type": ext.swizzle_type,
                        "ext_swizzle_n": ext.swizzle_n,
                        "pipeline_kind": be.pipeline_kind(cfg),
                        **attrs,
                    })
        if not rows:
            raise ValueError("the enumeration is empty. Check shapes / "
                             "max_kernels.")
        return cls(df=pd.DataFrame(rows), env=env, hw=hw,
                   source="enumerate:sm80",
                   extra_columns={"build_attrs": "analytic"})


def _build_attrs(be, cfg, hw, ext) -> dict:
    """An **analytic model** of the values only a build can give. For the
    enumeration mode only.

    A real table carries the `-Xptxas -v` measurements. Here they are
    approximated — the purpose of the synthetic table is pipeline
    validation, not kernel prediction.
    """
    warps_m = max(1, cfg.tile_m // ext.warp_m)
    warps_n = max(1, cfg.tile_n // ext.warp_n)
    warps_k = max(1, cfg.tile_k // ext.warp_k) if ext.warp_k else 1
    threads = warps_m * warps_n * warps_k * 32
    accum = (ext.warp_m * ext.warp_n) // 32   # one fp32 accumulator/register
    overhead = 28 + 4 * ext.stages
    want = accum + overhead
    regs = min(255, want)
    spill = max(0, want - 255) * 4
    smem = be.smem_bytes(cfg, 2)
    by_smem = hw.smem_per_block // max(1, smem)
    by_regs = hw.regs_per_sm // max(1, regs * threads)
    by_thr = hw.max_threads_per_sm // max(1, threads)
    return {
        "threads": int(threads), "regs_per_thread": int(regs),
        "smem_dynamic": int(smem), "smem_static_bytes": int(smem),
        "spill_stores": int(spill), "spill_loads": int(spill),
        "max_blocks_per_sm": int(max(0, min(by_smem, by_regs, by_thr))),
        "has_spill": bool(spill > 0),
        "regs_total_per_block": int(regs * threads),
        "inst_total": int(2000 + 40 * accum + 120 * ext.stages),
        "launchable": True,
    }


# ---------------------------------------------------------------------------
# The physics — ★ it does not use features/ (§28)
# ---------------------------------------------------------------------------
def synth_times(grid: Grid, preset: str = "normal", *, seed: int = 0,
                return_parts: bool = False):
    """Synthesises times onto the grid. It is vectorised."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset!r}. {sorted(PRESETS)}")
    ps = PRESETS[preset]
    df = grid.df
    hw = grid.hw
    rng = np.random.default_rng(seed)

    M = df["M"].to_numpy(np.float64)
    N = df["N"].to_numpy(np.float64)
    K = df["K"].to_numpy(np.float64)
    tm = df["tile_m"].to_numpy(np.float64)
    tn = df["tile_n"].to_numpy(np.float64)
    tk = df["tile_k"].to_numpy(np.float64)
    sk = df["split_k"].to_numpy(np.float64)
    eb = np.asarray([_DTYPE_BYTES.get(str(d), 2) for d in df["dtype"]],
                    dtype=np.float64)

    smem = _col(df, ("smem_dynamic", "smem_bytes", "smem_static_bytes"))
    spill = (_col(df, ("spill_stores",), 0.0) + _col(df, ("spill_loads",), 0.0))
    occ = np.maximum(1.0, _col(df, ("max_blocks_per_sm",), 1.0))
    stages = _col(df, ("ext_stages",), 3.0)

    # -- 1. the two roofline terms are kept **separate** ------------------
    flops = 2.0 * M * N * K
    bytes_moved = eb * (M * K + K * N + M * N)
    t_compute = flops / (hw.peak_tflops_f16 * 1e12) * 1e3        # ms
    t_memory = bytes_moved / (hw.bandwidth_gbps * 1e9) * 1e3     # ms

    # -- 2. wave quantization (structure #1) ------------------------------
    # ★ The same physics as the feature side's tail_waste, algebraically,
    #   written in a different form.
    tiles_m = np.ceil(M / tm)
    tiles_n = np.ceil(N / tn)
    tiles = tiles_m * tiles_n * sk
    waves = tiles / (hw.sm_count * occ)
    wave_pen = np.ceil(waves) / np.maximum(waves, 1e-12)

    # -- 3. ★ partial-tile waste (structure #11) — the main driver of the
    #    shape x config interaction ---------------------------------------
    # When a tile crosses the shape boundary it computes that part **too**.
    # A 128-row tile on M=1 throws away 99.2% of the work. That is why a
    # small M prefers a small tile_m.
    edge_pen = (tiles_m * tm / M) * (tiles_n * tn / N)

    # -- 4. occupancy (structure #2) --------------------------------------
    smem_pen = 1.0 + _C_SMEM * np.clip(smem / hw.smem_per_block, 0.0, 1.5)

    # -- 5. spilling (structure #3) ---------------------------------------
    spill_pen = 1.0 + _C_SPILL * (spill > 0)

    # -- 6. the split-K reduction (structure #5) --------------------------
    sk_pen = 1.0 + _C_SPLITK * (sk - 1.0)

    # -- 7. the mainloop warm-up (structure #6) ---------------------------
    iters = np.maximum(1.0, K / (tk * sk))
    warm_pen = 1.0 + _C_WARMUP * np.clip(stages / iters, 0.0, 2.0)

    # -- 8. tile/warp arithmetic intensity (structure #7) -----------------
    # ★ Above the reference tile the gain **saturates**. Beyond it, A/B smem
    #   traffic is no longer the bottleneck and registers/epilogue are.
    tile_eff = 2.0 * tm * tn / np.maximum(tm + tn, 1.0)
    tile_pen = np.power(_REF_TILE_EFF / np.clip(tile_eff, 1.0, _REF_TILE_EFF),
                        _C_TILE_EXP)
    wm = _col(df, ("ext_warp_m",), 64.0)
    wn = _col(df, ("ext_warp_n",), 64.0)
    warp_eff = 2.0 * wm * wn / np.maximum(wm + wn, 1.0)
    warp_pen = np.power(_REF_WARP_EFF / np.clip(warp_eff, 1.0, _REF_WARP_EFF),
                        _C_WARP_EXP)

    # -- 9. the two-stage pipeline (structure #8) -------------------------
    pipe_pen = 1.0 + _C_PIPELINED * (stages <= 2)

    # -- 10. instruction count / register pressure (structures #9, #10) ---
    # A **continuous** axis that differs slightly per kernel. Without it the
    # times fall on only a few discrete combinations and tick collisions
    # become far worse than in reality.
    inst = _col(df, ("inst_total",), 0.0)
    inst_ref = np.median(inst[inst > 0]) if np.any(inst > 0) else 1.0
    inst_pen = 1.0 + _C_INST * np.clip(inst / max(inst_ref, 1.0) - 1.0, -0.6, 3.0)
    regs_tot = _col(df, ("regs_total_per_block",), 0.0)
    reg_pen = 1.0 + _C_REG * np.clip(regs_tot / hw.regs_per_sm, 0.0, 2.0)

    penalty = (wave_pen * edge_pen * smem_pen * spill_pen * sk_pen * warm_pen
               * tile_pen * warp_pen * pipe_pen * inst_pen * reg_pen)

    # -- 11. structure strength (the preset) ------------------------------
    # struct=0 -> the penalty is exactly 1 -> the time becomes independent of
    # the config (null)
    penalty = np.power(penalty, float(ps["struct"]))

    # -- 12. multiplied onto the compute side only. The memory floor
    #    barely knows the config ------------------------------------------
    damp = float(ps["mem_damp"])
    t = np.maximum(t_compute * penalty,
                   t_memory * (1.0 + damp * (penalty - 1.0))) + _LAUNCH_MS

    ai = flops / np.maximum(bytes_moved, 1.0)
    ridge = (hw.peak_tflops_f16 * 1e12) / (hw.bandwidth_gbps * 1e9)
    mem_bound = ai < ridge

    # -- 13. measurement noise — ★ it depends on the kernel time (§30.2) --
    sigma = (_NOISE_A / np.maximum(t, 1e-9) + _NOISE_B) * float(ps["noise_scale"])
    t_noisy = t * (1.0 + rng.normal(0.0, 1.0, size=t.shape) * sigma)
    t_noisy = np.maximum(t_noisy, _TICK_MS)

    # -- 14. ★ timer quantisation (§30.2) --------------------------------
    # Without it the synthetic table gains an ordering that does not actually
    # exist.
    t_q = np.round(t_noisy / _TICK_MS) * _TICK_MS

    if return_parts:
        return t_q, {"t_compute": t_compute, "t_memory": t_memory,
                     "penalty": penalty, "waves": waves, "edge_pen": edge_pen,
                     "mem_bound": mem_bound, "t_clean": t, "sigma": sigma}
    return t_q


def _col(df, names, default=None) -> np.ndarray:
    for n in names:
        if n in df.columns:
            return df[n].fillna(0).to_numpy(np.float64)
    if default is None:
        raise KeyError(f"the grid has none of {names}.")
    return np.full(len(df), float(default))


# ---------------------------------------------------------------------------
# Saving as a bundle
# ---------------------------------------------------------------------------
def generate(preset: str, seed: int, out: str | Path, grid: Grid, *,
             bundle_id: str | None = None) -> Path:
    """Writes the synthetic table **in the same format as a real bundle**
    (§22.3).

    The same file format is what validates the loader, the adapter and
    `PerfTable`.

    ★ `SYNTHETIC` is nailed into the `bundle_id` and the directory name
    (§22.6). It blocks, at the level of names, any path by which a synthetic
    artefact could be mistaken for a real result.
    """
    out = Path(out)
    bid = bundle_id or f"SYNTHETIC-{preset}-s{seed}"
    if "SYNTHETIC" not in bid:
        raise ValueError("a synthetic bundle's bundle_id must contain "
                         "SYNTHETIC (§22.6).")
    path = out / bid
    path.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    t, parts = synth_times(grid, preset, seed=seed, return_parts=True)
    df = grid.df.copy()
    df["time_ms"] = t

    # The answer/outcome columns are filled in the same shape as a real
    # table.
    sig = parts["sigma"] * t
    df["time_std_ms"] = sig
    df["time_min_ms"] = np.maximum(t - sig, _TICK_MS)
    df["time_max_ms"] = t + sig
    df["n_reps"] = 30
    df["outlier_frac"] = 0.0
    df["status"] = "ok"
    df["error"] = ""
    df["max_rel_error"] = 0.0
    df["actual_split_k"] = df["split_k"]
    df["env_hash"] = _synth_env_hash(bid)
    for c, v in (("sm_clock_mhz", 1350.0), ("mem_clock_mhz", 7601.0),
                 ("gpu_temp_c", 55.0), ("power_w", 200.0),
                 ("soak_elapsed_s", 0.0), ("drift_ratio", 1.0)):
        df[c] = v
    df["timestamp"] = "1970-01-01T00:00:00Z"

    # Shape-level derived answers (the same definitions kernelTab computes
    # on export)
    g = df.groupby(["M", "N", "K"], sort=False)["time_ms"]
    best = g.transform("min")
    df["difficulty"] = g.transform("median") / best
    flops = 2.0 * df.M.astype(float) * df.N.astype(float) * df.K.astype(float)
    df["tflops"] = flops / (df["time_ms"] * 1e-3) / 1e12
    df["frac_of_peak"] = df["tflops"] / float(grid.hw.peak_tflops_f16)
    df["cublas_ms"] = np.nan
    df["vs_cublas"] = np.nan

    tbl = path / "table.parquet"
    df.to_parquet(tbl, index=False)
    (path / "env.json").write_text(json.dumps(grid.env, indent=2))

    n_shapes = int(df.groupby(["M", "N", "K"], sort=False).ngroups)
    info = {
        "bundle_id": bid,
        "schema_version": 2,
        "synthetic": {
            "preset": preset, "seed": seed,
            "coefficients": PRESETS[preset],
            "grid_source": grid.source,
            "generator": "kernelrule.tools.synth",
            "warning": ("this is a synthetic table. Do not report "
                        "performance numbers from it (§28). Do not carry a "
                        "rule from this table over to a real one."),
            **grid.extra_columns,
        },
        "gpu_name": grid.env.get("hardware", {}).get("name", "SYNTHETIC"),
        "arch": grid.hw.arch,
        "sm_count": grid.hw.sm_count,
        "env_hash": _synth_env_hash(bid),
        "n_shapes": n_shapes,
        "n_kernels": int(df.kernel_id.nunique()),
        "n_rows": int(len(df)),
        "peak_tflops_f16_effective": float(grid.hw.peak_tflops_f16),
        "bandwidth_gbps_effective": float(grid.hw.bandwidth_gbps),
        # ★ It is emitted as schema_version 2 — because it really carries
        #   tick_ms.
        "noise_floor": {
            "sigma_abs_ms": _NOISE_A, "sigma_rel": _NOISE_B,
            "tick_ms": _TICK_MS,
            "model": "noise_floor(t) = max(sigma_abs_ms/t + sigma_rel, tick_ms/t)",
            "source": "kernelrule.tools.synth (uses the measured A6000 "
                      "coefficients as they are)",
        },
        "shape_layers": _layers(df, grid),
        "created_seconds": round(time.perf_counter() - t0, 2),
    }
    files = {}
    for name in ("table.parquet", "env.json"):
        f = path / name
        files[name] = {"bytes": f.stat().st_size, "sha256": _sha256(f)}
    info["files"] = files
    (path / "BUNDLE.json").write_text(json.dumps(info, indent=2,
                                                 ensure_ascii=False))
    return path


def _layers(df, grid: Grid) -> dict:
    """The shape layers. If the grid came from a real bundle, its layers
    are used as they are."""
    try:
        from kerneltab.core.shapes import all_layers
        out = {}
        present = {tuple(r) for r in df[["M", "N", "K"]].drop_duplicates()
                   .to_numpy().tolist()}
        for name, probs in all_layers(grid.hw).items():
            rows = [[p.M, p.N, p.K] for p in probs
                    if (p.M, p.N, p.K) in present]
            if rows:
                out[name] = rows
        return out
    except Exception:                                  # pragma: no cover
        return {}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _synth_env_hash(bid: str) -> str:
    """The synthetic table's `env_hash`. A prefix is nailed on so that it
    **can never mix** with a real condition."""
    return "5y47he71c" + hashlib.sha256(bid.encode()).hexdigest()[:55]


# ---------------------------------------------------------------------------
# Self-check (§22.3, appendix C)
# ---------------------------------------------------------------------------
def self_check(path: str | Path) -> dict:
    """Checks that the generated table meets the target statistics.

    ⚠️ If the generator drifts off target, everything developed on top of it
    is detached from reality. It is pinned by a test.
    """
    from kerneltab.core.bundle import load_bundle

    b = load_bundle(path, verify=True)
    syn = b.info.get("synthetic") or {}
    preset = syn.get("preset", "?")
    y = b.scoring(ok_only=False)

    g = y.groupby(["M", "N", "K"])["time_ms"]
    best = g.min()
    med = g.median()
    diff = (med / best).to_numpy()
    n_cand = g.size().to_numpy()
    n_distinct = g.nunique().to_numpy()

    # Tie density — whether the tick dominates on short shapes (§22.3)
    short = best.to_numpy() < 0.05
    return {
        "preset": preset,
        "n_shapes": int(len(best)),
        "n_rows": int(len(y)),
        "difficulty_median": float(np.median(diff)),
        "difficulty_min": float(diff.min()),
        "difficulty_max": float(diff.max()),
        "best_ms_median": float(np.median(best.to_numpy())),
        "frac_small_shapes": float((best.to_numpy() < 0.5).mean()),
        "distinct_time_frac_median": float(np.median(n_distinct / n_cand)),
        "distinct_time_frac_short": (
            float(np.median((n_distinct / n_cand)[short])) if short.any()
            else float("nan")),
    }
