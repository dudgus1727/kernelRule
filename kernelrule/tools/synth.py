"""합성 성능 표 생성기 (§22, 부록 C).

알려진 물리 구조를 심고, 파이프라인이 그 구조를 **되찾아내는지**로 검증한다.
정답을 알고 있으므로 채점기 버그(부호 반전, 정답 누출)를 잡을 수 있다 —
무작위 표에서는 정답 누출이 있어도 regret 이 좋아 보이지 않아 못 잡는다.

★ **이 파일은 `kernelrule.features` 를 import 하지 않는다** (§28).
  같은 함수를 쓰면 파이프라인 검증이 동어반복이 된다. 물리는 같되 구현을
  분리한다 — 여기서는 wave 항을 `ceil(w)/w` 로 쓰고, 피처 쪽은 낭비 비율
  `tail_waste` 로 쓴다. 대수적으로 같은 물리이지만 코드도 계수도 다르다.

★ **생성기는 실측 시간을 볼 수 없다.** 격자를 실제 번들에서 가져올 때도
  `load_for_ranking`(정답 제거)만 쓴다. 구조적으로 실측을 베낄 수 없다.

## 심는 구조

    #1 wave quantization       ceil(waves)/waves
    #2 occupancy (smem 압력)    1 + c * smem/smem_per_block
    #3 스필                     1 + c * (spill > 0)
    #4 메모리 바운드 감쇠        penalty -> 1 + damp*(penalty-1)
    #5 split-K 리덕션 비용       1 + c * (split_k - 1)
    #6 mainloop 워밍업          stages 가 깊고 K 가 짧으면 손해
    #7 ★ 타일 연산 강도          작은 타일은 A/B 재사용이 적어 mainloop 이 느리다
    #8 2단 파이프라인            stages=2 는 다른 커널 계열이고 지연 은닉이 약하다
    #9 명령어 수                 mainloop 한 번의 실제 작업량
   #10 레지스터 압력             SM 당 상주 블록 수를 깎는다
   #11 ★ 부분 타일 낭비          M=1 에 128행 타일을 쓰면 일의 99%가 버려진다

## ★ 벌점은 roofline 의 **연산 쪽에만** 붙는다

`t = max(t_compute, t_memory)` 를 먼저 만들고 거기에 벌점을 곱하면, 형상과
config 의 **상호작용이 사라진다** — 어떤 config 가 어느 형상에서 좋은지가
거의 형상에 무관해지고, 그러면 고정 config 하나가 모든 형상에서 최적에 가까워
정적 top-1 이 1.03 까지 내려간다. 그런 표에서는 규칙이 배울 것이 없다.

맞는 순서는 이것이다.

    t_compute *= (config 벌점 전부)      # 연산량과 연산 효율의 문제
    t_memory  *= (거의 config 무관)       # 대역폭은 타일 모양을 모른다
    t = max(...)

이러면 **메모리 바운드 형상에서 config 영향이 약해지는 것(구조 #4)이
부과되는 대신 자연히 나온다.** M=1 형상이 쉬운 이유가 "감쇠 계수를 곱해서"가
아니라 "대역폭 바닥에 닿아서" 가 된다 — 실측이 말하는 물리와 같다.

**#7 이 난이도의 주 동인이다.** 실측 난이도 중앙 1.671 은 "무작위 config 가
최적보다 67% 느리다" 는 뜻인데, wave 항이나 스필로는 그 크기가 안 나온다
(스필은 소수 커널에만 있어 중앙값을 못 움직인다). 타일이 128x128 에서 32x64 로
작아지면 타일당 FLOP/바이트가 3배 줄고, 그것이 전형적인 config 의 손해다.

## 노이즈와 양자화를 둘 다 넣는 것이 중요하다 (§22.3)

노이즈를 고정값으로 두면 작은 형상의 어려움이 사라져 합성 표가 실제와
근본적으로 달라진다. 양자화를 빼면 **실제로는 존재하지 않는 순위**가 생겨
채점기의 눈금 처리를 검증할 수 없다.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["PRESETS", "Grid", "generate", "self_check", "synth_times"]

#: 난이도 프리셋 (§22.5).
#:
#: `struct` 가 0 이면 시간이 config 와 **완전히 무관**해진다 — 그것이 `null`
#: 이고, **정답 누출 탐지기**다. 그 표에서 어떤 규칙이든 regret 이 1.0 을
#: 크게 밑돌면 어딘가에서 정답이 새고 있다.
PRESETS: dict[str, dict] = {
    "easy":   dict(struct=1.5, noise_scale=0.4, mem_damp=0.05),
    "normal": dict(struct=1.0, noise_scale=1.0, mem_damp=0.15),
    "hard":   dict(struct=0.5, noise_scale=6.0, mem_damp=0.40),
    "null":   dict(struct=0.0, noise_scale=1.0, mem_damp=1.00),
}

# -- 생성 모델의 계수 --------------------------------------------------------
# ★ 피처 라이브러리와 공유하지 않는다 (§28). 값도 일부러 다르게 둔다.
_C_SMEM = 0.37          # occupancy 손해
_C_SPILL = 2.60         # 스필 커널의 벌점
_C_SPLITK = 0.023       # split-K 파티션당 리덕션 비용
_C_WARMUP = 0.055       # stages 워밍업 (mainloop 이 짧을 때)
_C_TILE_EXP = 0.34      # 타일 연산 강도 지수 (구조 #7 — 난이도의 주 동인)
_REF_TILE_EFF = 128.0   # 128x128 타일의 2*tm*tn/(tm+tn)
_C_WARP_EXP = 0.20      # warp 타일 연산 강도 지수 (구조 #7b)
_REF_WARP_EFF = 64.0    # 64x64 warp 타일의 2*wm*wn/(wm+wn)
_C_PIPELINED = 0.21     # stages=2 (MmaPipelined) 벌점
_C_INST = 0.70          # 명령어 수 (구조 #9). GBDT 가 상위로 꼽은 축이다 (§30.6)
_C_REG = 0.30           # 레지스터 압력 (구조 #10)
_LAUNCH_MS = 0.0132     # 런치 오버헤드 (§18.2 의 13µs)

# 노이즈/눈금 — A6000 실측 (§30.2). 프리셋의 noise_scale 이 곱해진다.
_NOISE_A = 0.000374
_NOISE_B = 0.00044
_TICK_MS = 0.001024

_DTYPE_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "f8": 1}


@dataclass
class Grid:
    """(형상, config) 격자 + 하드웨어. **시간은 없다.**

    `df` 는 `load_for_ranking` 과 같은 모양이다 — 정답 컬럼이 없다.
    """

    df: pd.DataFrame
    env: dict
    hw: object
    source: str = ""
    extra_columns: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.df)

    # -- 실제 번들에서 격자만 가져오기 ------------------------------------
    @classmethod
    def from_bundle(cls, ref: str | Path, *, env_hash: str,
                    shapes: list[tuple[int, int, int]] | None = None,
                    max_configs_per_shape: int | None = None,
                    seed: int = 0) -> Grid:
        """실제 번들의 (형상, config, 커널 정적 속성) 격자를 그대로 쓴다.

        ★ `ranking` 로더만 부른다 — **실측 시간을 볼 수 없다.** 그래서 합성
        표가 실측을 베끼는 것이 구조적으로 불가능하다.

        실제 alignment 구조(a888 형상과 a448 형상의 후보가 다르다)가 공짜로
        따라온다 (§22.3).
        """
        from kerneltab.core.bundle import load_bundle
        from kerneltab.core.hardware import hardware_from_env

        b = load_bundle(ref, verify=True)
        if not str(b.env_hash).startswith(str(env_hash)):
            raise ValueError(f"env_hash 불일치: {env_hash!r} vs {b.env_hash[:16]!r}")
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
            for _, g in df.groupby(["M", "N", "K"], sort=False):
                if len(g) > max_configs_per_shape:
                    idx = np.sort(rng.choice(len(g), max_configs_per_shape,
                                             replace=False))
                    g = g.iloc[idx]
                parts.append(g)
            df = pd.concat(parts, ignore_index=True)
        df = df.reset_index(drop=True)
        if df.empty:
            raise ValueError("격자가 비었다. shapes 필터를 확인하라.")
        return cls(df=df, env=env, hw=hardware_from_env(env),
                   source=f"bundle:{b.info.get('bundle_id')}")

    # -- 번들 없이 열거 ---------------------------------------------------
    @classmethod
    def enumerate(cls, env: dict, *,
                  shapes: list[tuple[int, int, int]] | None = None,
                  max_kernels: int | None = None, seed: int = 0) -> Grid:
        """kernelTab 의 `shapes.py` / `backends/sm80.py` 로 직접 열거한다.

        번들이 없을 때 쓴다. 커널 정적 속성(regs/spill/occupancy)은 실제
        빌드가 있어야 알 수 있으므로 **해석적으로 모델링**한다 — 그 사실을
        `BUNDLE.json` 의 `synthetic.build_attrs` 에 남긴다.
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
            raise ValueError("열거 결과가 비었다. shapes / max_kernels 확인.")
        return cls(df=pd.DataFrame(rows), env=env, hw=hw,
                   source="enumerate:sm80",
                   extra_columns={"build_attrs": "analytic"})


def _build_attrs(be, cfg, hw, ext) -> dict:
    """빌드해야 알 수 있는 값들의 **해석적 모델**. 열거 모드 전용.

    실제 표에는 `-Xptxas -v` 실측이 들어 있다. 여기서는 근사한다 —
    합성 표의 목적은 파이프라인 검증이지 커널 예측이 아니다.
    """
    warps_m = max(1, cfg.tile_m // ext.warp_m)
    warps_n = max(1, cfg.tile_n // ext.warp_n)
    warps_k = max(1, cfg.tile_k // ext.warp_k) if ext.warp_k else 1
    threads = warps_m * warps_n * warps_k * 32
    accum = (ext.warp_m * ext.warp_n) // 32          # fp32 누산기 1개/레지스터
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
# 물리 — ★ features/ 를 쓰지 않는다 (§28)
# ---------------------------------------------------------------------------
def synth_times(grid: Grid, preset: str = "normal", *, seed: int = 0,
                return_parts: bool = False):
    """격자에 시간을 합성한다. 벡터화되어 있다."""
    if preset not in PRESETS:
        raise ValueError(f"알 수 없는 프리셋: {preset!r}. {sorted(PRESETS)}")
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

    # -- 1. roofline 두 항을 **따로** 둔다 --------------------------------
    flops = 2.0 * M * N * K
    bytes_moved = eb * (M * K + K * N + M * N)
    t_compute = flops / (hw.peak_tflops_f16 * 1e12) * 1e3        # ms
    t_memory = bytes_moved / (hw.bandwidth_gbps * 1e9) * 1e3     # ms

    # -- 2. wave quantization (구조 #1) -----------------------------------
    # ★ 피처 쪽의 tail_waste 와 대수적으로 같은 물리를 다른 형태로 쓴다.
    tiles_m = np.ceil(M / tm)
    tiles_n = np.ceil(N / tn)
    tiles = tiles_m * tiles_n * sk
    waves = tiles / (hw.sm_count * occ)
    wave_pen = np.ceil(waves) / np.maximum(waves, 1e-12)

    # -- 3. ★ 부분 타일 낭비 (구조 #11) — 형상 x config 상호작용의 주 동인 --
    # 타일이 형상 경계를 넘으면 그 부분도 **전부 계산한다**. M=1 에 128행
    # 타일이면 일의 99.2%가 버려진다. 작은 M 이 작은 tile_m 을 선호하는 이유다.
    edge_pen = (tiles_m * tm / M) * (tiles_n * tn / N)

    # -- 4. occupancy (구조 #2) -------------------------------------------
    smem_pen = 1.0 + _C_SMEM * np.clip(smem / hw.smem_per_block, 0.0, 1.5)

    # -- 5. 스필 (구조 #3) -------------------------------------------------
    spill_pen = 1.0 + _C_SPILL * (spill > 0)

    # -- 6. split-K 리덕션 (구조 #5) --------------------------------------
    sk_pen = 1.0 + _C_SPLITK * (sk - 1.0)

    # -- 7. mainloop 워밍업 (구조 #6) --------------------------------------
    iters = np.maximum(1.0, K / (tk * sk))
    warm_pen = 1.0 + _C_WARMUP * np.clip(stages / iters, 0.0, 2.0)

    # -- 8. 타일/warp 연산 강도 (구조 #7) ----------------------------------
    # ★ 기준 타일 위로는 이득이 **포화**한다. 그 위에서는 A/B smem 트래픽이
    #   더 이상 병목이 아니고 레지스터/에필로그가 병목이 된다.
    tile_eff = 2.0 * tm * tn / np.maximum(tm + tn, 1.0)
    tile_pen = np.power(_REF_TILE_EFF / np.clip(tile_eff, 1.0, _REF_TILE_EFF),
                        _C_TILE_EXP)
    wm = _col(df, ("ext_warp_m",), 64.0)
    wn = _col(df, ("ext_warp_n",), 64.0)
    warp_eff = 2.0 * wm * wn / np.maximum(wm + wn, 1.0)
    warp_pen = np.power(_REF_WARP_EFF / np.clip(warp_eff, 1.0, _REF_WARP_EFF),
                        _C_WARP_EXP)

    # -- 9. 2단 파이프라인 (구조 #8) ---------------------------------------
    pipe_pen = 1.0 + _C_PIPELINED * (stages <= 2)

    # -- 10. 명령어 수 / 레지스터 압력 (구조 #9, #10) -----------------------
    # 커널마다 조금씩 다른 **연속적인** 축이다. 이게 없으면 시간이 소수의
    # 이산 조합에만 떨어져 눈금 충돌이 실제보다 훨씬 심해진다.
    inst = _col(df, ("inst_total",), 0.0)
    inst_ref = np.median(inst[inst > 0]) if np.any(inst > 0) else 1.0
    inst_pen = 1.0 + _C_INST * np.clip(inst / max(inst_ref, 1.0) - 1.0, -0.6, 3.0)
    regs_tot = _col(df, ("regs_total_per_block",), 0.0)
    reg_pen = 1.0 + _C_REG * np.clip(regs_tot / hw.regs_per_sm, 0.0, 2.0)

    penalty = (wave_pen * edge_pen * smem_pen * spill_pen * sk_pen * warm_pen
               * tile_pen * warp_pen * pipe_pen * inst_pen * reg_pen)

    # -- 11. 구조 강도 (프리셋) --------------------------------------------
    # struct=0 -> penalty 가 정확히 1 -> 시간이 config 와 무관해진다 (null)
    penalty = np.power(penalty, float(ps["struct"]))

    # -- 12. 연산 쪽에만 곱한다. 메모리 바닥은 config 를 거의 모른다 --------
    damp = float(ps["mem_damp"])
    t = np.maximum(t_compute * penalty,
                   t_memory * (1.0 + damp * (penalty - 1.0))) + _LAUNCH_MS

    ai = flops / np.maximum(bytes_moved, 1.0)
    ridge = (hw.peak_tflops_f16 * 1e12) / (hw.bandwidth_gbps * 1e9)
    mem_bound = ai < ridge

    # -- 13. 측정 노이즈 — ★ 커널 시간에 의존한다 (§30.2) -------------------
    sigma = (_NOISE_A / np.maximum(t, 1e-9) + _NOISE_B) * float(ps["noise_scale"])
    t_noisy = t * (1.0 + rng.normal(0.0, 1.0, size=t.shape) * sigma)
    t_noisy = np.maximum(t_noisy, _TICK_MS)

    # -- 14. ★ 타이머 양자화 (§30.2) ---------------------------------------
    # 빼면 합성 표에 실제로는 존재하지 않는 순위가 생긴다.
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
        raise KeyError(f"격자에 {names} 중 아무것도 없다.")
    return np.full(len(df), float(default))


# ---------------------------------------------------------------------------
# 번들로 저장
# ---------------------------------------------------------------------------
def generate(preset: str, seed: int, out: str | Path, grid: Grid, *,
             bundle_id: str | None = None) -> Path:
    """합성 표를 **진짜 번들과 같은 형식**으로 쓴다 (§22.3).

    같은 파일 형식이어야 로더/어댑터/`PerfTable` 이 검증된다.

    ★ `SYNTHETIC` 이 `bundle_id` 와 디렉토리 이름에 박힌다 (§22.6). 합성
    산출물이 실제 결과로 오인되는 경로를 이름 수준에서 막는다.
    """
    out = Path(out)
    bid = bundle_id or f"SYNTHETIC-{preset}-s{seed}"
    if "SYNTHETIC" not in bid:
        raise ValueError("합성 번들의 bundle_id 에는 SYNTHETIC 이 들어가야 한다 (§22.6).")
    path = out / bid
    path.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    t, parts = synth_times(grid, preset, seed=seed, return_parts=True)
    df = grid.df.copy()
    df["time_ms"] = t

    # 정답/결과 컬럼을 실제 표와 같은 모양으로 채운다.
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

    # 형상 수준 파생 정답 (kernelTab 이 export 시 계산하는 것과 같은 정의)
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
            "warning": ("합성 표다. 성능 수치를 보고하지 마라 (§28). "
                        "이 표의 규칙을 실제 표에 이어서 쓰지 마라."),
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
        # ★ schema_version 2 로 낸다 — tick_ms 를 실제로 싣기 때문이다.
        "noise_floor": {
            "sigma_abs_ms": _NOISE_A, "sigma_rel": _NOISE_B,
            "tick_ms": _TICK_MS,
            "model": "noise_floor(t) = max(sigma_abs_ms/t + sigma_rel, tick_ms/t)",
            "source": "kernelrule.tools.synth (A6000 실측 계수를 그대로 씀)",
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
    """형상 층. 격자가 실제 번들에서 왔으면 그 층을 그대로 쓴다."""
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
    """합성 표의 `env_hash`. 실제 조건과 **절대 섞이지 않게** 접두어를 박는다."""
    return "5y47he71c" + hashlib.sha256(bid.encode()).hexdigest()[:55]


# ---------------------------------------------------------------------------
# 자기 검사 (§22.3, 부록 C)
# ---------------------------------------------------------------------------
def self_check(path: str | Path) -> dict:
    """생성된 표가 목표 통계를 만족하는지 확인한다.

    ⚠️ 생성기가 목표를 벗어나면 그 위에서 개발한 모든 것이 현실과 동떨어진다.
    테스트로 고정한다.
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

    # 동점 밀집도 — 짧은 형상에서 눈금이 지배하는지 (§22.3)
    short = best.to_numpy() < 0.05
    out = {
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
    return out
