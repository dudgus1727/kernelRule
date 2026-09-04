"""★ 순위 손실 관문 — 대리 손실이 목표와 어긋나는가. LLM 0회.

    python3 experiments/rank_loss_gate.py

## 왜 순위 손실인가

config 를 일부만 재면 `regret` 을 못 쓴다 — **최적을 모르기 때문이다.**

```
regret = 시간 / ★ 최적 시간      최적을 모르면 정의가 안 된다
순위 손실 = 재본 것들끼리의 순서가 맞는가   ★ 최적을 몰라도 된다
```

**그러나 대리 손실은 목표와 어긋날 수 있다.** config 샘플링을 하기
전에 그것부터 잰다. **여기서는 config 를 전수로 쓴다** — 샘플링 변수를
섞으면 무엇이 원인인지 못 가른다.

```
A) 순위 손실로 적합 -> ★ regret 으로 채점
B) regret 으로 적합 -> regret 으로 채점    (지금 방식)
```

## 설계에서 지킨 것 셋

```
1 ★ 모든 쌍을 똑같이 세지 않는다
    19,635개 중 15,000등과 16,000등의 순서는 의미가 없다.
    양성은 **참 상위 K개**, 음성은 순위 구간별로 고르게 뽑는다

2 ★ 노이즈 바닥 이내인 쌍은 뺀다
    `NoiseModel.resolvable(t_i, t_j)` 가 그 판정을 한다.
    안 빼면 잡음에 맞춘다 (5090 정답 집합 중앙 9개, 최대 724개)

3 ★ 완화 온도를 튜닝하지 않는다 — T = 1 로 **고정**한다
    점수는 w 에 대해 선형이므로 (s = Φw) w 를 c 배 하면 s 가 c 배다.
    즉 T 는 w 의 크기와 **분리해서 식별되지 않는다.**
    T 를 두는 것은 자유도를 늘리는 것이 아니라 중복시키는 것이다.
    ★ 따라서 T=1 은 선택이 아니라 정규화다.
```

## 선형성

`s = Φ w` 를 쓰므로 **점수가 w 에 선형이어야 한다.** 구조마다 실제로
확인하고(2w 에서 2s 가 나오는가), 아니면 그 구조를 빼고 그 사실을 적는다.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

G5090 = ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403")
SRC_RUNS = [f"F3rw-p8-s{i}" for i in range(6)]
N_POS = 32          # 참 상위 K개
N_NEG = 256         # 순위 구간별로 고르게
TEMP = 1.0          # ★ 고정. 위 docstring 참고


def _splits(table: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def _phi(fn, matrix, table, p, n_w: int) -> np.ndarray:
    """항별 값 행렬 Φ (후보 x 항). 단위 기저 가중치로 뽑는다."""
    cand = table.candidates(p)
    cols = []
    for j in range(n_w):
        w = np.zeros(n_w)
        w[j] = 1.0
        cols.append(np.asarray(make_score_of(fn, matrix, w)(p, cand),
                               dtype=np.float64))
    return np.stack(cols, axis=1)


def _is_linear(phi: np.ndarray, fn, matrix, table, p, w: np.ndarray) -> bool:
    """s = Φw 인가. 아니면 이 방법을 못 쓴다 — 조용히 넘기지 않는다."""
    cand = table.candidates(p)
    s = np.asarray(make_score_of(fn, matrix, w)(p, cand), dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(phi @ w)
    if not ok.any():
        return False
    d = np.abs(s[ok] - (phi @ w)[ok])
    scale = np.maximum(np.abs(s[ok]), 1.0)
    return bool(np.max(d / scale) < 1e-9)


def _pairs(table, p, rng) -> tuple[np.ndarray, np.ndarray]:
    """(양성, 음성) 인덱스. **노이즈로 못 가르는 쌍은 뺀다.**"""
    t = table.times_of(p)
    order = np.argsort(t, kind="stable")
    pos = order[:N_POS]
    rest = order[N_POS:]
    if len(rest) == 0:
        return pos, rest
    # 순위 구간별로 고르게 — 뒤쪽만 뽑으면 쉬운 쌍만 배운다
    idx = np.unique(np.linspace(0, len(rest) - 1, N_NEG).astype(int))
    neg = rest[idx]
    return pos, neg


def _pair_data(fn, matrix, table, shapes, n_w, rng):
    """형상마다 Φ 부분행렬과 유효 쌍 마스크를 미리 만든다."""
    out = []
    for p in shapes:
        phi = _phi(fn, matrix, table, p, n_w)
        pos, neg = _pairs(table, p, rng)
        if len(neg) == 0:
            continue
        t = table.times_of(p)
        # ★ 노이즈 바닥으로 가를 수 있는 쌍만
        res = table.noise.resolvable(t[pos][:, None], t[neg][None, :])
        worse = t[neg][None, :] > t[pos][:, None]
        mask = res & worse
        if not mask.any():
            continue
        out.append((phi[pos], phi[neg], mask))
    return out


def _rank_loss_and_grad(w, data):
    """평균 로지스틱 쌍 손실과 기울기. s = Φw 이므로 해석적으로 나온다."""
    tot, n = 0.0, 0
    g = np.zeros_like(w)
    for phi_p, phi_n, mask in data:
        sp = phi_p @ w
        sn = phi_n @ w
        d = (sn[None, :] - sp[:, None]) / TEMP     # 양수여야 옳다
        # log(1 + exp(-d)) — 안정적으로
        loss = np.logaddexp(0.0, -d)
        sig = -1.0 / (1.0 + np.exp(d))             # d(loss)/d(d)
        m = mask.astype(np.float64)
        tot += float((loss * m).sum())
        n += int(mask.sum())
        gm = sig * m / TEMP
        g += (gm.sum(axis=0) @ phi_n) - (gm.sum(axis=1) @ phi_p)
    if n == 0:
        return 0.0, g
    return tot / n, g / n


def _regret_on(fn, matrix, table, shapes, ws) -> float:
    regs = []
    for p in shapes:
        cand = table.candidates(p)
        sc = make_score_of(fn, matrix, ws[regime_of(p, table.hw)])(p, cand)
        t = table.times_of(p)
        regs.append(float(t[cand.top_k(sc, 1)[0]] / t.min()))
    return geomean(np.array(regs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/rank-loss-gate.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    B = PerfTable.from_bundle(G5090[0], env_hash=G5090[1], ok_only=False)
    mB = FeatureMatrix(B, REGISTRY)
    sp = _splits(B)

    print("=" * 78)
    print("순위 손실 관문 — 대리 손실이 목표와 어긋나는가")
    print("=" * 78)
    print(f"  5090 학습 {len(sp.train.shapes)} / 홀드아웃 "
          f"{len(sp.val.shapes)}   ★ config 는 전수다")
    print(f"  쌍: 양성 상위 {N_POS}, 음성 순위 구간별 {N_NEG}, "
          f"노이즈로 못 가르는 쌍 제외, T = {TEMP} 고정\n")
    print(f"  {'구조':26s} {'A) 순위손실':>12} {'B) regret':>11} "
          f"{'A-B':>9}  선형")

    rows = []
    for run in SRC_RUNS:
        f = Path("runs") / run / "archive.jsonl"
        e = sorted((json.loads(x) for x in f.read_text().splitlines()
                    if x.strip()), key=lambda z: z["regret"])[0]
        fn = compile_rule(e["code"])
        w0 = np.asarray(e["w"], dtype=np.float64)
        n_w = len(w0)

        # 선형성 확인 — 한 형상으로 충분하다 (구조는 형상과 무관하다)
        p0 = sp.train.shapes[0]
        phi0 = _phi(fn, mB, B, p0, n_w)
        lin = _is_linear(phi0, fn, mB, B, p0, w0 * 1.7)
        if not lin:
            print(f"  {run:26s} {'—':>12} {'—':>11} {'—':>9}  ★ 비선형")
            rows.append({"run": run, "linear": False})
            continue

        ws_a, ws_b = {}, {}
        for nm in ("short", "long"):
            g = [q for q in sp.train.shapes if regime_of(q, B.hw) == nm]
            rng = np.random.default_rng(0)
            data = _pair_data(fn, mB, B, g, n_w, rng)
            r = minimize(_rank_loss_and_grad, w0, args=(data,), jac=True,
                         method="L-BFGS-B",
                         options={"maxiter": 500, "maxfun": 2000})
            ws_a[nm] = r.x
            # ★ B 팔은 정의상 regret 이다 (D-99). 명시한다 — 기본값이
            #   rank 로 바뀌었으므로 안 밝히면 두 팔이 같아진다.
            ws_b[nm] = fit_weights(fn, mB, B, Split("train", tuple(g)), w0,
                                   max_evals=300, objective="regret").w
        va = _regret_on(fn, mB, B, list(sp.val.shapes), ws_a)
        vb = _regret_on(fn, mB, B, list(sp.val.shapes), ws_b)
        print(f"  {run:26s} {va:12.4f} {vb:11.4f} {va - vb:+9.4f}  ✓")
        rows.append({"run": run, "linear": True, "rank_loss": va,
                     "regret": vb, "diff": va - vb})

    ok = [r for r in rows if r.get("linear")]
    if ok:
        A_ = np.array([r["rank_loss"] for r in ok])
        Bv = np.array([r["regret"] for r in ok])
        from scipy.stats import wilcoxon
        try:
            _, pv = wilcoxon(A_, Bv)
        except Exception:                                   # noqa: BLE001
            pv = float("nan")
        print(f"\n  중앙  A) {np.median(A_):.4f}   B) {np.median(Bv):.4f}   "
              f"차이 {np.median(A_) - np.median(Bv):+.4f}")
        print(f"  ★ 대응 Wilcoxon 양측 p = {pv:.4f}  "
              f"(A 가 나쁜 구조 {int(np.sum(A_ - Bv > 0))}/{len(ok)})")
        print("  ★ 판정선(σ 상한, n=6 대응) 참고: 0.0516 — "
              "sigma-5090.json")
    Path(a.out).write_text(json.dumps(
        {"bundle": G5090[0], "n_pos": N_POS, "n_neg": N_NEG, "temp": TEMP,
         "note": ("config 전수. 샘플링 변수를 섞지 않았다. "
                  "T 는 w 크기와 분리 식별되지 않으므로 1 로 고정"),
         "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
