"""★ 적합기가 **과적합을 만드나** — CMA vs Nelder-Mead. LLM 0회.

    python3 experiments/fitter_regress.py

사전 등록 `docs/artifacts/fitter-regress-prereg.md`.

§3 에서 기준선을 다시 뽑았더니 방향이 반대였다 (1.0762 -> 1.0987).
절차가 셋 달라(적합기·예산·시드 수) 판정에 못 썼다. **여기서는 변수를
적합기 하나로 줄인다** — 같은 규칙을 두 적합기로 다시 맞춘다.

```
NM   nelder-mead · 재시작 4 · 적합 300 · 다듬기 600   ← 지금까지의 측정 절차
CMA  cma         · 재시작 1 · 적합 300 · 다듬기 600   ← §3 이 쓴 것
```

⚠️ 학습 regret 은 **판정에 안 쓴다** — CMA 가 학습을 더 잘 맞추는 것은
설계상 당연하고, 그것은 과적합의 기전이지 판정이 아니다 (사전 등록 §3).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from regret_at_k import _measure as _measure_k
from scipy.stats import wilcoxon
from two_stage import A6000, _fit, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

#: 주 대상 — regret 진화 6구조. D-77·D-123 이 쓴 것과 **같은 여섯**이다.
MAIN = ("arch24 6구조", [f"f1pipe-F3-arch24-s{i}" for i in range(6)])
#: 부차 — §3 의 예산 8 팔. CMA 로 진화한 구조다. **따로** 본다.
SIDE = ("rb08 3구조 (CMA 진화)", [f"f1pipe-F3-rb08-s{i}" for i in range(3)])

ARMS = [("NM", "nelder-mead", 4), ("CMA", "cma", 1)]
REPORT_KS = (1, 10, 50, 100)
#: A6000 시드 폭. **크기를 읽는 자**이지 판정선이 아니다.
SIGMA = 0.0124
#: 비대응 참고선 (§29.5). 여기서 새로 정하지 않는다 (원칙 7).
DELTA = 0.0516


def _best(run: str) -> dict:
    f = Path("runs") / run / "archive.jsonl"
    arc = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return sorted(arc, key=lambda e: e["regret"])[0]


def _one(group: tuple, T, M, train, hold, out: dict) -> None:
    label, runs = group
    print("\n" + "=" * 88)
    print(f"{label} — 구조 {len(runs)}개, 변수는 **적합기 하나**")
    print("=" * 88)
    rows: dict = {}
    for name, method, nres in ARMS:
        ho, tr, ks = [], [], []
        for r in runs:
            e = _best(r)
            fn, ws = _fit(e["code"], e["w"], T, M, train, "regret",
                          method=method, n_restarts=nres)
            h = _measure_k(fn, ws, T, M, hold)
            t = _measure_k(fn, ws, T, M, train)
            ho.append(h["regret_at_k"][1])
            tr.append(t["regret_at_k"][1])
            ks.append([h["regret_at_k"][k] for k in REPORT_KS])
        rows[name] = {"hold": ho, "train": tr,
                      "ks": np.array(ks).tolist()}
    print(f"  {'':6s} {'학습 중앙':>10} {'홀드아웃 중앙':>13} {'격차':>9} "
          f"{'홀드아웃 범위':>19}")
    for name, _, _ in ARMS:
        d = rows[name]
        g = [h - t for h, t in zip(d["hold"], d["train"], strict=True)]
        print(f"  {name:6s} {np.median(d['train']):10.4f} "
              f"{np.median(d['hold']):13.4f} {np.median(g):+9.4f} "
              f"{min(d['hold']):9.4f}~{max(d['hold']):.4f}")
        d["gap"] = g

    dif = [c - n for n, c in zip(rows["NM"]["hold"], rows["CMA"]["hold"],
                                 strict=True)]
    worse = sum(1 for x in dif if x > 0)
    print("\n  ★ 짝지은 차이 (CMA - NM, 양수 = CMA 가 **나쁘다**)")
    print("     " + "  ".join(f"{x:+.4f}" for x in dif))
    print(f"     CMA 가 나쁜 구조 {worse}/{len(dif)}   "
          f"중앙 {np.median(dif):+.4f}   (σ={SIGMA}, 참고선 {DELTA})")
    p = float("nan")
    if len(dif) >= 5 and any(abs(x) > 0 for x in dif):
        p = float(wilcoxon(dif, alternative="greater").pvalue)
        print(f"     짝지은 Wilcoxon 단측 p = {p:.4f}  "
              f"{'★ 유의 (<=0.05)' if p <= 0.05 else '유의하지 않다'}")
    else:
        print("     ★ n<5 라 Wilcoxon 을 안 쓴다 — 부호만 읽는다")

    dtr = [c - n for n, c in zip(rows["NM"]["train"], rows["CMA"]["train"],
                                 strict=True)]
    print("\n  기전 확인 — 학습에서는? (CMA - NM, 음수 = CMA 가 잘 맞춘다)")
    print("     " + "  ".join(f"{x:+.4f}" for x in dtr)
          + f"   CMA 가 좋은 구조 {sum(1 for x in dtr if x < 0)}/{len(dtr)}")
    dg = [c - n for n, c in zip(rows["NM"]["gap"], rows["CMA"]["gap"],
                                strict=True)]
    print(f"     격차(홀드아웃-학습) 차 중앙 {np.median(dg):+.4f} "
          "— 양수면 CMA 가 더 벌어진다")

    print("\n  regret@k 중앙")
    print(f"  {'':6s} " + " ".join(f"{f'k={k}':>8s}" for k in REPORT_KS))
    for name, _, _ in ARMS:
        a = np.array(rows[name]["ks"])
        print(f"  {name:6s} " + " ".join(f"{m:8.3f}"
                                         for m in np.median(a, axis=0)))
    out[label] = {"rows": rows, "paired_hold": dif, "paired_train": dtr,
                  "paired_gap": dg, "p_hold": p, "n_worse": worse}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/fitter-regress.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    out: dict = {"sigma": SIGMA, "delta": DELTA, "ks": list(REPORT_KS),
                 "arms": [list(x) for x in ARMS]}
    for g in (MAIN, SIDE):
        _one(g, T, M, train, hold, out)

    m = out[MAIN[0]]
    print("\n" + "=" * 88)
    print("★ 판정 — 사전 등록 §3 그대로")
    print("=" * 88)
    if m["p_hold"] <= 0.05 and np.median(m["paired_hold"]) > 0:
        print("  ★ 적합기가 과적합을 만든다 — 예산 8 에서는 NM, 16차원이")
        print("     필요할 때만 CMA. 4090 전이의 (b) 재적합도 NM 으로 간다")
    elif m["p_hold"] <= 0.05:
        print("  ★ 예상 밖 — CMA 가 유의하게 좋다. 사전 등록에 없는 갈래다")
    else:
        print("  구분 불가 — 1.0762 -> 1.0987 은 적합기 탓이 아니다")
        print("     남는 후보: 적합 예산(200 vs 300) · 시드 수(6 vs 3) ·")
        print("     다듬기 수정(D-122) · 진화 경로 자체")
    print("  ⚠️ 이 결과는 D-123(§2 관문)을 뒤집지 않는다 — 거기는 **16차원")
    print("     도달률**이고 여기는 **8차원 일반화**다 (사전 등록 §5)")

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                      default=float))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
