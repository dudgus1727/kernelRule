"""★ 표마다 **정답 집합이 얼마나 넓은가** — 전이 손해의 세 번째 축. LLM 0회.

    python3 experiments/table_axes.py

D-126 이 관측한 것: 전이 손해가 **출처가 아니라 대상**이 정하고, 그 순서가
그 표의 원주민 성적 순서와 같다. ridge 도 바운드 뒤집힘도 그 순서를 못
만든다. **남는 후보가 "그 표에서 1등이 얼마나 좁은 표적인가" 다.**

```
정답 집합 크기   best 와 **노이즈로 구분 안 되는** config 수 (NoiseModel)
상위권 폭        (t_k - t_1) / t_1  — k 등이 1등보다 몇 % 느린가
```

★ 새 문턱을 만들지 않는다 — `table.noise.resolvable` 그대로다 (원칙 2).
⚠️ 이것은 **관측**이다. D-126 의 사후 관측에 붙는 것이고 인과가 아니다.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from kernelrule.core.crosstable import common_shapes
from kernelrule.core.table import PerfTable

TABLES = {
    "a6000": ("datasets/rtx-a6000-sm_86-c63710df", "c63710df"),
    "4090": ("datasets/rtx-4090-sm_89-ad95d455", "ad95d455"),
    "5090": ("datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403"),
    # ★ 2026-09-06 (D-141). ⚠️ 이 표의 정답 집합만 노이즈 계수 논란이 있다 —
    #   `--sigma-rel` 로 계수를 갈아 끼워 영향 범위를 괄호로 낼 수 있다.
    "h100": ("datasets/h100-nvl-sm_90-63684546", "63684546"),
}
KS = (10, 100)


def _axes(T: PerfTable, shapes) -> dict:
    ans, spread = [], {k: [] for k in KS}
    for p in shapes:
        t = np.sort(np.asarray(T.times_of(p), dtype=np.float64))
        best = t[0]
        # ★ best 와 노이즈로 못 가르는 config 수 = "정답 집합"
        ok = T.noise.resolvable(np.full(t.shape, best), t)
        ans.append(int((~ok).sum()))
        for k in KS:
            if len(t) > k:
                spread[k].append(float((t[k - 1] - best) / best))
    return {"n_shapes": len(list(shapes)),
            "answer_set": ans,
            "answer_median": float(np.median(ans)),
            "spread": {k: float(np.median(v)) for k, v in spread.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/table-axes.json")
    ap.add_argument("--sigma-rel", type=float, default=None, metavar="X",
                    help="이 표들의 sigma_rel 을 X 로 바꿔 다시 센다. "
                         "★ 계수가 정답 집합에 얼마나 닿는지 괄호로 내는 용도")
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = {n: PerfTable.from_bundle(b, env_hash=h, ok_only=False)
         for n, (b, h) in TABLES.items()}
    # ★ 등록된 표에 **다 있는 형상**으로만 잰다 — 형상 쓸이가 다르면 (4090 은
    #   큰 M 쪽이다) 표의 성질이 아니라 형상 구성을 재게 된다.
    common = common_shapes(T["a6000"], T["4090"])
    keys = {(p.M, p.N, p.K) for p in common} & {
        (p.M, p.N, p.K) for p in common_shapes(T["a6000"], T["5090"])}
    if "h100" in T:
        keys &= {(p.M, p.N, p.K)
                 for p in common_shapes(T["a6000"], T["h100"])}
    if a.only:
        T = {n: t for n, t in T.items() if n in a.only}
    if a.sigma_rel is not None:
        # ★ 계수만 갈아 끼운다. **다른 것은 그대로** — 영향 범위를 괄호로
        #   내기 위한 것이지 계수를 고치는 것이 아니다 (D-141).
        import dataclasses
        for n, t in T.items():
            # `noise` 는 읽기 전용 property 다 — 뒷필드를 바꾼다.
            t._noise = dataclasses.replace(
                t.noise, sigma_rel_coef=a.sigma_rel,
                source=t.noise.source + f" [★ sigma_rel 을 {a.sigma_rel} 로 갈아끼움]")
        print(f"  ★ sigma_rel 을 {a.sigma_rel} 로 바꿔 잰다 — 원본이 아니다")
    print("=" * 88)
    print(f"표의 축 — 표 {len(T)}개에 다 있는 {len(keys)}형상에서만 잰다 (LLM 0회)")
    print("=" * 88)
    print(f"  {'표':8s} {'정답 집합 (중앙)':>16s} {'분포':>22s} "
          f"{'10등이 1등보다':>14s} {'100등':>10s}")
    out: dict = {"n_common": len(keys), "ks": list(KS)}
    for n, t in T.items():
        sh = [p for p in t.shapes() if (p.M, p.N, p.K) in keys]
        r = _axes(t, sh)
        q = np.percentile(r["answer_set"], [25, 75])
        dist = f"[{q[0]:.0f}, {q[1]:.0f}]  최대 {max(r['answer_set'])}"
        print(f"  {n:8s} {r['answer_median']:16.1f} {dist:>22s} "
              f"{r['spread'][10]:13.1%} {r['spread'][100]:10.1%}")
        out[n] = r
    print()
    print("  ★ 정답 집합이 넓을수록 '아무거나 골라도 되는' 표다 —")
    print("     남의 가중치를 그대로 옮겨도 덜 벌받는다 (D-126 의 가설)")
    print("  ⚠️ 관측이다. 표 셋이고 실험 계획서에 없던 자름이다 (원칙 27)")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
