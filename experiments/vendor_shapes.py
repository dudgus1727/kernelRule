"""★ 형상별 승패 — **어디서 이기고 어디서 지나**. LLM 0회.

    python3 experiments/vendor_shapes.py

`vendor_compare.py` 가 낸 형상별 값에 **표의 성질**을 붙인다.
그림(정렬 막대)을 그릴 수 있게 `vendor-shapes.json` 으로 낸다.

⚠️ **사후 자름이다.** M 으로 가르는 것은 실험 계획서에 없었다 — 관측으로
적는다. 다만 체제 분할(SOL 0.5ms)은 이미 있던 축이고 같은 얘기다.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from math import comb
from pathlib import Path

import numpy as np

from kernelrule.core.splits import regime_of
from kernelrule.core.table import PerfTable

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
PAT = re.compile(r"M=(\d+), N=(\d+), K=(\d+)")


def _sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="docs/artifacts/vendor-nan-r11.json")
    ap.add_argument("--out", default="docs/artifacts/vendor-shapes.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    j = json.loads(Path(a.src).read_text())
    arm = next(iter(j["arms"]))
    ours, vend = j["arms"][arm]["per_shape_median"], j["vendor_per_shape"]
    T = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    by_key = {(p.M, p.N, p.K): p for p in T.shapes()}

    rows = []
    for k, o in ours.items():
        if k not in vend:
            continue
        m = PAT.search(k)
        mnk = tuple(int(x) for x in m.groups())
        p = by_key[mnk]
        t = np.sort(np.asarray(T.times_of(p), dtype=np.float64))
        best = t[0]
        ok = T.noise.resolvable(np.full(t.shape, best), t)
        rows.append({
            "M": mnk[0], "N": mnk[1], "K": mnk[2],
            "ours": o, "vendor": vend[k], "delta": o - vend[k],
            "regime": regime_of(p, T.hw),
            "answer_set": int((~ok).sum()),
            "gap100": float((t[99] - best) / best) if len(t) > 100 else float("nan"),
            "n_cand": int(len(t)),
        })
    rows.sort(key=lambda r: r["delta"])

    print("=" * 96)
    print(f"형상별 승패 — {arm}   (음수 = 우리가 이김)")
    print("=" * 96)
    print(f"  {'M':>6s} {'N':>6s} {'K':>6s} {'우리':>8s} {'벤더':>8s} "
          f"{'차':>9s} {'체제':>6s} {'정답집합':>7s} {'100등폭':>8s}")
    for r in rows:
        print(f"  {r['M']:6d} {r['N']:6d} {r['K']:6d} {r['ours']:8.4f} "
              f"{r['vendor']:8.4f} {r['delta']:+9.4f} {r['regime']:>6s} "
              f"{r['answer_set']:7d} {r['gap100']:8.1%}")

    def tally(sel, label):
        s = [r for r in rows if sel(r)]
        w = sum(1 for r in s if r["delta"] < -1e-9)
        l = sum(1 for r in s if r["delta"] > 1e-9)
        print(f"  {label:24s} 이김 {w:2d} / 짐 {l:2d} / 무 {len(s)-w-l:2d}"
              f"   부호검정 p = {_sign_p(w, l):.4f}")
        return {"win": w, "loss": l, "tie": len(s) - w - l,
                "p": _sign_p(w, l), "n": len(s)}

    print("\n" + "-" * 96)
    out = {"src": a.src, "arm": arm, "shapes": rows, "groups": {}}
    out["groups"]["all"] = tally(lambda r: True, "전체")
    # ★ 사후 자름이다. M<=32 는 결과를 보고 정한 경계가 아니라 **표의 M 축에서
    #   가장 작은 셋**이다 — 그 사실을 적는다.
    out["groups"]["M<=32"] = tally(lambda r: r["M"] <= 32, "M <= 32")
    out["groups"]["M>=128"] = tally(lambda r: r["M"] >= 128, "M >= 128")
    out["groups"]["fast"] = tally(lambda r: r["regime"] == "short", "빠른 체제")
    out["groups"]["slow"] = tally(lambda r: r["regime"] == "long", "느린 체제")

    print("\n  ⚠️ M 으로 가른 것은 **사후**다 (실험 계획서에 없다). 관측으로 읽어라.")
    print("     체제 분할은 이미 있던 축이고 같은 얘기다.")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}   (정렬 막대 그림용 자료)")


if __name__ == "__main__":
    main()
