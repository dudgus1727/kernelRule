"""★ Wins and losses per shape — **where it wins and where it loses**. 0 LLM
calls.

    python3 experiments/vendor_shapes.py

It attaches **the table's properties** to the per-shape values
`vendor_compare.py` produced. It writes `vendor-shapes.json` so a figure (a
sorted bar chart) can be drawn.

⚠️ **It is a post-hoc cut.** Splitting by M was not in the pre-registration —
it is written down as an observation. The regime split (SOL 0.5ms) is an axis
that already existed and says the same thing.
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
        d0 = T.frame_for(p)
        rows.append({
            "M": mnk[0], "N": mnk[1], "K": mnk[2],
            # ★ It is a shape-level axis — one value per shape (confirmed)
            "arith_intensity": float(d0["arith_intensity"].iloc[0]),
            "is_memory_bound": bool(d0["is_memory_bound"].iloc[0]),
            "ours": o, "vendor": vend[k], "delta": o - vend[k],
            "regime": regime_of(p, T.hw),
            "answer_set": int((~ok).sum()),
            "gap100": float((t[99] - best) / best) if len(t) > 100 else float("nan"),
            "n_cand": int(len(t)),
        })
    rows.sort(key=lambda r: r["delta"])

    print("=" * 96)
    print(f"wins and losses per shape — {arm}   (negative = we win)")
    print("=" * 96)
    print(f"  {'M':>6s} {'N':>6s} {'K':>6s} {'ours':>8s} {'vendor':>8s} "
          f"{'diff':>9s} {'regime':>7s} {'answers':>8s} {'100th gap':>10s}")
    for r in rows:
        print(f"  {r['M']:6d} {r['N']:6d} {r['K']:6d} {r['ours']:8.4f} "
              f"{r['vendor']:8.4f} {r['delta']:+9.4f} {r['regime']:>7s} "
              f"{r['answer_set']:8d} {r['gap100']:10.1%}")

    def tally(sel, label):
        s = [r for r in rows if sel(r)]
        w = sum(1 for r in s if r["delta"] < -1e-9)
        l = sum(1 for r in s if r["delta"] > 1e-9)
        print(f"  {label:26s} wins {w:2d} / losses {l:2d} / "
              f"ties {len(s)-w-l:2d}"
              f"   sign test p = {_sign_p(w, l):.4f}")
        return {"win": w, "loss": l, "tie": len(s) - w - l,
                "p": _sign_p(w, l), "n": len(s)}

    print("\n" + "-" * 96)
    out = {"src": a.src, "arm": arm, "shapes": rows, "groups": {}}
    out["groups"]["all"] = tally(lambda r: True, "all")
    # ★ It is a post-hoc cut. M<=32 is not a boundary chosen after seeing the
    #   result but **the smallest three on the table's M axis** — that fact is
    #   written down.
    out["groups"]["M<=32"] = tally(lambda r: r["M"] <= 32, "M <= 32")
    out["groups"]["M>=128"] = tally(lambda r: r["M"] >= 128, "M >= 128")
    out["groups"]["fast"] = tally(lambda r: r["regime"] == "short",
                                  "the fast regime")
    out["groups"]["slow"] = tally(lambda r: r["regime"] == "long",
                                  "the slow regime")
    # ★ Again on an existing axis — `is_memory_bound` is a shape-level value
    #   the table already has (AI < ridge). It looks at whether it can stand
    #   in for the post-hoc M.
    out["groups"]["mem_bound"] = tally(lambda r: r["is_memory_bound"],
                                       "memory bound (AI<ridge)")
    out["groups"]["compute_bound"] = tally(lambda r: not r["is_memory_bound"],
                                           "compute bound")
    # ★ The D-130/D-141 axis — the top-rank gap. It is split at the median
    #   (20 shapes)
    med_gap = float(np.median([r["gap100"] for r in rows]))
    out["groups"]["tight_top"] = tally(lambda r: r["gap100"] < med_gap,
                                       f"tight top (<{med_gap:.1%})")
    out["groups"]["wide_top"] = tally(lambda r: r["gap100"] >= med_gap,
                                      f"wide top (>={med_gap:.1%})")

    print("\n  ⚠️ splitting by M is **post-hoc** (it is not in the "
          "pre-registration). Read it as an observation.")
    print("     The regime split is an axis that already existed and says the "
          "same thing.")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}   (data for the sorted bar chart)")


if __name__ == "__main__":
    main()
