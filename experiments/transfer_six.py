"""★ Three pairs, **six directions** — what makes the transfer hard? 0 LLM
calls.

    python3 experiments/transfer_six.py

It gathers the json `transfer_29_5.py` produced per direction into one sheet.
**Nothing is refitted here** — the procedure must not change (principle 2).

The three questions in `docs/artifacts/transfer-4090-prereg.md` §5:

```
1  does (a) work on A6000<->4090 (0 bound flips)
2  does (b) ≈ (c) hold on that pair
3  across the three pairs, does (a)'s loss correlate with the ridge
   difference
   ★ there are only three pairs, so it is descriptive — no correlation
     coefficient is given (principle 27)
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: The six directions. (source, destination)
DIRS = [("a6000", "4090"), ("4090", "a6000"),
        ("a6000", "5090"), ("5090", "a6000"),
        ("4090", "5090"), ("5090", "4090")]
#: pair name -> the number of bound flips (measured by `transfer_check`,
#: pre-registration §1).
#: ★ The key is **the sorted order** — that is how `_pair` builds it (there
#: are two directions, so putting only one in raises a KeyError on the other.
#: It really did).
FLIP = {tuple(sorted(k)): v for k, v in
        {("a6000", "4090"): 0, ("a6000", "5090"): 4,
         ("4090", "5090"): 3}.items()}
#: The decision line. Pre-registration §4 — it is not set anew here
#: (principle 7).
DELTA = 0.0516


def _pair(src: str, dst: str) -> tuple:
    return tuple(sorted((src, dst)))


def _load(src: str, dst: str) -> dict | None:
    f = Path(f"docs/artifacts/transfer-{src}-{dst}.json")
    return json.loads(f.read_text()) if f.exists() else None


def _med(x) -> float:
    return float(np.median(np.asarray(x, dtype=np.float64)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/transfer-six.json")
    a = ap.parse_args()

    got = {(s, d): _load(s, d) for s, d in DIRS}
    missing = [f"{s}->{d}" for (s, d), v in got.items() if v is None]
    if missing:
        print(f"  ⚠️ directions not there yet: {missing}")
        print("     ★ no conclusion is drawn with the missing ones left out "
              "— run them all and call this again")

    print("=" * 96)
    print("three pairs, six directions — the holdout of 20 shapes (flips "
          "included) / the parenthesis excludes the flips")
    print("=" * 96)
    print(f"  {'direction':16s} {'ridge x':>8} {'flips':>6} "
          f"{'(a) transplant':>22} {'(b) refit':>22} {'(c) regrow':>22}")
    rows: dict = {}
    for s, d in DIRS:
        v = got[(s, d)]
        if v is None:
            print(f"  {s + ' -> ' + d:16s} {'—':>8} {'—':>6} "
                  f"{'(not run yet)':>22}")
            continue
        r = v["ridge"][1] / v["ridge"][0]
        f = FLIP[_pair(s, d)]
        cell = []
        for k in ("a", "b", "c"):
            cell.append(f"{_med(v[k]):.4f} ({_med(v[k + '_nf']):.4f})")
        print(f"  {s + ' -> ' + d:16s} {r:8.2f} {f:6d} "
              + " ".join(f"{c:>22s}" for c in cell))
        rows[f"{s}->{d}"] = {
            "ridge_ratio": r, "flipped": f,
            "a": _med(v["a"]), "b": _med(v["b"]), "c": _med(v["c"]),
            "a_nf": _med(v["a_nf"]), "b_nf": _med(v["b_nf"]),
            "c_nf": _med(v["c_nf"]),
            "a_range": [min(v["a"]), max(v["a"])],
            "b_range": [min(v["b"]), max(v["b"])],
            "c_range": [min(v["c"]), max(v["c"])],
            "baseline": v["baseline"], "n": [len(v["a"]), len(v["c"])]}

    # ------------------------------------------------------ questions 1·2
    print("\n" + "=" * 96)
    print("questions 1·2 — per direction, how far (a) and (b) are from (c)")
    print("=" * 96)
    print(f"  {'direction':16s} {'flips':>6} {'(a)-(c)':>10} {'(b)-(c)':>10}"
          f"   {'flips excl. (a)-(c)':>22} {'(b)-(c)':>10}")
    for s, d in DIRS:
        k = f"{s}->{d}"
        if k not in rows:
            continue
        w = rows[k]
        da, db = w["a"] - w["c"], w["b"] - w["c"]
        dan, dbn = w["a_nf"] - w["c_nf"], w["b_nf"] - w["c_nf"]
        w["a_minus_c"], w["b_minus_c"] = da, db
        w["a_minus_c_nf"], w["b_minus_c_nf"] = dan, dbn
        print(f"  {k:16s} {w['flipped']:6d} {da:+10.4f} {db:+10.4f}   "
              f"{dan:+22.4f} {dbn:+10.4f}")
    print(f"\n  the decision line delta = {DELTA} (pre-registration §4).  "
          "positive = worse than (c)")
    print("  ★ if (b)-(c) is within delta, 'the structure moves over and only "
          "the weights need refitting' holds in that direction")

    # -------------------------------------------------------- question 3
    print("\n" + "=" * 96)
    print("question 3 — does (a)'s loss go with the ridge difference  "
          "★ 3 pairs, descriptive")
    print("=" * 96)
    print(f"  {'pair':16s} {'ridge x':>8} {'flips':>6} "
          f"{'(a)-(c) mean of both dirs':>28} {'flips excl.':>14}")
    per_pair: dict = {}
    for p in sorted({_pair(s, d) for s, d in DIRS}):
        ks = [f"{s}->{d}" for s, d in DIRS if _pair(s, d) == p and
              f"{s}->{d}" in rows]
        if not ks:
            continue
        m = float(np.mean([rows[k]["a_minus_c"] for k in ks]))
        mn = float(np.mean([rows[k]["a_minus_c_nf"] for k in ks]))
        rr = float(np.mean([rows[k]["ridge_ratio"] for k in ks]))
        # ★ The ratio is the reciprocal in each direction, so a mean collapses
        #   towards 1. It is read as **the absolute value of the difference**
        #   — both directions of the same pair then share a value.
        far = abs(np.log(rows[ks[0]]["ridge_ratio"]))
        per_pair["+".join(p)] = {"ridge_log_dist": far, "flipped": FLIP[p],
                                 "a_minus_c_mean": m, "a_minus_c_nf_mean": mn,
                                 "dirs": ks, "ridge_ratio_mean": rr}
        print(f"  {'+'.join(p):16s} {far:8.3f} {FLIP[p]:6d} "
              f"{m:28.4f} {mn:14.4f}")
    print("\n  the ridge column is **|log(ratio)|** — it is the reciprocal in "
          "each direction, so a plain mean would flatten it to about 1")
    print("  ★ no correlation coefficient is given. There are three pairs "
          "(principle 27)")

    # ------------------------------- ★ a post-hoc observation (per target)
    print("\n" + "=" * 96)
    print("★ a post-hoc observation — the loss changes with **the target "
          "table**  ⚠️ not in the pre-registration")
    print("=" * 96)
    print(f"  {'target':10s} {'(c) native':>12s} {'baseline physics':>18s} "
          f"{'(a)-(c) per source':>28s} {'(b)-(c) per source':>28s}")
    by_dst: dict = {}
    for s_, d_ in DIRS:
        k = f"{s_}->{d_}"
        if k in rows:
            by_dst.setdefault(d_, []).append((s_, rows[k]))
    for d_, items in by_dst.items():
        c = items[0][1]["c"]
        base = items[0][1]["baseline"]
        aa = "  ".join(f"{s_}:{w['a_minus_c']:+.4f}" for s_, w in items)
        bb = "  ".join(f"{s_}:{w['b_minus_c']:+.4f}" for s_, w in items)
        print(f"  {d_:10s} {c:12.4f} {base:18.4f} {aa:>28s} {bb:>28s}")
        per_pair[f"dst:{d_}"] = {
            "c": c, "baseline": base,
            "a_minus_c": {s_: w["a_minus_c"] for s_, w in items},
            "b_minus_c": {s_: w["b_minus_c"] for s_, w in items}}
    print("\n  ★ the two sources give **almost the same loss** on the same "
          "target —")
    print("     the loss looks like a property of the target, not the source")
    print("  ⚠️ it is three targets x two sources. It is a cut that was not "
          "in the pre-registration, so it is **an observation**")

    Path(a.out).write_text(json.dumps(
        {"delta": DELTA, "dirs": rows, "pairs": per_pair,
         "missing": missing}, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
