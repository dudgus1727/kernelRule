"""★ Campaign spread — how far apart do two runs of the same condition land?
0 LLM calls (D-136).

    python3 experiments/campaign_spread.py

It reads only the curve raw data `round_curve.py` already produced (it does
not even read the table).

## Why it is needed

The current decision line `delta = 0.0516` was built from the **seed spread
within one campaign**, σ. But a good share of the comparisons we judge on are
**between different campaigns**. If there is a campaign effect, that
component does not shrink as seeds are added — it is the **floor** of the
decision line.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

ART = Path("docs/artifacts")
# (name, curve file, the group key inside that file)
#: ★ Everything is read from `round-curve-bests.json` — the curve rebuilt
#: from `bests.jsonl` (D-139). The old files (`round-curve*.json`) are **kept
#: as correction history** and are not read here.
#: ⚠️ 2026-09-08 (D-146): the names were translated together with
#: `docs/artifacts/campaign-spread.md` and the keys of `campaign-spread.json`.
#: The Korean originals are at commit `ee53b4d`.
CURVES = "round-curve-bests.json"
CAMPAIGNS = [
    ("old", CURVES, "F3rw-p8-old"),
    ("p3", CURVES, "F3rw-p8-p3"),
    ("new24", CURVES, "F3rw-p8"),
    ("nan", CURVES, "F3rw-p8-nan"),
]
# The power coefficient: two-sided 0.05 + power 0.8 -> z(0.975) + z(0.8)
ZSUM = 2.8016


def _curves(fn: str, key: str | None) -> dict[str, list[float]]:
    j = json.loads((ART / fn).read_text())
    g = j["groups"]
    k = key or next(iter(g))
    return g[k]["curves"]


def _source(fn: str, key: str | None) -> list[str]:
    j = json.loads((ART / fn).read_text())
    g = j["groups"]
    return g[key or next(iter(g))].get("source", ["?"])


def _at(c: list[float], r: int) -> float:
    return c[min(r, len(c) - 1)]


def _live(run: str) -> int:
    """The last round this seed **actually ran**.

    ⚠️ It must not be measured by the curve length — `archive.jsonl` writes a
    line **only when there is an improvement**, so a seed with no improvement
    in the last round looks like "it stopped early". The one place rounds are
    counted is `rounds.jsonl` (principle 2).
    """
    rs = [json.loads(x) for x
          in (Path("runs") / run / "rounds.jsonl").read_text().splitlines()
          if x.strip()]
    return max(r["round"] for r in rs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "campaign-spread.json"))
    a = ap.parse_args()

    cur = {name: _curves(fn, k) for name, fn, k in CAMPAIGNS
           if (ART / fn).exists()}
    print("=" * 88)
    print("1. up to which round is **every seed alive** in each campaign")
    print("=" * 88)
    live = {}
    for name, cs in cur.items():
        ends = {s: _live(s) for s in cs}
        live[name] = min(ends.values())
        print(f"  {name:5s} last round per seed "
              + " ".join(f"{s[-2:]}:r{e}" for s, e in sorted(ends.items()))
              + f"   -> all alive to r{live[name]}")

    # ★ p3 and new24 are the only two campaigns with the same prompt and the
    #   same condition
    r = min(live["p3"], live["new24"])
    print(f"\n  ★ p3 and new24 **have the same condition apart from "
          f"patience**. They are compared at r{r}, where both are alive")

    print("\n" + "=" * 88)
    print(f"2. at r{r} — the within-campaign spread and the between-campaign "
          f"difference")
    print("=" * 88)
    vals = {}
    for name in ("p3", "new24"):
        v = sorted(_at(c, r) for c in cur[name].values())
        vals[name] = v
        print(f"  {name:5s} n={len(v)}  " + " ".join(f"{x:.4f}" for x in v))
        print(f"        mean {st.mean(v):.4f}  median {st.median(v):.4f}"
              f"  σ(seed) {st.stdev(v):.4f}")
    d_mean = st.mean(vals["new24"]) - st.mean(vals["p3"])
    d_med = st.median(vals["new24"]) - st.median(vals["p3"])
    sw = math.sqrt(sum(st.variance(v) for v in vals.values()) / 2)
    n = len(vals["p3"])
    se_within = sw * math.sqrt(2.0 / n)
    print(f"\n  campaign difference (mean)     {d_mean:+.4f}")
    print(f"  campaign difference (median)   {d_med:+.4f}")
    print(f"  σ(seed) pooled                 {sw:.4f}")
    print(f"  ★ the standard deviation of the difference expected from the "
          f"seed spread alone  {se_within:.4f}")
    print(f"     the measured difference / that value = "
          f"{abs(d_mean) / se_within:.1f}x")

    print("\n" + "=" * 88)
    print("2-2. ★ the three campaigns side by side **at the round where all "
          "are alive**")
    print("=" * 88)
    print(f"  {'round':>6s} " + " ".join(f"{k:>22s}" for k in cur)
          + "   (mean · median · σ)")
    for rr in (4, 5, 11, 23):
        row = []
        for name, cs in cur.items():
            if rr > min(_live(s) for s in cs):
                row.append(f"{'— (some ended)':>22s}")
                continue
            v = [_at(c, rr) for c in cs.values()]
            row.append(f"{st.mean(v):7.4f} {st.median(v):7.4f} {st.stdev(v):6.4f}")
        print(f"  r{rr:<5d} " + " ".join(row))
    print("\n  source: " + " · ".join(
        f"{n}={'/'.join(_source(fn, k))}" for n, fn, k in CAMPAIGNS
        if (ART / fn).exists()))
    print("  ⚠️ the archive snapshot is **accurate only at the last round** "
          "(D-139) —")
    print("     only r11 of the old campaign can be used, and no round "
          "before it")
    print("  ⚠️ comparing a campaign where some seeds stopped at a later "
          "round means **the stopped side is frozen")
    print("     and only the running side improves** — that difference is "
          "not campaign spread but the")
    print("     fact that 'one side gave up' (this is where D-136 corrects "
          "D-135)")

    print("\n" + "=" * 88)
    print("3. variance decomposition — the campaign component")
    print("=" * 88)
    # Var(campaign mean) = sw^2/n + sc^2 ;  Var(difference) = 2(sw^2/n + sc^2)
    # There is only **one** observed difference, so only a point estimate is
    # possible (1 degree of freedom).
    var_diff = d_mean ** 2
    sc2_raw = var_diff / 2.0 - sw ** 2 / n
    # ★ The point estimate **can come out negative** — that means the
    #   observed campaign difference is smaller than the width expected from
    #   the seed spread alone. A variance cannot be negative, so it is
    #   clipped to 0, and **the clipping is written down** (hiding it would
    #   read as "there is a campaign component").
    sc2 = max(0.0, sc2_raw)
    sc = math.sqrt(sc2)
    if sc2_raw < 0:
        print(f"  ★ the σ(campaign)² point estimate is **negative** "
              f"({sc2_raw:+.3e}) — the observed difference is smaller than "
              f"the width expected from the seed spread alone. It is clipped "
              f"to 0")
    print(f"  σ(seed)  = {sw:.4f}   -> the standard error of the mean "
          f"{sw / math.sqrt(n):.4f}")
    print(f"  ★ σ(campaign) point estimate = {sc:.4f}   (1 degree of freedom "
          f"— no interval can be given)")
    print(f"  the standard deviation of the campaign mean = sqrt(σw²/n + σc²)"
          f" = {math.sqrt(sw ** 2 / n + sc2):.4f}")

    print("\n" + "=" * 88)
    print("4. the decision line — what changes")
    print("=" * 88)
    SIG_HI = 0.0319   # the σ 95% upper bound §29.5 used
    cur_line = ZSUM * SIG_HI * math.sqrt(2.0 / n)
    print(f"  the current decision line (seed σ upper bound {SIG_HI} · n={n} "
          f"· unpaired)  {cur_line:.4f}")
    new_line = ZSUM * math.sqrt(2.0 * (SIG_HI ** 2 / n + sc2))
    print(f"  ★ with the campaign component in (point estimate)             "
          f"      {new_line:.4f}")
    floor = ZSUM * math.sqrt(2.0) * sc
    print(f"  ★ the **floor** that remains however many seeds are added      "
          f"      {floor:.4f}")
    print("\n  the decision line as the number of seeds grows:")
    print(f"  {'n':>4s} {'seeds only':>11s} {'with campaign':>16s}")
    for m in (3, 6, 12, 24, 96):
        print(f"  {m:4d} {ZSUM * SIG_HI * math.sqrt(2.0 / m):11.4f}"
              f" {ZSUM * math.sqrt(2.0 * (SIG_HI ** 2 / m + sc2)):16.4f}")

    print("\n" + "=" * 88)
    print("5. ⚠️ the limit of this estimate — there is **one campaign pair**")
    print("=" * 88)
    # d ~ N(0, tau^2) was seen **once**, so the 95% upper bound of tau is
    # |d| / sqrt(chi2_{0.05,1}) = |d| / 0.0627 -> 16x. There is effectively
    # no information.
    print("  same-condition campaign pairs  1  ->  1 degree of freedom")
    print(f"  σ(campaign) 95% upper bound   {abs(d_mean) / 0.0627:.4f}"
          "   ★ 16x the measured value — it cannot be used as a bound")
    print("  ★ what can be said: **the measured difference equals the width "
          "expected from the seed spread alone**")
    print("     what cannot be said: a **guarantee** that the campaign "
          "component is small")
    print("  Measuring this needs 3 or more same-condition campaigns "
          "(2+ degrees of freedom)")

    out = {
        "round": r,
        "seed_values": vals,
        "diff_mean": d_mean, "diff_median": d_med,
        "sigma_within": sw, "sigma_campaign_point": sc,
        "sigma_campaign_var_raw": sc2_raw,
        "line_current": cur_line, "line_with_campaign": new_line,
        "line_floor": floor, "n": n, "sigma_hi_used": SIG_HI,
        "note": "σ(campaign) is a point estimate from a single campaign pair "
                "(1 degree of freedom)",
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
