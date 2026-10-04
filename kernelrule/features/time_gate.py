"""★ A time axis has to behave like a time — train-only checks (D-192).

## Why

Asked for "an axis that estimates the time from two effects pulling in
opposite directions as one config field grows" (role/feature.md, D-190 §2),
the FeatureWriter wrote axes that carried the unit "time ratio" and still
moved one way only: the split-K axis was exactly `1/split_k` in serial mode,
and every raster axis grew with the band width. The quota counted them by
their unit, so nothing told the model the axis did not do what the unit
claims. The hand-built time axes that worked (D-189) were written the other
way round — measured on the training shapes, fixed, measured again.

## What is measured — training shapes only

For an axis whose declared unit contains "time" (lower is better after the
declared direction is applied):

```
rank       mean over training shapes of the within-shape rank correlation
           between the axis and log(time). A time estimate rises with time
pick       geometric-mean regret when the config with the lowest value is
           picked (reported, not judged — a time axis is one term of a rule)
slices     through each training shape's fastest config, vary ONE field
           (split_k / stages / raster_width) with every other config field
           held. Where is the axis lowest, where is the time lowest?
             judged only when the slice has >= 3 values of the field AND
             the axis varies along it (an axis that ignores a field makes
             no claim about it)
             end miss = the time is lowest inside the range, the axis at an
                        end
```

## The verdict

```
rejected   rank < RANK_MIN (0.3)
           or, along some field where the time is lowest inside the range
           on at least a quarter of the judged shapes (and on >= 3), the
           axis puts its minimum at an end on more than half of those
```

⚠️ The thresholds were set **after** looking at ten known axes on fold 0's
training shapes (`experiments/time_gate_calib.py`,
`docs/artifacts/time-gate/calib.json`). The first rule — rank > 0, a field
judged from 3 interior shapes — caught the hand-built time axes on
`stages`, where the a6000 optimum sits inside the range on only 5 of 44
slices, and let through FeatureWriter axes correlating +0.05 with time.
0.3 is the conventional "medium" correlation; a quarter keeps a field from
being judged on a handful of shapes. Neither was swept. ⛔ Only `train_shapes` are read —
the holdout never reaches this file. ⚠️ It reads training **times**: unlike
the rest of the FeatureWriter path (which sees no data, D-75), a rejection
here tells the model how its axis ranks the training configs. That is the
deliberate change of D-192, recorded as the condition `time_gate`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["SLICE_FIELDS", "RANK_MIN", "INTERIOR_SHARE", "slice_stats",
           "diagnose", "verdict"]

#: The config fields a time axis is checked along.
SLICE_FIELDS = ("split_k", "stages", "raster_width")
#: The smallest mean within-shape rank correlation with log(time).
RANK_MIN = 0.3
#: A field is judged only where the time is lowest inside its range on at
#: least this share of the judged shapes (and on at least 3).
INTERIOR_SHARE = 0.25

#: What "the same kernel except along one field" holds fixed. Resource
#: columns (registers, smem, threads) follow from these and are left out.
_KEY = ("tile_m", "tile_n", "tile_k", "align_a", "align_b", "align_c",
        "split_k", "split_k_mode", "stages", "raster_order",
        "raster_width", "pipeline_kind", "warp_m", "warp_n", "warp_k")


def slice_stats(field_vals: np.ndarray, times: np.ndarray,
                axis: np.ndarray) -> dict | None:
    """One slice (rows that differ only in one field). `None` when it
    cannot be judged: fewer than 3 values of the field, or an axis that
    does not move along it."""
    vals = np.asarray(field_vals)
    uniq = np.unique(vals)
    if len(uniq) < 3:
        return None
    t_best = {u: float(np.min(times[vals == u])) for u in uniq}
    a_best = {u: float(np.min(axis[vals == u])) for u in uniq}
    if np.ptp(list(a_best.values())) == 0:
        return None                          # the axis ignores this field
    truth = min(uniq, key=lambda u: (t_best[u], u))
    pick = min(uniq, key=lambda u: (a_best[u], u))
    ends = (uniq[0], uniq[-1])
    interior = truth not in ends
    return {"hit": bool(pick == truth), "interior": bool(interior),
            "end_miss": bool(interior and pick in ends)}


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _columns(df) -> dict:
    """The config fields of one shape's rows, from the table frame.
    ⚠️ The same mapping as `core.types.raster_of` (D-190 §4)."""
    from kernelrule.core.types import raster_of

    out = {k: df[k].to_numpy() for k in ("tile_m", "tile_n", "tile_k",
                                         "align_a", "align_b", "align_c",
                                         "split_k", "split_k_mode",
                                         "pipeline_kind")}
    n = len(df)
    out["stages"] = (df["ext_stages"].to_numpy() if "ext_stages" in df
                     else np.zeros(n))
    st = df["ext_swizzle_type"].tolist() if "ext_swizzle_type" in df \
        else [None] * n
    sn = df["ext_swizzle_n"].tolist() if "ext_swizzle_n" in df \
        else [None] * n
    ro = [raster_of(a, b) for a, b in zip(st, sn, strict=True)]
    out["raster_order"] = np.asarray([r[0] for r in ro], dtype=object)
    out["raster_width"] = np.asarray([r[1] for r in ro])
    for k in ("warp_m", "warp_n", "warp_k"):
        c = f"ext_{k}"
        out[k] = df[c].to_numpy() if c in df else np.zeros(n)
    return out


def diagnose(name: str, direction: str, table, probe,
             train_shapes: Sequence) -> dict:
    """The numbers above, for axis `name` held in the matrix `probe`."""
    sign = -1.0 if direction == "higher_is_better" else 1.0
    ranks, picks = [], []
    per = {f: {"n": 0, "interior": 0, "end_miss": 0, "hit": 0}
           for f in SLICE_FIELDS}
    for p in train_shapes:
        feats, _info = probe.for_shape(p)
        a = sign * np.asarray(getattr(feats, name), dtype=np.float64)
        t = np.asarray(table.times_of(p), dtype=np.float64)
        if a.ndim != 1 or len(a) != len(t) or not np.isfinite(a).all():
            continue
        ranks.append(_rank_corr(a, np.log(t)))
        picks.append(float(t[int(np.argmin(a))] / np.min(t)))
        cols = _columns(table.frame_for(p).reset_index(drop=True))
        j = int(np.argmin(t))
        for fld in SLICE_FIELDS:
            mask = np.ones(len(t), dtype=bool)
            for k in _KEY:
                if k != fld:
                    mask &= cols[k] == cols[k][j]
            s = slice_stats(cols[fld][mask], t[mask], a[mask])
            if s is None:
                continue
            d = per[fld]
            d["n"] += 1
            d["interior"] += s["interior"]
            d["end_miss"] += s["end_miss"]
            d["hit"] += s["hit"]
    r = [x for x in ranks if np.isfinite(x)]
    return {"n_shapes": len(picks),
            "rank": float(np.mean(r)) if r else float("nan"),
            "pick_gm": (float(np.exp(np.mean(np.log(picks))))
                        if picks else float("nan")),
            "slices": per}


def verdict(d: dict) -> str | None:
    """`None` = it behaves like a time. Otherwise the message the
    FeatureWriter gets back — numbers about **its own axis** only."""
    bad = []
    if not (d["rank"] >= RANK_MIN):
        bad.append(f"it does not rise with the time: the mean within-shape "
                   f"rank correlation with log(time) is {d['rank']:+.2f} "
                   f"over {d['n_shapes']} training shapes")
    for fld, s in d["slices"].items():
        if (s["interior"] >= max(3, INTERIOR_SHARE * s["n"])
                and s["end_miss"] > 0.5 * s["interior"]):
            bad.append(
                f"along cfg.{fld} it moves one way only: holding every other "
                f"field of the fastest config, the time is lowest in the "
                f"middle of the range on {s['interior']} of {s['n']} "
                f"training shapes, and your axis is lowest at an end on "
                f"{s['end_miss']} of those")
    if not bad:
        return None
    return ("time-axis check — " + "; ".join(bad) + f". (Picking the lowest "
            f"value alone gives a regret of {d['pick_gm']:.3f} on these "
            f"shapes.) A time axis has to be lowest where the time is: write "
            f"it as the sum, or the larger, of an effect that falls and one "
            f"that rises as that field grows, each a time from hw.* rates.")
