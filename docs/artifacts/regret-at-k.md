# `regret@k` — the wall was not made by the metric

The pre-registration is [regret-at-k-prereg.md](regret-at-k-prereg.md).
**0 LLM calls.**

```
python3 experiments/regret_at_k.py     # -> docs/artifacts/regret-at-k.json
```

> ⚠️ 2026-09-08 (D-146): translated into English. The numbers and the verdicts
> are unchanged; the Korean original is at commit `ee53b4d`.

Everything is on the A6000 holdout of 20 shapes. Reported per seed, as **the
median and the range**.

---

## 1. ★ The suspicion was true — `tau` does not look at the noise

Every place that calls `kendalltau` was checked exhaustively
(`two_stage._measure`, `rank_evo_report`, `top_tau`, `degeneracy`). **All of
them pass `time_ms` as it is.** `variant="b"` treats values as tied **only
when the times are exactly equal**.

```
pairs within the top 100: 99,000
  pairs the noise cannot separate   46,679 (47.2%)
  pairs tau-b treats as tied        14,628 (14.8%)   ← only exactly equal times
★ the difference, 32.4%, **gets scored on an order that does not exist**
```

The rank loss **drops** those pairs (`NoiseModel.resolvable`). `tau` did not.
**Inside one experiment, training and evaluation used different criteria.**

⚠️ The instruction's "101,046 of 143,550 (70%)" is a **training-41-shape**
figure. On the holdout of 20 shapes it is 47.2%. They are not put side by side
(principle 4).

---

## 2. But the wall is unchanged — the verdict is **(c)**

```
                                     k=1    k=3    k=5   k=10   k=20   k=50  k=100
regret structure+regret w           1.076  1.066  1.072  1.075  1.084  1.099  1.106
★ rank structure+regret w           1.121  1.113  1.115  1.118  1.126  1.142  1.166
★ regret structure+rank w           1.528  1.458  1.451  1.448  1.472  1.488  1.496
rank structure+rank w               1.636  1.580  1.566  1.543  1.568  1.545  1.536
★ random floor                      1.715  1.945  2.066  2.268  2.339  2.370  2.378
```

### The seed ranges **do not overlap at any k**

The pre-registered criterion was the seed range.

```
   k   the regret arm's range   the rank arm's range   the gap
   1   1.0566~1.0919            1.1800~1.7047          +0.0880
   3   1.0549~1.0927            1.1497~1.6718          +0.0569
   5   1.0518~1.1038            1.1328~1.6298          +0.0290
  10   1.0657~1.1162            1.1468~1.6433          +0.0306
  20   1.0671~1.1138            1.1601~1.5806          +0.0463
  50   1.0690~1.1469            1.1526~1.5691          +0.0058   ← the closest
 100   1.0795~1.1774            1.2146~1.5572          +0.0372
```

**(c) the two curves stay apart even at k>=10.** The wall is real. The
current conclusion holds.

⚠️ At k=50 the gap of +0.0058 almost touches. What can be said is not "they
are definitely different" but **"they do not overlap"**.

### The random floor **grows** with k

```
the floor   1.715 -> 2.378   (k=1 -> 100)
the rules   all flat or better
```

`regret@k` is not a metric that gets easier as k grows — random gets worse.
**The result is that the gap between the two families holds regardless of k.**

---

## 3. (b) is **rejected** — the catastrophes are not concentrated in a few shapes

```
                                catastrophic (>1.15) shapes   seed range   union
                                (median/20)
regret structure+regret w                2.5                    1~ 5         8
★ rank structure+regret w                5.0                    5~ 7         9
★ regret structure+rank w               12.5                    5~20        20
rank structure+rank w                   14.0                   12~18        20
product term (prod)                     14.0                   13~16        20
k=10                                    15.0                   14~16        16
k=20                                    14.0                   14~16        16
k=50                                    17.0                   16~17        17
λ=1                                     13.0                   12~15        15
budget 16                               13.0                   12~14        15

★ shapes catastrophic in every arm: 7 / shapes catastrophic in any arm: 20
```

**The rank-weight arms are catastrophic on 12~18 of the 20 shapes.** It is
not "concentrated in a few shapes" but **failing to pick first place on most
of them.** The suspicion that the geomean was dragged by a few shapes is
rejected.

---

## 4. (b) The noise-aware `tau` — everything rises but **the order does not change**

The true ranks were tied together within the noise floor and re-measured. The
grouping is single linkage in ascending time — `NoiseModel.resolvable` is used
as it is (principle 2).

```
                                old tau   noise-aware   difference
regret structure+regret w        0.122       0.158       +0.036
★ rank structure+regret w        0.115       0.134       +0.019
★ regret structure+rank w        0.203       0.291       +0.089
rank structure+rank w            0.353       0.410       +0.057
product term (prod)              0.370       0.427       +0.057
k=20                             0.235       0.341       +0.106
k=50                             0.327       0.393       +0.065
λ=1                              0.290       0.377       +0.087
budget 16                        0.364       0.432       +0.068
```

**Everything rises** (+0.019 ~ +0.106) — direct evidence for §1. Scoring the
order inside the noise had been holding tau down.

★ **But the order of the arms does not change.** The regret-weight arms are
0.13~0.16 and the rank-weight arms 0.38~0.43. **The wall is there in the
fixed metric too.**

⚠️ The old tau values are not deleted — they are kept alongside (documentation
rule 2).

---

## 5. An aside — it is the **weights**, not the structure (D-103 reconfirmed)

```
★ rank structure + regret weights   regret@k 1.121 ~ 1.166
   regret structure + regret weights        1.076 ~ 1.106
```

Putting regret weights on a **structure** evolved with the rank loss brings it
almost up to the regret arm (a gap of 0.045~0.060 at every k). **The structure
is not broken** — D-103's conclusion that one set of weights cannot make both
orders at once holds in `regret@k` too.

---

## 6. What changed

```
changed      ★ it is now settled that tau did not look at the noise, and every
             value rose
             there was a problem in how the metric was chosen (§7)
unchanged    ★ the wall. The two families do not overlap at any k, the order
             is the same under the noise-aware tau, and the catastrophes are
             not concentrated in a few shapes
```

The conclusion of the six-direction experiment holds. **"tau did not move" was
a fact, and `regret@k` confirmed it with a different metric.**

---

## 7. ★ The way the metric was chosen was wrong

Setting the rank-loss experiment's decision line on `tau` happened at design
time. What the user first asked was **"does the regret rule show the
performance trend"**, and that turned into **"does it get the order exactly
right"**.

```
what we want to know   does it show the performance trend at the top
the metric used        tau — does it get the top **order** exactly right
★ instead of going back to what we want to know, **the metric that existed**
  was used
```

That the conclusion did not change is luck. Had `regret@k` been used from the
start, the noise problem of §4 would not have arisen either.
