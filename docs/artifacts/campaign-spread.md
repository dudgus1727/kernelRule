# Campaign spread — **it is not larger than the seed spread** (and this corrects D-135 §5)

> **Reproduce**: `python3 experiments/campaign_spread.py` (**0** LLM calls)
> The curves' raw data: `round-curve.json` · `round-curve-p3.json` ·
> `round-curve-new.json` (all made by `experiments/round_curve.py`)
> **Raw data**: `campaign-spread.json`

> ⚠️ 2026-09-08 (D-146): translated into English. The numbers and the verdicts
> are unchanged; the Korean original is at commit `ee53b4d`.

## 0. Why it was measured

The decision line `delta = 0.0516` is a value made from the **seed spread
inside one campaign**, σ. But a good many of the comparisons we decide on are
**between different campaigns** (the expressiveness set, the hw ladder, the
transfer (b) vs (c), F1/F2/F3). If there is a separate campaign effect, that
component **does not shrink when seeds are added** — it is the floor of the
decision line.

## 1. ⚠️ First, the 0.0185 written earlier is **a value read at the wrong place**

D-135 §5 wrote "two campaigns using the same prompt are 0.0185 apart at r5".
**At that r5 one side (p3) already had some seeds stopped.**

```
p3 last round per seed   s0:r6 s1:r5 s2:r4 s3:r6 s4:r6 s5:r5
                         ★ the last round where everyone is alive is r4
```

A stopped seed's curve is **pinned at its last value** while only the running
side improves. That difference is not campaign spread but the fact that
**one side quit**.

**This does not mean the old value is deleted** — the correction is that, for
the reason above, 0.0185 **cannot be used as an estimate of the campaign
spread**.

## 2. Again, at a round where everyone is alive

```
round     old (old prompt·12r)   p3 (new prompt)      new24 (new prompt·24r)
          mean   median   σ      mean   median   σ     mean   median   σ
r4      1.0943 1.0949 .0117   1.1054 1.1052 .0109   1.1144 1.1129 .0188
r5      1.0943 1.0949 .0117    — some finished       1.1144 1.1129 .0188
r11     1.0759 1.0762 .0124    — some finished       1.1141 1.1121 .0187
r23      — finished (12r)      — some finished       1.0734 1.0787 .0141
```

### ★ Two campaigns under the same condition (p3 ↔ new24, r4)

`p3` and `new24` **have every setting the same except `patience`** (the
configs were flattened and compared — the only other item is
`loop.patience`). And before it stops, `patience` does nothing.

```
p3     1.0922 1.0967 1.0989 1.1115 1.1138 1.1193   mean 1.1054  σ 0.0109
new24  1.0922 1.1006 1.1019 1.1240 1.1277 1.1401   mean 1.1144  σ 0.0188

the campaign difference (mean)                          +0.0090
the standard deviation expected from the seed spread     0.0089
★ measured / expected = 1.0x
```

**The difference between the two campaigns is exactly the width predicted by
the seed spread alone.** The point estimate of the campaign component is
`σ(campaign) = 0.0011` — effectively 0.

## 3. So the decision line **does not change**

```
the current decision line (seed σ upper bound 0.0319 · n=6 · unpaired)   0.0516
with the campaign component added (point estimate)                       0.0518
the floor that remains however far seeds are increased                   0.0044
```

| n | seeds only | with the campaign component |
|---:|---:|---:|
| 3 | 0.0730 | 0.0731 |
| 6 | **0.0516** | **0.0518** |
| 12 | 0.0365 | 0.0367 |
| 24 | 0.0258 | 0.0262 |
| 96 | 0.0129 | 0.0136 |

**The past "indistinguishable" verdicts do not need to be revisited.** The
decision line moves by 0.0002.

## 4. ⚠️ What this estimate cannot say

```
campaign pairs under the same condition   1     -> 1 degree of freedom
σ(campaign) 95% upper bound               0.1437   ★ 16x the measured value —
                                                     it cannot be used as a bound
```

**What can be said**: the measured difference equals the width expected from
the seed spread alone.
**What cannot be said**: a **guarantee** that the campaign component is small.

Measuring it needs **3 or more** campaigns under the same condition (2+
degrees of freedom). Whether that is worth doing now is a separate question —
one campaign is 5,600 LLM calls.

## 5. ★ But something else shows up — the old campaign **leads at every matched round**

```
r4    old 1.0949  <  p3 1.1052  <  new24 1.1129
r5    old 1.0949  <                new24 1.1129
r11   old 1.0762  <                new24 1.1121
```

The sign is the same all three times, and at r11 it is **0.0359** — inside the
decision line 0.0516 so **indistinguishable**, but the point estimate is not 0.

```
the "prompt effect -0.0005" D-131 wrote
  ★ old (r5) against p3 (r4~r6, wherever each stopped) — **a comparison with
    the rounds not matched**
  ★ matching the rounds, old leads by 0.0103 at r4
```

**The value the two campaigns delivered is the same** (1.0762 at old's r11,
1.0787 at new24's r23, a difference of +0.0025). Only, **the new prompt spent
twice the rounds getting to the same place.** This is an observation, not a
verdict — it was not in the pre-registration and there is one campaign pair.
**It is written down as a place to look at in the next re-measurement.**

## 6. The statements that remain

```
★ the decision line 0.0516 is used as it is (campaign component point
  estimate 0.0011)
★ the current decision line is right for comparisons between campaigns too —
  no need to revisit past verdicts
⚠️ but that is a point estimate out of **one pair**
★ the old prompt leads at every matched round (indistinguishable, sign 3/3)
  — an observation
```
