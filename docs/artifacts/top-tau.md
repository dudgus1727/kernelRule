# Top-rank ordering ability — with the resolution controlled (2026-09-01)

> **Why** `degeneracy.md`'s top-rank tau of 0.141 is **"in between"** on the
> pre-registered decision line (`>=0.30 it ranks / <=0.10 it cannot`), but I
> wrote it down as "it cannot". And there was a resolution confound.
> **Reproduce** `python3 experiments/top_tau.py` · 0 LLM calls
> **Raw data** [top-tau.json](top-tau.json)

> ⚠️ 2026-09-08 (D-146): translated into English. The numbers and the verdicts
> are unchanged; the Korean original is at commit `ee53b4d`.

## The control

tau-b is computed only on the shapes whose true top 100 has **N or more
distinct time values**. The number of shapes left and **the random floor of
that same subset** (20 draws) are reported alongside.

## The result — ★ controlling makes it **go down**

```
A6000 training 41 shapes (the pre-registration condition)
structure      no control   uniq>=30   uniq>=50
s0                -0.106     -0.153     -0.153
s1                 0.145      0.040      0.016
s2                -0.118     -0.167     -0.181
s3                 0.150      0.177      0.253
s4                 0.213      0.290      0.351
s5                 0.138      0.152      0.153
★ random floor     0.002     -0.002     -0.012
shapes left           40         17          5
★ median of the 6  0.141      0.096      0.084
```

```
A6000 holdout 20 shapes   median  0.122  ->  0.121  ->  0.067  (20/12/5 shapes)
5090  holdout 20 shapes   median  0.075  ->   ★ cannot be controlled (0 shapes left)
```

**It was not the resolution.** Controlling for it does not raise the value,
it **lowers** it.

⚠️ But controlling shrinks the sample from 40 → 17 → 5 shapes. The median at
`>=50` is over **5 shapes** and is unstable.

⚠️ **The sign differs per structure.** `s0` (-0.153) and `s2` (-0.181) order
the top ranks **backwards** while `s4` (+0.290) orders them correctly. "The
rule can / cannot rank the top" cannot be said of the 6 structures as one.

## ★ The premise that the 5090 has less confound was wrong

```
                 tick        best median   top-100 span         distinct times
A6000        1024.00 ns      353.28 µs     32.3 µs = 31.5 ticks       34
5090           16.00 ns      185.34 µs      4.1 µs = ★ 256 ticks       5
```

**The 5090's top 100 spans 256 ticks and yet has only 5 distinct values.**
The resolution is more than enough and the values still clump into 5 levels —
**it is not quantisation, the performances genuinely overlap.**

The A6000 is the opposite. 34 distinct values inside 31.5 ticks — **about one
value per tick, right up against the resolution limit.**

```
★ The reason "the top ranks clump" differs between the two tables.
  A6000  the resolution limit
  5090   ★ the performances genuinely cannot be told apart
```

So on the 5090 **not one shape** meets the `>=30` condition. The control
itself is impossible.

## The verdict — against the pre-registered line

```
no control   0.141 (A6000 training 41)   -> the pre-registration's "in between"
★ controlled 0.096 / 0.084               -> "it cannot rank (<=0.10)"
5090         0.075                       -> "it cannot rank", but the control
                                            is impossible
```

**After the control it is on the "cannot rank" side.** But the ground differs
from the first thought — taking the resolution out did not improve it, it
made it worse. And the sample is 5~17 shapes with the sign differing per
structure.

### ★ So the statement is written like this

```
can be said   "inside the top 100 it can hardly rank at all.
              It is only slightly above the random floor (0.00) and the sign
              differs per structure"
★ cannot say  "it cannot rank because of the resolution"  — controlling does
              not raise it
★ cannot say  "the 6 structures consistently cannot rank" — s4 is 0.29~0.35
```

Read together with the all-range tau (0.30~0.56, random 0.00) the picture
holds — **the rule ranks on a coarse scale and can hardly rank at the top.**
That is consistent with the description `a top-1 selector` (D-100).
