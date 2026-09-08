# The re-run pre-registration — ★ nailed down **before** the run

> **Status**: registered (2026-08-26). Before the verification run
> **Reproduce**: `python3 experiments/rerun.py --verify` (the short verification)
>          `python3 experiments/rerun.py` (the real run)
> **Model**: `gpt-5.6-luna` / responses / `reasoning_effort=medium`
> **Table**: `datasets/rtx-a6000-sm_86-c63710df` (dev, its numbers must not be reported externally)

> ⚠️ 2026-09-08 (D-146): this document was translated into English. It is a
> frozen pre-registration plus its results, so **nothing was deleted** — the
> Korean original is at commit `ee53b4d`
> (`git show ee53b4d:docs/artifacts/rerun-preregistration.md`). Every number,
> criterion and verdict is unchanged; only the language is.

**Setting the criteria after seeing the results is contamination** (D-50). So
they are written here first. The criteria in this document have to be **the
same content** as the `PREREG` dictionary in `experiments/rerun.py`, and a
test pins that down.

---

## 0. Why it is re-run

All 12 runs evolved in a state where **half of them were scored without a
fit** (D-54). If the scoring was wrong, then everything chosen on top of it —
parent selection, archive updates, early stopping — was wrong too. The rule
ranking was also confirmed to change before and after the polish (Kendall tau
median 0.875, the best rule 11/12, D-57).

```
Refitting fixes only the final artefact. The evolutionary trajectory cannot
be undone.  (principle 13)
```

## 1. The design

```
scale       one condition, 6 seeds x 12 rounds x 12 proposals
condition   feature_detail = "full" (the feature descriptions are given)
state       the polish on (the strengthened form, D-59) + the invariant
            warnings (D-54) + the self-reported reach rate
model       gpt-5.6-luna / responses / medium — pinned (D-52)
split       the structural holdout nk11008 (the same as before — changing the
            split breaks the comparison)
```

**★ What is not done**

```
an A/B comparison            the conclusion is already in (inside the seed
                             spread, indistinguishable; D-53/54)
a comparison with a deleted   there must be nothing to compare against
value
cherry-picking seeds          all of them are used or none (D-40/D-46/D-50)
```

The reason A/B is excluded is **"because the conclusion is already in"**, not
"because the fitter fails there". The other order would be contamination too
(principle 18).

## 2. ★ The purpose and the expected result

```
purpose    "to obtain a value from an uncontaminated state"
           ★ it is not to beat the vendor

artefact   the structural-holdout median + quartiles + the per-regime
           decomposition
           -> this becomes this repository's **representative performance
              numbers**
```

**★ The expected result: indistinguishable from the vendor.**

```
By the D-53 calculation, the difference 6 seeds can tell apart is 0.03 or
more.
The currently estimated gap is around 0.02.
So "indistinguishable" is expected, and getting it is not a failure.
```

Writing the expectation down in advance **blocks rationalising afterwards.**
Both "it was better than expected" and "it was worse than expected" can only
be said against this line.

## 3. ★ For the vendor comparison, per shape is the main metric

```
secondary metric  a sign test over the 6 runs
                  the p lower bound is 0.031. At 5/6 it is p=0.22 and says
                  nothing

★ main metric     the median of the 6 runs at each shape vs the vendor
                  a sign test over the 20 shapes -> a p lower bound of ~1e-6
                  the variance enters once (the between-run variance is
                  absorbed by the median)
```

Per shape also avoids §30.4's problem that "the geomean is dragged by a few
shapes". **Both are reported, and per shape is the main metric.**

## 4. ★ The fitter pass condition and what to do on failure

The pass condition: the reach rate (4000 random points, regret@1). **It is a
single condition, so 12/12 is expected** (in D-60 condition A gave 12/12 =
100%).

```
1 failure          record it and continue
                   ★ do not take that run out of the results — taking it out
                     is selection bias (D-50)
2 or more          stop and report. (b) reconsider the regret@3 surrogate loss
gap over 0.03      stop regardless of the count
```

**Taking a failed run out of the results is the most dangerous thing.**

## 5. ★ The cost cap and the abort conditions

The credit has been used up twice (D-43).

```
cost cap        the expected calls and tokens are computed first and recorded
                in config.json
                `Budget` is set from those values, and going over raises

abort           LLMUnreachable aborts immediately (implemented)
                3 runs in a row with an empty archive stops it

partial run     if only 4 of the 6 seeds finish, **report on the 4 but state
                explicitly that "the design was 6 seeds"**
                ★ seeds are not cherry-picked
```

## 6. The verification run (before the real run)

```
python3 experiments/rerun.py --verify      # 1~2 runs x 6 rounds
```

What is looked at here is not the performance but **whether the machinery
runs**.

```
[ ] does the fitter reach rate come out 12/12 (n/n, scaled to the
    verification run)
[ ] do the invariant warnings actually fire
[ ] does the fit movement appear in the round summary
[ ] do the pre-registration values go into config.json as they are
[ ] do the artefacts survive being killed mid-run
```

**A verification-run result is not a representative number.** It is 6 rounds
and it is not pooled with the real run.

---

## The verification-run result (2026-08-26)

Reproduce: `python3 experiments/rerun.py --verify`
then `python3 experiments/fitter_movement.py verify-s0 verify-s1`

**★ These numbers are not representative.** It is 2 runs x 6 rounds and it is
not pooled with the real run. What is looked at here is not the performance
but **whether the machinery runs**.

### The checklist

```
[x] the fitter reach rate       4/4 = 100.0%   pass
[x] invariant warnings fired    59 of them — negative weights / a 100x
                                blow-up / the evaluation cap
[x] fit movement in the round   12/12, 9/12, 12/12, 7/8, 8/9, 8/10 ...
    summary
[x] the condition in            feature_detail is recorded in the llm block
    config.json                 ★ the condition is no longer a guess (D-60)
[x] prereg.json                 the pre-registration values are in the
                                artefact as they are
[x] survives being killed       RoundLoop dumps in a finally (D-33) + a
                                SIGTERM handler
```

### The fitter is definitely different

```
                     before (12 runs)   the verification run
fit movement rate    45.8%              7/8 ~ 12/12 per round
reach rate           83.3%              4/4 = 100.0%
```

The biggest difference is `fit moved 12/12` appearing in the round summary.
Before, half of them stayed at the initial values and **nobody knew** (D-54).

### What the invariant warnings actually caught

```
negative weights        every feature is "larger is worse", so a negative is
                        a structural error
a 100x weight blow-up   it went two orders of magnitude away from the initial
                        value
hitting the evaluation  it always fires — it is a state, not a warning
cap                     (principle 11)
```

The first two **discriminate** — they fire on some candidates only. The third
fires on all of them, so it is not a signal, and that fact is recorded in
D-55.

### The measured cost

```
2 runs x 6 rounds   154 calls   1,695 s (28 min)
-> estimate for the real run, 6 runs x 12 rounds: ~930 calls, 3~4 hours
   that is inside the cost cap of 1,395 calls
```

### What remains

```
The performance numbers are not looked at — it is a verification run and not
representative.
The §2 deletion scope is carried out before the real run.
```

---

## ★ The real-run result — the human arm was that condition (2026-08-27)

Reproduce: `python3 experiments/f1_pipeline.py F3 --stage 3 --seed-source architect
--tag arch24 --n-seeds 6 --rounds 12` and then
`python3 experiments/vendor_compare.py`,
`python3 experiments/fitter_movement.py f1pipe-F3-arch24-s{0..5}`

**The "human 24" arm run as the F1 control has the same condition as this
pre-registration.** It was not re-run separately — the conditions were
checked against each other.

| | the pre-registration | actual (the human arm) | |
|---|---|---|---|
| seeds | 6 | 6 | ✅ |
| rounds | 12 | 12 | ✅ |
| proposals per round | 12 | 12 | ✅ |
| `feature_detail` | full | full | ✅ |
| split | nk11008 | nk11008 | ✅ |
| model | gpt-5.6-luna | gpt-5.6-luna | ✅ |
| endpoint | responses | responses | ✅ |
| reasoning | medium | medium | ✅ |
| **the seed** | **not stated** | `architect-try05` | ★ |

**8/8 match and only the seed was missing from the pre-registration.** When
the pre-registration was written `physics_seeded` was tacitly assumed but not
written down — **a gap in the specification**. The seed effect was already
measured as "it does not decide the final result" (the D-54 investigation),
so 6 more seeds are not run with the `physics_seeded` seed.

### The fitter pass condition — passed

```
★ reach rate 12/12 = 100.0%   (4000 random points, regret@1)
movement rate   polish off 41.7% / on 66.7%  (diagnostic)
```

The failure policy written in the pre-registration (record 1, stop at 2, stop
when the gap exceeds 0.03) never had occasion to fire.

### ★ The main metric — the per-shape vendor comparison

```
the structural holdout, 20 shapes   vendor geomean 1.0737
```

| | win/loss/tie | sign test p | our geomean | vendor |
|---|---|---:|---:|---:|
| **per shape (main)** | 9 / 11 / 0 | **0.824** | **1.0650** | 1.0737 |
| per run (secondary) | 3 / 6 | 1.000 | — | — |

**Indistinguishable from the vendor.** Exactly the expectation written in the
pre-registration.

> ★ Indistinguishable from the vendor. By the D-53 calculation the difference
> 6 seeds can tell apart is 0.03 or more and the currently estimated gap is
> around 0.02. Getting "indistinguishable" is not a failure.

Our geomean is 0.0087 lower, but **per shape it does not move: 9:11.** It is
§30.4's situation where a few shapes drag the geomean — **this is where the
reason per shape is the main metric shows itself.**

### ★ The per-regime decomposition — the direction splits

| regime | shapes | ours | vendor | win/loss | p |
|---|---:|---:|---:|---:|---:|
| fast (SOL<0.5ms) | 12 | **1.0660** | 1.0994 | 8/4 | 0.388 |
| slow (SOL≥0.5ms) | 8 | 1.0635 | **1.0363** | 1/7 | 0.070 |

**We are better on the fast shapes and the vendor is better on the slow
ones.** Neither reaches significance, but the slow side's p=0.070 is on the
line. The direction agrees with §30.5 ("size comes first") — the shorter the
kernel, the more the vendor heuristic misses.

**"We beat the vendor on the fast regime" cannot be said** — 8/12 is p=0.388.
What can be said reaches only as far as **"there is a direction of winning on
the fast side and losing on the slow side"**.

### The representative performance numbers

```
this repository's representative values (the structural holdout of 20 shapes,
the median of 6 runs per shape)
  ours     geomean 1.0650
  vendor   geomean 1.0737
  verdict  ★ indistinguishable (per-shape sign test p = 0.824)

the per-run final scoring (a different procedure — do not mix them)
  median 1.0762   quartiles [1.0706, 1.0834]   seed spread σ 0.0124
```

⚠️ **The two numbers come from different procedures** (principle 4). The
first is "the median of 6 runs per shape -> geomean", the second is "a
geomean per run -> the median of the 6 runs". Do not put them side by side.
