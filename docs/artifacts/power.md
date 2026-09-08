# The weight **in the exponent slot** — form (b)

The pre-registration is [power-prereg.md](power-prereg.md). Producing the
numbers took **0 LLM calls**.

```
python3 experiments/power_report.py     # -> docs/artifacts/power.json
```

> ⚠️ 2026-09-08 (D-146): translated into English. The numbers and the verdicts
> are unchanged; the Korean original is at commit `ee53b4d`.

3 seeds x 12 rounds, budget 8, k=100, λ=0, the same seed rule, the model
`gpt-5.6-luna`. The baseline is `rankevo` — **the same condition, only
without the hint.**

---

## 1. Was the form actually used

```
                        proposals          archive        the fitted exponent
baseline (no hint)      0/432 ( 0.0%)     0/38 ( 0.0%)
exponent slot stated   78/432 (18.1%)     1/26 ( 3.8%)   [0.355]  |w-1| 0.645
```

★ **In the baseline it is 0 out of 432 proposals.** The checker never blocked
this form (the same spot as D-110) and **nobody used it.** Told about it, it
became 18%.

★ **But only 3.8% survive into the archive.** And the one exponent that
survived is **0.355** — 0.645 away from 1, so when it did survive it really
used that degree of freedom.

⚠️ The prompt example was fixed once. At first it was only
`np.power(f.a, w[3]) * w[4]`, which read as "two budget units", and since the
parent had 8 terms it collided with the same prompt's "the budget is full" —
**round 1 measured 0/12.** Once "**changing** `f.a * w[i]` **into**
`np.power(f.a, w[i])` **does not increase the budget**" was put in front, it
became 1/12. The real run used the fixed prompt (power-prereg.md §5).

## 2. ★ Was it tried and lost

"18% of proposals / 3.8% of the archive" reads two ways — **it was tried and
lost** and **it was a bad proposal regardless of the form.** They have to be
separated. 20 were drawn at random from each side of the same run's proposals
and fitted **by the same procedure**.

```
                    train rank loss   holdout regret   terms   (rank-loss range)
with exponent              1.1814           1.8113       8     0.390~2.583
without exponent           1.0333           1.8046       8     0.350~2.399
```

**Proposals with an exponent term are worse on the training rank loss** (1.18
vs 1.03). The holdout regret is the same. The ranges overlap heavily, so it
cannot be said that "the exponent is harmful", but it can be said that **they
were not accepted not because of the form but because they were simply
worse.** It was tried, it competed, and it lost.

## 3. The final metric — the holdout of 20 shapes, the median of 3 seeds

### ★ The regret refit — the cell the decision line is on

```
                        regret   top-100 tau   all-range   (tau range)
baseline                1.1213       0.115       0.267     -0.021~+0.155
exponent slot stated    1.1293       0.055       0.318     -0.020~+0.057
★ random floor          1.8753      -0.007       0.000
```

### The rank fit

```
                        regret   top-100 tau   all-range
baseline                1.6364       0.353       0.320
exponent slot stated    1.5714       0.362       0.500   ★ the highest
                                                          all-range yet
★ random floor          1.8753      -0.007       0.000
```

## 4. The verdict (pre-registration §7)

```
tau >= 0.20 and regret <= 1.15   -> the wall got lower
actual  tau +0.055 / regret 1.1293   -> ★ the wall is not a problem of the
                                          form
```

★ **And "does the exponent move away from 1" has an answer too.** The
surviving exponent is 0.355, clearly away from it. That is, **it is not "the
form was given and not used"** — 18% used it, and the ones that used it
competed and lost. That is the stronger negative.

An aside: in the rank fit the **all-range tau goes 0.320 -> 0.500**, the
highest so far (the seed range +0.359~+0.391 is narrow). The exponent shapes
**the whole-range form** well. But it does not show up in the top 100 or at
first place.

## 5. In common

```
                      terms   rej rate   fitter reach    min
baseline                8       1.4%      100.0%        95.3
exponent slot stated    8       3.7%      100.0%        72.0
```

A rejection rate of 3.7% — the newly added exponent guard does not block
normal proposals. None of the four abort conditions fired.
