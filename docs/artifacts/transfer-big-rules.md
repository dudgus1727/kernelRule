# Do the **big** rules transfer — (a) transplant · (b) refit

```
python3 -m experiments.transfer_big_rules     # -> docs/artifacts/transfer-big-rules.json
```

**0 LLM calls.** Source `a6000`, targets `5090` · `4090` · `h100`.
⚠️ **(c) regrow is not touched here** — that is a different experiment
(`transfer-29-5.md` · `c-ladder.md`) and its numbers are not overwritten.

## The three rules on the ladder

| rule | terms | weights | paths | AST nodes |
|---|--:|--:|--:|--:|
| r5 (D-156) | 21 | 21 | 2 | 503 |
| r11 (D-156) | 38 | 38 | 4 | 910 |
| old 8p (D-140, s3) | 8 | 8 | 1 | 259 |

⚠️ r5 and r11 come from **one seed** (D-156). The old 8p rule is the
**lower median of the six D-140 seeds by training regret** — chosen without
looking at any holdout.

⚠️ The fitter here follows `fitter_for(len(w0))` (D-144), so the two big
rules refit with CMA and the 8-parameter one with Nelder-Mead.
`transfer_29_5.py` fits everything with Nelder-Mead/300, so **these numbers
do not go beside the old transfer numbers** (principle 4).

## The ladder

`(a)` = the A6000 structure **and** its A6000 weights, scored on the target.
`(b)` = the structure fixed, the weights refitted on the target's training
split. `gap` = (b) holdout − (b) training.

### a6000 -> 5090   (training 41 shapes, holdout 20 of 53 common)

```
baseline   static top-1 1.0452 · vendor 1.2562 · human_guided refit 1.1401

rule                par/train      (a)      (b)  b train      gap  moved
r5 (D-156)             21/41     1.1215  ★1.0358   1.0171  +0.0187   2/2
r11 (D-156)            38/41     1.1225   1.0549   1.0168  +0.0381   2/2
old 8p                  8/41     1.2244   1.0813   1.0432  +0.0380   2/2
```

### a6000 -> 4090   (training 39 shapes, holdout 20 of 50 common)

```
baseline   static top-1 1.0403 · vendor none · human_guided refit 1.0761

rule                par/train      (a)      (b)  b train      gap  moved
r5 (D-156)             21/39     1.0515  ⛔1.2532   1.0270  +0.2262   2/2
r11 (D-156)            38/39     1.0497  ★1.0407   1.0237  +0.0170   2/2
old 8p                  8/39     1.0488   1.0565   1.0288  +0.0277   2/2
```

### a6000 -> h100   (training 39 shapes, holdout 20 of 48 common)

```
baseline   static top-1 1.1921 · vendor none · human_guided refit 1.0937

rule                par/train      (a)      (b)  b train      gap  moved
r5 (D-156)             21/39     1.1543  ★1.0526   1.0320  +0.0206   2/2
r11 (D-156)            38/39     1.5000   1.0919   1.0394  +0.0525   2/2
old 8p                  8/39     1.1965  ⛔1.2841   1.0456  +0.2386   2/2
```

## What it says

**1. (b) beats the target's own baselines on two of three targets.**

```
5090   (b) r5  1.0358  <  static top-1 1.0452  ·  vendor 1.2562
h100   (b) r5  1.0526  <  static top-1 1.1921  ·  human_guided 1.0937
4090   (b) r11 1.0407  ≈  static top-1 1.0403        ★ a tie (+0.0004)
```

★ The H100 one is worth marking: D-141 measured transfer there as **not
beating the baseline**. It does here — with a rule 21 terms wide instead of
8. ⚠️ Different rules, different fitter, one seed.

**2. The bigger rule is not the better traveller — and the direction is not
the same on the three targets.**

```
        r5 (21p)   r11 (38p)   difference
5090     1.0358     1.0549      r5 better by 0.019   < σ 0.037 -> not told apart
h100     1.0526     1.0919      r5 better by 0.039   ≈ σ        -> borderline
4090     1.2532     1.0407      r11 better by 0.213  ⛔ but see 3
```

**3. Two refits blew up, and the fitter is not the cause.**

```
r5 on 4090     (b) 1.2532 with training 1.0270  -> gap +0.2262
old 8p on h100 (b) 1.2841 with training 1.0456  -> gap +0.2386
★ both moved 2/2 — the fitter did move, it landed somewhere that does not
  generalise
```

**4. "More parameters than shapes" did not produce the overfitting.**

```
r11 on 4090   38 parameters / 39 training shapes   gap +0.0170  ← the smallest
r5  on 4090   21 parameters / 39 training shapes   gap +0.2262  ← the largest
```

The blowups are at 21/39 and 8/39, not at 38/39. Whatever produced them, it
is not the parameter-to-shape ratio in this data.

⚠️ The target training splits are **39~41 shapes**, not the 20~27 that was
assumed when this was asked for. The holdout is 20.

## ⚠️ What this cannot say

```
r5 and r11 are **one seed**'s products (D-156)
three targets x three rules = 9 cells, one number each
no significance is attached — the seed spread σ ≈ 0.037 is wider than most
of the differences above
```
