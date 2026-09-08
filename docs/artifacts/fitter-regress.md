# ★ The fitter is **not the culprit** — 1.0762 -> 1.0987 is not the fitter's fault

> **Reproduce**: `python3 experiments/fitter_regress.py` (about 6 minutes,
> **0 LLM calls**)
> **Pre-registration**: [fitter-regress-prereg.md](fitter-regress-prereg.md)
> **Raw data**: `docs/artifacts/fitter-regress.json`
> **Table**: `datasets/rtx-a6000-sm_86-c63710df` (dev), split `nk11008`

> ⚠️ 2026-09-08 (D-146): translated into English. The numbers and the verdicts
> are unchanged; the Korean original is at commit `ee53b4d`.

## 1. What got caught

Re-taking the baseline in D-124 gave the opposite direction.

```
the old baseline (Nelder-Mead 200/600, 6 seeds)   1.0762
the new baseline (CMA-ES 300/600, 3 seeds)        1.0987   worse by 0.0225
```

It could have become "CMA fits the training better and the holdout gets worse
= the fitter makes overfitting". **The variables are cut down to the fitter
alone to tell them apart.**

## 2. ★ The result — the same structures, two fitters

The 6 `arch24` structures are **refitted** with the two fitters and the
holdout is measured. The structures, the seeds, the budget and the split are
all the same.

| arm | training median | holdout median | gap | holdout range |
|---|---:|---:|---:|---|
| NM (nelder-mead/4) | 1.0495 | **1.0762** | +0.0186 | 1.0566~1.0919 |
| CMA (cma/1) | 1.0485 | **1.0753** | +0.0135 | 1.0546~1.0919 |

```
the paired differences (CMA - NM, positive = CMA is worse)
  +0.0012  -0.0145  +0.0000  -0.0006  -0.0013  -0.0020
  structures where CMA is worse 1/6   median -0.0010
  ★ paired one-sided Wilcoxon p = 0.9375 — not significant
```

**They are indistinguishable.** If anything CMA is slightly better (median
-0.0010, one twelfth of σ).

★ **The NM arm reproduced `1.0762` exactly** — the final score that was
reported. That means the measurement path is sound (principle 1), and that is
why this comparison is trusted.

### The mechanism hypothesis does not stand either

```
structures where CMA is better in training   3/6      (expected "almost
                                                       certainly CMA")
median of the gap differences             -0.0008   (expected "CMA opens up
                                                     more")
```

**In 8 dimensions CMA does not even fit the training better.** What D-123
measured is the **16-dimensional reach**, and in 8 dimensions the two fitters
find practically the same point (the same story as D-77's A8 reach of 100%).

## 3. Then where did the 0.0225 come from

The 3 `rb08` structures (§3's budget-8 arm) were measured with the same two
fitters.

| | training | holdout | holdout per seed |
|---|---:|---:|---|
| rb08 + NM | 1.0858 | 1.1146 | 1.1463 / 1.1146 / 1.0923 |
| rb08 + CMA | 1.0836 | 1.0987 | 1.0926 / 1.1186 / 1.0987 |

**Whichever fitter measures it, the `rb08` structures are worse than the
`arch24` structures.**

```
arch24 6 seeds   1.0566 1.0702 1.0719 1.0806 1.0844 1.0919
rb08   3 seeds                                   1.0926 1.0987 1.1186
★ the three rb08 seeds are worse than **all six** arch24 seeds
```

**The difference is in the structures the evolution made, not in the
measurement.** It is not the fitter.

### The two campaigns differ in more than the fitter

```
same        seed code hash d5ee6da8 · 19 features · model gpt-5.6-luna
            split nk11008 · 12 rounds x 12 proposals · objective regret ·
            budget 8
different   fitting budget 200 -> 300 · in-loop fitter NM/4 -> CMA/1
            ★ and **every prompt that changed in between** — arch24 is before
              D-113·116·117 (generating the hw prompt from the bundle), and
              the board changed several times after D-117's hypothesis-field
              allow list and D-78's literal rule
```

**So `1.0762` and `1.0987` cannot be put side by side in the first place**
(principle 4). D-124 was right to re-take the budget-8 arm — the four arms ran
**on the same board on the same day**, and comparisons inside that set do not
have this problem.

## 4. An aside — the seed range is wider than σ

From the per-seed values of §3's four arms:

```
rb08   1.0926 1.1186 1.0987   standard deviation 0.0136
rb16   1.0906 1.0964 1.0764                      0.0103
rprod  1.0944 1.0840 1.0794                      0.0075
rpow   1.0839 1.1063 1.0319   ★                  0.0382
                                 the σ used for the verdict = 0.0124
```

**The exponent arm's seed range is 3x σ.** The decision line 0.0516 is a value
made from σ=0.0124, so in that arm **the power is lower than the
pre-registration assumed** — one more reason not to read "indistinguishable"
as "there is no difference".

## 5. The verdict — the third branch of the pre-registration's §3

```
★ indistinguishable — the fitter does not make overfitting
-> there is no reason to go back to NM at budget 8
-> the 4090 transfer's (b) refit uses **the procedure so far (NM) as it is**
   because it has to be put beside the old transfer numbers (principle 4)
-> CMA only when 16 dimensions are needed (D-123 stands as it is)
```

⚠️ This does not overturn D-123. That is about the **16-dimensional reach**
and this is about **8-dimensional generalisation** — both are facts (written
in advance in the pre-registration's §5).
