# The axes the two objectives use — is the wall the budget's fault

0 LLM calls. Measured **before** the budget re-run so that the expectation
rests on a ground.

```
python3 experiments/feature_overlap.py     # -> docs/artifacts/feature-overlap.json
```

> ⚠️ 2026-09-08 (D-146): translated into English. The numbers and the verdicts
> are unchanged; the Korean original is at commit `ee53b4d`.

The targets are **one final best rule each** from the 6 `regret`-evolution
runs (`f1pipe-F3-arch24-s0..5`) and the 3 rank-loss-evolution runs
(`f1pipe-F3-rankevo-s0..2`). The two families are the same ones the 2x2 of
D-102·D-103 used.

---

## 1. The two families **use the same axes**

```
regret union  19
rank   union  19
★ jaccard 1.000 — there is not one axis present on only one side
```

Of the registry's 24, only `arith_intensity` was used by no rule. The other
four (`can_use_cp_async` `is_memory_bound` `log_sol_ms` `roofline_ratio`)
were used not as features but as **predicates** (`p.*`).

## 2. Even by rule pairs, between families is no further apart than within

★ A jaccard cannot be read without a floor (principle 7).

```
                        n     jaccard median (range)     axis union median (range)
within regret          15    0.647 (0.444~0.938)        17.0 (15~18)
within rank             3    0.684 (0.632~0.941)        19.0 (17~19)
★ between families     18    0.697 (0.579~0.938)        17.5 (16~19)
★ random floor       2000    0.455                      20.7
```

**Between families (0.697) is if anything higher than within a family (0.647
/ 0.684).** All three are above the floor (0.455) but they are not
distinguishable from each other.

    The two objectives **do not differ in which physics they point at.**
    It is the same direction as D-103 pointing at the weights.

## 3. The decision line — and that line was inaccurate

The line in the instruction was `union <= 8`. The between-family union median
is **17.5** and **0** of the 18 pairs are 8 or under. By the line as written,
"there is room for the budget to break through".

⚠️ **But the number of axes is not the budget unit.** The budget counts
`numeric literals + the number of weights`. A real rule holds several axes in
one term:

```python
s = s + np.where(p.is_memory_bound, f.log_dram_traffic, f.log_inst_total) * w[4]
#          1 predicate + 2 features = 3 axes in 1 weight
```

Measured, one term holds **1.62~2.50 axes (median 2.12)**, predicates
included. All 9 rules have 8 terms while using 13~20 axes.

```
the between-family union (axes+predicates)   18~22
holding that at that density needs           ★ 9.4 terms
-> 8 is short. The shortfall is about 1.4 terms. 16 is comfortable.
```

**The "8 or under" line assumed 1 term = 1 axis, and the data denies that.**
The line's direction (there is room for the budget to break through) holds in
the converted unit too, but **that the margin is only 1.4 terms** was not
visible from the original line.

## 4. The ceiling measurement's five axes are in the regret rules too

```
                       regret     rank
split_k_cost            6/6        3/3
sm_idle_cost            5/6        3/3
waves                   4/6        3/3
pipeline_warmup_frac    4/6        2/3
tail_waste              2/6        2/3
```

**The five that move a lot inside the top 100 are already in the `regret`
rules.** The difference in ranking ability is not explained by "it did not
look at those axes".

---

## 5. So, the expectation for budget 16

The two branches point in different directions.

```
axis capacity   the union is 1.4 terms short of 8      -> 16 could help
mechanism       the two families use **the same axes** (jaccard 1.000)
                and the difference is in the weights (D-103)  -> 16 will not
                                                                 help
```

★ **Expectation: budget 16 does not break through the wall.** The regret
refit tau was −0.059~0.058 at 8, and it is expected to stay inside that band
at 16 as well. The ground is the jaccard of 1.000 — it is not that there are
too few axes to hold, but that **on the same axes one set of weights cannot
make both orders at once.**

That said, 16 terms do allow **the same axis to be used several times with
different predicates and weights**. If that degree of freedom breaks the
wall, the expectation was wrong, and it would then mean the wall was an
"expressiveness" problem. It is run to measure that.
