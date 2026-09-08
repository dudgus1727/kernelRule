# The conclusion — indistinguishable from the vendor. The library decides the result

> **Status**: canonical
> **Updated** 2026-09-03 · the rank axis is closed (D-104~D-112).
> ★ **Read it straight down from the top** (the order was put right on
> 2026-09-03). The correction history is **the last two sections** — those are
> not the current values.
>
> ```
> the canonical numbers -> the two artefacts -> the results that survived ->
> the wall -> the room that is left -> the comparison table / the regimes /
> shippable -> F1 -> the open questions -> what is not implemented
> -> ★ the correction history (at the very bottom)
> ```
>
> ⚠️ 2026-09-08 (D-146): this document was translated into English. The
> numbers, the verdicts and the correction history are unchanged; the Korean
> original is at commit `ee53b4d`. The reproduction command in the canonical
> section keeps its arm label verbatim — it is what produced the committed
> json's key.

This document is not a set of certainties — it collects **only what has a
reproduction procedure attached**.

## ★ What was built — not a "heuristic" but **a cost model that only gets the ranking right**

```
the structure   ★ a weighted sum of physical quantities — it is an analytical
                cost model
the coefficients ★ not derived from physics but fitted on the table — it is
                learning
the goal        ★ not absolute time but **the ranking**. Getting the argmin
                right is enough
```

**The structure is an analytical model and the coefficients are learned. Only,
it does not fit absolute time but only the ranking, so it is less constrained
than a prediction model.**

And that fact connects to §29's wall — because only the ranking has to be
right, a state where **top-1 is right while the whole order is wrong** is
possible, and that is what happens (tau inside the true top 100 is 0.141).
That is the reason it must not be read as a "performance model".

⚠️ The selection cost follows the same character — **evaluating** the rule is
297 µs for 15,000 configs, but with the feature computation of the current
Python implementation it is 13 ms ([select-cost.md](select-cost.md)).

> ## ★ 2026-09-01 — what was built got settled: **a top-1 selector**
>
> It came out of the degeneracy check ([degeneracy.md](degeneracy.md),
> following the pre-registration).
>
> ```
> it is not degenerate   a median of 14.5 distinct configs over 41 shapes, 0
>                        shapes overlapping with static top-1
>                        all-range tau 0.30~0.56 against random 0.00
> ★ but              tau inside the true top 100 is only 0.141 (0.096 after
>                    controlling for resolution), and the ranking ability
>                    drops from 16x random to 4x under transfer.
>                    ★ Refitting the weights does not bring it back
>                    (4 -> 4.6x)
> ```
>
> **So this artefact is not a "performance model" but a `top-1 selector`.**
> Any description inside this document that reads as "it learned physics /
> it explains runtime" has to be read narrowed to that scope.
>
> ★ **The claim does not shrink.** As a top-1 selector it is on a par with the
> vendor (9 wins 11 losses, p = 0.824 — ⚠️ **the canonical value changed on
> 2026-09-06: 8 wins 11 losses 1 tie, p = 0.648.** See the canonical section
> below), it runs inside µs, and **the structure transfers to a GPU of a
> different architecture** — (b) refit is indistinguishable from (c)
> regeneration and it takes 0 LLM calls (D-98). That is this project's
> strongest result.

> ## ★ 2026-09-01 (2) — **the objective decides the character of the artefact**
>
> From the same pipeline · the same features · the same seed, **changing only
> the objective** produced two artefacts of different character (D-101).
>
> ```
>                        evolved with        regret     top-100 tau
> ★ the top-1 selector   regret              1.049        0.141
> ★ the performance model the top-100 rank loss  1.47    ★ 0.389
> ```
>
> **The two are not a better and a worse version of one thing — they are
> different objects.**
>
> ### "it learned physics" is taken back, conditionally
>
> ```
> ⛔ cannot be used   "our rule learned physics"
> ★ can be used      "evolving it with the rank loss produces a rule that
>                     **explains the order of runtimes**. Only, it cannot be
>                     used for top-1 selection (regret 1.47)"
> ```
>
> There is one more piece of ground — the five axes the ceiling measurement
> pointed at as "what varies a lot inside the top 100"
> (`split_k_cost` `sm_idle_cost` `waves` `pipeline_warmup_frac` `tail_waste`)
> are the ones **the evolution actually picked**
> ([ranking-ceiling.md](ranking-ceiling.md) §3).
>
> ### Which one is put forward
>
> ```
> the practical claim (on a par with the vendor · µs · transfer)
>                                        ★ the top-1 selector side
> the interpretive claim (it represents physics)   ★ the rank side
> ```
>
> **Both are not claimed for one artefact.** The attempt to join the two is
> measured by [two-stage-prereg.md](two-stage-prereg.md).
>
> ⚠️ It is 3 seeds and the tau is measured on the **training 41 shapes**. The
> holdout value comes fresh out of that pre-registration's A — do not put them
> side by side now (principle 4).

---

> ## ⚠️ Block 3.5 of this run was in a contaminated state (confirmed 2026-08-21, D-28)
>
> The `table_facts` injected into the diagnostic report were sentences
> computed over **all 66 shapes / 61 shapes for a888**. Aggregates of the
> validation and final splits went into the prompt. §12.3 blocked only the
> "holdout score" and the aggregates slipped through.
>
> **It is not re-run.** How much of that information the LLM used cannot be
> known, so **a reservation is attached to the interpretation of this run's
> validation score.** The training score and the structure-related
> observations are little affected, and the generalisation claim against the
> validation/final splits gets weaker.
>
> The bypass path was closed in `report/table_facts.py` — later runs compute
> from the training split only.

## ★ The canonical performance numbers (settled 2026-08-27, D-69)

**A new session starts here.**

> ### ★ This section is **the canonical one** (2026-09-03)
>
> Other documents have performance numbers too and **they are all values of a
> different procedure**. They are not wrong, the procedures differ, and one
> had to read all of them to know which was canonical. **Now the canonical
> value is this one place** (principle 2).
>
> ```
> ★ updated 2026-09-06 (D-140) — the canonical run changed to F3rw-p8-nan
> 1.0827   ★ canonical — the median of 6 runs per shape -> geomean, the
>          structure holdout of 20 shapes
> 1.0737   ★ canonical — the vendor under the same procedure (it does not
>          change)
> 1.0886   geomean per run -> the median of 6 runs   (the same run, a
>          different aggregation)
>
> ⚠️ Below are the **old canonical values** (F3rw-p8-old). They are not deleted
> 1.0650   old — the median of 6 runs per shape -> geomean
> 1.0762   old — geomean per run -> the median of 6 runs   (the same run, a
>          different aggregation)
> 1.0797   design.md §30 — the dev table / a different split / a different
>          point in time
> 1.0757   design.md §29 — the initial structure-holdout value
> 1.0680   principles.md §? — the example that it is not physics_seeded
> ```
>
> **Do not put a value from another document beside this table** (principle 4
> / documentation rule 3). Those values are not deleted — they record the
> procedure of their time.

### How it was obtained

```
the fixed fitter (reach 12/12 = 100%, 4000 random points) + the tidied prompt
(-64%)
a single condition (with feature descriptions) 6 seeds x 12 rounds x 12
proposals
gpt-5.6-luna / responses / medium
seed: Architect (the 24 human ones, condition A, try05)
the structure holdout nk11008 (20 shapes)
```

The pre-registration is `rerun-preregistration.md`. 8/8 of the conditions
match.

**It passed the gate because it is a single condition.** With A/B mixed the
reach was 0.833 (20/24), below the bar, and all 4 failures were on the "no
feature descriptions" side (D-60).

### The main metric — the per-shape vendor comparison

| canonical run | win/loss/tie | sign-test p | ours | vendor |
|---|---|---:|---:|---:|
| ★ **F3rw-p8-nan r11 (now)** | 8 / 11 / 1 | **0.648** | **1.0827** | 1.0737 |
| F3rw-p8-nan r23 (for reference) | 11 / 9 / 0 | 0.824 | 1.0791 | 1.0737 |
| ⚠️ F3rw-p8-old (the old canonical) | 9 / 11 / 0 | 0.824 | 1.0650 | 1.0737 |

Secondary (per run): now 1/6 (p=0.219) · old 3/6 (p=1.000)

```
★ indistinguishable from the vendor — ★ that holds even though the canonical
  value changed (p 0.648~0.824). The pre-registration says in advance that
  "indistinguishable is expected and getting it is not a failure" — this is
  not rationalising after the fact.

⚠️ But **the sign of the geomean flipped**
   old   ours 1.0650  <  vendor 1.0737   (we are better)
   now   ours 1.0827  >  vendor 1.0737   (the vendor is better)
   -> "indistinguishable" holds but "we lead on the geomean" cannot be used
```

> **Reproduce**: `python3 experiments/vendor_compare.py --arm
> "이름=F3rw-p8-nan-s" --round 11 --out docs/artifacts/vendor-nan-r11.json`
> (the arm label is kept verbatim — it is the key in the committed json)
> Raw data `vendor-nan.json` · `vendor-nan-r11.json` · the full text
> [canon-nan.md](canon-nan.md) · [D-140](../decisions.md)

### By regime — the direction splits

★ The current canonical value (F3rw-p8-nan r11):

| regime | shapes | ours | vendor | win/loss | p |
|---|---:|---:|---:|---:|---:|
| fast (SOL<0.5ms) | 12 | **1.0932** | 1.0994 | 7/4 | 0.549 |
| slow (SOL≥0.5ms) | 8 | 1.0673 | **1.0363** | 1/7 | 0.070 |

⚠️ The old canonical value (F3rw-p8-old) — it is not deleted:

| regime | shapes | ours | vendor | win/loss | p |
|---|---:|---:|---:|---:|---:|
| fast (SOL<0.5ms) | 12 | **1.0660** | 1.0994 | 8/4 | 0.388 |
| slow (SOL≥0.5ms) | 8 | 1.0635 | **1.0363** | 1/7 | 0.070 |

**"We beat the vendor in the fast regime" cannot be written** — 8/12 is
p=0.388. What can be said goes as far as **there is a direction**. And **the
side where we lose (slow, p=0.070) is the closer one to significance.** The
slow regime has only 1.5% of room in it (§9.2b).

### ⚠️ Do not mix two numbers from different procedures (principle 4)

```
the median of 6 runs per shape -> geomean       1.0650   ★ the main metric
geomean per run -> the median of the 6 runs     1.0762   (quartiles [1.0706,
                                                          1.0834])
```

Both are legitimate and they are different procedures. They are not put side
by side.

---

## ★ The headline — it is not performance, it is performance per cost

The gate is not a single threshold but **a table** (§9.2c). "Passed / did not
pass" is not said from one cell.

| | build cost | regret@1 (structure holdout) | runtime |
|---|---|---:|---|
| a random config | 0 | 1.671 | µs |
| static top-1 | minutes | 1.115 | µs |
| **kernelRule** | **about 1 hour of LLM** | **1.0650** (the per-shape procedure) | µs |
| GBDT | hours (★ transfer fails) | 1.019 | ms |
| the vendor (nearest) | **years of people, a team** (estimated) | **1.0737** | µs |

### ⚠️ What is not in the cost column

```
counted       5,019 LLM calls (exploration) / ~2,830 (reproduction)
not counted   48 hours of kernelTab measurement / defining 24 features /
              15,000 lines of pipeline / dozens of Claude Code sessions
```

**"Overnight" is the cost of making one new rule when the pipeline and the
features already exist.** And **the vendor's "years" is not a value we
measured, it is an estimate**, and the cost of **updating** it with one more
architecture is not years. Do not compute a ratio out of this table
(`cost.md`).

### The shape of the claim

```
not   "more accurate than the vendor"
but   "it **fills the gap** before the vendor heuristic matures on new
       hardware" (§16.3)
```

⚠️ **Do not read 1.0650 and 1.0737 as "we won".** The per-shape sign test is
9:11, p=0.824 — **indistinguishable** (D-69). The geomean difference is pulled
by a few shapes (§30.4).

---

> ## ⚠️ The provenance problem of the performance numbers (2026-08-26)
>
> A good many of the tables below came out of **`gpt-5.4`**. That model **was
> introduced without instruction** while making the Architect control, and
> several experiments piled up on top of it (D-52).
>
> **The instructed model is `gpt-5.6-luna`.** Only what is reproduced on
> `luna` is used in the conclusion. The `gpt-5.4` results are not deleted but
> **they are not cited as grounds for the conclusion.**
>
> **The `gpt-5.4` runs were deleted from `runs/`** — those numbers cannot be
> reproduced, and the related artefacts are marked that way in their badge.
>
> What is settled on `luna` with 6 seeds right now:
> ```
> the structure holdout  median 1.1019  range 0.0805  standard deviation 0.0274
> the fast regime        seeds that beat the vendor 2/6, sign test p = 1.000
> the rejection rate     0.8%  100 seconds per round  output 19k
> ```
> **There is no performance claim yet.** The median is worse than the vendor's
> (1.0737).

> ## ⚠️ The table below is a single-seed value (confirmed 2026-08-23)
>
> Running the same condition with 10 seeds spreads the structure holdout over
> **1.0518 ~ 1.1496 (a range of 0.0977)**. **Any comparison below that is
> smaller than that range is "indistinguishable"** (D-40).
>
> | comparison | difference | against the range |
> |---|---:|---|
> | with / without feature descriptions | 0.058 | 60% — the only one that survives |
> | the Architect seed | 0.022 | 1/4 |
> | the new axes A/B | 0.016 | 1/6 |
> | "on a par with the vendor" | — | it was a single seed |
>
> **This repository has not yet passed the vendor (structure holdout
> 1.0737).** The median of 10 seeds is 1.0872, and re-checking the seed
> selection procedure on a new bundle gave 1.0865.
>
> ### The results that survived the range
>
> | result | why it is valid |
> |---|---|
> | **the feature-description effect 0.058** | 60% of the range 0.0977 |
> | **F1 rediscovery 7~9/24** | ★ an **observation**, not performance, so it is unrelated to the seed range |
> | **10 new F1 axes, 8/10 used in a rule** | the same as above |
> | **`split_k_io_amplification` is new information** | correlation 0.30~0.46. A deterministic computation |
> | **one `has_spill` term takes 1.1637 → 3.1841** | deterministic. Term ablation |
> | **the static top-1 / vendor / GBDT baselines** | nothing to do with the LLM |
>
> ### What is buried in the range
>
> | result | difference | against the range |
> |---|---:|---|
> | the Architect seed is harmful | 0.022 | 1/4 |
> | the new axes A/B | 0.016 | 1/6 |
> | "on a par with the vendor" | — | a single seed |
>
> **"The only thing that survived is the description effect" was inaccurate**
> — the seed range applies only to **the performance metrics of an evolution
> run**. A **deterministic computation** or an **observation**, like the
> rediscovery count, the correlations or the term ablation, is not affected
> by it.

## ★ The results that survived (2026-08-27)

```
✅ the canonical performance   indistinguishable from the vendor (per shape
                              9:11, p=0.824)                            D-69
✅ the regime direction        we win in fast and lose in slow (neither
                              significant)                              D-69
✅ a rule gets built out of the F1 library                              D-68
✅ F1 does not catch up with the 24 human ones (+0.0433, p=0.002)       D-68
✅ the range is narrowed by evolution (4.4x -> 1.42x)                   D-68
✅ the prompt rewrite raises rediscovery from 0 to 6                    D-63
✅ the LLM divides physics into 7 areas and is stable across three times D-63
✅ the discrimination limit 0.03 (seed range σ=0.0274)                  D-53
✅ the fitter reach 100% (a single condition)                     D-60, D-61
⚠️ ★ but **the refit reach is 0.75 in 8 dimensions** — refitting from random
   starting points finds a better place 3/12 of the time (a gap of up to
   0.0110). **A difference under 0.011 can be fitter starting-point noise.**
                                                                        D-77
❌ ★ **it collapses in 16 dimensions** — refit reach 0.42, a gap of up to
   0.0462. The budget 8 vs 16 experiment is not started; the fitter is fixed
   first                                                                D-77
✅ no expected_range leak (21/21 exactly as declared)                   D-71

❌ invalid   "5 out of 6 in the fast regime" (gpt-5.4, the original deleted)
❌ invalid   the feature-description effect 0.058 (gpt-5.4). On luna it is
             0.016, p=1.000
❌ invalid   F1 rediscovery 7~9/24 (gpt-5.4). It depends on the model and on
             the prompt
❌ invalid   Architect condition A 1.1942 (the state where the prompt had a
             contradiction in it, §30.10.2b)
```

### The cost

```
exploration     33 runs    about 5,019 LLM calls
reproduction    18 runs    about 2,830 LLM calls + 1,872 for the F1 pipeline
                           + 21 for Architect
```

---

## ★ The wall — it was pushed from six directions and it did not move (2026-09-03)

This is the conclusion of the rank-loss axis. **That itself is a result.**

### What the wall is

```
a rule whose regret is 1.11~1.13 has a top-100 tau near 0
a rule whose tau is 0.33~0.37 has a regret of 1.59~1.69
```

Even with the same structure, **which objective the weights are fitted with**
splits the tau into 0.02 and 0.37 (D-103). One weight vector cannot make
"picking first place" and "the order of the top" at the same time.

### It was pushed six times

| direction | what was changed | result |
|---|---|---|
| structure | `regret` evolution vs rank evolution | axis Jaccard **1.000** — they use the same axes |
| budget | term budget 8 vs 16 | indistinguishable. Even with the terms growing 8->13 (D-108) |
| order | `rank->regret` / `regret->rank` | the wall is the same in both (D-104) |
| expressiveness | feature products | **it already worked**. Stating it takes 66->88% but the wall is the same (D-110) |
| the goal's definition | `k` 10~100 / `λ` 0~3 | narrowing k is worse if anything, and λ is a straight line (D-109/D-111) |
| form | the weight **in the exponent slot** | 18% of the proposals used it and **it competed and lost** (D-112) |

★ The last two matter. **It is not "we do not know because we did not try"**
— the product was never blocked by the checker and 59% of the rules were
already using it, and once told about the exponent 18% tried it and it was
worse from the training objective onward.

### ★ Three of those six, the expressiveness ones, were **re-measured on the regret path** (D-124)

The budget, the product and the exponent above are **all on the rank-loss
path**. That objective turned out to be the wrong one (D-118·D-121), so the
conclusion was standing on a wrong objective. The fitter was resolved first
(D-123 — CMA-ES) and the three were re-run on the regret@1 path.

```
                the baseline 1.0987   (budget 8, CMA 300/600, 3 seeds)
budget 16        1.0906   +0.0081   indistinguishable
the product hint 1.0840   +0.0147   indistinguishable
the exponent hint 1.0839  +0.0148   indistinguishable   decision line 0.0516
★ all three indistinguishable — on the regret path too, expressiveness cannot
  lower the wall
```

⚠️ The baseline is not `1.0762`. The fitter differs so they cannot be put side
by side (principle 4) — that is why the budget-8 arm was re-taken under the
same condition.

★ Only, **in `regret@k` it does change**: the exponent arm at k=10/50/100 and
the budget-16 arm at k=10/50 have seed ranges that **do not overlap** with the
baseline. The decision line was not in the pre-registration and this is an
observation over 12 cells at 3 against 3, so **it is not used as a verdict**.
Expressiveness appears to work in **the top region** rather than in "picking
first place".

★ The exponent form **survives** on the regret path — archive survival 3.8%
(rank) -> **62.5%** (regret). D-112's "it competed and lost" was a conclusion
under that objective.

### So the statement that remains

```
★ **a linear combination of** our feature space does not predict the order of
   the top runtimes well enough
```

This is **the same story** as the gap to GBDT:

```
ours (the human arm)   1.0762
GBDT                   1.019     ★ the ceiling learning can reach
the gap                0.0572    4.6x the seed range σ 0.0124
```

GBDT uses the same features while using **non-linear interactions freely**.
What we widened (the number of terms, the number of nodes, products,
exponents) is all "the shape of one term", and it is still **a sum of terms**.
The wall and the GBDT gap are read as two expressions of one thing.

⚠️ **The limit of the reading.** This is not "a linear combination is
impossible in principle" but **"it was pushed in six directions and it did not
move at 3 seeds"**. Each experiment is 3 seeds so none is individually
significant, and the verdicts were made **only by range separation**.

### ★ The exact statement of the wall (2026-09-03, D-118)

```
the regret rule   finds a good region. ★ It cannot order what is inside it
the rank rule     orders things.       ★ It does not know which region is good
```

The rank rule **orders the true top 100 when it is given** (noise-aware tau
0.410). But **when it picks the top 100 itself it picks the wrong ones**
(regret@100 = 1.536, the random floor 2.378). It is right locally and wrong
globally.

Three metrics point the same way — the old tau / the noise-aware tau /
`regret@k`. **The wall is not an artefact of one metric**
([regret-at-k.md](regret-at-k.md)).

⚠️ In the course of that check **a real defect of the old `tau`** came out: it
was scoring pairs the noise cannot separate (47.2% of the holdout top 100).
The training side of the same experiment (the rank loss) dropped those pairs.
All the values rose (+0.019~+0.106) but **the order of the arms does not
change.**

### An aside — the wall is on the "top" side

When the exponent slot was opened, **the all-range tau went 0.320 -> 0.500**,
the highest so far (D-112). It makes the whole-range shape better while
nothing shows up in the top 100 or at first place. **What cannot be got right
is not the whole range but the top.**

### What was not tried (`pending_fixes` 11)

Splitting the regimes into three or more — the only remaining direction that
breaks the very fact that there is **one** weight vector per shape. The
parameters double, and 32 weights for 41 shapes is a bad ratio (D-98's refit
sample requirement grows too). **A new pre-registration is needed.**

## ★ The room that is left — what to aim at is not the difference between conditions

```
ours (the human arm)   1.0762
GBDT                   1.019    ★ the ceiling learning can reach
the gap                0.0572   4.6x the seed range σ 0.0124 — it can be told
                                apart
```

**A difference of order 0.02 between conditions cannot be told apart** (D-53)
**but the GBDT gap is different.** It does not slide into "improving
performance is hopeless".

## 1. ★ The only comparable table

**Procedure** a 2-way SOL split (a proxy metric) → fit the weights separately
per regime → evaluate per regime and combine over the 61 shapes. The holdout
is 1 in every 3 in SOL order inside each regime (19 shapes).
**Reproduce** `python3 experiments/rescore_canonical.py`

| | in-sample 61 | holdout 19 | holdout significance vs the vendor |
|---|---:|---:|---|
| **`evolved`** (the first run) | **1.0652** | **1.0628** | **11 wins 7 losses 1 tie** |
| **the vendor, nearest** ★the gate | 1.0797 | 1.0864 | — |
| Architect B (`gpt-5.4`) | 1.1564 | 1.1600 | 7 wins 11 losses 1 tie |
| `physics_seeded` | 1.1637 | 1.1652 | 6 wins 11 losses 2 ties |
| Architect A (`gpt-5.4`) | 1.1780 | 1.1961 | 7 wins 11 losses 1 tie |
| Architect A (`gpt-5.4-mini`) | 1.4797 | 2.1317 | 3 wins 16 losses 0 ties |

**Only the evolved rule beats the vendor.** Every structure written in one go
is 1.15~1.20.

> ⚠️ `evolved` was made with **the contaminated block 3.5** (D-28). How much
> that gained is what `experiments/seed_ablation.py` separates — until the
> value re-run with the cleaned report exists, a reservation is attached to
> 1.0652.

### The other things confirmed

**A structure comes out without the table.** Architect A (`gpt-5.4`) produced
something at the `physics_seeded` level (1.1780 vs 1.1637) without seeing a
single sentence that came from the table.

**The model is decisive.** Under the same condition A, `mini` gives 1.4797 and
`gpt-5.4` gives 1.1780. Had only `mini` been seen, it would nearly have gone
to "condition A is impossible".

**The table buys variance, not the best score.** From A to B the best improves
by 0.029, the median by 0.302 and the worst by 0.598
(`artifacts/architect-gate.md`).

## 2. The slow regime is not a testbed

There is only **1.5%** of room. Static top-1 (1.0145) beats the vendor
(1.0439), `evolved` (1.1174) and `physics_seeded` (1.1137) all, and only GBDT
(1.0056) is better than that. **It is a regime where no rule-based method can
beat a fixed config.**

→ The "transfer failure" that came out of here may be a property of **the
testbed** rather than of the structure. §29.5 (b) cannot be judged before
there are ≥30 shapes in the slow regime.
(`artifacts/regime-diagnosis.md`, `artifacts/structure-transfer.md`)

---

## 3. The regime-specific terms are not the cause

Removing `is_two_stage` / `pipeline_warmup_frac` / `log_mainloop_iters` does
not recover it (1.1137→1.1137 and so on), and adding them does not break it
(the fast regime goes 1.1889→**1.1276**, an improvement if anything).
**Both directions denied the causation.**

And `physics_seeded` uses `pipeline_warmup_frac` too — the premise did not
hold in the first place. (`artifacts/structure-transfer.md`)

---

## 4. ★ The numbers above are shippable — it is not an oracle

The regime boundary is defined by **the SOL proxy metric**
(`core/splits.py::regime_of`). It agreed with the `t_best` criterion on all 61
shapes.

**But 100% is this table's luck.** The median of `t_best/SOL` is 1.140, so
there are two shapes inside the danger band `SOL ∈ [438, 500] us`, and their
`t_best` values are 497.7us and 496.6us — **3us** from the boundary.
(`experiments/proxy_dispatch.py`)

**And `evolved` needs no dispatch to begin with** — a single rule applies to
every shape. 1.0850 is a number independent of the regime decision.

---

## ★ The F1 line — when the LLM makes the features too (D-63, D-68, D-74)

> ⚠️ **This is provisional.** These results came out with only **stage 1
> (FeatureWriter) improved** in the pipeline. Stage 2 (Architect) and stage 3
> (Analyst + Optimizer) only had their prompts tidied and **their structure is
> unchanged.**
>
> **The place that makes performance is stage 3** and effectively nothing was
> done there. What is below is **a conclusion on the library axis**, not on the
> performance axis.

```
✅ a rule can be built out of the library     0 refusals, 12 runs finished
❌ it does not catch up with the 24 human ones  +0.0433, p=0.002 (over the
                                                seed range 0.0274)
```

| | F1's 21 | the 24 human ones |
|---|---:|---:|
| structure holdout median | 1.1195 | 1.0762 |
| quartiles | [1.1108, 1.1328] | [1.0706, 1.0834] |
| seed range σ | 0.0177 | 0.0124 |
| in-sample-to-holdout gap | +0.0290 | +0.0267 |

**The generalisation gaps are the same → it is not overfitting, the starting
point is lower.**

### F1-K — when the five public facts are given (D-74)

| | start | generated | **library** | new axes | **seed terms** | holdout median | σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1 | 0 | 21 | **21** | 7 | 8 | 1.1195 | 0.0177 |
| **F1-K** | 5 | 13 | **18** | **9** | **7** | 1.1288 | 0.0140 |

F1-K quartiles [1.1227, 1.1400], 967 calls over all the stages (inside the
pre-registration's ceiling of 990).
| the 24 human ones | 24 | 0 | **24** | — | 8 | **1.0762** | 0.0124 |

```
F1-K vs F1            +0.0093   p = 0.589   ★ indistinguishable
F1-K vs the 24 human  +0.0526   p = 0.002
```

**The new axes went up (7→9, the ratio 44%→75%) and the performance did
not.**

⚠️ **The library sizes differ** (18 / 21 / 24). "Did F1-K lose because it is
smaller" cannot be separated with this data — **that confound is written into
the conclusion sentence.**

```
the range is narrowed by evolution   stage 2 (Architect, 10 calls) 4.4x ->
                                     stage 3 (evolution) 1.42x
                                     evolution absorbs a good part of the luck
                                     of the initial choice
```

### The prompt rewrite was F1's biggest variable

```
(a) the old prompt + free       strict rediscovery 0/24
(b) the new prompt + free       strict rediscovery 6/22   ★
(c) the new prompt + per area   strict rediscovery 3/22
```

The skew was **the prompt's fault, not the generation strategy's** (D-63,
principle 21).

---

## 5. The open questions

**Who makes the features.** The LLM in the current pipeline only does the job
of picking 8 out of the 52 in `physical.py`. **Defining** `tail_waste` is the
part that is understanding physics and that stage is already finished. It may
be natural that `evolved` did not beat `physics_seeded` — the material is the
same but one side knows why that material is there. §11.4's Instrumenter is
**not implemented**.

**Can a structure be made without the table** (Architect condition A). For
transfer to hold, a structure has to come out on new hardware without the
table. Not implemented.

---

## ★ Stage 3 is still not fixed

```
stage 1  FeatureWriter        the prompt rewrite / fixed areas / the 5 known
                              ones — much was fixed
stage 2  Architect            barely touched. The rule examples are only
                              placeholders
stage 3  Analyst + Optimizer  only the prompts were tidied. The structure is
                              unchanged
```

### 303 of the Analyst's requests are being thrown away (`analyst-requests.md`)

```
of 1,655 hypotheses, 303 have needs_new_feature filled in (18.3%)
there is no code in loop.py that reads that field
```

The top four requests (245 of them) **can be expressed with fields that
already exist** — the absolute wave/CTA counts, the L2 reuse gain, the
pipeline family, the split-K gain. Only `warp_*`/`swizzle` (28) are
deliberately forbidden because they are `cfg.ext` (§4.3).

### The priorities

```
implement Critic     physics_seeded went 1.758 -> 1.172 with 3 rounds of
                     physics-error comments
                     ★ a per-term physical explanation makes the difference
                     from GBDT visible
§16.1 ablation       the Analyst is 8% of the calls and we do not know whether
                     it contributes
24 rounds            whether 12 is enough has not been looked at
the Analyst -> FeatureWriter path
```

## The verification path (done)

```
4-1  results turned into JSON      the .md/.json agreement test
4-2  KERNELRULE_UNSEAL             the final split is sealed (not opened yet)
4-3  the seed selection order      the evidence is recorded in chosen.json
4-4  the provenance of expected_range   no leak (21/21)
```

### What was deferred

```
a binary-shape area request (luna)   digging further into F1's limits
the 4090 transfer                    waiting for the hardware
implementing Critic                  a tool that **explains** the artefact.
                                     Not a regret tool
opening the final split              exactly once, after everything is done
```

### What not to do

```
⛔ do not change the model · the endpoint · the reasoning effort without
   instruction (D-52, principle 25)
⛔ do not report a difference under 0.03 as an "improvement" (D-53)
⛔ do not look at the structure-holdout score and fix the prompt (§12.3d)
⛔ do not cite a gpt-5.4 artefact as grounds for the conclusion
⛔ do not put 1.0650 and 1.0762 side by side — the procedures differ
   (principle 4)
⛔ do not slip another model in on "it is one call so it is cheap"
   (principle 25)
```
## 0. ★ A correction — the contrast was set up wrong twice

> ⚠️ **From here down is the correction history. These are not the current
> values.** Documentation rule 2 — a wrong value is not deleted.

### The first correction (the terms)

```
wrong:  the human structure transfers and the LLM structure does not
```

The name `rules/handwritten.py` created a "human vs LLM" contrast. That rule
too was written in one go by Claude Code reading the physics document. The
names were fixed — `physics_seeded` / `evolved` (`docs/glossary.md`).

### The second correction (the numbers) — this one is bigger

```
wrong:  a structure fitted to physics transfers and one fitted to the score
        does not
right:  ★ the evolved structure is the strongest. physics_seeded loses to the
        vendor
```

Even after the first correction the contrast itself was kept alive, but **the
number holding that contrast up belonged to a different rule.** The value
cited as `physics_seeded refit per regime = 1.0680` is, under the final
scoring procedure, `evolved`'s value (1.0652), and `physics_seeded` under the
same procedure is **1.1637**, losing to the vendor (1.0797).

**The common cause is a gate number cited without a reproduction procedure**
(§30.8b). The same mistake happened twice. So this document now uses **only
one table with a procedure attached** as the conclusion, and the remaining
descriptions are moved down into the correction history.

> ## ★ The performance numbers are verifiable (2026-08-26)
>
> The final rules and **the fitted weights** were committed —
> `docs/artifacts/rules/`.
>
> ```
> python3 experiments/verify_rules.py      # all 12 match, a few seconds
> ```
>
> `runs/` is in `.gitignore` so it is not in the repository. This command does
> not read it — it recomputes the structure holdout **from the committed rules
> alone** and checks against `index.json`. **The LLM runs cannot be reproduced
> but the scoring is deterministic** (§24.4b) — half of the performance claim
> is verified this way.

## The correction history (documentation rule 2 — a wrong value is not deleted)

> ⚠️ **From here down is the correction history. These are not the current
> values.** Documentation rule 2 — a wrong value is not deleted.

| when | what was wrong | what is right |
|---|---|---|
| ~2026-08-20 | static top-1 = 1.394 | 1.115 (it was an artefact of the cover) |
| ~2026-08-20 | the vendor = 1.088 / mapping 79% | 1.080 / 92.9% (the status filter) |
| 2026-08-21 | "the human structure vs the LLM structure" | both are LLM. Physics vs score |
| 2026-08-21 | "short/long shapes" | the fast/slow regime (by SOL, not by dimension) |
| 2026-08-21 | "the physics structure transfers and the score structure does not" | it is the opposite. `evolved` 1.0652 < `physics_seeded` 1.1637 |
| 2026-08-21 | "`physics_seeded` refit per regime 1.0680, 6 wins 0 losses against the vendor" | that value is `evolved`'s |
| 2026-08-21 | "the only method that passed the gate is the human structure + refit" | `physics_seeded` has never passed the gate |

**Values presented as grounds in the work instructions that are in no run of
this repository:**
`1.0325`, `1.1957`, `1.2151`, `1.2116`, `1.2130`, `55%→5.6%`,
`balanced 1.1387`. If a source is confirmed, the relevant section is
rewritten.

---
