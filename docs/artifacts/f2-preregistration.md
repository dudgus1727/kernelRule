# The F1-K pre-registration — ★ nailed down **before** the run

> **Status**: registered (2026-08-27). **Written with 0 LLM calls made**
> **Reproduce**: `python3 experiments/f1_pipeline.py F1-K --categorize --per-category 3 --tag k1`
> **Model**: `gpt-5.6-luna` / responses / medium — pinned by `DEFAULT_MODEL` (D-45, principle 25)
> **Table**: `datasets/rtx-a6000-sm_86-c63710df` (dev, its numbers must not be reported externally)

> ⚠️ 2026-09-08 (D-146): this document was translated into English. It is a
> frozen pre-registration, so **nothing was deleted** — the Korean original
> is at commit `ee53b4d`
> (`git show ee53b4d:docs/artifacts/f2-preregistration.md`). The numbers, the
> criteria and the structure are unchanged; only the language is.

**Setting the criteria after seeing the results is contamination** (D-50). So
they are written here first. The values in this document have to be **the
same** as `F1K_PREREG` in `experiments/f1_pipeline.py`, and
`tests/test_f1k_prereg.py` pins that down.

## ★ Why now, and not "just before the run"

```
writing now         it is written from the plumbing result alone. The run
                    result is unknown
writing just before ★ while building the plumbing a feel for "this ought to
the run             work" has formed. That feel seeps into the criteria
```

**Just before the run is the more dangerous moment.** Now that the plumbing
check (`--dry-run`) is done, and with the LLM not called even once, it is
nailed down.

---

## 1. The design

```
condition        F1-K
start library    the 5 public facts (known5.py — the tidied form of
                 physical.py)
areas            a fixed 7 (prompts/areas.md). --recategorize is not used
generation       3~4 per area = 21~28 proposals
after that       Architect 10 calls -> a seed -> evolution, 6 seeds x 12
                 rounds
model            pinned by DEFAULT_MODEL
```

**The same as F1:** it does not see the table / only `p`·`hw`·`cfg` / it
passes the static checks and the §8.3 validation as they are / the final
scoring procedure / the structural holdout nk11008.

**Different from F1 — there are two, and they are not separated:**

```
1. the start library goes 0 -> 5
2. the example goes from an unrelated domain -> a real feature (code and all)
```

**Both are part of "public knowledge is given"**, so they are treated as one
variable. It is an exception to D-31 (do not change two variables at once)
and the reason is written down here. **"Which of the two did it" cannot be
separated by this experiment.**

## 2. The purpose and what is expected

```
purpose   "given known axes, does it make more new ones. And does the
          library get better"

expected  the number of new axes is larger than F1's (no budget is spent on
          rediscovery)
          ★ whether the evolution performance catches up with F3 is not
            known. Not catching up is not a failure
          ★ it may also be better than F1 — that too is an open question,
            not an expectation
```

**Because of the seed spread (σ 0.0274), a difference of the 0.02 class
between conditions cannot be told apart** (D-53). If the performance is
similar, **"indistinguishable" is the honest description**.

## 3. The main metrics

```
the number of new axes and their correlation   against the human 24. Strict
                                               (sp and pe both >0.95) /
                                               monotone
the structural holdout after evolution         **the same final scoring
                                               procedure** as F1 (the median
                                               of 6 runs per shape)
```

## 4. Observations (not performance)

```
the acceptance/refusal distribution per area
  ★ does the "memory traffic" area's output overlap the roofline axis
    -> the area mapping folded the two axes together (§30.18); it is checked
       whether they really are the same
physics_coverage       how much of the human 24's physics it covers
among the axes made, the ones a human did not make — are they physically
sound (qualitative)
does the rationale carry a source — does the example's habit get through
how many are judged to be shape_level
```

## 5. ★ What is **not** a criterion

```
the number of rediscoveries   five were given, so of course there is less to
                              rediscover. The condition is not evaluated by
                              this
```

## 6. What to do on failure

```
one area refused 3 times in a row   skip it and record it in
                                    skipped_categories. Continue
acceptance under half               stop and ★ look at the refusal-reason
                                    distribution first (below)
all 10 Architect calls refused      stop and report (the same as §30.9.6)
3 runs in a row with an empty        stop
archive during evolution
```

### Why "half" — it is not a quality judgement but waste prevention

```
80%   normal
50%   odd, but a result still comes out
20%   ★ the checker or the prompt is broken. 4 out of 21 proposals gives no
      library
```

**Half is "the minimum for the experiment to hold", not a level tightened
against F1's measurement (80%).** F1-K was given five, so **duplicate
refusals can rise and that is normal behaviour**. Setting it at 70% would
have stopped a normal run.

### ★ What to do after stopping — suspect in principle-1 order

```
look at the refusal-reason distribution first
  mostly duplicates          normal. Five were given, so it is expected.
                             Judge whether to continue
mostly §8.3 validation fails  a checker or field problem (the D-37 / D-38
                             family)
mostly schema fails          a prompt problem
```

**The order is infrastructure -> checker -> subject** (principle 8). When F1
gave 3/20 acceptances, the cause was **two validator defects** — it was not
the LLM.

⚠️ This section **does not change the criteria; it states what to do after
stopping**. The threshold (half) is unchanged. **It was added before the LLM
was called.**

## 7. What is not done

```
the remaining 19 are not put in — that is the F3 condition
--recategorize is not used — the fixed seven are used
the F1 results (the 21-feature library, 12 runs) are not deleted — they are
what it is compared against
the model is not changed (D-45, principle 25)
the prompt is not fixed after seeing the results (§12.3d)
```

## 8. The cost cap

Estimated from the F1 measurement.

```
stage 1   ~24 calls  (7 areas x 3, plus room for refusals)   F1 measured
                                                             20~22 calls /
                                                             40~45 min
stage 2   ~30 calls  (Architect 10 + retries)                F1 measured 11
                                                             calls
stage 3   ~936 calls (6 seeds x 12 rounds)                   F1 measured 936
                                                             calls
total     ~990 calls.  3~4 hours of wall clock
```
