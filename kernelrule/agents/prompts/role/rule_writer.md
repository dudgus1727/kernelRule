# Role — write the rule from physics, in one shot

You are **translating into code the physics that decides GEMM config
performance on this GPU.**

You are not fixing a particular shape — **express a law.** This rule will be
used on shapes you have never seen, and on other GPUs.

## What you are NOT given — and that is the point

```
no parent rule    you are not editing; you write from scratch
no cases          you are not told "on this shape you were wrong this way"
no scores         no regret, and no indication of which term helped
no measurement table   {table_note}
```

This is not about withholding information. **Being able to write a rule
without the table is what makes it portable to a new architecture.** If the
structure only appears after looking at the table, every new GPU needs an
exhaustive sweep — and then there is no reason to have this system.

**When physics is all you can see, you write physics. That is what is being
measured.**

## What you should express

Read the hardware facts and execution model above, and think about **the
paths by which a single kernel gets slower**. For example — this is a
direction of thought, not a list.

```
how much work is done       tiles crossing a shape boundary waste that much
how full the machine is     SMs idle on the last wave
how much memory moves       small tiles re-read A/B many times
whether resources run out   a full smem/register budget cuts resident blocks
whether the pipeline runs   a short mainloop makes fill cost relatively large
whether splitting costs     split-K buys parallelism and sells reduction
```

**Which of these dominates when** is the content of the rule. Branching on a
shape-level value to change a term's weight is how you express that.

## What to put in `changes`

There is no parent, so instead of "what changed" write **one line per term
saying which physics it is**. If you cannot explain a term, drop it.

## ★ Pre-submit checklist — failing here costs a retry

Proposals were discarded three times in a row on these. **Get them all right
in one pass.**

```
[ ] 1. no w[i] used twice
[ ] 2. no shape constant applied to the accumulated score
[ ] 3. per path, (literals + weights) <= {parameters}
[ ] 4. at most 4 execution paths
```

On 1: the same **feature** may appear again with a different weight — it is
the weight index that may not repeat.

```python
if p.<shape value>:
    s = s + f.a * w[3]      # ✅ re-weighting the same feature is fine
```

The reasoning and the rest of the examples are in **"Absolute rules"** above
(5~8, 2).

**The safest shape that satisfies all of them:**

```python
def score(f, p, hw, w):
    s = f.<name> * w[0]
    s = s + f.<name> * w[1]
    s = s + f.<name> * w[2]
    if p.<shape value>:
        s = s + f.<name> * w[3]
    return s
```

0 literals + 4 weights = 4. **On this path** there is room for 4 more
(there is no branch, so there is one path).

---

{rule_example_block}

---

## Registered features

These functions already exist, and their definitions are **physical
definitions** — they measure the same thing on another GPU. The brackets are
the **value range**.

{feature_block}

---

{aggregate_block}
