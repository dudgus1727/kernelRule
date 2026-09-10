<!-- ★ Hard constraints shared by the roles that write the rule **function**
     (RuleEditor + RuleWriter). Only what is true for both belongs here.
     If it applies to one of them, push it down into `_rules_edit.md`
     (RuleEditor) or into the role file. -->

## Shape of the rule function

```python
def score(f, p, hw, w):
    s = f.<name> * w[0]
    s = s + f.<name> * w[1]
    return s
```

Lower is better. All features are normalised so that **larger is worse**, so
weights are usually positive.

## Absolute rules

```
1. Do not branch on config-level features (`f.*`).
   `f.*` is an **array** over all candidates, so `if f.<name> < 1:` raises.
   This constraint is deliberate — conditional specialisation, repeated,
   becomes a lookup table and does not generalise.

   `np.where(...)` is the only way to write a config-level condition, and it
   is **one expression, not a branch**: both sides are computed and the term
   keeps one weight.

       s = s + np.where(f.<name> < 1, w[3], 0.0)   ⛔ a constant per side —
                                                      that is specialisation
       s = s + np.where(p.<shape value> < 1,
                        f.<A>, f.<B>) * w[3]       ✅ which physics to read
       s = s + f.<other name> * w[3]               ✅ a continuous term is
                                                      better still

2. You may branch on shape-level values (`p.*`). They are scalars.
   But **multiplying or adding a shape constant to the whole accumulated
   score does nothing.** The rule sorts within each shape independently, so
   a shape constant cancels out.

       if p.<shape value>:
           s = s * w[2]                    # ⛔ changes no ordering at all
           s = s + f.<name> * w[2]         # ✅ changes a term's weight

   **`p.M` / `p.N` / `p.K` may be used with inequalities.**

       if p.M == 4096:                     # ⛔ memorises one point. Rejected
       if p.M < 128:                       # ✅ splits a range. Generalises

   Only equality (`==`, `!=`) is rejected. Splitting a range states a
   physical fact ("small M means a skinny shape"), so it is allowed.

3. Access weights **by constant index only**: `w[0]`, `w[1]`, ...

4. No `import`. `np` is already provided, and only these functions:
   where clip minimum maximum log log2 sqrt abs exp power sign
   floor ceil round isfinite nan_to_num square fmin fmax
   `np.random` is forbidden — the rule must be deterministic.

5. **Split when the physics differs.** Memory-bound and compute-bound
   shapes have different bottlenecks, so the same term does not act in the
   same direction. When you split, **each branch gets its own weights.**

       s = f.<name> * w[0]                   # common — belongs to every path
       if p.<shape value> < 1:
           s = s + f.<A> * w[1] + f.<A2> * w[2] + ...
       else:
           s = s + f.<B> * w[3] + f.<B2> * w[4] + ...

   Physics must be the reason for a split — not the shape of the code.
   ⚠️ Terms **outside** the `if/else` belong to every path. `np.where` is
   not a branch — both sides are computed, so it is one path.
   ⚠️ There may be **at most 4 execution paths**.
   Two levels of nesting · `if/elif/elif/else` · two sequential `if`s —
   all of those are 4 paths.

6. ★ **`len(w0)` must equal the largest index you use + 1, exactly.**
   An index you never use is still fitted, so it would be a free parameter.

       s = f.<A> * w[0] + f.<B> * w[8]      ⛔ 1..7 are never used
       s = f.<A> * w[0] + f.<B> * w[1]      ✅

   The same `w[i]` **may** appear in several terms. That is a claim — "these
   carry the same weight" — and it costs one coefficient instead of two.

       s = f.<A> * w[0] + f.<B> * w[0]      ✅ the same weight, deliberately
```

## Match the magnitudes

The brackets in the feature list are the **value range**. If you add terms of
different magnitude with the same weight, **the widest term decides the whole
ordering** — a `[0, 300]` term added to a `[0, 1]` term with equal weights
makes the latter invisible.

```python
s = np.log2(f.<wide range name>) * w[0]   # compress to match magnitudes
s = s + f.<narrow range name> * w[1]      # already [0,1], leave it
```

Or give `w0` on the inverse scale of the range — for a `[0, 300]` term,
`w0 ≈ 0.003`. **Do one of the two.**

**Do not work around 6.** Reusing one weight across several terms hides
terms from the count and gives the fitter one coefficient for two different
physical quantities.

**Write the terms you can explain.** One line of physics per term is the
bar.{product_block}{power_block}

## You do not fit the weights

`w` is fitted by a numerical optimiser **before scoring**. (Which optimiser
depends on the objective and may change — you do not need to know.)

`p.n_candidates` is **how many configs were measured** for that shape. It
is enumeration information, not performance — it says nothing about which
config is fast. Splitting shapes by candidate count is not learning physics,
it is **learning the experimental design**.

**Focus on structure** — which physical quantities to use, how to combine
them, where to split regimes. "Is 2.0 right, or 2.7?" is not your job.

Still, do not give a careless `w0`. The objective is a step function and the
optimiser can get stuck on a plateau near the starting point. Give a
**starting point that reflects the physical magnitude of each term**.

## Keep the score dimensionless

Do not predict absolute time. If the hardware changes, all of it has to be
relearned. Ratios, logs and normalised quantities transfer.
