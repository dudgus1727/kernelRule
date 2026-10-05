<!-- D-193 — the RuleWriter example when the registry holds the time
     features (condition F4). The main term is the J1e/K1a form with the
     generic DRAM time; the corrections are left as placeholders on purpose:
     which effects to add, and in what form, is what the run measures. -->
## Rule example — ★ built **from the feature list above**

This is a shape example. **Do not submit it as is** — look at why each part
is where it is. Some names are placeholders.

★ **Start from one estimate of the kernel's time.** The time features above
are in one unit — a time over the ideal time of the whole problem — so they
can be combined into a single estimate before anything is weighted:

```python
def score(f, p, hw, w):
    # main term — one estimate of this config's time
    #   paths that run at the same time: the slower one decides  -> max
    #   paths that overlap only in part: combine them             -> sqrt(a^2 + b^2)
    sm = np.maximum(f.tm_crit_ratio, f.tm_l2_ratio)     # SM math vs its L2 feed
    t = np.sqrt(np.square(sm) + np.square(f.tm_dram_ratio))
    s = 10.0 * np.log2(t)          # fixed scale, not a weight
    # corrections — effects the estimate does not contain, one term each
    s = s + f.<correction axis> * w[0]
    s = s + f.<correction axis> * w[1]
    return s
```

What this is meant to convey:

```
why one estimate      max / sqrt pick the bottleneck per config. A weighted
                      sum of separate penalties keeps paying for a path that
                      is hidden behind a slower one
why log2 of a ratio   the estimate is a ratio to the ideal time — dimensionless,
                      so it transfers to another GPU; it is not "predicting
                      absolute time". log2 makes 10% slower the same step on
                      every shape
why no weight on it   it is the yardstick — each correction weight is measured
                      against it (1 score unit = 0.1 log2 of time)
the corrections       the estimate is only the skeleton. Which effects it
                      misses, and in what form, decides how good the rule is
```

★ **The main term needs no memory/compute branch** — max and sqrt already
pick the bottleneck per config, inside every shape. Branch on a shape-level
value only when a **correction's** physics differs between shapes, and name
that physics.

**Drop any term you cannot explain.** One line of physics per term is the bar.
