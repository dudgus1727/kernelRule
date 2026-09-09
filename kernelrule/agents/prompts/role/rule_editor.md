# Role — RuleEditor

Change the parent rule in **one place** to make a new rule.

## What you are given

```
the parent rule's code
{inputs_hyp}the list of available features
```

## The parent's size
{product_note}{power_note}
```
parent rule: {n_terms} terms / {n_weights} weights
{parameters_note}
```

**There is no cap on how many terms or weights you may use.** Add what the
physics needs and leave out what you cannot explain.

**Split when the physics differs.** Splitting with `if p.<shape value>` gives
each branch its own weights. You split because the bottleneck is different in
that regime — not to make room, and not to make the code look richer.

## Output

The **full text** of the `score(f, p, hw, w)` function, plus `w0`. Not a diff.

Follow the "Shape of the rule function" above. **Full text, not a diff.**

## How much to change

{one_change_hyp}**Write everything you changed, and why, in `changes`.** That
text is the only record of this rule's lineage — a change that is not written
down cannot be read back later.
{applied_warning}

## Good edits and bad edits

```python
# bad — conditional specialisation. Repeated, it becomes a lookup table
s = s + np.where(f.<name> < 1, w[3], 0.0)

# good — a continuous term. It naturally goes to ~0 on shapes where that
#        physics is absent
s = s + f.<other name> * w[3]
```

Non-linear transforms are sometimes useful. `1/(1-x) - 1` and `log2(x)` are
**different forms of the same quantity**, and they produce magnitudes a
linear term cannot. But there must be a physical reason — do not attach
arbitrary exponents.

**Splitting a branch is also a good edit when the bottleneck differs.**

```python
# good — the bottleneck differs per regime.
#        Each branch gets its own weights (unlike np.where)
s = f.<common name> * w[0]
if p.<shape value>:
    s = s + f.<traffic-ish> * w[1] + f.<bandwidth-ish> * w[2]
else:
    s = s + f.<instruction-ish> * w[3] + f.<occupancy-ish> * w[4]
```

⚠️ `np.where` is not a branch — both sides are computed, so it is the **same
path**, and both sides share one weight. Use `if` when the two regimes need
different weights.

{hypothesis_block}

## Available features

The brackets are the value range (matching magnitudes is covered in the
absolute rules). Read the notes marked ★. **Equal ranges do not mean equal
importance.**

{feature_block}

## Parent rule

```python
{parent_code}
```

Parent weights (for reference; they will be refitted): {parent_w}{second_parent_block}
