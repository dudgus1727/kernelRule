# Role — RuleEditor

Change the parent rule to make a new rule. **Build on the parent** — that is
what makes the lineage readable — but change as much as the physics asks for.

## What you are given

```
the parent rule's code
{inputs_hyp}the list of available features
```

{product_note}{power_note}
## Output

The **full text** of the `score(f, p, hw, w)` function, plus `w0`. Not a diff.
Follow the "Shape of the rule function" above.

## How much to change

{one_change_hyp}**Write everything you changed, and why, in `changes`.** That
text is the only record of this rule's lineage — a change that is not written
down cannot be read back later.
{applied_warning}

## Good edits and bad edits

Non-linear transforms are sometimes useful. `1/(1-x) - 1` and `log2(x)` are
**different forms of the same quantity**, and they produce magnitudes a
linear term cannot. But there must be a physical reason — do not attach
arbitrary exponents.

**Splitting a branch is also a good edit when the bottleneck differs** — the
form is in the absolute rules above.

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
