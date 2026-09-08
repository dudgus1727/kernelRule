# Role — partition the areas

Using the raw values you are given, partition the **physical areas that drive
performance** into {n_min}~{n_max} areas.

## Why partition

In the next step you will build a physical quantity per area. Without a
partition, the first few stay near wherever they landed — one axis actually
got split three times.

**The reason a human does not hand you the areas** is that doing so would be
handing over prior knowledge. It would amount to telling you "memory traffic
matters". **How you partition is itself the observation.**

## What you may use

{field_block}

## Rules

```
1. Areas must **not overlap**. Do not name the same physics twice
2. Each area must be **actually computable** from the raw values above
3. Describe it in one sentence — "what is wasted or constrained, by how much"
4. Area names: lower case + underscores
```

**Gaps are allowed.** If a physical effect cannot be expressed from the raw
values, do not create that area — forcing it produces a meaningless function
in the next step.

## Output

```
categories   [{{name, description}}] — {n_min}~{n_max} of them
notes        ★ If you left something out while partitioning, write it here.
             Things like "this physics matters but cannot be measured from
             the given values"
```
