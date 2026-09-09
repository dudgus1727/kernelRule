<!-- ★ Only the role that **edits a parent rule** receives this (RuleEditor).
     RuleWriter writes from scratch, so there is no "term to replace" and no
     "previous score" — giving it this file contradicts its own role file
     ("no scores, and no regret either"). -->

## How you are scored

{objective_block}

The few cases you are given are a **window for diagnosis**, not a list to
memorise.

## ★ What actually got rejected — write it this way and it is discarded

### 1. Adding terms by reusing a weight (the most common one)

```python
# ⛔ rejected: 19 terms built out of {parameters} weights
s = s + f.<nameA> * w[0]
s = s + f.<nameB> * w[0]      # w[0] reused
s = s + f.<nameC> * w[0]      # again
```

### 2. Using more than {parameters} parameters on one path

`len(w0)` **itself may exceed {parameters}** — if you split. What gets
rejected is ★ **one execution path** whose (literals + weights) exceeds
{parameters}.

```python
# ✅ accepted: <= {parameters} per path. len(w0) is larger than that
s = f.<nameA> * w[0]                       # common — belongs to every path
if p.<shape value> < 1:
    s = s + f.<nameB> * w[1] + ...         # the rest of this path's room
else:
    s = s + f.<nameC> * w[8] + ...         # the rest of that path's room
```

(The path cap itself is in the absolute rules above.)

### 3. Code that is too long

The cap is {ast_nodes} AST nodes. About {parameters} terms fits comfortably.
Going past it is a sign that you are piling up special cases.

### 4. Multiplying or adding a shape constant to the accumulated score

Re-read absolute rule 2. **It changes no ordering at all.**
