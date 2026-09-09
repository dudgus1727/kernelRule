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
# ⛔ rejected: 19 terms built out of 8 weights
s = s + f.<nameA> * w[0]
s = s + f.<nameB> * w[0]      # w[0] reused
s = s + f.<nameC> * w[0]      # again
```

### 2. Leaving a hole in `w0`

```python
# ⛔ rejected: w[0] and w[8] used with len(w0) = 9 — the six in between
#    are fitted anyway, so they are free parameters
```

`len(w0)` must equal the largest index you use + 1.

### 3. Code that is too long

The cap is {ast_nodes} AST nodes. Going past it is a sign that you are piling
up special cases, not adding physics.

### 4. Multiplying or adding a shape constant to the accumulated score

Re-read absolute rule 2. **It changes no ordering at all.**
