## Rule example — ★ built **from the feature list above**

This is a shape example. **Do not submit it as is** — look at why each term
is where it is. Some names are placeholders.

```python
def score(f, p, hw, w):
    # traffic — unbounded, so compress with a log
    s = np.log2(f.<wide-range traffic axis>) * w[0]
    # wave quantisation — already [0,1], leave it
    s = s + f.tail_waste * w[1]
    # spill — binary. Switching it on changes the magnitude, so start large
    s = s + f.has_spill * w[2]
    # resource pressure
    s = s + f.occupancy_deficit * w[3]
    # ★ look at **different physics** per regime (selection, not re-weighting)
    s = s + np.where(p.roofline_ratio < 1,
                     f.<memory-side axis>, f.<compute-side axis>) * w[4]
    return s
```

What this is meant to convey:

```
when to use log compression   for wide-range terms (edge_waste is [0,300])
initial weight of a binary term  large, since switching it on changes magnitude
★ re-weighting vs selection
    if p.x:  s += f.a * w[i]              see the same physics more/less
    np.where(p.x < 1, f.a, f.b) * w[i]    ★ see different physics
one comment per term          the shape of what goes into `changes`
```

**Drop any term you cannot explain.** Each of the five terms above is
explained in one line — that is the bar.

## ★ The split form — each branch has its own weights

When the bottleneck differs, split with `if`. **This differs from
`np.where`** — `np.where` computes both sides, so it is one path and gains no
room, while `if` splits the path and **each branch spends its own
parameters.**

```python
def score(f, p, hw, w):
    # common — belongs to every path
    s = np.log2(f.<wide-range traffic axis>) * w[0]
    if p.<regime shape value>:
        # one regime: bandwidth is the bottleneck
        s = s + f.<memory-side axis> * w[1]
        s = s + f.tail_waste * w[2]
    else:
        # the other regime: instructions and occupancy are the bottleneck
        s = s + f.<compute-side axis> * w[8]
        s = s + f.occupancy_deficit * w[9]
    return s
```

```
len(w0) = 10 but only 3 per path — both are within the cap
★ Do not split to gain room. Split only when the physics differs
⚠️ There may be at most 4 execution paths
```
