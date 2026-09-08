## Rule example — ★ from a different domain

Below is a scoring function from a **domain unrelated to this problem**
(choosing a cache replacement policy). Look only at **why each term is there**
and at **the shape**. Do not carry these concepts over to GEMM.

```python
def score(f, p, hw, w):
    # re-reference distance — unbounded, so compress with a log
    s = np.log2(f.<re-reference distance>) * w[0]
    # pollution ratio — already [0,1], leave it
    s = s + f.<pollution ratio> * w[1]
    # metadata overflow — binary. Switching it on changes magnitude, start large
    s = s + f.<metadata overflow> * w[2]
    # ★ look at **different physics** per regime (selection, not re-weighting)
    s = s + np.where(p.<workload ratio> < 1,
                     f.<latency-sensitive axis>, f.<bandwidth axis>) * w[3]
    return s
```

What this is meant to convey:

```
when to use log compression   for wide-range terms
initial weight of a binary term  large, since switching it on changes magnitude
★ re-weighting vs selection
    if p.x:  s += f.a * w[i]              see the same physics more/less
    np.where(p.x < 1, f.a, f.b) * w[i]    ★ see different physics
one comment per term          the shape of what goes into `changes`
```

**Replace the placeholders with real names.** Submitting the code above as is
will be rejected.
