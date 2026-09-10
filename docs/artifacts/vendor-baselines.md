# The vendor baseline on **four** tables

```
python3 experiments/vendor_extract.py <bundle> --env-hash <h> --count 8 --out <file>
python3 -m experiments.vendor_baselines     # -> docs/artifacts/vendor-baselines.json
```

**0 LLM calls.** `nvMatmulHeuristics` computes from a preset without a GPU.

## What was extracted

| file | gpu | preset | shapes | lib |
|---|---|---|--:|---|
| `vendor-a6000-c63710df.json` | RTX A6000 | `RTX_A6000` | 66 | 0.1.0.27 |
| `vendor-5090-5bb6f403.json` | RTX 5090 | `RTX_5090` | 66 | 0.1.0.27 |
| ★ `vendor-4090-ad95d455.json` | RTX 4090 | `RTX_4090` | 64 | 0.1.0.27 |
| ★ `vendor-h100-63684546.json` | H100 NVL | `H100_NVL` | 64 | 0.1.0.27 |

Conditions matched to the existing two: target CUTLASS · layout
TN_ROW_MAJOR · precision HSS · count 8 · lib 0.1.0.27.

**The watch after extraction, on both new files:**

```
cluster != (1,1)     0        the 2.x kernel space
instr   != (16,8,16) 0        f16 HMMA
parse failures       0
```

### ⚠️ The H100 preset was wrong in the table

`GPU_PRESETS` mapped every `"h100"` to **`H100_SXM`**. Our bundle is an
**H100 NVL** — a different card. `H100_NVL` exists in the library, so the
map now lists the variants (`h100 nvl` · `h100 pcie`) before the bare
`h100`. **Nothing was substituted** (D-158).

## ★ The four-table baseline

Whole table:

| gpu | shapes | vendor | vendor (strict) | no rec | exact match | static top-1 | random |
|---|--:|--:|--:|--:|--:|--:|--:|
| a6000 | 66 | 1.0830 | 1.1143 | 0 | 92.9% | 1.1151 | 1.6455 |
| 5090 | 66 | 1.1501 | 1.1708 | 0 | 96.4% | 1.0807 | 1.3243 |
| 4090 | 64 | 1.0787 | 1.0953 | 0 | 95.7% | 1.0577 | 1.3867 |
| h100 | 64 | 1.1210 | 1.1344 | 0 | 95.1% | 1.2009 | 2.1582 |

The transfer holdout of each table (`nk11008` val, 20 shapes):

| gpu | vendor | vendor (strict) | static top-1 |
|---|--:|--:|--:|
| a6000 | **1.0737** | 1.0737 | 1.0636 |
| 5090 | 1.1158 | 1.1158 | 1.0452 |
| 4090 | 1.0647 | 1.0647 | 1.0403 |
| h100 | 1.1522 | 1.1522 | 1.1921 |

★ The A6000 holdout comes out at **1.0737** — the canonical recorded value
(D-69 · D-140). That is the check that this procedure is the same one.

**No shape is missing a recommendation on any table** (`no rec` 0), so no
geomean here is taken over a reduced set.

## ★ The 5090 recheck — it was (나), a bug in the evaluation

D-157 reported the 5090 vendor as **1.2562**, an order away from the
A6000's 1.0737. The cause is in our code, not in the vendor.

```
vendor_order_fn() returns an **order** — an array of candidate indices
D-157 handed it to `evaluate_scores()`, which expects **scores**
-> the index array was read as a score vector. The number is meaningless
```

With `evaluate()` (the order-based one, which `vendor_compare.py` has always
used) the same holdout gives **1.1158**. The same bug made the A6000 read
2.1562 instead of 1.0830 in the first pass of this experiment, which is how
it was caught — the recorded 1.0737 did not reproduce.

On the 53 shapes both tables share:

```
5090   geomean 1.1580   median 1.1471   max 1.4512   over 1.5: ★ 0/53
a6000  geomean 1.0828
worst  1024x4096x4097  5090 1.4512 / a6000 1.1875
       1x11008x4096    5090 1.3871 / a6000 1.1304
```

**The 5090 vendor is worse than the A6000's, but by 0.075 on the geomean,
not by an order.** No shape is over 1.5, so nothing is dragging the geomean.

## Where each table's vendor stands

```
h100   ★ the vendor **beats static top-1** (1.1522 < 1.1921) — the only
       table where it does. It is also the table where static top-1 is
       worst
a6000  vendor 1.0737 · static 1.0636 — static is slightly ahead
5090   vendor 1.1158 · static 1.0452
4090   vendor 1.0647 · static 1.0403
```

⚠️ `strict` and `nearest` agree exactly on all four holdouts, so the mapping
choice does not carry these numbers.
