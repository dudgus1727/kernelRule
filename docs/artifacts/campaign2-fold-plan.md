# The (N,K) group split (D-171 §1)

> **reproduce** `python3 -m experiments.nk_fold_plan` · 0 LLM calls
> fold 0 is the layer holdout (`11008`, both orientations) — the old `nk11008` split is one fold of this design

| table | shapes | (N,K) groups | fold0 | fold1 | fold2 | fold3 |
|---|--:|--:|--:|--:|--:|--:|
| a6000 | 65 | 19 | 17 | 16 | 16 | 16 |
| 5090 | 65 | 18 | 18 | 16 | 16 | 15 |
| 4090 | 63 | 18 | 16 | 16 | 16 | 15 |
| h100 | 63 | 18 | 16 | 16 | 16 | 15 |

## The four checks

| table | 1 coverage | 2 (N,K) twins in train | 3 min train minority | ⚠️ M twins |
|---|---|--:|--:|--:|
| a6000 | ok | 0 | 29% | 58 |
| 5090 | ok | 0 | 18% | 56 |
| 4090 | ok | 0 | 17% | 52 |
| h100 | ok | 0 | 12% | 52 |

## ⚠️ The balance check and what caps it

| table | whole population memory | best train minority | can any split reach 25%? |
|---|--:|--:|---|
| a6000 | 31% | 29% | yes |
| 5090 | 18% | 18% | **no — the table itself is below it** |
| 4090 | 21% | 17% | **no — the table itself is below it** |
| h100 | 14% | 12% | **no — the table itself is below it** |

★ On the 5090 / 4090 / H100 the memory-bound share of the whole 63~65 shapes is 14~21%, under the 25% `check_balance` threshold (D-144). **No split of those tables can satisfy it** — their ridge points are lower, so fewer shapes fall memory-bound. The warning is a property of the table, not of this design, and it applied equally to every earlier run on those tables.

⚠️ **M twins are not a defect.** M=1024 appears in nearly every group, so a validation shape almost always has an M sibling in training. This split measures **"a layer shape never seen"**, not "a shape never seen".

⚠️ A fold's own regime counts are in the JSON as a design check. They are **not** a result — one fold's memory side is 4~8 shapes. Regime readings come from the folds pooled.

### a6000

```
fold0  val 17  train 48  (4096,4096)
fold1  val 16  train 49  (512,512) (4096,128) (4096,256) (4096,4098) (4096,11008) (4098,4096)
fold2  val 16  train 49  (1024,1024) (4096,512) (4096,1024) (4096,4100) (4100,4096) (11008,4096)
fold3  val 16  train 49  (2048,2048) (4096,2048) (4096,8192) (4096,16384) (8192,8192) (12288,4096)
```

### 5090

```
fold0  val 18  train 47  (4096,4096)
fold1  val 16  train 49  (2048,2048) (4096,256) (4096,2048) (4096,8192) (4096,11008) (8192,8192)
fold2  val 16  train 49  (4096,128) (4096,1024) (4096,4098) (4098,4096) (11008,4096) (16384,16384)
fold3  val 15  train 50  (4096,512) (4096,4100) (4096,16384) (4100,4096) (12288,4096)
```

### 4090

```
fold0  val 16  train 47  (4096,4096)
fold1  val 16  train 47  (2048,2048) (4096,256) (4096,2048) (4096,8192) (4096,11008) (8192,8192)
fold2  val 16  train 47  (4096,128) (4096,1024) (4096,4098) (4098,4096) (11008,4096) (16384,16384)
fold3  val 15  train 48  (4096,512) (4096,4100) (4096,16384) (4100,4096) (12288,4096)
```

### h100

```
fold0  val 16  train 47  (4096,4096)
fold1  val 16  train 47  (2048,2048) (4096,256) (4096,2048) (4096,8192) (4096,11008) (8192,8192)
fold2  val 16  train 47  (4096,128) (4096,1024) (4096,4098) (4098,4096) (11008,4096) (16384,16384)
fold3  val 15  train 48  (4096,512) (4096,4100) (4096,16384) (4100,4096) (12288,4096)
```
