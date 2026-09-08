# ★ 12 rounds is **not enough** — but the shortfall is smaller than the seed spread

> **Reproduce**: `python3 experiments/rounds_curve.py` (about a minute,
> **0 LLM calls**)
> **Pre-registration**: [rounds-prereg.md](rounds-prereg.md)
> **Raw data**: `docs/artifacts/rounds-curve.json`
> The round number is the file's `round` field = **from 0** (r0..r11 is 12
> rounds)

> ⚠️ 2026-09-08 (D-146): translated into English. The numbers and the verdicts
> are unchanged; the Korean original is at commit `ee53b4d`.

## 1. First — early stopping **never once fired**

Running 12 rounds with `patience=10` leaves only two decision windows, and it
does not stop if even one new cell appears. Measured:

```
where patience=10 would have stopped F3rw-p8's 6 seeds   all "-" (never fired)
had it been patience=3                                   r3 · r6 · r10 · r11 ·
                                                         none · none
had it been patience=4                                   r4 · none for the rest
```

**"It ran all 12" was not "it converged" but "it never had the chance to
stop".**

## 2. F3rw-p8 (formerly arch24), 6 seeds — the curves

The noise threshold is **0.0047 for all 6 seeds** (the value `is_significant`
uses).

| seed | last improvement | ★ last **significant** improvement | last new cell | the curve (best_val_regret) |
|---|---|---|---|---|
| s0 | r10 | **r10** | r9 | 1.1163 1.0973 1.0968 1.0968 1.0914 1.0914 1.0968 1.0968 1.0968 1.0968 **1.0908** 1.0908 |
| s1 | r10 | r7 | r8 | 1.1122 1.1288 1.1174 1.0768 1.0768 1.0768 1.0900 **1.0672** 1.0694 1.0694 1.0692 1.0692 |
| s2 | r5 | r5 | r9 | 1.0987 ×4 **1.0898** ×5 1.0898 1.1000 |
| s3 | r7 | r7 | **r11** | 1.1122 1.1033 1.0593 1.0593 1.0972 1.0914 1.0914 **1.0737** ×4 |
| s4 | r11 | **r11** | r7 | 1.1238 1.1104 1.1013 1.0901 1.0960 1.1050 1.1050 1.0864 ×4 **1.0809** |
| s5 | r6 | r6 | r8 | 1.1151 1.1151 1.1027 1.0915 1.0915 1.0683 **1.0593** 1.0630 ×5 |

```
the last significant improvement round   r10 · r7 · r5 · r7 · r11 · r6
★ seeds with a significant improvement in the last 3 rounds (r9·r10·r11)
                                         2/6 (s0, s4)
```

## 3. ★ The verdict — (b) 12 is not enough

It is the second of the three in pre-registration §2. **It is not a line made
after looking at the curves.**

```
(b) there are seeds that improve significantly in the last 3 rounds
    -> 12 is not enough
    -> round 24 has to be measured at n=6 (about 6,000 calls / 15 hours)
```

### ⚠️ But the **size** of the shortfall has to be read alongside

```
s0's improvement at r10   0.0060
s4's improvement at r11   0.0055
the noise threshold       0.0047   <- larger than this (hence "significant")
the seed spread σ         0.0124   <- ★ smaller than this
the decision line delta   0.0516   <- ★ one ninth of it
```

**"Significant" is measured against the noise floor, and "measurable" against
the seed spread.** The amount more rounds appear to gain (~0.006) is **smaller
than changing one seed.** Running 24 rounds at 6 seeds could still move the
median only within the seed spread.

★ **The verdict stays (b)** — the line is not changed after seeing the
result. But it is written down alongside that **the expected gain against the
cost (6,000 calls / 15 hours) is below σ.** Whether to run it is decided with
those two lines side by side.

## 4. The other conditions are the same

| condition | last significant improvement | significant improvement in the last 3 rounds |
|---|---|---|
| F3rw-p8 | r10 r7 r5 r7 r11 r6 | **2/6** |
| F1rw-p8 | r9 r4 r5 r9 r8 r7 | 2/6 |
| F2rw-p8 | r8 r8 r5 r5 r9 r8 | 1/6 |

⚠️ The threshold for F1/F2 is **an approximation.** Those runs' rules refer
to features made inside the loop, so they cannot be rescored with the base
registry. All 6 F3 seeds gave the same threshold (0.0047) — the threshold is
set by the holdout shapes, not by the rule — so that value was borrowed and
marked `(approx)` in the table. **The verdict was made on F3 alone.**

## 5. New cells keep appearing to the end

```
the last new-cell round  F3: r9 r8 r9 r11 r7 r8   F1: r11 r7 r10 r9 r11 r8
```

`should_stop`'s second condition (`new_cell_recent`) is **almost always
true**. So shrinking patience alone does not stop it either — the cell axes
are quartiles, so the cells keep being re-sorted.

★ Changing patience is a **condition change** and needs its own
pre-registration. It was not changed here.
