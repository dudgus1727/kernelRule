# The regime split axis — **whichever axis it splits on, or none, it is indistinguishable**

> **Reproduce**: `python3 experiments/regime_axis.py` (**0** LLM calls · 0 GPU)
> **Pre-registration**: [regime-axis-prereg.md](regime-axis-prereg.md)
> **Raw data**: `regime-axis.json`
> The rules: the **r11** artefacts of the six representative runs (D-140).
> The structures were not touched.

> ⚠️ 2026-09-08 (D-146): translated into English. The numbers and the verdicts
> are unchanged; the Korean original is at commit `ee53b4d`.

## 0. ★ Start with the reproduction check

`canonical_score` hardcodes the regime names and the axis, so that procedure
was reimplemented in the experiment script. So first:

```
arm ① (SOL 0.5), the median of 6 runs   1.0886
the known representative value          1.0886   ★ difference +0.0000
-> the reimplementation is right. The other arms may be read
```

## 1. The five arms (the holdout of 20 shapes, 6 runs)

| arm | median | range | σ | holdout per regime | vs ① | verdict |
|---|---:|---|---:|---|---:|---|
| **① SOL 0.5** | **1.0886** | 1.0677~1.1662 | 0.0364 | short 12 / long 8 | — | (the baseline) |
| ② roofline | 1.1074 | 1.0636~1.1884 | 0.0429 | mem 8 / comp 12 | +0.0188 | indistinguishable |
| ③ no split | 1.1054 | 1.0855~1.1884 | 0.0374 | all 20 | +0.0167 | indistinguishable |
| ①' SOL 0.25 | 1.0948 | 1.0744~1.1614 | 0.0306 | short 10 / long 10 | +0.0061 | indistinguishable |
| ①'' SOL 1.0 | 1.0888 | 1.0614~1.1734 | 0.0396 | short 14 / long 6 | +0.0002 | indistinguishable |

⚠️ `①'' SOL 1.0` has **7 long shapes in training, under `MIN_PER_REGIME`
(8)** — that regime's weights are hard to trust (written down in advance in
the pre-registration).

**All four are inside the decision line of 0.0516. They do not touch the
buffer band (0.0589) either.**

### 1-1. The paired comparison — ⚠️ not in the pre-registration

The five arms use **the same six rules**, so they can be paired. The
pre-registration set only the unpaired decision line (0.0516), so **this is
an observation.**

```
② - ①    mean +0.0090  median +0.0070   runs worse than ① 3/6   p = 1.000
③ - ①    mean +0.0139  median +0.0162   4/6                     p = 0.688
①'- ①    mean -0.0021  median +0.0011   3/6                     p = 1.000
①''- ①   mean -0.0039  median -0.0032   2/6                     p = 1.000
③ - ②    mean +0.0049  median +0.0000   2/6                     p = 1.000
```

**Even paired, nothing separates** (it does not come close to the paired
decision line of 0.0305 either).

## 2. Boundary sensitivity — **0.5 is not a knife edge**

```
SOL 0.25   1.0948   (+0.0061)
SOL 0.5    1.0886
SOL 1.0    1.0888   (+0.0002)
★ moving the boundary by a factor of 4 keeps it inside 0.006
```

**The answer to "why was 0.5 chosen" became "it is the same whether it is
chosen or not".** That is not a defence — **the question became
meaningless.**

## 3. ★ Which side collapses when it is not split

The per-regime holdout regret seen through the roofline split (the median of
6 runs):

| arm | mem (8 shapes) | comp (12 shapes) |
|---|---:|---:|
| ① SOL 0.5 | 1.0596 | **1.1146** |
| ② roofline | 1.0590 | 1.1433 |
| ③ no split | 1.0562 | **1.1558** |

```
the mem side    1.0562 ~ 1.0596 — ★ the three arms are effectively the same
the comp side   1.1146 -> 1.1558  ★ not splitting is 0.041 worse (inside the
                                    decision line, but there is a direction)
```

★ **The only place splitting earns its keep is the comp side, and even that
is inside the decision line.**

⚠️ And **② splits on mem/comp and is still worse than ① on the comp side**
(1.1433 vs 1.1146). It is not "splitting on that axis makes that axis
better".

## 4. ⚠️ Two branches of the pre-registration fired **at the same time**

§3 of the pre-registration read:

```
② and ① indistinguishable        ★ switch to ② anyway (it is easier to
                                   explain)
③ indistinguishable from ①·②     ★ there is no reason to split. Take it out
                                   of the design
```

**Both fired and the two conclusions differ.** The pre-registration did not
settle that case — **that fact is written down.**

### How to read it

```
③ is the stronger statement
  if "splitting and not splitting are indistinguishable",
  then **which axis to split on is a question that need not be asked at all**
  the preference for switching to ② presupposed "we do split"
```

## 5. What remains

```
★ all five arms indistinguishable (unpaired and paired alike)
★ no boundary sensitivity — SOL 0.25/0.5/1.0 within 0.006
★ the only place splitting earns its keep is the comp side's 0.041, and even
  that is inside the decision line
★ two branches of the pre-registration collided — ③ is the stronger statement
⚠️ n=6 and σ is 0.031~0.043. It is not "there is no difference" but
   **"this sample cannot tell them apart"**
```
