# ★ The final-scoring curve by round — **24 is not a convergence point either**

> **Status**: measured (2026-09-05), and the verdict is in the title — ★
> **neither 12 nor 24 is a convergence point**. The number of rounds is
> fixed at 12 (D-166); this curve does not reopen it, it is the reservation
> that goes with every number read out of a 12-round run.
> ⚠️ 2026-09-11 (D-166 §L): the badge was missing — `conclusion.md` still
> listed "whether 12 is enough" as unexamined because of it.
> **Reproduce**: `python3 experiments/round_curve.py` (about 40 minutes,
> **0 LLM calls**)
> **Pre-registration**: D-132 §2 — the decision line was pinned before the
> computation
> **Raw data**: `round-curve.json` (old) · `round-curve-new.json` (new)

> ⚠️ 2026-09-08 (D-146): translated into English. The numbers and the verdicts
> are unchanged; the Korean original is at commit `ee53b4d`.

From the archive up to each round the **best training score** is picked,
refitted per regime and measured on the holdout of 20 shapes = **the value
that would have been reported had it stopped at that round**.

## 1. The median curves of the two campaigns

```
old (12 rounds, patience 10, the old prompt)
r0     r1     r2     r3     r4     r5     r6     r7     r8     r9    r10    r11
1.0981 1.1011 1.1011 1.0981 1.0949 1.0949 1.0949 1.0919 1.0796 1.0796 1.0782 1.0762
★ the round that comes inside the final value's σ (0.0113) = r8
  improvement over the last 4 rounds +0.0156

new (24 rounds, no early stopping, the new prompt)
r0     r1     r2     r3 ... r12    r13    r14    r15    r16 ...  r22    r23
1.1258 1.1258 1.1221 1.1129   1.1121 1.1013 1.1013 1.1055 1.1055 … 1.0970 1.0787
★ the round that comes inside the final value's σ = r23 (the last)
  improvement over the last 4 rounds +0.0183
```

## 2. The verdict — the pre-registration's second branch

```
"it keeps rising to 24  ★ 24 is not enough either. That fact is written down"
```

**Both campaigns rise to the last round.** The point that comes inside the
final value's σ is r8 for the old one and **r23** for the new one.

## 3. ⚠️ But **the final values are the same as each other**

```
old  end of 12 rounds  1.0762
new  end of 24 rounds  1.0787      difference +0.0025  (8% of the decision
                                   line 0.0305)
```

**Doubling the rounds did not make the final value better.** The curve says
"it is still rising", but compared campaign against campaign that gain is not
visible.

```
**within** a campaign   r0 -> end   old +0.0218 · new +0.0471   ★ it rises
**between** campaigns   end vs end  +0.0025                     ★ indistinguishable
```

The starting points differ too (r0 is 1.0981 against 1.1258) — **with the same
seed**, the 0-round results are 0.028 apart. The spread between campaigns
covers the gain from rounds.

★ So the answer to "how many rounds are needed" is **"it cannot be settled at
this spread"**. By the curve 24 is not enough; by the result 12 is enough.

## 4. What would produce an answer

```
comparing 24 vs 12 within the same campaign has already been done (the curve)
  — it rises
reducing the spread between campaigns needs more seeds
  measuring the decision line 0.0025 at σ 0.0113 needs ★ hundreds of seeds
-> the number of rounds **cannot be settled by performance**. It has to be
   settled by cost
```

## 5. So what was decided

```
24 rounds · no early stopping is kept
grounds   the curve rises to the last round (no reason to cut it shorter)
          the final value is indistinguishable from 12 but **does not get
          worse either**
          the cost is about 470 calls per run (twice that of 12 rounds)
⚠️ it is not claimed that "24 is better than 12" — that difference cannot be
   measured
```

> ⚠️ 2026-09-11 (D-166 §L): ~~"24 rounds · no early stopping is kept"~~ — the
> number of rounds for the 21-run campaign is **12**, decided by the user.
> The paragraph above is left as the record of what was decided on
> 2026-09-05 and is not deleted (documentation rule 2). What survives it is
> §2 and §3: the curve is still rising at the last round in both campaigns,
> and the two final values are indistinguishable — so 12 is a **cost**
> decision, exactly as §4 concluded, and the reservation in the badge rides
> along with every 12-round number.
