# The two-stage objective · the transfer of a rank rule (2026-09-01)

> **Pre-registration** [two-stage-prereg.md](two-stage-prereg.md) — the
> decision line was nailed down first and **was not changed** · **0 LLM calls**
> **Reproduce** `python3 experiments/two_stage.py`
> **Raw data** [two-stage.json](two-stage.json)

> ⚠️ 2026-09-08 (D-146): translated into English. The numbers and the verdicts
> are unchanged; the Korean original is at commit `ee53b4d`.

⚠️ **Everything is on the holdout of 20 shapes.** Do not put it beside D-101's
tau (the training 41 shapes, 0.389) (principle 4).

---

# A. The structure by rank, the weights by regret — ★ it does not work

```
A6000 holdout 20 shapes             regret   top-100 tau   all-range tau
rank structure + rank weights       1.6364      0.353          0.320
★ rank structure + regret weights   1.1213    ★ 0.115          0.267
regret structure + regret weights   1.0762      0.122          0.300
★ random floor (20 draws)           1.8753     -0.007          0.000
```

## The verdict — it does not reach the pre-registered line

```
regret 1.1213 / tau 0.115  ->  ★ indistinguishable
  success                  regret <= 1.10 and tau >= 0.30
  the weights erase it     regret <= 1.10 but tau <= 0.15
  the structure misfits    regret >= 1.30
```

**It fell between the two lines.** `regret` is 0.021 over 1.10 and `tau` is
0.035 under 0.15. **The decision line is not moved** — as pre-registered, it
is "indistinguishable".

## But the direction is clear

```
the moment the weights are changed to regret, tau collapses 0.353 -> 0.115
and that 0.115 is ★ effectively the same as the regret-evolved structure's
0.122
```

**The structure holds almost none of the ranking ability.** The tau of 0.353
the rank evolution produced was **held by the weights**, and it disappears
when the weights change.

And nothing is gained either.

```
regret   rank structure 1.1213  vs  regret structure 1.0762
                                    ★ the rank structure is worse
tau      0.115                  vs  0.122   ★ no difference
```

**"the structure by rank, the weights by regret" gets neither side.**

## ★ (2) evolving on from there is not done

The pre-registration: *"if regret >= 1.30 the structure misfits → (2) is
needed."* `regret` is 1.12, so **that is not the branch.** And the
observation above breaks (2)'s premise — (2) assumes "the structure is usable
but a weight stage is needed", whereas **the structure does not hold the
ranking ability.**

---

# B. The transfer of a rank rule — ★ the top ranks do not move. The coarse order does

```
5090 holdout 20 shapes                regret   top-100 tau   all-range tau
(a) full transplant (A6000 weights)   1.1311      0.111          0.388
★ (b) refit (5090 rank loss)          1.1267    ★ 0.123          0.349
★ random floor (20 draws)             1.4217      0.001          0.000
```

## The top ranks — they do not come back

```
A6000 holdout (rank structure + rank weights)   0.353
5090 (a) full transplant                        0.111
5090 (b) ★ refitted with the 5090 rank loss     0.123
```

**Refitting on the 5090 table with the rank loss does not bring it back.**
(a) and (b) are 0.111 against 0.123, almost the same — **the refit does
almost nothing.**

This is **the same direction** as what D-100 saw with `regret`-evolved rules.
Evolving with the ranking as the objective does not move that ability to the
5090 either.

### ⚠️ The cause is not asserted — it was written down in advance

```
the time span of the 5090's top 100        ★ 1.2%   (A6000 6.6%)
distinct time values in the 5090's top 100 ★ 5      (A6000 24)
```

**"it cannot learn" and "there is nothing to learn" cannot be told apart from
this data.** The usable statement reaches only as far as **"it did not come
back. But the 5090's top ranks span 1.2% with 5 distinct values, so there is
little to learn in the first place"**.

## ★ The coarse order does move

```
all-range tau   A6000 holdout 0.320  ->  5090 (a) 0.388 / (b) 0.349
```

**It does not drop. It is slightly higher.** The random floor is 0.000, so it
did transfer.

```
★ the coarse order rides on the structure and moves.
★ the order inside the top 100 does not move.
```

D-100's "the ranking ability does not transfer" **has to be split along two
axes** — the all-range order transfers and only the top ranks do not.

---

## Overall

```
evolving with the rank loss produces top-rank      D-101, tau 0.353 (holdout)
ability
★ that ability is **held by the weights**          refitting with regret
                                                   gives 0.115
★ that ability **does not move to another GPU**    0.111~0.123 on the 5090
the coarse order rides on the structure and moves  all-range 0.32 ->
                                                   0.35~0.39
```

**The road to merging `the top-1 selector` and `the performance model` into
one artefact does not open in this direction.** The "which one do we put
forward" of `conclusion.md` stays as it is.
