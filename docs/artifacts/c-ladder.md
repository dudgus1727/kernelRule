# The (c) regrow ladder — **which part** of hw is used

The pre-registrations are [c-rerun-prereg.md](c-rerun-prereg.md) ·
[c-rerun3-prereg.md](c-rerun3-prereg.md). Producing the numbers took **0 LLM
calls**.

```
python3 experiments/c_ladder.py      # -> docs/artifacts/c-ladder.json
```

> ⚠️ 2026-09-08 (D-146): translated into English. The numbers and the verdicts
> are unchanged; the Korean original is at commit `ee53b4d`.

Each rung differs in **exactly one** thing. The 5090 table, the 53 common
shapes, the split `nk11008`, 3 seeds x 12 rounds, budget 8, the model
`gpt-5.6-luna`. The regret comes from **the same procedure** as
`sigma_5090.py` (the final scoring) — the old (c)'s value came out of that
procedure.

---

## 0. ⚠️ First — the recorded (c) `1.0485` is **a value mixing two seeds**

`transfer_29_5.TABLES["5090"]["runs"]` bundles six runs as (c). Opening the
`chosen.json` of the last three (`5090sigma-b-s*`):

```
5090sigma-s*     source = rule_writer-try03    ★ the definition of (c) — from
                                                 scratch on the 5090
5090sigma-b-s*   source = physics_seeded       ★ **the hand seed**
```

`physics_seeded` is a seed a human wrote while looking at the A6000. **It is
not "from scratch on the 5090 table".** §29.5's (c) 1.0485 is an average over
two conditions.

```
RuleWriter seed, 3 seeds   1.0416  (1.0315~1.0625)   ← the definition of (c)
the hand seed, 3 seeds     1.0506  (1.0463~1.0663)
the median of the six      1.0485                    ← the recorded value
```

**The ladder's first rung uses 1.0416.** The recorded 1.0485 is not deleted —
how it came about is written down here (documentation rule 2).

---

## 1. The ladder — the verdict is **indistinguishable on both steps**

```
                                        median      range
old (c)     numbers A6000 · warnings A6000  1.0416   1.0315~1.0625
middle (c)  numbers 5090  · warnings A6000  1.0384   1.0346~1.0422
new (c)     numbers 5090  · warnings 5090   1.0611   1.0422~1.0761
the hand seed (outside the ladder)          1.0506   1.0463~1.0663
```

The decision line is **delta = 0.0516** (the σ upper bound, the value §29.5
already used).

```
old -> middle (the numbers)             1.0416 -> 1.0384   +0.0032   ★ indistinguishable
middle -> new (the warnings section)    1.0384 -> 1.0611   -0.0227   ★ indistinguishable
```

### Following the pre-registration's decision path as written

```
middle -> new is indistinguishable   -> the warnings section is not used
then old -> middle is what decides
old -> middle is indistinguishable too  -> ★ the whole hw prose is not used.
                                           table_facts dominates
```

⚠️ **One reservation.** The new (c) is the side that got **worse** (−0.0227),
and the ranges `1.0346~1.0422` against `1.0422~1.0761` **touch at a single
point.** It is under the decision line so it is "indistinguishable", but
"there is no difference" and "we cannot tell them apart" are different
things. With 3 seeds this is as far as it goes.

---

## 2. tau — the top ranks **do not rise** when the warnings go

```
                                          top-100 tau   noise-aware   all-range
old (c)     numbers A6000 · warnings A6000     0.104        0.104        0.330
middle (c)  numbers 5090  · warnings A6000     0.078        0.071        0.352
new (c)     numbers 5090  · warnings 5090      0.097        0.091        0.229
```

★ **D-100's "the ranking ability does not transfer" is not hw's fault.**
Erasing the warning that says "give up on the short shapes" leaves the
top-100 tau at 0.08~0.10.

⚠️ These taus come from **the regret-refitted weights** (the final scoring
path). It is the same cell as D-118's picture of the wall — the regret arm's
tau is around 0.1 to begin with.

---

## 3. The seed structure — the `log_sol_ms` branch **did not come back**

```
                                          candidates splitting   in the      axes
                                          by shape length        chosen seed
old (c)     numbers A6000 · warnings A6000        2/10           ★ present   9~12
middle (c)  numbers 5090  · warnings A6000        0/10             absent    9~11
new (c)     numbers 5090  · warnings 5090         0/10             absent    8~12
```

**Changing the numbers to the 5090 alone made it 0/10, and changing the
warnings section too keeps it at 0/10.** The hw prose does reach **the seed's
form** — but that does not carry over into the performance after 12 rounds.

⚠️ n=10, so 2/10 vs 0/10 is Fisher p = 0.474. **Direction only.**

---

## 4. So

```
★ the hw prose changes the seed's **form** (the log_sol_ms branch 2/10 -> 0/10)
★ but the **performance** after 12 rounds does not change (all three
  indistinguishable)
-> the `table_facts` the evolution sees every round dominates
```

§29.5's conclusion **(b) ≈ (c)** holds. The (b) refit 1.0539 and the three
(c) arms (1.0384~1.0611) are all inside the decision line.

**The old (c)'s condition error did not change the result.** But that was
known only afterwards, and the fact that the condition was wrong for days
stays in the record (D-113, principle 39).

---

## 5. Against the pre-registration's expectations

```
expected (c-rerun3 §5)   middle -> new will be indistinguishable   ✓ correct
                         the seed's shape-length branch is worth   ✓ 0/10 held
                         looking at
                         a rise in tau would be unexpected         ✓ it did
                                                                     not rise
expected (c-rerun §6)    it will be indistinguishable              ✓ correct
```
