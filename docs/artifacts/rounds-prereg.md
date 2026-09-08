# Pre-registration — **is 12 rounds right** (from the curves, 0 LLM calls)

**Written before the computation.**

> ⚠️ 2026-09-08 (D-146): translated into English. The criteria and the numbers
> are unchanged; the Korean original is at commit `ee53b4d`.

## 0. Why — "early stopping did not fire" is not "it converged"

```python
def should_stop(self):          # loop.py, patience n = 10
    if len(self.rounds) < n + 1: return False, ""
    vals = [x.best_val_regret for x in self.rounds[-(n + 1):]]
    improved = vals[0] - vals[-1]
    significant = is_significant(improved, ev)      # compared to the noise floor
    new_cell_recent = archive.last_new_cell_round > len(self.rounds) - 1 - n
    if significant or new_cell_recent: return False, ""
```

```
patience is 10 and 12 rounds are run
-> the judgement starts at the 11th round, and there are only two windows,
   r0..r10 · r1..r11
-> and **it does not stop if even one new cell appeared recently**
★ early stopping effectively never fires. So running all 12 so far was not
  "because it converged" but "because there was no chance to stop"
```

## 1. What is measured — the recorded curves only (0 LLM calls)

```
targets   F3rw-p8 (formerly arch24), 6 seeds   ← the main one
          F1rw-p8 (formerly F1-free-roofline), 6 seeds · F2rw-p8 (formerly
          F1-K-k1), 6 seeds  ← the condition comparison
data      best_val_regret / n_cells in runs/<run>/rounds.jsonl
threshold ★ exactly what the loop uses — `is_significant(delta, ev)`, where
          `ev` is that run's final archive-best rule scored on the holdout.
          No new criterion is made (principle 2)
```

```
The round number is the file's `round` field = **from 0** (r0..r11 is 12
rounds).
"the last 3 rounds" = r9 · r10 · r11
```

What is produced:

```
1  the best_val_regret curve per seed
2  the last round that **improved** (by any amount) — what D-89 counted
3  the last round that improved **significantly** — above the threshold
4  the last round a new cell appeared (n_cells increased)
5  **when it would have stopped** with patience at 3·4·10 (by the same
   formula)
```

★ Separating 2 from 3 is the core of this experiment — which of the two
D-89's "r7/r8/r9" was is not known right now.

## 2. The verdict — chosen from three. **Not made after looking at the curves**

```
(a) all 6 seeds have their last **significant** improvement at r8 or earlier
    ★ 12 is enough. Rounds are not an axis
    -> a footnote records "it converged at 12" together with the curves

(b) there are seeds that improve **significantly** in the last 3 rounds
    (r9·r10·r11)
    ★ 12 is not enough
    -> round 24 has to be measured at n=6 (about 6,000 calls / 15 hours)

(c) ambiguous (e.g. the significant improvements end early but new cells keep
    appearing to the end)
    ★ adjusting patience is reviewed (10 -> 3~4)
    -> then rounds become **an artefact** rather than a condition
    ⚠️ changing patience is itself a condition change — it needs **its own
       pre-registration** and it is not changed here
```

**"improvement by any amount" is also reported, as a secondary.** The verdict
is made on the **significant** side alone.

## 3. What not to do

```
[ ] do not make a new significance criterion — `is_significant` as it is
[ ] do not fix the decision line after looking at the curves
[ ] do not change patience in this experiment (it needs its own
    pre-registration)
[ ] do not speak from one seed's curve — all 6 seeds are reported
```
