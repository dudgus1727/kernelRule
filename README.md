# kernelRule

**A pipeline that automatically produces GEMM config-selection rules** from a
performance table. An LLM writes the rule code, a pre-measured table scores
it, and an evolutionary loop iterates.

What ships is a pure Python function that runs in microseconds, plus the
record of why that function looks the way it does. **The LLM is a tool for
making the rule, not a part of the final system.**

```
input    a kernelTab bundle (table.parquet + env.json + kernels.jsonl
         + BUNDLE.json)
output   score(f, p, hw, w) -> a score array   +  the fitted weights W_FITTED
         + features/ (the validated physical feature library)
         + the hypothesis history (why the rule looks the way it does)
```

## ★ The reading order

A new session only needs to read **these three, in this order**. That should
be enough to answer "what is settled now and what has to be done next".

```
1. docs/principles.md            the 38 things stepped on repeatedly
                                 (1,084 lines). Start here
2. docs/artifacts/conclusion.md  ★ the representative numbers and the
                                 current conclusion (605 lines)
3. docs/design.md                the whole design (5,170 lines — only the
                                 sections you need)
```

⚠️ `design.md` is 5,170 lines because correction boxes have piled up on top
of the old description. **Splitting off a version that keeps only the current
implementation is pending** (`pending_fixes` 13).

The rest is read only when needed.

```
docs/glossary.md            the terms — a wrong name flipped a conclusion
                            twice
docs/decisions.md           D-1~D-114 in time order. ★ There is an index at
                            the head
docs/artifacts/*.md         the results per experiment. **Read the status
                            badge at the top first**
docs/artifacts/cost.md      the cost tally — exploration vs reproduction
docs/examples/              sample reports
experiments/README.md       the script ↔ artifact mapping
```

## The current state (2026-09-03, D-114 / principle 38)

★ **No number is written here.** The representative values live in exactly
one place, the "★ The canonical performance numbers" section of
[docs/artifacts/conclusion.md](docs/artifacts/conclusion.md) — putting them
in two places makes them diverge (principle 2).

```
settled     **indistinguishable** from the vendor (9/11 per shape, p=0.824)
            two shippable artefacts — the rule function + the validated
            feature library
            the regime boundary can be judged by a proxy (SOL) — 61/61
            agreement with t_best
            transfer (§29.5): the structure moves and **the weights are a
            hardware constant**
            ★ the wall on the rank axis — pushed from six directions and it
            did not move (D-104~D-112)

open        the 5090 (c) regrow — ★ it ran having received the A6000
            hardware facts (D-113).
            It is measured again with the three arms of the ladder (the
            numbers / the warnings section, one at a time)
            ★ the wall is being re-measured with regret@k
            (regret-at-k-prereg.md)
            design.md could not be split into current/historical
            (pending_fixes 13)
            D-42 is tested again with the cell axes redesigned
            (pending_fixes 7)
            putting one weight per shape (pending_fixes 11 — with a
            reservation attached)
```

**The pass condition is a cost-performance table, not a single threshold**
(§9.2c). We do not look at the one regret cell and say "it cleared it / it
did not".

### The rank axis is closed (2026-09-03)

```
A rule with low regret cannot order the top ranks, and the converse holds
too.
Structure / budget / ordering / expressiveness / the definition of the
objective / form — all six directions negative.
The statement that remains: **a linear combination of** this feature space
cannot get the top-rank ordering right.
```

For the details, see "★ The wall — it was pushed from six directions and it
did not move" in `conclusion.md`.

## The three documentation rules

1. Attach **a reproduction procedure** to every pass-condition number (the
   status filter / the definition of the cover / the shape set / the
   candidate set / the holdout split / the script that computed it)
2. **Do not delete** a wrong value — leave it as correction history
3. Before putting two numbers side by side, check that they came from **the
   same procedure / the same denominator / the same aggregation / the same
   data / the same model** (D-31)

## Installation

```bash
pip install -e ../kernelTab       # the table loader / answer isolation /
                                  # the noise model
pip install -e '.[test,llm]'      # ★ without llm the real execution path
                                  # dies
python3 -m pytest tests/ -q
```

`[llm]` is `pydantic` + `pydantic-ai-slim[openai]` + `openai`. **Leaving it
out skips 8 cases in `tests/test_openai_client.py`**, and that module is in
`CRITICAL_MODULES`, so the session fails — that is, the result of such a run
guarantees nothing (§26.3).

If you only want to score, `[test]` alone is enough. You just cannot run the
LLM experiments.

```bash
export OPENAI_API_KEY=...   # without it, it stops instead of silently
                            # falling back to MockLLM
```

## The absolute rule

**The rule function cannot see the table.** It is blocked in four layers.

| layer | the device |
|---|---|
| data structure | there is no time anywhere in `Problem`/`Config`/`CandidateSet` |
| the loader | `load_for_ranking` / `load_for_scoring` split in two + a whitelist |
| the static check | AST — forbidden names, imports, direct `M/N/K` comparisons, `if` on `f.*` |
| the behavioural check | the `null` preset, the constant-score test |

**A tie is broken by config identity alone.** In this table 29 of the 66
shapes are exact ties at the optimal time, with up to 84 tied — "the optimal
config of that shape" is a function of the tie-break rule, not a physical
fact. That is why `PerfTable` does not offer `best_config`.

**The noise floor differs per shape.** A fixed 1% is forbidden. In this
table the floor of the 11.3us shape is 9.1% and of the 9.7ms shape 0.048%.

## The tables

```
dev     CUDA 12.4 / host / schema_version 1   — for development
main    CUDA 13.3 / container                 — the representative numbers
                                                come from this one
5090    rtx-5090-sm_120-5bb6f403              — transfer (§29.5)
```

★ **The judgement is made per split.** The split x method table is always
given alongside (§30.6c). Performance numbers obtained on the development
table are not reported.

## The layout

```
kernelrule/
  core/     types adapter noise table scoring weights matrix splits
            sandbox archive loop
  features/ physical (25 of them) validate
  rules/    checks handwritten
  baselines/ static_topk vendor gbdt
  agents/   schemas mock
  report/   diagnostic
tools/synth.py    the synthetic table generator (4 presets)
```

## Running it

```bash
# the diagnostic report (a human reads it first — §12.4)
python3 -c "..."   # for an example see docs/decisions.md

# the MockLLM loop — ★ a __main__ guard is required (the sandbox spawns
# children)
def main(): ...
if __name__ == "__main__": main()
```
