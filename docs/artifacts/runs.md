# The run coordinates — **it is a generated file**

> ★ This table is built by `experiments/runs_table.py`. **Do not edit it by
> hand** — `runs/*/config.json` and the artifact json are the source, and
> `tests/test_docs.py` catches any divergence (`--check`).
>
> ```
> python3 experiments/runs_table.py            # build it again
> python3 experiments/runs_table.py --check    # only look at whether it diverged
> ```
>
> ⚠️ 2026-09-08 (D-146): this document and the strings the script writes into
> it were translated into English. Nothing but the language changed; the
> Korean original is at commit `ee53b4d`. Old `## D-N` entries in
> `docs/decisions.md` quote the old marks (`★ 미업로드`, `★갈림`) — those are
> the record of what the table said at the time and they stay.
>
> ⚠️ **While a campaign is running it is normal for `--check` to be red** —
> `config.json` is updated every round, so the table falls behind. Building it
> again once it finishes is enough. What this check is for is **a table edited
> by hand**.
>
> ★ **From 2026-09-05 (D-137) it is not red** — `--check` compares with the
> **rows** of the tags whose `rounds.jsonl` was written within the last 30
> minutes left out (`_live_tags`, it looks only at mtime). The check on the
> other rows stays alive, and the skipped tags are printed.

## The tag rule (D-128)

```
<features><seed rule>-p<parameters>[-<expressiveness>][-<experiment name>]

F3rw-p8        the F3 library · the RuleWriter seed · parameters 8 · default
F3rw-p16       parameters 16
F3rw-p8-prod   the product hint    F3rw-p8-pow   the exponent hint
F3hg-p8-d75-a  the human_guided seed
★ The table (GPU), the inheritance and the code version do not go in the tag
  — config.json holds them
★ A directory starting with `x-` is **discarded** (the rank-loss line · a
  condition error). It was not deleted, it is marked by name, and it does not
  go into this table
```

<!-- RUNS:BEGIN — experiments/runs_table.py builds it -->

| tag | seeds | features | seed rule | parameters | expressiveness | fitter | rounds | table | commit | trace | final score | source | status |
|---|--:|---|---|--:|---|---|---|---|---|---|--:|---|---|
| `D146r` | 1 | 19/F3 | rule_writer-try05 | 8 | default | nelder-mead/4/200 | 1 | a6000 | `ee53b4d` | ★ not uploaded | — | — | |
| `F1rw-p8` | 6 | 16/? | rule_writer-try09 | ? | default | nelder-mead/4/200 | 12 | a6000 | `?` | — | 1.1195 | conclusion.json | |
| `F2rw-p8` | 6 | 17/? | rule_writer-try01 | ? | default | nelder-mead/4/200 | 12 | a6000 | `?` | — | 1.1288 | conclusion.json | |
| `F3hg-p8-d75-a` | 3 | 20/F3 | human_guided | 8 | default | nelder-mead/4/200 | 4 | a6000 | `?` | — | — | — | |
| `F3hg-p8-d75-b` | 6 | 19/F3 | human_guided | 8 | default | nelder-mead/4/200 | 4 | a6000 | `?` | — | — | — | |
| `F3rw-p16` | 3 | 19/F3 | rule_writer-try05 | 16 | default | cma/1/300 | 12 | a6000 | `?` | — | 1.0906 | expressive-regret.json | |
| `F3rw-p8` | 6 | 19/F3 | rule_writer-try05 | 8 | default | nelder-mead/4/200 | 24 | a6000 | `?` | — | 1.0787 | canon-p8.json | a campaign that ran on faulty code — 2% of the proposals were thrown away by `__import__` (D-135). The canonical value is `F3rw-p8-nan` |
| `F3rw-p8-4090` | 3 | 19/F3 | rule_writer-try00 | 8 | default | nelder-mead/4/200 | 12 | 4090 | `?` | — | 1.0493 | sigma-4090.json | |
| `F3rw-p8-5090` | 3 | 19/F3 | rule_writer-try05 | 8 | default | nelder-mead/4/200 | 12 | 5090 | `?` | — | 1.0611 | c-ladder.json | |
| `F3rw-p8-abl-analyst` | 3 | 19/F3 | rule_writer-try05 | 8 | default | nelder-mead/4/200 | 12 | a6000 | `?` | — | — | — | |
| `F3rw-p8-abl-noanalyst` | 3 | 19/F3 | rule_writer-try05 | 8 | default | nelder-mead/4/200 | 12 | a6000 | `?` | — | — | — | |
| `F3rw-p8-abl-shuffled` | 3 | 19/F3 | rule_writer-try05 | 8 | default | nelder-mead/4/200 | 12 | a6000 | `?` | — | — | — | |
| `F3rw-p8-cma` | 3 | 19/F3 | rule_writer-try05 | 8 | default | cma/1/300 | 12 | a6000 | `?` | — | 1.0987 | expressive-regret.json | ⛔ retired — p8 and yet CMA — the current rule (fitter_for) does not produce it |
| `F3rw-p8-cross` | 3 | 19/F3 | rule_writer-try05 | 8 | default | nelder-mead/4/200 | 12 | a6000 | `?` | — | — | — | |
| `F3rw-p8-d75` | 6 | 21/F3 | rule_writer-try05 | 8 | default | nelder-mead/4/200 | 4 | a6000 | `?` | — | — | — | |
| `F3rw-p8-nan` | 6 | 19/F3 | rule_writer-try05 | 8 | default | nelder-mead/4/200 | 24 | a6000 | `21aee74·4803a1b·cdd9cc1` | trace-F3rw-p8-nan-cdd9cc1 | 1.0886 | round-curve-bests.json | ★ the current canonical value — `-nan` means the np.errstate guard in `compile_rule` (D-135). The campaign before it is `F3rw-p8`. ★ The final score is **read at r11** (round 12, D-140) — the campaign ran to 24. The commits differ because of a docs commit in the middle of the campaign, and `kernelrule/`·`prompts/` did not change in any of the pairs (checked, D-137) |
| `F3rw-p8-old` | 6 | 19/? | rule_writer-try05 | ? | default | nelder-mead/4/200 | 12 | a6000 | `?` | — | 1.0762 | conclusion.json | ⛔ retired — the old canonical value — the old prompt·12 rounds·patience 10 (D-129) |
| `F3rw-p8-p3` | 6 | 19/F3 | rule_writer-try05 | 8 | default | nelder-mead/4/200 | 5~6~7 | a6000 | `?` | — | — | — | a campaign stopped at r4~r6 by patience 3 (D-131) |
| `F3rw-p8-pow` | 3 | 19/F3 | rule_writer-try05 | 8 | pow | cma/1/300 | 12 | a6000 | `?` | — | 1.0839 | expressive-regret.json | ⛔ retired — p8 and yet CMA. To be re-measured |
| `F3rw-p8-prod` | 3 | 19/F3 | rule_writer-try05 | 8 | prod | cma/1/300 | 12 | a6000 | `?` | — | 1.0840 | expressive-regret.json | ⛔ retired — p8 and yet CMA. To be re-measured |
| `luna` | 3 | 19/? | ? | ? | default | nelder-mead/4/200 | 12 | a6000 | `?` | — | — | — | |
| `lunaNAMES` | 6 | 19/? | ? | ? | default | nelder-mead/4/200 | 12 | a6000 | `?` | — | — | — | |
| `nocap` | 1 | 19/F3 | human_guided | 8 | default | nelder-mead/4/200 | 2 | a6000 | `131d612` | trace-nocap-131d612 | — | — | ★ a behaviour check, not a result (D-152) — 1 seed x 2 rounds, fold 0, 3-term seed, **after the live cap in the schema validator was removed**. Do not read its numbers |
| `nocount` | 1 | 19/F3 | human_guided | 8 | default | nelder-mead/4/200 | 1 | a6000 | `1ecdd56` | trace-nocount-1ecdd56 | — | — | ★ a contrast, not a result (D-151) — 1 seed x 1 round, fold 0, after the size sentences were taken out of every surface. Do not read its numbers |
| `seed3` | 1 | 19/F3 | human_guided | 8 | default | nelder-mead/4/200 | 1 | a6000 | `1ecdd56` | trace-seed3-1ecdd56 | — | — | ★ a contrast, not a result (D-151) — the same round with a **3-term seed**, to see whether len(w0) follows the parent. Do not read its numbers |
| `smoke` | 1 | 19/F3 | human_guided | 8 | default | nelder-mead/4/200 | 2 | a6000 | `579bdf6` | trace-smoke-579bdf6 | — | — | ★ a wiring smoke run, not a result (D-149) — 1 seed x 2 rounds, fold 0. Do not read its numbers |
| `verify` | 2 | 19/? | ? | ? | default | nelder-mead/4/200 | 6 | a6000 | `?` | — | — | — | |

<!-- RUNS:END -->

## Footnotes

```
★ n=6 is the standard. A run at n=3 is weak in principle
  (3 against 3 gives a minimum p of 0.10 for "they do not overlap") — to be
  re-measured
objective       regret@1 (since D-128 this is the only evolution path)
split           nk11008 — training/holdout 20. The training count differs per
                table
model           fixed at gpt-5.6-luna
fitter          parameters 8 -> nelder-mead/4/200,  16 -> cma/1/300 (D-128)
                ⚠️ a marked row is **an old run that ran off that rule**
final score     refit per regime -> the holdout. **It is read from the json in
                the source column**
★ 12 rounds is an unverified value — D-127's verdict is "(b) it is not
  enough", and the shortfall (0.0055~0.0060) is smaller than the seed range σ
  (0.0124)
features column `n_features`/`feature_condition`. `?` is from the days that
                key did not exist
seed rule column the source in `chosen.json`. `?` is an old run with no such
                file
```

## The rows whose final score is still empty

`—` means **there is not yet an artifact json holding that tag's final
score**. The number is not written in here by hand — re-measure it, produce
the artefact, and it fills in.
