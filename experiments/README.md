# experiments/

**Everything is run from the repository root.** `runs/` and `datasets/` are
read by relative path.

Each script leaves its result as one document in `docs/artifacts/`. **A
script with no document is deleted** — a result that lives only in a commit
message is one a new session cannot find.

| script | LLM | the artifact it makes | state |
|---|---|---|---|
| `rerun.py` | ✅ | `rerun-preregistration.md` | ★ **the re-run representative runner** — the criteria are in `PREREG` |
| `f1_pipeline.py` | ✅ | `f1-guided.md` · `f1k-preregistration.md` · (§30.9) | ★ **the F0~F3 representative runner** — `--dry-run` verified |
| `seed_selection.py` | ✅ | `luna-baseline.md` · `conclusion.md` | F3 only (the old representative value) |
| `export_rules.py` | — | `docs/artifacts/rules/` | **exporting the rules** |
| `verify_rules.py` | — | (verification only) | ★ **the verification path** |

| `new_axes.py` | ✅ | `new-axes.md` | ⚠️ the originals were deleted |
| `feature_writer.py` | ✅ | `feature-writer-f1.md` | valid — the luna re-measurement is done |
| `rule_writer_gate.py` | ✅ | `rule-writer-gate.md` | ⚠️ the originals were deleted (only the mini runs remain) |

| `score_new_axes.py` | — | scoring for `new-axes.md` | ⚠️ the originals were deleted |
| `rescore_canonical.py` | — | the final scoring table of `conclusion.md` | valid |
| `regime_transfer.py` | — | `structure-transfer.md` · `regime-diagnosis.md` | valid |
| `proxy_dispatch.py` | — | `regime-diagnosis.md` · `glossary.md` | valid |
| `regime_count.py` | — | `regime-count.md` | ⚠️ the evolved arm's originals were deleted |
| `seed_spread.py` | — | the seed spread in `conclusion.md` | ⚠️ the originals were deleted — the list has to be filled in for it to run |
| `selection_spread.py` | — | `decisions.md` D-40 / D-42 | ⚠️ the originals were deleted — the list has to be filled in for it to run |
| `fitter_sweep.py` | — | `fitter-sweep.md` · D-55 | valid — it reads only the committed rules |
| `fitter_polish.py` | — | `fitter-sweep.md` · D-55 | valid — it reads only the committed rules |
| `fitter_movement.py` | — | D-56 · D-57 | ★ **the fitter pass condition** — a reach rate of 90% |
| `polish_ranking.py` | — | D-57 | ★ **the re-run decision** — does the ranking change |

## ★ How to verify a number

```bash
python3 experiments/verify_rules.py
```

`runs/` is in `.gitignore`, so it is not in the repository. This command does
**not** read it — it recomputes the structural holdout from the rules in
`docs/artifacts/rules/*.py` and their fitted weights alone, and checks that
against `index.json`. If they diverge it fails.

```
an LLM run        not reproducible (the randomness is not controlled —
                  §24.4b)
scoring/rescoring ★ deterministic. A few seconds
```

The performance numbers in the documents are verified along this path. If you
make a new run, run `export_rules.py` again — without it the documents go
stale silently (the tests do check that `rules/*.py` and `index.json` pair
up).

## The run condition is in the configuration, not in the code

The model, the endpoint and the reasoning effort all come from one place,
**`DEFAULT_MODEL` and `LLMConfig` in
`kernelrule/agents/openai_client.py`**. A script that nails them down itself
makes a test fail (D-45).

An experimental condition (whether the feature descriptions are given, and so
on) is put in **as a flag**, like `LLMConfig.feature_detail` — reverting the
code back and forth makes it impossible to tell which run was under which
condition. Every run leaves its configuration in `runs/{id}/config.json`
(D-51).

## Scoring goes through `core/canonical.py` alone

`canonical_score(code, w0, table=, matrix=, splits=)` runs **only if it is
given the loop's `SplitSet`**, and there is no path that picks the shapes
separately. Otherwise the holdout overlaps the training shapes — 11 of 19
actually did (D-36).

## ★ How to run F0~F3

```bash
python3 experiments/f1_pipeline.py F1 --dry-run      # plumbing check, 0 LLM
python3 experiments/f1_pipeline.py F1                # the real thing
python3 experiments/f1_pipeline.py F1 --stage 2      # reuse the stage-1
                                                     # artefacts
```

The condition decides exactly one thing: **which registry goes into all three
stages**.

```
F3  the human 24 + the human_guided seed   stage 1 is skipped (that is what
                                           the condition is)
F2  the base 5 + FeatureWriter
F1  raw values only -> FeatureWriter -> the RuleWriter seed
F0  no features
```

**The first thing to look at** after running it:

```bash
jq .human_features_present runs/f1pipe-F1-*/config.json   # it must be [] for F0/F1
jq .physics_coverage runs/f1pipe-F1-*/stage1-features/summary.json
```
