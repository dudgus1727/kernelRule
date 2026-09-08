"""★ The run trace (D-133) — **does the logging leave the computation
alone**?

Exactly as the instruction §4 says: the trace is not a condition. Running it
with the trace on and off, the artefacts **have to be the same**. If they
differ, the logging touched something.

It is checked with MockLLM — it is deterministic, so it has to be **exactly**
the same, and it costs 0 LLM calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.agents.mock import MockLLM
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet
from kernelrule.features import REGISTRY


def _run(table, tmp_path: Path, *, trace: bool, rounds: int = 2):
    """★ The same seed and the same MockLLM, on and off (`synth_table` is in
    conftest)."""
    m = FeatureMatrix(table, REGISTRY)
    sh = table.shapes()
    sp = SplitSet(train=Split("train", tuple(sh[:-2])),
                  val=Split("val", tuple(sh[-2:])))
    llm = MockLLM("mutate", seed=1, feature_names=m.feature_names(),
                  shape_values=["is_memory_bound"])
    cfg = LoopConfig(run_id=f"t{int(trace)}", max_rounds=rounds,
                     n_rules_per_round=4, max_evals=40, seed=0,
                     sandbox_first_seen=False, out_dir=str(tmp_path),
                     objective="regret", trace=trace)
    loop = RoundLoop(cfg=cfg, table=table, matrix=m, splits=sp, llm=llm)
    loop.run(rounds, verbose=False)
    loop.dump()
    return Path(tmp_path) / cfg.run_id


def test_trace_does_not_change_the_result(synth_table, tmp_path):
    """★ Run it on and off — **are the archive and the round records the
    same**?

    With the same seed, the same RNG and MockLLM it has to be exactly the
    same. If it differs, the logging had a side effect on the computation
    path (the instruction §4).
    """
    off = _run(synth_table, tmp_path, trace=False)
    on = _run(synth_table, tmp_path, trace=True)
    assert (off / "archive.jsonl").read_text() == \
        (on / "archive.jsonl").read_text()

    # ★ `seconds` is the wall clock, so two runs cannot help differing — it is
    #   not a computed result. Everything **but that** is compared.
    def rounds(d):
        return [{k: v for k, v in json.loads(x).items() if k != "seconds"}
                for x in (d / "rounds.jsonl").read_text().splitlines()]

    assert rounds(off) == rounds(on)
    assert not (off / "trace.jsonl").exists()
    assert (on / "trace.jsonl").exists()


def test_trace_first_line_is_self_sufficient(synth_table, tmp_path):
    """★ Can the condition be known from the first line alone (the
    instruction §3-3)?"""
    d = _run(synth_table, tmp_path, trace=True)
    first = json.loads((d / "trace.jsonl").read_text().splitlines()[0])
    assert first["ev"] == "run_start"
    for k in ("commit", "config", "split", "n_train", "n_val", "features"):
        assert k in first, k
    # The whole config — it has to be **the same thing** as `config.json`
    # (principle 2)
    saved = json.loads((d / "config.json").read_text())
    assert first["config"]["loop"]["seed"] == saved["loop"]["seed"]
    assert first["config"]["rule_constraints"] == saved["rule_constraints"]


def test_trace_records_the_order_and_the_failures(synth_table, tmp_path):
    """★ Are the order and **the failures** kept — what the scattered files
    could not do."""
    d = _run(synth_table, tmp_path, trace=True)
    evs = [json.loads(x)["ev"]
           for x in (d / "trace.jsonl").read_text().splitlines()]
    assert evs[0] == "run_start"
    assert evs.count("round_start") == evs.count("round_end") == 2
    assert "proposal" in evs and "scored" in evs and "archive" in evs
    # Is the time monotone — being time-ordered in one file is the point of
    # this format
    ts = [json.loads(x)["t"]
          for x in (d / "trace.jsonl").read_text().splitlines()]
    assert ts == sorted(ts)


def test_trace_is_not_a_condition():
    """★ It must not go into `runset.KEYS` (the instruction §8)."""
    from kernelrule.core.runset import KEYS

    assert "trace" not in KEYS
