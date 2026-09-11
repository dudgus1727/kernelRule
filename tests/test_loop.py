"""The round loop and the archive (§13, §14)."""
from __future__ import annotations

import numpy as np
import pytest

from kernelrule.agents.mock import MockLLM
from kernelrule.core.archive import (
    CELL_AXIS_NAMES,
    N_QUANTILES,
    Archive,
    Elite,
)
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.splits import Split, SplitSet


def _elite(regret=1.2, short=1.2, long=1.2, n=100, rnd=0, rid="r1",
           allv=None):
    return Elite(rule_id=rid, code="x", w=[1.0], regret=regret,
                 mem_objective=short, comp_objective=long,
                 all_objective=(regret if allv is None else allv),
                 code_len=n, round=rnd)


# ---------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------
def test_cell_axes_are_27_cells():
    """★ 3x3x3 = 27 cells (D-144). The old value was 4x4x4 = 64."""
    assert len(CELL_AXIS_NAMES) == 3
    assert N_QUANTILES ** len(CELL_AXIS_NAMES) == 27


def test_specialist_survives_even_with_bad_overall():
    """★ Even with a poor overall score, **best in one region is kept
    alive** (§13.1).

    ⚠️ 2026-09-10 (D-155): a specialist must also clear the cut line — the
    archive keeps different **kinds**, not bad rules. Both of these are
    inside the top fraction of the population.
    """
    a = Archive()
    for i in range(8):
        a.consider(_elite(regret=1.30 + 0.02 * i, short=1.2 + 0.05 * i,
                          long=1.5 - 0.05 * i, rid=f"mid{i}"))
    a.consider(_elite(regret=1.12, short=1.20, long=1.10, rid="all"))
    won = a.consider(_elite(regret=1.13, short=1.02, long=1.40,
                            rid="memspec"))
    assert won, "the memory-band specialist was discarded"
    assert a.best.rule_id == "all"
    assert any(e.rule_id == "memspec" for e in a.cells.values())


def test_a_bad_rule_does_not_take_a_cell():
    """★ The archive is where different kinds live, not where the worst is
    stored (D-155)."""
    a = Archive()
    for i in range(9):
        a.consider(_elite(regret=1.10 + 0.01 * i, rid=f"g{i}"))
    n_before = a.n_cells
    won = a.consider(_elite(regret=1.60, short=1.9, long=1.3, rid="bad"))
    assert not won, "a rule far below the cut line took a cell"
    assert a.n_cells == n_before
    assert all(e.rule_id != "bad" for e in a.cells.values())


def test_the_population_is_every_rule_scored():
    """★ The boundaries come from everything scored, not from the elites
    that are alive (D-155).

    With the live elites as the population it fed back on itself: a good
    candidate could not get in, so the population did not change, so the
    boundaries did not move, so the next good candidate could not get in.
    """
    a = Archive()
    for i in range(6):
        a.consider(_elite(regret=1.40 - 0.01 * i, rid=f"x{i}"))
    assert len(a._seen_keys) == 6
    seen, cells = a.n_seen, a.n_cells
    a.consider(_elite(regret=1.90, rid="refused"))
    assert a.n_seen == seen + 1, "a refusal must still enter the population"
    assert len(a._seen_keys) == 7
    assert a.n_cells == cells


def test_noise_tolerance_blocks_meaningless_updates():
    """Updating the **global best** on "slightly better" accumulates noise
    (§13.4).

    ⚠️ 2026-09-08 (D-144): with the cells now dynamic tertiles, "an empty
    list" no longer shows it — with two candidates the tertiles put them in
    **different cells**, so `new_cell` is legitimately taken. What this test
    holds is the **`best` update**.
    """
    a = Archive(noise_tol=0.01)
    a.consider(_elite(regret=1.20, rid="a"))
    assert "best" not in a.consider(_elite(regret=1.195, rid="b"))
    assert a.best.rule_id == "a"
    assert "best" in a.consider(_elite(regret=1.15, rid="c"))


def test_parent_mix_is_exploit_explore_cross():
    """★ 6 proposals = exploit 3 / explore 2 / cross 1 (D-144). The old
    value was 12 = 6/3/3."""
    a = Archive()
    for i in range(9):
        a.consider(_elite(regret=1.3 - 0.01 * i, short=1.0 + 0.05 * i,
                          long=1.0 + 0.02 * i, allv=1.2 - 0.01 * i,
                          n=50 + 30 * i, rid=f"r{i}"))
    kinds = [k for k, _ in a.parents(6, np.random.default_rng(0))]
    assert kinds.count("exploit") == 3
    assert kinds.count("explore") == 2
    assert kinds.count("cross") == 1


def test_new_cell_round_is_tracked():
    a = Archive()
    a.consider(_elite(rnd=0, rid="a"))
    a.consider(_elite(short=1.02, rnd=4, rid="b"))
    assert a.last_new_cell_round == 4


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
@pytest.fixture
def loop(synth_table, tmp_path):
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY

    fm = FeatureMatrix(synth_table, REGISTRY)
    sh = synth_table.shapes()
    splits = SplitSet(train=Split("train", tuple(sh[:-2])),
                      val=Split("val", tuple(sh[-2:])))
    llm = MockLLM("mutate", seed=1, feature_names=fm.feature_names(),
                  shape_values=["is_memory_bound"])
    cfg = LoopConfig(run_id="test", n_rules_per_round=4, max_rounds=3,
                     max_evals=40, seed=0, sandbox_first_seen=False,
                     out_dir=str(tmp_path))
    return RoundLoop(cfg=cfg, table=synth_table, matrix=fm, splits=splits,
                     llm=llm)


def test_loop_runs_and_fills_the_archive(loop):
    loop.run(3, verbose=False)
    assert len(loop.rounds) == 3
    assert loop.archive.best is not None
    assert loop.archive.n_cells >= 1
    assert sum(r.n_scored for r in loop.rounds) > 0


def test_llm_call_budget_matches_the_design(loop):
    """1 diagnosis + n rules per round (§11.1 — about 89% of the calls are
    RuleEditor)."""
    loop.run(3, verbose=False)
    for i, r in enumerate(loop.rounds):
        assert r.llm_calls["rule_editor"] == loop.cfg.n_rules_per_round
        # Round 1 has an empty archive, so it skips the diagnosis
        assert r.llm_calls["analyze"] == (0 if i == 0 else 1)


def test_adversarial_mode_scores_nothing(synth_table, tmp_path):
    """★ In adversarial mode **not one may be scored** (§24.3)."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY

    fm = FeatureMatrix(synth_table, REGISTRY)
    sh = synth_table.shapes()
    splits = SplitSet(train=Split("train", tuple(sh[:-2])),
                      val=Split("val", tuple(sh[-2:])))
    llm = MockLLM("adversarial", seed=0, feature_names=fm.feature_names())
    cfg = LoopConfig(run_id="adv", n_rules_per_round=12, max_rounds=1,
                     max_evals=20, seed=0, out_dir=str(tmp_path))
    lp = RoundLoop(cfg=cfg, table=synth_table, matrix=fm, splits=splits,
                   llm=llm)
    r = lp.run_round()
    assert r.n_proposed == 12
    assert r.n_scored == 0, (
        f"{r.n_scored} adversarial rules were scored")
    assert lp.archive.best is None


def test_loop_dump_writes_everything(loop, tmp_path):
    loop.run(2, verbose=False)
    d = loop.dump()
    for name in ("archive.jsonl", "rounds.jsonl", "failures.jsonl",
                 "hypotheses.jsonl"):
        assert (d / name).exists(), name
    assert (d / "llm_calls").exists()


def test_replay_reproduces_the_run(synth_table, tmp_path):
    """★ The same LLM responses reproduce the result (§24.4)."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY

    fm = FeatureMatrix(synth_table, REGISTRY)
    sh = synth_table.shapes()
    splits = SplitSet(train=Split("train", tuple(sh[:-2])),
                      val=Split("val", tuple(sh[-2:])))

    def build(llm, run_id):
        cfg = LoopConfig(run_id=run_id, n_rules_per_round=3, max_rounds=2,
                         max_evals=30, seed=0, sandbox_first_seen=False,
                         out_dir=str(tmp_path))
        return RoundLoop(cfg=cfg, table=synth_table, matrix=fm,
                         splits=splits, llm=llm)

    a = build(MockLLM("mutate", seed=4, feature_names=fm.feature_names()), "a")
    a.run(2, verbose=False)
    a.llm.dump(tmp_path / "calls")
    b = build(MockLLM("replay", replay_dir=tmp_path / "calls"), "b")
    b.run(2, verbose=False)
    assert [r.best_regret for r in a.rounds] == [r.best_regret
                                                 for r in b.rounds]


def test_early_stop_path_is_sealed(loop):
    """★ The early-stop path is **sealed** (D-144).

    The old implementation read `best_val_regret` — a path by which the
    validation split enters the stopping verdict. It did not run because
    `patience=0`, but **the moment someone turns it on the test is
    contaminated.** So turning it on is an error.

    The old test was "even when the score stalls it keeps running if a new
    cell appears (§14.3)".
    """
    loop.run(1, verbose=False)
    assert loop.cfg.patience == 0
    assert loop.should_stop() == (False, "")
    loop.cfg.patience = 2
    with pytest.raises(ValueError, match="sealed"):
        loop.should_stop()


def test_duplicate_code_is_not_rescored(loop):
    """The same code is not rescored (§15.4)."""
    loop.run(2, verbose=False)
    assert len(loop._seen_code) <= sum(r.n_scored for r in loop.rounds)


def test_seed_puts_the_baseline_in_the_archive(loop):
    """★ Without a seed the loop reads **the report of a different rule**.

    To test "can it read the report and fix that rule" with the hand rule as
    the baseline, it has to start there. The first 20 rounds were run without
    a seed and the experiment was measuring the wrong thing.
    """
    code = ("def score(f, p, hw, w):\n"
            "    return f.traffic_amplification * w[0]\n")
    e = loop.seed(code, [1.0], changes="baseline")
    assert loop.archive.best is not None
    assert loop.archive.best.rule_id == e.rule_id
    assert e.round == -1
    # The same code is not scored again
    assert code.strip() in loop._seen_code


def test_seed_rejects_a_bad_rule(loop):
    with pytest.raises(ValueError, match="the initial rule was refused"):
        loop.seed("def score(f, p, hw, w):\n    return f.nope * w[0]\n", [1.0])


def test_val_blowup_is_reported_not_hidden(loop):
    """★ The archive selects on the **training** score — a rule that
    collapses on validation can become the "best". It really happened (train
    1.164 / val 6.085).

    The selection rule is left as it is (using validation would contaminate
    the holdout), but **an alarm is raised.**
    """
    from kernelrule.core.archive import Elite
    from kernelrule.core.loop import VAL_GAP_ALARM

    loop.run(1, verbose=False)
    # The diagnostic report compiles the best rule, so it must be valid
    # code
    bad = Elite(rule_id="bad",
                code="def score(f, p, hw, w):\n"
                     "    return f.waves * w[0]\n", w=[1.0], regret=1.0,
                mem_objective=1.0, comp_objective=1.0, all_objective=1.0,
                code_len=10, round=0,
                # ★ The default acceptance criterion is rank (D-101).
                #   Without it the archive **refuses** — it does not silently
                #   fall back to regret
                rank_loss=0.5,
                val_regret=1.0 + VAL_GAP_ALARM * 10)
    loop.archive.consider(bad)
    r = loop.run_round()
    assert r.n_val_blowups >= 1, "the validation blowup was not reported"
    assert "blowups" in r.line()


# ---------------------------------------------------------------------------
# ★ Regime balance (§10.1) — it stops training sacrificing a minority
# regime
# ---------------------------------------------------------------------------
def test_regime_balance_flags_a_lopsided_train_split(real_bundle_path):
    """★ `M > 2048` splits training 82%/18% — it must warn.

    Measured: under that composition, evolution worsened overall regret from
    1.177 to 1.390 while the training score improved from 1.201 to 1.118.
    """
    import warnings

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.splits import MIN_REGIME_FRAC, check_balance, split_by_M_range
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                   ok_only=False)
        sh = [p for p in tb.shapes()
              if bool((tb.frame_for(p).align_a == 8).all()
                      and (tb.frame_for(p).align_b == 8).all()
                      and (tb.frame_for(p).align_c == 8).all())]
    sp = split_by_M_range(sh)
    with pytest.warns(UserWarning, match="minority regime"):
        bal = check_balance(sp.train, tb.hw)
    assert not bal.ok
    assert bal.minority()[1] < MIN_REGIME_FRAC
    assert bal.counts["long"] == 9 and bal.counts["short"] == 41


def test_regime_balance_accepts_a_crossing_split(real_bundle_path):
    """A split that crosses the regimes passes (69%/31% measured)."""
    import warnings

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.splits import by_predicate, check_balance
    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                   ok_only=False)
        sh = [p for p in tb.shapes()
              if bool((tb.frame_for(p).align_a == 8).all()
                      and (tb.frame_for(p).align_b == 8).all()
                      and (tb.frame_for(p).align_c == 8).all())]
    sp = by_predicate(sh, lambda p: (p.N, p.K) == (11008, 4096), name="nk")
    bal = check_balance(sp.train, tb.hw)      # it must not warn
    assert bal.ok and bal.counts["long"] == 16


def test_balance_check_is_strictable():
    """With `strict=True` it is an error. It is not waved through silently
    (§26.4)."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.splits import Split, SplitError, check_balance
    from kernelrule.core.types import Hardware, Problem

    hw = Hardware(name="t", arch="sm_86", sm_count=84, smem_per_block=101376,
                  max_threads_per_sm=1536, regs_per_sm=65536,
                  peak_tflops_f16=116.1, bandwidth_gbps=729.7,
                  l2_bytes=6291456)
    tiny = [Problem(1, 4096, 4096)] * 9 + [Problem(8192, 8192, 8192)]
    with pytest.raises(SplitError, match="minority regime"):
        check_balance(Split("train", tuple(tiny)), hw, strict=True)


def test_cell_axes_are_not_all_scores():
    """★ The three axes (D-155): how good · how complex · which band.

    ⚠️ They used to be `mem_objective` · `comp_objective` · `all_objective`
    — **three scores**. A good rule was top-band on all three and landed in
    `(0,0,0)`, while bad rules scattered and took cells of their own; at
    D-154, 4 of 5 held cells were rules at regret 1.29~1.41 and the archive
    stopped moving for five rounds.
    """
    from kernelrule.core.archive import CELL_AXIS_NAMES

    assert set(CELL_AXIS_NAMES) == {"regret", "n_terms", "regime_skew"}


def test_regime_skew_keeps_its_sign():
    """★ `regime_gap` is absolute, so a memory specialist and a compute
    specialist share a cell. The axis is the **signed** one (D-155)."""
    mem_spec = _elite(regret=1.2, short=1.05, long=1.40)
    comp_spec = _elite(regret=1.2, short=1.40, long=1.05)
    assert mem_spec.regime_gap == pytest.approx(comp_spec.regime_gap)
    assert mem_spec.regime_skew < 0 < comp_spec.regime_skew


def test_regime_gap_is_exposed():
    e = _elite(regret=1.1, short=1.05, long=1.40)
    assert e.regime_gap == pytest.approx(0.35)


def test_loop_warns_when_train_has_one_regime(synth_table, tmp_path):
    """If training holds only one regime the cell axes become meaningless —
    it warns."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import Split, SplitSet
    from kernelrule.features import REGISTRY

    fm = FeatureMatrix(synth_table, REGISTRY)
    sh = list(synth_table.shapes())
    import math

    from kernelrule.core.splits import regime_of
    # ★ The cell axes became the roofline (D-144) — the warning looks at
    #   that axis too.
    del math
    short = [p for p in sh
             if regime_of(p, synth_table.hw, axis="roofline") == "mem"]
    if len(short) < 2 or len(short) == len(sh):
        pytest.skip("this test needs both bands present in the synthetic "
                    "grid")
    splits = SplitSet(train=Split("train", tuple(short)),
                      val=Split("val", tuple(p for p in sh
                                              if p not in short)))
    llm = MockLLM("mutate", seed=0, feature_names=fm.feature_names())
    cfg = LoopConfig(run_id="one", n_rules_per_round=2, max_rounds=1,
                     max_evals=20, sandbox_first_seen=False,
                     out_dir=str(tmp_path))
    with pytest.warns(UserWarning, match="only one roofline band"):
        RoundLoop(cfg=cfg, table=synth_table, matrix=fm, splits=splits,
                  llm=llm)


# ---------------------------------------------------------------------------
# Telling LLM transport failures apart from schema refusals (D-43)
# ---------------------------------------------------------------------------
# An HTTP 429 (exhausted credit) was tallied as "144 schema refusals". An
# infrastructure failure **looks like a failure of the model** in the logs —
# the same class as D-39.

@pytest.mark.parametrize("exc,transport", [
    (RuntimeError("status_code: 429, body: You have no credits remaining"), True),
    (RuntimeError("invalid_api_key"), True),
    (ConnectionError("connection reset"), False),
    (ValueError("Exceeded maximum output retries (3)"), False),
    (ValueError("a weight is reused across terms"), False),
])
def test_transport_errors_are_told_apart(exc, transport):
    from kernelrule.core.loop import _is_transport_error
    assert _is_transport_error(exc) is transport


def test_named_transport_exceptions_are_caught():
    from kernelrule.core.loop import _is_transport_error

    class ModelHTTPError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    for cls in (ModelHTTPError, APIConnectionError):
        assert _is_transport_error(cls("an unrelated body"))


def test_round_of_total_transport_failure_stops_the_run():
    """★ A credit problem does not heal by itself. The remaining rounds
    are not burned."""
    from kernelrule.core.loop import LLMUnreachable, RoundResult

    res = RoundResult(round=0, n_proposed=12, n_llm_error=12)
    res.rejections.append(("llm-transport", "429 no credits"))
    assert res.n_llm_error == res.n_proposed
    assert "★LLM err 12" in res.line()
    assert issubclass(LLMUnreachable, RuntimeError)


def test_dump_records_what_it_ran_with(loop, tmp_path):
    """★ Without a record of what it ran with, they cannot be placed side
    by side later (D-51).

    Only 2 of 30 runs had a `config.json` — `dump()` did not write it, and
    those two were written by another script.
    """
    import json

    cfg = json.loads((loop.dump(tmp_path / "d") / "config.json").read_text())
    assert cfg["loop"]["run_id"] == "test"
    assert cfg["split"]["n_train"] == len(loop.splits.train.shapes)
    assert cfg["split"]["n_val"] == len(loop.splits.val.shapes)
    assert cfg["n_features"] > 0
    # For MockLLM at least the class name must survive — it must not be
    # blank
    assert cfg["llm"]


# ---------------------------------------------------------------------------
# D-75 — the Analyst -> FeatureWriter path
# ---------------------------------------------------------------------------
#: The seed rule. Unused features have to remain for the mock Analyst to
#: produce hypotheses.
_SEED_RULE = (("def score(f, p, hw, w):\n"
               "    return np.log2(f.traffic_amplification) * w[0]\n"), [1.0])


def _d75_loop(synth_table, tmp_path, *, cap: int):
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY, FeatureRegistry

    # ★ The registry is **copied**. The loop adds axes to it, so using the
    #   global `REGISTRY` directly would leak into other tests.
    reg = FeatureRegistry("d75")
    for name in REGISTRY.names():
        reg.add(REGISTRY[name])
    fm = FeatureMatrix(synth_table, reg)
    sh = synth_table.shapes()
    splits = SplitSet(train=Split("train", tuple(sh[:-2])),
                      val=Split("val", tuple(sh[-2:])))
    cfg = LoopConfig(run_id="d75", n_rules_per_round=2, max_rounds=1,
                     max_evals=30, seed=0, sandbox_first_seen=False,
                     out_dir=str(tmp_path),
                     max_new_features_per_round=cap)
    llm = MockLLM("mutate", seed=1, feature_names=fm.feature_names())
    return RoundLoop(cfg=cfg, table=synth_table, matrix=fm, splits=splits,
                     llm=llm), reg


def test_the_feature_path_is_on_by_default():
    """★ The default is **3 per round** (D-160).

    It was 0 and every campaign ran with the path shut. Under F1 the Analyst
    asked 30 times in 36 hypotheses and not one reached the FeatureWriter
    (D-159). The number is checked **on LoopConfig**, because that is the
    default the pipeline reads.
    """
    from kernelrule.core.loop import LoopConfig

    assert LoopConfig.max_new_features_per_round == 3


def test_cap_zero_shuts_the_feature_path(synth_table, tmp_path):
    """★ At 0 the path does not exist — the state every run before D-160
    was in. It is what makes the old runs readable."""
    loop, _reg = _d75_loop(synth_table, tmp_path, cap=0)
    assert loop.cfg.max_new_features_per_round == 0
    loop.seed(*_SEED_RULE)
    r = loop.run_round()
    assert r.n_feature_requests == 0 and r.n_features_made == 0
    assert loop.features_made == []
    assert r.llm_calls.get("feature", 0) == 0


def test_analyst_request_reaches_the_feature_writer(synth_table, tmp_path):
    """★ The request is **not thrown away** (D-75).

    `loop.py` was not reading a field that had been filled 303 times across
    33 runs. Whether the path exists is checked by "there was a request" and
    "there was a feature call".
    """
    loop, reg = _d75_loop(synth_table, tmp_path, cap=1)
    n_before = len(reg._items)
    loop.seed(*_SEED_RULE)
    r = loop.run_round()
    assert r.n_feature_requests >= 1, "the request was not read"
    assert r.llm_calls.get("feature", 0) >= 1, "FeatureWriter was not called"
    assert loop.features_made, "there is no record of the attempt"
    row = loop.features_made[0]
    assert row["requirement"], "the requirement sentence was not carried"
    print("D-75 attempt:", row)   # with -s it shows what was built
    if row.get("accepted"):
        assert len(reg._items) == n_before + 1
        assert row["name"] in loop.matrix.feature_names() \
            or row.get("shape_level"), "the column was not built"


def test_feature_writer_never_sees_the_diagnostic_report():
    """★ Condition 1 — the diagnostic report is not given (D-75).

    It blocks the channel by which a feature built inside the loop could be
    fitted to the training shapes. What crosses must be **one requirement
    sentence** only.
    """
    import ast
    import inspect

    from kernelrule.core import loop as loop_mod
    from kernelrule.core.loop import _feature_task

    # (1) **Nothing changes except the requirement sentence.** Written as a
    #     substring check it would catch the guidance text itself ("you see
    #     no table and no cases") — the same mistake as D-73, where the
    #     checker banned wording it had itself allowed.
    a = _feature_task("the CTA parallelism gain split-K creates")
    b = _feature_task("the L2 reuse gain")
    assert "the CTA parallelism gain split-K creates" in a
    assert a.replace("the CTA parallelism gain split-K creates", "<X>") \
        == b.replace("the L2 reuse gain", "<X>"), (
            "something other than the requirement changes")

    # (2) The call site passes **the requirement sentence only**. The prompt
    #     slot is an empty string.
    import textwrap
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(loop_mod.RoundLoop._write_features)))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "complete"]
    assert len(calls) == 1, "there is not exactly one FeatureWriter call"
    c = calls[0]
    assert c.args[0].value == "feature"
    assert c.args[1].value == "", "the report goes into the prompt slot"
    kw = {k.arg for k in c.keywords}
    assert kw == {"condition", "registry", "task"}, (
        f"more is being passed: {kw}")


def test_requirement_reads_the_old_field_name():
    """Both names are read — a run made under either must not silently
    become 0."""
    from kernelrule.core.loop import _requirement_of

    assert _requirement_of({"needs_new_feature": "the L2 reuse gain"}) \
        == "the L2 reuse gain"
    # ★ The name used briefly on 2026-08-28. The 3 runs made then must still
    #   be readable
    assert _requirement_of({"physical_requirement": "the briefly used name"}) \
        == "the briefly used name"
    assert _requirement_of({"needs_new_feature": None}) == ""


def test_over_cap_requests_are_recorded_not_dropped(synth_table, tmp_path):
    """★ A request that hit the cap is **not dropped silently**.

    The cap is needed because of the §21 cache, but without recording the
    overflow "how much was asked for" cannot be measured — and that is
    D-75's main observation.
    """
    from kernelrule.core.loop import RoundResult

    loop, _reg = _d75_loop(synth_table, tmp_path, cap=1)
    res = RoundResult(round=0)
    hyps = [{"id": "H0", "needs_new_feature": "the L2 reuse gain"},
            {"id": "H1", "needs_new_feature": "the absolute CTA count"},
            {"id": "H2", "needs_new_feature": "the split-K parallelism gain"}]
    loop._write_features(hyps, 0, res)
    assert res.n_feature_requests == 3
    assert res.n_feature_over_cap == 2
    over = [x for x in loop.features_made if x.get("over_cap")]
    assert len(over) == 2, "the over-cap requests were not recorded"
    assert {x["hypothesis_id"] for x in over} == {"H1", "H2"}
    assert all(x["requirement"] for x in over), (
        "the requirement sentences did not survive")


def test_analyze_prompt_matches_the_measured_baseline():
    """★ The guidance on the requirement field must be **exactly the
    wording the baseline was measured with** (D-81).

    The 17.9% (the old 6 runs) was measured with the three lines below.
    Adding or removing anything here changes what is being compared — the
    guidance really was expanded once and the rate was pressed down to
    0~5.9% (D-80). The suppressing sentence ("in most rounds this is null")
    is **part of the baseline** too, so it stays.

    If a reason to change it arises, change this test alongside and treat
    **runs from that point on as a new family**.
    """
    from kernelrule.agents.openai_client import load_prompt

    # ★ 2026-09-08 (D-146): the prompt became English. **The requirement
    #   to pin the wording is unchanged** — it is changed only here, and runs
    #   from that point on are treated as a new family.
    baseline = (
        "In `measurable_with`, use **only names from the lists below**. If "
        "you need a\nquantity that is not listed, put its name in "
        "`needs_new_feature` (in most\nrounds this is `null` — there are not "
        "that many physical quantities).")
    txt = load_prompt("role/analyze.md")
    assert baseline in txt, (
        "the guidance on the requirement field differs from the baseline "
        "wording. Leave it, or if you change it, fix this test and treat it "
        "as a new family (D-81)")
    # Have the things added on 2026-08-28 and then reverted come back?
    for gone in ("physical_requirement",
                 "is discarded rather than passed on",
                 "you had better use measurable_with"):
        assert gone not in txt, f"a reverted phrase came back: {gone!r}"


# ---------------------------------------------------------------------------
# §16.1 — Analyst ablation
# ---------------------------------------------------------------------------
def test_analyst_off_makes_no_analyze_call(synth_table, tmp_path):
    """★ With it off, the diagnostic report **is not even built** (§16.1,
    D-89).

    Building it and not giving it would be "there is a diagnosis and it is
    unused", a different condition. Both the call count and the hypothesis
    count must be 0.
    """
    loop, _reg = _d75_loop(synth_table, tmp_path, cap=0)
    loop.cfg.use_analyst = False
    loop.seed(*_SEED_RULE)
    r = loop.run_round()
    assert r.llm_calls.get("analyze", 0) == 0, (
        "the Analyst is off yet it was called")
    assert loop.hypotheses == []
    assert r.n_proposed > 0, "the RuleEditor must keep running"


def test_analyst_on_is_the_default(synth_table, tmp_path):
    """The default is on — every run so far is under that condition."""
    from kernelrule.core.loop import LoopConfig

    assert LoopConfig(run_id="x").use_analyst is True
    loop, _reg = _d75_loop(synth_table, tmp_path, cap=0)
    loop.seed(*_SEED_RULE)
    r = loop.run_round()
    assert r.llm_calls.get("analyze", 0) == 1


def test_borrowed_hypotheses_skip_the_same_seed_index(synth_table, tmp_path):
    """★ Control arm C — giving `abl-B-s1`'s hypotheses to `-s1` is not
    "another run".

    And with an empty pool it **does not silently run without hypotheses**
    (§26.4).
    """
    import json

    pool = tmp_path / "f1pipe-x-s0" / "hypotheses.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text("\n".join(json.dumps(
        {"id": f"H{i}", "round": i // 2, "claim": f"c{i}", "analyst_pass": 1},
        ensure_ascii=False) for i in range(6)))

    loop, _reg = _d75_loop(synth_table, tmp_path, cap=0)
    loop.cfg.use_analyst = False
    loop.cfg.hypothesis_pool = (str(pool),)
    loop.cfg.run_id = "f1pipe-y-s1"     # a different seed number -> used
    got = loop._pool_round(0)
    assert got and all("borrowed_from" in h for h in got)
    assert all("analyst_pass" not in h for h in got)

    loop2, _r2 = _d75_loop(synth_table, tmp_path, cap=0)
    loop2.cfg.use_analyst = False
    loop2.cfg.hypothesis_pool = (str(pool),)
    loop2.cfg.run_id = "f1pipe-y-s0"    # ★ the same seed number -> excluded
    with pytest.raises(ValueError, match="hypothesis pool is empty"):
        loop2._pool_round(0)


def test_borrowed_arm_calls_no_analyst_but_renders_the_section(synth_table,
                                                               tmp_path):
    """★ C does not call the Analyst but **the hypothesis section must
    exist**.

    Making A (no hypothesis) and C (someone else's hypothesis) identical
    down to the prompt structure makes it impossible to tell what differs —
    C's only difference must be **the provenance of the sentences**.
    """
    import json

    pool = tmp_path / "f1pipe-x-s2" / "hypotheses.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text("\n".join(json.dumps(
        {"id": f"H{i}", "round": 0, "claim": f"c{i}", "analyst_pass": 1},
        ensure_ascii=False) for i in range(3)))

    loop, _reg = _d75_loop(synth_table, tmp_path, cap=0)
    loop.cfg.use_analyst = False
    loop.cfg.hypothesis_pool = (str(pool),)
    loop.seed(*_SEED_RULE)
    r = loop.run_round()
    assert r.llm_calls.get("analyze", 0) == 0, (
        "this is C, yet the Analyst was called")
    assert loop.hypotheses, "the borrowed hypotheses were not recorded"
    assert all(h.get("analyst_pass") == 0 for h in loop.hypotheses)


# ---------------------------------------------------------------------------
# D-95 — parallel scoring and fitting
# ---------------------------------------------------------------------------
def _parallel_pair(synth_table, tmp_path, workers: int, *, seed_rule=None):
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY

    fm = FeatureMatrix(synth_table, REGISTRY)
    sh = synth_table.shapes()
    splits = SplitSet(train=Split("train", tuple(sh[:-2])),
                      val=Split("val", tuple(sh[-2:])))
    cfg = LoopConfig(run_id=f"par{workers}", n_rules_per_round=6, max_rounds=1,
                     max_evals=30, seed=0, sandbox_first_seen=False,
                     out_dir=str(tmp_path), n_workers=workers)
    llm = MockLLM("mutate", seed=3, feature_names=fm.feature_names())
    lp = RoundLoop(cfg=cfg, table=synth_table, matrix=fm, splits=splits,
                   llm=llm)
    lp.seed(*(seed_rule or _SEED_RULE))
    r = lp.run_round()
    elites = sorted(lp.archive.cells.values(), key=lambda e: e.rule_id)
    return r, [(e.rule_id, e.code, tuple(e.w), e.regret, e.mem_objective,
                e.comp_objective, e.val_regret) for e in elites]


#: ★ A rule with more than `FITTER_SWITCH_DIM` weights — the **CMA** path
#: (D-163). `_SEED_RULE` has one weight, so the pair test above only ever
#: exercised Nelder-Mead: measured, `fitter_for` saw len(w0) 1 and 2 and
#: nothing else.
_WIDE_SEED = (
    ("def score(f, p, hw, w):\n"
     "    s = np.log2(f.traffic_amplification) * w[0]\n"
     + "".join(f"    s = s + f.{n} * w[{i}]\n" for i, n in enumerate(
         ("waves", "edge_waste", "tail_waste", "reg_pressure",
          "smem_pressure", "spill_magnitude", "log_grid_tiles",
          "log_mainloop_iters", "split_k_cost", "sm_idle_cost",
          "log_dram_traffic"), start=1))
     + "    return s\n"), [1.0] * 12)


def test_the_parallel_pair_covers_the_cma_path(synth_table, tmp_path):
    """★ D-163 — the pair test has to reach the fitter the runs actually use.

    Every rule of the last 12-round run was fitted with CMA (64/64), and the
    pair test only ever saw 1 and 2 weights — Nelder-Mead. A test that
    passes on a path the campaign never takes proves nothing about the path
    it does take.
    """
    from kernelrule.rules.checks import FITTER_SWITCH_DIM, fitter_for

    assert len(_WIDE_SEED[1]) > FITTER_SWITCH_DIM
    assert fitter_for(len(_WIDE_SEED[1]))["fit_method"] == "cma"


def test_parallel_matches_sequential_on_the_cma_path(synth_table, tmp_path):
    """★ D-163 — the same values from a **CMA-dimension** seed.

    Turning the workers on must not change what a campaign produces. The
    fitter is chosen by `fitter_for(len(w0))` in both paths; this is the
    check that says so by behaviour rather than by reading the code.
    """
    seq_r, seq = _parallel_pair(synth_table, tmp_path / "wa", 0,
                                seed_rule=_WIDE_SEED)
    par_r, par = _parallel_pair(synth_table, tmp_path / "wb", 3,
                                seed_rule=_WIDE_SEED)
    assert seq, "the sequential run produced nothing — the test is meaningless"
    assert seq == par, "the parallel result differs from the sequential one"
    for f in ("n_proposed", "n_scored", "n_accepted", "n_fit_moved",
              "n_rejected_static", "n_rejected_fit", "n_cells"):
        assert getattr(seq_r, f) == getattr(par_r, f), f


def test_parallel_matches_sequential(synth_table, tmp_path):
    """★ Parallel must produce **exactly the same values** as sequential
    (D-95).

    A difference means there is hidden state inside `fit_weights`. Being
    faster is a gain only when the result is the same; otherwise the
    condition changed.

    ⚠️ It looks at the **values**, not the time — "it is fast" is not used as
    a performance metric (principle 29: in D-89 the arm with the Analyst off
    was fast because it did no work).
    """
    seq_r, seq = _parallel_pair(synth_table, tmp_path / "a", 0)
    par_r, par = _parallel_pair(synth_table, tmp_path / "b", 3)
    assert seq, ("the sequential run produced nothing — the test is "
                 "meaningless")
    assert seq == par, "the parallel result differs from the sequential one"
    for f in ("n_proposed", "n_scored", "n_accepted", "n_fit_moved",
              "n_rejected_static", "n_rejected_fit", "n_cells"):
        assert getattr(seq_r, f) == getattr(par_r, f), f


def test_workers_default_is_parallel():
    """★ D-163 — the default is **6 workers**, not sequential.

    ⚠️ It was 0 and every campaign ran that way. 74% of a round's wall clock
    of the D-162 run was the fitter, and it is the same computation either
    way: `test_parallel_matches_sequential_on_the_cma_path` is what says the
    values do not change. 6 = `n_rules_per_round`, so one round's candidates
    are fitted in a single wave.
    """
    from kernelrule.core.loop import LoopConfig

    cfg = LoopConfig(run_id="x")
    assert cfg.n_workers == 6
    assert cfg.n_workers == cfg.n_rules_per_round


def test_worker_does_not_do_the_sandbox(synth_table):
    """★ The static checks and the sandbox are done by **the parent**
    (D-95).

    Calling `run_isolated` inside a worker gives a nested process spawn.
    Those two are 3% of a round, so they are not worth parallelising anyway.
    """
    import ast
    import inspect
    import textwrap

    from kernelrule.core import loop as loop_mod

    src = textwrap.dedent(inspect.getsource(loop_mod._fit_and_score))
    names = {ast.unparse(n) for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call)}
    assert not any("run_isolated" in n for n in names)
    assert not any("check_rule" in n for n in names)


# ---------------------------------------------------------------------------
# D-96 — recording the cross lineage
# ---------------------------------------------------------------------------
def test_observation_keys_never_reach_the_prompt():
    """★ Request keys starting with `_` do not go to the LLM (D-96).

    The parent code was put into the request for observation. If that leaked
    into the prompt, `cross` would receive "two parents + two more parent
    codes", and then **the observation device changes the condition.**
    Unblocked here, it is the kind of contamination that leaks silently.
    """
    from kernelrule.core.loop import RoundLoop

    seen = []

    class _Spy:
        def complete(self, role, prompt, **kw):
            seen.append(kw)
            raise RuntimeError("stop")

    loop = RoundLoop.__new__(RoundLoop)
    loop.llm = _Spy()
    loop._call_optimizers([{"prompt": "p", "parent_kind": "cross",
                            "_codes": ["A", "B"], "analyst": False}])
    assert seen and all(not k.startswith("_") for k in seen[0]), seen[0]


def test_cross_lineage_counts_mixing_not_just_presence():
    """It is `mixed` only when the child used features unique to **each of
    the two parents**."""
    import numpy as np

    from kernelrule.core.loop import RoundLoop

    loop = RoundLoop.__new__(RoundLoop)
    loop.cross_lineage = []
    loop._feats = ["waves", "tail_waste", "bytes_per_flop"]
    loop._shape_vals = []
    # ★ `_record_cross` looks at the budget. Without it the `except`
    #   swallows the error and the feature set passes through empty — that
    #   really happened.
    loop._budget = 8
    loop._limits = None
    # ★ The per-axis minima for the exponent guard (D-112). Without them,
    #   `feats`' `except Exception` swallowed an `AttributeError` and the
    #   set came out empty **again** — which is why that except was narrowed
    #   to `SyntaxError`.
    loop._fmins = {}

    def _code(*fs):
        body = " + ".join(f"f.{x} * w[{i}]" for i, x in enumerate(fs))
        return f"def score(f, p, hw, w):\n    return {body}\n"

    a, b = _code("waves"), _code("tail_waste")
    loop._record_cross(0, [a, b], _code("waves", "tail_waste"))
    loop._record_cross(0, [a, b], a)               # one side copied whole
    loop._record_cross(0, [a, a], _code("waves"))  # nothing to mix
    m, c, same = loop.cross_lineage
    assert m["mixed"] and m["mixable"] and not m["copied"]
    assert not c["mixed"] and c["copied"]
    assert not same["mixable"], (
        "with no unique terms it must drop out of the denominator")
    assert np is not None


# ---------------------------------------------------------------------------
# D-104 — the objective switch
# ---------------------------------------------------------------------------
def test_switch_requires_matching_start():
    """★ If the starting objective differs from the left-hand side of the
    switch, it **stops**.

    It must not silently become a different experiment (§26.4).
    """
    import pytest

    from kernelrule.core.loop import RoundLoop, RoundResult

    loop = RoundLoop.__new__(RoundLoop)
    loop.cfg = type("C", (), {"objective_switch": "rank->regret",
                              "switch_window": 3,
                              "switch_min_improve": 0.01})()
    loop._objective = "regret"          # <- mismatched
    loop._switched = False
    loop.rounds = [RoundResult(round=i, best_rank_loss=1.0) for i in range(5)]
    with pytest.raises(ValueError, match="the starting objective"):
        loop._maybe_switch()


def test_switch_fires_only_when_improvement_stalls():
    """It switches only when the improvement is below the threshold. And
    **only once.**"""
    from kernelrule.core.loop import RoundLoop, RoundResult

    calls = []
    loop = RoundLoop.__new__(RoundLoop)
    loop.cfg = type("C", (), {"objective_switch": "rank->regret",
                              "switch_window": 3,
                              "switch_min_improve": 0.01})()
    loop._objective, loop._switched = "rank", False
    loop._switch_to = lambda dst: (calls.append(dst),
                                   setattr(loop, "_switched", True),
                                   setattr(loop, "_objective", dst))

    def rounds(vals):
        return [RoundResult(round=i, best_rank_loss=v)
                for i, v in enumerate(vals)]

    loop.rounds = rounds([1.00, 0.90, 0.80, 0.70])   # 30% improvement
    assert loop._maybe_switch() is False and not calls
    loop.rounds = rounds([1.00, 0.999, 0.998, 0.997])  # 0.3% improvement
    assert loop._maybe_switch() is True and calls == ["regret"]
    # ★ It does not switch twice — twice would make two "when do we
    #   switch" decisions
    loop._objective = "rank"
    assert loop._maybe_switch() is False and calls == ["regret"]


def test_switch_needs_enough_rounds():
    """With fewer rounds than the window it makes no verdict."""
    from kernelrule.core.loop import RoundLoop, RoundResult

    loop = RoundLoop.__new__(RoundLoop)
    loop.cfg = type("C", (), {"objective_switch": "rank->regret",
                              "switch_window": 3,
                              "switch_min_improve": 0.01})()
    loop._objective, loop._switched = "rank", False
    loop.rounds = [RoundResult(round=i, best_rank_loss=1.0) for i in range(3)]
    assert loop._maybe_switch() is False
