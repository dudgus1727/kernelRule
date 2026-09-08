"""★ The **condition sameness** of a run set (D-120).

```
D-113   arch_prompt was in config.json and was not read
D-119   the seed source was in chosen.json and was not read
★ it is the second time within a day of principle 39 being written — so it
  was made into a check
```
"""
from __future__ import annotations

import json

import pytest

from kernelrule.core.runset import (
    KEYS,
    RunSetError,
    assert_same_condition,
    condition_report,
    run_condition,
)


def _mk(root, run, *, source="rule_writer-try00", code="def score(): ...",
        objective="regret", budget=8, model="m", campaign=None,
        fit_method=None, fit_restarts=None):
    """With `fit_method=None` **the key itself is not written** — that is the
    shape of an old run."""
    d = root / run
    (d).mkdir(parents=True, exist_ok=True)
    loop = {"objective": objective, "rank_top_k": 100,
            "rank_lambda": 0.0, "feature_condition": "F3"}
    if fit_method is not None:
        loop["fit_method"] = fit_method
    if fit_restarts is not None:
        loop["fit_restarts"] = fit_restarts
    (d / "config.json").write_text(json.dumps({
        "loop": loop,
        "split": {"kind": "nk11008"},
        "rule_constraints": {"budget": budget},
        "llm": {"model": model, "arch_prompt": None,
                "hw_text": {"sha256": "abc"}}}))
    camp = root / (campaign or run.rsplit("-s", 1)[0]) / "stage2-rule-writer"
    camp.mkdir(parents=True, exist_ok=True)
    (camp / "chosen.json").write_text(
        json.dumps({"source": source, "code": code}))


def test_same_condition_passes(tmp_path):
    for i in range(3):
        _mk(tmp_path, f"camp-s{i}")
    got = assert_same_condition([f"camp-s{i}" for i in range(3)],
                                root=tmp_path)
    assert got["seed_source"] == "rule_writer-try00"
    assert set(got) == set(KEYS)


def test_mixed_seed_source_fails(tmp_path):
    """★ D-119 was exactly this."""
    for i in range(3):
        _mk(tmp_path, f"a-s{i}")
    for i in range(3):
        _mk(tmp_path, f"b-s{i}", source="human_guided", code="other")
    runs = [f"a-s{i}" for i in range(3)] + [f"b-s{i}" for i in range(3)]
    with pytest.raises(RunSetError, match="seed_source"):
        assert_same_condition(runs, root=tmp_path, label="(c) regrow")


def test_mixed_objective_fails(tmp_path):
    _mk(tmp_path, "c-s0")
    _mk(tmp_path, "c-s1", objective="rank")
    with pytest.raises(RunSetError, match="objective"):
        assert_same_condition(["c-s0", "c-s1"], root=tmp_path)


def test_mixed_seed_code_fails_even_with_same_source(tmp_path):
    """Even with the same source name, **a different code** is a different
    seed."""
    # ★ There are two campaigns — the seed is per campaign, so this is what
    #   makes it differ
    _mk(tmp_path, "d0-s0")
    _mk(tmp_path, "d1-s0", code="def score(): pass")
    with pytest.raises(RunSetError, match="seed_sha"):
        assert_same_condition(["d0-s0", "d1-s0"], root=tmp_path)


def test_missing_config_is_an_error_not_a_pass(tmp_path):
    """★ Treating what is absent as a pass makes the check silently 0."""
    _mk(tmp_path, "e-s0")
    with pytest.raises(RunSetError, match="config.json"):
        assert_same_condition(["e-s0", "e-s9"], root=tmp_path)


def test_single_run_needs_no_check(tmp_path):
    _mk(tmp_path, "f-s0")
    assert assert_same_condition(["f-s0"], root=tmp_path) == {}


def test_report_lists_observed_values(tmp_path):
    _mk(tmp_path, "g-s0")
    _mk(tmp_path, "g-s1", objective="rank")
    r = condition_report(["g-s0", "g-s1"], root=tmp_path)
    assert r["objective"] == ["rank", "regret"]
    assert r["split_kind"] == ["nk11008"]


def test_real_c_arms_are_clean_and_the_recorded_six_are_not():
    """★ Confirmed with a real run — does it catch it when turned back
    (principle 38)."""
    from pathlib import Path

    ok = [f"x-hwold-5090sigma-s{i}" for i in range(3)]
    if not all((Path("runs") / r / "config.json").exists() for r in ok):
        pytest.skip("no 5090 run")
    assert_same_condition(ok, label="(c)")
    mixed = ok + [f"x-hand-5090sigma-b-s{i}" for i in range(3)]
    with pytest.raises(RunSetError, match="human_guided"):
        assert_same_condition(mixed, label="the recorded (c), six of them")


def test_run_condition_reads_the_hardware_prompt_identity():
    """The D-113 spot — is hw caught as a condition."""
    from pathlib import Path

    r = "x-hwmid-5090sigma-hw-s0"
    if not (Path("runs") / r / "config.json").exists():
        pytest.skip("no run")
    assert run_condition(r)["hw"] is not None


# ---------------------------------------------------------------------------
# ★ The fitter condition (D-123)
# ---------------------------------------------------------------------------
def test_old_run_without_fit_method_reads_as_nelder_mead(tmp_path):
    """★ An old run has no key. **Absent = nelder-mead** — that was all the
    code had at the time. It is not being lenient, it is filling in a fact.
    """
    _mk(tmp_path, "old-s0")
    assert run_condition("old-s0", tmp_path)["fit_method"] == "nelder-mead"
    assert run_condition("old-s0", tmp_path)["fit_restarts"] == 4


def test_old_run_groups_with_an_explicit_nelder_mead_run(tmp_path):
    """An old run and a run that wrote `fit_method="nelder-mead"` are **the
    same condition**."""
    _mk(tmp_path, "camp-s0")
    _mk(tmp_path, "camp-s1", fit_method="nelder-mead", fit_restarts=4)
    got = assert_same_condition(["camp-s0", "camp-s1"], root=tmp_path)
    assert got["fit_method"] == "nelder-mead"


def test_mixed_fitter_fails(tmp_path):
    """★ Putting a CMA run and an old run in one set fails (D-123).

    In §3 the budget-8 and budget-16 arms have to run with the same fitter,
    and the old baseline (1.0762) is a different fitter — putting them side
    by side is caught here.
    """
    _mk(tmp_path, "camp-s0", fit_method="cma", fit_restarts=1)
    _mk(tmp_path, "camp-s1")
    with pytest.raises(RunSetError, match="fit_method"):
        assert_same_condition(["camp-s0", "camp-s1"], root=tmp_path)


def test_same_method_but_different_restarts_fails(tmp_path):
    """The number of restarts is a condition too — CMA is 1, Nelder-Mead
    is 4."""
    _mk(tmp_path, "camp-s0", fit_method="cma", fit_restarts=1)
    _mk(tmp_path, "camp-s1", fit_method="cma", fit_restarts=4)
    with pytest.raises(RunSetError, match="fit_restarts"):
        assert_same_condition(["camp-s0", "camp-s1"], root=tmp_path)
