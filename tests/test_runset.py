"""★ 묶음의 **조건 동일성** (D-120).

```
D-113   arch_prompt 가 config.json 에 있었고 안 읽었다
D-119   씨앗 source 가 chosen.json 에 있었고 안 읽었다
★ 원칙 39 가 생긴 지 하루 만에 두 번째다 — 그래서 검사로 만들었다
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
    """`fit_method=None` 이면 **키 자체를 안 쓴다** — 옛 실행의 모양이다."""
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
    """★ D-119 가 정확히 이것이었다."""
    for i in range(3):
        _mk(tmp_path, f"a-s{i}")
    for i in range(3):
        _mk(tmp_path, f"b-s{i}", source="human_guided", code="other")
    runs = [f"a-s{i}" for i in range(3)] + [f"b-s{i}" for i in range(3)]
    with pytest.raises(RunSetError, match="seed_source"):
        assert_same_condition(runs, root=tmp_path, label="(c) 재생성")


def test_mixed_objective_fails(tmp_path):
    _mk(tmp_path, "c-s0")
    _mk(tmp_path, "c-s1", objective="rank")
    with pytest.raises(RunSetError, match="objective"):
        assert_same_condition(["c-s0", "c-s1"], root=tmp_path)


def test_mixed_seed_code_fails_even_with_same_source(tmp_path):
    """출처 이름이 같아도 **코드가 다르면** 다른 씨앗이다."""
    # ★ 캠페인이 둘이다 — 씨앗은 캠페인 단위이므로 이렇게 해야 갈린다
    _mk(tmp_path, "d0-s0")
    _mk(tmp_path, "d1-s0", code="def score(): pass")
    with pytest.raises(RunSetError, match="seed_sha"):
        assert_same_condition(["d0-s0", "d1-s0"], root=tmp_path)


def test_missing_config_is_an_error_not_a_pass(tmp_path):
    """★ 없는 것을 통과로 처리하면 검사가 조용히 0 이 된다."""
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
    """★ 진짜 실행으로 확인한다 — 되돌려서 잡는지 (원칙 38)."""
    from pathlib import Path

    ok = [f"f1pipe-F3-5090sigma-s{i}" for i in range(3)]
    if not all((Path("runs") / r / "config.json").exists() for r in ok):
        pytest.skip("5090 실행 없음")
    assert_same_condition(ok, label="(c)")
    mixed = ok + [f"f1pipe-F3-5090sigma-b-s{i}" for i in range(3)]
    with pytest.raises(RunSetError, match="human_guided"):
        assert_same_condition(mixed, label="기록된 (c) 여섯")


def test_run_condition_reads_the_hardware_prompt_identity():
    """D-113 의 자리 — hw 가 조건으로 잡히는가."""
    from pathlib import Path

    r = "f1pipe-F3-5090sigma-hw-s0"
    if not (Path("runs") / r / "config.json").exists():
        pytest.skip("실행 없음")
    assert run_condition(r)["hw"] is not None


# ---------------------------------------------------------------------------
# ★ 적합기 조건 (D-123)
# ---------------------------------------------------------------------------
def test_old_run_without_fit_method_reads_as_nelder_mead(tmp_path):
    """★ 옛 실행에는 키가 없다. **없음 = nelder-mead** 다 — 그때 코드가
    그것뿐이었다. 봐주는 것이 아니라 사실을 채우는 것이다.
    """
    _mk(tmp_path, "old-s0")
    assert run_condition("old-s0", tmp_path)["fit_method"] == "nelder-mead"
    assert run_condition("old-s0", tmp_path)["fit_restarts"] == 4


def test_old_run_groups_with_an_explicit_nelder_mead_run(tmp_path):
    """옛 실행과 `fit_method="nelder-mead"` 를 적은 실행은 **같은 조건**이다."""
    _mk(tmp_path, "camp-s0")
    _mk(tmp_path, "camp-s1", fit_method="nelder-mead", fit_restarts=4)
    got = assert_same_condition(["camp-s0", "camp-s1"], root=tmp_path)
    assert got["fit_method"] == "nelder-mead"


def test_mixed_fitter_fails(tmp_path):
    """★ CMA 실행과 옛 실행을 한 묶음에 넣으면 실패한다 (D-123).

    §3 에서 예산 8 팔과 16 팔을 같은 적합기로 돌려야 하고, 옛 기준선
    (1.0762)은 다른 적합기다 — 나란히 놓으면 여기서 걸린다.
    """
    _mk(tmp_path, "camp-s0", fit_method="cma", fit_restarts=1)
    _mk(tmp_path, "camp-s1")
    with pytest.raises(RunSetError, match="fit_method"):
        assert_same_condition(["camp-s0", "camp-s1"], root=tmp_path)


def test_same_method_but_different_restarts_fails(tmp_path):
    """재시작 수도 조건이다 — CMA 는 1, Nelder-Mead 는 4 다."""
    _mk(tmp_path, "camp-s0", fit_method="cma", fit_restarts=1)
    _mk(tmp_path, "camp-s1", fit_method="cma", fit_restarts=4)
    with pytest.raises(RunSetError, match="fit_restarts"):
        assert_same_condition(["camp-s0", "camp-s1"], root=tmp_path)
