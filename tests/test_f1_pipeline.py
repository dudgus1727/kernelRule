"""The plumbing of the F1~F3 pipeline (§30.9).

**It tests the plumbing, not the experimental results.** The condition
decides one thing only — which registry goes into all three stages. If even
one of the 24 a human wrote leaks into F1, the question "can the LLM build
features" collapses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import kernelrule.features.physical  # noqa: F401, E402
from kernelrule.features import REGISTRY  # noqa: E402


@pytest.fixture(scope="module")
def pipe():
    import f1_pipeline
    return f1_pipeline


@pytest.mark.parametrize("cond", ["F1"])
def test_f0_f1_start_from_an_empty_registry(pipe, cond):
    """★ **Not one** of the human-written features may be there."""
    r = pipe._base_registry(cond)
    assert not r._items, (
        f"the {cond} starting registry is not empty: {sorted(r._items)}")


def test_f2_is_the_public_knowledge_seven(pipe):
    """★ F2 = **the seven public facts**
    (before the D-128 rename its name was `F1-K`).

    The old `F2` (5 raw physical quantities, `F2_BASE`) had 0 runs and was
    deleted. The name is the same, so **what F2 is** is pinned by a test —
    a change is caught here.

    ⚠️ 2026-09-13 (D-170 §4): **it was five until this date.** `log_min_dim`
    and `log_flops` were added so that F2 has more than one axis a rule can
    branch on. The condition name did not change and the content did, so a
    number from before and a number from after are not on the same
    condition — which is exactly what this test exists to make visible.
    ⛔ The two are shape-level, so `_base_registry("F2")` now hands the
    pipeline three branchable axes rather than one.
    """
    from kernelrule.features.known7 import KNOWN7

    r = pipe._base_registry("F2")
    assert sorted(r._items) == sorted(KNOWN7._items)
    assert len(r._items) == 7
    assert {"log_min_dim", "log_flops"} <= set(r._items)
    assert sum(1 for n in r._items if r[n].shape_level) == 3
    # The names are among the 24, but it is the **cleaned-up version** —
    # docstrings with the table observations removed (§12.3)
    assert set(r._items) <= set(REGISTRY._items)
    assert sorted(r._items) != sorted(REGISTRY._items)


def test_no_alias_for_the_old_condition_names(pipe):
    """★ There are no aliases (D-128). An old name must be an **error**."""
    for old in ("F0", "F1-K", "F1K"):   # all names D-128 removed
        with pytest.raises(ValueError, match="unknown condition"):
            pipe._base_registry(old)


def test_f3_is_the_human_24(pipe):
    r = pipe._base_registry("F3")
    assert sorted(r._items) == sorted(REGISTRY._items)


def test_unknown_condition_is_an_error(pipe):
    with pytest.raises(ValueError, match="unknown condition"):
        pipe._base_registry("F9")


def test_mock_llm_gets_only_the_given_registry(pipe, monkeypatch):
    """Does `_make_llm` take the names for the prompt **from the
    registry**?"""
    import argparse

    from kernelrule.agents.openai_client import Budget
    from kernelrule.features import Feature, FeatureRegistry

    reg = FeatureRegistry("only-mine")
    reg.add(Feature(name="mock_axis", fn=lambda p, hw, cfg: 0.0,
                    unit="dimensionless", expected_range=(0.0, 1.0),
                    direction="higher_is_worse", code_hash="h"))
    a = argparse.Namespace(dry_run=True, seed=0, model="m")
    llm = pipe._make_llm(a, registry=reg, budget=Budget())
    assert llm.features == ["mock_axis"]
    assert not set(llm.features) & set(REGISTRY._items)


def test_architect_mock_refuses_an_empty_feature_list():
    """With no features to build a seed from, it **does not silently fall
    back to the human 24**."""
    from kernelrule.agents.mock import MockLLM

    with pytest.raises(ValueError, match="silently fall back"):
        MockLLM("mutate", feature_names=[]).complete("rule_writer", "")


def test_regime_split_does_not_need_the_registry():
    """★ A regime is a property of (shape, hardware) — not of the feature
    list.

    It used to read the SOL value off `info`, so with the F1 registry both
    the loop and the report died entirely. It was gathered into `regime_of`
    (principle 2).
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    bad = []
    for rel in ("kernelrule/core/loop.py", "kernelrule/report/diagnostic.py"):
        tree = ast.parse((root / rel).read_text(), filename=rel)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and node.attr in ("log_" + "sol_ms", "is_memory_bound")
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ("info", "f", "feats")):
                bad.append(f"  {rel}:{node.lineno} {node.value.id}.{node.attr}")
    assert not bad, (
        "the regime verdict reads a registry feature — it dies under F1 "
        "(§30.9). Use `core.splits.regime_of`:\n" + "\n".join(bad))


def test_stage1_loader_redetects_shape_level(pipe):
    """★ `_load_stage1` must pass `table` for `shape_level` to be
    re-judged.

    Without it, the recorded value is used (mostly absent = False) and
    stages 2 and 3 run **with 0 shape-level features**. F1 stage 2 really
    was run once in that state (D-67).
    """
    import inspect

    sig = inspect.signature(pipe._load_stage1)
    assert "table" in sig.parameters, "_load_stage1 does not take the table"
    src = inspect.getsource(pipe._load_stage1)
    assert "table=table" in src, (
        "the table is not passed to load_generated")


# ---------------------------------------------------------------------------
# ★ 4-3 — **leaving evidence** that seed selection did not look at the
# holdout
#
#   Procedurally it is held (`score_only` does not return the holdout). When
#   someone asks later "did it really not look", there has to be an answer
#   (D-50).
# ---------------------------------------------------------------------------
def test_chosen_json_records_what_was_seen(pipe):
    import inspect

    src = inspect.getsource(pipe.stage2)
    for key in ("selected_on", "holdout_seen_at_selection", "unsealed"):
        assert f'"{key}"' in src, (
            f"chosen.json does not record {key}")
    assert '"holdout_seen_at_selection": False' in src


def test_score_only_does_not_return_holdout():
    """★ There is **no path at all** by which seed selection could see the
    holdout (principle 6)."""
    import inspect

    from kernelrule.core.loop import RoundLoop

    src = inspect.getsource(RoundLoop.score_only)
    assert "return float(e.regret)" in src, "the return changed"
    assert "val_regret" not in src, "it returns the holdout"


def test_config_records_seal_state(pipe):
    import inspect

    from kernelrule.core.loop import RoundLoop

    # ★ The place that builds the config moved to `_config_dict` (D-133 —
    #   the trace's first line uses the same thing). The subject of the check
    #   follows it.
    assert '"unsealed"' in inspect.getsource(RoundLoop._config_dict)
    assert '"unsealed": is_unsealed()' in inspect.getsource(pipe.main)
