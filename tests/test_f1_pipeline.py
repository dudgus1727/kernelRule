"""F1~F3 파이프라인의 배관 (§30.9).

**실험 결과가 아니라 배관을 시험한다.** 조건이 정하는 것은 하나뿐이다 —
어느 레지스트리가 세 단계 전부에 들어가는가. F1 에서 사람이 쓴 24개가
하나라도 새면 "LLM 이 피처를 만들 수 있는가" 라는 질문 자체가 무너진다.
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
    """★ 사람이 쓴 것이 **하나도** 없어야 한다."""
    r = pipe._base_registry(cond)
    assert not r._items, f"{cond} 출발 레지스트리가 비어 있지 않다: {sorted(r._items)}"


def test_f2_is_the_public_knowledge_five(pipe):
    """★ F2 = **공개 지식 다섯** (D-128 개명 전 이름은 `F1-K`).

    옛 `F2`(원시 물리량 5개, `F2_BASE`)는 실행이 0회라 삭제했다. 이름이
    같으므로 **무엇이 F2 인지**를 시험으로 고정한다 — 달라지면 여기서 잡는다.
    """
    from kernelrule.features.known5 import KNOWN5

    r = pipe._base_registry("F2")
    assert sorted(r._items) == sorted(KNOWN5._items)
    assert len(r._items) == 5
    # 이름은 24개 안에 있지만 **정리본**이다 — 표 관측을 뺀 docstring (§12.3)
    assert set(r._items) <= set(REGISTRY._items)
    assert sorted(r._items) != sorted(REGISTRY._items)


def test_no_alias_for_the_old_condition_names(pipe):
    """★ alias 를 두지 않는다 (D-128). 옛 이름은 **에러**여야 한다."""
    for old in ("F0", "F1-K", "F1K"):   # 전부 D-128 이 없앤 이름이다
        with pytest.raises(ValueError, match="알 수 없는 조건"):
            pipe._base_registry(old)


def test_f3_is_the_human_24(pipe):
    r = pipe._base_registry("F3")
    assert sorted(r._items) == sorted(REGISTRY._items)


def test_unknown_condition_is_an_error(pipe):
    with pytest.raises(ValueError, match="알 수 없는 조건"):
        pipe._base_registry("F9")


def test_mock_llm_gets_only_the_given_registry(pipe, monkeypatch):
    """`_make_llm` 이 프롬프트용 이름을 **레지스트리에서** 뽑는가."""
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
    """씨앗을 만들 피처가 없으면 **조용히 사람 24개로 안 떨어진다**."""
    from kernelrule.agents.mock import MockLLM

    with pytest.raises(ValueError, match="조용히"):
        MockLLM("mutate", feature_names=[]).complete("rule_writer", "")


def test_regime_split_does_not_need_the_registry():
    """★ 체제는 (형상, 하드웨어)의 성질이다 — 피처 목록의 성질이 아니다.

    전에는 `info.log_sol_ms` 를 읽어서 F1 레지스트리로는 루프도 리포트도
    통째로 죽었다. `regime_of` 로 모았다 (원칙 2).
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    bad = []
    for rel in ("kernelrule/core/loop.py", "kernelrule/report/diagnostic.py"):
        tree = ast.parse((root / rel).read_text(), filename=rel)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and node.attr in ("log_sol_ms", "is_memory_bound")
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ("info", "f", "feats")):
                bad.append(f"  {rel}:{node.lineno} {node.value.id}.{node.attr}")
    assert not bad, (
        "체제 판정이 레지스트리 피처를 읽는다 — F1 에서 죽는다 "
        "(§30.9). `core.splits.regime_of` 를 써라:\n" + "\n".join(bad))


def test_stage1_loader_redetects_shape_level(pipe):
    """★ `_load_stage1` 이 `table` 을 넘겨야 `shape_level` 이 다시 판정된다.

    안 넘기면 기록된 값(대부분 없음 = False)을 쓰고, **형상 수준 피처가
    0개인 채로** 2·3단계가 돈다. 실제로 F1 2단계를 그 상태로 한 번
    돌렸다 (D-67).
    """
    import inspect

    sig = inspect.signature(pipe._load_stage1)
    assert "table" in sig.parameters, "_load_stage1 이 표를 안 받는다"
    src = inspect.getsource(pipe._load_stage1)
    assert "table=table" in src, "load_generated 에 표를 안 넘긴다"


# ---------------------------------------------------------------------------
# ★ 4-3 — 씨앗 선택이 홀드아웃을 안 봤다는 **증거를 남긴다**
#
#   절차로는 지켜지고 있다 (`score_only` 가 홀드아웃을 안 돌려준다).
#   나중에 "정말 안 봤는가" 를 물으면 답할 것이 있어야 한다 (D-50).
# ---------------------------------------------------------------------------
def test_chosen_json_records_what_was_seen(pipe):
    import inspect

    src = inspect.getsource(pipe.stage2)
    for key in ("selected_on", "holdout_seen_at_selection", "unsealed"):
        assert f'"{key}"' in src, f"chosen.json 에 {key} 를 안 적는다"
    assert '"holdout_seen_at_selection": False' in src


def test_score_only_does_not_return_holdout():
    """★ 씨앗 선택이 홀드아웃을 볼 **경로 자체가 없다** (원칙 6)."""
    import inspect

    from kernelrule.core.loop import RoundLoop

    src = inspect.getsource(RoundLoop.score_only)
    assert "return float(e.regret)" in src, "반환이 바뀌었다"
    assert "val_regret" not in src, "홀드아웃을 돌려준다"


def test_config_records_seal_state(pipe):
    import inspect

    from kernelrule.core.loop import RoundLoop

    assert '"unsealed"' in inspect.getsource(RoundLoop.dump)
    assert '"unsealed": is_unsealed()' in inspect.getsource(pipe.main)
