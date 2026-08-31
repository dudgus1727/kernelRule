"""역할 이름은 하나다 — alias 를 두지 않는다 (D-93).

## 왜 alias 를 안 두나

옛 이름을 읽는 호환 경로를 만들면 **두 이름이 공존하고 그것이 갈린다.**
`is_reference` / `top_k` / `DEFAULT_MODEL` / `REGISTRY` /
`load_generated` / `approx_equal` / 예산 상수에 이은 여덟 번째가 된다
(원칙 2).

alias 를 안 두는 대신 **이 시험이 막는다.** 옛 이름을 직접 쓰면 실패한다.

## 무엇이 예외인가

`decisions.md` 의 "당시 RuleWriter 라 불렀다" 같은 **역사 서술**은 그대로
둔다 (문서 규칙 2 — 틀린 값을 지우지 않는다). 이 시험은 **코드**만 본다.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 더는 쓰지 않는 역할 이름. 문자열 리터럴로 나오면 실패한다.
#: ⚠️ 이 튜플은 **옛 이름**이다. 일괄 치환에 두 번 걸렸다 —
#: `rename_roles.py` 의 `RENAME` 과 여기. 둘 다 자기 자신을 치환해
#: 검사가 통째로 무력해졌다. 아래 `test_rename_map_is_not_identity` 와
#: 이 파일의 제외 목록이 그것을 잡는다.
_OLD_ROLES = ("architect", "optimize")
#: ★ `critique` 는 **루프 밖 역할**이라 `experiments/critic.py` 가
#: `register_role` 로 쓴다. 금지 대상은 `kernelrule/` 안에 남는 것이다 —
#: 아래 `test_no_critique_role_in_the_agent_package` 가 그것을 본다.


def _py_files():
    yield from (ROOT / "kernelrule").rglob("*.py")
    yield from (ROOT / "experiments").glob("*.py")
    yield from (ROOT / "tests").glob("*.py")


def test_no_old_role_string_literals():
    """★ 코드에 옛 역할 이름이 **문자열로** 남으면 실패한다."""
    bad: list[str] = []
    for f in _py_files():
        rel = f.relative_to(ROOT).as_posix()
        if rel in ("experiments/rename_roles.py", "tests/test_role_names.py"):
            continue                    # 이전 스크립트와 이 시험은 옛 이름을 안다
        try:
            tree = ast.parse(f.read_text(), filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            v = node.value
            if v in _OLD_ROLES or (v == "critique"
                                   and rel.startswith("kernelrule/")):
                bad.append(f"  {rel}:{node.lineno}  {v!r}")
    assert not bad, (
        "옛 역할 이름이 코드에 남아 있다. alias 를 두지 않으므로 이것이 "
        "유일한 방어선이다 (D-93):\n" + "\n".join(bad))


def test_no_critique_role_in_the_agent_package():
    """★ Critic 은 **루프 밖**이다 (D-92).

    루프가 부르지 않는 역할을 `kernelrule/agents/` 에 남겨 두면 "언젠가
    켤 것" 으로 읽히고 조건 목록과 ablation 표에 계속 끌려다닌다.
    실험 스크립트가 `register_role` 로 자기 프롬프트와 자기 스키마를
    들고 온다.
    """
    hits = [f.relative_to(ROOT).as_posix()
            for f in (ROOT / "kernelrule").rglob("*")
            if f.is_file() and f.suffix in (".py", ".md")
            and "critique" in f.read_text(errors="ignore")]
    assert not hits, f"kernelrule/ 에 critique 가 남아 있다: {hits}"
    assert not (ROOT / "kernelrule/agents/prompts/role/critique.md").exists()


def test_loop_roles_are_exactly_four():
    """루프가 부르는 역할은 넷이다 — 목록이 늘면 여기서 걸린다."""
    from kernelrule.agents.openai_client import (
        _EDITS_RULES,
        _NEEDS_HW,
        _WRITES_RULES,
    )

    assert set(_NEEDS_HW) == {"rule_writer"}
    assert set(_WRITES_RULES) == {"rule_editor", "rule_writer"}
    assert set(_EDITS_RULES) == {"rule_editor"}


@pytest.mark.parametrize("name", ["rule_writer", "rule_editor", "analyze",
                                  "feature", "categorize"])
def test_register_role_refuses_to_shadow_a_loop_role(name):
    """등록 역할이 루프 역할을 덮으면 **조용히 다른 것이 돈다**."""
    import os

    _ = os.environ.setdefault("OPENAI_API_KEY", "test-key")
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import REGISTRY

    llm = OpenAILLM(LLMConfig(),
                    feature_names=REGISTRY.names(shape_level=False),
                    shape_values=REGISTRY.names(shape_level=True),
                    registry=REGISTRY, cache=False)
    with pytest.raises(ValueError, match="루프 역할"):
        llm.register_role(name, instructions="x", output_type=dict)


def test_rename_map_is_not_identity():
    """★ 이전 스크립트가 **자기 자신**의 일괄 치환에 걸린 적이 있다.

    `RENAME` 이 항등 사상이 되면 "남은 옛 이름 0" 이 거짓이 된다 —
    아무것도 안 옮기고 통과한다.
    """
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    from rename_roles import RENAME

    assert RENAME, "이전 표가 비었다"
    for old, new in RENAME.items():
        assert old != new, f"{old!r} 이 자기 자신으로 매핑된다"
        assert old in _OLD_ROLES
