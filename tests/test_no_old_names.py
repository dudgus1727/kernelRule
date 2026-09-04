"""★ 개명한 이름이 **코드와 현재 문서에 남아 있지 않은가** (D-128).

```
F1-K / F1K / f1k     -> F2
F0 / 옛 F2           -> 삭제
physics_seeded       -> human_guided
architect-tryNN      -> rule_writer-tryNN
rule_budget          -> parameters
```

**alias 를 두지 않는다** (원칙 2) — 두 이름이 공존하면 달라진다.

## 예외는 **정정 이력**이다

옛 이름을 지우면 "왜 값이 다른가" 를 못 되짚는다 (문서 규칙 2).

```
파일 단위   역사 기록(decisions/design/glossary)과 산출물·실행 디렉토리
줄 단위     ★ 그 줄이 **`D-128` 을 인용**하면 통과한다
            = "옛 이름을 쓰려면 어느 결정이 바꿨는지 같이 적어라"
```
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 옛 이름들. 값은 (정규식, 새 이름).
OLD = {
    "F1-K": "F2", "F1K": "F2", "f1k": "f2",
    "physics_seeded": "human_guided",
    "architect-try": "rule_writer-try",
    "rule_budget": "parameters",
    "literal_budget_message": "literal_parameter_message",
}

#: ★ 정정 이력을 담은 파일 — 옛 이름이 **남아 있어야** 한다.
HISTORY = {
    "docs/decisions.md",            # 시간순 기록. 옛 이름이 사실이다
    "docs/design.md",               # 정정 상자가 옛 서술 위에 쌓여 있다
    "docs/glossary.md",             # 이름 대응표가 여기 있다
    "docs/pending_fixes.md",
    "kernelrule/rules/human_guided.py",   # 개명 이력을 docstring 에 적었다
    "kernelrule/features/known5.py",      # 같은 이유
    "kernelrule/core/runset.py",          # 옛 키를 읽는 자리
    "tests/test_no_old_names.py",         # 이 파일 자신
}
#: 산출물(`docs/artifacts/*.md`, `*.json`)은 **그때의 기록**이라 전부 예외다.
#: 실행 디렉토리(`runs/`)도 마찬가지 — 변환한 값 옆에 `_renamed` 를 남겼다.
HISTORY_DIRS = ("docs/artifacts/", "runs/")


def _files():
    for p in list(ROOT.glob("*.md")) + list((ROOT / "docs").rglob("*.md")) \
            + list((ROOT / "kernelrule").rglob("*.py")) \
            + list((ROOT / "experiments").rglob("*.py")) \
            + list((ROOT / "tests").rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if rel in HISTORY or rel.startswith(HISTORY_DIRS):
            continue
        yield rel, p


@pytest.mark.parametrize("old", sorted(OLD))
def test_old_name_is_gone(old):
    bad = [f"{rel}:{i}" for rel, p in _files()
           for i, line in enumerate(p.read_text().splitlines(), 1)
           # ★ 옛 이름은 **개명한 결정을 같이 인용할 때만** 남을 수 있다
           if old in line and "D-128" not in line]
    assert not bad, (
        f"옛 이름 {old!r} 이 남아 있다 (-> {OLD[old]}): {bad[:8]}\n"
        "★ alias 를 두지 마라. 정정 이력이면 HISTORY 목록에 넣어라.")


def test_history_files_still_carry_the_old_names():
    """★ 예외 목록이 **비어 있지 않은가** — 이력을 지우면 그것도 잘못이다.

    원칙 38 의 자리다: 예외 목록만 두고 아무것도 안 남아 있으면 이 시험은
    "통과" 하지만 이력은 사라진 것이다.
    """
    txt = (ROOT / "docs" / "decisions.md").read_text()
    for old in ("F1-K", "physics_seeded", "rule_budget"):
        assert old in txt, f"{old} 의 정정 이력이 decisions.md 에서 사라졌다"
