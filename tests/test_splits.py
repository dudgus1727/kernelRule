"""분할 (§10.2) — 역할이 타입에 있는 것과 최종 분할 봉인."""
from __future__ import annotations

import pytest

from kernelrule.core.types import Problem


# ---------------------------------------------------------------------------
# ★ 4-2 — 최종 분할 봉인 (§30.15)
#
#   "끝에 딱 한 번" 은 **의도이지 강제가 아니었다.** `splits.test.shapes` 를
#   그냥 읽으면 됐다. 코드로 막고, 연 실행은 기록에 남긴다.
# ---------------------------------------------------------------------------
def test_test_split_is_sealed(monkeypatch):
    from kernelrule.core.splits import UNSEAL_ENV, Split, SplitError

    monkeypatch.delenv(UNSEAL_ENV, raising=False)
    p = Problem(M=128, N=128, K=128)
    te = Split("test", (p,))
    with pytest.raises(SplitError, match="봉인"):
        _ = te.shapes


def test_train_and_val_are_not_sealed(monkeypatch):
    from kernelrule.core.splits import UNSEAL_ENV, Split

    monkeypatch.delenv(UNSEAL_ENV, raising=False)
    p = Problem(M=128, N=128, K=128)
    assert len(Split("train", (p,)).shapes) == 1
    assert len(Split("val", (p,)).shapes) == 1


def test_size_is_visible_without_unsealing(monkeypatch):
    """★ 분할이 **존재한다**는 것과 그 안을 보는 것은 다르다."""
    from kernelrule.core.splits import UNSEAL_ENV, Split

    monkeypatch.delenv(UNSEAL_ENV, raising=False)
    te = Split("test", (Problem(M=1, N=1, K=1),))
    assert len(te) == 1          # 봉인과 무관
    assert te.role == "test"


def test_unseal_flag_opens_it_and_is_reportable(monkeypatch):
    from kernelrule.core.splits import UNSEAL_ENV, Split, is_unsealed

    p = Problem(M=128, N=128, K=128)
    te = Split("test", (p,))
    monkeypatch.setenv(UNSEAL_ENV, "1")
    assert is_unsealed()
    assert len(te.shapes) == 1
    # 빈 값이나 "0" 은 봉인 유지 — 실수로 열리지 않게
    for off in ("", "0", "false"):
        monkeypatch.setenv(UNSEAL_ENV, off)
        assert not is_unsealed(), off


def test_nothing_reads_the_private_field(monkeypatch):
    """`_shapes` 를 직접 읽으면 봉인을 우회한다 — 라이브러리에 없어야 한다."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted([*(root / "kernelrule").rglob("*.py"),
                     *(root / "experiments").glob("*.py")]):
        rel = f.relative_to(root).as_posix()
        if rel == "kernelrule/core/splits.py":
            continue            # 구현 자신
        for node in ast.walk(ast.parse(f.read_text(), filename=rel)):
            if isinstance(node, ast.Attribute) and node.attr == "_shapes":
                bad.append(f"  {rel}:{node.lineno}")
    assert not bad, ("`_shapes` 를 직접 읽어 봉인을 우회한다 (§30.15):\n"
                     + "\n".join(bad))
