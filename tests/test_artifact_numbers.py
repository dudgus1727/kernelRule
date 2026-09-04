"""★ 4-1 — `.md` 의 숫자가 `.json` 과 달라지지 않는가 (§30.14).

지금까지 수치를 **손으로 옮겨 적었다.** 전사 오류가 안 잡힌다. 특히
`conclusion.md` 의 대표값 수치가 그렇다 — 그 문서를 다음 세션이 읽는다.

```
docs/artifacts/<name>.json   기계가 쓴 것 (대표값)
docs/artifacts/<name>.md     사람이 읽는 것
```

**`.json` 의 값이 `.md` 안에 소수 셋째 자리까지 그대로 있어야 한다.**
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parents[1] / "docs" / "artifacts"

#: `(json 경로, md 경로)`. 짝이 늘면 여기 추가한다.
PAIRS = [("conclusion.json", "conclusion.md"),
         ("fitter-dim16.json", "fitter-dim16.md")]


def _numbers(obj, path=""):
    """JSON 안의 숫자를 `(경로, 값)` 으로 펼친다. `_` 로 시작하는 키는 메타."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            yield from _numbers(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def _fmt(v: float) -> list[str]:
    """이 값이 문서에 나타날 수 있는 표기들."""
    out = {f"{v:.4f}", f"{v:.3f}", f"{v:g}"}
    if v == int(v):
        out.add(str(int(v)))
    if 0.0 < v < 1.0:
        out.add(f"{v:.3f}")
        out.add(f"{v * 100:g}%")
        out.add(f"{v:.1%}")
    if abs(v) >= 1000:
        out.add(f"{int(v):,}")
    return sorted(out)


@pytest.mark.parametrize(("js", "md"), PAIRS)
def test_md_numbers_match_json(js, md):
    data = json.loads((ART / js).read_text())
    body = (ART / md).read_text()
    missing = []
    for path, v in _numbers(data):
        if any(t in body for t in _fmt(v)):
            continue
        missing.append(f"  {path} = {v}   (표기 후보 {_fmt(v)})")
    assert not missing, (
        f"{js} 의 값이 {md} 에 없다 — 손으로 옮겨 적다 달라졌다 (§30.14):\n"
        + "\n".join(missing))


def test_canonical_json_states_its_procedure():
    """★ 절차가 안 붙은 숫자는 비교하면 안 된다 (원칙 4)."""
    d = json.loads((ART / "conclusion.json").read_text())
    for key in ("canonical", "canonical_alt", "by_regime", "f1_vs_human"):
        blob = json.dumps(d[key], ensure_ascii=False)
        assert "_procedure" in blob or "_label" in blob, (
            f"{key} 에 절차 설명이 없다")
    # 두 절차의 값이 다르다는 것을 문서가 알고 있어야 한다
    assert d["canonical"]["ours_geomean"] != d["canonical_alt"]["median"]
    assert "섞지 마라" in d["canonical_alt"]["_procedure"]


def test_dev_table_warning_is_present():
    """dev 표 수치를 대외 보고하지 않는다는 표시."""
    d = json.loads((ART / "conclusion.json").read_text())
    assert "대외 보고하지 마라" in d["_warning"]
