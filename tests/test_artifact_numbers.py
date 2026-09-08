"""★ 4-1 — do the numbers in a `.md` diverge from the `.json` (§30.14)?

Until now the numbers were **transcribed by hand.** A transcription error is
not caught. The representative numbers in `conclusion.md` especially — that
is the document the next session reads.

```
docs/artifacts/<name>.json   what the machine wrote (representative)
docs/artifacts/<name>.md     what a human reads
```

**A value in the `.json` has to appear in the `.md` as it is, to three
decimal places.**

⚠️ 2026-09-08 (D-146): the asserted strings were translated together with
`conclusion.md` and `conclusion.json`. The numbers are unchanged; the Korean
originals are at commit `ee53b4d`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parents[1] / "docs" / "artifacts"

#: `(json path, md path)`. When a pair is added, it goes here.
PAIRS = [("conclusion.json", "conclusion.md"),
         ("fitter-dim16.json", "fitter-dim16.md")]


def _numbers(obj, path=""):
    """It flattens the numbers in the JSON into `(path, value)`. A key
    starting with `_` is metadata."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            yield from _numbers(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def _fmt(v: float) -> list[str]:
    """The forms this value could take in the document."""
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
        missing.append(f"  {path} = {v}   (candidate forms {_fmt(v)})")
    assert not missing, (
        f"a value of {js} is not in {md} — it diverged while being "
        f"transcribed by hand (§30.14):\n"
        + "\n".join(missing))


def test_canonical_json_states_its_procedure():
    """★ A number with no procedure attached must not be compared
    (principle 4)."""
    d = json.loads((ART / "conclusion.json").read_text())
    for key in ("canonical", "canonical_alt", "by_regime", "f1_vs_human"):
        blob = json.dumps(d[key], ensure_ascii=False)
        assert "_procedure" in blob or "_label" in blob, (
            f"{key} has no procedure description")
    # The document has to know that the two procedures give different values
    assert d["canonical"]["ours_geomean"] != d["canonical_alt"]["median"]
    assert "Do not mix it" in d["canonical_alt"]["_procedure"]


def test_dev_table_warning_is_present():
    """The mark saying the dev-table numbers are not reported externally."""
    d = json.loads((ART / "conclusion.json").read_text())
    assert "Do not report these numbers externally" in d["_warning"]
