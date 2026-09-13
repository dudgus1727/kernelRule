"""★ Are the renamed names **gone from the code and the current documents**
(D-128)?

```
F1-K / F1K / f1k     -> F2
F0 / the old F2      -> deleted
physics_seeded       -> human_guided
architect-tryNN      -> rule_writer-tryNN
rule_budget          -> parameters
```

**No alias is kept** (principle 2) — two coexisting names diverge.

## The exception is **the correction history**

Deleting the old name makes "why is the value different" untraceable
(documentation rule 2).

```
per file   the historical records (decisions/design/glossary) and the
           artefact and run directories
per line   ★ the line passes if it **cites `D-128`**
           = "if you use an old name, write down which decision changed it"
```
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The old names. The value is (the regex, the new name).
OLD = {
    "F1-K": "F2", "F1K": "F2", "f1k": "f2",
    "physics_seeded": "human_guided",
    "architect-try": "rule_writer-try",
    "rule_budget": "parameters",
    "literal_budget_message": "literal_parameter_message",
    # ★ 2026-09-13 (D-170 §4)
    "known5": "known7", "KNOWN5": "KNOWN7",
}

#: ★ Which decision a line has to cite to be allowed to keep an old name.
#: It was `D-128` for every entry; the D-170 rename needed its own, and a
#: single hardcoded id would have made the new old-name unenforceable.
DECISION = {"known5": "D-170", "KNOWN5": "D-170"}

#: ★ The files that carry the correction history — the old names **have to
#: stay** in them.
HISTORY = {
    "docs/decisions.md",            # the time-ordered record. The old name is
                                    # the fact
    "docs/design.md",               # correction boxes piled on the old text
    "docs/glossary.md",             # the name mapping table is here
    "docs/pending_fixes.md",
    "kernelrule/rules/human_guided.py",   # the rename history is in the
                                          # docstring
    "kernelrule/features/known7.py",      # the same reason
    "kernelrule/core/runset.py",          # where the old key is read
    "tests/test_no_old_names.py",         # this file itself
}
#: The artefacts (`docs/artifacts/*.md`, `*.json`) are **the record of the
#: time**, so they are all exceptions.
#: The run directories (`runs/`) likewise — `_renamed` was left beside the
#: converted value.
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
           # ★ An old name may remain **only when the renaming decision is
           #   cited alongside it**
           if old in line and DECISION.get(old, "D-128") not in line]
    assert not bad, (
        f"the old name {old!r} is still there (-> {OLD[old]}, cite "
        f"{DECISION.get(old, 'D-128')}): {bad[:8]}\n"
        "★ Do not keep an alias. If it is correction history, put it in the "
        "HISTORY list.")


def test_history_files_still_carry_the_old_names():
    """★ Is the exception list **not empty** — deleting the history is a fault
    too.

    This is where principle 38 lives: if only the exception list remains and
    nothing is left in it, this test "passes" but the history is gone.
    """
    txt = (ROOT / "docs" / "decisions.md").read_text()
    for old in ("F1-K", "physics_seeded", "rule_budget"):
        assert old in txt, (f"the correction history of {old} disappeared "
                            f"from decisions.md")
