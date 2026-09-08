"""★ It moves the role names of the old artefacts in one go (D-93). 0 LLM
calls.

    python3 experiments/rename_roles.py --check     # count only
    python3 experiments/rename_roles.py --apply     # move them

## Why no alias is kept

Making a compatibility path that reads the old name means **two names coexist
and they diverge.** It would be the eighth after `is_reference` / `top_k` /
`DEFAULT_MODEL` / `REGISTRY` / `load_generated` / `approx_equal` / the budget
constants (principle 2). So **the data is moved instead of patching the
reading side.**

## ★ The tallies are checked after the move

The **sum** of the calls per role does not change when a name changes. If it
does not match, there is a place that was not moved — and passing over that
silently makes `cost.md` wrong.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

#: old name -> new name. **Written in one place only.**
#: ⚠️ This table is **old name -> new name**. Once, this file got caught in its
#: own bulk substitution and became the identity map (`{"rule_writer":
#: "rule_writer"}`). Then "0 old names left" becomes false —
#: `test_rename_map_is_not_identity` pins that down.
RENAME = {"architect": "rule_writer", "optimize": "rule_editor"}


def _role_counts(root: Path) -> Counter:
    """The number of calls per role in `llm_calls/*.json`. The names are
    normalised before counting."""
    c: Counter = Counter()
    for f in root.glob("*/llm_calls/*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        r = str(d.get("role", ""))
        c[RENAME.get(r, r)] += 1
    return c


#: The artefact directory names. The role name is baked into the path, so a
#: new run would use the old name too — and that is exactly "two names
#: coexisting".
DIR_RENAME = {"stage2-architect": "stage2-rule-writer"}


def _move_dirs(root: Path, apply: bool) -> int:
    n = 0
    for old, new in DIR_RENAME.items():
        for d in sorted(root.glob(f"*/{old}")):
            n += 1
            if apply:
                d.rename(d.parent / new)
    return n


def _walk(root: Path):
    yield from root.glob("*/llm_calls/*.json")
    yield from root.glob("*/config.json")
    yield from root.glob("*/rounds.jsonl")
    yield from root.glob("*/stage2-*/summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)

    before = _role_counts(root)
    print("=" * 68)
    print(f"moving the role names  {root}   "
          f"{'(count only)' if a.check else '(apply)'}")
    print("=" * 68)
    print(f"  the tally before (normalised): {dict(before)}")

    n_files = 0
    hits: Counter = Counter()
    for f in _walk(root):
        txt = f.read_text()
        found = {k: txt.count(f'"{k}"') for k in RENAME
                 if f'"{k}"' in txt}
        if not found:
            continue
        n_files += 1
        hits.update(found)
        if a.apply:
            for old, new in RENAME.items():
                txt = txt.replace(f'"{old}"', f'"{new}"')
            f.write_text(txt)
    print(f"  files containing an old name: {n_files}   {dict(hits)}")
    n_dirs = _move_dirs(root, a.apply)
    print(f"  directories with an old name: {n_dirs} {DIR_RENAME}")

    if a.apply:
        after = _role_counts(root)
        # ★ The tally check. Only the name changed, so **the sums must match.**
        ok = before == after
        print(f"  the tally after:   {dict(after)}")
        print("  ★ the tally check: "
              + ("matches" if ok else
                 "★ mismatch — there is a place that was not moved"))
        if not ok:
            raise SystemExit(1)
        left = sum(1 for f in _walk(root)
                   for k in RENAME if f'"{k}"' in f.read_text())
        print(f"  old names left: {left}")
        if left:
            raise SystemExit(1)
    print("\n  ★ no alias is kept — two coexisting names diverge "
          "(principle 2)")


if __name__ == "__main__":
    main()
