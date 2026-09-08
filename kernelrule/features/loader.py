"""Reload generated features across runs (§11.4).

## Why this is needed

Features created by F1 exist only in `runs/featwriter-*/proposals.jsonl`. To
use them in the rule loop they must be registered again, and **which ones you
put in is an experimental condition**, so that choice must live in code.

    condition A   the 24 human-written ones
    condition B   24 + N generated new axes

★ Rediscovered ones (Spearman 1.000 against an existing feature) are **not
included.** They trip the §8.4 duplicate check, and even when they do not,
giving the same information twice blurs "is the new axis useful".
"""

from __future__ import annotations

import json
from pathlib import Path

from kernelrule.features import Feature, FeatureRegistry

__all__ = ["load_generated", "extended_registry"]


def load_generated(path: str | Path, *, table, only: set[str] | None = None,
                   exclude: set[str] | None = None) -> list[Feature]:
    """Revive the accepted features from `proposals.jsonl`.

    `only` / `exclude` make the experimental condition **explicit** — with a
    default of "everything" you later cannot tell which set was used.

    ★ `table` is a **required argument** — it is what re-derives
    `shape_level` (§30.12). Why the recorded value is not trusted: that
    verdict was reached **on that bundle**, and a feature that references
    `cfg` but happened to be constant there may be config-dependent on
    another table.

    **There is no default.** With one, two call sites omitted it, and one of
    them was the stage-2 path, so RuleWriter ran with **zero shape-level
    features** (D-67). If re-deriving really is unnecessary, pass
    `table=None` **explicitly** — then the recorded value is used.
    """

    from kernelrule.features.generated import compile_feature

    out: list[Feature] = []
    seen: set[str] = set()
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("accepted") or not r.get("code"):
                continue
            name, fn = compile_feature(r["code"], known=frozenset(seen))
            if only is not None and name not in only:
                continue
            if exclude and name in exclude:
                continue
            seen.add(name)
            rng = r.get("expected_range") or (0.0, 1.0)
            if isinstance(rng, str):
                rng = tuple(float(x) for x in
                            rng.strip("()[] ").split(",")[:2])
            out.append(Feature(
                name=name, fn=fn, unit=str(r.get("unit", "dimensionless")),
                shape_level=bool(r.get("shape_level", False)),
                expected_range=(float(rng[0]), float(rng[1])),
                direction=str(r.get("direction", "higher_is_worse")),
                doc=str(r.get("rationale", ""))[:200],
                physical_meaning=str(r.get("rationale", "")),
                source=r["code"], code_hash=str(abs(hash(r["code"])))))
    if table is not None:
        from dataclasses import replace

        from kernelrule.features.generated import detect_shape_level
        redone = []
        for f in out:
            is_shape, _ = detect_shape_level(replace(f, shape_level=False),
                                             table)
            redone.append(replace(f, shape_level=is_shape))
        return redone
    return out


def extended_registry(base: FeatureRegistry, features: list[Feature], *,
                      name: str = "extended") -> FeatureRegistry:
    """Existing + generated. **Does not touch the original** — that would
    break the condition comparison."""
    reg = FeatureRegistry(name)
    for n in sorted(base._items):
        reg.add(base[n])
    for f in features:
        reg.add(f)
    return reg
