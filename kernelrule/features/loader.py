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

__all__ = ["load_generated", "extended_registry", "base_registry",
           "run_registry", "registry_spec", "registry_from_spec",
           "RegistryUnavailable"]


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

        from kernelrule.core.matrix import FeatureMatrix
        from kernelrule.features import FeatureRegistry
        from kernelrule.features.generated import detect_shape_level

        # ★ 2026-09-11 (D-161): **one pass, not one per feature.**
        #   `detect_shape_level` builds a matrix when it is not handed one,
        #   and since D-160 that matrix covers every shape — so loading 16
        #   axes cost 265.9s on the A6000 table, paid again at every stage
        #   that loads them. The verdict is unchanged: the same columns over
        #   the same shapes, computed together.
        #   Every candidate goes in as **config level** — the column per
        #   config is exactly what the verdict reads.
        probe_reg = FeatureRegistry("probe-shape-level")
        for f in out:
            probe_reg.add(replace(f, shape_level=False))
        probe = FeatureMatrix(table, probe_reg) if out else None
        redone = []
        for f in out:
            is_shape, _ = detect_shape_level(replace(f, shape_level=False),
                                             table, matrix=probe)
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


# ----------------------------------------------------------------------
# ★ 2026-09-11 (D-166 §M): the registry a **finished run** actually scored
# with, rebuilt from its artefacts.
#
# `transfer_generated.py` (D-165) needed this to move an F2 rule to another
# table, and built it inline. `export_rules.py` / `verify_rules.py` did not
# have it at all — they loaded the 24 human axes and any F2 rule fell into
# the "unregistered value" bucket (D-156), which is the bucket for rules
# that are **not verifiable at all**. Same trap, third site (principle 23),
# so the loader lives here and the call sites share it.
# ----------------------------------------------------------------------


def base_registry(condition: str, *,
                  human: FeatureRegistry | None = None) -> FeatureRegistry:
    """The **starting registry** the condition decides. `F1` is empty.

    ★ There are three conditions (D-128). A 0 -> 5 -> 24 ladder, with no
    aliases.

    ⚠️ `F3` is the human-written list, and **the library does not reach for
    the global registry** (§30.9) — `human=` has to be handed in from the
    call site. Moving this function into the library without that argument
    is exactly what `test_library_never_imports_the_global_registry`
    caught (D-166 §M): the body came from `experiments/`, where looking at
    `REGISTRY` is allowed, and carried the habit across the line.
    """
    if condition == "F1":
        return FeatureRegistry("F1-empty")
    if condition == "F2":
        # ★ The five public facts (§30.17). Not `physical.py`'s originals
        #   but the **cleaned-up version with the table observations
        #   removed** — the original docstrings carry measurement results
        #   such as "in this table a spilling kernel was optimal 0 times"
        #   (§12.3).
        from kernelrule.features.known5 import KNOWN5
        r = FeatureRegistry("F2-known5")
        for n in sorted(KNOWN5._items):
            r.add(KNOWN5[n])
        return r
    if condition == "F3":
        if human is None:
            raise ValueError(
                "condition F3 is the human-written axis list, and this "
                "library function does not reach for the global registry "
                "(§30.9). Pass `human=REGISTRY` from the call site.")
        r = FeatureRegistry("F3-human24")
        for n in sorted(human._items):
            r.add(human[n])
        return r
    raise ValueError(f"unknown condition: {condition!r}. F1/F2/F3")


def run_registry(run: str, *, table, seed: int = 0, root: str | Path = "runs",
                 condition: str | None = None,
                 human: FeatureRegistry | None = None
                 ) -> tuple[FeatureRegistry, dict[str, str]]:
    """The axis list of the run `run`, rebuilt on `table`.

    Three layers, in the order the run itself built them:

    ```
    the condition   F1 nothing · F2 known5 · F3 the 24 human ones
    stage 1         what the FeatureWriter made before the loop
                      runs/<run>/stage1-features/proposals.jsonl
    the loop        what it made during the loop (D-160)
                      runs/<run>-s<seed>/features.jsonl
    ```

    The third layer is the one that gets forgotten — it is in a different
    directory from the other two, and a rule that uses it raises
    `AttributeError: unregistered feature` when only the first two are
    loaded (D-165 §1, and again D-166 §M).

    Returns `(registry, origin)`; `origin[name]` says which layer it came
    from, so a report can say *where* an axis was made.

    ⚠️ `shape_level` is re-derived **on this table** for every layer
    (§30.12). That verdict is made from values, so the same formula can be
    shape level here and config level on another GPU — which is exactly
    what a transfer has to see.
    """

    from kernelrule.features import generated as gen

    root = Path(root)
    d = root / run
    if condition is None:
        cfg = d / "config.json"
        if not cfg.exists():
            raise FileNotFoundError(
                f"{cfg} does not exist, so the condition of {run!r} is "
                f"unknown. Pass `condition=` explicitly — it is not guessed "
                f"(§26.4).")
        condition = json.loads(cfg.read_text()).get("condition")

    base = base_registry(str(condition), human=human)
    pending: list[Feature] = []
    origin: dict[str, str] = {}
    for n in sorted(base._items):
        pending.append(base[n])
        origin[n] = f"{condition} base"

    # ★ `table=None` **on purpose**: the recorded verdict is discarded and
    #   every layer is re-derived together below, in one pass. Paying for a
    #   per-layer pass here would be the cost D-161 removed.
    seen = {f.name for f in pending}
    stage1 = d / "stage1-features" / "proposals.jsonl"
    if stage1.exists():
        for f in load_generated(stage1, table=None, exclude=seen):
            if f.name in seen:
                continue
            seen.add(f.name)
            pending.append(f)
            origin[f.name] = "stage1"
        revive = stage1.parent / "revalidated.jsonl"
        if revive.exists():
            for f in load_generated(revive, table=None, exclude=seen):
                if f.name in seen:
                    continue
                seen.add(f.name)
                pending.append(f)
                origin[f.name] = "stage1 (revalidated)"

    loop = root / f"{run}-s{seed}" / "features.jsonl"
    if not loop.exists():
        loop = d / "stage3-evolution" / f"s{seed}" / "features.jsonl"
    if loop.exists():
        for line in loop.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if not e.get("accepted") or e.get("name") in seen:
                continue
            name, fn = gen.compile_feature(e["code"],
                                           known=frozenset(seen))
            seen.add(name)
            rng = (e.get("range") or {}).get("declared") or (0.0, 1.0)
            pending.append(Feature(
                name=name, fn=fn, unit=str(e.get("unit", "dimensionless")),
                expected_range=(float(rng[0]), float(rng[1])),
                direction=str(e.get("direction", "higher_is_worse")),
                doc=str(e.get("requirement", ""))[:200],
                source=e["code"], code_hash=str(abs(hash(e["code"])))))
            origin[name] = f"loop r{e.get('round', '?')}"

    return _rederive(pending, table,
                     f"{run}-on-{getattr(table, 'bundle', '?')}"), origin


def _rederive(pending: list[Feature], table, name: str) -> FeatureRegistry:
    """`shape_level` re-derived for every axis in **one probe matrix**
    (§30.12, D-161).

    Every candidate goes in as **config level** — the column per config is
    exactly what the verdict reads. One matrix, not one per axis: per axis
    it was a full-table build, 384s on the 4090 and 724s on the H100.
    """
    from dataclasses import replace

    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import generated as gen

    probe_reg = FeatureRegistry("probe-shape-level")
    for f in pending:
        probe_reg.add(replace(f, shape_level=False))
    probe = FeatureMatrix(table, probe_reg) if pending else None
    reg = FeatureRegistry(name)
    for f in pending:
        is_shape, _ = gen.detect_shape_level(
            replace(f, shape_level=False), table, matrix=probe)
        reg.add(replace(f, shape_level=is_shape))
    return reg


class RegistryUnavailable(RuntimeError):
    """The recorded axis list **cannot be rebuilt here**.

    ★ It is not the same thing as "this rule uses a value the library
    retired" (D-156). That one is unverifiable by design and its recorded
    number stands; this one means a rule that *is* in scope for
    verification could not be checked — a failure (D-166 §M③).
    """


def registry_spec(reg: FeatureRegistry, origin: dict[str, str], *,
                  run: str | None = None,
                  condition: str | None = None) -> dict:
    """The axis list, **written down** so it can be rebuilt without `runs/`.

    `runs/` is in `.gitignore`, so a rule that uses generated axes could
    not be re-scored from a clean checkout — the same reason the rule and
    its fitted weights are committed at all (`experiments/export_rules.py`).
    The library axes go in **by name** (they are in the repository); the
    generated ones carry their source.
    """
    import hashlib

    axes = []
    for n in sorted(reg._items):
        f = reg[n]
        layer = origin.get(n, "?")
        generated = not layer.endswith("base") and layer != "human"
        axes.append({
            "name": n, "layer": layer,
            "source": f.source if generated else None,
            "unit": f.unit, "direction": f.direction,
            "expected_range": [float(f.expected_range[0]),
                               float(f.expected_range[1])],
            # ★ recorded, but **re-derived** on load (§30.12) — it is a
            #   verdict about the table, not a property of the formula.
            "shape_level_at_export": bool(f.shape_level)})
    blob = "\n".join(f"{a['name']}\x00{a['source'] or ''}" for a in axes)
    return {"run": run, "condition": condition, "n": len(axes),
            "hash": hashlib.sha256(blob.encode()).hexdigest()[:12],
            "axes": axes}


def registry_from_spec(spec: dict, *, table,
                       human: FeatureRegistry | None = None
                       ) -> FeatureRegistry:
    """Rebuild what `registry_spec` wrote. Raises `RegistryUnavailable` if
    a library axis it names is gone.

    ⚠️ A spec with no `condition` is the pre-condition era — its axes are
    the human-written ones, and `human=` must be handed in (§30.9).
    """
    from kernelrule.features import generated as gen

    cond = spec.get("condition")
    if cond:
        base = base_registry(str(cond), human=human)
    elif human is not None:
        base = human
    else:
        raise RegistryUnavailable(
            "the recorded axis list names no condition, so it is the "
            "human-written one — pass `human=REGISTRY` from the call site. "
            "The library does not reach for the global registry (§30.9).")
    pending: list[Feature] = []
    seen: set[str] = set()
    missing: list[str] = []
    for ax in spec["axes"]:
        if ax.get("source"):
            name, fn = gen.compile_feature(ax["source"],
                                           known=frozenset(seen))
            if name != ax["name"]:
                raise RegistryUnavailable(
                    f"the recorded axis {ax['name']!r} compiles as {name!r}")
            lo, hi = ax.get("expected_range") or (0.0, 1.0)
            pending.append(Feature(
                name=name, fn=fn, unit=str(ax.get("unit", "dimensionless")),
                expected_range=(float(lo), float(hi)),
                direction=str(ax.get("direction", "higher_is_worse")),
                source=ax["source"], code_hash=str(abs(hash(ax["source"])))))
        elif ax["name"] in base._items:
            pending.append(base[ax["name"]])
        else:
            missing.append(ax["name"])
            continue
        seen.add(ax["name"])
    if missing:
        raise RegistryUnavailable(
            f"{len(missing)} axes of the recorded list are not in "
            f"{base.name!r} any more: {missing[:5]}")
    return _rederive(pending, table, f"{spec.get('run') or 'exported'}-rebuilt")
