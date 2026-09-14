"""The feature registry (§8.2, §8.4).

A feature **does not judge.** It is a pure function
`(Problem, Hardware, Config) -> float`, and "good/bad" is what the rule
expresses through weights.

## Writing rules — enforced identically on the LLM

    - Pure function. No side effects
    - Computed from (Problem, Hardware, Config) only. No measurements, no
      profiler counters
    - Returns a single float. Normalising to 0~1 is recommended. **Unify the
      direction so that larger is worse**
    - Read hardware constants from hw.*. No hardcoding of 84 or 101376
    - No reference to cfg.ext (the architecture-transfer premise — §4.3)
    - It must have physical meaning. No arbitrary combinations
    - At most 10 lines

**Unifying the sign matters.** Fixing it to "larger is worse" makes the rule
always "weighted sum, then ascending sort", and removes any room for the LLM
to get the sign confused.

## Lifecycle — append-only, no rewriting (§8.4)

`deprecate` does not delete, it only marks. Existing rules keep running, and
the feature only drops out of the feature list in the prompt when new rules
are generated. **In-place modification is forbidden** — to fix a bug, add a
`_v2`. If past experiments become invalid, they cannot be reproduced.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Feature",
    "FeatureRegistry",
    "REGISTRY",
    "feature",
    "shape_feature",
    "render_features",
]

Direction = str   # "higher_is_worse" | "higher_is_better" | "neutral"


@dataclass(frozen=True)
class Feature:
    """One feature. The metadata is the material for the automatic checks
    (§8.3)."""

    name: str
    fn: Callable                      # (Problem, Hardware, Config) -> float
    unit: str                         # "dimensionless" | "bytes" | "count" | ...
    expected_range: tuple[float, float]
    direction: Direction
    doc: str = ""
    # ------------------------------------------------------------------
    # ★ The description is split into two layers (§8.2 / RuleWriter
    # condition A)
    # ------------------------------------------------------------------
    # To test the transfer claim you have to ask "can a rule be written
    # without looking at the table", and if sentences that came from the
    # table are mixed into the feature descriptions that test does not hold.
    # The two layers must be separated **as fields** so they do not mix when
    # the prompt is assembled — separating them by comment always mixes.
    #
    #   physical_meaning  "the fraction of SM slots idle in the last wave"
    #                     the same on a new GPU. It goes into condition A.
    #   observed          "on the A6000 a spilling kernel was never optimal"
    #                     ★ it came from the table. It is left out of A.
    #
    #: If empty, `doc` is used.
    physical_meaning: str = ""
    #: Properties observed in the table. It **must** have come from the
    #: training split only (§12.3 / D-28) — computing it on the full table
    #: leaks the holdout into the prompt.
    observed: tuple[str, ...] = ()
    #: Is it shape-level (Config-independent)? If True it enters
    #: `ShapeInfo` as a scalar.
    shape_level: bool = False
    #: Optional vectorised implementation. `(df, hw, info) -> np.ndarray`.
    #: If present FeatureMatrix uses it; otherwise the scalar is called a
    #: million times.
    #: **It must agree with the scalar implementation** —
    #: `verify_vectorized()` checks that.
    vec: Callable | None = None
    deprecated_at_round: int | None = None
    deprecation_reason: str = ""
    #: Source hash. Used in `features.lock` and in the FeatureMatrix cache
    #: key. ★ Build it with `code_hash_of` for a generated axis — the
    #: built-in `hash()` is salted per process (D-171 §S).
    #: key (§25).
    code_hash: str = ""
    #: ★ The source code. **For a feature built with `exec`,
    #: `inspect.getsource` fails** (OSError). §8.3's scale-invariance check
    #: reads "does it use hw" out of the source, so without this it rejects
    #: every hardware-independent feature — in the first F1 run a perfectly
    #: good feature was thrown away exactly that way (D-37).
    source: str = ""

    @property
    def active(self) -> bool:
        return self.deprecated_at_round is None

    @property
    def physics(self) -> str:
        """★ The architecture-independent definition. The only description
        the condition-A prompt uses."""
        return self.physical_meaning or self.doc

    def describe(self, *, include_observed: bool) -> str:
        """One prompt line. `include_observed=False` is condition A."""
        return self.describe_with(include_observed=include_observed)

    def describe_with(self, *, include_observed: bool,
                      extra: tuple[str, ...] = ()) -> str:
        # ★ Show the access form as it is. Giving only the name produces
        #   rules that use a config-level feature as `p.` — the very first
        #   RuleWriter call did exactly that.
        ref = f"{'p' if self.shape_level else 'f'}.{self.name}"
        # ★ The range and unit are **part of the feature's physical
        #   definition** (they are not the table). Without them relative
        #   weights cannot be set — terms of different magnitudes just get
        #   added, and the numerical optimiser cannot escape that point
        #   either. The first RuleWriter A attempt really did produce regret
        #   8.4.
        lo, hi = self.expected_range
        rng = f"[{lo:g}, {hi:g}]"
        head = f"{ref:28s} {rng:>14s}  {self.physics}"
        if include_observed:
            for o in (*self.observed, *extra):
                head += f"\n{'':28s}   [observed] {o}"
        return head


class FeatureRegistry:
    """Name -> Feature. **Keeps a typo from passing silently** (§21.3)."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._items: dict[str, Feature] = {}

    def add(self, f: Feature, *, replace: bool = False) -> Feature:
        if f.name in self._items and not replace:
            raise ValueError(
                f"feature {f.name!r} already exists. **In-place "
                f"modification is forbidden** (§8.4) — to fix a bug, add "
                f"{f.name}_v2 and deprecate the old version.")
        self._items[f.name] = f
        return f

    def annotate(self, name: str, *, physical_meaning: str = "",
                 expected_range: tuple[float, float] | None = None) -> None:
        """Attaches the description and range **afterwards**. It does not
        clutter the decorator.

        The "why does this drive performance" of all 24 features has to be
        reviewable in one place for a gap to be visible. Scattered across
        decorators, you fix `has_spill` only and miss the rest — which is
        what happened.

        ⚠️ **Do not put anything observed in the table here** (§12.3b). "Why
        is it slow" is physics; "how many times in this table" is an
        observation.
        """
        f = self[name]
        d = dict(f.__dict__)
        if physical_meaning:
            d["physical_meaning"] = physical_meaning
        if expected_range is not None:
            d["expected_range"] = tuple(expected_range)
        self._items[name] = Feature(**d)

    def deprecate(self, name: str, *, at_round: int, reason: str) -> None:
        f = self[name]
        self._items[name] = Feature(**{**f.__dict__,
                                       "deprecated_at_round": at_round,
                                       "deprecation_reason": reason})

    def __getitem__(self, name: str) -> Feature:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"unregistered feature: {name!r}. "
                f"available: {sorted(self._items)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def names(self, *, active_only: bool = True,
              shape_level: bool | None = None) -> list[str]:
        out = []
        for n, f in self._items.items():
            if active_only and not f.active:
                continue
            if shape_level is not None and f.shape_level != shape_level:
                continue
            out.append(n)
        return sorted(out)

    def items(self, **kw) -> list[Feature]:
        return [self._items[n] for n in self.names(**kw)]

    def lock_hash(self, *, active_only: bool = True) -> str:
        """A snapshot hash of the registry. The FeatureMatrix cache key +
        `features.lock`."""
        h = hashlib.sha256()
        for n in self.names(active_only=active_only):
            h.update(n.encode())
            h.update(self._items[n].code_hash.encode())
        return h.hexdigest()[:16]

    def lock(self) -> dict:
        return {n: {"code_hash": self._items[n].code_hash,
                    "unit": self._items[n].unit,
                    "direction": self._items[n].direction,
                    "deprecated_at_round": self._items[n].deprecated_at_round}
                for n in self.names(active_only=False)}


#: The global registry. `features/physical.py` registers into it.
REGISTRY = FeatureRegistry("physical")


def code_hash_of(code: str) -> str:
    """★ The identity hash of a generated axis's source (D-171 §S).

    ## What this replaced

    Four call sites used `str(abs(hash(code.strip())))`. Python's built-in
    `hash()` is **salted per process** for `str` (PYTHONHASHSEED), so the
    same axis got a different `code_hash` in every process:

    ```
    same code, two processes   59842078212829897  vs  5096144044792620548
    k7-1 registry lock_hash    f5d3822e314a505d   vs  6978c47d71d86426
    ★ the human REGISTRY       a8e4c6b89cd33cb1   ==  a8e4c6b89cd33cb1
    ```

    The human registry was stable because `_hash_fn` below already hashes
    the source with sha256 — **only the generated path was salted**, and
    that is the path every campaign since F1 runs on.

    ⛔ It made `FeatureMatrix._cache_key()` useless for generated
    registries: the key mixes `registry.lock_hash()`, which mixes
    `code_hash`, so a disk cache could **never** hit. A device that
    silently does nothing (principle 1).

    ## The normalisation, decided once

    `code.strip()` and nothing more. Leading and trailing whitespace cannot
    change what Python compiles; inner whitespace can (indentation), so
    collapsing it would make two different axes collide. ★ Two of the four
    sites stripped and two did not, so the **same axis already got two
    different hashes** depending on whether it arrived through
    `register_generated` or `load_generated` — one helper ends that too.
    """
    return hashlib.sha256(code.strip().encode()).hexdigest()[:16]


def _hash_fn(fn: Callable) -> str:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):       # pragma: no cover
        src = repr(fn)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def feature(*, unit: str = "dimensionless",
            expected_range: tuple[float, float] = (0.0, 1.0),
            direction: Direction = "higher_is_worse",
            vec: Callable | None = None,
            registry: FeatureRegistry | None = None,
            shape_level: bool = False,
            physical_meaning: str = "",
            observed: tuple[str, ...] = ()):
    """Registers a config-level feature. It is computed as an array and
    becomes `Feats.<name>`."""

    def deco(fn: Callable) -> Callable:
        f = Feature(name=fn.__name__, fn=fn, unit=unit,
                    expected_range=tuple(expected_range), direction=direction,
                    doc=(fn.__doc__ or "").strip().split("\n")[0],
                    shape_level=shape_level, vec=vec,
                    physical_meaning=physical_meaning,
                    observed=tuple(observed),
                    code_hash=_hash_fn(fn))
        # ★ The registry is **mandatory**. With a default, under the
        #   F0~F3 conditions it silently registers into the 24 a human
        #   wrote (§30.9). `registry or REGISTRY` will not do either —
        #   `__len__` exists, so **an empty registry is falsy**, and then
        #   only the first feature leaks into the global one, which is a
        #   worse shape of the same bug.
        if registry is None:
            raise ValueError(
                "feature(registry=...) is mandatory. With a default, "
                "which registry it registers into is invisible at the call "
                "site, and under the F0~F3 conditions the 24 a human wrote "
                "get mixed in silently (§26.4).")
        registry.add(f)
        fn.feature = f          # type: ignore[attr-defined]
        return fn

    return deco


def shape_feature(**kw):
    """A shape-level feature. It is a **scalar**, so a rule may write
    `if p.<name>:`.

    This is the kind of branch §8.1 (the appendix replacement) allows —
    "judge differently when memory-bound" generalises, whereas "config #17
    when M is 4096" is memorisation. A config-level feature is an array, so
    `if` raises `ValueError`, and that asymmetry makes conditional
    specialisation syntactically hard.
    """
    kw.setdefault("shape_level", True)
    kw["shape_level"] = True
    return feature(**kw)


def render_features(registry: FeatureRegistry, *,
                    include_observed: bool, active_only: bool = True,
                    extra_observed: dict[str, list[str]] | None = None) -> str:
    """The feature list to put in the prompt.

    ★ `include_observed=False` is **condition A** — not one sentence that
    came from the table goes in. Only then has "can a rule be written from
    physics alone, without looking at the table" actually been asked.

    Do not concatenate strings at the call site — **always use this
    function.** Assembling by hand mixes them; that is how block 3.5 got
    contaminated (D-28).

    `extra_observed` attaches per-feature observations **computed on the
    training split** (`report/table_facts.py` builds them). This is why
    numbers that came from the table are not put into the static
    `Feature.observed` field — such a value differs per split, and nailed
    into the source there is no way to tell which split it came from.

    ⚠️ With `include_observed=False`, `extra_observed` is **ignored too.**

    ⚠️ `registry` has no default. With one, the 24 a human wrote silently
    enter under the F0~F3 conditions — that experiment asks "can the LLM
    invent features", and the answer would be sitting in the prompt.
    """
    extra = extra_observed or {}
    # ★ No default. **Which registry goes into the prompt** is the
    #   experimental condition itself — leave it out and the 24 a human
    #   wrote get rendered silently into condition F1 (§30.9, principle 1).
    if registry is None:
        raise ValueError(
            "render_features must be given a registry. Which feature list "
            "goes into the prompt is an experimental condition (§26.4).")
    reg = registry
    items = [reg[n] for n in sorted(reg._items)]
    if active_only:
        items = [f for f in items if f.active]
    shape, cfg = [f for f in items if f.shape_level], [
        f for f in items if not f.shape_level]
    def line(f: Feature) -> str:
        return f.describe_with(include_observed=include_observed,
                               extra=tuple(extra.get(f.name, ())))

    h_shape = ("## Shape level — access only as `p.<name>`. "
               "Scalars, so `if p.<name>:` is allowed")
    h_cfg = ("## Config level — access only as `f.<name>`. "
             "**Arrays** (no `if`, raises ValueError)")
    tail = ("★ Swapping the prefix is rejected immediately. Using a name "
            "that is not in the `p.` list as `p.`, or the reverse.")
    out = [h_shape, *(line(f) for f in shape),
           "", h_cfg, *(line(f) for f in cfg), "", tail]
    return "\n".join(out)


def verify_vectorized(f: Feature, df, hw, info, *, n: int = 256,
                      rtol: float = 1e-9, seed: int = 0) -> None:
    """Checks that the vectorised implementation agrees with the scalar
    one.

    ⚠️ A mismatch is a **rejection** (§26.4). If the vectorisation silently
    differs, training (the matrix) and deployment (the scalar) use different
    functions, and that is the error §8.1 set out to remove.
    """
    from kernelrule.core.types import Problem, config_from_row

    if f.vec is None:
        raise ValueError(
            f"{f.name}: there is no vectorised implementation.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    sub = df.iloc[idx]
    got = np.asarray(f.vec(sub, hw, info), dtype=np.float64)
    p = Problem(M=int(sub.iloc[0]["M"]), N=int(sub.iloc[0]["N"]),
                K=int(sub.iloc[0]["K"]), dtype=str(sub.iloc[0]["dtype"]))
    want = np.asarray([float(f.fn(p, hw, config_from_row(r)))
                       for r in sub.to_dict("records")], dtype=np.float64)
    if not np.allclose(got, want, rtol=rtol, atol=1e-12, equal_nan=True):
        bad = int((~np.isclose(got, want, rtol=rtol, atol=1e-12,
                               equal_nan=True)).sum())
        raise ValueError(
            f"{f.name}: the vectorised implementation differs from the "
            f"scalar one ({bad}/{len(sub)} rows). Rejected — training and "
            "deployment would use different functions.")
