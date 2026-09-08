"""The noise floor — delegated to kernelTab, but **the bundle coefficients
are enforced** (§30, §4-3).

## It is not reimplemented

The model (`max(a/t + b, tick/t)`) lives in `kerneltab.core.noise`. It is not
rewritten here. All this file does is **hold on to which coefficients are
used**.

## Why a wrapper is needed — the fail-open path actually found

kernelTab has **two** noise entry points, of different character.

    kerneltab.core.noise.noise_floor(t)   module-level globals (A6000 values
                                          nailed in)
    Bundle.noise_floor                    a function built from the bundle's
                                          coefficients + a warning when tick
                                          is absent

And `kerneltab.core.table.answer_set()` **calls the former.** For now the
development bundle is an A6000 so the values agree and there is no problem,
but once a 4090/H100 bundle arrives, `answer_set()` silently uses the A6000
tick. Not even a warning — because it never goes through `Bundle.tick_ms`.

This is the same class of thing as §30.8's "a safeguard that silently does
nothing". Without fixing kernelTab (out of our scope), **the mismatch is
turned into an error on this side.**

    NoiseModel.from_bundle(b)  ->  reads the bundle coefficients
                                   compares them against the module globals
                                   errors if they differ (no silent
                                   continuation)

## ★ 2026-08-31 — kernelTab closed that hole. The check changes

**The description above is kept as correction history.** On first opening
the 5090 bundle (`5bb6f403`) the comparison above tripped, and on checking it
turned out `kerneltab.core.table.answer_set()` no longer uses the module
globals — injecting `noise` is **mandatory** and without it it raises
`NoiseCoefRequired`. The module-level `noise_floor()` function itself is
gone. kernelTab's own docstring writes down exactly the failure we predicted
("it uses the A6000 tick without warning when scoring a 4090/H100 bundle
too").

Then **a coefficient mismatch is no longer a danger.** Another GPU's bundle
differs from the A6000 constants by definition, so leaving the old check in
place **blocks every non-A6000 bundle forever** — not a safeguard but a road
closure.

```
old check   bundle coefficients == module globals ?  -> another GPU always fails
new check   ★ does answer_set demand injection ?     -> fails if the hole is alive
            a coefficient mismatch is **recorded as a fact** (a warning, and
            it stays in `source`)
```

★ The check becomes a **capability check**. Not "are the values equal" but
"is the dangerous path closed". If the hole comes back (someone restores a
default), the new check catches it — the old one could not (on the A6000,
where the values agree, it passed).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["NoiseModel", "NoiseMismatchError"]


class NoiseMismatchError(RuntimeError):
    """The bundle coefficients differ from kernelTab's module-level
    globals.

    In that state `kerneltab.core.table.answer_set()` runs **with the wrong
    coefficients** (because it uses the globals). Fixing only our own
    computation would not help, so this stops.
    """


def _require_injected_noise() -> None:
    """★ Do `answer_set`/`answer_tolerance` **demand** coefficient
    injection?

    If they do not, they run silently on the default (= the A6000 tick).
    That is §30.8's "a safeguard that silently does nothing".

    The old check (bundle coefficients == module globals) **could not catch
    this** — on the development A6000 bundle the values agreed so it passed,
    while on other GPUs it blocked despite there being no danger. Wrong in
    both directions.
    """
    from kerneltab.core.noise import NoiseCoefRequired
    from kerneltab.core.table import answer_tolerance

    try:
        answer_tolerance(1.0)
    except NoiseCoefRequired:
        return
    except Exception as e:                                  # noqa: BLE001
        raise NoiseMismatchError(
            "kerneltab.core.table.answer_tolerance fails unexpectedly: "
            f"{type(e).__name__}: {e}") from e
    raise NoiseMismatchError(
        "★ kerneltab.core.table.answer_tolerance() returns a value even "
        "without noise coefficients injected — a default has come back.\n"
        "  That default is the A6000 tick, and scoring another GPU's bundle "
        "makes the answer set silently wrong (the 5090 tick is 1/64 of the "
        "A6000's).\n"
        "  Check the kernelTab side. Our own scoring is safe through "
        "PerfTable.answer_mask(), but in this state any other code using "
        "the kernelTab helpers is silently wrong.")


@dataclass(frozen=True, slots=True)
class NoiseModel:
    """The noise floor of this measurement condition. **It carries the
    provenance of the coefficients.**

    `sigma_rel` and `noise_floor` are exposed **separately** (§30.3).
    Collapsing them into one function would make the answer set (which needs
    resolution) and noise-model validation (statistics only) use the same
    value.
    """

    sigma_abs_ms: float
    sigma_rel_coef: float
    tick_ms: float
    #: Where the coefficients came from. Needed for reports and
    #: reproduction.
    source: str
    #: Whether a fallback was used because `tick_ms` is absent from the
    #: bundle (schema_version 1).
    tick_is_fallback: bool = False

    # -- The two terms are exposed separately (§30.3) ---------------------
    def sigma(self, t):
        """The **statistical** noise only. `a/t + b`. Used when comparing
        against observed spread.

        It is the component that averages down with repeated measurement.
        """
        t = np.asarray(t, dtype=np.float64)
        out = np.where(t > 0, self.sigma_abs_ms / np.where(t > 0, t, 1.0)
                       + self.sigma_rel_coef, self.sigma_rel_coef)
        return float(out) if out.ndim == 0 else out

    def tick_pct(self, t):
        """The **resolution limit**. `tick/t`. Repetition does not shrink
        it.

        Two configs that land on the same tick are recorded with literally
        identical times.
        """
        t = np.asarray(t, dtype=np.float64)
        out = np.where(t > 0, self.tick_ms / np.where(t > 0, t, 1.0), np.inf)
        return float(out) if out.ndim == 0 else out

    def floor(self, t):
        """The **smallest relative difference that can distinguish** two
        measurements. `max(statistical, resolution)`.

        ⚠️ When it cannot be computed it returns a **conservatively large**
        value (§26.4), not 0 — returning 0 would mean "every difference is
        significant", and all the noise would be learned as signal.
        """
        t = np.asarray(t, dtype=np.float64)
        bad = ~(np.isfinite(t) & (t > 0))
        safe = np.where(bad, 1.0, t)
        out = np.maximum(self.sigma_abs_ms / safe + self.sigma_rel_coef,
                         self.tick_ms / safe)
        # With the time unknown, treat it as unresolvable. 1.0 = 100%, i.e.
        # "cannot distinguish anything".
        out = np.where(bad, 1.0, out)
        return float(out) if out.ndim == 0 else out

    def answer_tol(self, best_ms) -> float:
        """The answer-set tolerance. `2 * floor(best)` (§30.3's 2σ
        rationale).

        At 1σ, noise turns a right answer into a wrong one 32% of the time;
        at 2σ, 5%. For this tolerance **overestimating is safer than
        underestimating** — overestimating merely makes it lenient towards
        the rule, whereas underestimating learns noise as signal.
        """
        return 2.0 * float(self.floor(best_ms))

    def resolvable(self, t_a, t_b, k: float = 2.0):
        """Is the difference between two measurements not explained by
        noise?"""
        a = np.asarray(t_a, dtype=np.float64)
        b = np.asarray(t_b, dtype=np.float64)
        s = k * np.sqrt((self.floor(a) * a) ** 2 + (self.floor(b) * b) ** 2)
        out = np.abs(a - b) > s
        return bool(out) if out.ndim == 0 else out

    # -- Construction -----------------------------------------------------
    @classmethod
    def from_bundle(cls, bundle, *, strict: bool = True) -> NoiseModel:
        """Builds from the bundle's `noise_floor` coefficients. **It
        compares against the module globals.**

        With `strict=True` (the default) a mismatch is an error. That is
        because of the fail-open path in the docstring above — `answer_set()`
        uses the module globals, so a mismatch means the state "our
        computation is right and the kernelTab helper is wrong".
        """
        from kerneltab.core import noise as kt

        info = getattr(bundle, "info", None) or {}
        coefs = dict(info.get("noise_floor") or {})
        if not coefs:
            raise NoiseMismatchError(
                f"bundle {info.get('bundle_id')} has no noise_floor "
                f"coefficients. This is not papered over with a default "
                f"(§26.4).")

        missing = [k for k in ("sigma_abs_ms", "sigma_rel") if k not in coefs]
        if missing:
            raise NoiseMismatchError(
                f"the bundle's noise_floor lacks {missing}. Coefficients "
                f"are not guessed.")

        # tick_ms is absent in schema_version 1. `Bundle.tick_ms`
        # substitutes for it and **warns** — that warning is not swallowed
        # but recorded here as a fact (§30.3b).
        tick_present = bool(coefs.get("tick_ms"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tick = float(bundle.tick_ms)
        if not tick_present:
            for w in caught:
                warnings.warn(
                    f"[kernelRule] using a fallback for the bundle's "
                    f"tick_ms — {w.message}",
                    stacklevel=2)

        model = cls(
            sigma_abs_ms=float(coefs["sigma_abs_ms"]),
            sigma_rel_coef=float(coefs["sigma_rel"]),
            tick_ms=tick,
            source=(f"BUNDLE.json:{info.get('bundle_id')}"
                    f"{'' if tick_present else ' (tick_ms=fallback)'}"),
            tick_is_fallback=not tick_present,
        )

        # ★ A **capability check** — is the dangerous path closed
        #   (2026-08-31)? Not a value comparison. On another GPU the values
        #   differ, of course.
        if strict:
            _require_injected_noise()

        # A coefficient mismatch is **recorded as a fact**. It does not
        # block — another GPU's bundle differs from the A6000 constants by
        # definition.
        drift = {
            "sigma_abs_ms": (model.sigma_abs_ms, kt.SIGMA_ABS_MS),
            "sigma_rel": (model.sigma_rel_coef, kt.SIGMA_REL),
            "tick_ms": (model.tick_ms, kt.EVENT_TICK_MS),
        }
        bad = {k: v for k, v in drift.items()
               if abs(v[0] - v[1]) > 1e-12 * max(1.0, abs(v[1]))}
        if bad:
            warnings.warn(
                f"[kernelRule] the bundle coefficients differ from "
                f"kerneltab.core.noise's A6000 reference values (normal — "
                f"it is a different GPU): "
                f"{ {k: v[0] for k, v in bad.items()} }. "
                "Scoring uses the bundle coefficients "
                "(PerfTable.answer_mask).",
                stacklevel=2)
        return model

    @classmethod
    def a6000_reference(cls) -> NoiseModel:
        """The measured A6000 c63710df coefficients. **For the synthetic
        table and tests only.**

        For a real bundle use `from_bundle` — using these values on another
        GPU is wrong.
        """
        from kerneltab.core import noise as kt

        return cls(sigma_abs_ms=kt.SIGMA_ABS_MS, sigma_rel_coef=kt.SIGMA_REL,
                   tick_ms=kt.EVENT_TICK_MS,
                   source="kerneltab.core.noise (A6000 c63710df measured)")
