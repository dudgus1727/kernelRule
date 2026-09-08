"""★ A new bundle is checked **before it is used** (D-89). 0 LLM calls.

    python3 experiments/bundle_guard.py datasets/<bundle> --env-hash <hash>

## Why it exists

kernelTab decided to fix the noise coefficients in the middle of a campaign.
**If it is not fixed before the bundle is generated, the A6000 coefficients
ride into the 5090 bundle.**

```
SIGMA_ABS_MS  5090 0.000016   ⛔ A6000 0.000374  (23x)
EVENT_TICK_MS 5090 0.000032   ⛔ A6000 0.001024  (32x)
```

**If the A6000 coefficients ride in, the number of candidates bundled as
"indistinguishable" on an 83us shape grows a great deal** — the answer set
widens and the ranking information disappears. And it happens **silently**:
there is a value, so no warning fires.

**Run this first thing on receipt, and if it is wrong do not use it.**
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: The A6000 values. **If a new GPU's bundle has these, they were loaded in by
#: mistake.**
A6000_SIGMA_ABS_MS = 0.000374
A6000_TICK_MS = 0.001024
#: The tolerance. The decimal literal of the JSON is read as it is, so it can
#: be very small.
#: ★ `abs(a-b) < tol` is not written directly — it falls apart at `inf`
#: (D-71).
#:   The test `test_no_raw_float_comparison_outside_numerics` caught this.
_TOL = 1e-12


def _close(a: float, b: float) -> bool:
    from kernelrule.core.numerics import approx_equal

    return approx_equal(a, b, tol=_TOL)


def _close_at(value: float, expect_str: str) -> tuple[bool, float]:
    """★ Does it match **down to the digits of the announced value** (D-125).

    The release note writes a rounded value — the 4090 says
    `sigma_abs 0.000743` while the bundle has `0.0007433368963633708`.
    Comparing with `tol=1e-12` **rejects a perfectly good bundle** (which is
    what happened).

    The tolerance is built from the number of decimals in the announced
    string: `0.000743` gives a rounding width of 5e-7. **It is not loosening
    things but looking only as far as the announced precision** — giving more
    digits makes it stricter. The A6000 coefficient check (`_close`) stays
    strict as it is.
    """
    t = expect_str.strip().lower()
    if "e" in t:
        mant, _, exp = t.partition("e")
        dec = len(mant.partition(".")[2]) - int(exp)
    else:
        dec = len(t.partition(".")[2])
    tol = 0.5 * 10.0 ** (-dec) if dec > 0 else 0.5
    from kernelrule.core.numerics import approx_equal

    return approx_equal(value, float(t), tol=tol), tol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--env-hash", required=True)
    ap.add_argument("--expect-rows", type=int)
    # ★ It is not taken with `type=float` — **the digits written down** are
    #   the tolerance (D-125). It has to be taken as a string to know the
    #   precision of "0.000743".
    ap.add_argument("--expect-tick-ms",
                    help="the announced tick (0.000032 for the 5090). ★ It is "
                         "checked down to the digits written — the release "
                         "note is a rounded value")
    ap.add_argument("--expect-sigma-abs-ms",
                    help="the announced absolute noise. ★ Checked down to the "
                         "digits written")
    # ★ The things that appeared in schema_version 3. If an announced value is
    #   given, it is matched.
    ap.add_argument("--expect-env-hash-v2",
                    help="the announced env_hash_v2 (prefix match)")
    ap.add_argument("--expect-schema-version", type=int)
    ap.add_argument("--expect-aggregate-status",
                    help="'all' means everything was aggregated with no "
                         "status filter")
    a = ap.parse_args()

    root = Path(a.bundle)
    meta = json.loads((root / "BUNDLE.json").read_text())
    nf = meta.get("noise_floor") or {}
    bad: list[str] = []
    warn: list[str] = []

    print("=" * 72)
    print(f"bundle check  {root.name}")
    print("=" * 72)
    print(f"  gpu       {meta.get('gpu_name')}  ({meta.get('arch')})")
    print(f"  env_hash  {str(meta.get('env_hash'))[:16]}")
    print(f"  rows/shapes/kernels  {meta.get('n_rows')} / "
          f"{meta.get('n_shapes')} / "
          f"{meta.get('n_kernels')}")
    print(f"  ridge     {meta.get('ridge_point')}")
    print(f"  schema    v{meta.get('schema_version')}  "
          f"aggregate_status={meta.get('aggregate_status')!r}  "
          f"env_hash_v2={str(meta.get('env_hash_v2'))[:16]}")
    print(f"  noise     {json.dumps(nf, ensure_ascii=False)}")

    # 1) does env_hash match the request (§3.4)
    if not str(meta.get("env_hash", "")).startswith(a.env_hash):
        bad.append(f"env_hash mismatch: requested {a.env_hash!r}")

    # 2) ★ did the A6000 coefficients ride in
    sig = nf.get("sigma_abs_ms")
    tick = nf.get("tick_ms")
    if sig is not None and _close(float(sig), A6000_SIGMA_ABS_MS):
        bad.append(f"★ sigma_abs_ms is the A6000 value "
                   f"{A6000_SIGMA_ABS_MS} — the noise coefficients rode in "
                   f"unfixed")
    if tick is not None and _close(float(tick), A6000_TICK_MS):
        bad.append(f"★ tick_ms is the A6000 value {A6000_TICK_MS} — the tick "
                   f"rode in unfixed")
    if tick is None:
        warn.append("tick_ms is not in the bundle (schema_version 1) — a "
                    "fallback is used and that is the A6000 tick")

    # 3) does it match the announced value
    for name, got, want in (("tick_ms", tick, a.expect_tick_ms),
                            ("sigma_abs_ms", sig, a.expect_sigma_abs_ms)):
        if want is None or got is None:
            continue
        ok, tol = _close_at(float(got), want)
        if ok:
            print(f"  ✓ {name} {got} = announced {want} "
                  f"(digit tolerance ±{tol:g})")
        else:
            bad.append(f"{name} {got} != announced {want} "
                       f"(digit tolerance ±{tol:g})")
    if a.expect_rows is not None and meta.get("n_rows") != a.expect_rows:
        warn.append(f"row count {meta.get('n_rows')} != announced "
                    f"{a.expect_rows}")

    # 3-b) the new entries of schema_version 3
    if a.expect_env_hash_v2 and not str(
            meta.get("env_hash_v2", "")).startswith(a.expect_env_hash_v2):
        bad.append(f"env_hash_v2 mismatch: "
                   f"{str(meta.get('env_hash_v2'))[:16]!r}"
                   f" != announced {a.expect_env_hash_v2[:16]!r}")
    if a.expect_schema_version is not None \
            and meta.get("schema_version") != a.expect_schema_version:
        bad.append(f"schema_version {meta.get('schema_version')} != announced "
                   f"{a.expect_schema_version}")
    if a.expect_aggregate_status is not None \
            and meta.get("aggregate_status") != a.expect_aggregate_status:
        # ★ This is a rejection. If `aggregate_status` is 'ok' then **the
        #   answer set is different** — the 5090 has 22.21% with status != ok,
        #   far more than the A6000's 10.65%, and a table that threw those
        #   away cannot be put beside one that kept them all (the D-91 line).
        bad.append(f"★ aggregate_status {meta.get('aggregate_status')!r} != "
                   f"announced {a.expect_aggregate_status!r} — the answer set "
                   f"is different")

    # 4) is the noise source this GPU
    src_gpu = str(nf.get("gpu", ""))
    if src_gpu and meta.get("gpu_name") and src_gpu != meta["gpu_name"]:
        bad.append(f"★ the noise source GPU differs: {src_gpu!r} vs "
                   f"{meta['gpu_name']!r}")

    print()
    for w in warn:
        print(f"  ⚠️  {w}")
    for b in bad:
        print(f"  ⛔ {b}")
    if bad:
        print("\n  ★ do not use this bundle. Report it to kernelTab and get "
              "it again.")
        sys.exit(1)
    print("\n  ✅ passed — it can be used")


if __name__ == "__main__":
    main()
