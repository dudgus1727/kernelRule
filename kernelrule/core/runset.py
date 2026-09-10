"""★ Check that a set of runs shares **one condition** (D-120).

## Why

`transfer_29_5.TABLES["5090"]["runs"]` grouped six runs as (c), but the last
three used a `human_guided` **hand seed**. Something that was not "from
scratch on the 5090 table" was mixed into (c), and that fact **was written
down** in `chosen.json`'s `source`. Nobody read it.

    D-113   arch_prompt was in config.json and went unread
    D-119   the seed source was in chosen.json and went unread
    ★ the second occurrence one day after principle 39 was written

**So it becomes a check.** Called where the set is built, a set whose
condition differs **fails** (§26.4).

## What is compared

    seed         stage-2 `chosen.json` source + code hash
    objective    objective / rank_top_k / rank_lambda
    form         parameters / product_hint / power_hint
    hardware     arch_prompt or hw_text hash   <- D-113 was here
    split/cond   split.kind / feature_condition / model

A **missing** value (an older run) does not count as a divergence — missing
and different are not the same. But present on one side only is a divergence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = ["RunSetError", "run_condition", "assert_same_condition",
           "condition_report"]

ROOT = Path("runs")

#: The condition keys compared. **What is not here is not looked at** — when
#: this grows, the tests grow with it.
KEYS = ("seed_source", "seed_sha", "objective", "rank_top_k", "rank_lambda",
        "parameters", "product_hint", "power_hint", "hw", "split_kind",
        "feature_condition", "model", "fit_method", "fit_restarts")

#: ★ The **old default** of a newly added key (D-123). Older runs have no
#: such key in `config.json`, and this is what the code did back then — so
#: filling "missing" with this value is not leniency but **stating the
#: fact**. Add an entry here only when a new key goes into KEYS. Runs that do
#: have a value are compared as they are.
_OLD_DEFAULTS = {"fit_method": "nelder-mead", "fit_restarts": 4}


class RunSetError(ValueError):
    """The condition differs within the set. **Do not proceed silently**
    (§26.4)."""


def _parameters(rc: dict):
    """The parameter cap as recorded. `"none"` = **explicitly no cap**,
    `None` = the run does not say (D-160)."""
    if rc.get("no_parameter_cap"):
        return "none"
    v = rc.get("parameters")
    if v is None:
        v = rc.get("budget")
    if v is None and ("parameters" in rc or "budget" in rc):
        # The key is there and holds null — that is "no cap" written out.
        return "none"
    return v


def _campaign(run: str) -> str:
    """`f1pipe-F3-tag-s0` -> `f1pipe-F3-tag`. The seed is per campaign."""
    parts = run.rsplit("-s", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else run


def run_condition(run: str, root: Path | None = None) -> dict:
    """One run's condition. A missing value is `None`."""
    r = (root or ROOT)
    cfg_p = r / run / "config.json"
    if not cfg_p.exists():
        raise RunSetError(f"{cfg_p} does not exist. Cannot read the "
                          "condition.")
    c = json.loads(cfg_p.read_text())
    loop, llm = c.get("loop", {}), c.get("llm", {})
    rc = c.get("rule_constraints") or {}
    hw = llm.get("hw_text")
    hw_id = (hw.get("sha256") if isinstance(hw, dict)
             else llm.get("arch_prompt"))
    out = {
        "objective": loop.get("objective"),
        "rank_top_k": loop.get("rank_top_k"),
        "rank_lambda": loop.get("rank_lambda"),
        # ★ D-128 rename: `budget` -> `parameters`. The repository's old
        #   artefacts were converted, but an old run may arrive from outside,
        #   so both names are read.
        # ★ 2026-09-10 (D-160): `"none"` is **no cap**, and it is a
        #   different thing from `None` = the run does not say. Runs before
        #   D-160 wrote 8 even with the cap off, and those files are left
        #   alone — the commit is what separates them.
        "parameters": _parameters(rc),
        "product_hint": llm.get("product_hint"),
        "power_hint": llm.get("power_hint"),
        "hw": hw_id,
        "split_kind": (c.get("split") or {}).get("kind"),
        "feature_condition": loop.get("feature_condition"),
        "model": llm.get("model") or llm.get("class"),
        # ★ The fitter (D-123). Older runs have no key, and back then it was
        #   Nelder-Mead with 4 restarts only — filled from `_OLD_DEFAULTS`.
        "fit_method": loop.get(
            "fit_method", _OLD_DEFAULTS["fit_method"]),
        "fit_restarts": loop.get(
            "fit_restarts", _OLD_DEFAULTS["fit_restarts"]),
        "seed_source": None, "seed_sha": None,
    }
    ch = r / _campaign(run) / "stage2-rule-writer" / "chosen.json"
    if ch.exists():
        d = json.loads(ch.read_text())
        out["seed_source"] = d.get("source")
        out["seed_sha"] = hashlib.sha256(
            (d.get("code") or "").encode()).hexdigest()[:12]
    return out


def condition_report(runs, root: Path | None = None) -> dict:
    """The **observed values** per key. One value means one condition."""
    conds = {r: run_condition(r, root) for r in runs}
    return {k: sorted({str(c[k]) for c in conds.values()}) for k in KEYS}


def assert_same_condition(runs, *, keys=KEYS, label: str = "the set",
                          root: Path | None = None) -> dict:
    """Is the condition single within the set? If not, **raise** (§26.4).

    ★ Call it where the set is built. Looking later turns into "it was there
    and nobody looked" (principle 39).
    """
    runs = list(runs)
    if len(runs) < 2:
        return {}
    conds = {r: run_condition(r, root) for r in runs}
    bad = {}
    for k in keys:
        vals = {r: c[k] for r, c in conds.items()}
        uniq = {str(v) for v in vals.values()}
        if len(uniq) > 1:
            bad[k] = vals
    if bad:
        lines = [(f"the condition differs within {label} "
                  f"({len(bad)} keys):")]
        for k, vals in bad.items():
            lines.append(f"  {k}:")
            for r, v in vals.items():
                lines.append(f"    {r:34s} {v}")
        lines.append("aggregating runs of different conditions into one set "
                     "makes that number the average of two conditions "
                     "(D-119).")
        raise RunSetError("\n".join(lines))
    return {k: conds[runs[0]][k] for k in keys}
