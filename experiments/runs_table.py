"""★ It **builds** `docs/artifacts/runs.md` — it is not written by hand
(D-128).

    python3 experiments/runs_table.py            # build
    python3 experiments/runs_table.py --check    # check whether it diverged
                                                 # (the tests call this)

The conditions are read from `config.json` and the final scores **from the
artifacts json**. No number is written into this file — writing one makes it
diverge (the same way as `decisions_index.py`).

⚠️ 2026-09-08 (D-146): the strings that go **into `docs/artifacts/runs.md`**
(the table header, the note and retirement texts, the status marks) were
translated together with the document. Nothing but the language changed; the
Korean originals are at commit `ee53b4d`. Old `## D-N` entries in
`docs/decisions.md` quote the old marks (`★ 미업로드`, `★갈림`) — those stay,
they are the record of what the table said at the time.

## The tag rule (D-128 §1-7)

```
<feature><seed>-p<parameters>[-<expressiveness>][-<experiment name>]
  F3rw-p8   F3rw-p16   F3rw-p8-prod   F3hg-p8-d75-a
★ The table (GPU) and the inheritance do not go in the tag — config.json
  holds them
★ Anything starting with `x-` is **discarded** (the rank-loss line and so
  on). It does not go in the table
```

### ★ The code-version rule was corrected (2026-09-05, D-137)

D-128 said "the code version does not go in the tag". But the `__import__`
fix **makes the 2% of proposals that were being thrown away get scored** —
if it changes the result then it is a condition.

```
a code change that changes the result   ★ it is attached to the tag
                                        (like `-nan`)
any other change                        by commit only — ★ the `commit`
                                        column of this table
```

The commit is read from the first line of `trace.jsonl`
(`run_start.commit`). An old run with no trace is `?` — **it is not filled
in by guessing** (principle 39).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from kernelrule.core.runset import run_condition
from kernelrule.rules.checks import fitter_for

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "artifacts" / "runs.md"
BEGIN = "<!-- RUNS:BEGIN — experiments/runs_table.py builds it -->"
END = "<!-- RUNS:END -->"

#: **Where in which artefact** the final score is. The value is not written
#: here (principle 2).
#: `(file, key path, aggregation)` — the aggregation `med` is the median of
#: the list.
CANON: dict[str, tuple[str, tuple, str]] = {
    # ★ The old representative value. The new one (F3rw-p8) gets an artefact
    #   once the re-measurement is done
    "F3rw-p8": ("canon-p8.json", ("holdout_regret",), "med"),
    # ★ An old representative value (retired after D-131). conclusion.json
    #   holds the value
    "F3rw-p8-old": ("conclusion.json", ("f1_vs_human", "human_median"),
                    "one"),
    "F1rw-p8": ("conclusion.json", ("f1_vs_human", "f1_median"), "one"),
    "F2rw-p8": ("conclusion.json", ("f2", "median"), "one"),
    "F3rw-p8-cma": ("expressive-regret.json", ("r1", "rb08"), "med"),
    "F3rw-p16": ("expressive-regret.json", ("r1", "rb16"), "med"),
    "F3rw-p8-prod": ("expressive-regret.json", ("r1", "rprod"), "med"),
    "F3rw-p8-pow": ("expressive-regret.json", ("r1", "rpow"), "med"),
    "F3rw-p8-5090": ("c-ladder.json", ("regret", "5090sigma-hw2"), "med"),
    "F3rw-p8-4090": ("sigma-4090.json", ("holdout_regret",), "med"),
    # ★ The current representative value (D-140). The rounds are 12, so it is
    #   **read at r11** — the campaign ran to round 24 but the value we
    #   report is the one at r11
    "F3rw-p8-nan": ("round-curve-bests.json",
                    ("groups", "F3rw-p8-nan", "curves"), "med@11"),
}
#: Tags that are not retired but need an explanation. They go into the status
#: column as they are.
NOTES = {
    "F3rw-p8-nan": "★ the current canonical value — `-nan` means the "
                   "np.errstate guard in `compile_rule` (D-135). The "
                   "campaign before it is `F3rw-p8`. ★ The final score is "
                   "**read at r11** (round 12, D-140) — the campaign ran to "
                   "24. The commits differ because of a docs commit in the "
                   "middle of the campaign, and `kernelrule/`·`prompts/` did "
                   "not change in any of the pairs (checked, D-137)",
    "F3rw-p8": "a campaign that ran on faulty code — 2% of the proposals "
               "were thrown away by `__import__` (D-135). The canonical "
               "value is `F3rw-p8-nan`",
    "F3rw-p8-p3": "a campaign stopped at r4~r6 by patience 3 (D-131)",
    "smoke": "★ a wiring smoke run, not a result (D-149) — 1 seed x 2 rounds, "
             "fold 0. Do not read its numbers",
    "uncapped": "★ a behaviour check, not a result (D-150) — 1 seed x 2 "
                "rounds, fold 0, with the parameter cap removed. Do not read "
                "its numbers",
    "nocount": "★ a contrast, not a result (D-151) — 1 seed x 1 round, fold "
               "0, after the size sentences were taken out of every surface. "
               "Do not read its numbers",
    "seed3": "★ a contrast, not a result (D-151) — the same round with a "
             "**3-term seed**, to see whether len(w0) follows the parent. Do "
             "not read its numbers",
    "noregime": "★ 12 rounds with our regime axis taken out of what the "
                "model sees (D-156) — 1 seed, fold 0, 3-term seed. Do not "
                "read its numbers",
    "archive": "★ 12 rounds with the archive rebuilt (D-155) — 1 seed, fold "
               "0, 3-term seed. New population, cut line and axes. Do not "
               "read its numbers",
    "ast3000": "★ 12 rounds with the AST node cap at 3000 (D-154) — 1 seed, "
               "fold 0, 3-term seed. Refusals 3/72, none of them the cap. Do "
               "not read its numbers",
    "full12": "★ the first full 12 rounds under the new design (D-153) — 1 "
              "seed, fold 0, 3-term seed. ⚠️ 17 of its 18 refusals are the "
              "AST node cap. Do not read its numbers",
    "nocap": "★ a behaviour check, not a result (D-152) — 1 seed x 2 rounds, "
             "fold 0, 3-term seed, **after the live cap in the schema "
             "validator was removed**. Do not read its numbers",
}
#: The prefixes dropped from the table. **They are not deleted** — they are
#: only marked by name.
DROP = ("x-",)
#: ★ The retired (to be re-measured) tags and the reason. They **stay in the
#: table as a status** (D-129 §3-2).
RETIRED = {
    "F3rw-p8-cma": "p8 and yet CMA — the current rule (fitter_for) does "
                   "not produce it",
    "F3rw-p8-prod": "p8 and yet CMA. To be re-measured",
    "F3rw-p8-pow": "p8 and yet CMA. To be re-measured",
    "F3rw-p8-old": "the old canonical value — the old prompt·12 "
                   "rounds·patience 10 (D-129)",
}


def _canon(tag: str) -> tuple[str, str]:
    """(the value string, the source). Blank if there is none."""
    spec = CANON.get(tag)
    if spec is None:
        return "", ""
    f, path, how = spec
    p = ROOT / "docs" / "artifacts" / f
    if not p.exists():
        return "", ""
    v = json.loads(p.read_text())
    for k in path:
        if not isinstance(v, (dict, list)) or (
                isinstance(v, dict) and k not in v):
            return "", ""
        v = v[k]
    import statistics as st
    if how == "med":
        if not isinstance(v, list) or not v:
            return "", ""
        v = st.median(v)
    elif how.startswith("med@"):
        # ★ The median at round N. It reads the curve in
        #   `round-curve-bests.json`. The curve **carries the value forward
        #   while the best does not change**, so it can be short — taking
        #   `min(N, len-1)` is the same as "if it had stopped there" (D-139).
        r = int(how[4:])
        if not isinstance(v, dict) or not v:
            return "", ""
        v = st.median([c[min(r, len(c) - 1)] for c in v.values()])
    return f"{float(v):.4f}", f


def _commit(runs: list[str]) -> str:
    """The commit on the first line of `trace.jsonl`. `?` if there is none —
    it is not guessed."""
    got = set()
    for r in runs:
        p = RUNS / r / "trace.jsonl"
        if not p.exists():
            got.add("?")
            continue
        with p.open() as f:
            line = f.readline()
        try:
            got.add(json.loads(line).get("commit") or "?")
        except json.JSONDecodeError:
            got.add("?")
    # ★ Committing in the middle of a campaign makes it differ per seed. **It
    #   is not hidden, all of them are written down** — if the commit did not
    #   change `kernelrule/` the behaviour is the same, but that judgement is
    #   made by a human.
    return "·".join(sorted(got)) if got else "?"


#: While a campaign is running, that tag's row changes every round.
#: `--check` leaves **only that row** out — the check on the other rows stays
#: alive.
LIVE_SECONDS = 1800
#: The release ledger. `trace_release.py --upload` writes it (D-138).
TRACE_MANIFEST = ROOT / "docs" / "artifacts" / "trace-releases.json"
MISSING = "★ not uploaded"


def _trace(tag: str, runs: list[str]) -> str:
    """Which release the trace is in. If there is a trace but no release, it
    is written down as such."""
    if not any((RUNS / r / "trace.jsonl").exists() for r in runs):
        return ""
    m = (json.loads(TRACE_MANIFEST.read_text())
         if TRACE_MANIFEST.exists() else {})
    return m.get(tag, {}).get("release") or MISSING


def _live_tags() -> set[str]:
    """The tags with a run written to within the last `LIVE_SECONDS`. It
    looks only at mtime — it does not guess."""
    import time
    now, live = time.time(), set()
    for tag, runs in _groups().items():
        for r in runs:
            f = RUNS / r / "rounds.jsonl"
            if f.exists() and now - f.stat().st_mtime < LIVE_SECONDS:
                live.add(tag)
                break
    return live


def _groups() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in sorted(RUNS.iterdir()):
        m = re.match(r"^(.*)-s(\d+)$", d.name) if d.is_dir() else None
        if not m or not (d / "config.json").exists():
            continue
        if m.group(1).startswith(DROP):
            continue
        out.setdefault(m.group(1), []).append(d.name)
    return out


def _rows() -> list[dict]:
    rows = []
    for tag, runs in sorted(_groups().items()):
        conds = [run_condition(r) for r in runs]

        def one(key: str, _c=conds) -> str:
            """Do all the seeds have the same value? If they differ, **the
            table says so**."""
            vals = {str(c[key]) for c in _c}
            if len(vals) > 1:
                return "★split"
            v = vals.pop()
            # ★ What is absent is `?`. It is not filled in by guessing
            #   (principle 39).
            return "?" if v in ("None", "") else v

        cfg = json.loads((RUNS / runs[0] / "config.json").read_text())
        nr = sorted({sum(1 for _ in (RUNS / r / "rounds.jsonl").open())
                     for r in runs})
        hw = one("hw")
        gpu = {"37762692c36f5ba4": "4090", "cc392ddd72b4902d": "5090"}.get(
            hw, "a6000" if hw and hw != "None" else "?")
        val, src = _canon(tag)
        rows.append({
            "tag": tag, "n": len(runs),
            "features": str(cfg.get("n_features", "?")),
            "condition": one("feature_condition"),
            "seed": one("seed_source").split(" (")[0],
            "parameters": one("parameters"),
            "hint": ("prod" if one("product_hint") == "True" else "")
                    + ("pow" if one("power_hint") == "True" else "")
                    or "default",
            "fitter": f"{one('fit_method')}/{one('fit_restarts')}/"
                      f"{cfg['loop'].get('max_evals', '?')}",
            # ★ Is this run off the §1-6 rule (nelder-mead when p<=8)
            # ★ "none" = no cap (D-160). The fitter is then chosen per
            #   rule from `len(w0)`, so a campaign-level verdict does not
            #   apply — it is not marked as off the rule.
            "off_rule": (one("parameters") not in ("?", "★split", "none")
                         and one("fit_method") != fitter_for(
                             int(one("parameters")))["fit_method"]),
            "rounds": "~".join(map(str, nr)),
            "gpu": gpu, "canon": val, "canon_src": src,
            "commit": _commit(runs),
            "trace": _trace(tag, runs),
            "note": NOTES.get(tag, ""),
            "retired": RETIRED.get(tag, ""),
            "objective": one("objective")})
    return rows


def render() -> str:
    rows = _rows()
    head = ("| tag | seeds | features | seed rule | parameters | "
            "expressiveness | fitter | rounds | table | commit | trace | "
            "final score | source | status |")
    L = [BEGIN, "", head,
         "|---|--:|---|---|--:|---|---|---|---|---|---|--:|---|---|"]
    for r in rows:
        L.append(
            f"| `{r['tag']}` | {r['n']} | {r['features']}/{r['condition']} |"
            f" {r['seed']} | {r['parameters']} | {r['hint']} | {r['fitter']} |"
            f" {r['rounds']} | {r['gpu']} | `{r['commit']}` |"
            f" {r['trace'] or '—'} |"
            f" {r['canon'] or '—'} | {r['canon_src'] or '—'} |"
            + (f" ⛔ retired — {r['retired']}" if r["retired"]
               else " ⚠️ outside the fitter rule" if r["off_rule"] else "")
            + ((" · " if (r["retired"] or r["off_rule"]) else " ")
               + r["note"] if r["note"] else "") + " |")
    L += ["", END]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit with a non-zero code if it diverged (the tests "
                         "call this)")
    a = ap.parse_args()
    body = render()
    txt = OUT.read_text() if OUT.exists() else ""
    if BEGIN in txt and END in txt:
        new = txt[:txt.index(BEGIN)] + body + txt[txt.index(END) + len(END):]
    else:
        new = txt.rstrip() + "\n\n" + body + "\n"
    if a.check:
        live = _live_tags()

        def strip(t: str) -> list[str]:
            return [x for x in t.splitlines()
                    if not any(x.startswith(f"| `{g}` |") for g in live)]

        # ★ Catch a trace that has no release (D-138). A running one is left
        #   out.
        miss = [r["tag"] for r in _rows()
                if r["trace"] == MISSING and r["tag"] not in live]
        if miss:
            print(f"★ there is a trace but no release: {miss}. Run "
                  "`python3 experiments/trace_release.py --tag <tag> "
                  "--upload` and build it again.")
            sys.exit(1)
        if strip(new) != strip(txt):
            print("★ runs.md diverged from the run artefacts. Build it again "
                  "with `python3 experiments/runs_table.py`.")
            sys.exit(1)
        if live:
            print(f"  ★ tags skipped because they are running: "
                  f"{sorted(live)}")
        print(f"runs.md is up to date ({len(_rows())} tags)")
        return
    OUT.write_text(new)
    print(f"runs.md updated — {len(_rows())} tags")


if __name__ == "__main__":
    main()
