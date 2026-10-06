"""★ D-193 — does a seed rule carry a **time main term**? LLM 0 · GPU 0.

    python3 -m experiments.seed_structure --prefix f4 [--gpu a6000]

The pre-registered structure test (`docs/decisions.md` D-193):

```
time main term   one expression combines two or more of the path-time
                 features (PATH) inside max / sqrt / log / ... — a fitted
                 factor inside it (e.g. on the L2 path) is still one
                 expression, and is recorded as `weighted_inside`
not one          the path times appear only as separately weighted terms of
                 the score
```

Helper variables are followed: `sm = np.maximum(f.tm_crit_ratio,
f.tm_l2_ratio)` then `t = np.sqrt(np.square(sm) + ...)` counts as one
expression over three paths. Pure AST — it reads no table and no time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ★ D-195 — the analysis lives in the library (rules/time_term.py), where
#   the term cap reads it too. One copy (principle 2).
from kernelrule.rules.time_term import COMBINE, PATH, analyse  # noqa: E402,F401


def _final(run: Path) -> dict:
    """The loop's last best rule and what changed on the way (stage 3)."""
    bests = [json.loads(x) for x in (run / "bests.jsonl").read_text()
             .splitlines() if x.strip()]
    last = bests[-1]
    r = analyse(last["code"])
    ids = [b["rule_id"] for b in bests]
    change = next((b["round"] for b in bests if b["rule_id"] == ids[-1]),
                  None)
    return {"final": r, "rule_id": last["rule_id"],
            "last_best_round": change,
            "splitk_terms": sorted(x for x in r["features_used"]
                                   if x.startswith("splitk_")),
            "n_best_changes": len(dict.fromkeys(ids)) - 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--gpu", default="a6000")
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--out", type=Path, default=None)
    # ★ `final` reads the loop's last best (stage 3, loop seed 0) instead
    ap.add_argument("--source", choices=("chosen", "final"), default="chosen")
    a = ap.parse_args()
    out = {}
    for f in (int(x) for x in a.folds.split(",")):
        d = Path(f"runs/{a.prefix}-{a.gpu}-f{f}/stage2-rule-writer")
        if a.source == "final":
            out[f] = _final(Path(f"runs/{a.prefix}-{a.gpu}-f{f}-s0"))
            c = out[f]["final"]
            print(f"  f{f}  main term {'YES' if c['time_main_term'] else 'no ':3s}"
                  f"  w {c['n_weights']}  tm_cta_warps "
                  f"{'tm_cta_warps' in c['features_used']}  split-K "
                  f"{out[f]['splitk_terms']}  last best round "
                  f"{out[f]['last_best_round']}")
            continue
        ch = json.loads((d / "chosen.json").read_text())
        res = {"chosen": analyse(ch["code"]), "source": ch.get("source"),
               "train": ch.get("fit_regret")}
        # every candidate the RuleWriter wrote, not only the chosen one
        cands = sorted((d / "candidates").glob("try*.py"))
        res["candidates_with_main_term"] = sum(
            analyse(p.read_text())["time_main_term"] for p in cands)
        res["n_candidates"] = len(cands)
        out[f] = res
        c = res["chosen"]
        print(f"  f{f}  main term {'YES' if c['time_main_term'] else 'no ':3s}"
              f"  paths {c['main_term_paths']}  w {c['n_weights']}  "
              f"branches {c['branches']}  candidates with it "
              f"{res['candidates_with_main_term']}/{len(cands)}")
    key = "final" if a.source == "final" else "chosen"
    n = sum(v[key]["time_main_term"] for v in out.values())
    print(f"  structure: {n}/{len(out)} folds")
    if a.out:
        a.out.write_text(json.dumps({"prefix": a.prefix, "gpu": a.gpu,
                                     "n_with_main_term": n, "folds": out},
                                    ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
