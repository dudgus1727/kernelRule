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
import ast
import json
from pathlib import Path

#: The path-time features of condition F4 — each a time over the ideal time.
PATH = ("tm_crit_ratio", "tm_l2_ratio", "tm_dram_ratio", "dram_w1_id_ratio",
        "dram_w1_hz_ratio")
#: The calls that combine paths into one time (and their log).
COMBINE = ("maximum", "fmax", "sqrt", "hypot", "log2", "log", "minimum",
           "fmin", "power")


def _feats(node: ast.AST) -> set[str]:
    return {n.attr for n in ast.walk(node)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "f"}


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _has_w(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
               and n.value.id == "w" for n in ast.walk(node))


def analyse(code: str) -> dict:
    fn = next(n for n in ast.parse(code).body
              if isinstance(n, ast.FunctionDef))
    # variable -> (path features it carries, does a weight sit in it)
    var: dict[str, tuple[set[str], bool]] = {}

    def closure(node: ast.AST) -> tuple[set[str], bool]:
        fs, w = set(_feats(node)) & set(PATH), _has_w(node)
        for nm in _names(node):
            if nm in var and nm != "s":
                fs |= var[nm][0]
                w = w or var[nm][1]
        return fs, w

    found: list[dict] = []
    for st in ast.walk(fn):
        if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                and isinstance(st.targets[0], ast.Name):
            t = st.targets[0].id
            if t != "s":
                fs, w = closure(st.value)
                old = var.get(t, (set(), False))
                var[t] = (old[0] | fs, old[1] or w)
        if isinstance(st, ast.Call) and isinstance(st.func, ast.Attribute) \
                and st.func.attr in COMBINE:
            fs, w = set(), False
            for a in st.args:
                f2, w2 = closure(a)
                fs |= f2
                w = w or w2
            if len(fs) >= 2:
                found.append({"call": st.func.attr, "paths": sorted(fs),
                              "weighted_inside": bool(w),
                              "expr": ast.unparse(st)[:160]})
    n_w = len({n.slice.value for n in ast.walk(fn)
               if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
               and n.value.id == "w" and isinstance(n.slice, ast.Constant)})
    branches = [ast.unparse(n.test) for n in ast.walk(fn)
                if isinstance(n, ast.If)]
    paths_used = sorted(_feats(fn) & set(PATH))
    return {"time_main_term": bool(found),
            "main_term_paths": sorted({p for x in found for p in x["paths"]}),
            "combinations": found[:4], "n_weights": n_w,
            "branches": branches, "path_features_used": paths_used,
            "features_used": sorted(_feats(fn))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--gpu", default="a6000")
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = {}
    for f in (int(x) for x in a.folds.split(",")):
        d = Path(f"runs/{a.prefix}-{a.gpu}-f{f}/stage2-rule-writer")
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
    n = sum(v["chosen"]["time_main_term"] for v in out.values())
    print(f"  structure: {n}/{len(out)} folds")
    if a.out:
        a.out.write_text(json.dumps({"prefix": a.prefix, "gpu": a.gpu,
                                     "n_with_main_term": n, "folds": out},
                                    ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
