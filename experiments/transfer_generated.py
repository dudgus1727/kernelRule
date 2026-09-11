"""★ Does a rule that uses **generated** axes transfer? 0 LLM calls.

    python3 -m experiments.transfer_generated

## Why this exists

`transfer_big_rules.py` checks the rule against `REGISTRY` — the axes a
human wrote. Every rule transferred so far (D-157) came from an F3 run, so
that was the right list. **An F1/F2 rule uses axes the FeatureWriter built**,
and those are not in `REGISTRY`: the rule is refused before it is ever
scored. The 21-run campaign is F2, so without this there is nothing to
transfer.

## ★ What "transferring a rule" means from here on

```
before   move the rule. Both tables used the same 27 human axes
★ now    move the rule **and the axes it uses**
```

That is legitimate because a feature is a hardware-independent formula —
`waves = tiles / (hw.sm_count * cfg.max_blocks_per_sm)` is right whatever the
SM count is. ⚠️ But it **is a different condition from the D-157 numbers**,
and those are not overwritten (this writes its own file).

## ⚠️ Two things that cannot happen in an F3 transfer

```
(a) the shape-level verdict is made **from values** (§30.12). An axis that is
    shape level on the A6000 can be config level on the 5090 — and then a
    rule written as `p.X` is refused on the target
(b) an axis that varies on the source can be **constant** on the target, and
    a branch on it is dead
```

Both are counted and reported rather than smoothed over.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.transfer_29_5 import TABLES, _score_on, _splits
from kernelrule.baselines.static_topk import StaticTopK
from kernelrule.core.crosstable import common_shapes
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import geomean
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights
from kernelrule.features import Feature, FeatureRegistry
from kernelrule.features.generated import compile_feature, detect_shape_level
from kernelrule.features.known5 import KNOWN5
from kernelrule.features.loader import load_generated
from kernelrule.rules.checks import check_rule, fitter_for, limits_for

SRC = "a6000"
DSTS = ("5090", "4090", "h100")
OUT = "docs/artifacts/transfer-generated.json"
#: The run whose rule and axes are moved. ★ Fixed here, not chosen later.
RUN = "cap1"


def source_registry(table) -> tuple[FeatureRegistry, dict]:
    """★ The **source run's** axes, rebuilt on `table`.

    Three layers, in the order the run itself built them:

    ```
    known5      the F2 starting point
    stage 1     what the FeatureWriter made before the loop
    the loop    what it made during the loop (D-160)
    ```

    ⚠️ `shape_level` is re-derived **on this table** — that is the whole
    point of (a) above, and `load_generated(table=...)` already does it for
    the stage-1 layer (D-67).
    """
    reg = FeatureRegistry(f"{RUN}-on-{getattr(table, 'bundle', '?')}")
    origin: dict[str, str] = {}
    for n in sorted(KNOWN5._items):
        f = KNOWN5[n]
        is_shape, _ = detect_shape_level(replace(f, shape_level=False), table)
        reg.add(replace(f, shape_level=is_shape))
        origin[n] = "known5"
    for f in load_generated(f"runs/{RUN}/stage1-features/proposals.jsonl",
                            table=table):
        if f.name not in reg._items:
            reg.add(f)
            origin[f.name] = "stage1"
    made = Path(f"runs/{RUN}-s0/features.jsonl").read_text().splitlines()
    for line in made:
        if not line.strip():
            continue
        e = json.loads(line)
        if not e.get("accepted") or e["name"] in reg._items:
            continue
        name, fn = compile_feature(e["code"], known=frozenset(reg._items))
        f = Feature(name=name, fn=fn, unit="?", expected_range=(0.0, 1.0),
                    direction="higher_is_worse", code_hash=name,
                    source=e["code"])
        is_shape, _ = detect_shape_level(f, table)
        reg.add(replace(f, shape_level=is_shape))
        origin[name] = f"loop r{e['round']}"
    return reg, origin


def _fit(code, w0, table, matrix, train):
    """Per-regime refit — the same procedure as `transfer_big_rules`
    (principle 2)."""
    fn = compile_rule(code)
    ws, moved, tr = {}, {}, {}
    fit = fitter_for(len(w0))
    for name in ("short", "long"):
        g = [p for p in train if regime_of(p, table.hw) == name]
        if not g:
            continue
        fr = fit_weights(fn, matrix, table, Split("train", tuple(g)), w0,
                         method=fit["fit_method"],
                         n_restarts=fit["fit_restarts"],
                         max_evals=max(300, fit["max_evals"]),
                         objective="regret")
        ws[name] = fr.w
        moved[name] = bool(fr.moved)
        tr[name] = float(fr.fit_regret)
    return fn, ws, moved, tr, fit["fit_method"]


def _constant_axes(reg, matrix, shapes) -> list[str]:
    """Axes with **one value** over those shapes — a branch on one is dead
    (§(b))."""
    out = []
    for n in sorted(reg._items):
        f = reg[n]
        vals = []
        for p in shapes:
            fe, info = matrix.for_shape(p)
            v = (np.full(1, float(getattr(info, n))) if f.shape_level
                 else np.asarray(getattr(fe, n), dtype=np.float64))
            vals.append(v)
        col = np.concatenate(vals)
        if float(col.std()) == 0.0:
            out.append(n)
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    rule_path = Path(f"runs/{RUN}-s0/archive.jsonl")
    arc = [json.loads(x) for x in rule_path.read_text().splitlines()
           if x.strip()]
    rule = min(arc, key=lambda a: a["regret"])
    print("=" * 88)
    print(f"Does a rule built on generated axes transfer — source {RUN}, "
          f"0 LLM calls")
    print("=" * 88)
    print(f"  rule: archive best · regret {rule['regret']:.4f} · "
          f"weights {len(rule['w'])} · cell {rule['cell']}")

    t0 = time.perf_counter()
    A = PerfTable.from_bundle(TABLES[SRC]["bundle"],
                              env_hash=TABLES[SRC]["env_hash"], ok_only=False)
    regA, origin = source_registry(A)
    mA = FeatureMatrix(A, regA)
    spA = _splits(A)
    print(f"  source registry {len(regA._items)} axes "
          f"(known5 {sum(1 for v in origin.values() if v == 'known5')} + "
          f"stage1 {sum(1 for v in origin.values() if v == 'stage1')} + "
          f"loop {sum(1 for v in origin.values() if v.startswith('loop'))})"
          f"  built in {time.perf_counter() - t0:.0f}s")
    shapeA = {n for n in regA._items if regA[n].shape_level}
    constA = _constant_axes(regA, mA, list(spA.train.shapes))
    print(f"  on the source: shape level {sorted(shapeA)} · "
          f"constant on train {constA}")

    res: dict = {"run": RUN, "src": SRC,
                 "rule": {"regret_src": rule["regret"],
                          "weights": len(rule["w"]), "cell": rule["cell"]},
                 "src_registry": {"n": len(regA._items), "origin": origin,
                                  "shape_level": sorted(shapeA),
                                  "constant_on_train": constA},
                 "targets": {}}

    for dst in DSTS:
        D = TABLES[dst]
        print("\n" + "=" * 88)
        print(f"{SRC} -> {dst}")
        print("=" * 88)
        t0 = time.perf_counter()
        B = PerfTable.from_bundle(D["bundle"], env_hash=D["env_hash"],
                                  ok_only=False)
        regB, _ = source_registry(B)
        t_reg = time.perf_counter() - t0
        t0 = time.perf_counter()
        mB = FeatureMatrix(B, regB)
        t_mat = time.perf_counter() - t0
        spB = _splits(B)
        common = {(p.M, p.N, p.K) for p in common_shapes(A, B)}
        hold = [p for p in spB.val.shapes if (p.M, p.N, p.K) in common]

        shapeB = {n for n in regB._items if regB[n].shape_level}
        flipped = sorted((shapeA | shapeB) - (shapeA & shapeB))
        constB = _constant_axes(regB, mB, list(spB.train.shapes))
        new_const = sorted(set(constB) - set(constA))
        code = rule["code"]
        used = set(re.findall(r"[fp]\.(\w+)", code))
        branch = set(re.findall(r"if\s+[^:]*?p\.(\w+)", code))
        print(f"  registry rebuilt {t_reg:.0f}s · ★ matrix {t_mat:.0f}s "
              f"({t_mat / max(len(regB._items), 1):.1f}s per axis, "
              f"{len(regB._items)} axes)")
        print(f"  ★ shape-level verdict flipped: {len(flipped)} {flipped}")
        print(f"  ★ constant on this target but not on the source: "
              f"{len(new_const)} {new_const}")
        print(f"     of those, used by the rule: "
              f"{sorted(set(new_const) & used)} · "
              f"branched on: {sorted(set(new_const) & branch)}")

        rep = check_rule(code, feature_names=regB.names(shape_level=False),
                         shape_value_names=regB.names(shape_level=True),
                         n_weights=len(rule["w"]), limits=limits_for())
        print(f"  static check on the target: "
              f"{'passed' if rep.ok else '⛔ REFUSED'}"
              + ("" if rep.ok else "\n     " + "\n     ".join(rep.violations)))

        t: dict = {"n_train": len(spB.train.shapes), "n_holdout": len(hold),
                   "t_registry_s": round(t_reg, 1),
                   "t_matrix_s": round(t_mat, 1),
                   "n_axes": len(regB._items),
                   "shape_level_flipped": flipped,
                   "constant_new": new_const,
                   "constant_new_used_by_rule": sorted(set(new_const) & used),
                   "constant_new_branched_on": sorted(set(new_const) & branch),
                   "static_ok": bool(rep.ok),
                   "violations": list(rep.violations)}
        if not rep.ok:
            res["targets"][dst] = t
            continue

        st1 = StaticTopK(B, hold, coverage="union").run(ks=(1,))
        t["static_top1"] = float(st1.by_k[1]["all"])
        vend = Path(f"datasets/baselines/vendor-{dst}-{D['env_hash'][:8]}.json")
        if vend.exists():
            from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
            from kernelrule.core.scoring import evaluate
            ev = evaluate(
                vendor_order_fn(B, load_vendor(vend), mapping="nearest"),
                B, hold, ks=(1,), label="vendor")
            t["vendor"] = float(geomean(ev.regret[:, 0]))
        else:
            t["vendor"] = None
        fn = compile_rule(code)
        _, wsA, _, _, _ = _fit(code, rule["w"], A, mA, list(spA.train.shapes))
        t["a_transplant"] = float(_score_on(fn, wsA, B, mB, hold))
        fnB, wsB, moved, tr, meth = _fit(code, rule["w"], B, mB,
                                         list(spB.train.shapes))
        t["b_refit"] = float(_score_on(fnB, wsB, B, mB, hold))
        t["b_train"] = float(geomean(np.array(list(tr.values())))) if tr \
            else float("nan")
        t["moved"] = f"{sum(moved.values())}/{len(moved)}"
        t["fitter"] = meth
        print(f"  holdout {len(hold)} shapes · target train "
              f"{len(spB.train.shapes)}")
        print(f"  {'static top-1':22s} {t['static_top1']:.4f}")
        print(f"  {'vendor':22s} "
              + (f"{t['vendor']:.4f}" if t["vendor"] else "★ none"))
        print(f"  {'(a) transplant':22s} {t['a_transplant']:.4f}")
        print(f"  {'(b) refit':22s} {t['b_refit']:.4f}   train "
              f"{t['b_train']:.4f}  moved {t['moved']}  {meth}")
        res["targets"][dst] = t

    Path(OUT).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {OUT}")
    print("  ⚠️ the source rule is from a **fold 0** run and the targets are "
          "split `nk11008`. This is a check that the mechanism works; the "
          "numbers are for reference only.")


if __name__ == "__main__":
    main()
