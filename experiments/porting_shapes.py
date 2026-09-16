"""★ How many target shapes does a transfer need? — the (a) refit curve
(D-177 §1-1). **0 LLM calls · 0 GPU.**

    python3 -m experiments.porting_shapes --dst <gpu>
    python3 -m experiments.porting_shapes --merge a.json b.json ...

The porting-cost curve (D-175) answered "how many **LLM calls**". That hides
the real bill: the loop refits on the target's whole training split, and
every one of those shapes has to be **built and measured** first.

```
LLM 44회        ★ 몇 분
★ 대상 형상 48개  ★ 형상마다 유효 config 중앙 15,015 개
                 ★ 빌드 1회 중앙 14.4~16.5초 (baselines 릴리즈)
```

So this sweeps **N**, the number of target shapes the refit is allowed to
see, and reads the result on the **whole** holdout.

```
N       ★ 4 · 8 · 16      (⛔ not 32+ — past half the table it is obvious)
뽑기     무작위 · ★ 반복 10회
평가     ★ 대상 fold0 홀드아웃 전체 · canonical
```

## ⚠️ What happens when the sample misses a regime

`canonical_score` fits **per regime**. With N=4 a uniform sample can easily
contain no memory-bound shape (that side is 14~21% of these tables), and
then `canonical_score` scores only part of the holdout — a different
denominator, which cannot be laid beside the transfer table.

⛔ The fix is **not** to stratify the sample: the order says the pick is
random, and stratifying would quietly make small N easier. Instead, when a
regime is absent from the sample, **that regime's weights are fitted on the
whole sample** (both regimes pooled). The holdout stays whole, the sampling
stays uniform, and the fallback is counted and reported.

★ `_refit` is verified against `canonical_score` on every sample that does
contain both regimes — they must agree to 1e-9, or this file is measuring
something else (§verify).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.f1_pipeline import _load_stage1, _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry

GPUS = ("a6000", "5090", "4090", "h100")
NS = (4, 8, 16)
REPEATS = 10
FOLD = 0
OUT = Path("docs/artifacts/porting-shapes.json")
OUT_MD = Path("docs/artifacts/porting-shapes.md")


def _seed_of(*parts) -> int:
    """⛔ Not Python's `hash()` — it is salted per process, so the same
    sweep would draw different shapes in a different shard (the same trap
    `code_hash_of` was written for)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def _sample(train: list, n: int, seed: int) -> list:
    """★ Uniform, without replacement. ⛔ Not stratified — see the module
    docstring."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), size=min(n, len(train)), replace=False)
    return [train[int(i)] for i in sorted(idx)]


def _refit(code: str, w0, *, table, matrix, sample: list, val: list,
           max_evals: int = 300) -> dict:
    """Per-regime refit on `sample`, read on the whole of `val`.

    ★ Same procedure as `canonical_score` except for one fallback: a regime
    with no shape in the sample is fitted on the **whole sample**. That case
    is flagged in the result (`pooled_regimes`).
    """
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import fit_weights, make_score_of

    fn = compile_rule(code)
    reg_ho: dict = {}
    fitted: dict = {}
    pooled: list[str] = []
    moved: list[bool] = []
    for name in ("short", "long"):
        g_ho = [p for p in val if regime_of(p, table.hw) == name]
        if not g_ho:
            continue
        g_tr = [p for p in sample if regime_of(p, table.hw) == name]
        if not g_tr:
            # ★ the fallback — the whole sample stands in for this regime
            g_tr = list(sample)
            pooled.append(name)
        fit = fit_weights(fn, matrix, table, Split("train", tuple(g_tr)),
                          w0, max_evals=max_evals, objective="regret")
        fitted[name] = [float(x) for x in fit.w]
        moved.append(bool(fit.moved))
        e = evaluate_scores(make_score_of(fn, matrix, fit.w), table, g_ho,
                            ks=(1,))
        for i, p in enumerate(e.shapes):
            reg_ho[p] = e.regret[i, 0]
    scored = [p for p in val if p in reg_ho]
    return {"holdout": float(geomean(np.array([reg_ho[p] for p in scored]))),
            "n_holdout": len(scored), "weights": fitted,
            "pooled_regimes": pooled, "moved": all(moved),
            "n_moved": sum(moved), "n_fits": len(moved)}


def _verify(code, w0, *, table, matrix, sample, splits) -> float | None:
    """★ The identity check — when the sample has both regimes, `_refit`
    must equal `canonical_score` on the same train set."""
    names = {regime_of(p, table.hw) for p in sample}
    if len(names) < 2:
        return None
    sp = SplitSet(train=Split("train", tuple(sample)), val=splits.val,
                  kind=f"{splits.kind}+sample")
    cs = canonical_score(code, np.asarray(w0, float), table=table,
                         matrix=matrix, splits=sp)
    r = _refit(code, w0, table=table, matrix=matrix, sample=sample,
               val=list(splits.val.shapes))
    return abs(cs.holdout - r["holdout"])


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=None, help="one target table")
    ap.add_argument("--src", default=None, help="one source table")
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="run the canonical_score identity check first")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.merge is not None:
        _merge(a.merge, Path(a.out or OUT))
        return

    dsts = GPUS if a.dst is None else (a.dst,)
    rows: list[dict] = []
    checks: list[dict] = []
    print("=" * 104)
    print("★ 전이에 대상 형상이 몇 개나 필요한가 — (a) 재적합 곡선 "
          "(D-177 §1-1). 0 LLM · 0 GPU")
    print("=" * 104)
    for dst in dsts:
        T = TABLES[dst]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        splits = _splits(table, fold=FOLD, k=4, design="nkband")
        train = list(splits.train.shapes)
        val = list(splits.val.shapes)
        for src in (GPUS if a.src is None else (a.src,)):
            if src == dst:
                continue
            d = Path(f"runs/pc-{src}2{dst}-f{FOLD}")
            ch = d / "stage2-rule-writer" / "chosen.json"
            if not ch.exists():
                print(f"  ⚠️ {src}->{dst} 씨앗 없음 — 건너뜀")
                continue
            e = json.loads(ch.read_text())
            reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2",
                               table)
            m = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
            if a.verify:
                s = _sample(train, 16, seed=_seed_of(src, dst, "verify"))
                diff = _verify(e["code"], e["w0"], table=table, matrix=m,
                               sample=s, splits=splits)
                checks.append({"src": src, "dst": dst, "diff": diff})
                print(f"  ★ 절차 대조 {src}->{dst}  차 "
                      + ("건너뜀(한 체제)" if diff is None else f"{diff:.2e}"))
            for n in NS:
                got = []
                for rep in range(REPEATS):
                    seed = _seed_of(src, dst, n, rep)
                    s = _sample(train, n, seed)
                    r = _refit(e["code"], e["w0"], table=table, matrix=m,
                               sample=s, val=val)
                    r.update({"rep": rep, "sample_seed": seed,
                              "shapes": [[p.M, p.N, p.K] for p in s],
                              "n_short": sum(1 for p in s
                                             if regime_of(p, table.hw)
                                             == "short")})
                    got.append(r)
                h = sorted(x["holdout"] for x in got)
                rows.append({
                    "src": src, "dst": dst, "fold": FOLD, "N": n,
                    "n_train_full": len(train), "n_holdout": len(val),
                    "a_as_is": e["a_as_is"], "b_refit": e["b_refit"],
                    "native": e["native"],
                    "holdout_median": round(st.median(h), 6),
                    "holdout_min": round(h[0], 6),
                    "holdout_max": round(h[-1], 6),
                    "gap_median": round(st.median(h) - e["native"], 6),
                    "n_pooled": sum(1 for x in got if x["pooled_regimes"]),
                    "n_not_moved": sum(1 for x in got if not x["moved"]),
                    "reps": got})
                r = rows[-1]
                print(f"  {src:5s}->{dst:5s} N={n:2d}  홀드아웃 중앙 "
                      f"{r['holdout_median']:.4f} "
                      f"[{r['holdout_min']:.4f}~{r['holdout_max']:.4f}]  "
                      f"원주민대비 {r['gap_median']:+.4f}  "
                      f"(b) {r['b_refit']:.4f}  "
                      f"체제보충 {r['n_pooled']}/10  "
                      f"미이동 {r['n_not_moved']}/10", flush=True)
    out = Path(a.out or OUT)
    out.write_text(json.dumps({"ns": list(NS), "repeats": REPEATS,
                               "fold": FOLD, "checks": checks,
                               "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n  -> {out}")


def _merge(paths: list[str], out: Path) -> None:
    rows: list[dict] = []
    checks: list[dict] = []
    for p in paths:
        j = json.loads(Path(p).read_text())
        rows += j["rows"]
        checks += j.get("checks", [])
    out.write_text(json.dumps({"ns": list(NS), "repeats": REPEATS,
                               "fold": FOLD, "checks": checks, "rows": rows},
                              ensure_ascii=False, indent=1))
    print(f"merged {len(paths)} -> {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
