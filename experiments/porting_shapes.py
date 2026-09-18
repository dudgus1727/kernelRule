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

## ⛔ 2026-09-18 (D-182) — `_refit` is **no longer** `canonical_score`

`_refit` below **fits**, and that is correct: fitting the weights on N target
shapes **is the thing this experiment measures**. What changed is the other
side — `canonical_score` stopped fitting at D-182, and stopped splitting by
regime at D-179.

```
_refit            ★ N 형상으로 가중치를 맞춘다 — ★ 이 실험의 본체
canonical_score   ⛔ 적합하지 않는다 — 받은 가중치로 채점만 한다
-> ★ 두 절차는 더 이상 같지 않다. `--verify` 의 동일성 검사는 뜻이 없다
```

⛔ **2026-09-18 (D-186) — 고쳤다.** D-179 이후 이 파일은 `_refit` 이
`regime_of` 를 축 없이 불러 ★ 아예 돌지 않았다. 이제 `_refit` 이 **한 벌**을
맞춘다 — 적합은 유지하고(그것이 본체다) 체제 분할만 없앴다.

⚠️ D-177 · D-178 의 옛 수치는 **두 벌 적합 + nkgroup 원주민** 으로 나온 것이고
기록에 그대로 남는다. 이 파일이 지금 내는 값은 ★ 다른 수다.

## ⚠️ What happened when the sample missed a regime (⛔ 옛 절차)

`canonical_score` **then** fitted per regime. With N=4 a uniform sample can easily
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
from experiments.c2_ref import label, native, transfer
from experiments.f1_pipeline import _load_stage1, _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import Split, regime_of
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
    """★ Fit **one** weight vector on `sample`, read it on the whole of
    `val`.

    ★ This is the experiment's subject — "how many target shapes does a
    transfer need to refit on". ⛔ The fit **stays** (D-186 §1); what left is
    the regime split.

    ```
    ⛔ 2026-09-18 (D-186): 예전에는 `short`/`long` 두 벌을 맞췄고, 표본이 한
       체제를 빠뜨리면 그 체제를 표본 전체로 맞추는 대체 절차가 있었다.
       ★ D-179 가 그 축을 지웠고 `regime_of` 가 축 없이 불리면 이제 raise 한다
       -> 이 파일은 그동안 ★ 아예 돌지 않았다
    ★ 지금은 한 벌이다 — 대체 절차도 필요 없다
    ```

    ⛔ **Not** `canonical_score`, which since D-182 does not fit at all.
    """
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import fit_weights, make_score_of

    fn = compile_rule(code)
    fit = fit_weights(fn, matrix, table, Split("train", tuple(sample)),
                      np.asarray(w0, float), max_evals=max_evals,
                      objective="regret")
    e = evaluate_scores(make_score_of(fn, matrix, fit.w), table, val, ks=(1,))
    return {"holdout": float(geomean(e.regret[:, 0])),
            "n_holdout": len(e.shapes),
            "weights": {"all": [float(x) for x in fit.w]},
            # ⛔ D-186 — 체제가 없으니 대체도 없다. 키는 기록 호환으로 남긴다
            "pooled_regimes": [],
            "moved": bool(fit.moved), "n_moved": int(fit.moved), "n_fits": 1}


def _verify(code, w0, *, table, matrix, sample, splits) -> float | None:
    """⛔ 2026-09-18 (D-182) — this check is **retired**, not fixed.

    It compared `_refit` against `canonical_score` on the same train set and
    required them to agree (measured 0.0 across 12 directions). ★ They are
    no longer the same procedure: `_refit` fits, `canonical_score` does not.
    Making them agree again would mean putting the fit back into the scorer,
    which is exactly what D-182 removed.
    """
    raise SystemExit(
        "porting_shapes._verify: `_refit` and `canonical_score` are no "
        "longer the same procedure (D-182 removed the fit from the "
        "scorer). The identity this checked held when D-177/D-178 ran and "
        "is recorded there. ⛔ Do not restore it by refitting in the "
        "scorer.")
    return None


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
            # ★ D-186 — 옛 값이 박힌 chosen.json 대신 ★ 재집계된 c2 를 본다
            tr = transfer(src, dst, FOLD)
            e["a_as_is"], e["b_refit"] = tr["a_as_is"], tr["b_refit"]
            e["native"] = native(dst, FOLD)
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
                              # ⛔ D-186 — 옛 `n_short` 는 없어진 축의 값이라
                              #   뺐다. roofline 쪽 구성은 porting_strat 이 센다
                              "n_mem": sum(
                                  1 for p in s
                                  if regime_of(p, table.hw, axis="roofline")
                                  == "mem")})
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
                               "note": label(), "rows": rows},
                              ensure_ascii=False, indent=1))
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
