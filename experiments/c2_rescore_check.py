"""★ 캠페인 2 재집계 §2 — ★ 대조부터 (D-183). **0 LLM · 채점 다시 안 함.**

    python3 -m experiments.c2_rescore_check            # 64실행 (읽기만, 공짜)
    python3 -m experiments.c2_rescore_check --canonical 6   # 표본 대조

집계 스크립트가 `nkgroup` 분할로 채점했다(`campaign_nk4.py:98` 의 박힌 상수,
pending_fixes 22). ★ 루프 자신이 적은 값은 `nkband` 분할에서 나왔다.

```
rounds.jsonl 마지막 best_val_regret = ★ archive.best 의 val_regret
archive.best  ★ 학습 regret 최소 — 단 갱신에 허용오차를 쓴다
              `_key(e) < _key(best) - noise_tol`
_best_by_train (집계 스크립트)  ★ 단순 min
-> ★ 학습 점수가 거의 같은 두 규칙에서 다른 것을 고를 수 있다
```

⚠️ **왜 두 값이 같아야 하나.** `fit_weights(val_split=...)` 는 적합이 끝난
뒤 **그 가중치로** val 을 잰다 — 적합하지 않는다. D-182 뒤의
`canonical_score` 도 같다. ★ 그러므로 두 절차는 같은 자다.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

OUT = Path("docs/artifacts/c2-rescore-check.json")
GPUS = ("a6000", "5090", "4090", "h100")
SEEDS = (0, 1, 2, 3)


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=int, default=0,
                    help="이 개수만큼 canonical_score 로도 대조한다")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    rows, missing = [], []
    for gpu in GPUS:
        for fold in range(4):
            for seed in SEEDS:
                tag = f"c2-{gpu}-f{fold}-s{seed}"
                d = Path("runs") / tag
                if not (d / "rounds.jsonl").exists():
                    missing.append(tag)
                    continue
                rr = _rows(d / "rounds.jsonl")
                arc = _rows(d / "archive.jsonl")
                last = rr[-1]
                mn = min(arc, key=lambda e: e["regret"])
                rows.append({
                    "run": tag, "gpu": gpu, "fold": fold, "seed": seed,
                    "n_rounds": len(rr),
                    # ★ 루프가 적은 값
                    "loop_val": last["best_val_regret"],
                    "loop_train": last["best_regret"],
                    # ★ 단순 min 으로 고른 규칙
                    "min_rule": mn["rule_id"], "min_train": mn["regret"],
                    "best_rule": _rows(d / "bests.jsonl")[-1]["rule_id"],
                    "min_val": mn["val_regret"],
                    "same_pick": abs(last["best_regret"]
                                     - mn["regret"]) < 1e-12,
                    "val_diff": abs(last["best_val_regret"]
                                    - mn["val_regret"])})
    print("=" * 96)
    print("★ 캠페인 2 §2 대조 (D-183). 0 LLM · 채점 다시 안 함")
    print("=" * 96)
    diff = [r for r in rows if r["val_diff"] > 1e-9]
    print("\n  ① rounds.jsonl 마지막 vs 아카이브 단순 min")
    print(f"     실행 {len(rows)}개 · 빠진 실행 {len(missing)}")
    print(f"     ★ 고른 규칙이 같은 실행 {sum(1 for r in rows if r['same_pick'])}"
          f"/{len(rows)}")
    print(f"     ★ 홀드아웃 값이 다른 실행 ★ {len(diff)}/{len(rows)}")
    for r in diff[:8]:
        print(f"       {r['run']:20s} 루프 {r['loop_val']:.6f} "
              f"단순min {r['min_val']:.6f}  차 {r['val_diff']:.2e} "
              f"(학습 {r['loop_train']:.6f} vs {r['min_train']:.6f})")

    canon = []
    if a.canonical:
        canon = _canonical(rows[:: max(1, len(rows) // a.canonical)][:a.canonical])
    res = {"n_runs": len(rows), "missing": missing,
           "n_same_pick": sum(1 for r in rows if r["same_pick"]),
           "n_val_differs": len(diff),
           "max_val_diff": max((r["val_diff"] for r in rows), default=0.0),
           "canonical_checks": canon,
           "why": ("fit_weights(val_split=...) scores the fitted weights on "
                   "val without refitting; D-182's canonical_score does the "
                   "same — the two are the same quantity"),
           "rows": rows}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


def _canonical(pick: list[dict]) -> list[dict]:
    """★ 표본 몇 개를 `canonical_score` (D-182 절차 · nkband) 로도 낸다."""
    import numpy as np

    import kernelrule.features.physical  # noqa: F401
    from experiments.f1_pipeline import _load_stage1, _splits
    from experiments.transfer_29_5 import TABLES
    from kernelrule.core.canonical import canonical_score
    from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY
    from kernelrule.features.loader import base_registry, load_generated

    out = []
    print(f"\n  ② canonical_score 대조 — 표본 {len(pick)}개 "
          f"(D-182 절차 · nkband 분할)")
    for r in pick:
        tag, gpu, fold = r["run"], r["gpu"], r["fold"]
        d = Path("runs") / tag
        base = Path("runs") / f"c2-{gpu}-f{fold}"
        T = TABLES[gpu]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        splits = _splits(table, fold=fold, k=4, design="nkband")
        reg = _load_stage1(base, base_registry("F2", human=REGISTRY), "F2",
                           table)
        fp = d / "features.jsonl"
        if fp.exists():
            for f in load_generated(fp, table=table):
                if f.name not in reg._items:
                    reg.add(f)
        m = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
        # ★ the rule the **archive** ended with — `bests.jsonl` last line.
        #   ⛔ not `min(archive)`: on an exact tie those differ (measured on
        #   c2-h100-f2-s0, where three rules tie at 1.053852132000).
        e = _rows(d / "bests.jsonl")[-1]
        cs = canonical_score(e["code"], np.asarray(e["w"], float),
                             table=table, matrix=m, splits=splits)
        out.append({"run": tag, "loop_val": r["loop_val"],
                    "canonical": round(cs.holdout, 9),
                    "diff": abs(cs.holdout - r["loop_val"]),
                    "n_holdout": cs.n_holdout,
                    "n_val_expected": len(splits.val.shapes)})
        o = out[-1]
        print(f"     {tag:20s} 루프 {o['loop_val']:.6f}  "
              f"canonical {o['canonical']:.6f}  ★ 차 {o['diff']:.2e}  "
              f"n={o['n_holdout']}/{o['n_val_expected']}")
    bad = [o for o in out if o["diff"] > 1e-9]
    print(f"     ★ 불일치 {len(bad)}/{len(out)}")
    return out


if __name__ == "__main__":
    main()
