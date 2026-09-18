"""★ 홀드아웃을 val / test 로 쪼개 라운드를 고르면 나은가 (D-184).
**⛔ LLM 0회 · 적합 0회.**

    python3 -m experiments.valtest_split --gpu <g>      # 갈래
    python3 -m experiments.valtest_split --merge <f...>

라운드별 규칙은 이미 있다 — `bests.jsonl` 에 라운드마다 `code` 와 `w` 가 있다
(12줄 × 64실행). ⛔ 새로 만들지 않고 저장된 것을 쓴다.

```
지금   ★ 12라운드 중 ★ 학습 regret 이 가장 낮은 규칙을 홀드아웃에서 잰다
물음   ★ 홀드아웃을 반으로 쪼개 ★ val 로 라운드를 고르면 나은가
```

## ★ 같은 test 에서 셋

```
(A) ★ 학습 regret 최고 규칙을 test 에서    ← 지금 방식
(B) ★ val 최고 규칙을 test 에서            ← 새 방식
(C) ★ test 최고 규칙을 test 에서           ← ★ 오라클 상한
```

⛔ 셋을 **같은 `test`** 에서 잰다. 그래서 형상별 regret 을 라운드마다 **한 번**
내고, 쪼개기 20회는 그 위에서 평균만 바꾼다 — 채점은 반복되지 않는다.

## 층화

`M` 크기 3분위로 맞춰 쪼갠다. ⚠️ **roofline 으로는 안 된다** — `mem` 이 fold 당
0~6개이고 4090 f0 · h100 f0 은 val 에 `mem` 이 0개가 된다.

## ⚠️ 유보

```
⚠️ fold0 은 (4096,4096) 한 묶음 17형상이다
   -> val 과 test 가 ★ 같은 (N,K) 를 나눠 갖는다
   ★ train 대비로는 둘 다 안 본 레이어라 목적에는 맞지만
   ★ val 과 test 가 서로 독립은 아니다
⚠️ val 7~9형상 · test 7~9형상 — ★ 둘 다 작다
```

## 판정선 (⛔ 결과를 보기 전에 박는다)

```
★ (B − A) 의 중앙이 ★ 쪼개기 20회의 흔들림보다 크다   -> ★ 쓸 만하다
★ 그보다 작거나 음수다                                -> ★ 안 쓴다
★ (C − A) 는 완벽히 골랐을 때의 ★ 상한으로 보고한다
```
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
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.scoring import evaluate_scores
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry, load_generated

GPUS = ("a6000", "5090", "4090", "h100")
SEEDS = (0, 1, 2, 3)
N_SPLITS = 20
OUT = Path("docs/artifacts/valtest-split.json")


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _seed_of(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def _halve(shapes, rng) -> tuple[list[int], list[int]]:
    """★ M 3분위로 층화해 반씩. 홀수 층은 번갈아 한 쪽에 더 준다."""
    m = np.array([p.M for p in shapes], float)
    q1, q2 = float(np.quantile(m, 1 / 3)), float(np.quantile(m, 2 / 3))
    band = [0 if x <= q1 else (1 if x <= q2 else 2) for x in m]
    val, test = [], []
    for b in (0, 1, 2):
        idx = [i for i, x in enumerate(band) if x == b]
        idx = [idx[int(j)] for j in rng.permutation(len(idx))]
        h = len(idx) // 2
        extra = len(idx) - 2 * h
        # ★ 남는 하나는 번갈아 — 한쪽이 계속 커지지 않게
        if extra and rng.random() < 0.5:
            val += idx[:h + 1]
            test += idx[h + 1:]
        else:
            val += idx[:h]
            test += idx[h:]
    return sorted(val), sorted(test)


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.merge:
        rows: list[dict] = []
        for f in a.merge:
            rows += json.loads(Path(f).read_text())["rows"]
        res = {"n_runs": len(rows), "n_splits": N_SPLITS,
               "verdict_line": ("(B-A) median vs the spread over 20 splits; "
                                "fixed before looking (D-184 §4)"),
               "rows": rows, "summary": _summary(rows)}
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"merged -> {a.out}  ({len(rows)} runs)")
        _print(res["summary"])
        return

    rows, skipped = [], []
    print("=" * 104)
    print("★ val/test 쪼개기로 라운드 고르기 (D-184). ⛔ LLM 0회 · 적합 0회")
    print("=" * 104)
    for gpu in (GPUS if a.gpu is None else (a.gpu,)):
        T = TABLES[gpu]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        for fold in range(4):
            sp = _splits(table, fold=fold, k=4, design="nkband")
            hold = list(sp.val.shapes)
            base = Path(f"runs/c2-{gpu}-f{fold}")
            for seed in SEEDS:
                tag = f"c2-{gpu}-f{fold}-s{seed}"
                d = Path("runs") / tag
                bp = d / "bests.jsonl"
                if not bp.exists():
                    skipped.append({"run": tag, "why": "no bests.jsonl"})
                    continue
                best = _rows(bp)
                reg = _load_stage1(base,
                                   base_registry("F2", human=REGISTRY),
                                   "F2", table)
                fp = d / "features.jsonl"
                if fp.exists():
                    for f in load_generated(fp, table=table):
                        if f.name not in reg._items:
                            reg.add(f)
                m = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
                # ★ 형상별 regret 을 라운드마다 ★ 한 번만 낸다
                per = []
                bad = None
                for e in best:
                    try:
                        per.append(_per_shape(e["code"], e["w"], table, m,
                                              hold))
                    except Exception as exc:                # noqa: BLE001
                        bad = f"round {e['round']}: {exc}"
                        break
                if bad:
                    skipped.append({"run": tag, "why": bad})
                    print(f"  ⚠️ {tag} 건너뜀 — {bad}")
                    continue
                per = np.array(per)                        # (rounds, shapes)
                # ★ (A) 학습 regret 최고 = 마지막 줄 (누적 최고)
                ia = int(np.argmin([e["regret"] for e in best]))
                rng = np.random.default_rng(_seed_of(tag, "valtest"))
                sp_rows = []
                for it in range(N_SPLITS):
                    vi, ti = _halve(hold, rng)
                    gv = np.exp(np.log(per[:, vi]).mean(axis=1))
                    gt = np.exp(np.log(per[:, ti]).mean(axis=1))
                    ib, ic = int(np.argmin(gv)), int(np.argmin(gt))
                    sp_rows.append({
                        "i": it, "n_val": len(vi), "n_test": len(ti),
                        "A": float(gt[ia]), "B": float(gt[ib]),
                        "C": float(gt[ic]),
                        "round_A": best[ia]["round"],
                        "round_B": best[ib]["round"],
                        "round_C": best[ic]["round"],
                        "val_picked_test_best": ib == ic,
                        # ★ 층화가 됐나 — M 3분위 구성
                        "val_bands": _bands(hold, vi),
                        "test_bands": _bands(hold, ti)})
                rows.append({
                    "run": tag, "gpu": gpu, "fold": fold, "seed": seed,
                    "n_holdout": len(hold), "n_rounds": len(best),
                    "n_distinct_rules": len({e["rule_id"] for e in best}),
                    "round_A": best[ia]["round"],
                    "BA_median": round(float(st.median(
                        [r["B"] - r["A"] for r in sp_rows])), 6),
                    "CA_median": round(float(st.median(
                        [r["C"] - r["A"] for r in sp_rows])), 6),
                    "BA_spread": round(float(
                        max(r["B"] - r["A"] for r in sp_rows)
                        - min(r["B"] - r["A"] for r in sp_rows)), 6),
                    "agree_frac": round(sum(
                        r["val_picked_test_best"] for r in sp_rows)
                        / len(sp_rows), 4),
                    "splits": sp_rows})
                r = rows[-1]
                print(f"  {tag:20s} n={len(hold)}  규칙 "
                      f"{r['n_distinct_rules']}  (B−A) 중앙 "
                      f"{r['BA_median']:+.4f}  (C−A) {r['CA_median']:+.4f}  "
                      f"val이 test최고를 맞힘 {r['agree_frac']:.0%}",
                      flush=True)
    out = Path(a.out)
    out.write_text(json.dumps({"rows": rows, "skipped": skipped,
                               "n_splits": N_SPLITS},
                              ensure_ascii=False, indent=1))
    print(f"\n  건너뛴 실행 {len(skipped)}\n  -> {out}")


def _bands(hold, idx) -> dict:
    m = np.array([p.M for p in hold], float)
    q1, q2 = float(np.quantile(m, 1 / 3)), float(np.quantile(m, 2 / 3))
    b = [0 if m[i] <= q1 else (1 if m[i] <= q2 else 2) for i in idx]
    return {str(k): b.count(k) for k in (0, 1, 2)}


def _per_shape(code, w, table, matrix, hold) -> list[float]:
    """★ 저장된 `w` 를 그대로 쓴다 — ⛔ 적합하지 않는다."""
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import make_score_of

    fn = compile_rule(code)
    e = evaluate_scores(make_score_of(fn, matrix, np.asarray(w, float)),
                        table, hold, ks=(1,))
    order = {(p.M, p.N, p.K): i for i, p in enumerate(e.shapes)}
    return [float(e.regret[order[(p.M, p.N, p.K)], 0]) for p in hold]


def _summary(rows: list[dict]) -> dict:
    ba = [r["BA_median"] for r in rows]
    ca = [r["CA_median"] for r in rows]
    sp = [r["BA_spread"] for r in rows]
    return {"n_runs": len(rows),
            "BA_median_of_medians": round(st.median(ba), 6),
            "BA_mean": round(st.mean(ba), 6),
            "BA_spread_median": round(st.median(sp), 6),
            "B_better_than_A": sum(1 for x in ba if x < 0),
            "B_worse_than_A": sum(1 for x in ba if x > 0),
            "CA_median_of_medians": round(st.median(ca), 6),
            "agree_median": round(st.median(
                [r["agree_frac"] for r in rows]), 4),
            "★ verdict": ("쓸 만하다" if st.median(ba) < -st.median(sp)
                          else "안 쓴다")}


def _print(s: dict) -> None:
    print(f"\n  ★ 실행 {s['n_runs']}")
    print(f"     (B−A) 중앙 {s['BA_median_of_medians']:+.4f} "
          f"· 평균 {s['BA_mean']:+.4f} "
          f"· ★ 쪼개기 20회 폭 중앙 {s['BA_spread_median']:.4f}")
    print(f"     B 가 A 보다 나은 실행 {s['B_better_than_A']}/{s['n_runs']} "
          f"· 나쁜 실행 {s['B_worse_than_A']}/{s['n_runs']}")
    print(f"     (C−A) 중앙 {s['CA_median_of_medians']:+.4f}  ← 상한")
    print(f"     val 이 test 최고 라운드를 맞힌 비율 중앙 "
          f"{s['agree_median']:.0%}")
    print(f"     ★ 판정 {s['★ verdict']}")


if __name__ == "__main__":
    main()
