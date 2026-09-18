"""★ What the stage-2 seed already scores on the holdout, and how much the
loop adds (D-173 §1). **0 LLM calls.**

    python3 -m experiments.seed_holdout

`rounds.jsonl` r0 is already **the best of the seed plus six edits**, so the
seed's own holdout was never written down. Without it "how much does the
loop lift the holdout" cannot be split.

## ⛔ 2026-09-18 (D-183) — the two procedures became **one**

```
옛  canonical   ★ 채점기가 학습에서 재적합 -> val
    loop val    ★ 한 번 적합 -> val
    -> 서로 다른 두 값이었고 섞으면 안 됐다

새  canonical_score 는 ⛔ 적합하지 않는다 (D-182)
    -> 적합은 ★ 한 군데서만 일어난다: 학습 분할에서 한 번
    -> 두 값이 ★ 같은 값이 된다
```

⚠️ **그래서 씨앗에 무엇을 먹이는지가 중요해졌다.** `chosen.json` 의 `w0` 는
LLM 이 **제안한 초깃값**이지 적합된 값이 아니다 (`fit_regret` 은 적합 **뒤**의
점수다). 그것을 그대로 채점하면 1.47~2.78 이 나온다 — 측정하려는 것이 아니다.

★ 그러므로 씨앗도 **학습 분할에서 한 번 적합한 뒤** 채점한다. 최종 규칙이
루프의 적합된 가중치로 채점되는 것과 ★ 같은 자다.
"""

from __future__ import annotations

import argparse
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
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate_scores
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry
from kernelrule.rules.checks import fitter_for

GPUS = ("a6000", "5090", "4090", "h100")
OUT = Path("docs/artifacts/seed-holdout.json")
#: ★ D-174 — campaign 2 (`c2-*`, 4 seeds, `nkband`) beside the 48-run one.
CAMPAIGNS = {"nk4": ("nk4", (0, 1, 2), "nkgroup", "campaign-nk4.json"),
             "c2": ("c2", (0, 1, 2, 3), "nkband", "campaign2.json")}
PREFIX, SEEDS, DESIGN, CAMPFILE = CAMPAIGNS["nk4"]


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _fit_then_score(code: str, w0, table, matrix, splits) -> float:
    """★ Fit the weights **once** on train, then read val. ⛔ The loop is
    **not** run — no LLM call, no round.

    ★ Same call the worker makes (`loop._fit_and_score`), so this value can
    stand at the head of `rounds.jsonl`'s series.

    ⚠️ 2026-09-18 (D-183): this used to be called `_loop_val`, which read as
    "run the loop". It does one `fit_weights`. ★ And that fit is **stage 2's
    own** — `chosen.json` stores the LLM's proposed `w0` and the score
    **after** fitting (`fit_regret`), never the fitted vector. Measured: this
    call reproduces the recorded `fit_regret` on **16/16** seeds to 4.6e-07.
    """
    fn = compile_rule(code)
    ft = fitter_for(len(w0))
    fr = fit_weights(fn, matrix, table, splits.train, np.asarray(w0, float),
                     max_evals=ft["max_evals"], val_split=splits.val,
                     objective="regret", method=ft["fit_method"],
                     n_restarts=ft["fit_restarts"])
    ev = evaluate_scores(make_score_of(fn, matrix, fr.w), table,
                         list(splits.train.shapes), ks=(1,))
    # ★ D-183 returns the fitted weights too — the scorer no longer fits,
    #   so whoever owns the fit must hand them over.
    return float(fr.val_regret), float(ev.at(1)), int(fr.moved), fr.w


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", choices=tuple(CAMPAIGNS), default="nk4")
    ap.add_argument("--out", default=str(OUT))
    # ★ The segment table is read back from the artefact — recomputing the
    #   16 canonical scores to change how a ratio is summarised would be
    #   two hours for nothing.
    ap.add_argument("--segments-only", action="store_true")
    a = ap.parse_args()
    global PREFIX, SEEDS, DESIGN, CAMPFILE
    PREFIX, SEEDS, DESIGN, CAMPFILE = CAMPAIGNS[a.campaign]

    if a.segments_only:
        old = json.loads(Path(a.out).read_text())
        old["segments"] = _segments(old["rows"])
        Path(a.out).write_text(json.dumps(old, ensure_ascii=False, indent=1))
        _print_segments(old["segments"])
        return
    camp = json.loads(Path(f"docs/artifacts/{CAMPFILE}").read_text())
    finals = {(r["gpu"], r["fold"], r["seed"]): r
              for r in camp["runs"] if not r.get("missing")}

    rows: list[dict] = []
    print("=" * 104)
    print("★ the stage-2 seed on its own holdout (D-173 §1). 0 LLM calls")
    print("=" * 104)
    print(f"  {'표':6s} {'fold':4s} {'씨앗 학습':>9s} {'★ 씨앗 HO':>10s} "
          f"{'최종 HO(s0)':>11s} {'최종 HO(3시드 중앙)':>18s} "
          f"{'★ 루프 기여':>11s}")
    for gpu in GPUS:
        T = TABLES[gpu]
        from kernelrule.core.table import PerfTable
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        for fold in range(4):
            d = Path(f"runs/{PREFIX}-{gpu}-f{fold}")
            ch = d / "stage2-rule-writer" / "chosen.json"
            if not ch.exists():
                continue
            chosen = json.loads(ch.read_text())
            splits = _splits(table, fold=fold, k=4, design=DESIGN)
            # ★ The registry the seed was written against — the stage-1
            #   state, not a run's grown one.
            reg = _load_stage1(d, base_registry("F2", human=REGISTRY),
                               "F2", table)
            matrix = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
            # ★ D-183 — fit once on train, then score. ⛔ Feeding the raw
            #   `w0` to the scorer would measure the LLM's initial guess.
            lv, ltrain, moved, wfit = _fit_then_score(
                chosen["code"], chosen["w0"], table, matrix, splits)
            cs = canonical_score(chosen["code"], wfit, table=table,
                                 matrix=matrix, splits=splits)
            fin = [finals[(gpu, fold, s)]["holdout"] for s in SEEDS
                   if (gpu, fold, s) in finals]
            row = {
                "gpu": gpu, "fold": fold, "n_holdout": cs.n_holdout,
                "seed_train_recorded": chosen.get("fit_regret"),
                "seed_len_w0": len(chosen["w0"]),
                # -- canonical, comparable with campaign-nk4.json ----------
                "seed_holdout_canonical": round(cs.holdout, 6),
                "seed_in_sample_canonical": round(cs.in_sample, 6),
                "final_holdout_s0": (finals[(gpu, fold, 0)]["holdout"]
                                     if (gpu, fold, 0) in finals else None),
                "final_holdout_median": (round(st.median(fin), 6)
                                         if fin else None),
                # -- the loop's own procedure, comparable with rounds.jsonl -
                "seed_val_loop": round(lv, 6),
                "seed_train_loop": round(ltrain, 6),
                "seed_fit_moved": moved,
            }
            row["loop_gain_canonical_s0"] = (
                round(row["seed_holdout_canonical"]
                      - row["final_holdout_s0"], 6)
                if row["final_holdout_s0"] is not None else None)
            rows.append(row)
            print(f"  {gpu:6s} f{fold:<3d} {row['seed_train_recorded']:9.4f} "
                  f"{row['seed_holdout_canonical']:10.4f} "
                  f"{(row['final_holdout_s0'] or float('nan')):11.4f} "
                  f"{(row['final_holdout_median'] or float('nan')):18.4f} "
                  f"{(row['loop_gain_canonical_s0'] or float('nan')):+11.4f}")

    seg = _segments(rows)
    Path(a.out).write_text(json.dumps(
        {"note": ("★ D-183: the two procedures are now ONE. The scorer does "
                  "not fit (D-182), so the seed is fitted once on the "
                  "training split and then scored — the same shape as the "
                  "final rule. `*_canonical` and `*_loop` are therefore the "
                  "same quantity and must agree; the pair is kept as a "
                  "check, not as two readings."),
         "canonical_equals_loop": True,
         "rows": rows, "segments": seg}, ensure_ascii=False, indent=1))
    print()
    _print_segments(seg)
    print(f"\n  -> {a.out}")


def _segments(rows: list[dict]) -> dict:
    """§1-3 — seed -> r0 -> r5 -> r11, all in the **loop's** procedure."""
    out: dict = {}
    for tag, folds in (("fold0-2", (0, 1, 2)), ("★ fold3", (3,))):
        d = {"seed_to_r0": [], "r0_to_r5": [], "r5_to_r11": [],
             "seed_to_r11": [], "n": 0}
        for r in rows:
            if r["fold"] not in folds:
                continue
            for s in SEEDS:
                p = Path(f"runs/{PREFIX}-{r['gpu']}-f{r['fold']}-s{s}"
                         f"/rounds.jsonl")
                if not p.exists():
                    continue
                rr = _rows(p)
                v = {x["round"]: x.get("best_val_regret") for x in rr}
                if 0 not in v or 11 not in v or 5 not in v:
                    continue
                d["n"] += 1
                d["seed_to_r0"].append(v[0] - r["seed_val_loop"])
                d["r0_to_r5"].append(v[5] - v[0])
                d["r5_to_r11"].append(v[11] - v[5])
                d["seed_to_r11"].append(v[11] - r["seed_val_loop"])
        keys = ("seed_to_r0", "r0_to_r5", "r5_to_r11", "seed_to_r11")
        out[tag] = {"n": d["n"]}
        out[tag]["median"] = {k: round(st.median(d[k]), 6) if d[k] else None
                              for k in keys}
        # ★ ⚠️ Medians do not decompose: the median of the whole is not the
        #   sum of the per-segment medians. A share built from medians read
        #   0% + 30% + 22% = 52%, which is not a decomposition of anything.
        #   The **mean** does decompose exactly, so the share is taken from
        #   means and the medians stay beside it as the robust summary.
        out[tag]["mean"] = {k: round(st.mean(d[k]), 6) if d[k] else None
                            for k in keys}
        out[tag]["improved_seed_to_r11"] = sum(
            1 for x in d["seed_to_r11"] if x < 0)
        tot = out[tag]["mean"]["seed_to_r11"]
        if tot:
            out[tag]["share_of_mean"] = {
                k: f"{out[tag]['mean'][k] / tot:.0%}"
                for k in ("seed_to_r0", "r0_to_r5", "r5_to_r11")}
            # And the per-run share, then its median — a second reading that
            # does not let one run's large total dominate.
            per = {k: [] for k in ("seed_to_r0", "r0_to_r5", "r5_to_r11")}
            for i, whole in enumerate(d["seed_to_r11"]):
                if abs(whole) < 1e-9:
                    continue
                for k, v in per.items():
                    v.append(d[k][i] / whole)
            out[tag]["share_per_run_median"] = {
                k: f"{st.median(v):.0%}" for k, v in per.items() if v}
    return out


def _print_segments(seg: dict) -> None:
    print("★ §1-3 구간별 기여 — ⚠️ 전부 ★ 루프 절차 (rounds.jsonl 과 같은 자)")
    for tag, d in seg.items():
        m, mu = d["median"], d["mean"]
        print(f"  {tag}  n={d['n']}  좋아진 {d['improved_seed_to_r11']}/{d['n']}")
        print(f"    중앙  씨앗->r0 {m['seed_to_r0']:+.4f}  "
              f"r0->r5 {m['r0_to_r5']:+.4f}  r5->r11 {m['r5_to_r11']:+.4f}"
              f"  | 씨앗->r11 {m['seed_to_r11']:+.4f}")
        print(f"    평균  씨앗->r0 {mu['seed_to_r0']:+.4f}  "
              f"r0->r5 {mu['r0_to_r5']:+.4f}  r5->r11 {mu['r5_to_r11']:+.4f}"
              f"  | 씨앗->r11 {mu['seed_to_r11']:+.4f}")
        if "share_of_mean" in d:
            print(f"    ★ 비율(평균 분해) {d['share_of_mean']}")
            print(f"      실행별 비율의 중앙 {d['share_per_run_median']}")


if __name__ == "__main__":
    main()
