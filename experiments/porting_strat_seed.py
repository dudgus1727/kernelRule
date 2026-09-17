"""★ (b) 층화 루프 실행의 씨앗 — 방향 x N (D-178 §2). **0 LLM calls.**

    python3 -m experiments.porting_strat_seed

D-177 의 `porting_shapes_seed` 와 같은 자리이고, 다른 것은 **뽑기**뿐이다.

```
씨앗   ★ pc-<src>2<dst>-f0 와 같은 것 — 옮긴 규칙. ⛔ stage 2 는 안 돈다
학습   ★ 층화로 뽑은 N 형상만 (`--train-shapes`)
뽑기   ★ (a) 의 ★ 첫 반복 (rep 0) 을 고정해 쓴다
평가   ⛔ 손대지 않는다 — 홀드아웃은 fold0 전체
태그   ps2-<src>2<dst>-n<N>-f0   (D-177 의 ps- 와 섞이지 않게)
```
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

from experiments.porting_shapes import FOLD, GPUS, NS, _seed_of
from experiments.porting_strat import (
    m_tertiles,
    sample_stratified,
    strata_of,
)
from experiments.transfer_29_5 import TABLES

PICK_REP = 0
OUT_TAG = "ps2"


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="docs/artifacts/porting-strat.json")
    a = ap.parse_args()

    from experiments.f1_pipeline import _splits
    from kernelrule.core.table import PerfTable

    curve = {}
    p = Path(a.curve)
    if p.exists():
        for r in json.loads(p.read_text())["rows"]:
            rep = next((x for x in r["reps"] if x["rep"] == PICK_REP), None)
            curve[(r["src"], r["dst"], r["N"])] = rep

    made = []
    print("=" * 96)
    print(f"★ (b) 층화 루프 씨앗 — 방향 x N (D-178). 뽑기는 rep {PICK_REP}")
    print("=" * 96)
    for dst in GPUS:
        T = TABLES[dst]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        train = list(_splits(table, fold=FOLD, k=4,
                             design="nkband").train.shapes)
        strata = strata_of(train, table, m_tertiles(train))
        for src in GPUS:
            if src == dst:
                continue
            base = Path(f"runs/pc-{src}2{dst}-f{FOLD}")
            if not (base / "stage2-rule-writer" / "chosen.json").exists():
                continue
            ch = json.loads(
                (base / "stage2-rule-writer" / "chosen.json").read_text())
            for n in NS:
                tag = f"{OUT_TAG}-{src}2{dst}-n{n}-f{FOLD}"
                d = Path("runs") / tag
                if (d / "stage3-evolution").exists():
                    print(f"  ⚠️ {tag} 이미 돌았다 — 건너뜀")
                    continue
                d.mkdir(parents=True, exist_ok=True)
                if (d / "stage1-features").exists():
                    shutil.rmtree(d / "stage1-features")
                shutil.copytree(base / "stage1-features",
                                d / "stage1-features")
                seed = _seed_of("strat", src, dst, n, PICK_REP)
                pick, alloc = sample_stratified(strata, n, seed)
                (d / "train-shapes.json").write_text(json.dumps(
                    [[q.M, q.N, q.K] for q in pick]))
                rep = curve.get((src, dst, n))
                (d / "stage2-rule-writer").mkdir(exist_ok=True)
                (d / "stage2-rule-writer" / "chosen.json").write_text(
                    json.dumps({
                        "source": f"ported:{src}->{dst}",
                        "code": ch["code"], "w0": ch["w0"],
                        "fit_regret": (rep["holdout"] if rep
                                       else ch["fit_regret"]),
                        "why": ("★ D-178 — the ported rule as a seed, fitted "
                                "on N ★ stratified target shapes. The "
                                "holdout is the whole fold."),
                        "N": n, "sampling": "stratified",
                        "alloc": alloc,
                        "pick_rep": PICK_REP, "sample_seed": seed,
                        "train_shapes": [[q.M, q.N, q.K] for q in pick],
                        "a_refit_holdout": rep["holdout"] if rep else None,
                        "src_run": ch.get("src_run"),
                        "b_refit_full": ch["b_refit"],
                        "a_as_is": ch["a_as_is"],
                        "native": ch["native"]}, ensure_ascii=False,
                        indent=1))
                made.append(tag)
                av = f"{rep['holdout']:.4f}" if rep else "—"
                print(f"  {tag:30s} N={n:2d}  (a) {av}  "
                      f"원주민 {ch['native']:.4f}  층 {alloc}")
    print(f"\n  ★ {len(made)} runs")


if __name__ == "__main__":
    main()
