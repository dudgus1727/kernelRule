"""★ The (b) loop runs' seeds — one per (direction, N) (D-177 §1-2).
**0 LLM calls.**

    python3 -m experiments.porting_shapes_seed

```
씨앗      ★ D-175 의 pc-<src>2<dst>-f0 와 같은 것 — 옮긴 규칙의 구조와
          소스의 w. ⛔ stage 2 는 돌리지 않는다
학습      ★ N 형상만 — `--train-shapes` 가 읽는 파일로 쓴다
뽑기      ★ (a) 의 ★ 첫 반복 (rep 0) 을 고정해 쓴다. 10번을 다 루프로
          돌리면 비용이 10배다 (§1-2)
평가      ⛔ 손대지 않는다 — 홀드아웃은 fold0 전체
```

★ `fit_regret` 에는 그 (방향, N) 의 **(a) 재적합 홀드아웃**을 적는다. 루프의
r-1 이 (a) 와 같은 값인지 나중에 대조하기 위해서다 (D-175 §3 넷째 조건과
같은 확인).
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

from experiments.porting_shapes import FOLD, GPUS, NS, _sample, _seed_of
from experiments.transfer_29_5 import TABLES

PICK_REP = 0
OUT_TAG = "ps"


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="docs/artifacts/porting-shapes.json")
    a = ap.parse_args()

    from experiments.f1_pipeline import _splits
    from kernelrule.core.table import PerfTable

    curve = {}
    p = Path(a.curve)
    if p.exists():
        for r in json.loads(p.read_text())["rows"]:
            rep = next((x for x in r["reps"] if x["rep"] == PICK_REP), None)
            curve[(r["src"], r["dst"], r["N"])] = (r, rep)

    made = []
    print("=" * 96)
    print(f"★ (b) 루프 씨앗 — 방향 x N (D-177 §1-2). 뽑기는 rep {PICK_REP}. "
          f"0 LLM 호출")
    print("=" * 96)
    for dst in GPUS:
        T = TABLES[dst]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        train = list(_splits(table, fold=FOLD, k=4,
                             design="nkband").train.shapes)
        for src in GPUS:
            if src == dst:
                continue
            base = Path(f"runs/pc-{src}2{dst}-f{FOLD}")
            if not (base / "stage2-rule-writer" / "chosen.json").exists():
                print(f"  ⚠️ {src}->{dst} pc 씨앗 없음 — 건너뜀")
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
                seed = _seed_of(src, dst, n, PICK_REP)
                pick = _sample(train, n, seed)
                (d / "train-shapes.json").write_text(json.dumps(
                    [[p.M, p.N, p.K] for p in pick]))
                row, rep = curve.get((src, dst, n), (None, None))
                (d / "stage2-rule-writer").mkdir(exist_ok=True)
                (d / "stage2-rule-writer" / "chosen.json").write_text(
                    json.dumps({
                        "source": f"ported:{src}->{dst}",
                        "code": ch["code"], "w0": ch["w0"],
                        # ★ the (a) point this curve sits on, so r-1 can be
                        #   checked against it
                        "fit_regret": (rep["holdout"] if rep
                                       else ch["fit_regret"]),
                        "why": ("★ D-177 §1-2 — the ported rule as a seed, "
                                "fitted on N target shapes only. `--train-"
                                "shapes` restricts the TRAIN split; the "
                                "holdout is the whole fold."),
                        "N": n, "pick_rep": PICK_REP, "sample_seed": seed,
                        "train_shapes": [[p.M, p.N, p.K] for p in pick],
                        "a_refit_holdout": rep["holdout"] if rep else None,
                        "a_pooled_regimes": (rep["pooled_regimes"] if rep
                                             else None),
                        "src_run": ch.get("src_run"),
                        "b_refit_full": ch["b_refit"],
                        "a_as_is": ch["a_as_is"],
                        "native": ch["native"]}, ensure_ascii=False,
                        indent=1))
                made.append(tag)
                av = (f"{rep['holdout']:.4f}" if rep else "—")
                print(f"  {tag:28s} N={n:2d}  (a) {av}  "
                      f"(b)전체 {ch['b_refit']:.4f}  "
                      f"원주민 {ch['native']:.4f}")
    print(f"\n  ★ {len(made)} runs")


if __name__ == "__main__":
    main()
