"""★ 3-fold 구성을 **파일로 만든다** (D-144). LLM 0회 · GPU 0회.

    python3 experiments/fold_plan.py

fold 별 train/val 목록과 memory/compute 개수를 적는다. 실행 전에 구성이
의도대로인지 사람이 볼 수 있어야 한다.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

from kernelrule.core.splits import regime_of, stratified_kfold
from kernelrule.core.table import PerfTable

BUNDLE = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
#: ★ fold 를 만드는 난수. **진화 시드와 분리한다** (D-144).
SPLIT_SEED = 12345


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--out", default="docs/artifacts/fold-plan.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(BUNDLE[0], env_hash=BUNDLE[1], ok_only=False)

    def aligned(p) -> bool:
        d = T.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in T.shapes() if aligned(p)]
    tot = Counter(regime_of(p, T.hw, axis="roofline") for p in shapes)
    print("=" * 84)
    print(f"층별 {a.k}-fold — 분할 시드 {a.split_seed} (진화 시드와 분리)")
    print("=" * 84)
    print(f"  정렬 8 형상 {len(shapes)}   {dict(tot)}")
    folds = stratified_kfold(shapes, T.hw, k=a.k, seed=a.split_seed)
    out: dict = {"bundle": BUNDLE[0], "k": a.k, "split_seed": a.split_seed,
                 "n_shapes": len(shapes), "totals": dict(tot), "folds": []}
    print(f"\n  {'fold':6s} {'train':>6s} {'mem/comp':>10s}   "
          f"{'val':>4s} {'mem/comp':>10s}")
    for i, sp in enumerate(folds):
        ct = Counter(regime_of(p, T.hw, axis="roofline")
                     for p in sp.train.shapes)
        cv = Counter(regime_of(p, T.hw, axis="roofline")
                     for p in sp.val.shapes)
        print(f"  {i:<6d} {len(sp.train.shapes):6d} "
              f"{f'{ct[chr(109)+chr(101)+chr(109)]}/{ct[chr(99)+chr(111)+chr(109)+chr(112)]}':>10s}   "
              f"{len(sp.val.shapes):4d} "
              f"{f'{cv[chr(109)+chr(101)+chr(109)]}/{cv[chr(99)+chr(111)+chr(109)+chr(112)]}':>10s}")
        out["folds"].append({
            "fold": i, "kind": sp.kind,
            "n_train": len(sp.train.shapes), "n_val": len(sp.val.shapes),
            "train_regimes": dict(ct), "val_regimes": dict(cv),
            "val": [[p.M, p.N, p.K] for p in sp.val.shapes],
            "train": [[p.M, p.N, p.K] for p in sp.train.shapes]})

    # ★ fold 의 val 이 서로 겹치지 않는가 / 합집합이 전체인가
    seen: Counter = Counter()
    for f in out["folds"]:
        for v in f["val"]:
            seen[tuple(v)] += 1
    print(f"\n  ★ val 이 정확히 한 번씩 나오나: "
          f"{set(seen.values()) == {1} and len(seen) == len(shapes)}")
    print("  ⚠️ 한계 — 세 fold 의 val 이 서로 다른 fold 의 **train** 에 있다.")
    print("     그리고 우리가 61형상을 전부 봐 왔다. '완전히 새로운 형상' 이")
    print("     아니므로 주장은 **'특정 분할에 의존하지 않는다'** 까지다.")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
