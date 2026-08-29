"""★ 커밋된 규칙으로 문서의 숫자를 다시 낸다. `runs/` 없이도 돈다.

    python3 experiments/verify_rules.py

## 무엇을 검증하는가

```
docs/artifacts/rules/<run>.py      score() + W_FITTED
docs/artifacts/rules/index.json    그때 기록된 점수
```

**이 스크립트는 `.py` 를 실행해 점수를 다시 계산하고 `index.json` 과
대조한다.** 어긋나면 실패한다.

## 왜 이것이 중요한가

LLM 실행은 재현할 수 없다 (난수 통제 불가 — §24.4b). 하지만 **채점은
완전히 결정론적**이다. 규칙 파일을 커밋해 두면 **성능 주장의 절반이
검증 가능해진다** — 누구나 몇 초에 확인한다.

`runs/` 는 `.gitignore` 라 저장소에 없다. 이 스크립트는 그것을 안 읽는다.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.numerics import approx_equal
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
RULES = Path("docs/artifacts/rules")
TOL = 5e-4


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score, mod.W_FITTED


def main() -> None:
    import numpy as np

    idx_path = RULES / "index.json"
    if not idx_path.exists():
        raise SystemExit(f"{idx_path} 가 없다. "
                         "`python3 experiments/export_rules.py` 를 먼저 돌려라.")
    index = json.loads(idx_path.read_text())

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")

    print("=" * 66)
    print("커밋된 규칙 재채점 — index.json 과 대조한다")
    print("=" * 66)
    print(f"  {'실행':16s} {'기록':>9} {'재계산':>9}  판정")
    bad = []
    for row in index:
        run = row["run"]
        f = RULES / f"{run}.py"
        if not f.exists():
            bad.append(f"{run}: 규칙 파일이 없다")
            continue
        fn, W = _load(f)
        reg = {}
        for name in ("short", "long"):
            g = [p for p in splits.val.shapes
                 if regime_of(p, table.hw) == name]
            if not g or name not in W:
                continue
            e = evaluate_scores(make_score_of(fn, matrix, np.asarray(W[name])),
                                table, g, ks=(1,))
            for i, p in enumerate(e.shapes):
                reg[p] = e.regret[i, 0]
        got = geomean(np.array([reg[p] for p in splits.val.shapes if p in reg]))
        want = row["holdout"]
        ok = approx_equal(got, want, TOL)
        if not ok:
            bad.append(f"{run}: 기록 {want:.4f} != 재계산 {got:.4f}")
        print(f"  {run:16s} {want:9.4f} {got:9.4f}  {'✅' if ok else '❌'}")

    print()
    if bad:
        print("★ 어긋난 것:")
        for b in bad:
            print(f"    {b}")
        raise SystemExit(1)
    print(f"  ★ {len(index)}개 전부 일치 (허용 {TOL})")
    print("  문서의 구조 홀드아웃 숫자는 이 값들로 검증된다.")


if __name__ == "__main__":
    main()
