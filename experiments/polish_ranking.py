"""★ 다듬기가 규칙의 **순위**를 바꾸는가 — 삭제·재실행 여부의 근거. LLM 0회.

    python3 experiments/polish_ranking.py

절대값이 아니라 순위가 기준이다. 점수가 움직여도 상대 순서가 유지되면
아카이브 갱신과 부모 선택이 같았을 것이고, 그러면 진화 궤적도 같다.

```
순위 유지    -> 재실행 불필요
순위 뒤집힘  -> 궤적이 달랐다. 삭제 + 재실행
```

**각 실행의 "최고 규칙" 이 바뀌는가** 가 더 직접적인 판정이다 — 진화는
전체 순위가 아니라 아카이브 셀의 최고를 보고 부모를 고른다.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
#: 실행당 이만큼의 아카이브 후보를 본다. 진화가 실제로 줄 세운 대상이다.
N_CAND = 12


def _fit_regret(fn, matrix, table, train, w0, *, polish: bool) -> float:
    """두 체제에서 적합한 뒤 훈련 regret 을 합친다 (채점과 같은 절차)."""
    tot, n = 0.0, 0
    for name in ("short", "long"):
        g = [p for p in train if regime_of(p, table.hw) == name]
        fr = fit_weights(fn, matrix, table, Split("train", tuple(g)), w0,
                         max_evals=300, warn_invariants=False, polish=polish,
                          objective="regret")
        tot += np.log(fr.fit_regret) * len(g)
        n += len(g)
    return float(np.exp(tot / n))


def main() -> None:
    warnings.simplefilter("ignore")
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    index = json.loads(Path("docs/artifacts/rules/index.json").read_text())

    print("=" * 76)
    print("다듬기 전후 순위 — 진화가 같은 선택을 했겠는가")
    print("=" * 76)
    print(f"  실행마다 아카이브 상위 {N_CAND}개를 줄 세운다\n")
    print(f"  {'실행':16s} {'후보':>4} {'Kendall tau':>12} {'p':>7} "
          f"{'최고 규칙':>10} {'상위3 집합':>10}")

    taus, same_best, same_top3, n_run = [], 0, 0, 0
    for row in index:
        run = row["run"]
        with (Path("runs") / run / "archive.jsonl").open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        cand = sorted(arc, key=lambda e: e["regret"])[:N_CAND]
        if len(cand) < 3:
            print(f"  {run:16s} {len(cand):4d}  후보 부족 — 건너뜀")
            continue
        off, on = [], []
        for e in cand:
            fn = compile_rule(e["code"])
            off.append(_fit_regret(fn, matrix, table, train, e["w"],
                                   polish=False))
            on.append(_fit_regret(fn, matrix, table, train, e["w"],
                                  polish=True))
        t = kendalltau(off, on)
        b = int(np.argmin(off)) == int(np.argmin(on))
        t3 = (set(np.argsort(off)[:3].tolist())
              == set(np.argsort(on)[:3].tolist()))
        taus.append(t.statistic)
        same_best += b
        same_top3 += t3
        n_run += 1
        print(f"  {run:16s} {len(cand):4d} {t.statistic:12.3f} "
              f"{t.pvalue:7.3f} {'유지' if b else '★바뀜':>10} "
              f"{'유지' if t3 else '★바뀜':>10}")

    print()
    print(f"  Kendall tau 중앙 {np.median(taus):.3f}  "
          f"최소 {min(taus):.3f}  최대 {max(taus):.3f}")
    print(f"  최고 규칙 유지  {same_best}/{n_run}")
    print(f"  상위 3 집합 유지 {same_top3}/{n_run}")
    print()
    if same_best == n_run and np.median(taus) > 0.9:
        print("  판정: 순위 유지 — 진화가 같은 선택을 했을 것이다")
    else:
        print("  ★ 판정: 순위가 바뀐다 — 궤적이 달랐다. 삭제 + 재실행")


if __name__ == "__main__":
    main()
