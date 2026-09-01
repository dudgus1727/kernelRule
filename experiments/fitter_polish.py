"""★ 좌표 다듬기가 A/B 해석을 뒤집는가 (D-55). LLM 0회.

    python3 experiments/fitter_polish.py

Nelder-Mead 가 멈춘 점이 좌표 방향으로 국소 최적이 아니라는 것을 재고,
다듬기를 켠 채 A/B(설명 vs 이름만)를 다시 읽는다. 판정 기준은
`docs/artifacts/fitter-sweep.md` 에 실험 전에 박아 뒀다.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"


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
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    index = json.loads((Path("docs/artifacts/rules") / "index.json").read_text())

    out: dict[bool, dict[str, float]] = {False: {}, True: {}}
    for pol in (False, True):
        for row in index:
            run = row["run"]
            with (Path("runs") / run / "archive.jsonl").open() as fh:
                arc = [json.loads(ln) for ln in fh if ln.strip()]
            best = min(arc, key=lambda e: e["regret"])
            fn = compile_rule(best["code"])
            reg = {}
            for name in ("short", "long"):
                g_tr = [p for p in train if regime_of(p, table.hw) == name]
                g_ho = [p for p in held if regime_of(p, table.hw) == name]
                fr = fit_weights(fn, matrix, table, Split("train", tuple(g_tr)),
                                 best["w"], max_evals=300, warn_invariants=False,
                                 polish=pol,
                          objective="regret")
                e = evaluate_scores(make_score_of(fn, matrix, fr.w), table,
                                    g_ho, ks=(1,))
                for i, p in enumerate(e.shapes):
                    reg[p] = e.regret[i, 0]
            out[pol][run] = geomean(np.array([reg[p] for p in held if p in reg]))

    print("=" * 70)
    print("좌표 다듬기 아래 A/B — 설명(luna) vs 이름만(lunaNAMES)")
    print("=" * 70)
    print(f"  {'조건':12s} {'다듬기 끔':>12} {'다듬기 켬':>12} {'변화':>9}")
    for tag, pre in (("A 설명", "luna-"), ("B 이름만", "lunaNAMES-")):
        f = [v for k, v in out[False].items() if k.startswith(pre)]
        t = [v for k, v in out[True].items() if k.startswith(pre)]
        print(f"  {tag:12s} {np.median(f):12.4f} {np.median(t):12.4f} "
              f"{np.median(t) - np.median(f):+9.4f}")
    for pol in (False, True):
        a = [v for k, v in out[pol].items() if k.startswith("luna-")]
        b = [v for k, v in out[pol].items() if k.startswith("lunaNAMES-")]
        u = mannwhitneyu(a, b, alternative="two-sided")
        print(f"\n  다듬기 {'켬' if pol else '끔'}: B - A = "
              f"{np.median(b) - np.median(a):+.4f}  "
              f"Mann-Whitney p={u.pvalue:.3f}")
    print("\n  ※ 12실행은 시드 폭 sigma=0.0274 안에 있다 (D-53). "
          "이 차이는 폭 안이며 유의하지 않다.")


if __name__ == "__main__":
    main()
