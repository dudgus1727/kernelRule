"""★ 적합기가 왜 절반은 안 움직이나 — 심플렉스 스텝을 쓸어본다. LLM 0회.

    python3 experiments/fitter_sweep.py

판정 기준은 **실험 전에** `docs/artifacts/fitter-sweep.md` 에 박아 뒀다.
결과를 보고 기준을 정하면 오염이다 (D-50).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core import weights as W
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
RULES = Path("docs/artifacts/rules")
#: (이름, SIMPLEX_SCALE, SIMPLEX_ABS)
SWEEP = [("상대 0.6 (지금)", 0.6, 0.0), ("상대 1.5", 1.5, 0.0),
         ("상대 4.0", 4.0, 0.0), ("절대 1.0", 0.0, 1.0), ("절대 5.0", 0.0, 5.0)]


def main() -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    index = json.loads((RULES / "index.json").read_text())

    print("=" * 78)
    print("적합기 심플렉스 스텝 쓸기 — 판정 기준은 fitter-sweep.md 에 있다")
    print("=" * 78)
    print(f"  {len(index)}규칙 x 2체제 = {len(index) * 2}회\n")
    print(f"  {'설정':16s} {'움직임':>10} {'구조HO 중앙':>12} {'개선':>8} "
          f"{'악화 건수':>9} {'evals 중앙':>10}")

    base_ho = None
    for label, scale, absst in SWEEP:
        W.SIMPLEX_SCALE, W.SIMPLEX_ABS = scale, absst
        moved = tot = 0
        evals = []
        per_run = []
        worse = 0
        for row in index:
            run = row["run"]
            d = Path("runs") / run
            with (d / "archive.jsonl").open() as fh:
                arc = [json.loads(ln) for ln in fh if ln.strip()]
            best = min(arc, key=lambda e: e["regret"])
            fn = compile_rule(best["code"])
            reg = {}
            for name in ("short", "long"):
                g_tr = [p for p in train if regime_of(p, table.hw) == name]
                g_ho = [p for p in held if regime_of(p, table.hw) == name]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fr = fit_weights(fn, matrix, table,
                                     Split("train", tuple(g_tr)), best["w"],
                                     max_evals=300, warn_invariants=False)
                tot += 1
                moved += int(fr.moved)
                evals.append(fr.n_evals)
                e = evaluate_scores(make_score_of(fn, matrix, fr.w), table,
                                    g_ho, ks=(1,))
                for i, p in enumerate(e.shapes):
                    reg[p] = e.regret[i, 0]
            ho = geomean(np.array([reg[p] for p in held if p in reg]))
            per_run.append(ho)
            if base_ho is not None and ho > base_ho[len(per_run) - 1] + 1e-6:
                worse += 1
        med = float(np.median(per_run))
        if base_ho is None:
            base_ho, imp = per_run, 0.0
        else:
            imp = float(np.median(base_ho)) - med
        print(f"  {label:16s} {moved:4d}/{tot:<4d} ({moved / tot:3.0%}) "
              f"{med:12.4f} {imp:+8.4f} {worse:9d} {int(np.median(evals)):10d}")
    W.SIMPLEX_SCALE, W.SIMPLEX_ABS = 0.6, 0.0


if __name__ == "__main__":
    main()
