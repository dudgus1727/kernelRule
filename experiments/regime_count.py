"""8-2 체제를 몇 개로 나눌 것인가. LLM 호출 0회.

    python3 experiments/regime_count.py

## 무엇을 재는가

체제를 늘리면 체제당 형상이 줄어 §10.1 하한(15~20)에 걸린다. **경계를
늘려서 얻는 이득이 그 대가보다 큰가**를 잰다.

★ **표본 내로 재면 답이 나오지 않는다.** 체제를 늘리면 자유 가중치가
k배로 늘어나므로 적합한 형상에서의 regret 은 **반드시** 좋아진다. 그것은
체제가 실재한다는 증거가 아니라 파라미터가 늘었다는 뜻이다.

그래서 홀드아웃으로 잰다:

    SOL 순으로 정렬해 3개마다 1개를 홀드아웃 (체제 전체에 고르게 퍼진다)
    경계도 가중치도 **학습 형상에서만** 정한다
    홀드아웃 형상은 자기 SOL 이 속한 체제의 가중치로 채점

구조는 고정이므로 **경계의 값어치만** 남는다. 이득이 노이즈 바닥 근처면
"늘려도 소용없다" 가 결과이고, 그것 자체가 체제 수에 대한 증거다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate, evaluate_scores, geomean
from kernelrule.core.splits import _DUMMY_CFG, Split
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.features.physical import log_sol_ms
from kernelrule.rules.human_guided import CODE as PS
from kernelrule.rules.human_guided import W0 as PS_W0

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"

RUN_REAL = Path("runs/real-gpt-5.4-mini-2026-03-17/archive.jsonl")

#: §10.1 — 체제당 이 수 미만이면 그 체제의 적합은 믿을 수 없다.
MIN_PER_REGIME = 15


def main() -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    sol = {p: log_sol_ms(p, table.hw, _DUMMY_CFG) for p in shapes}
    ordered = sorted(shapes, key=lambda p: sol[p])

    # ★ SOL 순으로 3개마다 1개를 홀드아웃. 체제 전체에 고르게 퍼지므로
    #   어떤 k 에서도 모든 체제에 홀드아웃 형상이 들어간다.
    holdout = [p for i, p in enumerate(ordered) if i % 3 == 2]
    train = [p for i, p in enumerate(ordered) if i % 3 != 2]

    def run(k: int, code: str, w0) -> tuple[float, float, list[int]]:
        """k 등분. **경계도 가중치도 학습 형상에서만** 정한다."""
        m = len(train)
        bounds = [round(i * m / k) for i in range(k + 1)]
        groups = [train[a:b]
                  for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        # 체제 경계 = 학습 형상 기준 SOL 컷
        cuts = [sol[groups[i][0]] for i in range(1, k)]

        def regime_index(p) -> int:
            return sum(1 for c in cuts if sol[p] >= c)

        per_tr: dict = {}
        per_ho: dict = {}
        for gi, g in enumerate(groups):
            fit = fit_weights(compile_rule(code), matrix, table,
                              Split("train", tuple(g)), w0, max_evals=300,
                          objective="regret")
            so = make_score_of(compile_rule(code), matrix, fit.w)
            ev = evaluate_scores(so, table, g, ks=(1,))
            for i, p in enumerate(ev.shapes):
                per_tr[p] = ev.regret[i, 0]
            mine = [p for p in holdout if regime_index(p) == gi]
            if mine:
                eh = evaluate_scores(so, table, mine, ks=(1,))
                for i, p in enumerate(eh.shapes):
                    per_ho[p] = eh.regret[i, 0]
        return (geomean(np.array([per_tr[p] for p in train])),
                geomean(np.array([per_ho[p] for p in holdout])),
                [len(g) for g in groups])

    print("=" * 72)
    print("8-2. 체제 수 — 경계를 늘리면 이득이 있는가")
    print("=" * 72)
    print("  구조 고정, 체제마다 가중치만 재적합 — 경계의 값어치만 남는다")
    with RUN_REAL.open() as fh:
        archive = [json.loads(ln) for ln in fh if ln.strip()]
    evolved = min(archive, key=lambda e: e["regret"])

    for label, code, w0 in (("human_guided", PS, PS_W0),
                            ("evolved", evolved["code"], evolved["w"])):
        print(f"\n  [{label}]")
        print(f"  {'분할':>4} {'학습41':>9} {'★홀드아웃20':>11}  "
              f"{'학습 체제당':<20}")
        for k in (1, 2, 3, 4, 5):
            tr, ho, sizes = run(k, code, w0)
            warn = ("  ★ §10.1 하한 미달"
                    if min(sizes) < MIN_PER_REGIME else "")
            print(f"  {k:4d} {tr:9.4f} {ho:11.4f}  {str(sizes):<20}{warn}")

    v = evaluate(vendor_order_fn(table, load_vendor(VENDOR), mapping="nearest"),
                 table, shapes, ks=(1,), label="벤더")
    print(f"\n  벤더 {v.at(1):.4f}   ★ 관문 1.080")
    print("\n  ※ 학습 점수는 k 가 커지면 반드시 좋아진다 (자유 가중치가"
          " k배).\n     ★ 판단은 홀드아웃 열로만 한다.")
    print("  ※ 등분 경계는 0.5ms 와 다르다 — 재는 것은 '경계를 더 두면"
          " 이득이\n     있는가' 이지 0.5ms 가 최적인가가 아니다.")


if __name__ == "__main__":
    main()
