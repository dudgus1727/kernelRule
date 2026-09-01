"""★ 폭이 진화가 만든 것인가 선택이 만든 것인가. LLM 호출 0회.

    python3 experiments/selection_spread.py

## 가설

같은 조건의 시드 6개가 구조 홀드아웃에서 폭 0.098 을 낸다. 그런데 아카이브
안에는 규칙이 여러 개 있고, 우리는 **학습 점수 최소** 하나만 고른다.

    첫 실행: 학습 최고 규칙 홀드아웃 1.2035 / 검증 최고 규칙 1.1305
             -> 어느 것을 고르느냐로 0.073 이 갈렸다

학습 41형상에서 상위 규칙들이 0.001 차이로 갈리는데 홀드아웃은 다른
형상이다. 순위가 뒤집히는 것이 자연스럽다.

## 재는 것

각 실행에서 학습 점수 상위 k개를 뽑아 홀드아웃 분포를 본다.

    k=1 의 시드 간 폭            0.098 (알고 있음)
    상위 k 중 홀드아웃 **최고** 의 폭   좁으면 "좋은 규칙이 있는데 못 고른다"
    상위 k 의 홀드아웃 **중앙** 의 폭   좁으면 "어느 것을 골라도 비슷한데
                                       최고만 튄다"

대응이 다르다. 그리고 **앙상블**(상위 k의 순위 평균)도 같이 잰다 — 선택
자체를 없애는 방법이고, 분산을 줄이는 것은 잘 알려진 성질이다.

⚠️ 홀드아웃을 **선택에 쓰지 않는다** (§10.2 / D-36). 여기서 홀드아웃
최고를 보는 것은 "아카이브에 그런 규칙이 있는가" 를 재는 **진단**이지
선택 기준이 아니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate, geomean
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
#: ★ 이 스크립트가 읽던 `gpt-5.4` 실행은 **삭제됐다** (D-52 — 지시 없이
#: 도입된 모델의 산출물). 다시 쓰려면 `experiments/seed_selection.py` 처럼
#: 지시된 모델로 먼저 실행을 만들고 아래 목록을 그것으로 바꿔라.
#: 없는 실행을 조용히 건너뛰면 **표본이 줄어든 줄 모르고 결론을 낸다.**
def _require(runs: list[str]) -> list[str]:
    from pathlib import Path as _P
    if not runs:
        raise SystemExit(
            "비교할 실행 목록이 비어 있다. 이 스크립트가 읽던 gpt-5.4 산출물은 "
            "삭제됐다 (D-52).\n"
            "지시된 모델로 실행을 만들고 목록을 채워라 — 빈 목록으로 돌면 "
            "표본 0으로 결론을 내게 된다.")
    missing = [r for r in runs if not (_P("runs") / r / "archive.jsonl").exists()]
    if missing:
        raise SystemExit(
            "이 스크립트가 읽던 실행이 없다 (gpt-5.4 산출물은 삭제됐다 — "
            "D-52):\n  " + "\n  ".join(missing)
            + "\n지시된 모델로 실행을 만들고 목록을 바꿔라.")
    return runs


#: 비교할 실행 목록. ★ 지시된 모델의 실행으로 바꿔서 쓴다.
RUNS: list[str] = []
KS = (1, 3, 5, 10)


def main() -> None:
    _require(RUNS)
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
    train = list(splits.train.shapes)

    def fit_per_regime(code, w0):
        """체제별 가중치. 앙상블도 같은 절차를 써야 비교가 성립한다."""
        out = {}
        fn = compile_rule(code)
        for name in ("short", "long"):
            g = [p for p in train if regime_of(p, table.hw) == name]
            out[name] = fit_weights(fn, matrix, table,
                                    Split("train", tuple(g)), w0,
                                    max_evals=300,
                          objective="regret").w
        return fn, out

    def ensemble_regret(fitted: list) -> float:
        """상위 k개의 **순위 평균**으로 고른다. 선택 자체를 없앤다."""
        regs = []
        for p in held:
            reg = regime_of(p, table.hw)
            cand = table.candidates(p)
            ranks = np.zeros(len(cand.tiebreak), dtype=float)
            for fn, ws in fitted:
                sc = make_score_of(fn, matrix, ws[reg])(p, cand)
                # ★ **동률을 보존하는** 순위여야 한다. `argsort(argsort(x))`
                #   는 같은 점수에 서로 다른 순위를 주고, 그 순서는 배열
                #   인덱스가 정한다 — 그러면 `top_k` 의 정준 tie-break 이
                #   무력화된다 (§30.7: 29/66 형상이 최적에서 동률이다).
                #   실제로 k=1 앙상블이 단일 규칙과 0.009 달라졌다.
                ranks += np.unique(sc, return_inverse=True)[1].astype(float)
            pick = cand.top_k(ranks, 1)[0]
            t = table.times_of(p)
            regs.append(float(t[pick] / t.min()))
        return geomean(np.array(regs))

    print("=" * 76)
    print("폭이 진화가 만든 것인가 선택이 만든 것인가")
    print("=" * 76)
    print(f"  같은 조건 {len(RUNS)}실행 / 구조 홀드아웃 {len(held)}형상\n")
    hdr = "  ".join(f"k={k}" for k in KS)
    print(f"  {'실행':30s} {'k=1':>8}   {'상위k 중 홀드아웃 최고':>22}")
    print(f"  {'':30s} {'':8s}   {hdr:>22}")

    best_of: dict[int, list] = {k: [] for k in KS}
    med_of: dict[int, list] = {k: [] for k in KS}
    ens_of: dict[int, list] = {k: [] for k in KS}
    for run in RUNS:
        f = Path("runs") / run / "archive.jsonl"
        if not f.exists():
            continue
        with f.open() as fh:
            arc = sorted((json.loads(ln) for ln in fh if ln.strip()),
                         key=lambda e: e["regret"])
        # 상위 10개까지의 홀드아웃 점수를 한 번씩만 계산한다
        hos, fitted = [], []
        for e in arc[:max(KS)]:
            r = canonical_score(e["code"], e["w"], table=table, matrix=matrix,
                                splits=splits)
            hos.append(r.holdout)
            fitted.append(fit_per_regime(e["code"], e["w"]))
        line = []
        for k in KS:
            sub = hos[:k]
            best_of[k].append(min(sub))
            med_of[k].append(float(np.median(sub)))
            ens_of[k].append(ensemble_regret(fitted[:k]))
            line.append(f"{min(sub):.4f}")
        print(f"  {run:30s} {hos[0]:8.4f}   " + "  ".join(line), flush=True)

    v = evaluate(vendor_order_fn(table, load_vendor(VENDOR),
                                 mapping="nearest"),
                 table, held, ks=(1,))
    print(f"\n{'=' * 76}")
    print("시드 간 폭 (최대 - 최소)")
    print("=" * 76)
    print(f"  {'':22s} " + "  ".join(f"k={k:<6d}" for k in KS))
    for label, d in (("상위k 중 최고", best_of), ("상위k 중앙", med_of),
                     ("★ 앙상블(순위평균)", ens_of)):
        sp = [max(d[k]) - min(d[k]) for k in KS]
        print(f"  {label:22s} " + "  ".join(f"{x:8.4f}" for x in sp))
    print()
    print(f"  {'':22s} " + "  ".join(f"k={k:<6d}" for k in KS))
    for label, d in (("상위k 중 최고", best_of), ("상위k 중앙", med_of),
                     ("★ 앙상블(순위평균)", ens_of)):
        md = [float(np.median(d[k])) for k in KS]
        print(f"  {label:22s} " + "  ".join(f"{x:8.4f}" for x in md)
              + "   (중앙값)")
    print(f"\n  벤더 {v.at(1):.4f}   ★ 관문")


if __name__ == "__main__":
    main()
