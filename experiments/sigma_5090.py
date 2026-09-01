"""★ 5090 시드 폭 σ — §29.5 판정선의 근거. LLM 0회.

    python3 experiments/sigma_5090.py

## 왜 다시 재나

A6000 의 σ 를 5090 에 그대로 쓰면 안 된다 (사전 등록
`transfer-prereg.md`). 눈금이 1/64 이고 ridge 가 0.74배다 — 동률 구조가
다르므로 폭도 다를 수 있다.

## 재는 것

**정준 절차 그대로**: 학습 점수로 아카이브에서 하나 고르고, 체제별로
가중치를 다시 맞추고, 구조 홀드아웃에서 채점한다. 시드 간 표준편차가 σ 다.

## ★ n=3 의 σ 는 아주 넓다

카이제곱 신뢰구간을 **함께** 낸다. 점추정만 쓰면 판정선이 실제보다
좁아진다 — 그것이 바로 "못 재는 차이를 보고" 하는 길이다 (원칙 7).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY


def _splits(table: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    return SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")


def _sigma_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """표본 표준편차와 그 카이제곱 신뢰구간. `n` 이 작으면 아주 넓다."""
    from scipy.stats import chi2

    n = len(x)
    s = float(np.std(x, ddof=1))
    lo = s * float(np.sqrt((n - 1) / chi2.ppf(1 - (1 - conf) / 2, n - 1)))
    hi = s * float(np.sqrt((n - 1) / chi2.ppf((1 - conf) / 2, n - 1)))
    return s, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="datasets/rtx-5090-sm_120-5bb6f403")
    ap.add_argument("--env-hash", default="5bb6f403")
    ap.add_argument("--runs", nargs="+",
                    default=[f"f1pipe-F3-5090sigma-s{i}" for i in range(3)])
    ap.add_argument("--out", default="docs/artifacts/sigma-5090.json")
    a = ap.parse_args()

    warnings.simplefilter("ignore")
    table = PerfTable.from_bundle(a.bundle, env_hash=a.env_hash, ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    sp = _splits(table)

    print("=" * 74)
    print(f"시드 폭 σ   표={Path(a.bundle).name}")
    print("=" * 74)
    print(f"  학습 {len(sp.train.shapes)} / 구조 홀드아웃 "
          f"{len(sp.val.shapes)}  ({sp.kind})")
    print("  절차: 학습 점수로 아카이브에서 1개 -> 체제별 재적합 -> 홀드아웃\n")

    hold, train = [], []
    for r in a.runs:
        f = Path("runs") / r / "archive.jsonl"
        arc = sorted((json.loads(x) for x in f.read_text().splitlines()
                      if x.strip()), key=lambda e: e["regret"])
        e = arc[0]                       # ★ 학습 점수로 고른다
        res = canonical_score(e["code"], e["w"], table=table, matrix=matrix,
                              splits=sp)
        hold.append(res.holdout)
        train.append(float(e["regret"]))
        print(f"  {r:28s} 학습 {e['regret']:.4f}   홀드아웃 {res.holdout:.4f}",
              flush=True)

    h = np.array(hold)
    s, lo, hi = _sigma_ci(h)
    print(f"\n  중앙 {np.median(h):.4f}   범위 {h.min():.4f}~{h.max():.4f}   "
          f"폭 {h.max() - h.min():.4f}")
    print(f"  ★ σ = {s:.4f}   95% 신뢰구간 [{lo:.4f}, {hi:.4f}]   (n={len(h)})")
    print(f"\n  ⚠️ n={len(h)} 의 구간이다. **상한을 쓴다** — 점추정으로")
    print("     판정선을 정하면 못 재는 차이를 보고하게 된다 (원칙 7).")

    def need(delta: float, sig: float) -> float:
        """양측 0.05, 검정력 0.8 의 2표본 t 근사 시드 수."""
        return 2.0 * (2.8 * sig / delta) ** 2

    print(f"\n  {'차이 delta':>12}  {'시드 수 (σ 점추정)':>20}  "
          f"{'시드 수 (σ 상한)':>18}")
    for d in (0.02, 0.03, 0.05, 0.10):
        print(f"  {d:12.2f}  {need(d, s):20.0f}  {need(d, hi):18.0f}")

    Path(a.out).write_text(json.dumps({
        "bundle": a.bundle, "env_hash": a.env_hash, "runs": a.runs,
        "split_kind": sp.kind, "n_train": len(sp.train.shapes),
        "n_holdout": len(sp.val.shapes),
        "train_regret": train, "holdout_regret": hold,
        "sigma": s, "sigma_ci95": [lo, hi], "n_seeds": len(h),
        "procedure": ("학습 점수로 아카이브 1개 -> 체제별 재적합 -> "
                      "구조 홀드아웃(nk11008)"),
    }, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
