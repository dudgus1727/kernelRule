"""★ 시드 폭을 체제별로 분해한다 (D-40). LLM 호출 0회.

    python3 experiments/seed_spread.py

## 왜

같은 조건의 시드 6개가 구조 홀드아웃에서 1.0518~1.1496 (폭 0.098) 을 낸다.
**지금까지 비교한 차이 대부분이 이 폭 안에 들어간다** — A/B 0.016,
Architect 씨앗 0.022, "벤더와 대등" 은 시드 하나였다.

폭이 어디서 오는지 모르면 실험을 어디서 해야 할지도 모른다.

    느린 체제 20형상   정적 top-1 1.015   여지 1.5%   ← 여기가 폭을 만드나?
    빠른 체제 41형상   정적 top-1 1.163   여지 16.3%

여지가 1.5% 인 구간에서는 어차피 아무도 못 이기는데, 시드마다 그 형상들의
순위가 흔들려 전체 geomean 을 끌고 다닐 수 있다.

## 판정

```
느린 쪽 폭이 크면   -> 빠른 체제로 실험을 옮긴다
비슷하면          -> 다른 원인. 라운드 수나 온도를 의심
```
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.scoring import evaluate
from kernelrule.core.splits import Split, SplitSet, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"

#: 같은 조건(씨앗 없음 + 피처 설명, 24개 피처)의 실행 6개.
SAME_CONDITION = [f"seedabl-desc-다-noseed-s{s}" for s in range(3)] + \
                 [f"newaxes-A-base-s{s}" for s in range(3)]


def main() -> None:
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
    n_fast = sum(1 for p in held if regime_of(p, table.hw) == "short")

    print("=" * 74)
    print("시드 폭을 체제별로 분해 — 같은 조건 6실행 (씨앗 없음 + 설명)")
    print("=" * 74)
    print(f"  구조 홀드아웃 {len(held)}형상 = 빠른 {n_fast} / 느린 "
          f"{len(held) - n_fast}\n")
    print(f"  {'실행':30s} {'전체':>8} {'빠른':>8} {'느린':>8}")

    rows = []
    for run in SAME_CONDITION:
        f = Path("runs") / run / "archive.jsonl"
        if not f.exists():
            print(f"  {run:30s} (없음)")
            continue
        with f.open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        best = min(arc, key=lambda e: e["regret"])
        r = canonical_score(best["code"], best["w"], table=table,
                            matrix=matrix, splits=splits)
        fast = r.by_regime.get("short", float("nan"))
        slow = r.by_regime.get("long", float("nan"))
        rows.append((r.holdout, fast, slow))
        print(f"  {run:30s} {r.holdout:8.4f} {fast:8.4f} {slow:8.4f}")

    if len(rows) < 2:
        return
    a = np.array(rows)
    print(f"\n  {'':30s} {'전체':>8} {'빠른':>8} {'느린':>8}")
    for label, fn in (("중앙", np.median), ("최소", np.min), ("최대", np.max)):
        print(f"  {label:30s} " + " ".join(f"{fn(a[:, i]):8.4f}"
                                           for i in range(3)))
    spread = a.max(axis=0) - a.min(axis=0)
    print(f"  {'★ 폭 (최대-최소)':30s} " + " ".join(f"{x:8.4f}" for x in spread))
    print(f"  {'변동계수 (표준편차/평균)':30s} "
          + " ".join(f"{a[:, i].std() / a[:, i].mean():8.4f}"
                     for i in range(3)))

    v = load_vendor(VENDOR)
    for name, group in (("빠른", [p for p in held
                                if regime_of(p, table.hw) == "short"]),
                        ("느린", [p for p in held
                                if regime_of(p, table.hw) == "long"])):
        e = evaluate(vendor_order_fn(table, v, mapping="nearest"),
                     table, group, ks=(1,))
        print(f"  {'벤더 ' + name:30s} {e.at(1):8.4f}")

    print()
    if spread[2] > spread[1] * 1.5:
        print("  ★ 느린 체제의 폭이 빠른 체제의 1.5배를 넘는다 — 가설 확정.")
        print("     실험을 빠른 체제로 옮긴다.")
    elif spread[1] > spread[2] * 1.5:
        print("  ★ 빠른 체제의 폭이 더 크다 — 가설 기각. 다른 원인이다.")
    else:
        print("  ★ 두 체제의 폭이 비슷하다 — 느린 체제만의 문제가 아니다.")
        print("     라운드 수나 온도를 의심해야 한다.")

    # -- split_k_io_amplification 이 새 정보인가 형태 개선인가 --------------
    print(f"\n{'=' * 74}")
    print("split_k_io_amplification 은 새 정보인가 형태 개선인가")
    print("=" * 74)
    from kernelrule.features.loader import extended_registry, load_generated
    from kernelrule.features.validate import _pearson, _spearman

    gen = load_generated("runs/featwriter-F1-gpt-5.4/proposals.jsonl",
                         only={"split_k_io_amplification"})
    if not gen:
        print("  (피처를 못 찾았다)")
        return
    ext = extended_registry(REGISTRY, gen)
    mat = FeatureMatrix(table, ext)
    cols: dict[str, list] = {}
    for p in list(table.shapes())[:12]:
        fe, _info = mat.for_shape(p)
        for n in ("split_k_io_amplification", "split_k_cost",
                  "log_workspace_bytes"):
            cols.setdefault(n, []).append(np.asarray(getattr(fe, n), float))
    c = {n: np.concatenate(v) for n, v in cols.items()}
    base = c["split_k_io_amplification"]
    for n in ("split_k_cost", "log_workspace_bytes"):
        sp, pe = abs(_spearman(base, c[n])), abs(_pearson(base, c[n]))
        verdict = ("형태 개선 (같은 정보)" if sp > 0.95
                   else "부분 겹침" if sp > 0.7 else "새 정보")
        print(f"  vs {n:24s} sp {sp:.3f}  pe {pe:.3f}   {verdict}")


if __name__ == "__main__":
    main()
