"""2번 채점 — 조건 A(24개) vs B(24 + 새 축 10개).

    python3 experiments/score_new_axes.py

★ 표본내와 "쓰이는가" 로 먼저 판정하고, **구조 홀드아웃은 마지막에 한 번만**
본다 (§12.3d). 그 숫자가 다음 수정의 근거가 되면 홀드아웃이 소진된다.

채점은 `core.canonical.canonical_score` 하나만 쓴다 — 루프의 `SplitSet` 을
받아야 돌고 형상을 따로 뽑는 경로가 없다 (D-36).

⚠️ 조건 B 의 규칙은 생성 피처를 참조하므로 **확장 레지스트리로 만든 행렬**
로 채점해야 한다. 기본 레지스트리로 채점하면 `AttributeError` 가 난다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.scoring import compare, evaluate
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import extended_registry
from kernelrule.rules.physics_seeded import CODE as PS
from kernelrule.rules.physics_seeded import W0 as PS_W0

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"


def main() -> None:
    # ★ 스크립트로 직접 돌면 `experiments` 가 패키지로 안 잡힌다.
    #   `__init__.py` 를 두고 저장소 뿌리를 경로에 넣는다.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.new_axes import novel_axes, used_features

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")

    novel = novel_axes(table, list(table.shapes())[:12])
    ext = extended_registry(REGISTRY, novel)
    new_names = {f.name for f in novel}
    mats = {"A-base": FeatureMatrix(table, REGISTRY),
            "B-extended": FeatureMatrix(table, ext)}

    v = evaluate(vendor_order_fn(table, load_vendor(VENDOR),
                                 mapping="nearest"),
                 table, list(splits.val.shapes), ks=(1,), label="벤더")

    print("=" * 78)
    print("2번 — 새 축의 쓸모")
    print("=" * 78)
    print(f"  {'조건':14s} {'시드':>4} {'표본내':>9} {'★구조HO':>9} "
          f"{'새 축 사용':>10}")
    rows: dict[str, list] = {}
    for cond in ("A-base", "B-extended"):
        for s in range(3):
            d = Path("runs") / f"newaxes-{cond}-s{s}" / "archive.jsonl"
            if not d.exists():
                continue
            with d.open() as fh:
                arc = [json.loads(ln) for ln in fh if ln.strip()]
            best = min(arc, key=lambda e: e["regret"])
            try:
                r = canonical_score(best["code"], best["w"], table=table,
                                    matrix=mats[cond], splits=splits)
            except Exception as exc:                        # noqa: BLE001
                print(f"  {cond:14s} {s:4d}  실패 {type(exc).__name__}: "
                      f"{str(exc)[:50]}")
                continue
            rows.setdefault(cond, []).append(r)
            used = sorted(used_features(best["code"]) & new_names)
            print(f"  {cond:14s} {s:4d} {r.in_sample:9.4f} {r.holdout:9.4f} "
                  f"{len(used):10d}  {used if used else ''}")

    print(f"\n  {'조건':14s} {'표본내 중앙':>11} {'최악':>8} "
          f"{'★구조HO 중앙':>13} {'최악':>8}")
    for cond, rs in rows.items():
        ins = sorted(x.in_sample for x in rs)
        hos = sorted(x.holdout for x in rs)
        print(f"  {cond:14s} {ins[len(ins) // 2]:11.4f} {ins[-1]:8.4f} "
              f"{hos[len(hos) // 2]:13.4f} {hos[-1]:8.4f}")
    print(f"  {'벤더 ★관문':14s} {'':11s} {'':8s} {v.at(1):13.4f}")
    ps = canonical_score(PS, PS_W0, table=table, matrix=mats["A-base"],
                         splits=splits)
    print(f"  {'physics_seeded':14s} {ps.in_sample:11.4f} {'':8s} "
          f"{ps.holdout:13.4f}")

    print(f"\n{'=' * 78}")
    print("유의성 — 구조 홀드아웃, 조건별 중앙 시드")
    print("=" * 78)
    for cond, rs in rows.items():
        mid = sorted(rs, key=lambda x: x.holdout)[len(rs) // 2]
        c = compare(mid.evaluation, v, table, name_a="A", name_b="벤더")
        print(f"  {cond:14s} {c.geo_a:.4f} vs {c.geo_b:.4f}   "
              f"이김 {int(c.a_wins.sum()):2d} / 짐 {int(c.a_loses.sum()):2d}"
              f" / 구분불가 {int(c.tied.sum()):2d}")

    print(f"\n{'=' * 78}")
    print(f"새 축 {len(new_names)}개가 아카이브에서 얼마나 쓰였나")
    print("=" * 78)
    tally: dict[str, int] = dict.fromkeys(sorted(new_names), 0)
    total = 0
    for s in range(3):
        d = Path("runs") / f"newaxes-B-extended-s{s}" / "archive.jsonl"
        if not d.exists():
            continue
        with d.open() as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                total += 1
                for n in used_features(json.loads(ln)["code"]) & new_names:
                    tally[n] += 1
    for n, c in sorted(tally.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(20 * c / max(total, 1))
        print(f"  {n:30s} {c:3d}/{total:3d}  {bar}")
    print(f"\n  한 번도 안 쓰인 축: "
          f"{[n for n, c in tally.items() if c == 0] or '없음'}")


if __name__ == "__main__":
    main()
