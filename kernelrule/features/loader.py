"""생성된 피처를 실행 간에 다시 불러온다 (§11.4).

## 왜 필요한가

F1 이 만든 피처는 `runs/featwriter-*/proposals.jsonl` 에만 있다. 그것을
규칙 루프에 쓰려면 다시 등록해야 하는데, **어느 것을 넣었는지가 실험
조건**이므로 그 선택이 코드에 남아야 한다.

    조건 A   사람이 쓴 24개
    조건 B   24 + 생성된 새 축 N개

★ 재발견된 것(기존과 스피어만 1.000)은 **넣지 않는다.** §8.4 의 중복
판정에 걸리고, 걸리지 않더라도 같은 정보를 두 번 주는 것이라 "새 축이
쓸모 있나" 를 흐린다.
"""

from __future__ import annotations

import json
from pathlib import Path

from kernelrule.features import Feature, FeatureRegistry

__all__ = ["load_generated", "extended_registry"]


def load_generated(path: str | Path, *, table, only: set[str] | None = None,
                   exclude: set[str] | None = None) -> list[Feature]:
    """`proposals.jsonl` 에서 채택된 피처를 되살린다.

    `only` / `exclude` 로 실험 조건을 **명시**한다 — 기본이 "전부" 면
    나중에 어떤 집합으로 돌렸는지 알 수 없게 된다.

    ★ `table` 은 **필수 인자다** — `shape_level` 을 다시 판정하기 위해서다
    (§30.12). 기록에 있는 값을 그대로 믿지 않는 이유: 그 판정은 **그
    번들에서** 내려진 것이고, `cfg` 를 참조하는데 우연히 상수였던 피처는
    다른 표에서 config 의존일 수 있다.

    **기본값을 두지 않는다.** 두었더니 두 호출부가 빠뜨렸고, 그중 하나가
    2단계 경로여서 **형상 수준 피처 0개**로 RuleWriter 가 돌았다 (D-67).
    재판정이 정말 필요 없으면 `table=None` 을 **명시**하라 — 그러면
    기록된 값을 쓴다.
    """

    from kernelrule.features.generated import compile_feature

    out: list[Feature] = []
    seen: set[str] = set()
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("accepted") or not r.get("code"):
                continue
            name, fn = compile_feature(r["code"], known=frozenset(seen))
            if only is not None and name not in only:
                continue
            if exclude and name in exclude:
                continue
            seen.add(name)
            rng = r.get("expected_range") or (0.0, 1.0)
            if isinstance(rng, str):
                rng = tuple(float(x) for x in
                            rng.strip("()[] ").split(",")[:2])
            out.append(Feature(
                name=name, fn=fn, unit=str(r.get("unit", "dimensionless")),
                shape_level=bool(r.get("shape_level", False)),
                expected_range=(float(rng[0]), float(rng[1])),
                direction=str(r.get("direction", "higher_is_worse")),
                doc=str(r.get("rationale", ""))[:200],
                physical_meaning=str(r.get("rationale", "")),
                source=r["code"], code_hash=str(abs(hash(r["code"])))))
    if table is not None:
        from dataclasses import replace

        from kernelrule.features.generated import detect_shape_level
        redone = []
        for f in out:
            is_shape, _ = detect_shape_level(replace(f, shape_level=False),
                                             table)
            redone.append(replace(f, shape_level=is_shape))
        return redone
    return out


def extended_registry(base: FeatureRegistry, features: list[Feature], *,
                      name: str = "extended") -> FeatureRegistry:
    """기존 + 생성. **원본을 건드리지 않는다** — 조건 비교가 깨진다."""
    reg = FeatureRegistry(name)
    for n in sorted(base._items):
        reg.add(base[n])
    for f in features:
        reg.add(f)
    return reg
