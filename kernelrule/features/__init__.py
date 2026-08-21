"""피처 레지스트리 (§8.2, §8.4).

피처는 **판단하지 않는다.** `(Problem, Hardware, Config) -> float` 의 순수
함수이고, "좋다/나쁘다" 는 규칙이 가중치로 표현한다.

## 작성 규칙 — LLM 에게도 동일하게 강제한다

    - 순수 함수. 부작용 없음
    - (Problem, Hardware, Config) 만으로 계산. 실측값/프로파일러 지표 금지
    - float 하나 반환. 0~1 정규화 권장. **클수록 나쁜 방향으로 통일**
    - 하드웨어 상수는 hw.* 에서 읽기. 84, 101376 하드코딩 금지
    - cfg.ext 참조 금지 (아키텍처 전이 전제 — §4.3)
    - 물리적 의미가 있을 것. 임의 조합 금지
    - 10줄 이내

**부호 통일이 중요하다.** "클수록 나쁨" 으로 맞추면 규칙이 항상 "가중합 후
오름차순 정렬" 이 되고, LLM 이 부호를 헷갈릴 여지가 사라진다.

## 라이프사이클 — append-only 금지 (§8.4)

`deprecate` 는 지우지 않고 표시만 한다. 기존 규칙은 계속 돌고, 새 규칙 생성
시 프롬프트의 피처 목록에서만 빠진다. **in-place 수정은 금지** — 버그를
고칠 때는 `_v2` 를 새로 추가한다. 과거 실험이 무효가 되면 재현이 안 된다.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Feature",
    "FeatureRegistry",
    "REGISTRY",
    "feature",
    "shape_feature",
]

Direction = str   # "higher_is_worse" | "higher_is_better" | "neutral"


@dataclass(frozen=True)
class Feature:
    """피처 하나. 메타데이터가 자동 검증(§8.3)의 재료다."""

    name: str
    fn: Callable                      # (Problem, Hardware, Config) -> float
    unit: str                         # "dimensionless" | "bytes" | "count" | ...
    expected_range: tuple[float, float]
    direction: Direction
    doc: str = ""
    #: 형상 수준인가 (Config 무관). True 면 `ShapeInfo` 에 스칼라로 들어간다.
    shape_level: bool = False
    #: 선택적 벡터화 구현. `(df, hw, info) -> np.ndarray`.
    #: 있으면 FeatureMatrix 가 이쪽을 쓰고, 없으면 스칼라를 100만 번 부른다.
    #: **스칼라 구현과 일치해야 한다** — `verify_vectorized()` 가 검사한다.
    vec: Callable | None = None
    deprecated_at_round: int | None = None
    deprecation_reason: str = ""
    #: 소스 해시. `features.lock` 과 FeatureMatrix 캐시 키에 쓴다 (§25).
    code_hash: str = ""

    @property
    def active(self) -> bool:
        return self.deprecated_at_round is None


class FeatureRegistry:
    """이름 -> Feature. **오타가 조용히 통과하지 않게 한다** (§21.3)."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._items: dict[str, Feature] = {}

    def add(self, f: Feature, *, replace: bool = False) -> Feature:
        if f.name in self._items and not replace:
            raise ValueError(
                f"피처 {f.name!r} 가 이미 있다. **in-place 수정 금지** (§8.4) — "
                f"버그를 고치려면 {f.name}_v2 를 추가하고 구 버전을 deprecate 하라.")
        self._items[f.name] = f
        return f

    def deprecate(self, name: str, *, at_round: int, reason: str) -> None:
        f = self[name]
        self._items[name] = Feature(**{**f.__dict__,
                                       "deprecated_at_round": at_round,
                                       "deprecation_reason": reason})

    def __getitem__(self, name: str) -> Feature:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"등록되지 않은 피처: {name!r}. "
                f"사용 가능: {sorted(self._items)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def names(self, *, active_only: bool = True,
              shape_level: bool | None = None) -> list[str]:
        out = []
        for n, f in self._items.items():
            if active_only and not f.active:
                continue
            if shape_level is not None and f.shape_level != shape_level:
                continue
            out.append(n)
        return sorted(out)

    def items(self, **kw) -> list[Feature]:
        return [self._items[n] for n in self.names(**kw)]

    def lock_hash(self, *, active_only: bool = True) -> str:
        """레지스트리 스냅샷 해시. FeatureMatrix 캐시 키 + `features.lock`."""
        h = hashlib.sha256()
        for n in self.names(active_only=active_only):
            h.update(n.encode())
            h.update(self._items[n].code_hash.encode())
        return h.hexdigest()[:16]

    def lock(self) -> dict:
        return {n: {"code_hash": self._items[n].code_hash,
                    "unit": self._items[n].unit,
                    "direction": self._items[n].direction,
                    "deprecated_at_round": self._items[n].deprecated_at_round}
                for n in self.names(active_only=False)}


#: 전역 레지스트리. `features/physical.py` 가 여기 등록한다.
REGISTRY = FeatureRegistry("physical")


def _hash_fn(fn: Callable) -> str:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):       # pragma: no cover
        src = repr(fn)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def feature(*, unit: str = "dimensionless",
            expected_range: tuple[float, float] = (0.0, 1.0),
            direction: Direction = "higher_is_worse",
            vec: Callable | None = None,
            registry: FeatureRegistry | None = None,
            shape_level: bool = False):
    """config 수준 피처를 등록한다. 배열로 계산되어 `Feats.<name>` 이 된다."""

    def deco(fn: Callable) -> Callable:
        f = Feature(name=fn.__name__, fn=fn, unit=unit,
                    expected_range=tuple(expected_range), direction=direction,
                    doc=(fn.__doc__ or "").strip().split("\n")[0],
                    shape_level=shape_level, vec=vec,
                    code_hash=_hash_fn(fn))
        # ★ `registry or REGISTRY` 를 쓰면 안 된다 — `__len__` 이 있어서
        # **빈 레지스트리가 falsy** 이고, 그러면 첫 피처가 조용히 전역
        # 레지스트리에 등록된다. 조용히 틀리는 종류의 버그다 (§30.8).
        (REGISTRY if registry is None else registry).add(f)
        fn.feature = f          # type: ignore[attr-defined]
        return fn

    return deco


def shape_feature(**kw):
    """형상 수준 피처. **스칼라**라서 규칙이 `if p.<name>:` 을 쓸 수 있다.

    이것이 §8.1(부록 대체본)이 허용하는 종류의 분기다 — "메모리 바운드면
    다르게 판단한다" 는 일반화되지만 "M 이 4096 이면 17번 config" 는 암기다.
    config 수준 피처는 배열이라 `if` 가 `ValueError` 를 내고, 그 비대칭이
    조건부 특수화를 문법적으로 어렵게 만든다.
    """
    kw.setdefault("shape_level", True)
    kw["shape_level"] = True
    return feature(**kw)


def verify_vectorized(f: Feature, df, hw, info, *, n: int = 256,
                      rtol: float = 1e-9, seed: int = 0) -> None:
    """벡터화 구현이 스칼라 구현과 일치하는지 검사한다.

    ⚠️ 불일치는 **기각**이다 (§26.4). 벡터화가 조용히 다르면 학습(행렬)과
    배포(스칼라)가 다른 함수를 쓰게 되고, 그건 §8.1 이 없애려던 오류다.
    """
    from kernelrule.core.types import Problem, config_from_row

    if f.vec is None:
        raise ValueError(f"{f.name}: 벡터화 구현이 없다.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    sub = df.iloc[idx]
    got = np.asarray(f.vec(sub, hw, info), dtype=np.float64)
    p = Problem(M=int(sub.iloc[0]["M"]), N=int(sub.iloc[0]["N"]),
                K=int(sub.iloc[0]["K"]), dtype=str(sub.iloc[0]["dtype"]))
    want = np.asarray([float(f.fn(p, hw, config_from_row(r)))
                       for r in sub.to_dict("records")], dtype=np.float64)
    if not np.allclose(got, want, rtol=rtol, atol=1e-12, equal_nan=True):
        bad = int((~np.isclose(got, want, rtol=rtol, atol=1e-12,
                               equal_nan=True)).sum())
        raise ValueError(
            f"{f.name}: 벡터화 구현이 스칼라와 다르다 ({bad}/{len(sub)}행). "
            "기각한다 — 학습과 배포가 다른 함수를 쓰게 된다.")
