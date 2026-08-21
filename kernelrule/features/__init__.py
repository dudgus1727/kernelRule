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
    "render_features",
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
    # ------------------------------------------------------------------
    # ★ 설명을 두 층으로 나눈다 (§8.2 / Architect A 조건)
    # ------------------------------------------------------------------
    # 전이 주장을 시험하려면 "표를 안 보고 규칙을 쓸 수 있는가" 를 물어야
    # 하는데, 피처 설명에 표에서 나온 문장이 섞여 있으면 그 시험이 성립하지
    # 않는다. 두 층을 **필드로** 갈라 놓아야 프롬프트를 조립할 때 섞이지
    # 않는다 — 주석으로 구분하면 반드시 섞인다.
    #
    #   physical_meaning  "마지막 wave 에서 노는 SM 슬롯의 비율"
    #                     새 GPU 에서도 그대로. A 조건에 넣는다.
    #   observed          "A6000 에서 스필 커널은 최적을 낸 적이 없다"
    #                     ★ 표에서 나왔다. A 조건에서 뺀다.
    #
    #: 비면 `doc` 을 쓴다.
    physical_meaning: str = ""
    #: 표에서 관측된 성질. **반드시 학습 분할에서만** 나온 것이어야 한다
    #: (§12.3 / D-28) — 전수 표에서 계산하면 홀드아웃이 프롬프트로 샌다.
    observed: tuple[str, ...] = ()
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

    @property
    def physics(self) -> str:
        """★ 아키텍처 무관한 정의. A 조건 프롬프트가 쓰는 유일한 설명."""
        return self.physical_meaning or self.doc

    def describe(self, *, include_observed: bool) -> str:
        """프롬프트 한 줄. `include_observed=False` 가 A 조건이다."""
        return self.describe_with(include_observed=include_observed)

    def describe_with(self, *, include_observed: bool,
                      extra: tuple[str, ...] = ()) -> str:
        # ★ 접근 형태를 그대로 보여준다. 이름만 주면 config 수준 피처를
        #   `p.` 로 쓰는 규칙이 나온다 — 실제로 첫 Architect 호출이 그랬다.
        ref = f"{'p' if self.shape_level else 'f'}.{self.name}"
        # ★ 범위와 단위는 **피처의 물리적 정의의 일부**다 (표가 아니다).
        #   없으면 상대 가중치를 세울 수 없다 — 자릿수가 다른 항을 그냥
        #   더하게 되고, 수치 최적화기도 그 지점에서 못 빠져나온다.
        #   실제로 Architect A 첫 시도가 regret 8.4 를 냈다.
        lo, hi = self.expected_range
        rng = f"[{lo:g}, {hi:g}]"
        head = f"{ref:28s} {rng:>14s}  {self.physics}"
        if include_observed:
            for o in (*self.observed, *extra):
                head += f"\n{'':28s}   [관측] {o}"
        return head


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
            shape_level: bool = False,
            physical_meaning: str = "",
            observed: tuple[str, ...] = ()):
    """config 수준 피처를 등록한다. 배열로 계산되어 `Feats.<name>` 이 된다."""

    def deco(fn: Callable) -> Callable:
        f = Feature(name=fn.__name__, fn=fn, unit=unit,
                    expected_range=tuple(expected_range), direction=direction,
                    doc=(fn.__doc__ or "").strip().split("\n")[0],
                    shape_level=shape_level, vec=vec,
                    physical_meaning=physical_meaning,
                    observed=tuple(observed),
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


def render_features(registry: FeatureRegistry | None = None, *,
                    include_observed: bool, active_only: bool = True,
                    extra_observed: dict[str, list[str]] | None = None) -> str:
    """프롬프트에 넣을 피처 목록.

    ★ `include_observed=False` 가 **A 조건**이다 — 표에서 나온 문장이 하나도
    들어가지 않는다. 그래야 "표를 안 보고 물리만으로 규칙을 쓸 수 있는가" 를
    실제로 물은 것이 된다.

    호출부에서 문자열을 이어붙이지 말고 **반드시 이 함수를 쓴다.** 조립을
    손으로 하면 섞인다 — 블록 3.5 가 그렇게 오염됐다 (D-28).

    `extra_observed` 는 **학습 분할에서 계산한** 관측을 피처별로 붙인다
    (`report/table_facts.py` 가 만든다). 정적 `Feature.observed` 필드에
    표에서 나온 수치를 넣지 않는 이유가 이것이다 — 그 값은 분할마다 다르고,
    소스에 박아 두면 어느 분할에서 나왔는지 알 수 없게 된다.

    ⚠️ `include_observed=False` 면 `extra_observed` 도 **무시된다.**
    """
    extra = extra_observed or {}
    reg = REGISTRY if registry is None else registry
    items = [reg[n] for n in sorted(reg._items)]
    if active_only:
        items = [f for f in items if f.active]
    shape, cfg = [f for f in items if f.shape_level], [
        f for f in items if not f.shape_level]
    def line(f: Feature) -> str:
        return f.describe_with(include_observed=include_observed,
                               extra=tuple(extra.get(f.name, ())))

    h_shape = ("## 형상 수준 — `p.<이름>` 으로만 접근한다. "
               "스칼라라서 `if p.<이름>:` 분기 가능")
    h_cfg = ("## config 수준 — `f.<이름>` 으로만 접근한다. "
             "**배열이다** (`if` 금지, ValueError)")
    tail = ("★ 접두사를 바꿔 쓰면 즉시 거부된다. `p.` 목록에 없는 이름을 "
            "`p.` 로 쓰거나 그 반대도 마찬가지다.")
    out = [h_shape, *(line(f) for f in shape),
           "", h_cfg, *(line(f) for f in cfg), "", tail]
    return "\n".join(out)


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
