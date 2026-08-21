"""피처 행렬 사전 계산 (§21) — 채점을 300배 빠르게 한다.

## 왜 필요한가 (부록 ★ 수정 1)

원 설계는 "채점 수 밀리초" 라고 했으나 틀렸다. 형상당 후보가 약 15,000개이고
라운드당 `score()` 호출이 12규칙 x 66형상 x 15,000 ≈ 1,190만 회다. 순수
파이썬이면 라운드당 2분, 50라운드면 채점에만 100분이다. 그러면 "채점이 공짜라
규칙 12개를 병렬로 만든다" 는 설계 전제가 무너진다.

피처는 `(Problem, Hardware, Config)` 의 **순수 함수**이고 규칙과 무관하다.
같은 (형상, config) 쌍에 대해 어떤 규칙이 오든 값이 같다. **한 번만 계산해
행렬로 올려두고, 규칙은 그 행렬 위의 numpy 연산이 되게 한다.**

## Feats / ShapeInfo 의 비대칭이 설계다 (§8.1 대체본)

    f.<name>  ->  (n_candidates,) 배열   `if f.waves < 1:` 은 ValueError
    p.<name>  ->  스칼라                  `if p.is_memory_bound:` 은 된다

config 수준 조건부 특수화(= 룩업 테이블로 가는 길)를 **문법적으로** 어렵게 하고,
형상 수준 분기(= 일반화되는 종류)는 그대로 허용한다.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kernelrule.core.table import PerfTable
from kernelrule.core.types import Config, Hardware, Problem, config_from_row
from kernelrule.features import FeatureRegistry

__all__ = ["FeatureMatrix", "Feats", "ShapeInfo"]


class _DictAttr:
    """이름으로 값을 꺼내되 **오타가 조용히 통과하지 않는다** (§21.3)."""

    __slots__ = ("_cols", "_kind")

    def __init__(self, cols: dict, kind: str) -> None:
        object.__setattr__(self, "_cols", cols)
        object.__setattr__(self, "_kind", kind)

    def __getattr__(self, name: str):
        # ★ 밑줄로 시작하는 이름은 즉시 거부한다. 안 그러면 `_cols` 가 아직
        #   없는 상태(예: pickle 복원)에서 `__getattr__("_cols")` 가 자기를
        #   다시 불러 **무한 재귀**에 빠진다. 샌드박스가 이걸 잡았다.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            cols = object.__getattribute__(self, "_cols")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return cols[name]
        except KeyError:
            kind = object.__getattribute__(self, "_kind")
            raise AttributeError(
                f"등록되지 않은 {kind}: {name!r}. "
                f"사용 가능: {sorted(cols)}") from None

    def __setattr__(self, name, value):     # pragma: no cover
        raise AttributeError(f"{self._kind} 는 읽기 전용이다.")

    # -- pickle. 샌드박스가 자식 프로세스로 넘길 때 쓴다 --------------------
    # `__slots__` + `__setattr__` 금지 조합이라 기본 경로가 안 돈다.
    def __getstate__(self):
        return {"_cols": object.__getattribute__(self, "_cols"),
                "_kind": object.__getattribute__(self, "_kind")}

    def __setstate__(self, state):
        object.__setattr__(self, "_cols", state["_cols"])
        object.__setattr__(self, "_kind", state["_kind"])

    def __contains__(self, name: str) -> bool:
        return name in self._cols

    def __len__(self) -> int:
        return len(self._cols)

    def keys(self):
        return self._cols.keys()

    def as_dict(self) -> dict:
        return dict(self._cols)


class Feats(_DictAttr):
    """config 수준 피처. 각 값이 `(n_candidates,)` numpy 배열이다.

    `if f.tail_waste < 0.1:` 은 `ValueError` 를 낸다 — 배열이기 때문이다.
    조건부 특수화 대신 `np.where` 를 쓰라는 압력이 문법에서 나온다.
    """

    def __init__(self, cols: dict[str, np.ndarray]) -> None:
        super().__init__(cols, "피처")


class ShapeInfo(_DictAttr):
    """형상 수준 값. 전부 **스칼라**라서 `if` 가 그대로 된다."""

    def __init__(self, cols: dict[str, float]) -> None:
        super().__init__(cols, "형상 수준 값")


@dataclass
class MatrixStats:
    n_shapes: int
    n_rows: int
    n_features: int
    build_seconds: float
    bytes_: int
    from_cache: bool

    def __str__(self) -> str:
        src = "캐시" if self.from_cache else "계산"
        return (f"FeatureMatrix({src}): {self.n_shapes}형상 x {self.n_rows}행 "
                f"x {self.n_features}피처, {self.bytes_/1e6:.0f}MB, "
                f"{self.build_seconds:.1f}s")


class FeatureMatrix:
    """(형상, config) -> 모든 피처값. 표 로드 시 한 번 계산한다."""

    def __init__(self, table: PerfTable, registry: FeatureRegistry, *,
                 hw: Hardware | None = None,
                 cache_dir: str | Path | None = None,
                 verbose: bool = False) -> None:
        self.table = table
        self.registry = registry
        self.hw = hw if hw is not None else table.hw
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cols: dict[tuple, dict[str, np.ndarray]] = {}
        self._info: dict[tuple, dict[str, float]] = {}
        self.stats: MatrixStats | None = None
        self._build(verbose=verbose)

    # -- 조회 -------------------------------------------------------------
    def for_shape(self, p: Problem) -> tuple[Feats, ShapeInfo]:
        k = p.key
        return Feats(self._cols[k]), ShapeInfo(self._info[k])

    def feature_names(self) -> list[str]:
        return self.registry.names(shape_level=False)

    def shape_value_names(self) -> list[str]:
        return sorted(next(iter(self._info.values())))

    def column(self, name: str) -> np.ndarray:
        """모든 형상을 이어붙인 열. GBDT 베이스라인/상관 분석용."""
        return np.concatenate([self._cols[p.key][name]
                               for p in self.table.shapes()])

    def invalidate(self, feature_name: str) -> None:
        """피처가 추가/수정되면 **그 열만** 다시 계산한다 (§21.2)."""
        f = self.registry[feature_name]
        for p in self.table.shapes():
            df = self.table.frame_for(p)
            info = self._info[p.key]
            if f.shape_level:
                info[f.name] = self._scalar(f, p, df)
            else:
                self._cols[p.key][f.name] = self._vector(f, p, df, info)

    # -- 계산 -------------------------------------------------------------
    def _cache_key(self) -> str:
        m = self.table.meta
        h = hashlib.sha256()
        h.update(str(self.table.env_hash).encode())
        h.update(str(m.get("ok_only")).encode())
        h.update(str(m.get("bundle_id")).encode())
        h.update(self.registry.lock_hash().encode())
        return h.hexdigest()[:16]

    def _build(self, *, verbose: bool) -> None:
        t0 = time.perf_counter()
        path = (self.cache_dir / f"featmat-{self._cache_key()}.npz"
                if self.cache_dir else None)
        if path is not None and path.exists():
            self._load_cache(path)
            self._finish_stats(t0, from_cache=True)
            return

        cfg_feats = self.registry.items(shape_level=False)
        shp_feats = self.registry.items(shape_level=True)
        for p in self.table.shapes():
            df = self.table.frame_for(p)
            info: dict[str, float] = {
                "M": float(p.M), "N": float(p.N), "K": float(p.K),
                "n_candidates": float(len(df)),
            }
            for f in shp_feats:
                info[f.name] = self._scalar(f, p, df)
            cols: dict[str, np.ndarray] = {}
            for f in cfg_feats:
                cols[f.name] = self._vector(f, p, df, info)
            self._cols[p.key] = cols
            self._info[p.key] = info
            if verbose:
                print(f"  {p.key} {len(df)}행", flush=True)

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._save_cache(path)
        self._finish_stats(t0, from_cache=False)

    def _scalar(self, f, p: Problem, df) -> float:
        """형상 수준 피처. 대표 config 하나로 계산한다 (Config 무관해야 한다).

        ⚠️ 예외를 삼키지 않는다 (§26.4). 피처가 터지면 기각이지 승인이 아니다.
        """
        cfg = config_from_row(df.iloc[0].to_dict())
        v = float(f.fn(p, self.hw, cfg))
        if not np.isfinite(v):
            raise ValueError(f"{f.name}({p.key}) 가 비유한 값 {v} 를 냈다. 기각.")
        return v

    def _vector(self, f, p: Problem, df, info: dict) -> np.ndarray:
        if f.vec is not None:
            out = np.asarray(f.vec(df, self.hw, ShapeInfo(info)),
                             dtype=np.float64)
            if out.shape != (len(df),):
                raise ValueError(
                    f"{f.name}: 벡터화 구현이 {out.shape} 를 냈다. "
                    f"({len(df)},) 여야 한다.")
        else:
            out = np.asarray(
                [float(f.fn(p, self.hw, config_from_row(r)))
                 for r in df.to_dict("records")], dtype=np.float64)
        if not np.all(np.isfinite(out)):
            n_bad = int((~np.isfinite(out)).sum())
            raise ValueError(
                f"{f.name}({p.key}) 에 비유한 값 {n_bad}개. 기각한다 (§26.4).")
        return out

    def _finish_stats(self, t0: float, *, from_cache: bool) -> None:
        n_rows = sum(len(next(iter(c.values()))) if c else 0
                     for c in self._cols.values())
        nfeat = len(self.feature_names())
        self.stats = MatrixStats(
            n_shapes=len(self._cols), n_rows=n_rows, n_features=nfeat,
            build_seconds=time.perf_counter() - t0,
            bytes_=n_rows * nfeat * 8, from_cache=from_cache)

    # -- 캐시 -------------------------------------------------------------
    def _save_cache(self, path: Path) -> None:
        blob: dict[str, np.ndarray] = {}
        for k, cols in self._cols.items():
            tag = "|".join(str(x) for x in k)
            for name, arr in cols.items():
                blob[f"c::{tag}::{name}"] = arr
            blob[f"i::{tag}"] = np.asarray(
                [self._info[k][n] for n in sorted(self._info[k])],
                dtype=np.float64)
            blob[f"n::{tag}"] = np.asarray(sorted(self._info[k]), dtype=object)
        np.savez_compressed(path, **blob, allow_pickle=True)

    def _load_cache(self, path: Path) -> None:
        z = np.load(path, allow_pickle=True)
        for key in z.files:
            if key.startswith("c::"):
                _, tag, name = key.split("::", 2)
                k = self._tag_to_key(tag)
                self._cols.setdefault(k, {})[name] = z[key]
            elif key.startswith("i::"):
                tag = key[3:]
                k = self._tag_to_key(tag)
                names = list(z[f"n::{tag}"])
                self._info[k] = dict(zip([str(n) for n in names],
                                         [float(v) for v in z[key]],
                                         strict=True))

    @staticmethod
    def _tag_to_key(tag: str) -> tuple:
        m, n, kk, dt = tag.split("|")
        return (int(m), int(n), int(kk), dt)
