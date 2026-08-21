"""`PerfTable` — 표 조회. **측정 시간은 여기에만 존재한다** (§7.1, §3).

## 구조적 격리

규칙/피처 쪽에 나가는 것과 시간이 **다른 객체**다.

    table.candidates(p)  -> CandidateSet    시간 없음. 규칙/피처가 본다
    table.times_of(p)    -> np.ndarray      채점기만 부른다. 읽기 전용

`CandidateSet` 에 시간 필드가 없으므로 `sorted(..., key=(score, time))` 이나
`idxmin()` 을 쓰려면 없는 필드를 참조해야 하고, 그러면 `AttributeError` 다.
이것이 §30.7 의 버그를 **자료구조 수준에서** 막는 방법이다.

실제로 이 표에서 확인했다: 66형상 중 **29개가 최적시간에 정확한 동점**이고
최대 84중 동점이다 (타이머 양자화). "그 형상의 최적 config" 는 tie-break
규칙의 함수이지 물리적 사실이 아니다. 그래서 이 클래스는 `best_config()` 를
**제공하지 않는다** — 정의 가능한 것은 `best_time()` (스칼라, tie-break 무관)과
`answer_mask()` (집합)뿐이다.

## env_hash 는 조인 키가 아니라 격리 경계다 (§3.4)

`env_hash` 는 **기본값 없는 필수 인자**다. kernelTab 이 이 함정을 다섯 번
밟았고 전부 "여러 조건이 섞인 데이터를 필터 없이 집계" 였으며 **에러 없이
조용히 틀렸다.**
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kernelrule.core.adapter import normalize
from kernelrule.core.noise import NoiseModel
from kernelrule.core.types import (
    CandidateSet,
    Config,
    Hardware,
    Problem,
    ShapeKey,
    config_from_row,
    hardware_from_env,
    make_tiebreak,
)

__all__ = ["PerfTable", "TableError"]

#: 조인 키. ranking 표와 scoring 표가 같은 행을 가리키는지 검사하는 데 쓴다.
_JOIN = ("M", "N", "K", "kernel_id", "split_k", "split_k_mode")

#: 크기 층화 경계 (§30.4). 이보다 짧으면 노이즈가 순위를 지배한다.
SIZE_STRATA_MS = 0.5

#: 표 전체에서 값이 하나인 **조건 메타데이터**. 행마다 들고 있을 이유가 없다.
#: 980,915행 x 문자열이면 100MB 를 넘는다. `PerfTable.meta` 로 옮긴다.
_CONSTANT_META = ("bundle_id", "gpu_name", "cutlass_commit", "nvcc_arch",
                  "clock_locked", "locked_mhz", "sm_count",
                  "peak_tflops_used", "ridge_point_spec")

#: 반복되는 짧은 문자열. category 로 두면 메모리가 한 자릿수 줄어든다.
_CATEGORICAL = ("dtype", "acc_dtype", "layout_a", "layout_b", "layout_c",
                "split_k_mode", "pipeline_kind", "arch", "ext_swizzle_type",
                "workspace_dtype", "partials_dtype")


class TableError(RuntimeError):
    """표를 신뢰할 수 없다. **기본값으로 진행하지 않는다.**"""


@dataclass(frozen=True, slots=True)
class ShapeStats:
    """형상 하나의 정답 쪽 통계. **채점기/층화 전용. 규칙에 넘기지 마라.**

    `difficulty` 는 `ANSWER_COLS` 다 — 규칙이 "여긴 어려우니 신중하게" 를
    알면 정답을 훔쳐본 것이다. 평가 층화(§7.3)에만 쓴다.
    """

    key: ShapeKey
    n_candidates: int
    best_ms: float
    median_ms: float
    difficulty: float
    noise_floor: float
    answer_tol: float
    n_answers: int
    n_tied_at_best: int
    n_distinct_times: int

    @property
    def distinct_time_frac(self) -> float:
        """계측 분해능 지표 (§30.4b). 난이도와 **다른 축**이다.

        난이도 낮음 = 실제로 성능이 비슷하다 (물리)
        이 값 낮음  = 측정이 구분을 못 한다 (계측)
        """
        return self.n_distinct_times / self.n_candidates if self.n_candidates else 0.0

    @property
    def is_small(self) -> bool:
        """노이즈 지배 구간인가 (§30.4). A6000 표에서 66형상 중 45개."""
        return self.best_ms < SIZE_STRATA_MS


class PerfTable:
    """(형상, config) -> 시간 조회.

    ⚠️ 생성자를 직접 부르지 말고 `from_bundle` / `from_frames` 를 써라.
    """

    __slots__ = ("_X", "__times", "_rows", "_stats", "_cands", "_order",
                 "_hw", "_noise", "_meta", "_env_hash")

    def __init__(self, X: pd.DataFrame, times: np.ndarray, *,
                 hw: Hardware | None, noise: NoiseModel, env_hash: str,
                 meta: dict) -> None:
        if len(X) != len(times):
            raise TableError(
                f"피처 행 {len(X)} != 시간 행 {len(times)}. 조인이 어긋났다.")
        self._X = X.reset_index(drop=True)
        # 이름 맹글링 + 읽기 전용. 실수로 정렬 키에 섞이는 것을 어렵게 한다.
        t = np.asarray(times, dtype=np.float64).copy()
        t.setflags(write=False)
        self.__times = t
        self._hw = hw
        self._noise = noise
        self._env_hash = env_hash
        self._meta = dict(meta)

        # ★ 형상 그룹은 **벡터화**로 만든다. 파이썬 루프로 100만 개 튜플과
        #   리스트를 만들면 표 자체보다 메모리를 더 쓴다 (실측 3.7GB -> ).
        idx = self._X.groupby(["M", "N", "K", "dtype"], sort=False,
                              observed=True).indices
        self._rows = {(int(k[0]), int(k[1]), int(k[2]), str(k[3])):
                      np.asarray(v, dtype=np.int64) for k, v in idx.items()}
        self._order = list(self._rows)
        self._cands: dict[ShapeKey, CandidateSet] = {}
        self._stats: dict[ShapeKey, ShapeStats] = {}
        self._build_stats()

    # -- 생성 -------------------------------------------------------------
    @classmethod
    def from_bundle(cls, ref: str | Path, *, env_hash: str,
                    ok_only: bool = False,
                    unexpected: str = "warn") -> PerfTable:
        """kernelTab 번들에서 만든다.

        `env_hash` 는 **필수**다 (§3.4). 기본값을 두지 않는다.

        `ok_only=False` 가 기본인 이유: `high_outlier_frac` 도 유효한 측정이다
        (전체의 10.7%). 이 결정은 `RunConfig` 에 기록되어야 하며, 관문 리포트는
        양쪽을 다 낸다.
        """
        from kerneltab.core.bundle import load_bundle

        b = load_bundle(ref, verify=True)
        full = str(b.env_hash)
        if not full.startswith(str(env_hash)):
            raise TableError(
                f"env_hash 불일치. 요청 {env_hash!r}, 번들 {full[:16]!r}\n"
                "  env_hash 는 조인 키가 아니라 격리 경계다 (§3.4). "
                "다른 조건의 데이터를 섞지 마라.")

        noise = NoiseModel.from_bundle(b)
        env = b.env()
        hw = hardware_from_env(env)

        X = b.ranking(ok_only=ok_only, unknown_columns="warn")
        y = b.scoring(ok_only=ok_only)
        meta = {
            "bundle_id": b.info.get("bundle_id"),
            "schema_version": b.schema_version,
            "gpu_name": b.info.get("gpu_name"),
            "arch": b.info.get("arch"),
            "sm_count": b.info.get("sm_count"),
            "ok_only": ok_only,
            "shape_layers": b.shape_layers(),
            "tick_is_fallback": noise.tick_is_fallback,
        }
        return cls.from_frames(X, y, hw=hw, noise=noise, env_hash=full,
                              meta=meta, unexpected=unexpected)

    @classmethod
    def from_frames(cls, X: pd.DataFrame, y: pd.DataFrame, *,
                    hw: Hardware | None, noise: NoiseModel, env_hash: str,
                    meta: dict | None = None,
                    unexpected: str = "warn") -> PerfTable:
        """정규화 + 조인 검증 후 만든다.

        `X` 는 `load_for_ranking` 결과(정답 없음), `y` 는 `load_for_scoring`
        결과(정답 포함)여야 한다. **두 프레임이 같은 행을 가리키는지 검사한다** —
        `ok_only` 를 다르게 줘서 어긋나면 채점이 조용히 틀린다.
        """
        if len(X) != len(y):
            raise TableError(
                f"ranking {len(X)}행 != scoring {len(y)}행. 같은 ok_only 로 "
                "로드했는지 확인하라.")
        Xn = normalize(X, unexpected=unexpected)
        for c in _JOIN:
            if c not in y.columns:
                raise TableError(f"scoring 표에 조인 키 {c!r} 가 없다.")
            a = Xn[c].to_numpy()
            b_ = y[c].to_numpy()
            if a.dtype.kind in "OU" or b_.dtype.kind in "OU":
                same = np.asarray([str(u) == str(v)
                                   for u, v in zip(a, b_, strict=True)])
            else:
                same = a == b_
            if not same.all():
                bad = int((~same).sum())
                raise TableError(
                    f"ranking/scoring 조인이 어긋났다: {c!r} 에서 {bad}행 불일치. "
                    "행 순서가 같다는 전제가 깨졌다.")
        if "time_ms" not in y.columns:
            raise TableError("scoring 표에 time_ms 가 없다.")
        if "env_hash" in y.columns and y["env_hash"].nunique() > 1:
            raise TableError(
                f"scoring 표에 env_hash 가 {y['env_hash'].nunique()}개 섞여 있다. "
                "조건을 섞어 집계하지 마라 (§3.4).")

        meta = dict(meta or {})
        # 상수 메타데이터는 행에서 빼고 `meta` 로 옮긴다.
        drop = ["env_hash"]
        for c in _CONSTANT_META:
            if c in Xn.columns:
                vals = Xn[c].unique()
                if len(vals) == 1:
                    meta.setdefault(c, vals[0])
                    drop.append(c)
        Xn = Xn.drop(columns=[c for c in drop if c in Xn.columns])
        for c in _CATEGORICAL:
            if c in Xn.columns and str(Xn[c].dtype) != "category":
                Xn[c] = Xn[c].astype("category")
        return cls(Xn, y["time_ms"].to_numpy(dtype=np.float64),
                   hw=hw, noise=noise, env_hash=env_hash, meta=meta)

    # -- 조회 -------------------------------------------------------------
    @property
    def hw(self) -> Hardware:
        if self._hw is None:
            raise TableError("이 표에 Hardware 가 없다 (env.json 미제공).")
        return self._hw

    @property
    def noise(self) -> NoiseModel:
        return self._noise

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    @property
    def env_hash(self) -> str:
        return self._env_hash

    def shapes(self) -> list[Problem]:
        """형상 목록. 표에 나타난 순서를 유지한다 (결정론)."""
        return [Problem(M=k[0], N=k[1], K=k[2], dtype=k[3])
                for k in self._order]

    def frame_for(self, p: Problem) -> pd.DataFrame:
        """형상 하나의 **피처 행들**. 정답 없음. FeatureMatrix 가 쓴다."""
        return self._X.iloc[self._rows[self._k(p)]]

    def candidates(self, p: Problem) -> CandidateSet:
        """규칙/피처에 넘기는 후보 집합. **시간 없음.**"""
        k = self._k(p)
        cs = self._cands.get(k)
        if cs is None:
            rows = self._rows[k]
            sub = self._X.iloc[rows]
            kid = sub["kernel_id"].astype(str).to_numpy()
            sk = sub["split_k"].to_numpy(dtype=np.int64)
            mode = sub["split_k_mode"].astype(str).to_numpy()
            cs = CandidateSet(
                n=len(rows), kernel_id=kid, split_k=sk, split_k_mode=mode,
                tiebreak=make_tiebreak(kid, sk, mode), row_index=rows)
            self._cands[k] = cs
        return cs

    def configs(self, p: Problem) -> tuple[Config, ...]:
        """`Config` 객체 배열. 배포 shim / 리포트용. 채점 경로에서는 안 쓴다."""
        sub = self.frame_for(p)
        return tuple(config_from_row(r) for r in sub.to_dict("records"))

    # -- 정답 쪽 (채점기만) -------------------------------------------------
    def times_of(self, p: Problem) -> np.ndarray:
        """★ **채점기 전용.** 후보 순서와 정렬된 시간 배열 (읽기 전용).

        규칙 함수에 이 배열을 넘기면 §3 의 격리가 무너진다. 호출부를 세 곳
        (scoring / baselines / report) 으로 한정한다.
        """
        v = self.__times[self._rows[self._k(p)]]
        v.setflags(write=False)
        return v

    def best_time(self, p: Problem) -> float:
        return self._stats[self._k(p)].best_ms

    def stats(self, p: Problem) -> ShapeStats:
        return self._stats[self._k(p)]

    def all_stats(self) -> list[ShapeStats]:
        return [self._stats[k] for k in self._order]

    def difficulty(self, p: Problem) -> float:
        """중앙값 시간 / 최적 시간. **`ANSWER_COLS` 다. 층화에만 쓴다.**"""
        return self._stats[self._k(p)].difficulty

    def answer_mask(self, p: Problem) -> np.ndarray:
        """정답으로 인정할 후보의 불리언 마스크.

        허용치는 **형상별 노이즈 바닥의 2σ** 다 (§30.3). 고정 1% 가 아니다 —
        15µs 커널에서 1% 는 재현되지 않는 차이라 노이즈를 정답/오답으로 가른다.

        ⚠️ `kerneltab.core.table.answer_set()` 대신 이걸 쓴다. 그쪽은 **모듈
        전역 상수**를 쓰므로 다른 GPU 번들에서 A6000 눈금을 조용히 쓴다
        (`core/noise.py` 참조).
        """
        t = self.times_of(p)
        st = self._stats[self._k(p)]
        return t <= st.best_ms * (1.0 + st.answer_tol)

    # -- 내부 -------------------------------------------------------------
    def _k(self, p: Problem) -> ShapeKey:
        k = p.key if isinstance(p, Problem) else tuple(p)
        if k not in self._rows:
            raise KeyError(f"표에 없는 형상: {k}")
        return k

    def _build_stats(self) -> None:
        t_all = self.__times
        for k, rows in self._rows.items():
            t = t_all[rows]
            finite = t[np.isfinite(t) & (t > 0)]
            if finite.size == 0:
                raise TableError(
                    f"형상 {k} 에 유효한 측정이 하나도 없다. 표를 확인하라.")
            best = float(finite.min())
            med = float(np.median(finite))
            tol = self._noise.answer_tol(best)
            self._stats[k] = ShapeStats(
                key=k, n_candidates=int(t.size), best_ms=best, median_ms=med,
                difficulty=med / best,
                noise_floor=float(self._noise.floor(best)),
                answer_tol=tol,
                n_answers=int((t <= best * (1.0 + tol)).sum()),
                n_tied_at_best=int((t == best).sum()),
                n_distinct_times=int(np.unique(finite).size),
            )

    def summary(self) -> pd.DataFrame:
        """형상별 정답 쪽 통계. **리포트/층화 전용.**"""
        return pd.DataFrame([{
            "M": s.key[0], "N": s.key[1], "K": s.key[2], "dtype": s.key[3],
            "n_candidates": s.n_candidates, "best_ms": s.best_ms,
            "difficulty": s.difficulty, "noise_floor": s.noise_floor,
            "answer_tol": s.answer_tol, "n_answers": s.n_answers,
            "n_tied_at_best": s.n_tied_at_best,
            "distinct_time_frac": s.distinct_time_frac,
            "is_small": s.is_small,
        } for s in self.all_stats()])
