"""정적 top-k — 형상과 **무관하게** 고정된 config k개를 쓸 때의 regret.

## ★ 절차가 답을 바꾼다 (§30.5b)

`status` 필터와 덮개 정의를 어떻게 잡느냐로 답이 세 갈래로 갈린다. 실측:

    ok 만 + 개별 전덮개      k=1 1.394   k=3 1.383   k=8 1.383   ← 포화
    ok 만 + 합집합 덮개      k=1 1.394   k=3 1.060   k=8 1.009
    전체 + 합집합 덮개 (정본) k=1 1.115   k=3 1.031   k=8 1.006

**세 절차를 전부 계산해 병기하고 정본을 명시한다. 단일 숫자로 보고하지 마라.**

### (1) `status` 필터

`high_outlier_frac` 은 그 **측정 한 건**의 속성이지 config 의 성질이 아니다.
반복 수가 많을수록 IQR 밖 하나가 걸릴 확률이 커질 뿐이고 시간 중앙값은
유효하다. `ok` 만 남기면 "모든 형상에서 우연히 깨끗한 측정이 나온 config" 를
요구하게 되어 61형상 전부에서 ok 인 config 가 17,325개 중 **3개**만 남는다.

### (2) 덮개 정의

개별 config 가 61형상 전부에서 유효할 것을 요구하면 `split_k>1` 이 사실상
배제된다 — `split_k=3` 은 K 가 3의 배수인 형상에서만 유효하다. 실제
라이브러리는 그런 config 를 당연히 포함하고 형상마다 그중 유효한 것을 쓴다.
**k개의 합집합이 덮으면 된다.**

### (3) 완화하면 반대로 무너진다

"덮은 형상에서만 재고 덮개율을 병기" (§9.1) 를 그대로 쓰면 그리디가 덮개
23% 로 도망가 1.074 가 나온다. 그래서 **합집합이 전 형상을 덮을 것을
요구**하고, 못 덮으면 그 자체를 실패로 보고한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kernelrule.core.scoring import Strata, geomean
from kernelrule.core.table import PerfTable

__all__ = ["StaticTopK", "TopKResult", "PROCEDURES", "run_all_procedures"]

#: 덮이지 않은 형상에 매기는 벌점 배수. 그리디가 **덮개를 먼저** 확보하게 한다.
#: 완화(덮은 형상에서만 채점)하면 23% 덮개로 도망간다.
_UNCOVERED_PENALTY = 1e3

#: 병기할 절차 세 개. 마지막이 정본이다.
PROCEDURES = (
    ("ok_individual", dict(ok_only=True, coverage="individual"),
     "ok 만 + 개별 전덮개"),
    ("ok_union", dict(ok_only=True, coverage="union"),
     "ok 만 + 합집합 덮개"),
    ("canonical", dict(ok_only=False, coverage="union"),
     "★ 전체 status + 합집합 덮개 (정본)"),
)


@dataclass
class TopKResult:
    procedure: str
    description: str
    ks: tuple[int, ...]
    #: k -> 층별 regret dict
    by_k: dict[int, dict[str, float]] = field(default_factory=dict)
    #: k -> 덮개율
    coverage: dict[int, float] = field(default_factory=dict)
    n_shapes: int = 0
    n_configs_considered: int = 0
    n_configs_total: int = 0
    chosen: list[tuple] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"[{self.procedure}] {self.description}",
                 f"    형상 {self.n_shapes}, 후보 config "
                 f"{self.n_configs_considered}/{self.n_configs_total} "
                 f"({100 * self.n_configs_considered / max(1, self.n_configs_total):.2f}%)",
                 f"    {'k':>3} {'전체':>7} {'>=0.5ms':>8} {'<0.5ms':>8} "
                 f"{'어려움':>7} {'쉬움':>7} {'덮개':>7}"]
        for k in self.ks:
            d = self.by_k[k]
            lines.append(
                f"    {k:>3} {d['all']:7.3f} {d['large(>=0.5ms)']:8.3f} "
                f"{d['small(<0.5ms)']:8.3f} {d['hard']:7.3f} {d['easy']:7.3f} "
                f"{self.coverage[k]:7.1%}")
        return "\n".join(lines)


class StaticTopK:
    """facility-location greedy. submodular 이므로 (1-1/e) 보장이 있다."""

    def __init__(self, table: PerfTable, shapes=None, *,
                 coverage: str = "union") -> None:
        if coverage not in ("union", "individual"):
            raise ValueError(f"알 수 없는 덮개 정의: {coverage!r}")
        self.table = table
        self.coverage = coverage
        self.shapes = tuple(shapes if shapes is not None else table.shapes())
        if not self.shapes:
            raise ValueError("형상이 하나도 없다 (§26.4).")
        self.strata = Strata.build(table, self.shapes)
        self._build()

    def _build(self) -> None:
        """config x 형상 행렬. 미측정은 NaN."""
        keys: dict[tuple, int] = {}
        cols = []
        for p in self.shapes:
            cand = self.table.candidates(p)
            t = np.asarray(self.table.times_of(p), dtype=np.float64)
            best = self.table.best_time(p)
            rel = t / best
            col: dict[int, float] = {}
            for i in range(cand.n):
                key = (str(cand.kernel_id[i]), int(cand.split_k[i]),
                       str(cand.split_k_mode[i]))
                j = keys.get(key)
                if j is None:
                    j = keys[key] = len(keys)
                # 같은 key 가 한 형상에 두 번 오면 안 되지만, 오면 빠른 쪽.
                if j not in col or rel[i] < col[j]:
                    col[j] = float(rel[i])
            cols.append(col)

        n_cfg, n_sh = len(keys), len(self.shapes)
        A = np.full((n_cfg, n_sh), np.nan, dtype=np.float64)
        for s, col in enumerate(cols):
            idx = np.fromiter(col.keys(), dtype=np.int64, count=len(col))
            val = np.fromiter(col.values(), dtype=np.float64, count=len(col))
            A[idx, s] = val
        self.keys = list(keys)
        self.n_configs_total = n_cfg
        if self.coverage == "individual":
            keep = np.flatnonzero(~np.isnan(A).any(axis=1))
            if keep.size == 0:
                raise ValueError(
                    "모든 형상에서 측정된 config 가 하나도 없다. "
                    "'individual' 덮개로는 정적 top-k 를 정의할 수 없다.")
            self.rows = keep
        else:
            self.rows = np.arange(n_cfg)
        self.A = A[self.rows]
        self.logA = np.log(self.A)

    def run(self, ks=(1, 2, 3, 5, 8, 10, 20)) -> TopKResult:
        ks = tuple(sorted(int(k) for k in ks))
        n_sh = len(self.shapes)
        cur = np.full(n_sh, np.nan)
        chosen: list[tuple] = []
        res = TopKResult(procedure="", description="", ks=ks,
                         n_shapes=n_sh,
                         n_configs_considered=int(self.rows.size),
                         n_configs_total=self.n_configs_total)
        pen = np.log(_UNCOVERED_PENALTY)
        for k in range(1, max(ks) + 1):
            # 각 후보를 넣었을 때의 목적함수. 미덮개는 벌점으로 채운다.
            cand = np.fmin(np.broadcast_to(cur, self.logA.shape), self.logA)
            filled = np.where(np.isnan(cand), pen, cand)
            obj = filled.mean(axis=1)
            i = int(np.argmin(obj))
            cur = np.fmin(cur, self.logA[i])
            chosen.append(self.keys[self.rows[i]])
            if k in ks:
                covered = ~np.isnan(cur)
                res.coverage[k] = float(covered.mean())
                rel = np.exp(cur)
                res.by_k[k] = self._strat(rel, covered)
        res.chosen = chosen
        return res

    def _strat(self, rel: np.ndarray, covered: np.ndarray) -> dict:
        s = self.strata

        def g(mask):
            m = mask & covered
            return geomean(rel[m]) if m.any() else float("nan")

        allm = np.ones(len(rel), dtype=bool)
        return {"all": g(allm), "hard": g(s.hard), "easy": g(~s.hard),
                "large(>=0.5ms)": g(~s.small), "small(<0.5ms)": g(s.small)}


def run_all_procedures(bundle_ref, env_hash: str, *, shapes_filter=None,
                       ks=(1, 2, 3, 5, 8, 10, 20)) -> list[TopKResult]:
    """세 절차를 전부 돌린다. **정본만 내지 않는다** (§30.5b)."""
    out = []
    for name, kw, desc in PROCEDURES:
        table = PerfTable.from_bundle(bundle_ref, env_hash=env_hash,
                                      ok_only=kw["ok_only"])
        shapes = ([p for p in table.shapes() if shapes_filter(p, table)]
                  if shapes_filter else None)
        try:
            r = StaticTopK(table, shapes, coverage=kw["coverage"]).run(ks)
        except ValueError as e:
            r = TopKResult(procedure=name, description=f"{desc} — 실패: {e}",
                           ks=tuple(ks))
            out.append(r)
            continue
        r.procedure, r.description = name, desc
        out.append(r)
    return out
