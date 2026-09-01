"""MAP-Elites 아카이브 (§13, §27).

## 왜 단일 최고로는 안 되는가

항상 최고에서만 출발하면 근처만 뒤진다. 언덕 꼭대기까지는 가지만 옆의 더
높은 산은 못 본다. **특정 영역 최고면 전체가 낮아도 살려둔다.**

    전체 최고        규칙#47   1.12
    mem-bound 최고   규칙#31   1.08 (전체 1.24)   <- 단일 최고 방식이면 버려진다
    compute 최고     규칙#40   1.03 (전체 1.31)

그리고 **둘을 합치면 양쪽 다 잘하는 규칙이 나올 수 있다.** 그것이 교차이고
도약이 나오는 지점이다.

## 셀 축 (§27) — ★ 크기 체제로 바꿨다

    code_len      AST 노드 수                  4구간
    short_objective  학습 분할 안의 **짧은** 형상   4구간
    long_objective   학습 분할 안의 **긴** 형상     4구간
                                                -> 64 셀

**원래는 mem-bound / compute-bound 였다.** 크기 층화가 난이도 층화보다
5배 더 갈린다는 §30.5 결과, 그리고 진화가 **소수 크기 체제를 희생한다**는
실측(§10.1)에 맞춰 바꿨다. 전이가 되는 규칙을 별도 셀에 보존하는 것이
목적이다 — 균형 잡힌 학습에서도 9개 중 1개는 여전히 폭발한다.

⚠️ **검증 분할을 셀 축에 쓰면 홀드아웃이 오염된다** (§10.2).
축은 **학습 분할 안에서** 체제를 가른다.

초반 20라운드에 채워지는 셀 수를 보고 조정한다 — **10개 미만이면 경계가
너무 성기고 50개 이상이면 너무 촘촘하다.**

## 갱신은 노이즈 바닥으로 판정한다 (§7.4, §13.4)

"조금 좋아졌다" 로 갱신하면 아카이브가 노이즈를 축적한다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Archive", "Elite", "CELL_AXES", "cell_of", "N_QUANTILES"]

#: ★ 절대 경계 (`cell_mode="absolute"`). **regret 규모에 맞춰 정한 값이다** —
#: 목적함수가 바뀌면 못 쓴다 (순위 손실은 0.4 근처라 전부 첫 칸에 몰린다).
#: 그때는 `cell_mode="quantile"` 을 쓴다.
CELL_AXES: dict[str, list[float]] = {
    "code_len": [0, 60, 120, 200, float("inf")],
    "short_objective": [1.0, 1.05, 1.15, 1.35, float("inf")],
    "long_objective": [1.0, 1.05, 1.15, 1.35, float("inf")],
}

#: 순위 기반 칸 수 (체제 축마다). 절대 경계와 같은 4칸이다.
N_QUANTILES = 4


def _bin(v: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return len(edges) - 2


def cell_of(code_len: int, short_objective: float,
            long_objective: float) -> tuple:
    """절대 경계 칸. `cell_mode="quantile"` 이면 `Archive` 가 다르게 센다."""
    return (_bin(code_len, CELL_AXES["code_len"]),
            _bin(short_objective, CELL_AXES["short_objective"]),
            _bin(long_objective, CELL_AXES["long_objective"]))


@dataclass
class Elite:
    rule_id: str
    code: str
    w: list[float]
    regret: float
    #: 학습 분할 안의 짧은 형상(roofline 하한 < 0.5ms) **목적함수 값**.
    #: ★ 이름이 `short_regret` 이었다 (D-101). 목적함수가 regret 뿐이라는
    #: 가정이 이름에 박혀 있었고, 순위 손실을 넣으면서 거짓이 됐다.
    short_objective: float
    #: 학습 분할 안의 긴 형상 목적함수 값
    long_objective: float
    code_len: int
    round: int
    changes: str = ""
    hypothesis_id: str = ""
    parent_ids: list[str] = field(default_factory=list)
    val_regret: float = float("nan")
    #: ★ 순위 손실 (D-101). `Archive(select_by="rank")` 일 때 채택 기준이
    #: 된다. `regret` 은 그때도 **계속 채워진다** — 기록은 양쪽 다 한다.
    rank_loss: float = float("nan")

    @property
    def regime_gap(self) -> float:
        """긴 형상과 짧은 형상의 regret 격차. **전이 신호다.**

        크면 그 규칙은 한 체제를 희생하고 있다. 아카이브가 이 축으로
        갈리므로 격차가 작은 규칙이 따로 보존된다.
        """
        return abs(self.long_objective - self.short_objective)

    @property
    def cell(self) -> tuple:
        return cell_of(self.code_len, self.short_objective,
                       self.long_objective)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["cell"] = list(self.cell)
        return d


class Archive:
    """셀당 최고 하나 + 전체 최고."""

    def __init__(self, noise_tol: float = 0.0, *,
                 select_by: str = "regret",
                 cell_mode: str = "absolute") -> None:
        #: 갱신을 인정할 최소 개선. `is_significant` 가 준다 (§7.4).
        self.noise_tol = float(noise_tol)
        #: ★ 무엇으로 채택하나 (D-101). 기본은 `regret` — 지금까지의 모든
        #: 실행이 그 조건이다. `"rank"` 는 **명시할 때만** 돈다.
        #:
        #: ⚠️ 셀 **축**은 안 바뀐다 (코드 길이 / 체제별 regret). 축은
        #: 다양성을 만드는 장치이고 채택이 목표를 정한다 — 둘을 함께
        #: 바꾸면 변수가 둘이 된다 (`rank-evo-prereg.md` 정정).
        if select_by not in ("regret", "rank"):
            raise ValueError(f"알 수 없는 채택 기준: {select_by!r}")
        self.select_by = select_by
        #: ★ 칸을 어떻게 나누나 (D-101).
        #:
        #: ```
        #: absolute   CELL_AXES 의 절대 경계. ★ regret 규모에 맞춘 값이다
        #: quantile   ★ 보유 엘리트 + 새 후보를 **체제별 값으로** 정렬해 4분위
        #: ```
        #:
        #: 절대 경계는 목적함수가 바뀌면 못 쓴다 — 순위 손실은 0.4 근처라
        #: 전부 첫 칸에 몰린다. 순위 기반은 **경계값이 필요 없고** 목적함수가
        #: 무엇이든 그대로 돈다.
        #:
        #: ⚠️ 대가: 칸의 뜻이 라운드마다 바뀐다. 전체가 좋아지면 1분위의
        #: 절대값이 내려간다. 그 대신 칸이 고르게 찬다.
        if cell_mode not in ("absolute", "quantile"):
            raise ValueError(f"알 수 없는 칸 방식: {cell_mode!r}")
        self.cell_mode = cell_mode
        self.cells: dict[tuple, Elite] = {}
        self.best: Elite | None = None
        self.history: list[dict] = []
        self.n_seen = 0
        self.n_accepted = 0
        #: 셀이 새로 채워진 라운드. 조기 종료 판정에 쓴다 (§14.3).
        self.last_new_cell_round = -1

    def _key(self, e: Elite) -> float:
        """채택에 쓰는 값. **작을수록 좋다.**"""
        if self.select_by == "regret":
            return e.regret
        v = e.rank_loss
        if not math.isfinite(v):
            raise ValueError(
                "select_by='rank' 인데 Elite.rank_loss 가 없다. "
                "조용히 regret 으로 떨어지지 않는다 (§26.4).")
        return v

    @property
    def _tol(self) -> float:
        """★ 순위 손실에는 `noise_tol` 을 안 쓴다.

        `noise_tol` 은 regret 규모에 맞춰 `is_significant` 가 준 값이고
        순위 손실은 규모가 다르다. 그리고 **순위 손실의 쌍은 이미
        `resolvable` 로 걸러져 있어** 그 자체가 노이즈를 반영한다 —
        허용치를 또 붙이면 두 번 빼는 것이 된다.
        """
        return self.noise_tol if self.select_by == "regret" else 0.0

    def _quantile_cells(self, pool: list[Elite]) -> dict:
        """★ 보유분 + 후보를 **체제별 값으로** 정렬해 4분위 칸을 매긴다.

        ⚠️ 정렬하는 것은 `short_objective`/`long_objective` 이고, 이 값은
        **언제나 체제별 regret** 이다 (`ev.at(1, mask=...)`). 목적함수가
        `rank` 여도 그렇다 — **축은 다양성 장치이고 채택이 목표를 정한다**
        는 설계 그대로다 (D-101). 처음에 "목적함수로 정렬" 이라고 적었는데
        부정확했다 (D-104 에서 정정).

        경계값이 없는 것이 요점이다 — 절대 경계는 regret 규모에 맞춰
        정한 값이라 값 분포가 바뀌면 전부 한 칸에 몰린다. 아카이브가
        최대 64개라 정렬이 공짜다.

        **동률은 같은 칸**이다 (`argsort(argsort(.))` 를 안 쓴다, D-41).
        """
        n = len(pool)
        out: dict[int, tuple] = {}
        axes = {}
        for name in ("short_objective", "long_objective"):
            vals = [getattr(x, name) for x in pool]
            order = sorted(range(n), key=lambda i: (vals[i], i))
            rank = [0] * n
            r = 0
            for pos, i in enumerate(order):
                if pos and vals[i] > vals[order[pos - 1]]:
                    r = pos
                rank[i] = r
            axes[name] = [min(N_QUANTILES - 1, x * N_QUANTILES // max(n, 1))
                          for x in rank]
        for i, x in enumerate(pool):
            out[i] = (_bin(x.code_len, CELL_AXES["code_len"]),
                      axes["short_objective"][i], axes["long_objective"][i])
        return out

    def _consider_quantile(self, e: Elite) -> list[str]:
        """칸을 다시 매기고 칸마다 최선만 남긴다. `e` 가 남으면 이겼다."""
        pool = [*self.cells.values(), e]
        cells = self._quantile_cells(pool)
        best: dict[tuple, int] = {}
        for i, x in enumerate(pool):
            c = cells[i]
            cur = best.get(c)
            if cur is None or self._key(x) < self._key(pool[cur]):
                best[c] = i
        new = {c: pool[i] for c, i in best.items()}
        won: list[str] = []
        if e in new.values():
            won.append("new_cell" if len(new) > len(self.cells) else "cell")
        self.cells = new
        if won:
            self.last_new_cell_round = e.round
        return won

    def consider(self, e: Elite) -> list[str]:
        """넣어 본다. 어느 자리를 차지했는지 돌려준다. 빈 리스트면 폐기.

        ⚠️ 검사를 **맨 앞에서** 한다. `self.best is None or _key(e) < ...`
        는 아카이브가 비었을 때 단락 평가로 `_key` 를 건너뛴다 — 첫
        후보만 검사 없이 들어가는 fail-open 이었다 (시험이 잡았다).
        """
        self.n_seen += 1
        self._key(e)          # ★ 검사. 값은 아래에서 다시 쓴다
        won: list[str] = []
        if self.best is None or self._key(e) < self._key(self.best) - self._tol:
            won.append("best")
            self.best = e
        if self.cell_mode == "quantile":
            won.extend(self._consider_quantile(e))
            # ★ 순위 칸에서는 `e.cell`(절대 경계)이 뜻이 없다. 기록에는
            #   실제로 들어간 칸을 남긴다 — 못 찾으면 절대 칸을 적고
            #   그 사실이 보이게 한다.
            c = next((k for k, v in self.cells.items() if v is e), e.cell)
        else:
            c = e.cell
            cur = self.cells.get(c)
            if cur is None:
                won.append("new_cell")
                self.cells[c] = e
                self.last_new_cell_round = e.round
            elif self._key(e) < self._key(cur) - self._tol:
                won.append("cell")
                self.cells[c] = e
        if won:
            self.n_accepted += 1
        self.history.append({"round": e.round, "rule_id": e.rule_id,
                             "regret": e.regret, "rank_loss": e.rank_loss,
                             "select_by": self.select_by, "cell": list(c),
                             "won": won, "changes": e.changes})
        return won

    # -- 부모 선택 (§13.3) ------------------------------------------------
    def parents(self, n: int, rng) -> list[tuple[str, list[Elite]]]:
        """6 착실한 개선 / 3 다른 언덕 탐색 / 3 교차 (n=12 기준 비율)."""
        elites = list(self.cells.values())
        if not elites:
            return [("fresh", []) for _ in range(n)]
        n_exploit = max(1, round(n * 0.5))
        n_random = max(1, round(n * 0.25))
        n_cross = max(0, n - n_exploit - n_random)
        out: list[tuple[str, list[Elite]]] = []
        for _ in range(n_exploit):
            out.append(("exploit", [self.best or elites[0]]))
        for _ in range(n_random):
            out.append(("explore", [elites[int(rng.integers(len(elites)))]]))
        for _ in range(n_cross):
            if len(elites) >= 2:
                i, j = rng.choice(len(elites), size=2, replace=False)
                out.append(("cross", [elites[int(i)], elites[int(j)]]))
            else:
                out.append(("exploit", [self.best or elites[0]]))
        return out[:n]

    # -- 상태 -------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def summary(self) -> dict:
        return {"n_cells": self.n_cells, "n_seen": self.n_seen,
                "n_accepted": self.n_accepted,
                "best_regret": self.best.regret if self.best else float("nan"),
                "last_new_cell_round": self.last_new_cell_round}

    def dump(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            for e in self.cells.values():
                fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
