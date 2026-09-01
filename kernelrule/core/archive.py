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
    short_regret  학습 분할 안의 **짧은** 형상   4구간
    long_regret   학습 분할 안의 **긴** 형상     4구간
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

__all__ = ["Archive", "Elite", "CELL_AXES", "cell_of"]

CELL_AXES: dict[str, list[float]] = {
    "code_len": [0, 60, 120, 200, float("inf")],
    "short_regret": [1.0, 1.05, 1.15, 1.35, float("inf")],
    "long_regret": [1.0, 1.05, 1.15, 1.35, float("inf")],
}


def _bin(v: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return len(edges) - 2


def cell_of(code_len: int, short_regret: float, long_regret: float) -> tuple:
    return (_bin(code_len, CELL_AXES["code_len"]),
            _bin(short_regret, CELL_AXES["short_regret"]),
            _bin(long_regret, CELL_AXES["long_regret"]))


@dataclass
class Elite:
    rule_id: str
    code: str
    w: list[float]
    regret: float
    #: 학습 분할 안의 짧은 형상(roofline 하한 < 0.5ms) regret
    short_regret: float
    #: 학습 분할 안의 긴 형상 regret
    long_regret: float
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
        return abs(self.long_regret - self.short_regret)

    @property
    def cell(self) -> tuple:
        return cell_of(self.code_len, self.short_regret, self.long_regret)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["cell"] = list(self.cell)
        return d


class Archive:
    """셀당 최고 하나 + 전체 최고."""

    def __init__(self, noise_tol: float = 0.0, *,
                 select_by: str = "regret") -> None:
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
