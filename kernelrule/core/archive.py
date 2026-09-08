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
5배 더 달라진다는 §30.5 결과, 그리고 진화가 **소수 크기 체제를 희생한다**는
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

__all__ = ["Archive", "Elite", "CELL_AXIS_NAMES", "N_QUANTILES"]

#: ★ 셀 축 셋 (2026-09-08, D-144).
#:
#: ```
#: 옛   code_len · short_objective(SOL<0.5ms) · long_objective   4x4x4
#: ★ 새 mem_objective · comp_objective · all_objective          3x3x3
#: ```
#:
#: 구간은 `t_memory > t_compute` (roofline 하한 두 항의 비교)로 가른다 —
#: `SOL 0.5 ms` 같은 **임의 문턱을 쓰지 않는다** (D-143 이 그 문턱을
#: 방어할 수 없음을 보였다).
#:
#: `all_objective` 가 중복이 아닌 것을 확인했다 — 같은 (mem, comp) 칸
#: 안에서 전체 구간이 갈리는 칸이 13/16 이었다.
CELL_AXIS_NAMES = ("mem_objective", "comp_objective", "all_objective")

#: 축마다 칸 수. ★ 동적 3분위다 — **절대 경계는 없다** (아래 참고).
N_QUANTILES = 3


@dataclass
class Elite:
    rule_id: str
    code: str
    w: list[float]
    regret: float
    #: 학습 분할 안의 **memory 구간**(t_memory > t_compute) 목적함수 값.
    #: ★ 이름 이력: `short_regret` -> `short_objective`(D-101) ->
    #: `mem_objective`(D-144). 축이 크기(SOL 0.5ms)에서 roofline 으로 바뀌었다.
    mem_objective: float
    #: 학습 분할 안의 **compute 구간** 목적함수 값
    comp_objective: float
    #: ★ 학습 분할 **전체**의 목적함수 값 (D-144 에서 새로 생긴 축)
    all_objective: float
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
        """compute 구간과 memory 구간의 격차. **전이 신호다.**

        크면 그 규칙은 한 구간을 희생하고 있다.
        """
        return abs(self.comp_objective - self.mem_objective)

    def to_dict(self) -> dict:
        """★ `cell` 은 여기서 못 만든다 — 3분위는 **모집단**이 정한다.
        `Archive.dump` 가 그때의 칸을 채워 넣는다 (D-144)."""
        return dict(self.__dict__)


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
        # ★ 칸은 **언제나 동적 3분위**다 (D-144). `cell_mode="absolute"` 를
        #   없앴다 — 절대 경계 [1.0, 1.05, 1.15, 1.35, inf] 로 두면 진화가
        #   진행되며 한 칸에 몰린다 (실측: 셀 점유 2칸 / 채택 3-12 /
        #   전구간 tau -0.093, D-42 셋째 후보).
        #
        #   ⚠️ 대가: 칸의 뜻이 라운드마다 바뀐다. 전체가 좋아지면 1분위의
        #   절대값이 내려간다. 그 대신 칸이 고르게 찬다. 경계가 바뀌면
        #   **보유 엘리트를 전부 다시 배치한다** (`_consider_quantile`).
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

        ⚠️ 정렬하는 것은 `CELL_AXIS_NAMES` 의 셋이고, 이 값은
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
        for name in CELL_AXIS_NAMES:
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
        for i, _x in enumerate(pool):
            out[i] = tuple(axes[nm][i] for nm in CELL_AXIS_NAMES)
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
        won.extend(self._consider_quantile(e))
        # ★ 3분위 칸은 **모집단**이 정하므로 Elite 혼자서는 자기 칸을 모른다.
        #   기록에는 실제로 들어간 칸을 남긴다. 밀려났으면 빈 튜플이다.
        c = next((k for k, v in self.cells.items() if v is e), ())
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
        """★ 마지막 아카이브 **상태**를 쓴다 (D-139 — 개선 이력이 아니다).

        칸은 3분위라 Elite 혼자서는 모른다 — 지금 배치를 함께 적는다.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            for c, e in self.cells.items():
                d = e.to_dict()
                d["cell"] = list(c)
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
