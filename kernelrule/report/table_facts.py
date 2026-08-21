"""블록 3.5 의 표 구조 관찰 — **학습 분할에서만** 계산한다 (§12.3).

## 왜 별도 모듈인가

전에는 `build_report(table_facts=[...])` 가 **자유 문자열 리스트**를 받았다.
리포트 생성기 자체는 `train.role != "train"` 을 거부하는데, 이 인자는 그
검사를 **완전히 우회한다** — 호출자가 전수 표에서 계산한 문장을 그대로
넣을 수 있었고, 실제로 그렇게 됐다.

    ⛔ 첫 실제 실행(`runs/real-gpt-5.4-mini-2026-03-17`)의 블록 3.5 는
       66형상 전수 / a888 61형상에서 계산된 값이었다. 검증·최종 분할이
       프롬프트에 들어갔다. `docs/artifacts/first-real-run.md` 참조.

§12.3 은 "홀드아웃 **점수**를 넣지 마라" 만 말했고 **집계가 빠져나갔다.**
점수든 집계든 홀드아웃에서 나온 것은 프롬프트에 들어가면 안 된다 — 사람이
그것을 보고 시스템을 고치면 결국 홀드아웃에 맞춰 튜닝하게 된다 (§10.2).

## 구조적 강제

`TableFacts` 는 `compute()` 로만 만들어지고, `compute()` 는 학습 분할만
받는다. `build_report` 는 이제 문자열 리스트를 거부한다 — **우회 경로를
없앤다** (§26.4: 조용히 나쁜 상태로 굴러가지 않는다).

## 최소 지지 형상 수

기반 형상이 적은 집계는 암기와 구분되지 않는다. "K<512 에서 stages 가
중요" 인데 그 형상이 4개뿐이면 그것은 물리가 아니라 그 4개다. 따라서
`MIN_SUPPORT` 미만에서 나온 문장은 **내지 않고, 냈어야 할 것이 있으면
그 사실을 적는다** — 조용히 빠지지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kernelrule.core.splits import Split, SplitError, regime_of
from kernelrule.core.table import PerfTable

__all__ = ["TableFacts", "MIN_SUPPORT"]

#: 이 수 미만의 형상에서 나온 집계는 제시하지 않는다.
MIN_SUPPORT = 15


@dataclass(frozen=True, slots=True)
class TableFacts:
    """학습 분할에서 계산된 표 구조 관찰.

    ★ `compute()` 로만 만든다. 자유 문자열을 넣는 경로를 두지 않는다.
    """

    lines: tuple[str, ...]
    n_shapes: int
    #: 지지 형상이 모자라 빼야 했던 관찰. **조용히 빠지지 않는다.**
    withheld: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def compute(cls, table: PerfTable, train: Split) -> TableFacts:
        """★ 학습 분할만 받는다. 다른 역할이면 예외다."""
        if not isinstance(train, Split) or train.role != "train":
            raise SplitError(
                "표 구조 관찰은 학습 분할에서만 계산한다 (§12.3). 집계도 "
                "홀드아웃을 넘지 않는다 — 점수만 막아서는 새어 나간다.")

        shapes = list(train.shapes)
        n = len(shapes)
        lines: list[str] = []
        withheld: list[str] = []

        def emit(text: str, support: int) -> None:
            (lines if support >= MIN_SUPPORT else withheld).append(text)

        lines.append(f"학습 분할 {n}형상에서만 계산했다 (§12.3).")

        # -- 정답 집합의 구성 — 어떤 축 값이 최적으로 뽑히는가 -------------
        #    ★ 컬럼이 없으면 **조용히 빠지지 않는다** (§26.4). 표마다 축이
        #      다를 수 있고, 없어진 관찰을 모르면 리포트를 잘못 읽는다.
        axes = (("has_spill", lambda c: c > 0, "스필 커널"),
                ("ext_stages", lambda c: c == 2, "stages=2(pipelined)"),
                ("ext_warp_m", lambda c: c == 128, "warp_m=128"),
                ("split_k_mode", lambda c: c.astype(str) == "parallel",
                 "split_k_mode=parallel"))
        cols = set(table.frame_for(shapes[0]).columns)
        counts: dict[str, int] = {}
        present = [a for a in axes if a[0] in cols]
        for p in shapes:
            frame = table.frame_for(p)
            win = frame.loc[table.answer_mask(p)]
            for col, pred, _ in present:
                if bool(pred(win[col]).any()):
                    counts[col] = counts.get(col, 0) + 1
        for col, _, label in present:
            emit(f"{label}이 정답 집합에 든 형상: {counts.get(col, 0)}/{n}개", n)
        for col, _, label in axes:
            if col not in cols:
                withheld.append(f"{label}: 표에 {col!r} 컬럼이 없어 못 쟀다")

        # -- 고정 config 하나로 어디까지 가는가 -----------------------------
        from kernelrule.baselines.static_topk import StaticTopK

        res = StaticTopK(table, shapes, coverage="union").run(ks=(1, 3, 8))
        emit("고정 config 하나로 얼마나 가는가 (형상 무관):  "
             f"top-1 {res.by_k[1]['all']:.3f}   top-3 {res.by_k[3]['all']:.3f}"
             f"   top-8 {res.by_k[8]['all']:.3f}", n)

        # -- 체제별 분해. ★ 크기가 먼저다 (§30.5) --------------------------
        fast = [p for p in shapes if regime_of(p, table.hw) == "short"]
        slow = [p for p in shapes if regime_of(p, table.hw) == "long"]
        for group, label in ((fast, "빠른 체제(SOL<0.5ms)"),
                             (slow, "느린 체제(SOL>=0.5ms)")):
            if not group:
                continue
            r = StaticTopK(table, group, coverage="union").run(ks=(1,))
            emit(f"  {label} {len(group):2d}형상만: top-1 "
                 f"{r.by_k[1]['all']:.3f}", len(group))

        # -- 정답의 뾰족함 — 노이즈 안에서 순위가 사라지는 형상 -------------
        n_flat = sum(1 for p in shapes if int(table.answer_mask(p).sum()) > 1)
        emit(f"정답이 하나로 정해지지 않는 형상(노이즈 안 동률): "
             f"{n_flat}/{n}개", n)
        sizes = np.array([int(table.answer_mask(p).sum()) for p in shapes])
        emit(f"  동률 폭 중앙값 {int(np.median(sizes))}개, 최대 "
             f"{int(sizes.max())}개", n)

        if withheld:
            lines.append(f"★ 제시하지 못한 관찰 {len(withheld)}건 "
                         f"(지지 형상 {MIN_SUPPORT}개 미만이거나 컬럼 부재). "
                         "조용히 빠지지 않는다 (§26.4).")

        return cls(lines=tuple(lines), n_shapes=n, withheld=tuple(withheld))
