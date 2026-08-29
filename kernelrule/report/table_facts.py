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
    #: 피처 이름 -> 그 피처에 붙일 관측. `features.render_features` 가 쓴다.
    #: ★ 학습 분할에서 나왔으므로 A 조건 프롬프트에서는 빠진다.
    by_feature: dict[str, list[str]] = field(default_factory=dict)
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

        # ★ 여기 있던 "정답 집합의 구성" 은 **삭제됐다** (2026-08-26, §12.3).
        #
        #   `스필 커널이 정답 집합에 든 형상: 0/61개` 같은 줄이었다. 이것은
        #   **축을 지목하는 정답 요약**이다 — "has_spill 은 볼 필요 없다" 를
        #   표에서 읽어서 알려주는 것이고, LLM 이 스스로 찾아야 할 것을
        #   대신 답해 준다. `by_feature` 로 각 피처 설명에 붙기까지 했다.
        #
        #   같은 이유로 `design.md` 의 "GBDT 피처 중요도를 블록 3.5 에
        #   넣어라" 도 철회했다 (§30.6 정정).
        #
        #   판정 기준 (§12.3b):
        #     가능  "고정 config 하나로 top-1 1.115"   여지의 크기
        #     불가  "스필 커널이 최적인 형상 0개"        축을 지목
        #     불가  "GBDT 가 mainloop_iters 를 중요하게 봤다"
        #
        #   **F0~F3 에서는 더 심각하다.** LLM 이 만든 피처는 이름이 전부
        #   다르므로 `feat_of` 매핑이 아예 안 맞고, 그런데도 축 이름은
        #   프롬프트에 남는다.
        by_feature: dict[str, list[str]] = {}

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

        return cls(lines=tuple(lines), n_shapes=n, by_feature=by_feature,
                   withheld=tuple(withheld))
