"""★ 재집계된 c2 값 하나만 보는 창구 (D-186). **0 LLM · 계산 없음.**

네 릴리즈(`porting-cost` · `porting-shapes` · `porting-strat` · `baselines`)가
c2 의 홀드아웃을 **원주민**으로 쓴다. 그 값이 D-183 에서 다시 나왔으므로
넷이 같은 자리를 보게 한다.

```
★ 원주민       campaign2.json 의 (표, fold) 홀드아웃
               ⚠️ 4시드의 ★ 중앙 — 캠페인 보고가 대표로 쓰는 것과 같은 자
★ 전이 (a)(b)  campaign2-transfer.json — ⛔ 재집계본
⛔ 옛 값        runs/*/stage2-rule-writer/chosen.json 에 박혀 있다.
   그것은 "적합 있던 채점 + nkgroup 원주민" 이라 ★ 읽지 않는다
```

⚠️ **적합이 있는 곳과 없는 곳** (D-186 §1)

```
적합 있음   (b) 가중치 재적합 · porting_shapes._refit 의 N 형상 적합
적합 없음   ★ 원주민 · (a) 그대로 · 루프 곡선의 각 라운드 채점
```
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

CAMPAIGN = Path("docs/artifacts/campaign2.json")
TRANSFER = Path("docs/artifacts/campaign2-transfer.json")
_C: dict = {}
_T: dict = {}


def native(gpu: str, fold: int) -> float:
    """★ 그 (표, fold) 의 원주민 — c2 4시드 홀드아웃의 **중앙**.

    ⛔ 적합 없음: c2 의 홀드아웃은 루프가 끝낸 가중치를 그대로 잰 값이다
    (D-182 · D-183).
    """
    if not _C:
        for r in json.loads(CAMPAIGN.read_text())["runs"]:
            if not r.get("missing"):
                _C.setdefault((r["gpu"], r["fold"]), []).append(r["holdout"])
    v = _C.get((gpu, fold))
    if not v:
        raise KeyError(f"c2 has no ({gpu}, fold{fold})")
    return float(st.median(v))


def transfer(src: str, dst: str, fold: int) -> dict:
    """★ 재집계된 전이 칸 — `a_as_is` · `b_refit` · `native` · `vendor`."""
    if not _T:
        for c in json.loads(TRANSFER.read_text())["cells"]:
            _T[(c["src"], c["dst"], c["fold"])] = c
    return _T[(src, dst, fold)]


def label() -> str:
    """산출물 note 에 박는 한 줄."""
    return ("★ D-186: 원주민과 (a) 는 ★ 적합 없이 채점한 값이고 (b) 와 "
            "_refit 만 적합한다. 옛 값은 '적합 있던 채점 + nkgroup 원주민' "
            "이었다 (pending_fixes 22 · D-182 · D-183).")
