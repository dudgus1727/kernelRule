"""프롬프트 문서에 표 유래 명제가 들어가지 않는가 (§12.3b / D-32).

## 왜 필요한가

§3 은 규칙 함수가 표를 못 보게 네 겹으로 막는다. 그런데 **그 표에서 나온
결론을 문장으로 옮겨 넣으면** 그 네 겹이 아무 일도 하지 않는다.

    "warp_m=128 은 최적을 낸 적이 없다"
      = 66형상 전부의 최적을 알아야 나오는 문장
      = 정답을 요약해 프롬프트에 넣은 것

학습 분할에서 세어도 마찬가지다 (§12.3b). 그리고 정정 이력을 프롬프트에
남기면 **뺐다고 적으면서 그 문장을 그대로 다시 쓰게 된다** (§12.3c) —
실제로 `hw/sm_86.md` 에서 그렇게 됐다.

## 한계

문자열 검사라 완벽하지 않다. 목적은 **다음에 누가 넣을 때 눈에 띄게**
하는 것이지 증명이 아니다. 구조적 방어는 `report/table_facts.py` 다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROMPTS = Path(__file__).resolve().parents[1] / "kernelrule/agents/prompts"

#: 표를 봐야만 쓸 수 있는 문장의 흔적.
_LEAK = (
    (re.compile(r"\d+\s*형상\s*중\s*\d+"), "N형상 중 M개 — 전수 집계"),
    (re.compile(r"\d+\s*/\s*66|\d+\s*/\s*61"), "N/66 · N/61 — 이 표의 분모"),
    (re.compile(r"최적(을)?\s*(낸 적이|한 적이)\s*없"), "'최적을 낸 적이 없다'"),
    (re.compile(r"최적\s*0\s*회|0\s*회\s*최적"), "'최적 0회'"),
    (re.compile(r"가설은?\s*기각"), "표로 기각한 가설"),
    (re.compile(r"이 표(에서|의)"), "'이 표에서/의' — 표를 가리킨다"),
    (re.compile(r"(정답 집합|answer set)에\s*(든|들어)"), "정답 집합 집계"),
)

#: 하드웨어 스펙은 표 없이 안다. 숫자가 있다고 누출이 아니다.
_HW_OK = re.compile(
    r"SM\s*84|101,?376|65,?536|1,?536|116\.1|729\.7|159\.1|6\s*MB|"
    r"1\.024|154\.8|768\s*GB|1350|7601|99KB")


def _prompt_files() -> list[Path]:
    return sorted(PROMPTS.rglob("*.md"))


def test_prompts_exist():
    assert _prompt_files(), f"프롬프트를 못 찾았다: {PROMPTS}"


@pytest.mark.parametrize("path", _prompt_files(), ids=lambda p: p.name)
def test_prompt_has_no_table_derived_claim(path: Path):
    """★ 표에서 나온 명제가 프롬프트에 있으면 안 된다 (§12.3b).

    걸리면 두 갈래로 처리한다.
      - 정말 표에서 나온 것   -> 뺀다. 정정 이력은 `docs/` 에 쓴다 (§12.3c)
      - 하드웨어/실행 모델    -> `_HW_OK` 에 추가하고 **왜 표 없이 아는지**
                               한 줄 남긴다
    """
    hits = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if _HW_OK.search(line):
            continue
        for pat, why in _LEAK:
            if pat.search(line):
                hits.append(f"  {path.name}:{i}  [{why}]\n    {line.strip()}")
                break
    assert not hits, (
        "프롬프트에 표 유래 명제가 있다 (§12.3b). 규칙 함수가 표를 못 보게 "
        "막아 놓고 그 표의 결론을 문장으로 넣으면 §3 이 아무 일도 하지 "
        "않는다:\n" + "\n".join(hits))
