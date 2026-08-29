"""프롬프트 두 축 배치 (§30.10).

원 설계는 "하드웨어 무관/의존" 한 축으로만 나눴다. **"역할 무관/의존"
으로는 안 나눠서** 역할별로 필요 없는 것이 공용에 쌓였다 — FeatureWriter
가 regret 정의와 가중치 예산 8개와 규칙 거부 사례를 매번 받았다.

              하드웨어 무관      하드웨어 의존
    역할 무관  _base.md          hw/sm_86.md
    역할 의존  role/*.md         (없음)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kernelrule.agents.openai_client import (
    _EDITS_RULES,
    _NEEDS_HW,
    _WRITES_RULES,
    load_prompt,
)

ROLES = ("analyze", "optimize", "feature", "architect")
PROMPTS = Path(__file__).resolve().parents[1] / "kernelrule/agents/prompts"


def _instructions(role: str) -> str:
    """`_agent()` 가 조립하는 것과 **같은 순서로** 만든다."""
    parts = [load_prompt("_base.md")]
    if role in _NEEDS_HW:
        parts.append(load_prompt("hw/sm_86.md"))
    if role in _WRITES_RULES:
        parts.append(load_prompt("role/_rules_common.md"))
    if role in _EDITS_RULES:
        parts.append(load_prompt("role/_rules_edit.md"))
    parts.append(load_prompt(f"role/{role}.md"))
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# ★ FeatureWriter 가 규칙 얘기를 받지 않는가
# ---------------------------------------------------------------------------

#: FeatureWriter 프롬프트에 있으면 안 되는 것. 전부 **규칙**의 얘기다.
_RULE_ONLY = ("regret", "가중치 8개", "리터럴", "w[0]", "np.random",
              "부모 규칙", "가설", "룩업 테이블")


def test_feature_prompt_has_no_rule_material():
    body = _instructions("feature")
    hit = [t for t in _RULE_ONLY if t in body]
    assert not hit, (
        f"FeatureWriter 프롬프트에 규칙 얘기가 있다: {hit}\n"
        "FeatureWriter 의 일은 원시 값으로 물리량을 찾는 것뿐이다 — "
        "뒷단 파이프라인을 알 필요가 없다 (§30.10).")


def test_feature_and_optimize_prompts_have_no_hardware_constants():
    """★ hw 를 안 보면 그 프롬프트는 **GPU 무관**해진다 (§16.2)."""
    hw = load_prompt("hw/sm_86.md")
    marks = [m for m in ("RTX A6000", "sm_86", "84", "101376") if m in hw]
    assert marks, "하드웨어 파일에서 표식을 못 찾았다 — 검사가 무의미하다"
    for role in ("feature", "optimize"):
        body = _instructions(role)
        hit = [m for m in ("RTX A6000", "sm_86") if m in body]
        assert not hit, f"{role} 프롬프트에 하드웨어 상수가 샜다: {hit}"


def test_rule_writers_get_the_budget():
    for role in ("optimize", "architect"):
        body = _instructions(role)
        assert "8" in body and "w[0]" in body, f"{role} 에 예산이 없다"


def test_hw_goes_only_to_roles_that_need_it():
    """★ Architect 뿐이다. Analyst 는 리포트 블록 1 에서 같은 사실을 받는다."""
    assert set(_NEEDS_HW) == {"architect"}


def test_architect_does_not_get_the_edit_block():
    """`role/architect.md` 가 "점수 없음" 이라고 써 놓고 regret 정의를
    받으면 정면으로 모순이다 (§30.10)."""
    assert "architect" not in _EDITS_RULES
    body = _instructions("architect")
    assert "점수 없음" in body, "역할 파일이 바뀌었다 — 검사가 무의미하다"
    assert "regret` = " not in body, "Architect 에 regret 정의가 샜다"


def test_optimizer_gets_the_edit_block():
    body = _instructions("optimize")
    assert "regret` = " in body and "실제로 거부된 것들" in body


# ---------------------------------------------------------------------------
# ★ 같은 문장이 두 역할 파일에 있으면 갈린다 (원칙 2)
# ---------------------------------------------------------------------------

#: 중복으로 세지 않는 줄. 마크다운 구조나 너무 짧은 것.
def _meaningful(line: str) -> bool:
    t = line.strip()
    return (len(t) >= 30 and not t.startswith(("#", "```", "|", "-", ">", "<!--"))
            and t not in ("", "---"))


def test_no_duplicate_sentences_between_role_files():
    seen: dict[str, str] = {}
    dupes: list[str] = []
    for f in sorted(PROMPTS.glob("role/*.md")):
        for line in f.read_text().splitlines():
            if not _meaningful(line):
                continue
            t = line.strip()
            if t in seen and seen[t] != f.name:
                dupes.append(f"  {seen[t]} <-> {f.name}: {t[:60]}")
            seen.setdefault(t, f.name)
    assert not dupes, (
        "역할 파일 사이에 같은 문장이 있다 — 하나만 고치면 갈린다 "
        "(원칙 2). 진짜 공용이면 `_base.md` 나 `role/_rules_common.md` 로 "
        "올려라:\n" + "\n".join(dupes))


def test_base_is_not_duplicated_into_role_files():
    base_lines = {ln.strip() for ln in load_prompt("_base.md").splitlines()
                  if _meaningful(ln)}
    dupes = []
    for f in sorted(PROMPTS.glob("role/*.md")):
        for line in f.read_text().splitlines():
            if _meaningful(line) and line.strip() in base_lines:
                dupes.append(f"  {f.name}: {line.strip()[:60]}")
    assert not dupes, ("`_base.md` 의 문장이 역할 파일에 복사돼 있다:\n"
                       + "\n".join(dupes))


# ---------------------------------------------------------------------------
# 예시가 답을 건네주지 않는가 (D-35)
# ---------------------------------------------------------------------------
def test_feature_examples_are_from_another_domain():
    """★ **F0/F1 에서는** GEMM config 축을 건드리는 예시가 답을 건네준다.

    F1-K/F2/F3 는 공개 지식을 주는 것이 조건의 정의이므로 실제 피처를
    코드까지 보여준다 (§30.17) — 그쪽은 `examples/known5.md` 이고 이
    검사의 대상이 아니다.
    """
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.features import REGISTRY

    block = load_prompt("examples/other_domain.md")
    leaked = [n for n in REGISTRY._items if n in block]
    assert not leaked, f"예시가 실제 피처를 담고 있다: {leaked}"
    # config 축 이름도 나오면 안 된다
    axes = ("tile_m", "tile_n", "tile_k", "split_k", "stages", "warp_m",
            "smem", "cp_async")
    hit = [a for a in axes if a in block]
    assert not hit, f"예시가 GEMM config 축을 건드린다: {hit}"


@pytest.mark.parametrize("role", ROLES)
def test_every_role_gets_the_base(role):
    assert "측정값은 배포 시점에 없습니다" in _instructions(role)


def test_hw_block_does_not_reference_cases():
    """★ `hw/*.md` 는 이제 **Architect 만** 받는데 Architect 는 사례를
    안 받는다. "사례에 붙은 ... 을 보세요" 는 없는 것을 가리킨다 (§30.10).
    """
    hw = load_prompt("hw/sm_86.md")
    assert "사례에 붙은" not in hw
    arch = _instructions("architect")
    assert "사례 없음" in arch, "역할 파일이 바뀌었다 — 검사가 무의미하다"


def test_analyst_gets_hardware_facts_from_the_report_not_a_file():
    """리포트 블록 1 과 `hw/*.md` 는 **같은 사실**이다. 리포트는 표에서
    매번 생성되고 파일은 고정이라, 둘 다 주면 번들이 바뀔 때 갈린다.
    """
    import warnings

    from kernelrule.core.table import PerfTable
    from kernelrule.report.diagnostic import hardware_block

    body = _instructions("analyze")
    assert "RTX A6000" not in body, "Analyst 시스템 프롬프트에 hw 가 있다"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = PerfTable.from_bundle("datasets/rtx-a6000-sm_86-c63710df",
                                  env_hash="c63710df", ok_only=False)
    blk = hardware_block(t.hw, t.noise)
    for fact in ("RTX A6000", "84", "ridge point"):
        assert fact in blk, f"리포트 블록 1 에 {fact!r} 가 없다"


# ---------------------------------------------------------------------------
# ★ 프롬프트 어디에도 **실제 피처 이름**이 박혀 있으면 안 된다 (D-35, D-65)
#
#   `role/architect.md` 의 크기 맞추기 예시가 `f.traffic_amplification` /
#   `f.tail_waste` 를 하드코딩하고 있었다. F1 조건에서 그것은 **답을
#   건네주는 것**이다 — 레지스트리에 없는 이름인데 물리를 지목한다.
#   `_base.md` 의 `if p.is_memory_bound:` 도 같다.
# ---------------------------------------------------------------------------
def test_no_prompt_hardcodes_a_registry_feature_name():
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.features import REGISTRY

    bad: list[str] = []
    for f in sorted(PROMPTS.rglob("*.md")):
        rel = f.relative_to(PROMPTS).as_posix()
        if rel in _KNOWN_BY_DESIGN:
            continue
        body = f.read_text()
        for n in REGISTRY._items:
            if n in body:
                bad.append(f"  {rel}: {n}")
    assert not bad, (
        "프롬프트가 실제 피처 이름을 박아 뒀다 — F0/F1 에서는 레지스트리에 "
        "없는 이름이고 물리를 지목한다 (D-35). `f.<이름>` 같은 자리표시자를 "
        "써라. 공개 지식을 주는 조건의 예시면 `_KNOWN_BY_DESIGN` 에 "
        "넣되 **그 파일이 F0/F1 에 안 간다는 것**을 확인하라:\n"
        + "\n".join(bad))


def test_known_by_design_files_never_reach_f0_or_f1():
    """★ 예외 파일이 정말 F0/F1 에 안 가는가 — 예외의 전제를 검사한다."""
    from kernelrule.agents.openai_client import _EXAMPLES

    for cond in ("F0", "F1"):
        assert f"examples/{_EXAMPLES[cond]}.md" not in _KNOWN_BY_DESIGN, cond


# ---------------------------------------------------------------------------
# ★ F1-K — 공개 지식 다섯으로 시작하는 조건 (§30.17)
# ---------------------------------------------------------------------------
def _feature_prompt(condition: str):
    import os

    import kernelrule.features.known5 as K
    from kernelrule.agents.openai_client import LLMConfig, OpenAILLM
    from kernelrule.features import FeatureRegistry

    os.environ.setdefault("OPENAI_API_KEY", "t")
    reg = FeatureRegistry(condition)
    if condition not in ("F0", "F1"):
        for n in sorted(K.KNOWN5._items):
            reg.add(K.KNOWN5[n])
    llm = OpenAILLM(LLMConfig(), feature_names=[], shape_values=[],
                    registry=reg)
    return llm._user_prompt("feature", "", condition=condition, registry=reg)


#: 실제 피처 이름을 **의도적으로** 담는 파일.
#:
#:   examples/known5.md      공개 지식을 주는 조건의 피처 예시 (§30.17)
#:   examples/rule_known.md  같은 조건의 **규칙** 예시 (§30.20)
#:
#: 둘 다 "레지스트리에 이미 있는 이름만" 쓴다. 그 불변식은
#: `test_rule_example_never_names_a_feature_outside_the_registry` 와
#: `test_known_by_design_files_never_reach_f0_or_f1` 이 지킨다 —
#: **예외를 만들면서 그 예외가 새는지를 함께 검사한다.**
_KNOWN_BY_DESIGN = {"examples/known5.md", "examples/rule_known.md"}


#: 표를 봐야만 아는 서술. 하나라도 프롬프트에 있으면 §12.3 위반이다.
_MEASURED = ("이 표에서", "최적 0회", "최적으로 뽑힌", "rel 중앙", "7.4%",
             "13.6", "37.2", "26% 어긋", "정답 집합")


def test_f1k_prompt_has_no_measurement():
    """★ 이번 작업의 가장 중요한 지점 — `has_spill` 의 표 관측을 빼는 것."""
    body = _feature_prompt("F1-K")
    hit = [m for m in _MEASURED if m in body]
    assert not hit, (
        f"F1-K 프롬프트에 측정 서술이 있다: {hit}\n"
        "표 없이 알 수 있는 것만 남긴다 (§12.3, §30.17).")


def test_f1k_shows_the_five_with_sources():
    import kernelrule.features.known5 as K

    body = _feature_prompt("F1-K")
    for n in K.KNOWN5._items:
        assert f"f.{n}" in body or f"p.{n}" in body, f"{n} 이 안 뜬다"
    assert body.count("출처:") >= 5, "출처가 다섯 미만이다"


def test_f1k_does_not_leak_the_other_nineteen():
    """★ 나머지 19개는 F3 조건이다."""
    import re

    import kernelrule.features.known5 as K
    from kernelrule.features import REGISTRY

    body = _feature_prompt("F1-K")
    rest = sorted(set(REGISTRY._items) - set(K.KNOWN5._items))
    leak = [n for n in rest if re.search(rf"\b{re.escape(n)}\b", body)]
    assert not leak, f"나머지 19개가 샜다: {leak}"


def test_examples_differ_by_condition():
    """F0/F1 은 무관 도메인, 공개 지식을 주는 조건은 실제 피처 (D-35)."""
    f1 = _feature_prompt("F1")
    f1k = _feature_prompt("F1-K")
    assert "branch_divergence_cost" in f1 and "queue_backlog" in f1
    assert "branch_divergence_cost" not in f1k
    assert "def tail_waste" in f1k and "다시 만들지 마세요" in f1k


def test_areas_are_fixed_and_do_not_name_features():
    """영역은 "무엇을 재는 자리" 일 뿐 "무엇을 만들어라" 가 아니다 (§30.18)."""
    from kernelrule.agents.openai_client import load_prompt

    areas = load_prompt("areas.md")
    body = areas[areas.index("```") + 3:areas.rindex("```")]
    rows = [ln for ln in body.splitlines() if "|" in ln]
    assert len(rows) == 7, f"영역이 일곱이 아니다: {len(rows)}"
    # 만들 피처를 지목하는 항목 나열이 없어야 한다
    for banned in ("wave 양자화", "타일 낭비", "wave quantization"):
        assert banned not in areas, banned


def test_known5_values_are_identical_to_physical(perf_table_for_known5):
    """★ 정리본이 원본과 **같은 값**을 내야 "알려진 피처를 줬다" 가 참이다."""
    import numpy as np

    import kernelrule.features.known5 as K
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import REGISTRY, FeatureRegistry

    t = perf_table_for_known5
    shapes = list(t.shapes())[:3]
    for n in sorted(K.KNOWN5._items):
        ra, rb = FeatureRegistry("a"), FeatureRegistry("b")
        ra.add(REGISTRY[n])
        rb.add(K.KNOWN5[n])
        ma, mb = FeatureMatrix(t, ra), FeatureMatrix(t, rb)
        sl = REGISTRY[n].shape_level
        for p in shapes:
            fa, ia = ma.for_shape(p)
            fb, ib = mb.for_shape(p)
            a = np.atleast_1d(np.asarray(getattr(ia if sl else fa, n), float))
            b = np.atleast_1d(np.asarray(getattr(ib if sl else fb, n), float))
            assert np.allclose(a, b, rtol=1e-9, atol=1e-12), n


@pytest.fixture(scope="module")
def perf_table_for_known5():
    import warnings

    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PerfTable.from_bundle("datasets/rtx-a6000-sm_86-c63710df",
                                     env_hash="c63710df", ok_only=False)


def test_internal_notes_never_reach_the_model():
    """★ `<!-- ... -->` 는 사람이 읽는 메모다. 모델에 가면 안 된다.

    실제로 `§30.18`, `D-45`, `D-47` 같은 **내부 결정 번호**가 그대로
    나가고 있었다 — 토큰을 쓰고, 내부 참조가 새고, 조건에 따라서는
    답을 건네줄 수도 있다.
    """
    from kernelrule.agents.openai_client import load_prompt

    bad = []
    for f in sorted(PROMPTS.rglob("*.md")):
        rel = f.relative_to(PROMPTS).as_posix()
        if "<!--" not in f.read_text():
            continue                       # 메모가 없는 파일
        if "<!--" in load_prompt(rel):
            bad.append(f"  {rel}")
    assert not bad, ("주석이 걷히지 않는다:\n" + "\n".join(bad))

    # 렌더링된 전문에도 없어야 한다
    for cond in ("F0", "F1", "F1-K", "F2", "F3"):
        body = _feature_prompt(cond)
        assert "<!--" not in body, cond
        for tag in ("§30.", "D-45", "D-47", "D-63"):
            assert tag not in body, f"{cond} 에 내부 참조 {tag} 가 있다"


# ---------------------------------------------------------------------------
# ★ §30.20 — Architect 규칙 예시도 조건별로 갈린다
#
#   FeatureWriter 는 예시가 조건별로 갈리는데 Architect 는 자리표시자
#   하나뿐이었다. 좋은 예시를 주되 **답을 건네지 않아야** 한다 (D-35).
#
#   조건 이름을 키로 쓰지 않는다 — Architect 의 `condition` 은 A/B(표
#   관측 유무)라 피처 조건과 축이 다르다. **레지스트리를 보고 정한다.**
# ---------------------------------------------------------------------------
def _reg(names):
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.features import REGISTRY, FeatureRegistry

    r = FeatureRegistry("probe")
    for n in names:
        r.add(REGISTRY[n])
    return r


def test_rule_example_is_chosen_by_registry_contents():
    import kernelrule.features.known5 as K
    from kernelrule.agents.openai_client import _rule_example_for
    from kernelrule.features import REGISTRY, FeatureRegistry

    human = _reg(sorted(REGISTRY._items))
    k5 = FeatureRegistry("k5")
    for n in sorted(K.KNOWN5._items):
        k5.add(K.KNOWN5[n])

    assert "f.tail_waste" in _rule_example_for(human)
    assert "f.tail_waste" in _rule_example_for(k5)
    # 이름이 없는 레지스트리면 무관 도메인으로 떨어진다
    assert "f.tail_waste" not in _rule_example_for(FeatureRegistry("empty"))


def test_rule_example_never_names_a_feature_outside_the_registry():
    """★ 이것이 진짜 불변식이다 — 조건 이름이 아니라 **누출 여부**."""
    import re

    import kernelrule.features.known5 as K
    from kernelrule.agents.openai_client import _rule_example_for
    from kernelrule.features import REGISTRY, FeatureRegistry

    k5 = FeatureRegistry("k5")
    for n in sorted(K.KNOWN5._items):
        k5.add(K.KNOWN5[n])
    cases = {"사람24": _reg(sorted(REGISTRY._items)), "known5": k5,
             "빈": FeatureRegistry("empty"),
             "일부": _reg(["waves", "edge_waste"])}
    for tag, r in cases.items():
        ex = _rule_example_for(r)
        leak = [n for n in REGISTRY._items
                if re.search(rf"[fp]\.{re.escape(n)}\b", ex)
                and n not in r._items]
        assert not leak, f"{tag}: 레지스트리 밖 이름이 예시에 있다 {leak}"


def test_rule_examples_keep_placeholders():
    """완성된 규칙을 주면 그대로 제출하고 구조 비교가 무너진다 (D-35)."""
    from kernelrule.agents.openai_client import load_prompt

    for f in ("examples/rule_known.md", "examples/rule_other_domain.md"):
        body = load_prompt(f)
        assert "<" in body and ">" in body, f"{f} 에 자리표시자가 없다"
        assert "재가중" in body and "선택" in body, f"{f} 에 둘의 차이가 없다"
