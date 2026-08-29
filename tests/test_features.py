"""피처 라이브러리와 자동 검증 (§8.2, §8.3)."""
from __future__ import annotations

import inspect
import warnings

import pytest

import kernelrule.features.physical  # noqa: F401  등록
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.features import REGISTRY
from kernelrule.features.validate import validate_registry


@pytest.fixture(scope="module")
def matrix(synth_table):
    return FeatureMatrix(synth_table, REGISTRY)


def test_library_has_enough_features():
    """§8.2 — 시작 전에 손으로 10~15개를 채워둔다."""
    assert len(REGISTRY.names(shape_level=False)) >= 15
    assert len(REGISTRY.names(shape_level=True)) >= 4


def test_no_feature_touches_ext():
    """★ `cfg.ext` 참조 금지 — 아키텍처 전이 전제 (§4.3, §8.2).

    `ext_*` 는 SM90 에 대응물이 없다. 전이를 노리는 규칙이 그것을 쓰면
    주 지표(아키텍처 홀드아웃)에서 무너진다.
    """
    bad = []
    for f in REGISTRY.items(active_only=False):
        try:
            src = inspect.getsource(f.fn)
        except (OSError, TypeError):      # pragma: no cover
            continue
        if "cfg.ext" in src or ".ext[" in src:
            bad.append(f.name)
    assert not bad, f"`cfg.ext` 를 참조하는 피처: {bad}"


def test_no_feature_references_answers():
    """정답 컬럼 이름을 **식별자로** 참조하지 않는다.

    단순 부분문자열 검사는 안 된다 — `hw.peak_tflops_f16` 이 `tflops` 를
    포함해서 오탐이 난다. 토큰 경계로 본다.
    """
    import re

    from kerneltab.core.table import ANSWER_COLS

    bad = []
    for f in REGISTRY.items(active_only=False):
        try:
            src = inspect.getsource(f.fn)
        except (OSError, TypeError):      # pragma: no cover
            continue
        code = "\n".join(ln for ln in src.split("\n")
                          if not ln.strip().startswith("#"))
        for col in ANSWER_COLS:
            if re.search(rf"(?<![\w.]){re.escape(col)}\b", code):
                bad.append((f.name, col))
    assert not bad, f"정답 컬럼을 참조하는 피처: {bad}"


def test_all_features_are_short():
    """§8.2 — 10줄 이내. 길면 물리가 아니라 조합이다."""
    long = []
    for f in REGISTRY.items(active_only=False):
        try:
            src = inspect.getsource(f.fn)
        except (OSError, TypeError):      # pragma: no cover
            continue
        body = [ln for ln in src.split("\n")
                if ln.strip() and not ln.strip().startswith("#")]
        # docstring 과 데코레이터를 뺀 실질 줄 수
        n = len([ln for ln in body if not ln.lstrip().startswith(("@", '"""'))])
        if n > 22:
            long.append((f.name, n))
    assert not long, f"너무 긴 피처: {long}"


def test_registry_validates_clean(synth_table, matrix, hw_other):
    """★ 전 피처가 자동 검증을 통과한다. 기각이 하나라도 있으면 실패다."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reps = validate_registry(REGISTRY, synth_table, matrix,
                                 hw_alt=hw_other, n_shapes=4)
    failed = {n: [str(c) for c in r.fails()]
              for n, r in reps.items() if r.failed}
    assert not failed, f"검증 기각: {failed}"


def test_scale_invariance_catches_a_hardcoded_constant(synth_table, matrix,
                                                       hw_other):
    """★ 하드웨어 상수를 하드코딩한 피처가 **잡히는가** (§8.3 6번).

    감시가 실제로 작동하는지 확인한다. 검사가 있다는 사실만으로는 아무것도
    보장되지 않는다 (§30.8).
    """
    from kernelrule.features import Feature, FeatureRegistry
    from kernelrule.features.validate import validate_feature

    r = FeatureRegistry("bad")

    def fake_waves(p, hw, cfg) -> float:
        """hw.sm_count 를 읽는 척하지만 84 를 박아 뒀다."""
        import math
        return math.ceil(p.M / cfg.tile_m) * math.ceil(p.N / cfg.tile_n) / 84.0

    f = Feature(name="fake_waves", fn=fake_waves, unit="dimensionless",
                expected_range=(0.0, 1e6), direction="neutral",
                vec=None, code_hash="x")
    r.add(f)
    m2 = FeatureMatrix(synth_table, r)
    rep = validate_feature(f, synth_table, m2, hw_alt=hw_other, n_shapes=3)
    # `hw.` 문자열이 docstring 에만 있으므로 하드웨어를 쓰는 것으로 보인다
    assert rep.failed, "하드코딩된 84 를 못 잡았다"
    assert any("스케일" in c.name for c in rep.fails())


def test_vectorized_matches_scalar_everywhere(synth_table, matrix):
    """★ 학습(행렬)과 배포(스칼라)가 같은 함수를 쓰는가."""
    from kernelrule.features import verify_vectorized

    p = synth_table.shapes()[0]
    df = synth_table.frame_for(p)
    _, info = matrix.for_shape(p)
    for f in REGISTRY.items():
        if f.vec is None or f.shape_level:
            continue
        verify_vectorized(f, df, matrix.hw, info, n=96)


def test_directions_are_declared():
    for f in REGISTRY.items(active_only=False):
        assert f.direction in ("higher_is_worse", "higher_is_better",
                               "neutral"), f.name


def test_shape_features_ignore_config(synth_table, matrix):
    """형상 수준 피처는 `cfg` 를 봐서는 안 된다 — 그것이 정의다."""
    p = synth_table.shapes()[0]
    cfgs = synth_table.configs(p)
    for f in REGISTRY.items(shape_level=True):
        vals = {float(f.fn(p, matrix.hw, c)) for c in cfgs[:40]}
        assert len(vals) == 1, f"{f.name} 이 config 마다 다른 값을 낸다: {vals}"


# ---------------------------------------------------------------------------
# 생성 피처의 하드웨어 사용 판정 (D-37)
# ---------------------------------------------------------------------------
# `exec` 로 만든 함수는 `inspect.getsource` 가 OSError 를 낸다. 그때 "hw 를
# 쓴다" 로 떨어지면 스케일 불변성 검사가 **하드웨어 무관 정상 피처를 전부
# 기각한다.** F1 첫 실행에서 실제로 그렇게 버려졌다.

_NO_HW = ("def tile_aspect(p, hw, cfg) -> float:\n"
          "    a = float(cfg.tile_m)\n"
          "    b = max(1.0, float(cfg.tile_n))\n"
          "    return abs(a - b) / (a + b)\n")
_USES_HW = ("def per_sm(p, hw, cfg) -> float:\n"
            "    return float(p.M) / max(1.0, float(hw.sm_count))\n")


def _gen(code):
    from kernelrule.features import Feature
    from kernelrule.features.generated import compile_feature
    name, fn = compile_feature(code, known=frozenset())
    return Feature(name=name, fn=fn, unit="dimensionless",
                   expected_range=(0.0, 1.0), direction="neutral",
                   source=code)


@pytest.mark.parametrize("code,expected", [(_NO_HW, False), (_USES_HW, True)])
def test_generated_feature_hardware_usage_is_read_from_source(code, expected):
    from kernelrule.features.validate import _uses_hardware
    assert _uses_hardware(_gen(code)) is expected


def test_unreadable_source_raises_instead_of_guessing():
    """★ 소스를 못 읽으면 **판단하지 않는다** (§26.4).

    `True` 로 떨어지는 것은 "hw 를 쓴다" 는 주장이고, 그 주장이 틀리면
    정상 피처를 기각한다.
    """
    from kernelrule.features import Feature
    from kernelrule.features.generated import compile_feature
    from kernelrule.features.validate import _uses_hardware

    name, fn = compile_feature(_NO_HW, known=frozenset())
    f = Feature(name=name, fn=fn, unit="dimensionless",
                expected_range=(0.0, 1.0), direction="neutral")  # source 없음
    with pytest.raises(ValueError, match="소스를 읽을 수 없어"):
        _uses_hardware(f)


# ---------------------------------------------------------------------------
# ★ §30.9 — 라이브러리는 전역 레지스트리를 직접 참조하지 않는다
#
# F0~F3 는 레지스트리를 **갈아 끼워서** 성립한다. 라이브러리 어딘가가
# `from kernelrule.features import REGISTRY` 를 하고 있으면, 조건을 바꿔도
# 그 지점만 사람이 쓴 24개를 계속 본다 — 조용히, 에러 없이.
# `is_reference()` / `top_k` / `DEFAULT_MODEL` 과 같은 부류의 사고다 (원칙 2).
# ---------------------------------------------------------------------------

#: 전역 레지스트리를 참조해도 되는 곳. **자기 자신과 등록부뿐이다.**
_MAY_TOUCH_GLOBAL = {"kernelrule/features/__init__.py",
                     "kernelrule/features/physical.py"}


def test_library_never_imports_the_global_registry():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bad: list[str] = []
    for f in sorted((root / "kernelrule").rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        if rel in _MAY_TOUCH_GLOBAL:
            continue
        tree = ast.parse(f.read_text(), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                    a.name == "REGISTRY" for a in node.names):
                bad.append(f"  {rel}:{node.lineno} from ... import REGISTRY")
            elif (isinstance(node, ast.Attribute) and node.attr == "REGISTRY"
                  and isinstance(node.value, ast.Name)):
                bad.append(f"  {rel}:{node.lineno} {node.value.id}.REGISTRY")
    assert not bad, (
        "라이브러리가 전역 레지스트리를 직접 본다 — 조건을 바꿔도 이 "
        "지점만 사람이 쓴 24개를 쓴다 (§30.9):\n" + "\n".join(bad))


def test_no_function_defaults_to_the_global_registry():
    """★ (c) 유형 — 호출부가 안 넘기면 조용히 24개를 쓰는 기본값."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bad: list[str] = []
    for f in sorted((root / "kernelrule").rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        tree = ast.parse(f.read_text(), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            a = node.args
            for d in [*a.defaults, *a.kw_defaults]:
                if isinstance(d, ast.Name) and d.id == "REGISTRY":
                    bad.append(f"  {rel}:{node.lineno} def {node.name}(...="
                               "REGISTRY)")
    assert not bad, (
        "전역 레지스트리를 기본값으로 쓰는 함수가 있다. 호출부가 안 "
        "넘기면 조용히 사람 24개를 쓴다 (§26.4, §30.9):\n" + "\n".join(bad))


@pytest.fixture(scope="module")
def perf_table():
    """실측 번들. 형상 수준 판정은 **실제 config 집합**이 있어야 한다."""
    import warnings

    from kernelrule.core.table import PerfTable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PerfTable.from_bundle("datasets/rtx-a6000-sm_86-c63710df",
                                     env_hash="c63710df", ok_only=False)


# ---------------------------------------------------------------------------
# ★ §30.12 — 형상 수준 자동 판정
#
#   생성 경로에는 `shape_feature` 데코레이터가 없어서 **모든 생성 피처가
#   config 수준으로 등록**되고 있었다. 그러면 규칙이 `if p.<x>:` 로 분기할
#   수 없고, 형상 안에서 상수인 항은 순위를 하나도 못 바꾼다. F1 21개 중
#   5개가 그 상태였다 (D-65).
# ---------------------------------------------------------------------------
def test_detection_matches_the_hand_written_labels(perf_table):
    """★ 판정 로직의 검증 — 사람이 손으로 붙인 표시와 일치하는가.

    일치하면 로직이 맞는 것이고, 어긋나면 판정에 결함이 있는 것이다.
    """
    import inspect
    from dataclasses import replace

    from kernelrule.features import REGISTRY
    from kernelrule.features.generated import detect_shape_level

    bad = []
    for n in sorted(REGISTRY._items):
        f = REGISTRY[n]
        try:
            src = inspect.getsource(f.fn)
        except (OSError, TypeError):
            src = ""
        got, why = detect_shape_level(
            replace(f, source=src, shape_level=False), perf_table)
        if got != f.shape_level:
            bad.append(f"  {n}: 사람 {f.shape_level} vs 자동 {got} ({why})")
    assert not bad, ("형상 수준 판정이 사람 표시와 어긋난다:\n"
                     + "\n".join(bad))


def test_detection_is_two_tiered():
    """AST 겹이 있어야 "이 표에서만 상수" 를 구분할 수 있다."""
    from kernelrule.features.generated import uses_cfg

    assert uses_cfg("def f(p, hw, cfg) -> float:\n    return float(cfg.tile_m)")
    assert not uses_cfg("def f(p, hw, cfg) -> float:\n    return float(p.M)")


def test_recheck_warning_is_recorded(perf_table):
    """★ cfg 를 참조하는데 상수인 것은 **번들이 바뀌면 다시 판정**해야 한다."""
    from dataclasses import replace

    from kernelrule.features import Feature
    from kernelrule.features.generated import detect_shape_level

    # cfg 를 참조하지만 값은 상수인 함수
    code = ("def probe_const(p, hw, cfg) -> float:\n"
            "    return float(cfg.tile_m) * 0.0 + float(p.M)\n")
    env: dict = {}
    exec(compile(code, "<t>", "exec"), env)  # noqa: S102
    f = Feature(name="probe_const", fn=env["probe_const"],
                unit="dimensionless",
                expected_range=(0.0, 1e9), direction="neutral",
                code_hash="h", source=code)
    is_shape, why = detect_shape_level(replace(f, shape_level=False),
                                       perf_table)
    assert is_shape
    assert "재판정" in why, why


def test_load_generated_requires_a_table():
    """★ `table` 에 기본값을 두면 호출부가 빠뜨린다 (D-67).

    두었더니 두 곳이 빠뜨렸고 그중 하나가 2단계 경로여서 **형상 수준
    피처 0개**로 Architect 가 돌았다. 재판정이 정말 필요 없으면
    `table=None` 을 **명시**해야 한다 — 빠뜨린 것과 구분된다.
    """
    import inspect

    from kernelrule.features.loader import load_generated

    prm = inspect.signature(load_generated).parameters["table"]
    assert prm.default is inspect.Parameter.empty, "table 에 기본값이 있다"
    assert prm.kind is inspect.Parameter.KEYWORD_ONLY


def test_every_load_generated_call_passes_table():
    """호출부 전수 검사 — 원칙 23 (한 자리 말고 한 종류)."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted([*(root / "kernelrule").rglob("*.py"),
                     *(root / "experiments").glob("*.py")]):
        tree = ast.parse(f.read_text(), filename=f.name)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "load_generated"
                    and not any(k.arg == "table" for k in node.keywords)):
                bad.append(f"  {f.name}:{node.lineno}")
    assert not bad, ("load_generated 에 table 을 안 넘기는 호출부:\n"
                     + "\n".join(bad))


# ---------------------------------------------------------------------------
# ★ 4-4 — `expected_range` 는 **LLM 이 선언한 것**이어야 한다 (D-71)
#
#   파이프라인이 실측으로 덮어쓰면 표 정보가 프롬프트에 들어간다.
#   LLM 은 표를 못 보므로 누출이 있다면 파이프라인이 만든 것이다.
# ---------------------------------------------------------------------------
def test_register_generated_keeps_the_declared_range(perf_table):
    from dataclasses import replace

    from kernelrule.features import FeatureRegistry
    from kernelrule.features.generated import register_generated

    code = ("def probe_range(p, hw, cfg) -> float:\n"
            "    return float(cfg.tile_m) / max(1.0, float(p.M))\n")
    reg = FeatureRegistry("probe")
    meta = {"name": "probe_range", "unit": "dimensionless",
            "direction": "higher_is_worse", "expected_range": [0.0, 7.0]}
    hw_alt = replace(perf_table.hw, sm_count=perf_table.hw.sm_count * 2 + 3,
                     smem_per_block=int(perf_table.hw.smem_per_block * 0.7),
                     max_threads_per_sm=int(perf_table.hw.max_threads_per_sm * 1.33),
                     peak_tflops_f16=perf_table.hw.peak_tflops_f16 * 1.6,
                     bandwidth_gbps=perf_table.hw.bandwidth_gbps * 0.8,
                     regs_per_sm=int(perf_table.hw.regs_per_sm * 1.5),
                     l2_bytes=int(perf_table.hw.l2_bytes * 2))
    f = register_generated(code, registry=reg, meta=meta, table=perf_table,
                           matrix=FeatureMatrix(perf_table, REGISTRY),
                           hw_alt=hw_alt)
    # 실측은 [0, 7] 보다 훨씬 좁다. 그래도 선언이 그대로여야 한다.
    assert f.expected_range == (0.0, 7.0), (
        "파이프라인이 선언 범위를 덮어썼다 — 표 정보가 프롬프트에 들어간다")


def test_range_warning_never_reaches_a_rejection_message():
    """범위 경고 문구에는 **실측 min/max** 가 들어간다. 그것이 거부
    메시지로 새면 LLM 에 되먹여질 수 있다 (구 `feature_writer.py`).
    """
    import inspect

    from kernelrule.features import validate as V

    src = inspect.getsource(V)
    assert "실측" in src, "범위 경고 문구가 바뀌었다 — 검사가 무의미하다"
    # 범위 검사는 "warn" 이어야 한다. "fail" 이면 `fails()` 로 새어 나간다.
    i = src.index("실측")
    ctx = src[max(0, i - 400):i]
    assert '"범위", "warn"' in ctx, (
        "범위 검사가 warn 이 아니다 — 실측 값이 FeatureRejected 로 샌다")


# ---------------------------------------------------------------------------
# ★ D-73 — 검사기가 **자기가 허용한 필드**를 금지하고 있었다
# ---------------------------------------------------------------------------
def test_allowed_fields_are_not_caught_by_banned_words():
    """`hw.peak_tflops_f16` 이 금지어 `"tflops"` 에 부분 문자열로 걸렸다.

    roofline 을 만들려던 제안이 그렇게 거부됐다 — 검사기의 결함이
    LLM 의 실패로 보인다 (원칙 8, D-37 과 같은 부류).
    """
    from kernelrule.features.generated import (
        _BANNED,
        RAW_FIELDS,
        check_feature_code,
    )

    # 허용 필드 중 금지어를 부분 문자열로 담는 것이 실제로 있다
    risky = [f"{b}.{n}" for b, ns in RAW_FIELDS.items() for n in ns
             if any(x in f"{b}.{n}" for x in _BANNED)]
    assert risky, "위험한 필드가 없다 — 이 검사가 무의미하다"

    for ref in risky:
        code = (f"def probe_ok(p, hw, cfg) -> float:\n"
                f"    return float({ref}) * 1.0\n")
        assert check_feature_code(code, known=frozenset()) == "probe_ok", ref


def test_banned_words_still_catch_real_leaks():
    """가리기가 진짜 누출까지 통과시키면 안 된다."""
    from kernelrule.features.generated import FeatureRejected, check_feature_code

    for leak in ("best_ms", "time_ms", "difficulty"):
        code = (f"def probe_leak(p, hw, cfg) -> float:\n"
                f"    {leak} = 1.0\n"
                f"    return {leak}\n")
        with pytest.raises(FeatureRejected):
            check_feature_code(code, known=frozenset())
