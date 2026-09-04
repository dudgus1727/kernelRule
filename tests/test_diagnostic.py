"""진단 리포트 (§12). **템플릿이 결론을 미리 쓰지 못하게 강제한다.**"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import numpy as np
import pytest
from toy import make_table

import kernelrule.report.diagnostic as D
from kernelrule.core.splits import Split, SplitError

#: 계산 없이 쓰면 안 되는 비교어. 이 데이터가 아니라 **다른 데이터의 결론**을
#: 템플릿에 박아 두는 것이 이 프로젝트가 밟은 함정이다.
COMPARATIVES = ("보다", "더 크", "더 작", "지배적", "대부분", "훨씬",
                "가장 큰", "가장 작", "대체로", "일반적으로")

#: 예외 — 물리 상수이지 이 분할의 관측이 아니다.
EXEMPT_FUNCS = {"hardware_block"}


def _rendered_literals(fn_name: str, node: ast.AST):
    """`add(...)` / `L.append(...)` 로 렌더되는 문자열 리터럴을 뽑는다.

    반환: (텍스트, 계산값을 포함하는가)
    """
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        f = sub.func
        name = (f.id if isinstance(f, ast.Name) else
                f.attr if isinstance(f, ast.Attribute) else "")
        if name not in ("add", "append"):
            continue
        for arg in sub.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append((arg.value, False))
            elif isinstance(arg, ast.JoinedStr):
                lit = "".join(v.value for v in arg.values
                              if isinstance(v, ast.Constant))
                computed = any(isinstance(v, ast.FormattedValue)
                               for v in arg.values)
                out.append((lit, computed))
            elif isinstance(arg, ast.BinOp):
                for side in (arg.left, arg.right):
                    if isinstance(side, ast.Constant) and isinstance(
                            side.value, str):
                        out.append((side.value, False))
                    elif isinstance(side, ast.JoinedStr):
                        lit = "".join(v.value for v in side.values
                                      if isinstance(v, ast.Constant))
                        computed = any(isinstance(v, ast.FormattedValue)
                                       for v in side.values)
                        out.append((lit, computed))
    return out


def test_template_never_states_an_uncomputed_comparison():
    """★ 이 프로젝트가 실제로 밟은 함정을 코드로 막는다.

    블록 3 에 "크기 층화가 난이도 층화보다 크게 갈린다" 를 미리 적어 뒀는데
    학습 분할의 실제 숫자는 반대였다 (0.0007 vs 0.1651). **리포트가 자기
    데이터와 모순되면 LLM 은 데이터가 아니라 문장을 믿는다.**

    규칙: 렌더되는 문자열에 비교어가 있으면 그 문자열은 **계산된 값을
    포함해야 한다.** 순수 리터럴 비교 서술은 금지다.
    """
    tree = ast.parse(Path(D.__file__).read_text())
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name in EXEMPT_FUNCS:
            continue
        for text, computed in _rendered_literals(node.name, node):
            if computed:
                continue
            for w in COMPARATIVES:
                if w in text:
                    bad.append(f"{node.name}(): {w!r} in {text[:70]!r}")
    assert not bad, (
        "템플릿이 계산되지 않은 비교 서술을 담고 있다:\n  "
        + "\n  ".join(bad)
        + "\n  계산 결과로 문장을 만들어라 (§12).")


def _toy(regret_by_shape):
    """형상별 regret 을 지정한 장난감 표 + 항등 규칙."""
    times = {}
    for (m, n, k), rel in regret_by_shape.items():
        times[(m, n, k)] = [1.0, float(rel)]
    return make_table(times)


def test_stratification_sentence_follows_the_data():
    """★ 같은 템플릿이 **반대 데이터에서 반대 문장**을 내야 한다.

    소스 검사만으로는 부족하다 — 계산된 f-string 안에 결론을 박을 수도 있다.
    행동으로 확인한다.
    """
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.noise import NoiseModel
    from kernelrule.features import Feature, FeatureRegistry

    reg = FeatureRegistry("toy")
    reg.add(Feature(name="idx", fn=lambda p, hw, c: 0.0, unit="dimensionless",
                    expected_range=(0.0, 10.0), direction="neutral",
                    vec=lambda df, hw, p: np.arange(len(df), dtype=float),
                    code_hash="x"))
    reg.add(Feature(name="log_sol_ms", fn=lambda p, hw, c: 0.0,
                    expected_range=(-30.0, 30.0), direction="neutral",
                    unit="dimensionless", shape_level=True, code_hash="y"))
    reg.add(Feature(name="is_memory_bound", fn=lambda p, hw, c: 0.0,
                    expected_range=(0.0, 1.0), direction="neutral",
                    unit="dimensionless", shape_level=True, code_hash="z"))

    def build(times):
        t = make_table(times, noise=NoiseModel.a6000_reference())
        m = FeatureMatrix(t, reg)
        return t, m

    code = "def score(f, p, hw, w):\n    return f.idx * w[0]\n"

    def score(f, p, hw, w):
        return f.idx * w[0]

    # 짧은 형상이 나쁜 표 / 긴 형상이 나쁜 표
    short_bad = {(64, 64, 64): [0.02, 0.10], (4096, 4096, 4096): [2.0, 2.02]}
    long_bad = {(64, 64, 64): [0.02, 0.0202], (4096, 4096, 4096): [2.0, 10.0]}
    texts = []
    for times in (short_bad, long_bad):
        t, m = build(times)
        rep = D.build_report(run_id="toy", table=t, matrix=m, score_fn=score,
                             weights=[-1.0], code=code,
                             train=Split("train", tuple(t.shapes())))
        texts.append(rep.render())
    a, b = texts
    assert a != b, "데이터가 반대인데 리포트가 같다"
    # 두 리포트의 크기 격차 부호가 반대여야 한다
    import re
    ga = float(re.search(r"격차 ([+-][\d.]+)", a).group(1))
    gb = float(re.search(r"격차 ([+-][\d.]+)", b).group(1))
    assert ga * gb < 0, f"크기 격차가 데이터를 따르지 않는다: {ga} vs {gb}"


def test_report_refuses_non_train_split():
    """★ 홀드아웃 점수가 프롬프트에 들어가는 경로를 만들지 않는다 (§12.3)."""
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import Feature, FeatureRegistry

    reg = FeatureRegistry("t2")
    reg.add(Feature(name="idx", fn=lambda p, hw, c: 0.0, unit="dimensionless",
                    expected_range=(0.0, 10.0), direction="neutral",
                    vec=lambda df, hw, p: np.arange(len(df), dtype=float),
                    code_hash="x"))
    t = make_table({(1024, 4096, 4096): [1.0, 2.0]})
    m = FeatureMatrix(t, reg)
    for role in ("val", "test"):
        with pytest.raises(SplitError, match="학습 분할만"):
            D.build_report(run_id="x", table=t, matrix=m,
                           score_fn=lambda f, p, hw, w: f.idx * w[0],
                           weights=[1.0], code="def score(f, p, hw, w):\n"
                                                "    return f.idx * w[0]\n",
                           train=Split(role, tuple(t.shapes())))


def test_sigma_is_relative_to_noise_floor():
    from kernelrule.core.noise import NoiseModel
    m = NoiseModel.a6000_reference()
    # 4ms 형상: 바닥 0.053%. 1% 차이면 약 19시그마
    assert D._sigma(4.04, 4.0, m) == pytest.approx(0.01 / m.floor(4.0), rel=1e-6)
    # 11us 형상: 바닥 9.1%. 같은 1% 차이면 0.11시그마 — 노이즈 안이다
    assert D._sigma(0.0113 * 1.01, 0.0113, m) < 1.0


@pytest.mark.needs_bundle
def test_real_report_has_no_holdout_and_fits_budget(real_bundle_path):
    """실제 표로 만든 리포트가 예산 안이고 홀드아웃 형상을 안 담는다."""
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import split_by_M_range
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY
    from kernelrule.rules.human_guided import CODE, W0, score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = PerfTable.from_bundle(real_bundle_path, env_hash="c63710df",
                                   ok_only=False)
        fm = FeatureMatrix(tb, REGISTRY)
        sp = split_by_M_range(tb.shapes())
        rep = D.build_report(run_id="t", table=tb, matrix=fm, score_fn=score,
                             weights=W0, code=CODE, train=sp.train)
    txt = rep.render()
    assert rep.token_estimate() < 9000, rep.token_estimate()
    held = {f"{p.M}x{p.N}x{p.K}" for p in sp.val}
    for h in held:
        assert h not in txt, f"홀드아웃 형상 {h} 이 리포트에 들어갔다"
    # 사례마다 σ 와 사용여부가 있다
    assert "노이즈 바닥의" in txt and "미사용" in txt


# ---------------------------------------------------------------------------
# 집계도 홀드아웃을 넘지 않는다 (§12.3 / D-28)
# ---------------------------------------------------------------------------
# `build_report` 는 train 분할만 받는데, `table_facts` 가 자유 문자열이라
# 그 검사를 **완전히 우회했다**. 첫 실제 실행의 블록 3.5 가 전수 표에서
# 계산된 값이었다. 우회 경로 자체를 없앴고, 그것을 여기서 고정한다.

def _facts_table():
    return make_table({(1024, 4096, 4096): [1.0, 2.0, 3.0],
                       (512, 4096, 4096): [1.0, 1.5],
                       (2048, 4096, 4096): [2.0, 2.2, 9.0]})


def test_table_facts_rejects_holdout_splits():
    from kernelrule.report.table_facts import TableFacts
    t = _facts_table()
    shapes = tuple(t.shapes())
    TableFacts.compute(t, Split("train", shapes))       # train 은 통과
    for role in ("val", "test"):
        with pytest.raises(SplitError, match="12.3"):
            TableFacts.compute(t, Split(role, shapes))


def test_table_facts_only_sees_the_train_shapes():
    """★ 학습 분할에 없는 형상이 집계에 섞이면 안 된다."""
    from kernelrule.report.table_facts import TableFacts
    t = _facts_table()
    shapes = list(t.shapes())
    part = TableFacts.compute(t, Split("train", tuple(shapes[:2])))
    whole = TableFacts.compute(t, Split("train", tuple(shapes)))
    assert part.n_shapes == 2
    assert whole.n_shapes == len(shapes)
    assert "학습 분할 2형상" in part.lines[0]
    assert part.lines != whole.lines


def test_build_report_refuses_raw_strings():
    """자유 문자열을 받으면 분할 검사가 아무 일도 하지 않는다."""
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import Feature, FeatureRegistry

    reg = FeatureRegistry("t3")
    reg.add(Feature(name="idx", fn=lambda p, hw, c: 0.0, unit="dimensionless",
                    expected_range=(0.0, 10.0), direction="neutral",
                    vec=lambda df, hw, p: np.arange(len(df), dtype=float),
                    code_hash="x"))
    t = make_table({(1024, 4096, 4096): [1.0, 2.0]})
    m = FeatureMatrix(t, reg)
    with pytest.raises(SplitError, match="TableFacts.compute"):
        D.build_report(run_id="x", table=t, matrix=m,
                       score_fn=lambda f, p, hw, w: f.idx * w[0],
                       weights=[1.0],
                       code="def score(f, p, hw, w):\n    return f.idx * w[0]\n",
                       train=Split("train", tuple(t.shapes())),
                       table_facts=["전수 표에서 계산한 문장"])


# ---------------------------------------------------------------------------
# ★ §12.3b — 블록 3.5 는 **축을 지목하면 안 된다**
#
#   "스필 커널이 정답 집합에 든 형상 0/61개" 는 정답 요약이다. 그 줄이
#   있으면 LLM 은 `has_spill` 을 볼 필요가 없다는 것을 표에서 배운 것이
#   아니라 **받은** 것이다. 같은 이유로 GBDT 피처 중요도도 뺐다.
#
#   F0~F3 에서는 더 심각하다 — LLM 이 만든 피처는 이름이 전부 다르므로
#   축 이름 매핑이 아예 안 맞는데도 프롬프트에는 남는다.
# ---------------------------------------------------------------------------

#: 블록 3.5 에 나오면 안 되는 것 — config 축 컬럼 이름과 그 값.
_FORBIDDEN_AXES = ("has_spill", "스필", "ext_stages", "stages=2", "pipelined",
                   "ext_warp_m", "warp_m", "split_k_mode", "parallel",
                   "mainloop_iters", "workspace_bytes", "waves_occ",
                   "grid_tiles", "inst_total", "tile_m",
                   "regs_total_per_block", "GBDT")


def test_block_3_5_never_names_an_axis():
    import warnings

    from kernelrule.core.splits import Split
    from kernelrule.core.table import PerfTable
    from kernelrule.report.table_facts import TableFacts

    bundle = "datasets/rtx-a6000-sm_86-c63710df"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = PerfTable.from_bundle(bundle, env_hash="c63710df", ok_only=False)
    train = Split("train", tuple(p for p in t.shapes()
                                 if 11008 not in (p.N, p.K)))
    facts = TableFacts.compute(t, train)

    body = "\n".join(facts.lines)
    hit = [a for a in _FORBIDDEN_AXES if a in body]
    assert not hit, (
        f"블록 3.5 가 축을 지목한다: {hit}\n"
        "이것은 정답 요약이다 (§12.3b). 남겨도 되는 것은 '여지의 크기' "
        "뿐이다 — 고정 config top-k, 체제별 분해, 동률 폭.\n" + body)


def test_block_3_5_never_annotates_individual_features():
    """`by_feature` 는 각 피처 **설명에** 표 관측을 붙인다 — 축 지목이다."""
    import warnings

    from kernelrule.core.splits import Split
    from kernelrule.core.table import PerfTable
    from kernelrule.report.table_facts import TableFacts

    bundle = "datasets/rtx-a6000-sm_86-c63710df"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = PerfTable.from_bundle(bundle, env_hash="c63710df", ok_only=False)
    train = Split("train", tuple(p for p in t.shapes()
                                 if 11008 not in (p.N, p.K)))
    assert not TableFacts.compute(t, train).by_feature, (
        "피처별 표 관측이 살아 있다. 프롬프트의 피처 설명에 '이 축은 "
        "정답에 0번 들어간다' 가 붙는다 (§12.3b).")
