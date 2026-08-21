"""공용 픽스처 + **스킵 감시** (§26.3). GPU 를 전혀 쓰지 않는다.

kernelTab 의 `tests/conftest.py` 에서 검증된 구조를 그대로 가져왔다 (R-1).
거기서 `test_table.py`/`test_bundle.py` 41개가 pyarrow 부재로 통째로 스킵되는데
요약은 "2 skipped" 초록불이었다. 하필 그 둘이 **정답 누출 방지를 검증하는
모듈**이었다.

kernelRule 에서 그에 해당하는 것이 `CRITICAL_MODULES` 다. 이 모듈들이
수집되지 않았거나 전부 스킵되면 **세션을 실패시킨다.**

우회: `KERNELRULE_ALLOW_SKIP=1`. 우회하면 큰 경고가 나가고, 그 실행 결과로
누출 방지를 보증해서는 안 된다.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

#: 중요 모듈이 이 비율 넘게 스킵되면 경고한다.
#:
#: kernelTab R-1 은 **모듈 전체**가 스킵되는 것을 잡았다. 그런데 한 층
#: 아래가 남아 있었다 — `ran == 0` 조건이라 4개가 돌고 11개가 스킵되면
#: 그냥 통과한다. `datasets/` 가 `.gitignore` 되어 있으므로 **새로 클론한
#: 사람은 이 상태가 기본**인데 아무 신호도 없었다.
#:
#: 실측: 번들 없이 `test_leakage.py` 는 4 passed / 11 skipped (73% 스킵)다.
#: 누출 방지 검증의 1/4 만 돌면서 초록불이 뜬다.
MAX_SKIP_FRAC = 0.5

#: 이 모듈들이 안 돌면 **실패**다.
CRITICAL_MODULES = {
    "test_leakage.py", "test_scoring.py", "test_adapter.py",
    "test_noise.py", "test_weights.py", "test_synth.py",
    # 2단계 추가 — 정적 검사와 샌드박스가 조용히 안 도는 것을 막는다
    "test_checks.py", "test_sandbox.py", "test_features.py",
    "test_baselines.py",
    # 3단계 추가 — 리포트 자기모순 검사와 adversarial 차단이 핵심이다
    "test_diagnostic.py", "test_agents.py", "test_loop.py",
    # 4단계 추가 — 키 부재 폴백 금지와 예산 상한이 조용히 꺼지면 안 된다
    "test_openai_client.py",
}
ALLOW_SKIP_ENV = "KERNELRULE_ALLOW_SKIP"

REPO = Path(__file__).resolve().parent.parent
REAL_BUNDLE = REPO / "datasets" / "rtx-a6000-sm_86-c63710df"
REAL_ENV_HASH = "c63710df"

_seen: dict[str, dict] = {}


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.skipped):
        d = _seen.setdefault(Path(str(report.fspath)).name,
                             {"ran": 0, "skipped": 0})
        if report.skipped:
            d["skipped"] += 1
        else:
            d["ran"] += 1


def _bad_modules() -> list[str]:
    """세션을 **실패시켜야** 하는 것."""
    bad = []
    for mod in sorted(CRITICAL_MODULES):
        d = _seen.get(mod)
        if d is None or (d["ran"] == 0 and d["skipped"] == 0):
            bad.append(f"{mod}: 수집되지 않았다 (import 실패이거나 파일이 없다)")
        elif d["ran"] == 0:
            bad.append(f"{mod}: {d['skipped']}개가 전부 스킵됐다 — 실제로 돈 것 0개")
    return bad


def _partial_skips() -> list[str]:
    """실패까지는 아니지만 **보증할 수 없는** 것 (§26.3 한 층 아래).

    거부하지 않는 이유: 번들 없이 개발하는 것은 정상 경로다. 다만 그
    실행으로 누출 방지를 보증해서는 안 되고, 그 사실이 **보여야** 한다.
    """
    out = []
    for mod in sorted(CRITICAL_MODULES):
        d = _seen.get(mod)
        if not d:
            continue
        total = d["ran"] + d["skipped"]
        if total and d["ran"] and d["skipped"] / total > MAX_SKIP_FRAC:
            out.append(f"{mod}: {d['skipped']}/{total} "
                       f"({d['skipped'] / total:.0%}) 스킵")
    return out


class _SkipGuardItem(pytest.Item):
    """맨 마지막에 도는 합성 항목.

    `pytest_sessionfinish` 에서 `exitstatus` 를 바꾸는 방법은 pytest 버전에
    따라 전파되지 않는다. 감시가 종료 코드로 이어지지 않으면 CI 에서
    무의미하므로 진짜 테스트 항목으로 만들어 정상 실패 경로를 탄다.
    """

    def runtest(self):
        # ★ 먼저 부분 스킵을 보고한다. 실패는 아니지만 조용히 넘어가면
        #   "누출 방지 검증됨" 을 거짓으로 믿게 된다.
        partial = _partial_skips()
        if partial:
            self._warn_loudly(
                "중요 모듈이 절반 넘게 스킵됐다\n"
                + "\n".join("  - " + x for x in partial) + "\n\n"
                "★ 이 실행 결과로 누출 방지를 보증하지 마라.\n"
                "  대부분 `datasets/` 아래 번들이 없어서다. 받는 법:\n"
                "    docs/kernelrule_design_addendum.md 의 데이터 절 참조\n"
                "  번들 없이 개발하는 것은 정상이지만, 그때는 합성 표만\n"
                "  검증된 것이다.")

        bad = _bad_modules()
        if not bad:
            return
        msg = ("중요 테스트 모듈이 실제로 돌지 않았다\n"
               + "\n".join("  - " + b for b in bad) + "\n\n"
               "이 모듈들은 **정답 누출 방지와 노이즈 바닥**을 검증한다.\n"
               "스킵된 채 초록불이 뜨면 '검증됨' 을 거짓으로 믿게 된다.\n\n"
               "  고치는 법:            pip install -e '.[test]'\n"
               f"  의도적으로 넘기려면:  {ALLOW_SKIP_ENV}=1 pytest")
        if os.environ.get(ALLOW_SKIP_ENV) == "1":
            self._warn_loudly(msg)
            pytest.skip(f"{ALLOW_SKIP_ENV}=1 로 우회 — 이 실행 결과로 "
                        "누출 방지를 보증하지 마라")
        raise AssertionError(msg)

    def _warn_loudly(self, msg: str) -> None:
        """우회 경고를 **반드시 보이게** 쓴다.

        두 번 틀렸다.

        1. `print` — 스킵된 항목의 캡처 출력은 표시되지 않는다.
        2. `terminalreporter.write_line` — pytest 버전에 따라 `runtest`
           안에서 **전역 캡처에 삼켜진다.** 이 환경(9.1.1)에서는 통과했지만
           다른 버전에서는 실패했다. **감시의 보장이 pytest 버전에
           의존하면 안 된다.**

        `capturemanager` 로 캡처를 명시적으로 끄고 쓴다.
        """
        lines = ("[경고] " + msg.replace("\n", "\n[경고] ")).split("\n")
        tr = self.config.pluginmanager.get_plugin("terminalreporter")
        cm = self.config.pluginmanager.get_plugin("capturemanager")

        def _emit() -> None:
            for line in lines:
                if tr is not None:
                    tr.write_line(line, red=True, bold=True)
                else:                                    # pragma: no cover
                    print(line, file=sys.stderr, flush=True)

        if cm is not None and hasattr(cm, "global_and_fixture_disabled"):
            with cm.global_and_fixture_disabled():
                _emit()
        else:                                            # pragma: no cover
            _emit()
        # 스킵 사유에도 남긴다 — 터미널 출력이 어떤 이유로든 안 보여도
        # `-rs` 요약에는 뜬다. 한 겹 더 둔다.
        self.user_properties.append(("allow_skip_bypass", msg[:200]))

    def repr_failure(self, excinfo, style=None):
        return str(excinfo.value)

    def reportinfo(self):
        return self.path, 0, "스킵 감시 (§26.3)"


def _config_filtered(config) -> bool:
    """**일부만** 고른 실행인가. 그럴 때만 감시를 떼어 낸다.

    두 가지를 실제로 밟아 봤다.

    1. `invocation_params.args` 를 직접 훑으면 `--deselect X` 의 X 같은
       **옵션 값**을 위치 인자로 오해한다 (kernelTab 에서 그 버그로 감시가
       통째로 무력화됐다). pytest 가 이미 파싱해 둔 것을 쓴다.
    2. "위치 인자가 있으면 필터" 로 두면 `pytest /repo` / `pytest $(pwd)` 처럼
       **전체를 도는 실행에서도 감시가 꺼진다.** 메타 테스트가 이걸 잡았다.
       그래서 **디렉토리 지정은 필터가 아니다** — 파일 지정만 필터로 본다.
    """
    opt = config.option
    if getattr(opt, "keyword", "") or getattr(opt, "markexpr", ""):
        return True
    if getattr(opt, "deselect", None):
        return True
    for raw in getattr(opt, "file_or_dir", []) or []:
        # "path::TestClass::test_x" 형태도 파일 지정이다.
        path = Path(str(raw).split("::")[0])
        if path.is_file() or path.suffix == ".py":
            return True
    return False


def pytest_report_header(config):
    """번들 유무를 **헤더에 항상 표시한다.** 없으면 무엇이 안 도는지 알린다."""
    if _have_real_bundle():
        return f"kernelRule: 실제 번들 있음 ({REAL_BUNDLE.name})"
    return ("kernelRule: ⚠️ 실제 번들 없음 — 계약/누출 검증의 상당수가 "
            "스킵된다. 이 실행으로 누출 방지를 보증하지 마라")


def pytest_collection_modifyitems(session, config, items):
    for it in items:
        _seen.setdefault(Path(str(it.fspath)).name, {"ran": 0, "skipped": 0})
    if _config_filtered(config):
        return
    items.append(_SkipGuardItem.from_parent(
        session, name="test_critical_modules_actually_ran"))


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def hw_a6000():
    """A6000 을 1350/7601 MHz 로 고정했을 때의 **실효** 스펙 (§6.2)."""
    from kernelrule.core.types import Hardware
    return Hardware(name="NVIDIA RTX A6000", arch="sm_86", sm_count=84,
                    smem_per_block=101376, max_threads_per_sm=1536,
                    regs_per_sm=65536, peak_tflops_f16=116.1,
                    bandwidth_gbps=729.7, l2_bytes=6291456)


@pytest.fixture(scope="session")
def hw_other():
    """가상 GPU. 하드웨어 상수 하드코딩 검출용 (§8.3 6번).

    ★ **모든 수치 필드가 달라야 한다.** 하나라도 A6000 과 같으면 그 필드를
    쓰는 피처는 스케일 검사를 그냥 통과한다 — `regs_per_sm` 을 같게 뒀다가
    `reg_pressure` 가 "hw 를 안 쓴다" 로 잘못 걸렸다.
    """
    from kernelrule.core.types import Hardware
    return Hardware(name="FAKE", arch="sm_86", sm_count=128,
                    smem_per_block=65536, max_threads_per_sm=2048,
                    regs_per_sm=131072, peak_tflops_f16=200.0,
                    bandwidth_gbps=1000.0, l2_bytes=4194304)


@pytest.fixture(scope="session")
def noise_a6000():
    from kernelrule.core.noise import NoiseModel
    return NoiseModel.a6000_reference()


def _have_real_bundle() -> bool:
    return (REAL_BUNDLE / "BUNDLE.json").exists()


@pytest.fixture(scope="session")
def real_bundle_path():
    """실제 번들. 없으면 스킵하되 **그 사실을 표시한다** (§23.4)."""
    if not _have_real_bundle():
        pytest.skip(f"실제 번들 없음: {REAL_BUNDLE} — 계약 검증이 건너뛰어졌다")
    return REAL_BUNDLE


@pytest.fixture(scope="session")
def tiny_grid():
    """작은 격자. 실제 번들이 있으면 거기서, 없으면 열거로 만든다.

    ★ 두 경로 모두 `load_for_ranking` 만 쓴다 — 실측 시간을 볼 수 없다.
    """
    from kernelrule.tools.synth import Grid
    shapes = [(1, 4096, 4096), (128, 4096, 4096), (1024, 4096, 4096),
              (4096, 4096, 4096), (512, 512, 512), (1024, 4096, 512),
              # ★ alignment 엣지. 없으면 `can_use_cp_async` 가 상수가 되어
              #   "설명력 0" 으로 기각된다 — 격자가 좁은 것이지 피처 문제가
              #   아니다. 층 D 를 반드시 하나는 넣는다.
              (1024, 4096, 4097), (1024, 4098, 4096)]
    if _have_real_bundle():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Grid.from_bundle(REAL_BUNDLE, env_hash=REAL_ENV_HASH,
                                    shapes=shapes,
                                    max_configs_per_shape=400, seed=7)
    pytest.skip("실제 번들 없음 — 열거 격자 경로는 test_synth.py 에서 따로 시험한다")


@pytest.fixture(scope="session")
def synth_bundles(tmp_path_factory, tiny_grid):
    """프리셋별 작은 합성 번들. 세션 내내 재사용한다."""
    from kernelrule.tools.synth import generate
    out = tmp_path_factory.mktemp("synth")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {p: generate(p, seed=11, out=out, grid=tiny_grid)
                for p in ("easy", "normal", "hard", "null")}


@pytest.fixture(scope="session")
def synth_table(synth_bundles):
    from kernelrule.core.table import PerfTable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PerfTable.from_bundle(synth_bundles["normal"],
                                     env_hash="5y47he71c", ok_only=False)


@pytest.fixture(scope="session")
def null_table(synth_bundles):
    from kernelrule.core.table import PerfTable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PerfTable.from_bundle(synth_bundles["null"],
                                     env_hash="5y47he71c", ok_only=False)
