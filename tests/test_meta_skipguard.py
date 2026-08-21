"""★ 감시에 감시 (§30.8).

`conftest.py` 의 스킵 감시가 **정말로 세션을 실패시키는가**. kernelTab 은
감시를 만들어 놓고 `config_filtered` 의 인자 파싱 버그로 감시가 통째로
무력화된 적이 있다. 감시가 있다는 사실만으로는 아무것도 보장되지 않는다.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).parent / "conftest.py"
STUB = '''
def test_ok():
    assert True
'''
#: 모듈이 **수집조차 안 되는** 경우 (import 단계에서 걸린다).
MODULE_SKIP_STUB = '''
import pytest
pytest.skip("환경이 안 맞아서", allow_module_level=True)

def test_never_runs():
    assert False
'''

#: 모듈은 수집되지만 **모든 테스트가 스킵되는** 경우. 다른 분기다.
ALL_SKIPPED_STUB = '''
import pytest

@pytest.mark.skip(reason="환경이 안 맞아서")
def test_a():
    assert False

@pytest.mark.skip(reason="환경이 안 맞아서")
def test_b():
    assert False
'''


def _run(tmp: Path, *args, env_extra=None) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ)
    env.pop("KERNELRULE_ALLOW_SKIP", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         *args, str(tmp)],
        capture_output=True, text=True, cwd=tmp, env=env, timeout=180)


@pytest.fixture
def guarded(tmp_path: Path) -> Path:
    """감시만 남기고 중요 모듈은 스텁으로 채운 최소 테스트 트리."""
    src = CONFTEST.read_text()
    # 픽스처가 무거우므로 감시 부분만 남긴다.
    head = src.split("# ---------------------------------------------------------------------------\n# 픽스처")[0]
    (tmp_path / "conftest.py").write_text(head)
    from conftest import CRITICAL_MODULES
    for name in CRITICAL_MODULES:
        (tmp_path / name).write_text(STUB)
    return tmp_path


def test_guard_passes_when_everything_runs(guarded):
    r = _run(guarded)
    assert r.returncode == 0, r.stdout[-2000:]


def test_guard_fails_when_a_critical_module_is_missing(guarded):
    (guarded / "test_leakage.py").unlink()
    r = _run(guarded)
    assert r.returncode != 0
    assert "test_leakage.py" in r.stdout and "수집되지 않았다" in r.stdout


def test_guard_fails_when_a_critical_module_is_not_collected(guarded):
    """모듈 최상단에서 스킵되면 수집 자체가 안 된다."""
    (guarded / "test_noise.py").write_text(MODULE_SKIP_STUB)
    r = _run(guarded)
    assert r.returncode != 0
    assert "수집되지 않았다" in r.stdout


def test_guard_fails_when_every_test_in_a_module_is_skipped(guarded):
    """수집은 됐지만 실제로 돈 것이 0개인 경우 — 다른 분기다."""
    (guarded / "test_noise.py").write_text(ALL_SKIPPED_STUB)
    r = _run(guarded)
    assert r.returncode != 0
    assert "전부 스킵됐다" in r.stdout


def test_guard_can_be_bypassed_loudly(guarded):
    (guarded / "test_noise.py").write_text(ALL_SKIPPED_STUB)
    r = _run(guarded, env_extra={"KERNELRULE_ALLOW_SKIP": "1"})
    assert r.returncode == 0
    assert "[경고]" in r.stdout, "우회가 조용히 일어났다"


def test_guard_does_not_fire_on_filtered_runs(guarded):
    """`-k` 로 일부만 고른 실행에는 감시가 붙지 않는다."""
    (guarded / "test_leakage.py").unlink()
    r = _run(guarded, "-k", "test_ok")
    assert r.returncode == 0, r.stdout[-2000:]
