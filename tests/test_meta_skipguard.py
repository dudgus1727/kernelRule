"""★ Watching the watch (§30.8).

Does `conftest.py`'s skip guard **really fail the session**? kernelTab once
built a guard and then disabled it entirely through an argument-parsing bug
in `config_filtered`. The mere fact that a guard exists guarantees nothing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).parent / "conftest.py"
STUB = '''
def test_ok():
    assert True
'''
#: The case where the module **is not even collected** (it is caught at the
#: import stage).
MODULE_SKIP_STUB = '''
import pytest
pytest.skip("the environment does not fit", allow_module_level=True)

def test_never_runs():
    assert False
'''

#: The case where the module is collected but **every test is skipped**. It is
#: a different branch.
ALL_SKIPPED_STUB = '''
import pytest

@pytest.mark.skip(reason="the environment does not fit")
def test_a():
    assert False

@pytest.mark.skip(reason="the environment does not fit")
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
        capture_output=True, text=True, cwd=tmp, env=env, timeout=180,
        check=False)      # the return code itself is what is checked


@pytest.fixture
def guarded(tmp_path: Path) -> Path:
    """A minimal test tree with only the guard kept and the critical modules
    filled in with stubs."""
    src = CONFTEST.read_text()
    # The fixtures are heavy, so only the guard part is kept.
    head = src.split("# ---------------------------------------------------------------------------\n# Fixtures")[0]
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
    assert "test_leakage.py" in r.stdout and "not collected" in r.stdout


def test_guard_fails_when_a_critical_module_is_not_collected(guarded):
    """A module skipped at its top level is not collected at all."""
    (guarded / "test_noise.py").write_text(MODULE_SKIP_STUB)
    r = _run(guarded)
    assert r.returncode != 0
    assert "not collected" in r.stdout


def test_guard_fails_when_every_test_in_a_module_is_skipped(guarded):
    """It was collected but 0 actually ran — a different branch."""
    (guarded / "test_noise.py").write_text(ALL_SKIPPED_STUB)
    r = _run(guarded)
    assert r.returncode != 0
    assert "were skipped" in r.stdout


def test_guard_can_be_bypassed_loudly(guarded):
    (guarded / "test_noise.py").write_text(ALL_SKIPPED_STUB)
    r = _run(guarded, env_extra={"KERNELRULE_ALLOW_SKIP": "1"})
    assert r.returncode == 0
    assert "[warning]" in r.stdout, "the bypass happened silently"


def test_guard_does_not_fire_on_filtered_runs(guarded):
    """The guard does not attach to a run filtered down with `-k`."""
    (guarded / "test_leakage.py").unlink()
    r = _run(guarded, "-k", "test_ok")
    assert r.returncode == 0, r.stdout[-2000:]
