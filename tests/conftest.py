"""Shared fixtures + the **skip watchdog** (§26.3). It uses no GPU at all.

The structure was taken as validated from kernelTab's `tests/conftest.py`
(R-1). There, 41 tests in `test_table.py`/`test_bundle.py` were skipped
wholesale because pyarrow was missing, and the summary was a green
"2 skipped". Those two happened to be **the modules that validate the
answer-leak defences**.

The equivalent here is `CRITICAL_MODULES`. If those modules are not collected
or are skipped entirely, **the session fails.**

The bypass is `KERNELRULE_ALLOW_SKIP=1`. Bypassing prints a loud warning, and
the result of that run must not be used to guarantee the leak defences.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

#: Warns when a critical module is skipped above this fraction.
#:
#: kernelTab's R-1 caught **a whole module** being skipped. But one layer
#: below remained — the condition was `ran == 0`, so 4 running and 11
#: skipped simply passed. `datasets/` is `.gitignore`d, so **for someone who
#: has just cloned, that state is the default** and there was no signal at
#: all.
#:
#: Measured: without a bundle, `test_leakage.py` is 4 passed / 11 skipped
#: (73% skipped). A quarter of the leak-defence validation runs and the light
#: is green.
MAX_SKIP_FRAC = 0.5

#: If these modules do not run, it is a **failure**.
CRITICAL_MODULES = {
    "test_leakage.py", "test_scoring.py", "test_adapter.py",
    "test_noise.py", "test_weights.py", "test_synth.py",
    # Added in stage 2 — it stops the static checks and the sandbox
    # silently not running
    "test_checks.py", "test_sandbox.py", "test_features.py",
    "test_baselines.py",
    # Added in stage 3 — the report self-contradiction check and the
    # adversarial blocking are the core
    "test_diagnostic.py", "test_agents.py", "test_loop.py",
    # Added in stage 4 — the no-fallback-on-missing-key rule and the budget
    # caps must not switch off silently
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
    """What **must fail** the session."""
    bad = []
    for mod in sorted(CRITICAL_MODULES):
        d = _seen.get(mod)
        if d is None or (d["ran"] == 0 and d["skipped"] == 0):
            bad.append(f"{mod}: not collected (an import failure, or the "
                       f"file is missing)")
        elif d["ran"] == 0:
            bad.append(f"{mod}: all {d['skipped']} were skipped — 0 actually "
                       f"ran")
    return bad


def _partial_skips() -> list[str]:
    """Not a failure, but **not something that can be guaranteed** (one
    layer below §26.3).

    Why it does not refuse: developing without a bundle is a normal path. It
    is only that the leak defences must not be guaranteed from such a run,
    and that fact has to be **visible**.
    """
    out = []
    for mod in sorted(CRITICAL_MODULES):
        d = _seen.get(mod)
        if not d:
            continue
        total = d["ran"] + d["skipped"]
        if total and d["ran"] and d["skipped"] / total > MAX_SKIP_FRAC:
            out.append(f"{mod}: {d['skipped']}/{total} "
                       f"({d['skipped'] / total:.0%}) skipped")
    return out


class _SkipGuardItem(pytest.Item):
    """A synthetic item that runs last.

    Changing `exitstatus` in `pytest_sessionfinish` does not propagate on
    every pytest version. A watchdog that does not reach the exit code is
    meaningless in CI, so it is made a real test item and takes the normal
    failure path.
    """

    def runtest(self):
        # ★ Partial skips are reported first. It is not a failure, but
        #   passing over it silently makes "the leak defences are validated"
        #   a false belief.
        partial = _partial_skips()
        if partial:
            self._warn_loudly(
                "a critical module was skipped more than half\n"
                + "\n".join("  - " + x for x in partial) + "\n\n"
                "★ Do not guarantee the leak defences from this run.\n"
                "  Usually it is because there is no bundle under "
                "`datasets/`. How to get one:\n"
                "    see the data section of docs/design.md\n"
                "  Developing without a bundle is normal, but then only the "
                "synthetic\n"
                "  table has been validated.")

        bad = _bad_modules()
        if not bad:
            return
        msg = ("a critical test module did not actually run\n"
               + "\n".join("  - " + b for b in bad) + "\n\n"
               "These modules validate **the answer-leak defences and the "
               "noise floor**.\n"
               "A green light while they are skipped makes 'validated' a "
               "false belief.\n\n"
               "  How to fix:              pip install -e '.[test]'\n"
               f"  To skip deliberately:    {ALLOW_SKIP_ENV}=1 pytest")
        if os.environ.get(ALLOW_SKIP_ENV) == "1":
            self._warn_loudly(msg)
            pytest.skip(f"bypassed with {ALLOW_SKIP_ENV}=1 — do not "
                        "guarantee the leak defences from this run")
        raise AssertionError(msg)

    def _warn_loudly(self, msg: str) -> None:
        """Writes the bypass warning so that it is **certainly visible**.

        It was wrong twice.

        1. `print` — the captured output of a skipped item is not shown.
        2. `terminalreporter.write_line` — on some pytest versions it is
           **swallowed by the global capture** inside `runtest`. It passed in
           this environment (9.1.1) and failed on another version. **A
           watchdog's guarantee must not depend on the pytest version.**

        The capture is disabled explicitly through `capturemanager`.
        """
        lines = ("[warning] "
                 + msg.replace("\n", "\n[warning] ")).split("\n")
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
        # It is recorded in the skip reason too — even if the terminal
        # output is invisible for any reason, it shows in the `-rs` summary.
        # One more layer.
        self.user_properties.append(("allow_skip_bypass", msg[:200]))

    def repr_failure(self, excinfo, style=None):
        return str(excinfo.value)

    def reportinfo(self):
        return self.path, 0, "the skip watchdog (§26.3)"


def _config_filtered(config) -> bool:
    """Is this a run that selected **only part**? Only then is the watchdog
    detached.

    Two things were actually stepped on.

    1. Sweeping `invocation_params.args` directly mistakes **option values**
       such as the X of `--deselect X` for positional arguments (in
       kernelTab that bug disarmed the watchdog entirely). What pytest has
       already parsed is used instead.
    2. "a positional argument means a filter" switches the watchdog off even
       on **runs that cover everything**, such as `pytest /repo` /
       `pytest $(pwd)`. A meta test caught this. So **naming a directory is
       not a filter** — only naming a file counts.
    """
    opt = config.option
    if getattr(opt, "keyword", "") or getattr(opt, "markexpr", ""):
        return True
    if getattr(opt, "deselect", None):
        return True
    for raw in getattr(opt, "file_or_dir", []) or []:
        # A "path::TestClass::test_x" form names a file too.
        path = Path(str(raw).split("::")[0])
        if path.is_file() or path.suffix == ".py":
            return True
    return False


def pytest_report_header(config):
    """**Always shows whether a bundle is present in the header.** Without
    one it says what will not run."""
    if _have_real_bundle():
        return f"kernelRule: real bundle present ({REAL_BUNDLE.name})"
    return ("kernelRule: ⚠️ no real bundle — much of the contract/leak "
            "validation is skipped. Do not guarantee the leak defences from "
            "this run")


def pytest_collection_modifyitems(session, config, items):
    for it in items:
        _seen.setdefault(Path(str(it.fspath)).name, {"ran": 0, "skipped": 0})
    if _config_filtered(config):
        return
    items.append(_SkipGuardItem.from_parent(
        session, name="test_critical_modules_actually_ran"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def a6000_hardware():
    """The **effective** specs of the A6000 with clocks locked at
    1350/7601 MHz (§6.2)."""
    from kernelrule.core.types import Hardware
    return Hardware(name="NVIDIA RTX A6000", arch="sm_86", sm_count=84,
                    smem_per_block=101376, max_threads_per_sm=1536,
                    regs_per_sm=65536, peak_tflops_f16=116.1,
                    bandwidth_gbps=729.7, l2_bytes=6291456)


@pytest.fixture(scope="session")
def hw_a6000():
    return a6000_hardware()


def a6000_hw_text() -> str:
    """★ The hardware-facts prompt, **generated** (D-113).

    `hw/sm_86.md`, the frozen file the prompt tests used to load, was deleted
    on 2026-09-08 (D-146). The text is built here the way the pipeline builds
    it, so the tests need no bundle and nothing gets skipped.

    `min_ms` is the A6000's shortest kernel (11.3 us). It is a test value —
    the real path reads it from the table (D-117).
    """
    from kernelrule.agents.hwprompt import render_hw_prompt
    from kernelrule.core.noise import NoiseModel

    return render_hw_prompt(a6000_hardware(),
                            noise=NoiseModel.a6000_reference(), env={},
                            min_ms=0.0113)


@pytest.fixture(scope="session")
def hw_other():
    """A fictitious GPU. For detecting hardcoded hardware constants (§8.3
    item 6).

    ★ **Every numeric field must differ.** If even one matches the A6000, a
    feature using that field simply passes the scale check — leaving
    `regs_per_sm` equal once made `reg_pressure` wrongly flagged as "it does
    not use hw".
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
    """The real bundle. Without it, skip — but **say so** (§23.4)."""
    if not _have_real_bundle():
        pytest.skip(f"no real bundle: {REAL_BUNDLE} — the contract "
                    f"validation was skipped")
    return REAL_BUNDLE


@pytest.fixture(scope="session")
def tiny_grid():
    """A small grid. From the real bundle if there is one, otherwise by
    enumeration.

    ★ Both paths use `load_for_ranking` only — they cannot see the measured
    times.
    """
    from kernelrule.tools.synth import Grid
    shapes = [(1, 4096, 4096), (128, 4096, 4096), (1024, 4096, 4096),
              (4096, 4096, 4096), (512, 512, 512), (1024, 4096, 512),
              # ★ An alignment edge case. Without it `can_use_cp_async`
              #   becomes a constant and is rejected as "zero explanatory
              #   power" — that is the grid being narrow, not a problem with
              #   the feature. At least one layer-D shape is always
              #   included.
              (1024, 4096, 4097), (1024, 4098, 4096)]
    if _have_real_bundle():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Grid.from_bundle(REAL_BUNDLE, env_hash=REAL_ENV_HASH,
                                    shapes=shapes,
                                    max_configs_per_shape=400, seed=7)
    pytest.skip("no real bundle — the enumeration-grid path is tested "
                "separately in test_synth.py")


@pytest.fixture(scope="session")
def synth_bundles(tmp_path_factory, tiny_grid):
    """A small synthetic bundle per preset. Reused across the session."""
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
