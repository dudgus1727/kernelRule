"""★ Do the documents diverge (the 2026-09-03 review §4)?

If a number is in two places, knowing which one is representative means
reading both. A hand-written index diverges every time one D is added.
"""
def test_decisions_index_is_current():
    """★ The index is **a generated artefact**. If it diverges it fails
    (D-115).

    Writing it by hand makes it diverge every time one D is added — and a
    diverged index is on the "worse than none" side.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(root / "experiments/decisions_index.py"),
         "--check"], capture_output=True, text=True, cwd=root, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_canonical_numbers_live_in_one_place():
    """The representative numbers live in one place, `conclusion.md`
    (principle 2)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    concl = (root / "docs/artifacts/conclusion.md").read_text()
    assert "이 절이 **대표값이다**" in concl
    readme = (root / "README.md").read_text()
    for n in ("1.0650", "1.0737", "1.0762", "1.0797"):
        assert n not in readme, (
            f"the performance number {n} is in the README — the "
            f"representative values are in conclusion.md")


def test_runs_table_is_not_stale():
    """★ Has `runs.md` diverged from the run artefacts (D-128)?

    Writing the table by hand makes it diverge — it is kept as **a generated
    artefact**, the same way as `decisions_index.py`, and checked here.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "experiments/runs_table.py", "--check"],
                       cwd=root, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_trace_column_flags_missing_release(tmp_path, monkeypatch):
    """★ If there is a trace but no release, the table says so (D-138).

    The `트레이스` column of `runs.md` reads **only** the ledger
    (`trace-releases.json`) — calling `gh` would put the test on the network
    (principle 2).
    """
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "experiments"))
    import runs_table as rt

    run = tmp_path / "runs" / "ZZfake-p8-s0"
    run.mkdir(parents=True)
    (run / "trace.jsonl").write_text('{"ev": "run_start"}\n')
    monkeypatch.setattr(rt, "RUNS", tmp_path / "runs")

    man = tmp_path / "m.json"
    monkeypatch.setattr(rt, "TRACE_MANIFEST", man)
    assert rt._trace("ZZfake-p8", ["ZZfake-p8-s0"]) == rt.MISSING

    man.write_text(json.dumps({"ZZfake-p8": {"release": "trace-ZZ-abc1234"}}))
    assert rt._trace("ZZfake-p8", ["ZZfake-p8-s0"]) == "trace-ZZ-abc1234"

    # With no trace it is blank — not "★ 미업로드"
    (run / "trace.jsonl").unlink()
    assert rt._trace("ZZfake-p8", ["ZZfake-p8-s0"]) == ""


def test_live_tags_uses_mtime_only(tmp_path, monkeypatch):
    """★ 'running' is decided by mtime alone — it is not guessed (D-137)."""
    import os
    import sys
    import time
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "experiments"))
    import runs_table as rt

    d = tmp_path / "runs" / "ZZfake-p8-s0"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "rounds.jsonl").write_text("{}\n")
    monkeypatch.setattr(rt, "RUNS", tmp_path / "runs")
    assert "ZZfake-p8" in rt._live_tags()

    old = time.time() - rt.LIVE_SECONDS - 60
    os.utime(d / "rounds.jsonl", (old, old))
    assert "ZZfake-p8" not in rt._live_tags()
