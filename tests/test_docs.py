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
    assert "This section is **the canonical one**" in concl
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

    The `trace` column of `runs.md` reads **only** the ledger
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

    # With no trace it is blank — not "★ not uploaded"
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
    # ★ 2026-09-12 (D-168): a run directory is one whose `config.json`
    #   carries a `"loop"` block — `RoundLoop.dump()` writes it and a
    #   pipeline tag directory does not. The stub used to be `{}`.
    (d / "config.json").write_text('{"loop": {"run_id": "ZZfake-p8-s0"}}')
    (d / "rounds.jsonl").write_text("{}\n")
    monkeypatch.setattr(rt, "RUNS", tmp_path / "runs")
    assert "ZZfake-p8" in rt._live_tags()

    old = time.time() - rt.LIVE_SECONDS - 60
    os.utime(d / "rounds.jsonl", (old, old))
    assert "ZZfake-p8" not in rt._live_tags()


def test_a_pipeline_tag_directory_is_not_a_run(tmp_path, monkeypatch):
    """★ D-168 — the 21-run campaign's tags end in `-s<digit>`.

    `runs/c21-a6000-f0-s0/` is the **pipeline** directory (config, stage 1,
    stage 2) and `runs/c21-a6000-f0-s0-s0/` is the loop run. Both carry a
    `config.json` and both match the seed pattern, so the table builder
    took the tag directory for a run and died on its missing
    `rounds.jsonl`. Only the loop's config has a `"loop"` block.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "experiments"))
    import runs_table as rt

    runs = tmp_path / "runs"
    tag = runs / "ZZpipe-nk-s0"          # the pipeline tag directory
    tag.mkdir(parents=True)
    (tag / "config.json").write_text(
        '{"condition": "F2", "bundle": "datasets/x", "registry": {"n": 3}}')

    run = runs / "ZZpipe-nk-s0-s0"       # the loop run under it
    run.mkdir()
    (run / "config.json").write_text('{"loop": {"run_id": "ZZpipe-nk-s0-s0"}}')
    (run / "rounds.jsonl").write_text("{}\n")

    monkeypatch.setattr(rt, "RUNS", runs)
    groups = rt._groups()
    assert "ZZpipe-nk-s0" in groups, groups
    assert groups["ZZpipe-nk-s0"] == ["ZZpipe-nk-s0-s0"]
    assert "ZZpipe-nk" not in groups, (
        "the pipeline tag directory was counted as a run")


def test_the_table_column_comes_from_the_recorded_gpu_name(tmp_path,
                                                           monkeypatch):
    """★ D-168 — the GPU column must not default to a real table name.

    It used to be a `hw_text.sha256` lookup with `a6000` as the fallback.
    D-166 §E changed the hardware prompt, so every sha256 changed, the
    lookup missed on all four tables and 9 campaign runs were filed under
    `a6000`. A wrong table name is indistinguishable from a right one.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "experiments"))
    import runs_table as rt

    def cfg(name):
        return {"llm": {"hw_text": {"gpu": name, "sha256": "deadbeef"}}}

    assert rt._gpu_of(cfg("NVIDIA H100 NVL (sm_90)"), "deadbeef") == "h100"
    assert rt._gpu_of(cfg("NVIDIA GeForce RTX 4090 (sm_89)"), "x") == "4090"
    assert rt._gpu_of(cfg("NVIDIA RTX A6000 (sm_86)"), "x") == "a6000"
    # ★ an unknown GPU shows its own name — it is not silently a6000
    assert rt._gpu_of(cfg("NVIDIA B200"), "x") == "NVIDIA B200"
    # runs from before `hw_text.gpu` still go through the sha map
    assert rt._gpu_of({}, "37762692c36f5ba4") == "4090"
    assert rt._gpu_of({}, "None") == "?"
