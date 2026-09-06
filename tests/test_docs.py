"""★ 문서가 달라지지 않는가 (2026-09-03 리뷰 §4).

수치가 두 곳에 있으면 어느 것이 대표값인지 다 읽어야 안다.
색인이 손으로 쓰인 것이면 D 하나 추가할 때마다 달라진다.
"""
def test_decisions_index_is_current():
    """★ 색인은 **생성물**이다. 달라지면 실패한다 (D-115).

    손으로 쓰면 D 하나 추가할 때마다 달라진다 — 그리고 달라진 색인은
    "없는 것보다 나쁘다" 쪽이다.
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
    """대표값 수치는 `conclusion.md` 한 곳이다 (원칙 2)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    concl = (root / "docs/artifacts/conclusion.md").read_text()
    assert "이 절이 **대표값이다**" in concl
    readme = (root / "README.md").read_text()
    for n in ("1.0650", "1.0737", "1.0762", "1.0797"):
        assert n not in readme, (
            f"README 에 성능 수치 {n} 이 있다 — 대표값은 conclusion.md 다")


def test_runs_table_is_not_stale():
    """★ `runs.md` 가 실행 산출물과 달라지지 않았는가 (D-128).

    표를 손으로 쓰면 달라진다 — `decisions_index.py` 와 같은 방식으로
    **생성물**로 두고 여기서 검사한다.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "experiments/runs_table.py", "--check"],
                       cwd=root, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_trace_column_flags_missing_release(tmp_path, monkeypatch):
    """★ 트레이스가 있는데 릴리즈가 없으면 표가 그렇게 적는다 (D-138).

    `runs.md` 의 `트레이스` 열은 대장(`trace-releases.json`) **하나만**
    읽는다 — `gh` 를 부르면 시험이 네트워크를 탄다 (원칙 2).
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

    # 트레이스가 없으면 빈칸이다 — "미업로드" 가 아니다
    (run / "trace.jsonl").unlink()
    assert rt._trace("ZZfake-p8", ["ZZfake-p8-s0"]) == ""


def test_live_tags_uses_mtime_only(tmp_path, monkeypatch):
    """★ '도는 중' 은 mtime 으로만 정한다 — 추측하지 않는다 (D-137)."""
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
