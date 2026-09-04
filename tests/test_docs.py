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
