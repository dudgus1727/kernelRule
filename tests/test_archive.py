

# ---------------------------------------------------------------------------
# D-101 — 채택 기준
# ---------------------------------------------------------------------------
def _e(rid, regret, rank_loss=float("nan"), code_len=10, short=1.0, long=1.0):
    from kernelrule.core.archive import Elite

    return Elite(rule_id=rid, code=f"# {rid}", w=[1.0], regret=regret,
                 short_regret=short, long_regret=long, code_len=code_len,
                 round=0, rank_loss=rank_loss)


def test_archive_default_selects_by_regret():
    """★ 기본은 regret 이다 — 지금까지의 모든 실행이 그 조건이다."""
    from kernelrule.core.archive import Archive

    a = Archive()
    assert a.select_by == "regret"
    a.consider(_e("r1", 1.10, rank_loss=0.9))
    a.consider(_e("r2", 1.05, rank_loss=9.9))   # regret 더 좋고 rank 더 나쁨
    assert a.best.rule_id == "r2"


def test_archive_rank_mode_selects_by_rank_loss():
    """`select_by="rank"` 면 **rank_loss** 로 고른다 (셀 축은 그대로)."""
    from kernelrule.core.archive import Archive

    a = Archive(select_by="rank")
    a.consider(_e("r1", 1.10, rank_loss=0.9))
    a.consider(_e("r2", 1.05, rank_loss=9.9))   # regret 은 좋지만 rank 나쁨
    assert a.best.rule_id == "r1", "regret 으로 고르고 있다"


def test_archive_rank_mode_refuses_missing_rank_loss():
    """★ `rank_loss` 가 없으면 **조용히 regret 으로 안 떨어진다** (§26.4)."""
    import pytest

    from kernelrule.core.archive import Archive

    a = Archive(select_by="rank")
    with pytest.raises(ValueError, match="rank_loss"):
        a.consider(_e("r1", 1.10))


def test_archive_refuses_unknown_select_by():
    import pytest

    from kernelrule.core.archive import Archive

    with pytest.raises(ValueError, match="채택 기준"):
        Archive(select_by="nope")
