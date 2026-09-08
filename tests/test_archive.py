

# ---------------------------------------------------------------------------
# D-101 — the acceptance criterion
# ---------------------------------------------------------------------------
def _e(rid, regret, rank_loss=float("nan"), code_len=10, short=1.0, long=1.0):
    from kernelrule.core.archive import Elite

    return Elite(rule_id=rid, code=f"# {rid}", w=[1.0], regret=regret,
                 mem_objective=short, comp_objective=long,
                 all_objective=regret, code_len=code_len,
                 round=0, rank_loss=rank_loss)


def test_archive_default_selects_by_regret():
    """★ The default is regret — every run so far is under that
    condition."""
    from kernelrule.core.archive import Archive

    a = Archive()
    assert a.select_by == "regret"
    a.consider(_e("r1", 1.10, rank_loss=0.9))
    a.consider(_e("r2", 1.05, rank_loss=9.9))   # better regret, worse rank
    assert a.best.rule_id == "r2"


def test_archive_rank_mode_selects_by_rank_loss():
    """With `select_by="rank"` it selects on **rank_loss** (the cell axes
    are unchanged)."""
    from kernelrule.core.archive import Archive

    a = Archive(select_by="rank")
    a.consider(_e("r1", 1.10, rank_loss=0.9))
    a.consider(_e("r2", 1.05, rank_loss=9.9))   # good regret, bad rank
    assert a.best.rule_id == "r1", "it is selecting on regret"


def test_archive_rank_mode_refuses_missing_rank_loss():
    """★ Without `rank_loss` it **does not silently fall back to regret**
    (§26.4)."""
    import pytest

    from kernelrule.core.archive import Archive

    a = Archive(select_by="rank")
    with pytest.raises(ValueError, match="rank_loss"):
        a.consider(_e("r1", 1.10))


def test_archive_refuses_unknown_select_by():
    import pytest

    from kernelrule.core.archive import Archive

    with pytest.raises(ValueError, match="acceptance criterion"):
        Archive(select_by="nope")
