

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


# ---------------------------------------------------------------------------
# ★ D-160 — the tertiles are drawn from the rules that passed the cut
# ---------------------------------------------------------------------------
def test_the_tertiles_are_drawn_from_the_passers():
    """★ The cut line and axis 1 used to be the same quantity read twice.

    `TOP_FRACTION` lets the best 30% in, and axis 1 was a tertile of
    **everything scored** — so anything that passed was automatically in
    band 0 and 18 of the 27 cells could not be reached. Measured on the
    D-156 run: all 25 accepted rules sat at axis-1 band 0.

    Checked by **behaviour**: with a wide spread of regrets, a rule that
    passes the cut must still be able to land in band 1 or 2.
    """
    from kernelrule.core.archive import Archive

    a = Archive()
    # ★ The bad rules come **first**, the way a run actually goes — a rule
    #   that arrives after the population has improved cannot pass the cut,
    #   so feeding them in increasing order would let only the first in and
    #   the test would prove nothing.
    regrets = [1.60 - i * 0.02 for i in range(30)]
    for i, g in enumerate(regrets):
        a.consider(_e(f"r{i}", g, code_len=10 + i, short=1.0 + (i % 5) * 0.1,
                      long=1.0))
    bands = {c[0] for c in a.cells}
    assert bands != {0}, (
        "every accepted rule is in axis-1 band 0 — the cut line and the "
        f"axis are the same quantity again (cells {sorted(a.cells)})")
    assert len(a._population()) < a.n_seen, (
        "the population must be the passers, not everything scored")


def test_the_population_is_never_empty():
    """The very first rule has nothing to be a quantile of — `_cut_line` is
    `inf` then, so it passes and the population is itself."""
    from kernelrule.core.archive import Archive

    a = Archive()
    a.consider(_e("only", 1.5))
    assert a.n_cells == 1
    assert len(a._population()) == 1
