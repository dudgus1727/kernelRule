"""★ D-193 — the pre-registered structure test (experiments/seed_structure)."""

from experiments.seed_structure import analyse

K1A_LIKE = """def score(f, p, hw, w):
    if p.M > p.N:
        d = f.dram_w1_hz_ratio
    else:
        d = f.dram_w1_id_ratio
    sm = np.maximum(f.tm_crit_ratio, f.tm_l2_ratio)
    t = np.sqrt(np.square(sm) + np.square(d))
    s = 10.0 * np.log2(t)
    s = s + w[0] * f.tm_regstaged + w[1] * f.tm_cta_warps
    return s
"""

PENALTIES = """def score(f, p, hw, w):
    s = f.tm_crit_ratio * w[0] + f.tm_l2_ratio * w[1]
    s = s + np.log2(f.tm_dram_ratio) * w[2]
    if p.roofline_ratio < 1:
        s = s + f.occupancy_deficit * w[3]
    return s
"""


def test_a_combined_time_is_a_main_term():
    r = analyse(K1A_LIKE)
    assert r["time_main_term"]
    assert {"tm_crit_ratio", "tm_l2_ratio", "dram_w1_hz_ratio"} <= set(
        r["main_term_paths"])
    assert r["n_weights"] == 2


def test_separately_weighted_paths_are_not():
    r = analyse(PENALTIES)
    assert not r["time_main_term"]
    assert r["branches"] == ["p.roofline_ratio < 1"]


def test_a_fitted_factor_inside_still_counts():
    code = ("def score(f, p, hw, w):\n"
            "    s = np.log2(np.sqrt(np.square(f.tm_crit_ratio)"
            " + np.square(w[0] * f.tm_l2_ratio)))\n    return s\n")
    r = analyse(code)
    assert r["time_main_term"] and r["combinations"][0]["weighted_inside"]
