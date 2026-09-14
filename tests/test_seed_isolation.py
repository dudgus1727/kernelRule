"""★ D-172 §X — an axis one seed builds must not reach the next seed.

`stage3()` used to take `matrix` and `reg` as arguments and share them
across its seed loop. `RoundLoop._write_features` registers a new axis into
`self.matrix.registry`, and nothing put it back, so seed 1 started with
seed 0's axes in the library.

⛔ The stage-1 library **is** shared on purpose (one library per campaign).
What must not carry over is only what the loop built.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _leak_report(run_dir_of, base_names: set[str], seeds=(0, 1, 2)) -> dict:
    """★ The audit's own comparison, as a function (D-172 §Y-3).

    `features.jsonl` says what each seed built; `archive.jsonl` says what
    its rules reference. **The registry alone is not enough** — "not in the
    registry but used by a rule" is the case a registry check misses, and
    it is the one that actually happened.
    """
    made: dict[int, set[str]] = {}
    for s in seeds:
        p = run_dir_of(s) / "features.jsonl"
        made[s] = {json.loads(x)["name"]
                   for x in p.read_text().splitlines() if x.strip()} \
            if p.exists() else set()
    out: dict[int, set[str]] = {}
    for s in seeds[1:]:
        earlier = set().union(*[made[e] for e in seeds if e < s]) \
            if any(e < s for e in seeds) else set()
        arc = run_dir_of(s) / "archive.jsonl"
        if not arc.exists():
            continue
        refs: set[str] = set()
        for x in arc.read_text().splitlines():
            if not x.strip():
                continue
            refs |= {m.group(1) for m in
                     re.finditer(r"\b[fp]\.(\w+)", json.loads(x)["code"])}
        leak = refs & (earlier - base_names - made[s])
        if leak:
            out[s] = leak
    return out


@pytest.mark.needs_bundle
def test_the_recorded_48_run_campaign_leaked_across_seeds():
    """⛔ **This is the sabotage check** (D-172 §Y-4).

    It asserts the defect **is** present in the 48-run campaign's recorded
    artefacts. If a later change made this pass, the detector stopped
    detecting — and the fix below would be verified by a test that cannot
    fail.

    ★ The campaign's numbers are not corrected by the fix; they were
    produced by the leaking code and stay as they are, with the reservation
    recorded in D-172 §Z.
    """
    from kernelrule.features.known7 import KNOWN7

    props = ROOT / "runs/k7-1/stage1-features/proposals.jsonl"
    if not props.exists():
        pytest.skip("runs/ is .gitignore'd — the campaign artefacts are not "
                    "in this checkout")
    base = set(KNOWN7._items) | {
        json.loads(x)["name"] for x in props.read_text().splitlines()
        if x.strip() and json.loads(x).get("accepted")}
    hits = 0
    combos = 0
    for gpu in ("a6000", "5090", "4090", "h100"):
        for fold in range(4):
            d = ROOT / f"runs/nk4-{gpu}-f{fold}-s0"
            if not d.exists():
                continue
            combos += 1
            rep = _leak_report(
                lambda s, g=gpu, f=fold: ROOT / f"runs/nk4-{g}-f{f}-s{s}",
                base)
            hits += len(rep)
    if combos == 0:
        pytest.skip("the nk4 campaign runs are not in this checkout")
    assert hits > 0, (
        "the leak detector found nothing in the 48-run campaign, whose "
        "artefacts are known to carry it (32 of 32 combinations, 270 "
        "rules). The detector is broken, not the campaign.")


def test_each_seed_starts_from_the_stage1_registry(synth_table, tmp_path,
                                                   monkeypatch):
    """★ Y-1 · Y-2 · Y-3 on a synthetic table — 2 seeds, 2 rounds, MockLLM.

    It runs `stage3` itself rather than imitating it, so the guard sits on
    the code the campaign calls.
    """
    import types

    from experiments import f1_pipeline as pipe
    from kernelrule.core.splits import Split, SplitSet
    from kernelrule.features import Feature, FeatureRegistry

    base = FeatureRegistry("base")
    for n in ("tail_waste", "edge_waste"):
        from kernelrule.features import REGISTRY
        base.add(REGISTRY[n])

    sizes: list[int] = []
    names: list[set] = []

    class _FakeLoop:
        """Stands in for `RoundLoop` — it records the registry it was handed
        and then **adds an axis**, the way `_write_features` does."""

        def __init__(self, *, cfg, table, matrix, splits, llm):
            self.matrix = matrix
            sizes.append(len(matrix.registry._items))
            names.append(set(matrix.registry._items))

        def seed(self, code, w0, *, changes=""):
            pass

        def run(self, n):
            r = self.matrix.registry
            nm = f"leaked_axis_{len(sizes)}"
            r.add(Feature(name=nm, fn=lambda p, hw, c: 1.0, unit="x",
                          expected_range=(0.0, 1.0), direction="neutral",
                          code_hash="x"))

    monkeypatch.setattr(pipe, "RoundLoop", _FakeLoop)
    monkeypatch.setattr(pipe, "_make_llm", lambda *a, **k: object())
    monkeypatch.setattr(pipe, "_load_stage1",
                        lambda d, b, c, t: _stage1_like(b))

    sh = synth_table.shapes()
    splits = SplitSet(train=Split("train", tuple(sh[:-2])),
                      val=Split("val", tuple(sh[-2:])), kind="t")
    a = types.SimpleNamespace(
        n_seeds=2, rounds=2, seed=0, max_new_features=1, condition="F2",
        no_analyst=True, workers=1, parameters=None, hypothesis_pool=[],
        model="mock", dry_run=True)
    pipe.stage3(a, tmp_path, synth_table, base, splits,
                {"code": "def score(f, p, hw, w):\n    return f.tail_waste",
                 "w0": [1.0]})

    # ★ Y-1 — both seeds start at the same size
    assert sizes[0] == sizes[1], sizes
    # ★ Y-2 — seed 1's registry does not hold what seed 0 built
    assert not any(n.startswith("leaked_axis") for n in names[1]), names[1]


def _stage1_like(base):
    """The stage-1 state: the condition's base plus what stage 1 accepted.
    Here the base alone — the point is that it is **rebuilt**, not carried."""
    from kernelrule.features import FeatureRegistry

    r = FeatureRegistry("stage1")
    for n in sorted(base._items):
        r.add(base[n])
    return r
