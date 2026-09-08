"""MockLLM (§24) — develop and debug the whole loop without API cost.

## Four modes

    canned        cycles through prepared rules   loop plumbing / archive /
                                                  scoring
    mutate        perturbs the parent's           evolution dynamics /
                  **structure**                   convergence curves
    adversarial   deliberately emits bad code     ★ the static checks and
                                                  the sandbox
    replay        replays a previous run's        deterministic reproduction
                  responses

## ★ `mutate` perturbs the **structure**, not the weights

§24.2 said "randomly perturb the parent rule's weights". That tests nothing —
the weights are fitted by `fit_weights` anyway (§29.3), so perturbing the
initial values barely affects the score.

**Adding, removing and swapping terms** is the unit of evolution. So this
mock adds, removes and swaps feature terms and attaches and detaches
shape-level branches. Only then can "does the pipeline find the missing term"
actually be tested.

## Determinism

Every mode is deterministic from the seed. That is required for `replay` to
hold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from kernelrule.agents.schemas import (
    Hypothesis,
    RuleProposal,
    SchemaViolation,
)

__all__ = ["LLMClient", "MockLLM", "LLMCall", "ADVERSARIAL_CASES",
           "CANNED_RULES"]


class LLMClient(Protocol):
    def complete(self, role: str, prompt: str, **kw) -> Any: ...


@dataclass
class LLMCall:
    role: str
    prompt_hash: str
    response: Any
    seq: int
    mode: str = ""


# ---------------------------------------------------------------------------
# canned — for checking the plumbing
# ---------------------------------------------------------------------------
CANNED_RULES: tuple[tuple[str, list[float]], ...] = (
    ("""def score(f, p, hw, w):
    return np.log2(f.traffic_amplification) * w[0]
""", [1.0]),
    ("""def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    return s + f.has_spill * w[1]
""", [1.0, 3.0]),
    ("""def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.has_spill * w[1]
    return s + f.is_two_stage * w[2]
""", [1.0, 3.0, 0.5]),
    ("""def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.has_spill * w[1]
    s = s + f.sm_idle_cost * w[2]
    return s + f.log_workspace_bytes * w[3]
""", [1.0, 3.0, 0.5, 0.1]),
    ("""def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.has_spill * w[1]
    s = s + f.smem_pressure * w[2]
    return s + f.split_k_cost * w[3]
""", [1.0, 3.0, 0.4, 0.3]),
)

# ---------------------------------------------------------------------------
# adversarial — ★ one of them passing means a hole in the defence (§24.3)
# ---------------------------------------------------------------------------
ADVERSARIAL_CASES: tuple[tuple[str, str, list[float]], ...] = (
    ("memorisation", """def score(f, p, hw, w):
    if p.M == 4096:
        return f.waves * w[0]
    return f.waves * w[0]
""", [1.0]),
    ("answer leak", """def score(f, p, hw, w):
    return f.waves * w[0] + time_ms
""", [1.0]),
    ("difficulty reference", """def score(f, p, hw, w):
    return f.waves * w[0] * difficulty
""", [1.0]),
    ("infinite loop", """def score(f, p, hw, w):
    while True:
        pass
    return f.waves * w[0]
""", [1.0]),
    ("sandbox escape", """def score(f, p, hw, w):
    import os
    os.system('echo PWNED')
    return f.waves * w[0]
""", [1.0]),
    ("typo", """def score(f, p, hw, w):
    return f.tail_wast * w[0]
""", [1.0]),
    ("if on an array", """def score(f, p, hw, w):
    if f.waves < 1:
        return f.waves * w[0]
    return f.waves * w[0]
""", [1.0]),
    ("too many literals", """def score(f, p, hw, w):
    return (f.waves*1.1 + f.tail_waste*2.2 + f.smem_pressure*3.3
            + f.has_spill*4.4 + f.edge_waste*5.5 + 6.6 + 7.7 + 8.8) * w[0]
""", [1.0]),
    ("non-determinism", """def score(f, p, hw, w):
    return np.random.rand(3) * w[0]
""", [1.0]),
    ("syntax error", """def score(f, p, hw, w)
    return 1
""", [1.0]),
    ("w slicing", """def score(f, p, hw, w):
    return f.waves * w[0] + sum(w[1:])
""", [1.0, 2.0]),
    ("dunder detour", """def score(f, p, hw, w):
    return f.waves * w[0] + score.__globals__['x']
""", [1.0]),
)


# ---------------------------------------------------------------------------
# mutate — ★ it perturbs the structure
# ---------------------------------------------------------------------------
_TEMPLATE_HEAD = "def score(f, p, hw, w):\n"


def _render_rule(terms: list[str], branch: tuple[str, str] | None) -> tuple:
    """A term list -> code + `w0`. The weights are `w[i]` in order."""
    lines, i = [], 0
    for t in terms:
        op = "s = " if i == 0 else "s = s + "
        lines.append(f"    {op}{t} * w[{i}]")
        i += 1
    if branch is not None:
        cond, term = branch
        lines.append(f"    if p.{cond}:")
        lines.append(f"        s = s + {term} * w[{i}]")
        i += 1
    lines.append("    return s")
    w0 = [1.0] * i
    return _TEMPLATE_HEAD + "\n".join(lines) + "\n", w0


def _parse_terms(code: str) -> tuple[list[str], tuple[str, str] | None]:
    """Reads back the code `_render_rule` produced. A mock-only parser."""
    terms, branch, cond = [], None, None
    for ln in code.split("\n"):
        t = ln.strip()
        if t.startswith("if p."):
            cond = t[5:].rstrip(":")
        elif t.startswith(("s = ", "s = s + ")) and " * w[" in t:
            expr = t.split(" * w[")[0]
            expr = expr.replace("s = s + ", "").replace("s = ", "")
            if cond is not None:
                branch = (cond, expr)
            else:
                terms.append(expr)
    return terms, branch


class MockLLM:
    """An LLM stand-in that runs without the API. **Deterministic.**"""

    def __init__(self, mode: str = "canned", *, seed: int = 0,
                 feature_names: list[str] | None = None,
                 shape_values: list[str] | None = None,
                 replay_dir: str | Path | None = None) -> None:
        if mode not in ("canned", "mutate", "adversarial", "replay"):
            raise ValueError(f"unknown mode: {mode!r}")
        self.mode = mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.features = list(feature_names or [])
        self.shape_values = list(shape_values or ["is_memory_bound"])
        self.calls: list[LLMCall] = []
        self._seq = 0
        self._n_features = 0
        self.replay_dir = Path(replay_dir) if replay_dir else None
        self._replay: list[LLMCall] = []
        if mode == "replay":
            self._load_replay()

    # -- Recording / replay -----------------------------------------------
    def _load_replay(self) -> None:
        if self.replay_dir is None or not self.replay_dir.exists():
            raise FileNotFoundError(
                f"replay mode, but {self.replay_dir} does not exist. It "
                f"does not silently fall back to canned (§26.4).")
        for f in sorted(self.replay_dir.glob("*.json")):
            d = json.loads(f.read_text())
            self._replay.append(LLMCall(role=d["role"],
                                        prompt_hash=d["prompt_hash"],
                                        response=d["response"],
                                        seq=d["seq"], mode=d.get("mode", "")))

    def dump(self, out: str | Path) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for c in self.calls:
            (out / f"{c.seq:05d}-{c.role}.json").write_text(json.dumps(
                {"role": c.role, "prompt_hash": c.prompt_hash, "seq": c.seq,
                 "mode": c.mode, "response": c.response},
                ensure_ascii=False, indent=1))

    # -- Entry point ------------------------------------------------------
    def complete(self, role: str, prompt: str, **kw) -> Any:
        h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        seq = self._seq
        self._seq += 1
        if self.mode == "replay":
            if seq >= len(self._replay):
                raise SchemaViolation(
                    f"the replay has no call #{seq} (only "
                    f"{len(self._replay)}). The loop changed — it does not "
                    f"silently make a new one.")
            rec = self._replay[seq]
            if rec.role != role:
                raise SchemaViolation(
                    f"replay mismatch: #{seq} is {rec.role!r} but {role!r} "
                    f"was requested")
            return rec.response
        resp = self._generate(role, prompt, **kw)
        self.calls.append(LLMCall(role=role, prompt_hash=h, response=resp,
                                  seq=seq, mode=self.mode))
        return resp

    def _generate(self, role: str, prompt: str, **kw) -> Any:
        if role == "analyze":
            return self._diagnose(prompt)
        if role == "rule_editor":
            return self._optimize(prompt, **kw)
        if role == "feature":
            return self._feature()
        if role == "rule_writer":
            return self._rule_writer(**kw)
        if role == "categorize":
            return {"categories": [
                {"name": f"mock_area_{i}",
                 "description": f"mock area {i}"}
                for i in range(5)],
                "notes": "areas partitioned by the mock"}
        raise ValueError(f"unknown role: {role!r}")

    # -- FeatureWriter / RuleWriter — ★ for checking the plumbing (§30.9) --
    #
    #   These two roles are needed to see whether the F0~F3 pipeline runs
    #   **all the way through** without a real LLM. The features the mock
    #   builds have no physical meaning — the purpose of `--dry-run` is
    #   plumbing, not performance.

    #: Feature templates built from raw values only. They use only names
    #: inside `RAW_FIELDS`.
    _FEATURE_FORMS = (
        ("mock_tile_area_ratio", "tile area / problem area",
         "float(cfg.tile_m * cfg.tile_n) / max(1.0, float(p.M) * p.N)"),
        ("mock_k_depth", "iteration depth along K",
         "float(p.K) / max(1.0, float(cfg.tile_k))"),
        ("mock_thread_load", "output elements per thread",
         "float(cfg.tile_m * cfg.tile_n) / max(1.0, float(cfg.threads))"),
        ("mock_smem_share", "shared-memory occupancy of the SM",
         "float(cfg.smem_bytes) / max(1.0, float(hw.smem_per_block))"),
        ("mock_grid_per_sm", "tiles per SM",
         ("float(p.M) * p.N / max(1.0, float(cfg.tile_m * cfg.tile_n))"
          " / max(1.0, float(hw.sm_count))")),
    )

    def _feature(self) -> dict:
        """It emits a **different** feature per proposal. Repeating the
        same one gets everything caught by the duplication check and the
        plumbing cannot be checked."""
        i = self._n_features % len(self._FEATURE_FORMS)
        self._n_features += 1
        name, doc, expr = self._FEATURE_FORMS[i]
        suffix = "" if self._n_features <= len(self._FEATURE_FORMS) else \
            f"_{self._n_features}"
        code = (f"def {name}{suffix}(p, hw, cfg) -> float:\n"
                f'    """{doc}."""\n'
                f"    return {expr}\n")
        return {"name": name + suffix, "code": code, "unit": "dimensionless",
                "direction": "higher_is_worse", "expected_range": [0.0, 1e6],
                "rationale": "a feature built by the mock — for checking "
                             "the plumbing"}

    def _rule_writer(self, **kw) -> dict:
        """A seed rule. It uses **only the feature names it was given**
        (under F1, the F1 features).

        When `self.features` is empty it raises rather than silently falling
        back to the human features — that is the path §30.9 blocks.
        """
        if not self.features:
            raise ValueError(
                "MockLLM(feature_names=...) is empty. There are no "
                "features to build a seed from — it does not silently fall "
                "back to the human 24 (§26.4).")
        n = min(4, len(self.features))
        pick = [self.features[int(i)] for i in
                self.rng.choice(len(self.features), size=n, replace=False)]
        code, w0 = _render_rule([f"f.{x}" for x in pick], None)
        return {"code": code, "w0": w0,
                "changes": "mock RuleWriter seed — picked from the given "
                           "features"}

    def _diagnose(self, prompt: str) -> dict:
        """Diagnosis — it reads the **unused features** out of the report
        and turns them into hypotheses.

        This is the only "intelligence" the mock has. The report emits an
        `★ unused` column, so it reads that. The purpose is to test
        **whether the loop plumbing carries that information**.
        """
        missing = []
        for ln in prompt.split("\n"):
            # ★ 2026-09-08 (D-146): the report became English. **The old
            #   form is read too** — this mock must still run on an old
            #   report.
            if "★ unused" in ln or "★ 미사용" in ln:
                name = ln.split()[0]
                if name in self.features and name not in missing:
                    missing.append(name)
        hyps = [Hypothesis(
            id=f"H{i}", claim=f"the rule does not use {n}. In the cases, "
                              f"its value differs greatly between the pick "
                              f"and the optimum",
            measurable_with=[n], proposed_direction=f"add a {n} term",
            risk="it could dilute the effect of the other terms")
            for i, n in enumerate(missing[:5])]
        # ★ Only the first hypothesis asks for an axis that does not exist
        #   (to check the D-75 path's plumbing). The loop reads it only when
        #   `max_new_features_per_round > 0`, so existing dry-runs are
        #   unaffected.
        if hyps:
            hyps[0].needs_new_feature = (
                "how much of one tile stays in L2 for the next tile to "
                "reuse")
        return {"hypotheses": [h.__dict__ for h in hyps]}

    def _optimize(self, prompt: str, *, parent: RuleProposal | None = None,
                  hypothesis: dict | None = None, **kw) -> dict:
        if self.mode == "adversarial":
            name, code, w0 = ADVERSARIAL_CASES[
                self._seq % len(ADVERSARIAL_CASES)]
            return {"code": code, "w0": w0, "changes": f"[adversarial] {name}"}
        if self.mode == "canned":
            code, w0 = CANNED_RULES[self._seq % len(CANNED_RULES)]
            return {"code": code, "w0": w0, "changes": "[canned]"}
        return self._mutate(parent, hypothesis)

    def _mutate(self, parent: RuleProposal | None,
                hypothesis: dict | None) -> dict:
        """★ It perturbs the structure, not the weights (see the module
        docstring)."""
        base = (parent.code if parent is not None
                else "def score(f, p, hw, w):\n"
                     "    s = f.traffic_amplification * w[0]\n    return s\n")
        terms, branch = _parse_terms(base)
        if not terms:
            terms = ["f.traffic_amplification"]
        pool = [f"f.{n}" for n in self.features]
        unused = [t for t in pool if t not in terms
                  and (branch is None or t != branch[1])]

        # If the hypothesis names a particular feature, add that first
        want = None
        if hypothesis:
            for n in hypothesis.get("measurable_with", []):
                if f"f.{n}" in unused:
                    want = f"f.{n}"
                    break

        r = self.rng.random()
        changes = ""
        if want is not None and r < 0.65:
            terms.append(want)
            changes = f"added the {want} term the hypothesis named"
        elif unused and r < 0.55:
            t = unused[int(self.rng.integers(len(unused)))]
            terms.append(t)
            changes = f"added the {t} term (random exploration)"
        elif len(terms) > 1 and r < 0.72:
            i = int(self.rng.integers(len(terms)))
            changes = f"removed the {terms[i]} term"
            terms.pop(i)
        elif unused and len(terms) > 1 and r < 0.86:
            i = int(self.rng.integers(len(terms)))
            t = unused[int(self.rng.integers(len(unused)))]
            changes = f"swapped {terms[i]} -> {t}"
            terms[i] = t
        elif branch is None and unused:
            cond = self.shape_values[int(self.rng.integers(
                len(self.shape_values)))]
            t = unused[int(self.rng.integers(len(unused)))]
            branch = (cond, t)
            changes = (f"added a shape-level branch: if p.{cond} -> "
                       f"reweight {t}")
        else:
            branch = None
            changes = "removed the shape-level branch"

        # Trim into the literal budget (8). Over it, the static checks
        # refuse.
        n_w = len(terms) + (1 if branch else 0)
        while n_w > 8 and len(terms) > 1:
            terms.pop()
            n_w = len(terms) + (1 if branch else 0)
        code, w0 = _render_rule(terms, branch)
        return {"code": code, "w0": w0, "changes": changes}
