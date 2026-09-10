"""The round loop (§14) — the code controls the order.

**The agents do not talk to each other** (§11.1). Making it a true
multi-agent system would make it impossible to trace which stage broke.

## One round (§14.1, appendix ★fix 1)

    1. score the archive's best rule -> build the diagnostic report  [code]
    2. 3~5 hypotheses                                               [1 LLM call]
    3. generate features if needed + review + auto-validation        [0~4 calls]
    4. generate 12 rules (parent/hypothesis combinations)            [12 calls]
    5. static checks -> sandbox -> ★ weight optimisation -> scoring  [code]
    6. update the archive
    7. record the hypothesis history / the failure list
    8. check the stopping condition

**Step 5's weight optimisation coming before scoring is the crux** (§29.3).
Without it a good structure is thrown away because of bad initial values, and
evolution selects not structure but **luck in the weights**.

## Stopping condition (§14.3)

    10 consecutive rounds of improvement below the noise floor
      AND
    no new cell filled in the archive

If the score has stalled but diversity is still growing, it is still
exploring. **Both conditions** are used. Early stopping is judged on the
**validation split** — judged on the training regret it keeps running long
after overfitting has begun (§10.2).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kernelrule.agents.schemas import SchemaViolation, validate_rule_proposal
from kernelrule.core.archive import Archive, Elite
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import SandboxError, compile_rule, run_isolated
from kernelrule.core.scoring import evaluate_scores
from kernelrule.core.splits import SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import FitError, fit_weights, make_score_of
from kernelrule.report.diagnostic import build_report
from kernelrule.rules.checks import FITTER_SWITCH_DIM as _FITTER_DIM
from kernelrule.rules.checks import (
    check_rule,
    fitter_for,
    limits_for,
    weight_bounds,
)

__all__ = ["RoundLoop", "RoundResult", "LoopConfig", "LLMUnreachable"]


@dataclass
class LoopConfig:
    run_id: str
    #: ★ 2026-09-08 (D-144): 12 -> 6. `Archive.parents`'s ratio makes it
    #: exploit 3 / explore 2 / cross 1.
    n_rules_per_round: int = 6
    max_rounds: int = 20
    max_evals: int = 200
    #: ★ The execution trace (D-133). It records events into `trace.jsonl`
    #: in time order.
    #: **It is not a condition** — it only logs and does not touch the
    #: compute path. That is why it is not in `runset.KEYS`
    #: (`tests/test_trace.py` verifies it).
    trace: bool = True
    #: ★ Early stopping is **off** (D-132). At `0` it runs to the round
    #: cap.
    #:
    #: Why it was turned off — the problem was not the value but the
    #: **criterion**:
    #: ```
    #: even when the loop's internal score (best_val_regret) is flat,
    #: ★ the final scoring (per-regime refit -> holdout) keeps improving
    #: -> stopping on the internal score loses final performance in
    #:    principle (D-131: +0.0187)
    #: raising patience to 5 only shrinks the size; the problem is the same
    #: ```
    #: ★ It must stay 0. Above 0 `should_stop` **raises** — the early-stop
    #: path was sealed at D-144 because it read the validation split. The
    #: field is kept so that turning it on fails loudly instead of silently.
    patience: int = 0
    seed: int = 0
    sandbox_first_seen: bool = True
    out_dir: str = "runs"
    #: ★ The Analyst -> FeatureWriter path (D-75). The cap on how many new
    #: axes may be created per round. **At 0 the path does not exist.**
    #:
    #: ⚠️ 2026-09-10 (D-160): the default was **0** and every campaign so far
    #: ran with the path shut. Under F3 nobody noticed — the Analyst asked
    #: for a new axis 0 times in 36 hypotheses, because `log_flops` and
    #: `log_min_dim` were already there. Under F1 it asked **30 times in 36**
    #: and every one of them was dropped before the FeatureWriter (D-159).
    #:
    #: ★ 3 per round, **the same for F1/F2/F3** — a condition that differs
    #: between arms cannot be compared later. There is no cap on the total,
    #: nor on how many one feature may use: an axis nobody uses does not
    #: enter the archive and dies there.
    #:
    #: The per-round cap exists because of §21 — the feature matrix
    #: recomputes every shape for each new axis and the cache key is the
    #: registry hash, so a registry that changes every round never hits the
    #: cache. 12 rounds x 3 is up to 36 more axes.
    max_new_features_per_round: int = 3
    #: The FeatureWriter condition (F1/F2/F3). It must be given **the same
    #: condition** as stage 1 outside the loop — otherwise the condition
    #: changes inside a round.
    feature_condition: str = "F3"
    #: ★ §16.1's control arm C (D-91) — the Analyst is off, but
    #: **hypotheses from another run and another round** go in its place.
    #: Paths to `hypotheses.jsonl` files.
    #:
    #: What it separates:
    #:   close to B  a hypothesis **need not fit the current state**
    #:               = the diagnostic report's contribution is diversity
    #:                 injection
    #:   close to A  ★ it must fit this round's diagnosis (the strong form
    #:               of §16.1)
    #:
    #: The Analyst is not called, so **the call count matches A** — the cost
    #: comparison is clean.
    hypothesis_pool: tuple[str, ...] = ()
    #: ★ The weight objective (D-101). Default `"regret"` — **the
    #: behaviour so far**.
    #: With `"rank"` it fits on the weighted pairwise loss within the true
    #: top `rank_top_k`, and **archive acceptance uses that too** (the cell
    #: axes do not change).
    #: `regret` keeps being computed and recorded even then (experiment plan
    #: §4 — not used for the verdict, used when placing two evolutions side
    #: by side).
    #: ★ 2026-09-04 (D-128): the default is `"regret"` again, and
    #: **`"rank"` is refused** — the rank loss was concluded to be the wrong
    #: objective (D-118 · D-121).
    #: The field itself stays: old `config.json` files must still be read.
    #: The rank loss and tau are still used **as metrics**
    #: (`weights.rank_loss` and the like).
    objective: str = "regret"
    rank_top_k: int = 100
    #: ★ The sum of two losses (D-109).
    #: `L = rank_loss(k) + lambda * rank_loss_top1(k)`
    #: `rank_loss_top1` covers **only pairs involving the true first place**
    #: — a smooth surrogate for `regret`. `regret` itself is not put in (it
    #: is a step function, so L-BFGS cannot be used).
    rank_lambda: float = 0.0
    #: ★ The fitter (D-123). Default `"nelder-mead"` — **every run so
    #: far**.
    #: `"cma"` is the arm §2's gate chose (100% reach rate in 16 dimensions,
    #: needs the `cma` package).
    #: ★ It is a condition, so it stays in `config.json` and the bundle
    #: check looks at it (principle 39).
    fit_method: str = "nelder-mead"
    #: The number of fit restarts. It is 1 for CMA-ES — splitting the
    #: budget four ways leaves only six generations, which is not CMA
    #: (`fitter-regret-prereg.md` §2).
    fit_restarts: int = 4
    #: ★ The two-order experiment (D-104). `"rank->regret"` or
    #: `"regret->rank"`.
    #: With `None` the objective does not change — the behaviour so far.
    #:
    #: **The switch point is not a hyperparameter but a stopping condition**
    #: — it switches when the improvement over the last `switch_window`
    #: rounds is below `switch_min_improve`. Rationale: s2 was flat from r5
    #: to r9, 0.2945 -> 0.2915 (1.0%).
    #: ⚠️ Was the parameter cap (D-104). **There is no cap** (D-150/152).
    #: Recorded in `config.json`; it refuses nothing.
    parameters: int | None = None
    objective_switch: str | None = None
    switch_min_improve: float = 0.01
    switch_window: int = 3
    #: ★ Parallel scoring and fitting (D-95). At 0 it is sequential — **the
    #: behaviour so far**.
    #: The results must be identical (`test_parallel_matches_sequential`).
    n_workers: int = 0
    #: ★ The §16.1 ablation — with the Analyst off there is neither a
    #: diagnostic report nor hypotheses. The RuleEditor edits from the
    #: parent rule and the feature list alone.
    #: **The default is on** (as in every run so far).
    use_analyst: bool = True


class LLMUnreachable(RuntimeError):
    """The LLM could not be reached. **Our problem, not a failure of the
    model** (D-43)."""


#: Names that identify a failure to reach the LLM. They are credit,
#: authentication or network problems, not failures of the model (D-43).
_TRANSPORT_HINTS = ("HTTPError", "APIError", "APIConnection", "APIStatus",
                    "Timeout", "RateLimit", "Authentication", "Permission",
                    "ConnectError", "ReadError", "ServiceUnavailable")
#: Finding one of these in the body settles it.
_TRANSPORT_BODY = ("no credits", "insufficient_quota", "invalid_api_key",
                   "401", "402", "403", "429", "500", "502", "503")


def _is_transport_error(exc: BaseException) -> bool:
    """Was the LLM **unreachable**, or did the model fail to match the
    schema?"""
    name = type(exc).__name__
    if any(h in name for h in _TRANSPORT_HINTS):
        return True
    text = str(exc).lower()
    return any(h in text for h in _TRANSPORT_BODY)


#: If **every** proposal of a round is a transport failure, it stops. A
#: credit or authentication problem does not heal by itself, and burning the
#: remaining rounds leaves nothing but an empty archive.
#: 12 rounds x 48 seconds really were spent that way.
STOP_ON_TOTAL_LLM_FAILURE = True


#: A validation gap larger than this counts as a **regime transfer
#: failure**.
#: It differs from overfitting — even a rule with only 3 terms exceeds this
#: value (+4.99 measured).
VAL_GAP_ALARM = 0.5


#: ★ What a worker sees (D-95). A child created by `fork` **inherits the
#: parent's memory as it is** — the table + feature matrix is 4.2GB, so
#: reloading per worker would be 50GB across 12. With copy-on-write it
#: barely grows in practice.
#:
#: ⚠️ It must be filled **before** the pool is created. The snapshot at fork
#: time is all there is.
_WORKER: dict = {}


def _fit_and_score(job: tuple) -> dict:
    """★ Fits and scores one candidate. **It runs in a worker** (D-95).

    96% of a round's wall clock is `fit_weights` (5.4 s x 12 candidates).
    Because of the GIL it cannot be threads; it has to be processes.

    ⚠️ **The static checks and the sandbox are done by the parent.** Doing
    them here would have the worker start yet another process
    (`run_isolated`), a nested spawn. Together those two are 3%.

    What comes back is **pure data** — the ids and order of the `Elite`s are
    decided by the parent (determinism).
    """
    idx, code, w0 = job
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.scoring import evaluate_scores
    from kernelrule.core.weights import fit_weights, make_score_of

    c = _WORKER
    try:
        fn = compile_rule(code)
        # ★ The fitter is decided by **this rule's len(W0)** (D-144). With
        #   a per-path budget the dimension differs per rule — one campaign
        #   setting cannot decide it.
        _ft = fitter_for(len(w0))
        fr = fit_weights(fn, c["matrix"], c["table"], c["train"], w0,
                         max_evals=_ft["max_evals"], val_split=c["val"],
                         objective=c.get("objective", "regret"),
                         rank_top_k=c.get("rank_top_k", 100),
                         rank_lambda=c.get("rank_lambda", 0.0),
                         method=_ft["fit_method"],
                         n_restarts=_ft["fit_restarts"],
                         bounds=weight_bounds(code, len(w0)))
    except (FitError, SchemaViolation) as e:
        return {"i": idx, "err": ("fit", str(e)[:90])}
    except Exception as e:                                  # noqa: BLE001
        return {"i": idx, "err": ("run", f"{type(e).__name__}: {e}"[:90])}
    ev = evaluate_scores(make_score_of(fn, c["matrix"], fr.w), c["table"],
                         list(c["train"].shapes), ks=(1, 3))
    return {"i": idx, "w": [float(x) for x in fr.w],
            "regret": fr.fit_regret, "moved": bool(fr.moved),
            "rank_loss": _rank_loss_of(fn, c, fr.w),
            "val_regret": fr.val_regret,
            "mem": ev.at(1, mask=c["short_mask"]),
            "comp": ev.at(1, mask=c["long_mask"]),
            "all": ev.at(1)}


def _rank_loss_of(fn, c: dict, w) -> float:
    """★ The rank loss used for acceptance. Under `objective="regret"` it
    leaves NaN.

    It is not forced under the `regret` condition — that would add the cost
    of building the pairs to every existing run.
    """
    import math

    if c.get("objective") != "rank":
        return float("nan")
    from kernelrule.core.weights import _Problem

    k = c.get("rank_top_k", 100)
    lam = float(c.get("rank_lambda", 0.0) or 0.0)
    pr = _Problem(c["matrix"], c["table"], c["train"].shapes, 1)
    pr.build_pairs(c["table"], k)
    v = pr.rank_loss(fn, w)
    # ★ Acceptance must use **the same objective as the fit** (principle
    #   37). Selecting without lambda picks by something other than what was
    #   optimised.
    if lam and math.isfinite(v):
        pr.build_top1_pairs(c["table"], k)
        v += lam * pr.rank_loss_top1(fn, w)
    return float(v) if math.isfinite(v) else float("nan")


def _requirement_of(h: dict) -> str:
    """The sentence naming the physical quantity a hypothesis asked for.
    **Both names are read.**

    The field is `needs_new_feature`. On 2026-08-28 it was briefly renamed
    to `physical_requirement`, three runs were made, and then it was
    **reverted** (D-81) — because the baseline was measured under the old
    name. The runs made in between must not silently become 0, so both are
    read.
    """
    v = h.get("physical_requirement") or h.get("needs_new_feature")
    return str(v).strip() if v else ""


def _feature_task(text: str) -> str:
    """**Everything** that goes to the FeatureWriter. ★ The diagnostic
    report does not (D-75).

    No case numbers, no scores, no shape list. Only the one sentence of
    physical requirement — it blocks the channel by which a feature built
    inside the loop could be fitted to the training shapes.
    """
    return ("## What to build now\n\n"
            "Build one feature that measures the quantity below.\n\n"
            f"> {text}\n\n"
            "There is no context beyond this sentence. You see no table and "
            "no cases — **derive it from physics**.")


@dataclass
class RoundResult:
    round: int
    n_proposed: int = 0
    n_rejected_static: int = 0
    n_rejected_sandbox: int = 0
    n_rejected_schema: int = 0
    #: ★ How many times the LLM was **unreachable**. It must not be mixed
    #: with schema refusals — exhausted credit, failed authentication or a
    #: network error would look like "the model produced bad rules" (D-43).
    #: A 429 really was read as "144 schema refusals" across 12 rounds.
    n_llm_error: int = 0
    n_rejected_fit: int = 0
    #: ★ How many candidates the weight fitter actually moved (D-54). When
    #: it is low that round was **scored on the initial values**, and
    #: evolution selects luck in the weights rather than structure (§29.3).
    #: On screen it is noticed in the first round.
    n_fit_moved: int = 0
    n_scored: int = 0
    n_accepted: int = 0
    best_regret: float = float("nan")
    best_val_regret: float = float("nan")
    #: ★ The archive best's rank loss when the acceptance criterion is
    #: `rank` (D-101). Under the `regret` condition it is NaN — it is not
    #: forced.
    best_rank_loss: float = float("nan")
    n_cells: int = 0
    val_gap: float = float("nan")
    n_val_blowups: int = 0
    #: ★ Proposal / duplicate / scored counts per parent kind (exploit /
    #: explore / cross) (D-94).
    #: `{"exploit": {"n": 6, "dup": 1, "scored": 5}, ...}`
    #:
    #: Why it is recorded here: the parent kind lived **only in the prompt
    #: string**, so it could not be read in runs where `llm_calls`'
    #: `prompt` was empty — an attempt to measure "hypothesis-parent
    #: mismatch" from old material failed.
    by_parent_kind: dict = field(default_factory=dict)
    #: ★ New axes the Analyst asked for this round / axes actually built
    #: (D-75).
    n_feature_requests: int = 0
    n_features_made: int = 0
    #: Requests **not built** because of the per-round cap. They are not
    #: discarded silently.
    n_feature_over_cap: int = 0
    seconds: float = 0.0
    llm_calls: dict = field(default_factory=dict)
    rejections: list[tuple] = field(default_factory=list)

    def line(self) -> str:
        err = f"★LLM err {self.n_llm_error} " if self.n_llm_error else ""
        mv = (f"fit moved {self.n_fit_moved}/{self.n_scored} | "
              if self.n_scored else "")
        gap = f"{self.val_gap:+.3f}"
        alarm = "!" if self.val_gap > VAL_GAP_ALARM else " "
        over = f"({self.n_feature_over_cap} over cap) " \
            if self.n_feature_over_cap else ""
        feat = (f"new axes {self.n_features_made}/{self.n_feature_requests} "
                f"{over}| " if self.n_feature_requests else "")
        return (
            f"r{self.round:<3d} proposed {self.n_proposed:2d} | {err}{feat}"
            f"refused schema {self.n_rejected_schema} static "
            f"{self.n_rejected_static} sandbox {self.n_rejected_sandbox} "
            f"fit {self.n_rejected_fit} | scored {self.n_scored:2d} "
            f"accepted {self.n_accepted:2d} | {mv}best {self.best_regret:.4f} "
            + (f"rank {self.best_rank_loss:.4f} "
               if self.best_rank_loss == self.best_rank_loss else "")
            + f"val {self.best_val_regret:.4f}({gap}{alarm})"
            f"| cells {self.n_cells:2d} blowups {self.n_val_blowups} | "
            f"{self.seconds:.1f}s")


def _output_schemas() -> dict:
    """Every JSON schema that goes to the model, in full (D-152).

    ★ It is what the model reads, not what we think it reads — a cap of 8
    lived in a field description for two days after the checker had dropped
    it (D-151). Without pydantic it is `{}`; a run without the LLM extra
    does not fail for this.
    """
    try:
        from kernelrule.agents import schemas as _s
        if not _s.HAVE_PYDANTIC:                    # pragma: no cover
            return {}
        out = {"RuleOutput": _s.rule_output_for().model_json_schema()}
        for name in ("AnalysisOutput", "FeatureOutput", "CategoryOutput"):
            cls = getattr(_s, name, None)
            if cls is not None and hasattr(cls, "model_json_schema"):
                out[name] = cls.model_json_schema()
        return out
    except Exception:                               # noqa: BLE001
        return {}


def _git_commit() -> str:
    """The current commit. For the trace's first line to stand on its own
    it needs the code version."""
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=False,
                              timeout=5).stdout.strip() or "?"
    except Exception:                                     # noqa: BLE001
        return "?"


def _sha(code: str) -> str:
    """A short hash of the code. Used in the trace to link the same
    proposal together (D-133)."""
    import hashlib

    return hashlib.sha256(code.encode()).hexdigest()[:12]


class RoundLoop:
    def __init__(self, *, cfg: LoopConfig, table: PerfTable,
                 matrix: FeatureMatrix, splits: SplitSet, llm) -> None:
        # ★ The rank loss is taken out of the evolution path (D-128). It
        #   is checked **first** — caught later, it would die after the
        #   whole table had been read (principle 1).
        if cfg.objective != "regret" or cfg.objective_switch is not None:
            raise ValueError(
                f"the evolution objective is only regret "
                f"(objective={cfg.objective!r}, "
                f"objective_switch={cfg.objective_switch!r}). The rank loss "
                f"was concluded to be the wrong objective (D-118 · D-121) — "
                f"it is used as a metric only. To reproduce an old run, go "
                f"back to that commit (D-128).")
        self.cfg = cfg
        # ★ The trace (D-133). It accumulates in the same place as
        # `dump()`.
        from kernelrule.core.trace import Tracer
        self.trace = Tracer(
            Path(cfg.out_dir) / cfg.run_id / "trace.jsonl"
            if cfg.trace else None)
        self.table = table
        self.matrix = matrix
        self.splits = splits
        self.llm = llm
        # ★ It is not injected but **computed here from the training
        #   split** (§12.3 / D-28). If the caller could pass a sentence
        #   computed on the full table, the report's split check would do
        #   nothing — the first real run was contaminated exactly that way.
        from kernelrule.report.table_facts import TableFacts
        self.table_facts = TableFacts.compute(table, splits.train)
        self.rng = np.random.default_rng(cfg.seed)
        # ★ The cells are always dynamic tertiles (D-144) — `cell_mode`
        # was removed.
        self.archive = Archive(
            noise_tol=0.0,
            select_by=("rank" if cfg.objective == "rank" else "regret"))
        self.rounds: list[RoundResult] = []
        self.failures: list[dict] = []
        self.hypotheses: list[dict] = []
        #: ★ Features built inside a round (D-75). They stay in the
        #: artefacts as they are.
        self.features_made: list[dict] = []
        #: ★ Did a cross child mix terms from both parents (D-96
        #: observation 1)? The parent code does not survive, so it is
        #: computed on the spot.
        self.cross_lineage: list[dict] = []
        #: ★ The archive best per round (code included). For the D-101
        #: observations.
        self.bests: list[dict] = []
        #: ★ The objective-switch state (D-104). `switch_round` is -1 when
        #: it never switches.
        #: ⚠️ 2026-09-09 (D-150): there is no parameter budget. The field is
        #: kept because `config.json` records `parameters` and old configs
        #: carry it — nothing refuses a rule for it.
        self._budget = (cfg.parameters if cfg.parameters is not None
                        else _FITTER_DIM)
        self._limits = limits_for()
        #: ★ Where a refusal happened -> (why, the full message), keyed by
        #: code (D-154). Written by `_admit` / `_evaluate_candidate`, drained
        #: by the trace event. It is not state the loop reads.
        self._why: dict[str, tuple[str, str]] = {}
        self._objective = cfg.objective
        self._switched = False
        self.switch_round = -1
        #: The **single source** of hypothesis ids. The ids the model
        #: attaches are unique only within a response, so they collide
        #: across rounds and passes.
        self._hyp_seq = 0
        #: The borrowed hypothesis blocks (control arm C). Read once, on
        #: first use.
        self._pool: list[list[dict]] | None = None
        #: The process pool for parallel scoring (D-95). It is not rebuilt
        #: every round.
        self._pool_exec = None
        self._rule_seq = 0
        self._seen_code: dict[str, float] = {}      # the cache (§15.4)
        self._feats = matrix.feature_names()
        #: ★ For the exponent-slot guard (D-112). It is refreshed together
        #: when axes are added.
        self._fmins = matrix.feature_mins()
        self._shape_vals = matrix.shape_value_names()
        self._short_mask, self._long_mask = self._regime_masks()
        # ★ The pool is created **in the constructor** (D-95). `fork` is
        #   dangerous while threads are running (CPython raises a
        #   DeprecationWarning) and asyncio LLM calls create threads — the
        #   constructor is the earliest point this object controls. The
        #   masks go into `_WORKER`, so it must come after that.
        #   ⚠️ It is not complete — if an earlier stage in the same process
        #      already called the LLM, threads may exist. Fall back to
        #      `n_workers=0` then (it is the default).
        self._pool_or_none()

    # -- ★ Control arm C — borrowing someone else's hypotheses (§16.1,
    #    D-91) --------------------------------------------------------------
    def _pool_round(self, r: int) -> list[dict]:
        """Borrows **a whole round** of another run, entire.

        There are two reasons for taking them per (run, round) rather than
        mixing individual hypotheses.

        ```
        1  One Analyst output is a set that complements itself. Mixing them
           individually gives not "a hypothesis that does not fit" but "a
           self-contradictory block", which measures something else
        2  The per-round count distribution automatically matches B's
           (4.4 on average)
        ```

        ⚠️ **Runs with the same seed number are excluded** — giving
        `abl-B-s1`'s hypotheses to `abl-C-s1` is not "another run".
        """
        if self._pool is None:
            import json as _json
            groups: dict[tuple, list[dict]] = {}
            mine = self.cfg.run_id.rsplit("-s", 1)[-1]
            for path in self.cfg.hypothesis_pool:
                pp = Path(path)
                src = pp.parent.name
                if src.rsplit("-s", 1)[-1] == mine:
                    continue        # the same seed number is excluded
                for ln in pp.read_text().splitlines():
                    if not ln.strip():
                        continue
                    h = _json.loads(ln)
                    if h.get("analyst_pass", 1) != 1:
                        continue
                    groups.setdefault((src, h.get("round")), []).append(h)
            self._pool = [groups[k] for k in sorted(groups)]
            if not self._pool:
                raise ValueError(
                    "the hypothesis pool is empty. Either only the same "
                    "seed number was given or the paths are wrong — it does "
                    "not silently run without hypotheses (§26.4).")
            self.rng.shuffle(self._pool)
        block = self._pool[r % len(self._pool)]
        out = []
        for h in block:
            g = dict(h)
            # ★ The provenance is recorded. Without being able to ask
            #   later "where did this hypothesis come from", this arm's
            #   result cannot be interpreted.
            g["borrowed_from"] = f"{h.get('id')}@r{h.get('round')}"
            g.pop("analyst_pass", None)
            out.append(g)
        return out

    # -- ★ Analyst -> FeatureWriter (D-75) --------------------------------
    def _write_features(self, hyps: list[dict], r: int,
                        res: RoundResult) -> list[str]:
        """**Builds features** for the physical quantities the hypotheses
        asked for. Returns the names built.

        ## Why it exists

        `needs_new_feature` was filled in 303 times across 33 runs, and
        **there was no code in `loop.py` that read it.** Roughly one in five
        said "measuring this needs a new axis", and all of it was thrown
        away.

        ## Three conditions — this function holds them

        ```
        1  ★ The FeatureWriter is **not given the diagnostic report.**
           Only the one requirement sentence is passed. It blocks the
           channel by which a feature built inside the loop could be fitted
           to the training shapes — that keeps condition F1 intact inside
           the loop too
        2  The per-round cap (`cfg.max_new_features_per_round`)
           §21: the feature matrix recomputes every shape for each new axis
        3  The verdict is by observation, not performance — frequency /
           usage rate / what was asked for
        ```

        ⚠️ Anything caught by validation (§8.3) is **not thrown away
        silently.** The refusal reason stays in `features_made` — "what did
        it try to build and fail" is an observation.
        """
        from kernelrule.features.generated import (
            FeatureRejected,
            register_generated,
        )
        from kernelrule.features.validate import alt_hw

        reqs = [(h.get("id", "?"), _requirement_of(h)) for h in hyps]
        reqs = [(hid, t) for hid, t in reqs if t]
        res.n_feature_requests = len(reqs)
        if not reqs:
            return []

        made: list[str] = []
        reg = self.matrix.registry
        cap = self.cfg.max_new_features_per_round
        # ★ The cap is applied **in code**. Suppressing it through the
        #   prompt would directly press down the very thing being measured
        #   (the request frequency) — the 17.9% baseline was measured
        #   without such wording. Requests over the cap are **dropped and
        #   recorded**: "how many hit the cap" is itself an observation.
        for hid, text in reqs[cap:]:
            self.features_made.append(
                {"round": r, "hypothesis_id": hid, "requirement": text,
                 "accepted": False, "over_cap": True,
                 "error": f"over the per-round cap of {cap} — not built"})
        res.n_feature_over_cap = max(0, len(reqs) - cap)
        for hid, text in reqs[:cap]:
            row = {"round": r, "hypothesis_id": hid, "requirement": text}
            try:
                out = self.llm.complete(
                    "feature", "", condition=self.cfg.feature_condition,
                    registry=reg, task=_feature_task(text))
                row["name"] = out.get("name")
                row["code"] = out.get("code")
                f = register_generated(out["code"], registry=reg, meta=out,
                                       table=self.table, matrix=self.matrix,
                                       hw_alt=alt_hw(self.table.hw))
                # ★ The column is built now. Without it, a rule using the
                #   name gets a KeyError.
                self.matrix.invalidate(f.name)
                self._feats = self.matrix.feature_names()
                self._fmins = self.matrix.feature_mins()
                self._shape_vals = self.matrix.shape_value_names()
                # ★ The workers hold the matrix as of fork time — they
                #   cannot see the new column. The pool is discarded and
                #   rebuilt. Without that, **the workers score on a stale
                #   matrix** and that silently becomes a different score
                #   (D-95).
                self._restart_pool()
                row.update(accepted=True, shape_level=f.shape_level)
                # ★ Declared vs actual, for an axis built inside the loop
                #   too (D-160). ⛔ Recorded, not shown.
                row["range"] = self._range_rows().get(f.name)
                made.append(f.name)
            except FeatureRejected as e:
                row.update(accepted=False, error=str(e)[:200])
            except Exception as e:                          # noqa: BLE001
                row.update(accepted=False,
                           error=f"{type(e).__name__}: {e}"[:200])
            self.features_made.append(row)
            # ★ The attempt goes into the trace as well (D-160). Until now
            #   "was an axis built inside the loop" could only be read from
            #   a separate file, and the trace is what gets released.
            self.trace.ev("feature", round=r, hypothesis_id=hid,
                          requirement=text, name=row.get("name"),
                          accepted=bool(row.get("accepted")),
                          shape_level=row.get("shape_level"),
                          range=row.get("range"), error=row.get("error"),
                          code=row.get("code"))
        res.n_features_made = len(made)
        return made

    def _range_rows(self) -> dict:
        """`name -> {declared, observed}` on the **training** shapes (D-160).

        ⛔ Not for a prompt — see `FeatureMatrix.observed_ranges`.
        """
        obs = self.matrix.observed_ranges(self.splits.train.shapes)
        reg = self.matrix.registry
        return {n: {"declared": [float(x) for x in reg[n].expected_range],
                    "observed": [round(v, 6) for v in obs[n]],
                    "shape_level": bool(reg[n].shape_level)}
                for n in sorted(obs)}

    # -- Regime masks (the cell axes) — ★ cut by size (§10.1, §30.5) ------
    def _regime_masks(self):
        """Separates short and long shapes within the training split.

        ⚠️ The boundary is taken from the **roofline lower bound**, not from
        `best_ms` (the answer).
        ⚠️ It cuts **within the training split only** — using validation as
        a cell axis contaminates the holdout (§10.2).
        """
        # ★ It uses `regime_of` — the same verdict in two places diverges
        #   (principle 2). This used to compute
        #   `info.log_sol_ms < log2(0.5)` directly here, which **assumed
        #   `log_sol_ms` was in the registry**. The F0/F1 registries do not
        #   have it, so the whole loop dies (§30.9). A regime is a property
        #   of (shape, hardware), not of the feature list.
        from kernelrule.core.splits import regime_of

        # ★ The axis changed from size (SOL 0.5ms) to the **roofline**
        #   (D-144).
        short = np.asarray([regime_of(p, self.table.hw, axis="roofline")
                            == "mem" for p in self.splits.train.shapes])
        if not short.any() or short.all():
            import warnings
            warnings.warn(
                f"the training split holds only one roofline band "
                f"(memory {int(short.sum())} / compute "
                f"{int((~short).sum())}). "
                f"The cell axes become meaningless, and evolution "
                f"sacrificing the other regime stays invisible (§10.1).",
                stacklevel=3)
        return short, ~short

    # -- Scoring ----------------------------------------------------------
    # -- Parallel scoring (D-95) ------------------------------------------
    def _pool_or_none(self):
        """A process pool made with `fork`. **It does not copy the table
        and the matrix.**

        The table + feature matrix is 4.2GB. Reloading per worker would be
        50GB across 12 and 3 seconds each just to load. `fork` hands the
        parent's memory over copy-on-write, so neither cost applies — but
        **`_WORKER` has to be filled before the pool is created.**

        ⚠️ `fork` is dangerous while threads are running. This is called
        only **after** `_call_optimizers` (asyncio) has finished.
        """
        if self.cfg.n_workers <= 0:
            return None
        if self._pool_exec is None:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context
            _WORKER.update(
                table=self.table, matrix=self.matrix,
                train=self.splits.train, val=self.splits.val,
                max_evals=self.cfg.max_evals,
                objective=self._objective,
                rank_top_k=self.cfg.rank_top_k,
                rank_lambda=self.cfg.rank_lambda,
                fit_method=self.cfg.fit_method,
                fit_restarts=self.cfg.fit_restarts,
                short_mask=self._short_mask, long_mask=self._long_mask)
            self._pool_exec = ProcessPoolExecutor(
                max_workers=self.cfg.n_workers, mp_context=get_context("fork"))
        return self._pool_exec

    def _restart_pool(self) -> None:
        """Discards and rebuilds the pool. **Mandatory whenever the matrix
        changes** (D-95)."""
        if self._pool_exec is not None:
            self._pool_exec.shutdown(wait=True)
            self._pool_exec = None
        self._pool_or_none()

    def _evaluate_batch(self, props: list, res: RoundResult) -> list:
        """Several candidates at once. The result **must equal the
        sequential one** (D-95).

        ★ How determinism is kept:
        ```
        results are assembled in submission order (`sorted(by i)`)
        rule ids and counters are assigned by the parent in that order
        fit_weights has a fixed seed, so it is independent of worker order
        ```
        """
        pool = self._pool_or_none()
        admitted = []
        for prop in props:
            got = self._admit(prop, res)
            if got is not None:
                admitted.append((prop, got[1]))
        if not admitted:
            return []
        if pool is None:               # sequential — the path so far
            out = []
            for prop, _rep in admitted:
                e = self._evaluate_candidate(prop, res)
                if e is not None:
                    out.append(e)
            return out

        jobs = [(i, prop.code, list(prop.w0))
                for i, (prop, _r) in enumerate(admitted)]
        got = sorted(pool.map(_fit_and_score, jobs), key=lambda d: d["i"])
        elites = []
        for d in got:
            prop, rep = admitted[d["i"]]
            if "err" in d:
                res.n_rejected_fit += 1
                res.rejections.append(d["err"])
                self._why[prop.code.strip()] = (
                    str(d["err"][0]), str(d["err"][1]))
                continue
            if d["moved"]:
                res.n_fit_moved += 1
            res.n_scored += 1
            elites.append(self._elite_from(prop, rep, d))
        return elites

    def _score(self, score_fn, w, shapes):
        return evaluate_scores(make_score_of(score_fn, self.matrix, w),
                               self.table, list(shapes), ks=(1, 3))

    def _admit(self, prop, res: RoundResult):
        """Static checks -> compile -> sandbox. **fail-closed.** It runs in
        the parent.

        ★ These three are 3% of a round. They are not sent to the workers
        not for cost but because `run_isolated` starts a process — doing it
        inside a worker would be a nested spawn (D-95).
        """
        rep = check_rule(prop.code, limits=self._limits,
                         feature_names=self._feats,
                         feature_mins=self._fmins,
                         shape_value_names=self._shape_vals,
                         n_weights=len(prop.w0))
        if not rep.ok:
            res.n_rejected_static += 1
            res.rejections.append(("static", rep.violations[0][:90]))
            # ★ **Every** violation, not the first (D-154). The static check
            #   returns several at once, and the trace used to keep none of
            #   them — a refusal read as `fit_or_run` and the reason had to
            #   be recovered by re-checking the code by hand.
            self._why[prop.code.strip()] = (
                "static", " | ".join(rep.violations))
            return None

        try:
            fn = compile_rule(prop.code)
        except SandboxError as e:
            res.n_rejected_sandbox += 1
            res.rejections.append(("compile", str(e)[:90]))
            self._why[prop.code.strip()] = ("compile", str(e))
            return None

        if self.cfg.sandbox_first_seen:
            p0 = self.splits.train.shapes[0]
            f, info = self.matrix.for_shape(p0)
            out = run_isolated(prop.code, (f, info, self.table.hw,
                                           np.asarray(prop.w0)), timeout=5.0)
            if not out.ok:
                res.n_rejected_sandbox += 1
                res.rejections.append(("sandbox", str(out)[:90]))
                self._why[prop.code.strip()] = ("sandbox", str(out))
                return None
        return fn, rep

    def _elite_from(self, prop, rep, out: dict) -> Elite:
        """Worker result -> `Elite`. ★ The ids and order are decided by
        **the parent** (determinism)."""
        self._rule_seq += 1
        return Elite(
            rule_id=f"r{self._rule_seq:04d}", code=prop.code,
            w=out["w"], regret=out["regret"],
            mem_objective=out["mem"], comp_objective=out["comp"],
            all_objective=out["all"],
            code_len=rep.n_nodes, code_terms=rep.n_terms,
            round=len(self.rounds),
            changes=prop.changes, hypothesis_id=prop.hypothesis_id,
            val_regret=out["val_regret"],
            rank_loss=float(out.get("rank_loss", float("nan"))))

    def _evaluate_candidate(self, prop, res: RoundResult):
        """Static checks -> sandbox -> weight optimisation -> scoring.
        **All fail-closed.**"""
        got = self._admit(prop, res)
        if got is None:
            return None
        fn, rep = got

        try:
            # ★ The fitter is decided by **this rule's len(W0)** (D-144).
            _ft = fitter_for(len(prop.w0))
            fr = fit_weights(fn, self.matrix, self.table, self.splits.train,
                             prop.w0, max_evals=_ft["max_evals"],
                             val_split=self.splits.val,
                             objective=self._objective,
                             rank_top_k=self.cfg.rank_top_k,
                             rank_lambda=self.cfg.rank_lambda,
                             method=_ft["fit_method"],
                             n_restarts=_ft["fit_restarts"],
                             bounds=weight_bounds(prop.code, len(prop.w0)))
        except (FitError, SchemaViolation) as e:
            res.n_rejected_fit += 1
            res.rejections.append(("fit", str(e)[:90]))
            self._why[prop.code.strip()] = ("fit", str(e))
            return None
        except Exception as e:                            # noqa: BLE001
            # A rule blowing up during scoring is a **rejection**. It is
            # not swallowed (§26.4).
            res.n_rejected_fit += 1
            res.rejections.append(("run", f"{type(e).__name__}: {e}"))
            self._why[prop.code.strip()] = ("run", f"{type(e).__name__}: {e}")
            return None

        if fr.moved:
            res.n_fit_moved += 1
        ev = self._score(fn, fr.w, self.splits.train.shapes)
        res.n_scored += 1
        return self._elite_from(prop, rep, {
            "w": [float(x) for x in fr.w], "regret": fr.fit_regret,
            "rank_loss": _rank_loss_of(fn, {
                "objective": self._objective,
                "rank_top_k": self.cfg.rank_top_k,
                "rank_lambda": self.cfg.rank_lambda,
                "matrix": self.matrix, "table": self.table,
                "train": self.splits.train}, fr.w),
            "val_regret": fr.val_regret, "moved": fr.moved,
            "mem": ev.at(1, mask=self._short_mask),
            "comp": ev.at(1, mask=self._long_mask),
            "all": ev.at(1)})

    def score_only(self, code: str, w0) -> float:
        """Scores one rule **on the training split only**. It does not
        enter the archive.

        It is used to rank RuleWriter candidates (§30.9 stage 2).
        `fit_weights` computes the holdout for reporting only and it is not
        returned here — if seed selection looks at the holdout, that holdout
        is not a holdout (§26.4, principle 6).
        """
        from kernelrule.agents.schemas import RuleProposal

        res = RoundResult(round=-2)
        e = self._evaluate_candidate(
            RuleProposal(code=code, w0=list(w0), changes="score_only"), res)
        if e is None:
            raise ValueError(f"the candidate was refused: {res.rejections}")
        return float(e.regret)

    def seed(self, code: str, w0, *, changes: str = "seed") -> Elite:
        """Puts the initial rule into the archive.

        **Without it, round 1 starts from an empty parent.** To take the
        hand rule as the baseline and test "can it read the report and fix
        that rule", it has to start here — otherwise the loop reads the
        report of an entirely different rule.
        """
        from kernelrule.agents.schemas import RuleProposal

        res = RoundResult(round=-1)
        e = self._evaluate_candidate(
            RuleProposal(code=code, w0=list(w0), changes=changes), res)
        if e is None:
            raise ValueError(
                f"the initial rule was refused: {res.rejections}")
        e.round = -1
        self.archive.consider(e)
        self._seen_code[code.strip()] = e.regret
        return e

    # -- One round --------------------------------------------------------
    def _record_cross(self, r: int, codes: list[str], child: str) -> None:
        """★ Did the child use features **unique to each of the two
        parents** (D-96)?

        If `cross` takes two parents and copies only one, it is no different
        from `explore` — that is observation 1 of the experiment plan.
        Without counting **the cases where there is nothing to mix** (when
        A-unique or B-unique is empty) separately, the denominator is wrong.
        """
        def feats(code: str) -> set:
            # ★ It used to be `except Exception`. Adding `self._fmins` on
            #   2026-09-03 swallowed that `AttributeError` and the feature
            #   set passed through **empty** — a place where the observation
            #   device silently becomes 0. Only the case where the rule
            #   cannot be parsed is swallowed.
            try:
                return check_rule(code, limits=self._limits,
                                  feature_names=self._feats,
                                  feature_mins=self._fmins,
                                  shape_value_names=self._shape_vals,
                                  n_weights=self._budget).features_used
            except SyntaxError:
                return set()
        a, b, c = feats(codes[0]), feats(codes[1]), feats(child)
        only_a, only_b = a - b, b - a
        self.cross_lineage.append({
            "round": r,
            "a": sorted(a), "b": sorted(b), "child": sorted(c),
            "only_a": sorted(only_a), "only_b": sorted(only_b),
            # Was there anything to mix — if not, it is taken out of the
            # denominator
            "mixable": bool(only_a and only_b),
            "took_a": sorted(c & only_a), "took_b": sorted(c & only_b),
            "mixed": bool(c & only_a) and bool(c & only_b),
            # Did it copy one side whole (character-identical to a parent's
            # code)
            "copied": child.strip() in (codes[0].strip(), codes[1].strip())})

    def run_round(self) -> RoundResult:
        t0 = time.perf_counter()
        r = len(self.rounds)
        res = RoundResult(round=r)
        calls = {"analyze": 0, "rule_editor": 0, "feature": 0}
        self.trace.ev("round_start", round=r,
                      archive_best=(self.archive.best.regret
                                    if self.archive.best else None),
                      cells=self.archive.n_cells)

        # 1~2. the diagnostic report -> hypotheses
        #  ★ With `use_analyst=False` this whole block is skipped (§16.1).
        #    The report is not even built — building it and not giving it
        #    would be "there is a diagnosis and it is unused", a different
        #    condition.
        hyps: list[dict] = []
        if self.cfg.use_analyst and self.archive.best is not None:
            def analyze() -> list[dict]:
                """**Rebuilds** the report and receives hypotheses.

                ★ The report is not reused — if stage 3 creates an axis the
                feature list changes, and handing the old report back would
                leave the Analyst blind to the axis just built.
                """
                fn = compile_rule(self.archive.best.code)
                rep = build_report(
                    run_id=f"{self.cfg.run_id}-r{r:03d}", table=self.table,
                    matrix=self.matrix, score_fn=fn,
                    weights=self.archive.best.w,
                    code=self.archive.best.code, train=self.splits.train,
                    table_facts=self.table_facts,
                    failures=self.failures[-20:],
                    hypotheses_applied=[h["claim"][:70]
                                        for h in self.hypotheses[-3:]])
                out = self.llm.complete("analyze", rep.render())
                calls["analyze"] += 1
                return list((out or {}).get("hypotheses", []))

            first = analyze()
            for h in first:
                h["analyst_pass"] = 1
            hyps = first
            # ★ When it goes back, **both responses are kept.** It used to
            #   overwrite `hyps`, so the first response vanished from the
            #   record — and that is the one holding the requests, so the
            #   **denominator of "request frequency" disappeared entirely.**
            replaced: list[dict] = []

            # 3. ★ If it asked for an axis that does not exist, build it
            #    (D-75)
            if self.cfg.max_new_features_per_round > 0 and first:
                made = self._write_features(first, r, res)
                calls["feature"] += min(res.n_feature_requests,
                                        self.cfg.max_new_features_per_round)
                if made:
                    # ★ **It goes back to the Analyst.** Building an axis
                    #   and not using it in that round is half a job —
                    #   waiting for the next round breaks the link to the
                    #   hypothesis that asked for it.
                    second = analyze()
                    if second:
                        for h in second:
                            h["analyst_pass"] = 2
                        replaced, hyps = first, second

            # Ids are attached to the hypotheses. They are used in the
            # RuleEditor prompt and for lineage tracking.
            # ⚠️ When comparing request frequency against old runs, count
            #    only `analyst_pass == 1` — old runs called the Analyst once
            #    per round (principle 4).
            # ★ **We** assign the ids. The model labels `H0..H4` within its
            #   own response, so leaving those as they are makes them
            #   collide across rounds and between the two responses of a
            #   go-back. A collision silently misaligns the lineage.
            for h in replaced + hyps:
                h["id"] = f"H{self._hyp_seq}"
                self._hyp_seq += 1
                h["round"] = r
                h.setdefault("analyst_pass", 1)
            self.hypotheses.extend(replaced)
            self.hypotheses.extend(hyps)
            self.trace.llm_calls(self.llm, round=r)
            self.trace.ev("hypotheses", round=r,
                          ids=[h.get("id") for h in hyps],
                          claims=[h.get("claim", "") for h in hyps],
                          # ★ `_requirement_of` — the field is
                          #   `needs_new_feature` (both names, D-81). Reading
                          #   a name that exists nowhere wrote null every
                          #   time, and under F1 30 of 36 hypotheses ask for
                          #   a feature (D-159)
                          needs_feature=[_requirement_of(h) or None
                                         for h in hyps],
                          n_replaced=len(replaced))
        elif self.cfg.hypothesis_pool and self.archive.best is not None:
            # ★ Control arm C — the Analyst is not called and someone
            # else's hypotheses go in (D-91)
            hyps = self._pool_round(r)
            for h in hyps:
                h["id"] = f"H{self._hyp_seq}"
                self._hyp_seq += 1
                h["round"] = r
                h["analyst_pass"] = 0        # 0 = borrowed
            self.hypotheses.extend(hyps)

        # 4. Rule generation — ★ parallel calls (§4-0). 12 or 1, the wall
        #    clock is similar
        parents = self.archive.parents(self.cfg.n_rules_per_round, self.rng)
        applied = [f"{h.get('id','?')}: {h.get('claim','')[:80]}"
                   for h in self.hypotheses[-4:]]
        # ★ The cells are tertiles, so an Elite alone does not know them —
        #   the current placement is looked up and written down.
        _where = {id(v): list(k) for k, v in self.archive.cells.items()}
        self.trace.ev("parents", round=r, picks=[
            {"kind": k, "rules": [x.rule_id for x in ps],
             "cells": [_where.get(id(x)) for x in ps]}
            for k, ps in parents])
        # ★ The exploit slots are given **different hypotheses** (D-144).
        #   The old way randomised across all slots, so two exploits could
        #   draw the same hypothesis, and in the traces 67 of 77 exploit
        #   duplicates (87%) were "the same parent + the same hypothesis".
        #   The parent is the same (one global best), so if the hypothesis
        #   matches too the prompts become literally identical.
        #   ⚠️ The randomisation across all slots is kept — only the
        #      exploits are made distinct from each other.
        n_exploit = sum(1 for k, _ in parents if k == "exploit")
        _exploit_hyps: list = []
        if hyps and n_exploit:
            idx = list(self.rng.permutation(len(hyps)))
            while len(_exploit_hyps) < n_exploit:
                _exploit_hyps.extend(hyps[i] for i in idx)
            _exploit_hyps = _exploit_hyps[:n_exploit]
        reqs = []
        for kind, ps in parents:
            parent, parent2, n_terms, path_params = None, None, 0, 0
            if ps:
                from kernelrule.agents.schemas import RuleProposal
                parent = RuleProposal(code=ps[0].code, w0=ps[0].w)
                # ★ The second parent is **actually passed** (D-96).
                #   `archive.parents` gives `cross` two Elites, but only
                #   `ps[0]` was used here, so **§13's crossover was never
                #   implemented** — `cross` was the same as `explore`.
                if len(ps) > 1:
                    parent2 = RuleProposal(code=ps[1].code, w0=ps[1].w)
                # The parent's term count is measured and put into the
                # prompt (the replacement frame)
                pr = check_rule(ps[0].code, limits=self._limits,
                                feature_names=self._feats,
                                feature_mins=self._fmins,
                                shape_value_names=self._shape_vals,
                                n_weights=len(ps[0].w))
                n_terms = pr.n_terms
                # The heaviest path's parameter count — **reported to the
                # model, not enforced** (D-150).
                path_params = pr.parameters_used
            # ★ Hypothesis assignment is **random** (D-94).
            #   `hyps[i % len(hyps)]` uses the earlier hypotheses more often
            #   — with 5 hypotheses it is 3 3 2 2 2, with 7 it is
            #   2 2 2 2 2 1 1. **That is 12 % n, not a design**, and it also
            #   correlates with the parent kind (i=0~5 exploit / 6~8 explore
            #   / 9~11 cross). It uses `self.rng`, so it is reproducible from
            #   the seed.
            if kind == "exploit" and _exploit_hyps:
                hyp = _exploit_hyps.pop(0)
            else:
                hyp = (hyps[int(self.rng.integers(len(hyps)))]
                       if hyps else None)
            reqs.append({"prompt": f"round={r} parent={kind}",
                         # ★ The parent kind is recorded as **its own
                         #   field**. Living only in the prompt string, it
                         #   cannot be read in runs where `llm_calls`'
                         #   `prompt` is empty — "hypothesis-parent
                         #   mismatch" really could not be measured from old
                         #   material (D-94).
                         "parent_kind": kind,
                         "parent": parent, "parent2": parent2,
                         # ★ The parent code, for observation (D-96
                         #   observation 1). It does not enter the prompt —
                         #   it is a key `_user_prompt` does not read.
                         #   Counting whether the child used features unique
                         #   to each parent needs the parents' feature sets.
                         "_codes": [x.code for x in ps[:2]],
                         # ★ The parent ids, for the trace (D-133). A key
                         #   `_user_prompt` does not read — it does not enter
                         #   the prompt.
                         "_parent_ids": [x.rule_id for x in ps[:2]],
                         "parent_n_terms": n_terms,
                         "parent_path_params": path_params,
                         "hypothesis": hyp,
                         "hypotheses_applied": applied,
                         # ★ Whether to build the hypothesis section
                         #   (§16.1). It must not be guessed from
                         #   `hypothesis=None` — that would mix "a round
                         #   with no hypothesis" with "no Analyst at all".
                         # Whether to build the hypothesis section. Control
                         # arm C does not call the Analyst but **does
                         # receive hypotheses**, so the section must exist
                         "analyst": bool(self.cfg.use_analyst
                                         or self.cfg.hypothesis_pool)})
        raws = self._call_optimizers(reqs)
        calls["rule_editor"] += len(reqs)
        self.trace.llm_calls(self.llm, round=r)

        elites: list[Elite] = []
        #: Candidates to send in parallel — the parent kind travels with
        #: them (for the D-94 counts).
        batch: list = []

        def bump(kind: str, key: str) -> None:
            res.by_parent_kind.setdefault(
                kind, {"n": 0, "dup": 0, "scored": 0})[key] += 1

        for i, (req, raw) in enumerate(zip(reqs, raws, strict=True)):
            hyp = req["hypothesis"]
            kind = req.get("parent_kind", "?")
            bump(kind, "n")
            res.n_proposed += 1
            if isinstance(raw, BaseException):
                # ★ Two things are separated (D-43).
                #   transport failure  credit / auth / network. **Our
                #                      problem**
                #   retries exhausted  the model failed to match the schema.
                #                      Discarded (§26.4)
                #   Mixed, it reads as "the model produced bad rules".
                if _is_transport_error(raw):
                    res.n_llm_error += 1
                    res.rejections.append(("llm-transport", (
                        f"{type(raw).__name__}: {str(raw)[:70]}")))
                    self.trace.ev("reject", round=r, i=i, kind=kind,
                                  why="llm-transport",
                                  detail=f"{type(raw).__name__}: {raw}"[:300])
                else:
                    res.n_rejected_schema += 1
                    res.rejections.append(("llm", (
                        f"{type(raw).__name__}: {str(raw)[:70]}")))
                    self.trace.ev("reject", round=r, i=i, kind=kind,
                                  why="llm",
                                  detail=f"{type(raw).__name__}: {raw}"[:300])
                continue
            try:
                # ★ The budget is passed (D-107). Without it, 16-term
                #   rules are silently refused on the MockLLM path and on
                #   any path not using structured output.
                prop = validate_rule_proposal(raw,
                                              parameters=self.cfg.parameters)
            except SchemaViolation as e:
                res.n_rejected_schema += 1
                res.rejections.append(("schema", str(e)[:90]))
                self.trace.ev("reject", round=r, i=i, kind=kind,
                              why="schema", detail=str(e)[:300])
                continue
            if hyp:
                prop.hypothesis_id = hyp.get("id", "")
            if kind == "cross" and len(req.get("_codes") or ()) > 1:
                self._record_cross(r, req["_codes"], prop.code)
            self.trace.ev("proposal", round=r, i=i, kind=kind,
                           parents=req.get("_parent_ids") or [],
                           hyp=(hyp or {}).get("id"),
                           changes=prop.changes, n_weights=len(prop.w0),
                           code=prop.code, code_sha=_sha(prop.code))
            key = prop.code.strip()
            if key in self._seen_code:      # it is not rescored (§15.4)
                bump(kind, "dup")
                self.trace.ev("duplicate", round=r, i=i, kind=kind,
                              code_sha=_sha(prop.code))
                continue
            # ★ Deduplication is finished **before entering the parallel
            #   path** — duplicates within the same round have to be caught
            #   here too, or the workers do the same job twice.
            self._seen_code[key] = float("nan")
            batch.append((prop, kind))

        # ★ Fitting and scoring happen in one batch (D-95). At
        #   `n_workers=0` it stays sequential.
        got = self._evaluate_batch([b for b, _k in batch], res)
        by_code = {e.code.strip(): e for e in got}
        for prop, kind in batch:
            e2 = by_code.get(prop.code.strip())
            if e2 is None:
                self._seen_code.pop(prop.code.strip(), None)
                # ★ It never reached scoring. ⚠️ 2026-09-10 (D-154): this used
                #   to say `why="fit_or_run"` with no detail whatever the
                #   real cause was — 17 AST-cap refusals were recorded that
                #   way and had to be recovered by re-checking the code
                #   (D-153). The reason is now taken from where it happened,
                #   with **the full message and the code**.
                why, detail = self._why.pop(prop.code.strip(),
                                            ("fit_or_run", ""))
                self.trace.ev("reject", round=r, kind=kind, why=why,
                              detail=detail, code=prop.code,
                              n_weights=len(prop.w0),
                              code_sha=_sha(prop.code))
                continue
            bump(kind, "scored")
            self.trace.ev("scored", round=r, kind=kind, rule=e2.rule_id,
                          code_sha=_sha(e2.code), fit=e2.regret,
                          val=e2.val_regret,
                          # ★ Did the fitter fail to move — this is the
                          #   "silently did nothing" spot (D-54)
                          moved=bool(getattr(e2, "moved", True)))
            self._seen_code[prop.code.strip()] = e2.regret
            elites.append(e2)

        # 6~7. Update the archive + record the failures
        before = self.archive.best.regret if self.archive.best else float("inf")
        for e in elites:
            won = self.archive.consider(e)
            # ★ The tertile cell is decided by the population — the
            #   current placement is looked up and written down (D-144).
            _cell = next((list(k) for k, v in self.archive.cells.items()
                          if v is e), None)
            # ★ 2026-09-10 (D-155): the **raw axis values** go in beside the
            #   cell. Without them the trace shows which cell a rule landed
            #   in but not why, and "the cells are spent on bad rules" took a
            #   re-analysis of the code to see.
            from kernelrule.core.archive import CELL_AXIS_NAMES as _AX
            self.trace.ev("archive", round=r, rule=e.rule_id,
                          accepted=bool(won), cell=_cell, regret=e.regret,
                          axes={nm: getattr(e, nm) for nm in _AX},
                          mem=e.mem_objective, comp=e.comp_objective,
                          cut_line=self.archive.summary()["cut_line"])
            if won:
                res.n_accepted += 1
            else:
                self.failures.append({
                    "round": r, "idea": e.changes,
                    "regret_before": round(before, 4),
                    "regret_after": round(e.regret, 4),
                    "verdict": "made_worse" if e.regret > before
                               else "no_effect"})
        res.n_cells = self.archive.n_cells
        if self.archive.best:
            res.best_regret = self.archive.best.regret
            res.best_rank_loss = self.archive.best.rank_loss
            res.best_val_regret = self.archive.best.val_regret
            # ★ **The best at that moment** is recorded per round, code
            #   included (the D-101 observations). `archive.jsonl` holds
            #   only the final state, so "does tau rise per round" cannot be
            #   retraced from it.
            self.bests.append({
                "round": len(self.rounds), "rule_id": self.archive.best.rule_id,
                "code": self.archive.best.code, "w": self.archive.best.w,
                "regret": self.archive.best.regret,
                "rank_loss": self.archive.best.rank_loss})
            res.val_gap = res.best_val_regret - res.best_regret
        # ★ The archive selects on the **training** score (using
        #   validation would contaminate the holdout). So a rule that
        #   collapses on validation can become the "best" — it really
        #   happened (train 1.164 / val 6.085). The selection is left as it
        #   is, but **an alarm is raised.**
        res.n_val_blowups = sum(
            1 for e in self.archive.cells.values()
            if np.isfinite(e.val_regret) and e.val_regret - e.regret
            > VAL_GAP_ALARM)
        res.llm_calls = calls
        res.seconds = time.perf_counter() - t0
        self.rounds.append(res)
        self.trace.ev("round_end", round=r, best=res.best_regret,
                      val=res.best_val_regret, cells=res.n_cells,
                      calls=dict(calls), proposed=res.n_proposed,
                      scored=res.n_scored, accepted=res.n_accepted,
                      rejected=(res.n_rejected_schema + res.n_rejected_static
                                + res.n_rejected_sandbox + res.n_rejected_fit),
                      seconds=round(res.seconds, 1))
        return res

    def _call_optimizers(self, reqs: list[dict]) -> list:
        """Calls for 12 rules. **In parallel** if the client supports it.

        `MockLLM` is synchronous and `OpenAILLM` provides `many()`. The loop
        does not distinguish between them — they sit behind the `LLMClient`
        Protocol.
        """
        # ★ Keys starting with `_` are **for observation** and do not go
        #   into the prompt (D-96). `_user_prompt` does not read them, so
        #   passing them is harmless today, but that is **an accident** —
        #   the day someone sweeps `kw`, they leak in silently.
        reqs = [{k: v for k, v in q.items() if not k.startswith("_")}
                for q in reqs]
        many = getattr(self.llm, "many", None)
        if many is None:
            out = []
            for req in reqs:
                q = dict(req)
                prompt = q.pop("prompt", "")
                try:
                    out.append(self.llm.complete("rule_editor", prompt, **q))
                except Exception as e:                    # noqa: BLE001
                    out.append(e)
            return out
        import asyncio
        return asyncio.run(many("rule_editor", [dict(q) for q in reqs]))

    # -- The objective switch (D-104) -------------------------------------
    def _maybe_switch(self) -> bool:
        """★ Switches when the improvement over the last `switch_window`
        rounds is below the threshold.

        The switch happens **only once.** Switching twice makes two "when do
        we switch" decisions, and that is a hyperparameter.
        """
        sw = self.cfg.objective_switch
        if not sw or self._switched:
            return False
        src, dst = sw.split("->")
        if self._objective != src:
            raise ValueError(
                f"objective_switch={sw!r} but the starting objective is "
                f"{self._objective!r}. It must match the left-hand side.")
        n = self.cfg.switch_window
        if len(self.rounds) < n + 1:
            return False
        key = ("best_rank_loss" if self._objective == "rank"
               else "best_regret")
        vals = [getattr(x, key) for x in self.rounds[-(n + 1):]]
        if not np.all(np.isfinite(vals)) or vals[0] <= 0:
            return False
        if (vals[0] - vals[-1]) / vals[0] >= self.cfg.switch_min_improve:
            return False
        self._switch_to(dst)
        return True

    def _switch_to(self, dst: str) -> None:
        """Switches the objective and **re-sorts the archive under the new
        criterion.**"""
        from kernelrule.core.archive import Archive

        self._objective = dst
        self._switched = True
        self.switch_round = len(self.rounds)
        old = list(self.archive.cells.values())
        if self.archive.best is not None and self.archive.best not in old:
            old.append(self.archive.best)
        # ★ If the new criterion's value is missing, it is filled in.
        #   Leaving it silently NaN makes the archive refuse it (which is
        #   the right behaviour).
        if dst == "rank":
            for e in old:
                if not np.isfinite(e.rank_loss):
                    e.rank_loss = _rank_loss_of(
                        compile_rule(e.code),
                        {"objective": "rank",
                         "rank_top_k": self.cfg.rank_top_k,
                         "rank_lambda": self.cfg.rank_lambda,
                         "matrix": self.matrix, "table": self.table,
                         "train": self.splits.train}, np.asarray(e.w))
        self.archive = Archive(
            noise_tol=0.0,
            select_by=("rank" if dst == "rank" else "regret"))
        for e in sorted(old, key=lambda x: (x.rank_loss if dst == "rank"
                                            else x.regret)):
            self.archive.consider(e)
        # ★ The goal definition in the prompt changes too. The cached
        #   agents are discarded — without that, the old instructions keep
        #   going out (principle 1).
        if hasattr(self.llm, "objective"):
            self.llm.objective = dst
            if hasattr(self.llm, "_agents"):
                self.llm._agents.clear()
        self._restart_pool()
        print(f"  ★ objective switch at r{self.switch_round}: -> {dst} "
              f"({len(old)} archive entries re-sorted -> "
              f"{len(self.archive.cells)} cells)")

    # -- The stopping verdict (§14.3) -------------------------------------
    def should_stop(self) -> tuple[bool, str]:
        """⛔ **Sealed** (2026-09-08, D-144). It is always `(False, "")`.

        Early stopping was turned off in D-132 and `patience` is 0. But the
        old implementation below **read `best_val_regret`** — a path by
        which the validation split enters the stopping verdict. It does not
        run today because `patience=0`, but **the moment someone turns it on
        the test is contaminated** (§29.7 — no path by which validation or
        the final split enters the objective or the stopping rule).

        ★ So the path is removed. To bring early stopping back, write a new
        criterion **that does not look at validation**, and write the
        experiment plan first.

        ```
        the old implementation (sealed, kept rather than deleted):
            n = self.cfg.patience
            if n <= 0: return False, ""
            if len(self.rounds) < n + 1: return False, ""
            vals = [x.best_val_regret for x in self.rounds[-(n + 1):]]
            if not np.all(np.isfinite(vals)): return False, ""
            improved = vals[0] - vals[-1]
            ev = self._score(compile_rule(self.archive.best.code),
                             self.archive.best.w, self.splits.val.shapes)
            significant = is_significant(improved, ev)
            new_cell_recent = (self.archive.last_new_cell_round
                               > len(self.rounds) - 1 - n)
            if significant or new_cell_recent: return False, ""
            return True, "..."
        ```
        """
        if self.cfg.patience:
            raise ValueError(
                f"patience={self.cfg.patience} but the early-stop path is "
                f"sealed (D-144). That path read the validation split — to "
                f"bring it back, write a new criterion that does not look "
                f"at validation.")
        return False, ""

    def run(self, n_rounds: int | None = None, *, verbose: bool = True,
            dump_each_round: bool = True):
        """Runs the rounds. ★ **It always saves at the end** (D-33).

        `dump()` used to be the caller's job, and a runner that did not call
        it lost the entire result of 78 minutes and 1,400 calls. The rule
        code lived only in memory, so rescoring was impossible — all that
        remained was the summary on stdout.

        The reason for the `finally` is that **whatever got that far must
        survive even if it dies partway.** Budget overruns, rate limits and
        Ctrl-C all land here. `dump_each_round` overwrites every round and
        is the insurance for long runs (the archive is small, so the cost is
        negligible).
        """
        n = n_rounds or self.cfg.max_rounds
        # ★ The first line must stand on its own (D-133 §3-3) — it holds
        #   the whole config and the commit hash, so the conditions can be
        #   known from the trace alone.
        # ★ 2026-09-10 (D-152): **the output schema goes in whole.** Finding
        #   where a phantom cap of 8 came from took two days because the
        #   field descriptions the model actually receives were not in the
        #   trace — `pydantic-ai` hands them over as the tool schema. With
        #   this, the same investigation is one `jq` away.
        self.trace.ev("run_start", run_id=self.cfg.run_id, n_rounds=n,
                      commit=_git_commit(), config=self._config_dict(),
                      table=str(getattr(self.table, "bundle", "")),
                      split=self.splits.kind,
                      n_train=len(self.splits.train.shapes),
                      n_val=len(self.splits.val.shapes),
                      features=sorted(self.matrix.feature_names()),
                      # ★ Declared range vs the range actually taken on the
                      #   training configs (D-160). The declaration is the
                      #   model's word and this is the table's. ⛔ It is
                      #   recorded, never shown — putting it in a prompt
                      #   makes the run condition B.
                      feature_ranges=self._range_rows(),
                      output_schemas=_output_schemas())
        try:
            for _ in range(n):
                res = self.run_round()
                if verbose:
                    print(res.line(), flush=True)
                # ★ The objective switch (D-104). It is checked **after**
                #   the round ends — so the verdict uses the improvement up
                #   to and including that round.
                self._maybe_switch()
                # ★ If **every** proposal is a transport failure, stop
                #   (D-43). A credit or authentication problem does not heal
                #   by itself — burning the remaining rounds leaves nothing
                #   but an empty archive. 12 rounds really were spent that
                #   way.
                if (STOP_ON_TOTAL_LLM_FAILURE and res.n_proposed
                        and res.n_llm_error == res.n_proposed):
                    raise LLMUnreachable(
                        f"r{res.round}: all {res.n_proposed} proposals are "
                        f"LLM transport failures. Last reason: "
                        + next((m for k, m in reversed(res.rejections)
                                if k == "llm-transport"), "?"))
                if dump_each_round:
                    self.dump()
                stop, why = self.should_stop()
                if stop:
                    if verbose:
                        print(f"early stop: {why}")
                    break
        finally:
            path = self.dump()
            # ★ No workers are left behind. With 12 of them floating
            #   around sharing 4.2GB, the next run's fork grows the memory.
            if self._pool_exec is not None:
                self._pool_exec.shutdown(wait=True)
                self._pool_exec = None
            if verbose:
                print(f"  -> {path}", flush=True)
        return self.rounds

    def _config_dict(self) -> dict:
        """The content of `config.json`. ★ The trace's first line uses
        **this** too — built separately in two places they diverge
        (principle 2, D-133)."""
        from kernelrule.core.splits import is_unsealed

        cfg: dict = {"loop": dict(self.cfg.__dict__),
                     "split": {"kind": self.splits.kind,
                               "n_train": len(self.splits.train.shapes),
                               "n_val": len(self.splits.val.shapes),
                               # ★ Did this run go with the final split
                               #   open (§30.15)? If it did, its numbers are
                               #   **possibly contaminated**.
                               "unsealed": is_unsealed()},
                     "n_features": len(self.matrix.feature_names()),
                     # ★ Axes built **inside** the loop (D-75). Mixed with
                     #   those received from outside, "how many were in the
                     #   library" cannot be retraced.
                     "n_features_made_in_loop": sum(
                         1 for x in self.features_made if x.get("accepted")),
                     # ★ The rule constraints. **They are conditions, so
                     #   they are recorded per run** (D-78). Before and
                     #   after the branch-comparison-constant exemption are
                     #   not the same family.
                     # ★ The objective switch (D-104). **A condition, so it
                     #   is recorded.**
                     "objective": self.cfg.objective,
                     "fit_method": self.cfg.fit_method,
                     "fit_restarts": self.cfg.fit_restarts,
                     "objective_switch": self.cfg.objective_switch,
                     "switch_round": self.switch_round,
                     "final_objective": self._objective,
                     # ★ 2026-09-10 (D-160): when there is no cap this
                     #   used to write `_FITTER_DIM` (8) — the **fitter
                     #   switch dimension**, which is a different quantity.
                     #   "ran with a cap of 8" and "ran with no cap" then
                     #   read the same in `config.json`, and `runs.md`
                     #   tagged both `p8`. A cap that is absent is written
                     #   as absent.
                     #   ⚠️ Old `config.json` files are **not** rewritten —
                     #   back then the cap really was 8. The commit is what
                     #   separates them.
                     "rule_constraints": {
                         "parameters": self.cfg.parameters,
                         "no_parameter_cap": self.cfg.parameters is None,
                         "branch_constants_exempt": True}}
        llm_cfg = getattr(self.llm, "cfg", None)
        if llm_cfg is not None and hasattr(llm_cfg, "to_dict"):
            cfg["llm"] = llm_cfg.to_dict()
        else:                                   # MockLLM and the like
            cfg["llm"] = {"class": type(self.llm).__name__}
        return cfg

    def dump(self, out: str | Path | None = None) -> Path:
        # ★ **What it was run with** is recorded (D-31, D-45, D-51).
        #   Without it there is no way to know later which run used which
        #   model / endpoint / reasoning effort, and then they cannot be
        #   placed side by side.
        d = Path(out or (Path(self.cfg.out_dir) / self.cfg.run_id))
        d.mkdir(parents=True, exist_ok=True)
        cfg = self._config_dict()
        (d / "config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=1, default=str))
        self.archive.dump(d / "archive.jsonl")
        (d / "rounds.jsonl").write_text("\n".join(
            json.dumps(x.__dict__, ensure_ascii=False, default=str)
            for x in self.rounds))
        (d / "failures.jsonl").write_text("\n".join(
            json.dumps(x, ensure_ascii=False) for x in self.failures))
        (d / "hypotheses.jsonl").write_text("\n".join(
            json.dumps(h, ensure_ascii=False) for h in self.hypotheses))
        # ★ Axes built inside a round (D-75). **The refused ones are kept
        #   too** — "what did it try to build and fail" is an observation.
        # ★ The cross lineage (D-96 observation 1). The parent code survives
        #   nowhere else, so it can only be retraced here.
        if self.bests:
            (d / "bests.jsonl").write_text("\n".join(
                json.dumps(x, ensure_ascii=False) for x in self.bests))
        if self.cross_lineage:
            (d / "cross.jsonl").write_text("\n".join(
                json.dumps(x, ensure_ascii=False) for x in self.cross_lineage))
        if self.features_made:
            (d / "features.jsonl").write_text("\n".join(
                json.dumps(x, ensure_ascii=False) for x in self.features_made))
        if hasattr(self.llm, "dump"):
            self.llm.dump(d / "llm_calls")
        return d
