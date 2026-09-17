"""★ Critic — it judges a rule **term by term**. ★ **It is outside the loop**
(D-92).

Both pass conditions (D-85 "does it not get worse when removed", D-87 "does
only the training get worse when removed") failed, and **the mock Critic
scored higher than the real one** (0.79 vs 0.50 / 0.36). Explainability does
not attach to performance, so **there is no ground for turning the direction
of evolution with a penalty, and if it does not turn the direction there is
no reason for it to be in the loop.**

So **this file holds** the prompt and the schema. `kernelrule/agents/` knows
only the four the loop calls (`register_role`, D-92).

Two of the three uses that were kept are here.

```
(a) identity-transform check -> moved into `rules/checks.py` (D-92). Not here
(b) after-the-run commentary -> `judge`. ★ Called **on one final rule only**
(c) transfer verification    -> `gate --other <5090>` (when the table comes)
```


```
python3 experiments/critic.py judge  --runs F3rw-p8-s0 ...   # LLM
python3 experiments/critic.py ablate --judged docs/artifacts/critic.json  # 0 LLM calls
python3 experiments/critic.py rank   --campaign runs/F3rw-p8     # LLM
```

## Why it exists

"an interpretable rule" is this study's claim, but right now that claim can
only be confirmed by **a human reading the code directly**. Attaching a
physical explanation per term is what makes the difference from GBDT visible.

## ★ The judgement is measured by **accuracy**, not by performance

`ablate` is the pass condition. The terms the Critic said it "cannot
explain" are removed and refitted.

```
regret does not get worse when removed   ★ the judgement was right
it gets much worse                       it was wrong, or the term is useful
                                         even though it cannot be explained
```

**★ And the other terms are removed one at a time and compared.** "it is
fine even with the unexplainable term removed" says nothing at all **if that
holds for every term**. The metric is **the damage rank of the unexplainable
term** — if the judgement is accurate it should be down at the bottom (the
less painful end).

The absolute value of regret is not reported (D-56 §2). Only differences and
ranks are used.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path


#: ⛔ 2026-09-17 (D-179) — the SOL lower-bound value this script used was
#: **removed from the code**. The 0.5 ms boundary was ours, chosen by looking
#: at the a6000 table, and the scoring path used it to fit two weight vectors
#: while the loop evolved one. See `docs/decisions.md` D-179.
#:
#: ⚠️ This script is kept because the numbers it produced are on the record
#: and this is how they were produced. ⛔ It **cannot be re-run** — and it
#: says so rather than quietly substituting another cut, which would put
#: different numbers under the same name.
def _sol_removed(where: str):
    raise SystemExit(
        f"{where}: the SOL lower-bound value was removed at D-179. This "
        f"script cannot be re-run. Its recorded numbers stay in "
        f"docs/decisions.md; ⛔ do not substitute another boundary.")


BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"


def _table_and_matrix():
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    return table, FeatureMatrix(table, REGISTRY), REGISTRY


def _train_groups(table):
    from kernelrule.core.splits import experiment_shapes, regime_of

    shapes = experiment_shapes(table)
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    return {n: [p for p in train if regime_of(p, table.hw) == n]
            for n in ("short", "long")}


def _best_rule(run: str) -> dict:
    """★ **One final rule per run**. Once per seed (D-92).

    It is not called every round inside the loop — that would turn the
    direction, and the two pass conditions could not give a ground for that.
    And there is the advantage that **the context is whole**, because the
    rule is finished.
    """
    arc = [json.loads(ln) for ln in
           (Path("runs") / run / "archive.jsonl").read_text().splitlines()
           if ln.strip()]
    return min(arc, key=lambda e: e["regret"])


# ---------------------------------------------------------------- judge (LLM)
def _register_critic(llm) -> None:
    """★ The roles outside the loop are registered here (D-92).

    Both the prompt and the schema are kept beside this file — leaving them
    in `kernelrule/agents/` reads as "something to switch on some day" and
    drags them into the condition list and the ablation table.
    """
    from pydantic import BaseModel, Field

    class TermOut(BaseModel):
        index: int = Field(description="the index of the w multiplied into "
                                       "this term")
        expression: str = Field(description="that term's expression "
                                            "(excluding w[i])")
        physics: str = Field(
            description="**one sentence** on the physical quantity this term "
                        "measures and the mechanism by which it governs "
                        "performance. 'bigger is better' is not a mechanism")
        explainable: bool = Field(
            description="can it be explained by a physical mechanism? ★ There "
                        "are terms for which False is the right answer — do "
                        "not make one up by force")
        why_not: str = Field(default="",
                             description="if it cannot, why")
        regime_dependent: bool = Field(default=False)
        regime: str = Field(default="")

    class CritiqueOut(BaseModel):
        terms: list[TermOut] = Field(
            description="one per term that has a w[i] multiplied in. Do not "
                        "leave any out")
        overall: str = Field(description="what the rule as a whole does. One "
                                         "paragraph")
        defects: list[str] = Field(default_factory=list)

    body = (Path(__file__).parent / "prompts" / "critique.md").read_text()
    llm.register_role("critique", instructions=body, output_type=CritiqueOut)


def _critic_user_prompt(code: str, registry) -> str:
    """The rule code + **only the physical quantities that were used**.

    What is not given, and why:

    ```
    score / cases / parent  knowing the context it was made in pulls the
                            judgement that way
    the weight values       they are values fitted to the table, so it turns
                            into "the table chose them, so they must be right"
    hardware constants      it asks "does it have a physical meaning", not
                            "does it suit an A6000"
    the whole library       suggestions of "use what was not used" get mixed
                            in — that is not the Critic's job
    ```
    """
    import re

    from kernelrule.features import render_features

    if not code.strip():
        raise ValueError("critique must be given the rule code")
    used = set(re.findall(r"\b[fp]\.(\w+)", code))
    sub = type(registry)(f"{registry.name}-used")
    for n in sorted(used & set(registry._items)):
        sub.add(registry[n])
    block = (render_features(sub, include_observed=False) if sub._items
             else "(no feature list)")
    return ("## The rule function\n\n```python\n" + code.strip()
            + "\n```\n\n## The physical quantities used\n\n" + block + "\n")


def cmd_judge(a) -> None:
    import numpy as np

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import Budget, LLMConfig, OpenAILLM
    from kernelrule.features import REGISTRY
    from kernelrule.rules.ablate import reorder_terms, term_exprs

    llm = OpenAILLM(LLMConfig(),
                    feature_names=REGISTRY.names(shape_level=False),
                    shape_values=REGISTRY.names(shape_level=True),
                    registry=REGISTRY, cache=False,
                    budget=Budget(max_calls=len(a.runs) * 3))
    _register_critic(llm)
    # ★ Shuffling the order (D-86). It changes the term order and renumbers
    #   the `w` indices — it looks at whether the Critic pointing at "the last
    #   term" is **a position bias**. The expressions themselves are
    #   unchanged, so pointing at the same expression means it is not a bias.
    rng = np.random.default_rng(a.shuffle) if a.shuffle is not None else None
    out = []
    for run in a.runs:
        best = _best_rule(run)
        code, order = best["code"], None
        if rng is not None:
            exprs = term_exprs(code)
            order = [int(i) for i in rng.permutation(sorted(exprs))]
            code = reorder_terms(code, order)
            print(f"  {run}  order {order}")
        res = llm.complete("critique", _critic_user_prompt(code, REGISTRY))
        n_un = sum(1 for t in res["terms"] if not t.get("explainable", True))
        print(f"  {run:32s} terms {len(res['terms']):2d}  "
              f"unexplainable {n_un}")
        for t in res["terms"]:
            mark = "   " if t.get("explainable", True) else "★UNEXPLAINABLE"
            print(f"     w[{t['index']}] {mark} {t.get('physics','')[:70]}")
        out.append({"run": run, "code": code, "critique": res,
                    "shuffle_order": order,
                    "exprs": {str(k): v for k, v in
                              term_exprs(code).items()}})
    # ★ An LLM call cannot be made again (D-33 / D-51). It is always kept.
    llm.dump(Path(a.out).with_suffix("") / "llm_calls")
    Path(a.out).write_text(json.dumps(
        {"_model": llm.cfg.model,
         "_note": "the same model wrote it and judged it — the errors may be "
                  "correlated (D-85)", "rules": out},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}   calls {len(a.runs)}")


# ------------------------------------------------- ablate (0 LLM calls)
def cmd_ablate(a) -> None:
    import numpy as np

    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.splits import Split
    from kernelrule.core.weights import fit_weights
    from kernelrule.rules.ablate import AblateError, drop_terms, term_indices

    warnings.simplefilter("ignore")
    d = json.loads(Path(a.judged).read_text())
    table, matrix, _reg = _table_and_matrix()
    groups = _train_groups(table)

    def regret_of(code: str, w0) -> float:
        """It fits per regime and combines — the same axis as the final
        scoring procedure."""
        fn = compile_rule(code)
        tot = n = 0.0
        for _name, g in groups.items():
            fr = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                             w0, max_evals=300, warn_invariants=False,
                          objective="regret")
            tot += fr.fit_regret * len(g)
            n += len(g)
        return tot / n

    rows = []
    for item in d["rules"]:
        run, code = item["run"], item["code"]
        crit = item["critique"]
        flagged = {t["index"] for t in crit["terms"]
                   if not t.get("explainable", True)}
        idx = term_indices(code)
        base = regret_of(code, [1.0] * len(idx))
        deltas: dict[int, float] = {}
        refused: dict[int, str] = {}
        for i in idx:
            try:
                cut = drop_terms(code, {i})
            except AblateError as e:
                refused[i] = str(e)
                continue
            try:
                deltas[i] = regret_of(cut, [1.0] * (len(idx) - 1)) - base
            except Exception as e:                          # noqa: BLE001
                refused[i] = f"{type(e).__name__}: {e}"[:80]

        order = sorted(deltas, key=lambda i: deltas[i])   # least painful first
        ranks = {i: k for k, i in enumerate(order)}
        print(f"\n  {run}   terms {len(idx)}  "
              f"unexplainable {sorted(flagged)}"
              + (f"  not removable {sorted(refused)}" if refused else ""))
        for i in order:
            m = "★UNEXPLAINABLE" if i in flagged else "              "
            print(f"     w[{i}] {m} damage {deltas[i]:+.4f}  "
                  f"rank {ranks[i]}/"
                  f"{len(order) - 1}")
        for i, why in refused.items():
            print(f"     w[{i}] — not removable: {why[:60]}")
        rows.append(dict(run=run, n_terms=len(idx), flagged=sorted(flagged),
                         deltas={str(k): v for k, v in deltas.items()},
                         refused={str(k): v for k, v in refused.items()},
                         ranks={str(k): v for k, v in ranks.items()}))

    # ★ The metric: do the damage ranks of the unexplainable terms gather at
    #   the bottom?
    fr_ranks, other_ranks = [], []
    for r in rows:
        n = len(r["ranks"])
        if n < 2:
            continue
        for k, v in r["ranks"].items():
            (fr_ranks if int(k) in r["flagged"] else other_ranks).append(
                v / (n - 1))
    print("\n" + "=" * 70)
    if fr_ranks:
        print(f"  relative damage rank of the unexplainable terms "
              f"(0=least painful)  "
              f"median {np.median(fr_ranks):.2f}  n={len(fr_ranks)}")
        print(f"  the remaining terms                               "
              f"median {np.median(other_ranks):.2f}  n={len(other_ranks)}")
        from scipy.stats import mannwhitneyu
        p = mannwhitneyu(fr_ranks, other_ranks, alternative="less").pvalue
        print(f"  Mann-Whitney (unexplainable < the rest)  p = {p:.4f}")
        print("  ★ the sample unit is **a term** and the terms of one rule "
              "are not independent (principle 28) — if the number of rules "
              "is small, do not trust this p")
    else:
        print("  ★ no term was pointed at as unexplainable — the Critic "
              "explained them all")
    Path(a.out).write_text(json.dumps(
        {"_note": "the absolute value of regret is not a reporting target "
                  "(D-56 §2). Only the differences are used",
         "rules": rows}, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


# ----------------------------------------------------------- rank (LLM)
def cmd_rank(a) -> None:
    """It judges the RuleWriter candidates and looks at whether **the order
    comes out different from the training score**.

    ★ Right now the seed is chosen by the training regret alone. What the
    4th campaign (D-84) showed is that the seed's **feature diversity**
    governs what comes downstream, and the training score cannot see that.
    **It is not changed** — only how different the order is gets recorded.
    """
    from scipy.stats import spearmanr

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import Budget, LLMConfig, OpenAILLM
    from kernelrule.features import REGISTRY

    # ★ The candidates have the code and the training score together in
    #   `tries` in `summary.json`. The `candidates/` directory is a `.py`
    #   copy, so it has no score.
    summ = Path(a.campaign) / "stage2-rule-writer" / "summary.json"
    tries = json.loads(summ.read_text())["tries"]
    cands = [t for t in tries if t.get("ok") and t.get("code")
             and t.get("fit_regret") is not None]
    if not cands:
        raise SystemExit(f"there is no scored candidate in {summ}")

    llm = OpenAILLM(LLMConfig(),
                    feature_names=REGISTRY.names(shape_level=False),
                    shape_values=REGISTRY.names(shape_level=True),
                    registry=REGISTRY, cache=False,
                    budget=Budget(max_calls=len(cands) * 2))
    _register_critic(llm)
    rows = []
    for c in cands:
        res = llm.complete("critique",
                           _critic_user_prompt(c["code"], REGISTRY))
        n_ok = sum(1 for t in res["terms"] if t.get("explainable", True))
        rows.append(dict(name=f"try{c['i']:02d}", train=c["fit_regret"],
                         n_terms=len(res["terms"]), n_explainable=n_ok,
                         frac=n_ok / max(len(res["terms"]), 1),
                         defects=len(res.get("defects", []))))
        print(f"  {rows[-1]['name']:20s} explainable "
              f"{n_ok}/{len(res['terms'])}")
    llm.dump(Path(a.out).with_suffix("") / "llm_calls")     # D-33
    rho, p = spearmanr([r["train"] for r in rows], [-r["frac"] for r in rows])
    print(f"\n  training-regret rank vs 'explainable fraction' rank  "
          f"rho={rho:+.3f} p={p:.4f}  n={len(rows)}")
    print("  ★ it only records — the seed selection rule is not changed "
          "(§13.4)")
    Path(a.out).write_text(json.dumps(
        {"_model": llm.cfg.model, "rows": rows,
         "spearman": {"rho": rho, "p": p}}, ensure_ascii=False, indent=1))
    print(f"  -> {a.out}")


# ------------------------------------------------------- gate (0 LLM calls)
#: The number of folds the training 41 is split into. It is a round robin in
#: ascending SOL order, so the regimes balance out automatically (3 slow
#: shapes per fold).
N_FOLDS = 4


def _folds(table, matrix, groups):
    """★ A round robin in ascending SOL order. A single 30/11 split is thin
    because it has only 3 slow shapes in validation — with 4 folds the same
    shape comes to validation one time in four."""
    train = [p for g in groups.values() for p in g]
    _sol_removed("critic._folds")
    srt = train
    out = [[] for _ in range(N_FOLDS)]
    for i, p in enumerate(srt):
        out[i % N_FOLDS].append(p)
    return out


def cmd_gate(a) -> None:
    """★ Pass condition 2 — it measures **"does not generalise"**, not
    "is useless" (D-87).

    D-85's pass condition was "if the training score does not get worse when
    it is removed, the judgement was right". **That was a wrong definition**
    — a term can be important and unexplainable at the same time, and that is
    exactly the situation we worry about.

    ```
    an explainable term     it is physics, so it applies to other shapes too
                            -> removing it makes fit and validation get worse
                               by a similar amount
    an unexplainable term   it is tuned to the training shapes
                            -> ★ removing it makes only the fit much worse
                               and validation less so

    metric   (fit damage) - (validation damage).  ★ the larger the positive
             value, the stronger the overfitting signature
    ```

    ⚠️ **The structural holdout 20 is not touched.** The split is made only
    inside the training 41.
    """
    import statistics

    import numpy as np

    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.splits import Split, regime_of
    from kernelrule.core.weights import fit_weights, make_score_of
    from kernelrule.rules.ablate import AblateError, drop_terms, term_indices

    warnings.simplefilter("ignore")
    d = json.loads(Path(a.judged).read_text())
    table, matrix, _reg = _table_and_matrix()
    groups = _train_groups(table)
    folds = _folds(table, matrix, groups)
    print(f"  {N_FOLDS}-fold  " + " ".join(
        f"[{len(f)} shapes, slow "
        f"{sum(1 for p in f if regime_of(p, table.hw) == 'long')}]"
        for f in folds))

    def one_fold(code: str, k: int) -> tuple[float, float]:
        """With fold k as validation, (fit regret, validation regret)."""
        fn = compile_rule(code)
        n_w = len(term_indices(code))
        va = folds[k]
        tr = [p for j, f in enumerate(folds) if j != k for p in f]
        fit_t = fit_n = val_t = val_n = 0.0
        for name in ("short", "long"):
            gtr = [p for p in tr if regime_of(p, table.hw) == name]
            gva = [p for p in va if regime_of(p, table.hw) == name]
            if not gtr:
                continue
            fr = fit_weights(fn, matrix, table, Split("train", tuple(gtr)),
                             [1.0] * n_w, max_evals=300,
                             warn_invariants=False,
                          objective="regret")
            fit_t += fr.fit_regret * len(gtr)
            fit_n += len(gtr)
            if gva:
                sc = make_score_of(fn, matrix, fr.w)
                from kernelrule.core.scoring import evaluate_scores
                ev = evaluate_scores(sc, table, gva, ks=(1,))
                val_t += ev.at(1) * len(gva)
                val_n += len(gva)
        return fit_t / max(fit_n, 1e-9), val_t / max(val_n, 1e-9)

    rows = []
    for item in d["rules"]:
        run, code = item["run"], item["code"]
        crit = item["critique"]
        flagged = sorted({t["index"] for t in crit["terms"]
                          if not t.get("explainable", True)})
        idx = term_indices(code)
        base = [one_fold(code, k) for k in range(N_FOLDS)]
        diffs: dict[int, float] = {}
        refused: dict[int, str] = {}
        for i in idx:
            try:
                cut = drop_terms(code, {i})
            except AblateError as e:
                refused[i] = str(e)
                continue
            ds = []
            for k in range(N_FOLDS):
                cf, cv = one_fold(cut, k)
                ds.append((cf - base[k][0]) - (cv - base[k][1]))
            diffs[i] = float(np.mean(ds))
        rows.append(dict(run=run, flagged=flagged, n_terms=len(idx),
                         diffs={str(k): v for k, v in diffs.items()},
                         refused={str(k): v for k, v in refused.items()}))
        order = sorted(diffs, key=lambda i: -diffs[i])   # largest positive
        print(f"\n  {run}  unexplainable {flagged}")
        for r, i in enumerate(order):
            m = "★UNEXPLAINABLE" if i in flagged else "              "
            print(f"     w[{i}] {m} (fit-val) {diffs[i]:+.4f}  "
                  f"rank {1 - r / max(len(order) - 1, 1):.2f}")

    def score_flagger(pick) -> float | None:
        """Per rule, the median relative rank of the pointed-at terms -> the
        median of those (principle 28)."""
        per = []
        for r in rows:
            ds = {int(k): v for k, v in r["diffs"].items()}
            if len(ds) < 2:
                continue
            order = sorted(ds, key=lambda i: -ds[i])
            rank = {i: 1 - k / (len(order) - 1) for k, i in enumerate(order)}
            f = [rank[i] for i in pick(r, sorted(ds)) if i in rank]
            if f:
                per.append(statistics.median(f))
        return statistics.median(per) if per else None

    real = score_flagger(lambda r, idx: r["flagged"])
    mock1 = score_flagger(lambda r, idx: idx[-1:])
    mockk = score_flagger(lambda r, idx: idx[-len(r["flagged"]):]
                          if r["flagged"] else [])
    rng = np.random.default_rng(20260828)
    rand = statistics.median([
        score_flagger(lambda r, idx: list(rng.choice(
            idx, size=min(len(r["flagged"]), len(idx)), replace=False))
            if r["flagged"] else [])
        for _ in range(200)])
    print("\n" + "=" * 70)
    print("  the (fit-val) relative rank of the pointed-at terms — median "
          "per rule (1=most positive)")
    print(f"    ★ the real Critic         "
          f"{real if real is None else f'{real:.2f}'}")
    print(f"    mock: the last term alone "
          f"{mock1 if mock1 is None else f'{mock1:.2f}'}")
    print(f"    mock: the last k (count matched) "
          f"{mockk if mockk is None else f'{mockk:.2f}'}")
    print(f"    random k (median of 200)  {rand:.2f}")
    print("    the line: >=0.65 the judgement is right / <=0.35 it is wrong "
          "/ in between is indistinguishable")
    print("    ★ if the mock goes over 0.65 then, whatever the real value "
          "is, it is **not judgeable**")
    Path(a.out).write_text(json.dumps(
        {"_note": "the metric is (fit damage)-(validation damage). The larger "
                  "the positive value, the stronger the overfitting "
                  "signature. The absolute value of regret is not a "
                  "reporting target (D-56 §2)",
         "n_folds": N_FOLDS, "rules": rows,
         "score": {"critic": real, "mock_last1": mock1,
                   "mock_lastk": mockk, "random_k": rand}},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


# ------------------------------------------------ compare (0 LLM calls)
def cmd_compare(a) -> None:
    """It compares the original judgement with the **order-shuffled** one
    (D-86 position bias).

    The comparison is by **expression**, not by term number — because
    shuffling changes the numbers.
    """
    import statistics

    from kernelrule.rules.ablate import term_exprs

    def exprs_of(r: dict) -> dict:
        # ★ The first judgement artefact has no `exprs` — it is read again
        #   from the code.
        return r.get("exprs") or {str(k): v
                                  for k, v in term_exprs(r["code"]).items()}

    base = json.loads(Path(a.base).read_text())["rules"]
    shuf = json.loads(Path(a.shuffled).read_text())["rules"]
    by = {r["run"]: r for r in shuf}
    jac, pos_hits, pos_tot = [], 0, 0
    print(f"  {'run':4s} {'orig':>5} {'shuf':>6} {'both':>5} {'jaccard':>7}")
    for b in base:
        sh = by.get(b["run"])
        if sh is None:
            continue
        be, se = exprs_of(b), exprs_of(sh)
        A = {be[str(t["index"])] for t in b["critique"]["terms"]
             if not t.get("explainable", True) and str(t["index"]) in be}
        B = {se[str(t["index"])] for t in sh["critique"]["terms"]
             if not t.get("explainable", True) and str(t["index"]) in se}
        u = A | B
        j = len(A & B) / len(u) if u else 1.0
        jac.append(j)
        n = len(se)
        for t in sh["critique"]["terms"]:
            if not t.get("explainable", True):
                pos_tot += 1
                pos_hits += t["index"] >= n - 2
        print(f"  {b['run'][-2:]:4s} {len(A):5d} {len(B):6d} "
              f"{len(A & B):5d} {j:7.2f}")
    med = statistics.median(jac)
    print(f"\n  ★ median jaccard {med:.2f}  (n={len(jac)} rules)")
    print("     >=0.60 not a bias / <=0.20 a bias / in between, partly a bias")
    if pos_tot:
        print(f"  ★ after shuffling, the rate of pointing at the last two "
              f"positions {pos_hits}/{pos_tot} "
              f"= {pos_hits / pos_tot:.0%}   (originally 8/13 = 62%)")
    Path(a.out).write_text(json.dumps(
        {"jaccard": jac, "jaccard_median": med,
         "late_position_rate": (pos_hits / pos_tot if pos_tot else None),
         "_note": "the comparison is by expression — shuffling changes the "
                  "w numbers"},
        ensure_ascii=False, indent=1))
    print(f"  -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--runs", nargs="+", required=True)
    j.add_argument("--out", default="docs/artifacts/critic.json")
    j.add_argument("--shuffle", type=int, metavar="SEED",
                   help="judge with the term order shuffled (the D-86 "
                        "position-bias check). The expressions stay the same "
                        "and only the w indices are renumbered")
    j.set_defaults(fn=cmd_judge)
    b = sub.add_parser("ablate")
    b.add_argument("--judged", default="docs/artifacts/critic.json")
    b.add_argument("--out", default="docs/artifacts/critic-ablation.json")
    b.set_defaults(fn=cmd_ablate)
    r = sub.add_parser("rank")
    r.add_argument("--campaign", required=True)
    r.add_argument("--out", default="docs/artifacts/critic-rank.json")
    r.set_defaults(fn=cmd_rank)
    c = sub.add_parser("compare")
    c.add_argument("--base", default="docs/artifacts/critic.json")
    c.add_argument("--shuffled", default="docs/artifacts/critic-shuffled.json")
    c.add_argument("--out", default="docs/artifacts/critic-shuffle.json")
    c.set_defaults(fn=cmd_compare)
    g = sub.add_parser("gate")
    g.add_argument("--judged", default="docs/artifacts/critic.json")
    g.add_argument("--out", default="docs/artifacts/critic-gate.json")
    g.set_defaults(fn=cmd_gate)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
