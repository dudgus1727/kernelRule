"""★ The re-run — one condition, 6 seeds. Judged only by the criteria
written in the pre-registration.

    python3 experiments/rerun.py --verify     # a short verification
                                              # (2 runs x 6 rounds)
    python3 experiments/rerun.py              # the real run (6 runs x 12
                                              # rounds)
    python3 experiments/rerun.py --score-only # only score what already ran

## Why it is re-run

All 12 runs evolved in a state where **half of them were scored without a
fit** (D-54). If the scoring was wrong, then everything chosen on top of it —
parent selection, archive updates, early stopping — was wrong too. The rule
ranking also changes before and after the polish (D-57).

    Refitting fixes only the final artefact. The evolutionary trajectory
    cannot be undone.  (principle 13)

## The pre-registration

**The decision criteria are in `PREREG`, and it is the same content as
`docs/artifacts/rerun-preregistration.md`.** A test pins that down. Changing
the criteria after seeing the results is contamination (D-50).
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401  — it fills REGISTRY
from kernelrule.agents.openai_client import Budget, LLMConfig, OpenAILLM
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet, check_balance
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
OUT = Path("runs")

#: ★ The pre-registration. It is **the same content** as
#: `docs/artifacts/rerun-preregistration.md`.
#:   `tests/test_rerun_prereg.py` checks that the two do not diverge.
#:
#: ⚠️ 2026-09-08 (D-146): **the values stay in Korean.** They are a frozen
#: record mirroring a `docs/` document, and `docs/` is not translated —
#: translating here would break the doc-code equality the test enforces.
PREREG = {
    "purpose": ("오염 없는 상태의 값을 얻는 것. **벤더를 이기는 것이 "
                "아니다.**"),
    "expected": ("★ 벤더와 구분 불가. D-53 계산상 6시드로 가릴 수 있는 "
                 "차이는 0.03 이상이고 현재 추정 격차는 0.02 근처다. "
                 "'구분 불가' 가 나와도 실패가 아니다."),
    "n_seeds": 6,
    "rounds": 12,
    "n_rules_per_round": 12,
    "feature_detail": "full",
    "split_kind": "nk11008",
    "not_doing": ["A/B 비교 — 이미 결론이 났다 (D-53/54)",
                  "삭제한 값과의 비교 — 비교할 대상이 없어야 한다",
                  "시드 골라 쓰기 — 전부 쓰거나 전부 안 쓴다 (D-50)"],
    "primary_metric": ("각 형상에서 6실행의 **중앙값** vs 벤더, 형상 20개 "
                       "부호검정. 분산이 한 번만 든다"),
    "secondary_metric": "실행 6개의 부호검정 (p 하한 0.031)",
    "gate": "도달률(무작위 4000점, regret@1). 단일 조건이라 12/12 예상",
    "on_gate_failure": {
        "1건": "기록하고 진행. ★ 그 실행을 결과에서 빼지 말 것 (D-50)",
        "2건 이상": "멈추고 보고. (나) regret@3 대리 손실 재검토",
        "격차 0.03 초과": "건수와 무관하게 멈춤"},
    "on_partial": ("6시드 중 일부만 끝나면 끝난 것만으로 보고하되 "
                   "'설계는 6시드였다' 를 명시한다. 시드를 고르지 않는다"),
    "abort": ["LLMUnreachable 즉시", "3실행 연속 아카이브가 비면"],
}

#: The cost cap. It is **the per-run maximum of the existing 6 luna runs
#: x 6 x 1.5 headroom**.
#: The credit has been used up twice (D-43) — going over stops with an
#: exception.
BUDGET = {"max_calls": 1395, "max_input_tokens": 12_140_928,
          "max_output_tokens": 2_157_264}

#: Three runs in a row with an empty archive stops it. That is the
#: infrastructure being dead, not the experiment.
MAX_EMPTY_STREAK = 3


class Terminated(KeyboardInterrupt):
    """SIGTERM as an exception — otherwise `finally` does not run
    (principle 17)."""


def _install_signal_handlers() -> None:
    import contextlib

    def _die(signum, _frame):
        raise Terminated(f"signal {signal.Signals(signum).name}")

    # ★ SIGTERM only. SIGHUP is the background-detach signal and must not be
    #   caught.
    with contextlib.suppress(OSError, ValueError):
        signal.signal(signal.SIGTERM, _die)


def _splits(table: PerfTable) -> SplitSet:
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    s = SplitSet(train=Split("train", tuple(p for p in shapes
                                            if p not in held)),
                 val=Split("val", tuple(held)), kind=PREREG["split_kind"])
    check_balance(s.train, table.hw)
    return s


def main() -> None:
    _install_signal_handlers()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="a short verification — 2 runs x 6 rounds. ★ Not a "
                         "representative number")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()

    n_seeds = 2 if a.verify else PREREG["n_seeds"]
    rounds = 6 if a.verify else PREREG["rounds"]
    tag = a.tag or ("verify" if a.verify else "rerun")

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    splits = _splits(table)

    print("=" * 78)
    print(f"the re-run — one condition, {n_seeds} seeds x {rounds} rounds"
          + ("   ★ a verification run (not a representative number)"
             if a.verify else ""))
    print("=" * 78)
    print(f"  purpose: {PREREG['purpose']}")
    print(f"  expected: {PREREG['expected']}")
    print(f"  training {len(splits.train.shapes)} / structural holdout "
          f"{len(splits.val.shapes)}")
    print(f"  the cost cap: calls {BUDGET['max_calls']} / input "
          f"{BUDGET['max_input_tokens']:,} / output "
          f"{BUDGET['max_output_tokens']:,}\n")

    budget = Budget(**BUDGET)
    d = OUT / f"{tag}-summary"
    d.mkdir(parents=True, exist_ok=True)
    run_ids = [f"{tag}-s{s}" for s in range(n_seeds)]
    _dump(d / "prereg.json", {"prereg": PREREG, "budget": BUDGET,
                              "n_seeds": n_seeds, "rounds": rounds,
                              "verify": a.verify, "runs": run_ids,
                              "bundle": BUNDLE})

    t0 = time.perf_counter()
    empty_streak = 0
    if not a.score_only:
        for s, run_id in enumerate(run_ids):
            if (OUT / run_id / "archive.jsonl").exists():
                print(f"  [{run_id}] already there. Skipped")
                continue
            llm = OpenAILLM(
                LLMConfig(concurrency=6,
                          feature_detail=PREREG["feature_detail"]),
                feature_names=matrix.feature_names(),
                shape_values=matrix.shape_value_names(),
                registry=REGISTRY, budget=budget, cache=False)
            loop = RoundLoop(
                cfg=LoopConfig(run_id=run_id, max_rounds=rounds,
                               n_rules_per_round=PREREG["n_rules_per_round"],
                               seed=100 + s),
                table=table, matrix=matrix, splits=splits, llm=llm)
            print(f"\n  --- {run_id} ---", flush=True)
            try:
                loop.run(rounds)
            except Exception as e:                          # noqa: BLE001
                print(f"  ★ stopped: {type(e).__name__}: {str(e)[:120]}")
            arc = OUT / run_id / "archive.jsonl"
            n = sum(1 for ln in arc.open() if ln.strip()) if arc.exists() else 0
            empty_streak = 0 if n else empty_streak + 1
            print(f"  archive {n}  cumulative calls {budget.calls}  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
            if empty_streak >= MAX_EMPTY_STREAK:
                print(f"\n  ★ the archive was empty {MAX_EMPTY_STREAK} runs "
                      "in a row. That is an infrastructure problem — it "
                      "stops (the pre-registration).")
                break

    _score(table, matrix, splits, run_ids, d, verify=a.verify)


def _dump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def _score(table, matrix, splits, run_ids, d: Path, *, verify: bool) -> None:
    """Reports only the metrics written in the pre-registration. **It does
    not compare against a deleted value.**"""
    from kernelrule.core.canonical import canonical_score

    print("\n" + "=" * 78)
    print("scoring — only the metrics written in the pre-registration"
          + ("  ★ a verification run" if verify else ""))
    print("=" * 78)

    done, missing = [], []
    for run_id in run_ids:
        arc = OUT / run_id / "archive.jsonl"
        if not arc.exists():
            missing.append(run_id)
            continue
        rows = [json.loads(ln) for ln in arc.open() if ln.strip()]
        if not rows:
            # ★ An empty archive is not "a bad run" but **a run that did not
            #   happen**. Filling it in as 0 pollutes the distribution
            #   (§26.4).
            print(f"  {run_id:16s} ⚠️ the archive is empty — excluded from "
                  f"the scoring")
            missing.append(run_id)
            continue
        best = min(rows, key=lambda e: e["regret"])
        cs = canonical_score(best["code"], best["w"], table=table,
                             matrix=matrix, splits=splits)
        done.append((run_id, cs))
        warn = f"  ★{len(cs.warnings)}" if cs.warnings else ""
        print(f"  {run_id:16s} holdout {cs.holdout:.4f}{warn}")

    if missing:
        # ★ A partial completion. **The designed scale is stated** — seeds
        #   are not cherry-picked.
        print(f"\n  ★ the design was {len(run_ids)} seeds and {len(done)} "
              f"finished. Unfinished: {missing}")
    if not done:
        print("  there is no run to score.")
        return

    ho = np.array([c.holdout for _, c in done])
    q1, med, q3 = np.percentile(ho, [25, 50, 75])
    print(f"\n  structural holdout  median {med:.4f}  "
          f"quartiles [{q1:.4f}, {q3:.4f}]  "
          f"n={len(ho)}")
    _dump(d / "scores.json", {
        "runs": {r: {"holdout": c.holdout, "in_sample": c.in_sample,
                     "by_regime": c.by_regime, "warnings": list(c.warnings)}
                 for r, c in done},
        "median": float(med), "q1": float(q1), "q3": float(q3),
        "n_done": len(done), "n_designed": len(run_ids), "missing": missing,
        "verify": verify,
        "note": ("★ this is a verification run — not a representative number"
                 if verify else
                 "the representative performance numbers of this repository")})
    print(f"\n  recorded: {d / 'scores.json'}")
    if verify:
        print("  ★ a verification-run number is not representative. It is "
              "not pooled with the real run.")


if __name__ == "__main__":
    main()
