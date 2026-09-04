"""★ 재실행 — 단일 조건 6시드. 실험 계획서에 적힌 기준으로만 판정한다.

    python3 experiments/rerun.py --verify     # 짧은 검증 (2실행 x 6라운드)
    python3 experiments/rerun.py              # 본 실행 (6실행 x 12라운드)
    python3 experiments/rerun.py --score-only # 이미 돈 것을 채점만

## 왜 재실행하나

12실행 전부가 **절반이 적합 없이 채점된** 상태에서 진화했다 (D-54).
채점이 틀렸으면 그 위의 선택 — 부모 선택, 아카이브 갱신, 조기 종료 —
이 전부 틀렸다. 다듬기 전후로 규칙 순위도 바뀐다 (D-57).

    재적합은 최종 산출물만 고친다. 진화 궤적은 못 되돌린다.  (원칙 13)

## 실험 계획서

**판정 기준은 `PREREG` 에 있고 `docs/artifacts/rerun-preregistration.md`
와 같은 내용이다.** 테스트가 그것을 고정한다. 결과를 보고 기준을 바꾸면
오염이다 (D-50).
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401  — REGISTRY 를 채운다
from kernelrule.agents.openai_client import Budget, LLMConfig, OpenAILLM
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet, check_balance
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
OUT = Path("runs")

#: ★ 실험 계획서. `docs/artifacts/rerun-preregistration.md` 와 **같은 내용**이다.
#:   `tests/test_rerun_prereg.py` 가 둘이 안 달라지는지 검사한다.
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

#: 비용 상한. 기존 luna 6실행의 **실행당 최댓값 x 6 x 1.5 여유**다.
#: 크레딧 소진을 두 번 겪었다 (D-43) — 넘으면 예외로 멈춘다.
BUDGET = {"max_calls": 1395, "max_input_tokens": 12_140_928,
          "max_output_tokens": 2_157_264}

#: 3실행 연속 아카이브가 비면 멈춘다. 인프라가 죽은 것이지 실험이 아니다.
MAX_EMPTY_STREAK = 3


class Terminated(KeyboardInterrupt):
    """SIGTERM 을 예외로 — 안 그러면 `finally` 가 안 돈다 (원칙 17)."""


def _install_signal_handlers() -> None:
    import contextlib

    def _die(signum, _frame):
        raise Terminated(f"신호 {signal.Signals(signum).name}")

    # ★ SIGTERM 만. SIGHUP 은 백그라운드 분리 신호라 잡으면 안 된다.
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
                    help="짧은 검증 — 2실행 x 6라운드. ★ 대표값 수치가 아니다")
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
    print(f"재실행 — 단일 조건 {n_seeds}시드 x {rounds}라운드"
          + ("   ★ 검증 실행 (대표값 수치 아님)" if a.verify else ""))
    print("=" * 78)
    print(f"  목적: {PREREG['purpose']}")
    print(f"  예상: {PREREG['expected']}")
    print(f"  학습 {len(splits.train.shapes)} / 구조 홀드아웃 "
          f"{len(splits.val.shapes)}")
    print(f"  비용 상한: 호출 {BUDGET['max_calls']} / 입력 "
          f"{BUDGET['max_input_tokens']:,} / 출력 "
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
                print(f"  [{run_id}] 이미 있다. 건너뛴다")
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
                print(f"  ★ 중단: {type(e).__name__}: {str(e)[:120]}")
            arc = OUT / run_id / "archive.jsonl"
            n = sum(1 for ln in arc.open() if ln.strip()) if arc.exists() else 0
            empty_streak = 0 if n else empty_streak + 1
            print(f"  아카이브 {n}개  누적 호출 {budget.calls}  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
            if empty_streak >= MAX_EMPTY_STREAK:
                print(f"\n  ★ {MAX_EMPTY_STREAK}실행 연속 아카이브가 비었다. "
                      "인프라 문제다 — 멈춘다 (실험 계획서).")
                break

    _score(table, matrix, splits, run_ids, d, verify=a.verify)


def _dump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def _score(table, matrix, splits, run_ids, d: Path, *, verify: bool) -> None:
    """실험 계획서에 적힌 지표만 낸다. **삭제한 값과 비교하지 않는다.**"""
    from kernelrule.core.canonical import canonical_score

    print("\n" + "=" * 78)
    print("채점 — 실험 계획서에 적힌 지표만" + ("  ★ 검증 실행" if verify else ""))
    print("=" * 78)

    done, missing = [], []
    for run_id in run_ids:
        arc = OUT / run_id / "archive.jsonl"
        if not arc.exists():
            missing.append(run_id)
            continue
        rows = [json.loads(ln) for ln in arc.open() if ln.strip()]
        if not rows:
            # ★ 빈 아카이브는 "나쁜 실행" 이 아니라 **실행이 안 된 것**이다.
            #   0 으로 채워 넣으면 분포가 오염된다 (§26.4).
            print(f"  {run_id:16s} ⚠️ 아카이브가 비었다 — 채점에서 제외")
            missing.append(run_id)
            continue
        best = min(rows, key=lambda e: e["regret"])
        cs = canonical_score(best["code"], best["w"], table=table,
                             matrix=matrix, splits=splits)
        done.append((run_id, cs))
        warn = f"  ★{len(cs.warnings)}건" if cs.warnings else ""
        print(f"  {run_id:16s} 홀드아웃 {cs.holdout:.4f}{warn}")

    if missing:
        # ★ 부분 완주. **설계 규모를 명시한다** — 시드를 골라 쓰지 않는다.
        print(f"\n  ★ 설계는 {len(run_ids)}시드였고 {len(done)}개가 끝났다. "
              f"미완: {missing}")
    if not done:
        print("  채점할 실행이 없다.")
        return

    ho = np.array([c.holdout for _, c in done])
    q1, med, q3 = np.percentile(ho, [25, 50, 75])
    print(f"\n  구조 홀드아웃  중앙 {med:.4f}  사분위 [{q1:.4f}, {q3:.4f}]  "
          f"n={len(ho)}")
    _dump(d / "scores.json", {
        "runs": {r: {"holdout": c.holdout, "in_sample": c.in_sample,
                     "by_regime": c.by_regime, "warnings": list(c.warnings)}
                 for r, c in done},
        "median": float(med), "q1": float(q1), "q3": float(q3),
        "n_done": len(done), "n_designed": len(run_ids), "missing": missing,
        "verify": verify,
        "note": ("★ 검증 실행이다 — 대표값 수치가 아니다" if verify else
                 "이 저장소의 대표값 성능 수치")})
    print(f"\n  기록: {d / 'scores.json'}")
    if verify:
        print("  ★ 검증 실행 수치는 대표값이 아니다. 본 실행과 합치지 않는다.")


if __name__ == "__main__":
    main()
