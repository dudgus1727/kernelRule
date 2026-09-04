"""★ 한 라운드의 시간을 구간별로 나눈다 (§5-1). **LLM 호출 0회**.

    python3 experiments/round_profile.py

병렬화를 하기 전에 **무엇이 병목인지** 본다. LLM 대기가 대부분이면
채점·적합을 병렬화해도 이득이 작다.

## 두 축을 따로 잰다

```
LLM 대기   ★ 실제 실행의 `llm_calls/*.json` 에 기록된 `seconds` 로 잰다
           동시성 6 이므로 벽시계는 합/6 에 가깝다
그 밖      ★ MockLLM 으로 한 라운드를 돌려 구간별로 잰다
           정적 검사 / 샌드박스 / fit_weights / 채점
```

**둘을 한 실행에서 재려면 LLM 호출이 필요하다.** 그래서 나눠 재고
합쳐서 읽는다 — 그 사실을 결과에 적는다.
"""

from __future__ import annotations

import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
RUNS = Path("runs")


def llm_share() -> None:
    """실제 실행에서 LLM 대기가 벽시계의 몇 %인가."""
    print("=" * 74)
    print("A. LLM 대기 — 실제 실행 기록에서")
    print("=" * 74)
    print(f"  {'실행':28s} {'라운드':>5} {'벽시계/R':>9} "
          f"{'LLM합/R':>9} {'/동시성6':>9} {'비중':>6}")
    for d in sorted(RUNS.glob("*/rounds.jsonl")):
        run = d.parent
        R = [json.loads(x) for x in d.read_text().splitlines() if x.strip()]
        if not R:
            continue
        secs = [float(json.loads(g.read_text()).get("seconds") or 0.0)
                for g in (run / "llm_calls").glob("*.json")]
        if not any(secs):
            continue
        wall = sum(r["seconds"] for r in R) / len(R)
        llm = sum(secs) / len(R)
        print(f"  {run.name:28s} {len(R):5d} {wall:9.1f} {llm:9.1f} "
              f"{llm / 6:9.1f} {llm / 6 / wall:6.0%}")
    print("  ★ 동시성 6 이므로 '/동시성6' 이 LLM 벽시계의 하한이다")
    print("    (재시도·직렬 구간이 있어 실제는 그보다 크다)")


def non_llm_breakdown() -> None:
    """MockLLM 으로 한 라운드를 돌려 LLM 밖 구간을 잰다."""
    print("\n" + "=" * 74)
    print("B. LLM 밖 — MockLLM 한 라운드, 구간별")
    print("=" * 74)
    warnings.simplefilter("ignore")
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core import loop as loop_mod
    from kernelrule.core import scoring as scoring_mod
    from kernelrule.core import weights as weights_mod
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import Split, SplitSet
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY

    t0 = time.perf_counter()
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)
    print(f"  표 + 피처 행렬 로드 {time.perf_counter() - t0:.1f}s "
          "(라운드마다 다시 하지 않는다)")

    def aligned(p) -> bool:
        x = table.frame_for(p)
        return bool((x.align_a == 8).all() and (x.align_b == 8).all()
                    and (x.align_c == 8).all())

    sh = [p for p in table.shapes() if aligned(p)]
    train = [p for p in sh if 11008 not in (p.N, p.K)]
    val = [p for p in sh if 11008 in (p.N, p.K)]
    splits = SplitSet(train=Split("train", tuple(train)),
                      val=Split("val", tuple(val)))

    acc: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)

    def timed(mod, name, key):
        orig = getattr(mod, name)

        def wrap(*a, **k):
            s = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                acc[key] += time.perf_counter() - s
                cnt[key] += 1
        setattr(mod, name, wrap)
        return orig

    olds = [(loop_mod, "check_rule", timed(loop_mod, "check_rule", "정적 검사")),
            (loop_mod, "fit_weights", timed(loop_mod, "fit_weights",
                                            "fit_weights")),
            (loop_mod, "run_isolated", timed(loop_mod, "run_isolated",
                                             "샌드박스")),
            (loop_mod, "evaluate_scores",
             timed(loop_mod, "evaluate_scores", "채점")),
            (loop_mod, "build_report", timed(loop_mod, "build_report",
                                             "진단 리포트"))]
    _ = (scoring_mod, weights_mod)

    from kernelrule.agents.mock import MockLLM
    from kernelrule.core.loop import LoopConfig, RoundLoop

    seed = json.loads(
        (RUNS / "F3rw-p8" / "stage2-rule-writer"
         / "chosen.json").read_text())
    cfg = LoopConfig(run_id="profile", n_rules_per_round=12, max_rounds=1,
                     seed=0, out_dir="/tmp")
    llm = MockLLM("mutate", seed=1, feature_names=matrix.feature_names(),
                  shape_values=matrix.shape_value_names())
    lp = RoundLoop(cfg=cfg, table=table, matrix=matrix, splits=splits, llm=llm)
    lp.seed(seed["code"], seed["w0"])
    # ★ 씨앗 채점도 `fit_weights` 를 부른다 — 안 지우면 라운드 시간의
    #   118% 가 나온다 (실제로 그렇게 찍혔다).
    acc.clear()
    cnt.clear()
    t0 = time.perf_counter()
    r = lp.run_round()
    wall = time.perf_counter() - t0
    for mod, name, orig in olds:
        setattr(mod, name, orig)

    print(f"\n  라운드 벽시계 {wall:.1f}s   제안 {r.n_proposed} 채점 {r.n_scored}")
    print(f"  {'구간':14s} {'초':>8} {'비중':>6} {'호출':>6}")
    for k in ("진단 리포트", "정적 검사", "샌드박스", "fit_weights", "채점"):
        print(f"  {k:14s} {acc[k]:8.1f} {acc[k] / wall:6.0%} {cnt[k]:6d}")
    other = wall - sum(acc[k] for k in acc)
    print(f"  {'그 밖':14s} {other:8.1f} {other / wall:6.0%}")
    if cnt["fit_weights"]:
        print(f"\n  후보당 fit_weights {acc['fit_weights'] / cnt['fit_weights']:.1f}s"
              f"  (채점된 후보 {cnt['fit_weights']}개)")
        print(f"  ★ 12개가 다 채점되면 "
              f"{acc['fit_weights'] / cnt['fit_weights'] * 12:.0f}s 가 된다 — "
              "중복이 많을수록 이 구간이 싸 보인다")
    print("\n  ★ MockLLM 이라 LLM 대기가 0 이다. 실제 라운드에서는 A 의 "
          "비중만큼 이 값들이 희석된다")


def main() -> None:
    llm_share()
    non_llm_breakdown()
    parallel_speedup()


if __name__ == "__main__":
    main()


def parallel_speedup(workers: int = 6) -> None:
    """★ 같은 라운드를 순차/병렬로 돌려 **시간과 값**을 비교한다 (D-95).

    시간만 보면 안 된다 — 값이 같아야 이득이다 (원칙 29).
    """
    print("\n" + "=" * 74)
    print(f"C. 병렬화 — 순차 vs {workers} 워커 (MockLLM, 같은 시드)")
    print("=" * 74)
    warnings.simplefilter("ignore")
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.mock import MockLLM
    from kernelrule.core.loop import LoopConfig, RoundLoop
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.splits import Split, SplitSet
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        x = table.frame_for(p)
        return bool((x.align_a == 8).all() and (x.align_b == 8).all()
                    and (x.align_c == 8).all())

    sh = [p for p in table.shapes() if aligned(p)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in sh if 11008 not in (p.N, p.K))),
        val=Split("val", tuple(p for p in sh if 11008 in (p.N, p.K))))
    seed = json.loads(
        (RUNS / "F3rw-p8" / "stage2-rule-writer"
         / "chosen.json").read_text())

    out = {}
    for n in (0, workers):
        llm = MockLLM("mutate", seed=1, feature_names=matrix.feature_names(),
                      shape_values=matrix.shape_value_names())
        lp = RoundLoop(
            cfg=LoopConfig(run_id=f"prof{n}", n_rules_per_round=12,
                           max_rounds=1, seed=0, out_dir="/tmp", n_workers=n),
            table=table, matrix=matrix, splits=splits, llm=llm)
        lp.seed(seed["code"], seed["w0"])
        t0 = time.perf_counter()
        r = lp.run_round()
        el = sorted(lp.archive.cells.values(), key=lambda e: e.rule_id)
        out[n] = (time.perf_counter() - t0, r.n_scored,
                  [(e.rule_id, round(e.regret, 12), tuple(e.w)) for e in el])
        if lp._pool_exec is not None:
            lp._pool_exec.shutdown(wait=True)
    (ts, ns, es), (tp, np_, ep) = out[0], out[workers]
    print(f"  순차   {ts:6.1f}s  채점 {ns}")
    print(f"  병렬   {tp:6.1f}s  채점 {np_}   ★ {ts / tp:.1f}배")
    print(f"  ★ 값이 같은가: {'예' if es == ep else '★아니오 — 숨은 상태가 있다'}")
    print("  ⚠️ 빠른 것이 좋은 것이 아니다 — 값이 같을 때만 이득이다 (원칙 29)")
