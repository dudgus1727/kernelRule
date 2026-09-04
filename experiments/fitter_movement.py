"""★ 1단계 — 고침이 듣는가. 저장된 12규칙 재적합. LLM 0회.

    python3 experiments/fitter_movement.py

**기준: 도달률 90% 이상** (2026-08-26 승인, D-56). 원래는 "적합 이동률 90%"
였으나 그 지표는 **초기값이 이미 최적일 때도 실패로 센다** — 측정 대상이
아니라 측정 도구가 틀렸다. 이동률은 진단용으로 계속 기록한다.

```
이동률   출발점에서 벗어났는가              과정
도달률   손 닿는 곳보다 나쁜 데서 끝났는가   결과   <- "제 일을 했는가"
```

**도달률을 "적합기가 최적을 찾는다" 로 읽지 마라.** 8차원에 무작위 4000점은
성기고, 로그균등 0.05~50 이 실제 최적을 덮는다는 보장도 없다. 정확한 서술은
"적합기가 무작위 4000점 탐색보다 나쁜 곳에서 끝나는 경우가 N% 다" 이다.

regret 의 **절대값은 보고하지 않는다.** 폐기 대상 수치다 (D-56 §2).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.core import weights as W
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
#: 다듬기 예산. 실험 계획서에 적은 비용 상한은 적합 305 의 2배(610)이고 기본값이
#: 600 이다. 인자로 올릴 수 있게 뺐다 — **올려서 잰 값은 기준 통과가
#: 아니다** (D-59). 상한을 넘긴 사실과 함께 보고한다.
POLISH_BUDGET = int(__import__("os").environ.get("KERNELRULE_POLISH_BUDGET",
                                                 "600"))
TARGET_MOVED = 0.90
TARGET_REACH = 0.90
#: 무작위 탐침. 부호는 양수(피처는 전부 '클수록 나쁨'), 크기는 로그균등 1000배.
N_PROBE = 4000
PROBE_LO, PROBE_HI = 0.05, 50.0


def _condition_of(run: str) -> str:
    """이 실행이 어느 조건인가 — 피처 설명을 줬는가.

    `runs/*/config.json` 이 있으면 거기서 읽는다. **luna/lunaNAMES 12실행은
    `dump()` 가 config.json 을 쓰기 전에 돌아서 없다** — 그때는 이름 접두사로
    떨어진다. 추측이므로 그 사실을 화면에 말한다 (§26.4).
    """
    cfg = Path("runs") / run / "config.json"
    if cfg.exists():
        d = json.loads(cfg.read_text())
        det = (d.get("llm") or {}).get("feature_detail") or d.get("feature_detail")
        if det:
            return "A 설명" if det == "full" else "B 이름만"
    return "B 이름만" if run.startswith("lunaNAMES") else "A 설명"


def main() -> None:
    warnings.simplefilter("ignore")
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    # ★ 기본은 커밋된 12규칙이지만, 임의 실행도 받는다 — 검증 실행에서
    #   통과 조건을 걸려면 그 실행들을 봐야 한다 (D-60).
    import sys
    args = [x for x in sys.argv[1:] if not x.startswith("-")]
    if args:
        index = [{"run": r} for r in args]
        print(f"  대상: 인자로 준 {len(index)}실행 {args}\n")
    else:
        index = json.loads(
            Path("docs/artifacts/rules/index.json").read_text())

    print("=" * 76)
    print(f"1단계 — 적합기 품질. 통과선: ★도달률 {TARGET_REACH:.0%} (D-56 승인)")
    if POLISH_BUDGET != 600:
        print(f"  ★ 다듬기 예산 {POLISH_BUDGET} — 실험 계획서 상한 610 을 "
              "넘겼다면 이 결과는 **기준 통과가 아니다** (D-59)")
    print("=" * 76)
    print(f"  {'규칙':16s} {'체제':6s} {'끔':>6} {'켬':>6} "
          f"{'무작위가 이김':>13} {'도달':>6}")

    n = n_off = n_on = n_probe_beats = n_reach = 0
    gaps: list[float] = []
    #: 조건 -> [도달 수, 전체]. ★ 실패가 한 조건에 몰려 있는지가 재실행
    #: 설계를 바꾼다 (D-60) — 몰려 있으면 그 조건을 안 쓰면 된다.
    by_cond: dict[str, list[int]] = {}
    named = any((Path("runs") / r["run"] / "config.json").exists()
                for r in index)
    for row in index:
        run = row["run"]
        with (Path("runs") / run / "archive.jsonl").open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        best = min(arc, key=lambda e: e["regret"])
        fn = compile_rule(best["code"])
        for name in ("short", "long"):
            g = [p for p in train if regime_of(p, table.hw) == name]
            sp = Split("train", tuple(g))
            # ★ `polish` 를 **명시한다.** 기본값이 True 로 바뀐 뒤 이 줄이
            #   생략돼 있어서 "다듬기 끔" 열이 실제로는 켬이었고, 두 열이
            #   똑같은 62.5% 로 나왔다 (원칙 1 — 조용히 한쪽으로 판정).
            off = fit_weights(fn, matrix, table, sp, best["w"], max_evals=300,
                              warn_invariants=False, polish=False,
                          objective="regret")
            on = fit_weights(fn, matrix, table, sp, best["w"], max_evals=300,
                             warn_invariants=False, polish=True,
                             polish_budget=POLISH_BUDGET,
                          objective="regret")
            prob = W._Problem(matrix, table, tuple(g), 1)
            rng = np.random.default_rng(7)
            bv = np.inf
            w0 = np.asarray(best["w"], dtype=np.float64)
            for _ in range(N_PROBE):
                c = np.exp(rng.uniform(np.log(PROBE_LO), np.log(PROBE_HI),
                                       size=w0.size))
                v = prob.regret(fn, c)
                if np.isfinite(v) and v < bv:
                    bv = v
            beat = bv < on.fit_regret - 1e-9
            n += 1
            n_off += off.moved
            n_on += on.moved
            n_probe_beats += beat
            if beat:
                gaps.append(on.fit_regret - bv)
            n_reach += not beat
            c = _condition_of(run)
            by_cond.setdefault(c, [0, 0])
            by_cond[c][0] += not beat
            by_cond[c][1] += 1
            print(f"  {run:16s} {name:6s} "
                  f"{'이동' if off.moved else '정지':>6} "
                  f"{'이동' if on.moved else '정지':>6} "
                  f"{('★ ' + format(bv - on.fit_regret, '+.4f')) if beat else '—':>13} "
                  f"{'—' if beat else 'OK':>6}")

    print()
    print(f"  이동률  다듬기 끔 {n_off:2d}/{n} = {n_off / n:5.1%}")
    print(f"  이동률  다듬기 켬 {n_on:2d}/{n} = {n_on / n:5.1%}   (진단용)")
    r = n_reach / n
    print(f"  ★도달률 {n_reach:2d}/{n} = {r:5.1%}   "
          f"{'통과' if r >= TARGET_REACH else '★미달 (통과선 90%)'}")
    print(f"          = 무작위 {N_PROBE}점이 적합 결과를 못 이긴 비율")
    print()
    print("  조건별 도달률 — ★ 실패가 몰려 있는가 (D-60)")
    for c in sorted(by_cond):
        ok, tot = by_cond[c]
        print(f"    {c:12s} {ok:2d}/{tot} = {ok / tot:6.1%}")
    if not named:
        print("    ※ 조건은 **실행 이름 접두사**로 추정했다 — 이 실행들은 "
              "config.json 이 없다")
    if gaps:
        print(f"\n  도달 실패 {len(gaps)}건의 격차: "
              f"{', '.join(f'{g:.4f}' for g in sorted(gaps, reverse=True))}")
        print(f"  최대 {max(gaps):.4f} — "
              + ("무시 가능(<0.005)" if max(gaps) < 0.005 else
                 "★ 적합기가 실제로 못 찾고 있다. 대안 검토"))


if __name__ == "__main__":
    main()
