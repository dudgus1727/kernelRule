"""★ 최종 채점 절차 재채점 — 같은 절차 / 같은 분모 / 같은 집계로만 비교한다.

    python3 experiments/rescore_canonical.py

## 왜 필요한가

RuleWriter A/B 는 `nk11008` 구조 분할에서 train41/val20 로 보고했고, 이전
보고들은 61형상 **체제별 재적합**으로 냈다. 두 숫자를 나란히 놓고 "통과 조건
미달" 이라고 쓴 것은 §30.8 이 반복해서 잡아온 패턴이다 — 절차가 다른 두
값을 비교했다.

여기서 **모든 후보를 같은 절차로** 다시 잰다.

    체제 판정   SOL 2분할 (빠른 41 / 느린 20). ★ t_best 를 쓰지 않는다
    가중치      체제마다 따로 적합
    집계        체제별 평가 후 61형상에서 결합
    유의성      형상별 (t_A - t_B) / (t_최적 x noise_floor), 2σ

★ 그리고 **표본 내와 홀드아웃을 함께** 낸다. 체제별 적합은 자유
파라미터가 2배이므로 표본 내 값은 반드시 좋아진다 (8-2 에서 확인했다).
표본 내만 보고하면 그 이득을 실력으로 오해한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import compare, evaluate, evaluate_scores, geomean
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.rules.human_guided import CODE as PS
from kernelrule.rules.human_guided import W0 as PS_W0

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
RUNS = Path("runs")


def _best(path: Path, key: str = "train"):
    with path.open() as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    ok = [r for r in rows if "code" in r]
    return min(ok, key=lambda r: r[key]) if ok else None


def main() -> None:
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    fast = [p for p in shapes if regime_of(p, table.hw) == "short"]
    slow = [p for p in shapes if regime_of(p, table.hw) == "long"]

    # ★ 홀드아웃 — 각 체제에서 SOL 순 3개마다 1개. 체제 안에 고르게 퍼진다.
    def thirds(g):
        g = sorted(g, key=lambda p: table.frame_for(p).index[0])
        return ([p for i, p in enumerate(g) if i % 3 != 2],
                [p for i, p in enumerate(g) if i % 3 == 2])

    f_fit, f_ho = thirds(fast)
    s_fit, s_ho = thirds(slow)
    holdout = f_ho + s_ho

    def per_regime(code, w0, fit_groups, eval_groups):
        """체제마다 따로 적합하고, 지정한 형상들을 그 체제 가중치로 채점."""
        reg, tol = {}, {}
        for grp, ev_grp in zip(fit_groups, eval_groups, strict=True):
            fit = fit_weights(compile_rule(code), matrix, table,
                              Split("train", tuple(grp)), w0, max_evals=300,
                          objective="regret")
            so = make_score_of(compile_rule(code), matrix, fit.w)
            e = evaluate_scores(so, table, ev_grp, ks=(1,))
            for i, p in enumerate(e.shapes):
                reg[p], tol[p] = e.regret[i, 0], e.tol[i]
        return reg, tol

    cands: list[tuple[str, str, list]] = [("human_guided", PS, PS_W0)]
    for cond, model in (("A", "gpt-5.4"), ("B", "gpt-5.4"),
                        ("A", "gpt-5.4-mini-2026-03-17")):
        r = _best(RUNS / f"architect-{cond}-{model}" / "tries.jsonl")
        if r:
            tag = "mini" if "mini" in model else "5.4"
            cands.append((f"RuleWriter {cond} ({tag})", r["code"], r["w"]))
    ev_run = RUNS / "real-gpt-5.4-mini-2026-03-17" / "archive.jsonl"
    if ev_run.exists():
        with ev_run.open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        e = min(arc, key=lambda x: x["regret"])
        cands.append(("evolved (첫 실행)", e["code"], e["w"]))

    vendor = load_vendor(VENDOR)
    v_all = evaluate(vendor_order_fn(table, vendor, mapping="nearest"),
                     table, shapes, ks=(1,), label="벤더")
    v_ho = evaluate(vendor_order_fn(table, vendor, mapping="nearest"),
                    table, holdout, ks=(1,), label="벤더")

    print("=" * 78)
    print("최종 채점 절차 재채점 — 체제별(SOL 2분할) 재적합, 61형상 결합")
    print("=" * 78)
    print(f"  빠른 {len(fast)} / 느린 {len(slow)}   "
          f"홀드아웃 {len(holdout)} (체제 안 3개마다 1개)")
    print(f"\n  {'':24s} {'표본내61':>9} {'★홀드아웃':>10} "
          f"{'빠른':>8} {'느린':>8}")

    results = {}
    for name, code, w0 in cands:
        try:
            r_in, t_in = per_regime(code, w0, [fast, slow], [fast, slow])
            r_ho, t_ho = per_regime(code, w0, [f_fit, s_fit], [f_ho, s_ho])
        except Exception as exc:                            # noqa: BLE001
            print(f"  {name:24s} 실패 {type(exc).__name__}: {str(exc)[:40]}")
            continue
        g_in = geomean(np.array([r_in[p] for p in shapes]))
        g_ho = geomean(np.array([r_ho[p] for p in holdout]))
        g_f = geomean(np.array([r_in[p] for p in fast]))
        g_s = geomean(np.array([r_in[p] for p in slow]))
        results[name] = (r_ho, t_ho)
        print(f"  {name:24s} {g_in:9.4f} {g_ho:10.4f} {g_f:8.4f} {g_s:8.4f}")
    print(f"  {'벤더 nearest ★통과 조건':24s} {v_all.at(1):9.4f} "
          f"{v_ho.at(1):10.4f}")

    # -- 유의성. ★ 홀드아웃에서만 판정한다 --------------------------------
    print(f"\n{'=' * 78}")
    print("유의성 — ★ 홀드아웃에서 벤더와 (표본 내로 판정하지 않는다)")
    print("=" * 78)
    from dataclasses import replace
    base = evaluate_scores(make_score_of(compile_rule(PS), matrix,
                                         np.ones(len(PS_W0))),
                           table, holdout, ks=(1,))
    for name, (r_ho, t_ho) in results.items():
        ev = replace(base,
                     regret=np.array([r_ho[p] for p in holdout]).reshape(-1, 1),
                     tol=np.array([t_ho[p] for p in holdout]), label=name)
        c = compare(ev, v_ho, table, name_a="A", name_b="벤더")
        print(f"  {name:24s} {c.geo_a:.4f} vs {c.geo_b:.4f}   "
              f"이김 {int(c.a_wins.sum()):2d} / 짐 {int(c.a_loses.sum()):2d}"
              f" / 구분불가 {int(c.tied.sum()):2d}")


if __name__ == "__main__":
    main()
