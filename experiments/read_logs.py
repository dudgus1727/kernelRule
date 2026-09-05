"""★ 옛 로그로 셋을 센다 — 중복 / 거부 / 죽은 항. LLM 0회 (D-135 §2).

    python3 experiments/read_logs.py

트레이스가 없는 실행에서도 `llm_calls/` + `rounds.jsonl` + `archive.jsonl`
로 셀 수 있는 것만 센다. **셀 수 없는 것은 그렇게 적는다** — 그것이
트레이스가 필요한 자리다.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

from sigma_5090 import _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights
from kernelrule.features import REGISTRY

A6000 = ("datasets/rtx-a6000-sm_86-c63710df", "c63710df")
_BIG = 1e9   # ★ `bounds` 의 "안 묶음" 쪽. `_proj` 가 클립만 하므로 넉넉히
RUNS = [f"F3rw-p8-s{i}" for i in range(6)]


def _rows(run: str, name: str) -> list[dict]:
    p = Path("runs") / run / name
    return ([json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            if p.exists() else [])


def _proposals(run: str) -> list[dict]:
    """`llm_calls` 에서 RuleEditor 응답을 순서대로."""
    out = []
    for f in sorted((Path("runs") / run / "llm_calls").glob(
            "*-rule_editor.json")):
        j = json.loads(f.read_text())
        r = j.get("response")
        if isinstance(r, dict) and r.get("code"):
            out.append({"seq": j.get("seq"), "code": r["code"],
                        "changes": r.get("changes", "")})
    return out


def dup_report() -> None:
    print("=" * 92)
    print("1. 중복 — 이미 있는 코드를 다시 낸다")
    print("=" * 92)
    print(f"  {'실행':16s} {'제안':>5s} {'중복':>5s} {'비율':>7s}   부모 종류별 중복")
    tot = Counter()
    for run in RUNS:
        rs = _rows(run, "rounds.jsonl")
        by: Counter = Counter()
        n = d = 0
        for x in rs:
            for k, v in (x.get("by_parent_kind") or {}).items():
                by[k] += v.get("dup", 0)
                n += v.get("n", 0)
                d += v.get("dup", 0)
        tot.update(by)
        print(f"  {run:16s} {n:5d} {d:5d} {d / max(1, n):7.1%}   {dict(by)}")
    print(f"\n  ★ 합계 부모 종류별 중복: {dict(tot)}")
    # 같은 코드가 몇 번 반복되나 — 응답에서 직접 센다
    print(f"\n  {'실행':16s} {'같은 코드가 2회 이상 나온 횟수':>28s}  최다 반복")
    for run in RUNS:
        c = Counter(p["code"].strip() for p in _proposals(run))
        rep = [v for v in c.values() if v > 1]
        print(f"  {run:16s} {sum(rep) - len(rep):28d}  {max(c.values())}회")
    print("\n  ⚠️ **어떤 가설이 배정됐을 때 중복이 나오나는 옛 로그로 못 센다** —")
    print("     제안별 가설 배정이 채택된 규칙에만 남는다. 트레이스가 그 자리다")


def reject_report() -> None:
    print("\n" + "=" * 92)
    print("2. 거부 — 사유와 분포")
    print("=" * 92)
    keys = ("n_rejected_schema", "n_rejected_static", "n_rejected_sandbox",
            "n_rejected_fit", "n_llm_error")
    print(f"  {'실행':16s} " + " ".join(f"{k[2:]:>10s}" for k in keys)
          + "   사유 (rejections)")
    why: Counter = Counter()
    per_round: Counter = Counter()
    for run in RUNS:
        rs = _rows(run, "rounds.jsonl")
        tot = {k: sum(x.get(k, 0) for x in rs) for k in keys}
        for x in rs:
            for kind, _detail in (x.get("rejections") or []):
                why[kind] += 1
                per_round[x["round"]] += 1
        det = Counter(k for x in rs for k, _ in (x.get("rejections") or []))
        print(f"  {run:16s} " + " ".join(f"{tot[k]:10d}" for k in keys)
              + f"   {dict(det)}")
    print(f"\n  ★ 사유별 합계: {dict(why)}")
    print("  ★ 라운드별 거부 수: "
          + " ".join(f"r{r}:{n}" for r, n in sorted(per_round.items())))
    if not per_round:
        print("     (거부가 없다)")


def dead_terms_report() -> None:
    print("\n" + "=" * 92)
    print("3. 죽은 항 — 빼도 학습이 안 나빠지는 자리")
    print("=" * 92)
    warnings.simplefilter("ignore")
    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M, sp = FeatureMatrix(T, REGISTRY), _splits(T)
    train = list(sp.train.shapes)
    print(f"  {'실행':16s} {'항':>3s} {'죽은 항':>7s}  자리별 손해 (0 에 묶었을 때 학습 regret 증가)")
    for run in RUNS:
        arc = _rows(run, "archive.jsonl")
        if not arc:
            continue
        best = sorted(arc, key=lambda e: e["regret"])[0]
        fn = compile_rule(best["code"])
        n = len(best["w"])
        base = []
        for nm in ("short", "long"):
            g = [q for q in train if regime_of(q, T.hw) == nm]
            base.append(fit_weights(fn, M, T, Split("train", tuple(g)),
                                    best["w"], max_evals=200,
                                    objective="regret",
                                    warn_invariants=False))
        dead, cost = [], {}
        for i in range(n):
            w = list(best["w"])
            w[i] = 0.0
            # ★ 그 자리를 **0 에 묶는다** (`bounds`). 묶지 않으면 적합기가
            #   0 에서 되돌아 나오므로, 재는 것이 "죽은 항" 이 아니라
            #   "시작점을 흔들어도 같은 점을 찾나" 가 된다 — D-136 의 정정.
            bnd = [(-_BIG, _BIG)] * n
            bnd[i] = (0.0, 0.0)
            ok, worst = True, 0.0
            for j, nm in enumerate(("short", "long")):
                g = [q for q in train if regime_of(q, T.hw) == nm]
                fr = fit_weights(fn, M, T, Split("train", tuple(g)), w,
                                 max_evals=200, objective="regret",
                                 warn_invariants=False, bounds=bnd)
                d = fr.fit_regret - base[j].fit_regret
                worst = max(worst, d)
                if d > 1e-6:
                    ok = False
            cost[i] = worst
            if ok:
                dead.append(i)
        # ⚠️ "언제 들어왔나" 는 안 센다 — 씨앗 규칙이 이미 8자리를 다 쓰므로
        #    `bests.jsonl` 의 코드에서 첫 등장은 전부 r0 이고, 아무것도
        #    말해주지 않는다 (D-136). 대신 **자리마다 묶었을 때의 손해**를 적는다.
        print(f"  {run:16s} {n:3d} {len(dead):7d}  " + " ".join(
            f"{'★' if i in dead else ' '}w{i}:{cost[i]:+.4f}"
            for i in range(n)))
    print("\n  ★ 는 죽은 자리 (0 에 묶어도 두 체제 다 안 나빠진다)."
          " 숫자는 묶었을 때 학습 regret 이 나빠지는 폭")
    print("  ⚠️ '들어온 뒤 지워질 뻔했나' 는 옛 로그로 못 센다 — 제안의")
    print("     changes 와 부모가 제안 단위로 안 남는다. 트레이스가 그 자리다")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dead", action="store_true")
    a = ap.parse_args()
    dup_report()
    reject_report()
    if not a.skip_dead:
        dead_terms_report()


if __name__ == "__main__":
    main()
