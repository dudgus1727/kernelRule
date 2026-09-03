"""★ 가중치를 지수 자리에 — 형태 (b) 의 결과. LLM 0회.

    python3 experiments/power_report.py

사전 등록 `docs/artifacts/power-prereg.md`. 기준선은 `rankevo`
(같은 조건, 힌트만 없음).

## 왜 채택률만으로는 부족한가

"제안에는 18% 나왔는데 아카이브에는 1개" 는 두 가지로 읽힌다 —
**시도했는데 졌다** 와 **애초에 나쁜 제안이었다**(형태와 무관). 그래서
제안 자체를 **직접 맞춰서** 겨룬다 (§2).
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from pathlib import Path

import numpy as np
from two_stage import A6000, _fit, _floor, _measure, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import _Problem
from kernelrule.features import REGISTRY
from kernelrule.rules.checks import exponent_indices, weight_bounds

ARMS = [("rankevo", "기준선 (힌트 없음)"), ("pow", "지수 자리 명시")]
SEEDS = 3
N_HEAD = 20          # ★ 머리 맞대기 표본 (양쪽 같은 수)


def _rows(d: Path, name: str) -> list[dict]:
    return [json.loads(x) for x in (d / name).read_text().splitlines()
            if x.strip()]


def _best(d: Path) -> dict:
    return sorted(_rows(d, "archive.jsonl"),
                  key=lambda e: e.get("rank_loss", 1e9))[0]


def _proposals(d: Path) -> list[dict]:
    out = []
    for f in sorted((d / "llm_calls").glob("*-rule_editor.json")):
        r = json.loads(f.read_text()).get("response")
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except Exception:                           # noqa: BLE001
                continue
        if isinstance(r, dict) and r.get("code") and r.get("w0"):
            out.append(r)
    return out


def _n_terms(code: str) -> int:
    return max([n.slice.value for n in ast.walk(ast.parse(code))
                if isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id == "w"
                and isinstance(n.slice, ast.Constant)] + [-1]) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/power.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    dirs = {t: [Path("runs") / f"f1pipe-F3-{t}-s{i}" for i in range(SEEDS)]
            for t, _ in ARMS}
    out: dict = {"n_holdout": len(hold)}

    # ----------------------------------------------------- §1 실제로 썼나
    print("=" * 84)
    print("§1  형태를 실제로 썼나 — 제안 / 아카이브 / 적합된 지수")
    print("=" * 84)
    for tag, lab in ARMS:
        pr = ex = arc_n = arc_e = 0
        expo_w = []
        for d in dirs[tag]:
            for p in _proposals(d):
                pr += 1
                if exponent_indices(p["code"]):
                    ex += 1
            for e in _rows(d, "archive.jsonl"):
                arc_n += 1
                idx = exponent_indices(e["code"])
                if idx:
                    arc_e += 1
                    expo_w += [float(e["w"][i]) for i in idx if i < len(e["w"])]
        print(f"  {lab:18s} 제안 {ex:3d}/{pr:3d} ({ex / max(1, pr):5.1%})   "
              f"아카이브 {arc_e:2d}/{arc_n:2d} ({arc_e / max(1, arc_n):5.1%})")
        if expo_w:
            print(f"  {'':18s} ★ 적합된 지수 {[round(x, 3) for x in expo_w]}  "
                  f"|w-1| 중앙 {np.median(np.abs(np.array(expo_w) - 1)):.3f}")
        out.setdefault("uptake", {})[tag] = {
            "prop": pr, "prop_expo": ex, "arc": arc_n, "arc_expo": arc_e,
            "expo_w": expo_w}

    # ------------------------------------------- §2 제안끼리 머리 맞대기
    print("\n" + "=" * 84)
    print("§2  ★ 시도했는데 진 것인가 — 제안을 **직접 맞춰서** 겨룬다")
    print("=" * 84)
    props = [p for d in dirs["pow"] for p in _proposals(d)]
    with_e = [p for p in props if exponent_indices(p["code"])]
    without = [p for p in props if not exponent_indices(p["code"])]
    rng = np.random.default_rng(0)
    pick = {"지수 있음": [with_e[i] for i in
                       rng.choice(len(with_e), N_HEAD, replace=False)],
            "지수 없음": [without[i] for i in
                       rng.choice(len(without), N_HEAD, replace=False)]}
    prob = _Problem(M, T, sp.train.shapes, 1)
    prob.build_pairs(T, 100)
    print(f"  같은 실행의 제안에서 각 {N_HEAD}개씩 무작위로 뽑아 "
          f"**같은 절차로** 맞춘다\n")
    print(f"  {'':12s} {'학습 순위손실':>12} {'홀드아웃 regret':>14} "
          f"{'항':>4}   (순위손실 범위)")
    for lab, ps in pick.items():
        rl, rg, tm = [], [], []
        for p in ps:
            try:
                w0 = list(p["w0"])
                fn, ws = _fit(p["code"], w0, T, M, train, "rank")
            except Exception:                           # noqa: BLE001
                continue
            w = ws["long"]
            rl.append(float(prob.rank_loss(compile_rule(p["code"]),
                                           np.asarray(w, float))))
            rg.append(_measure(fn, ws, T, M, hold)[0])
            tm.append(_n_terms(p["code"]))
        rl, rg = np.array(rl), np.array(rg)
        print(f"  {lab:12s} {np.median(rl):12.4f} {np.median(rg):14.4f} "
              f"{np.median(tm):4.1f}   ({rl.min():.3f}~{rl.max():.3f})")
        out.setdefault("head", {})[lab] = {
            "rank_loss": rl.tolist(), "regret": rg.tolist(),
            "terms": tm}

    # ------------------------------------------------- §3 판정선이 걸린 칸
    print("\n" + "=" * 84)
    print("§3  최종 지표 — 홀드아웃 20형상, 3시드 중앙")
    print("=" * 84)
    fl = _floor(T, hold)
    for obj, head in (("regret", "★ regret 재적합 — 판정선이 걸린 칸"),
                      ("rank", "순위 적합")):
        print(f"\n  --- {head} ---")
        print(f"  {'':18s} {'regret':>8} {'상위100 tau':>12} {'전구간':>9}"
              f"   (tau 범위)")
        for tag, lab in ARMS:
            vals = []
            for d in dirs[tag]:
                e = _best(d)
                fn, ws = _fit(e["code"], e["w"], T, M, train, obj)
                vals.append(_measure(fn, ws, T, M, hold))
            v = np.array(vals)
            und = int(v[:, 3].sum())
            print(f"  {lab:18s} {np.median(v[:, 0]):8.4f} "
                  f"{np.median(v[:, 1]):12.3f} {np.median(v[:, 2]):9.3f}"
                  f"   ({v[:, 1].min():+.3f}~{v[:, 1].max():+.3f})"
                  + (f"  ⚠️ tau 정의 안 됨 {und}형상" if und else ""))
            out.setdefault(obj, {})[tag] = [list(x) for x in v]
        print(f"  {'★ 무작위 바닥':18s} {fl[0]:8.4f} {fl[1]:12.3f} "
              f"{fl[2]:9.3f}")
    out["floor"] = fl

    r = out["regret"]["pow"]
    med_r, med_t = float(np.median([x[0] for x in r])), \
        float(np.median([x[1] for x in r]))
    print("\n  판정 — 사전 등록에 박은 선")
    print(f"    regret {med_r:.4f} / 상위100 tau {med_t:+.3f}  ->  " + (
        "★ 벽이 낮아졌다" if med_t >= 0.20 and med_r <= 1.15
        else "벽은 형태의 문제가 아니다"))

    # ------------------------------------------------------------ 공통
    print("\n" + "=" * 84)
    print("공통")
    print("=" * 84)
    print(f"  {'':18s} {'항':>4} {'거부율':>7} {'적합기':>7} {'분':>7}")
    for tag, lab in ARMS:
        arc = [e for d in dirs[tag] for e in _rows(d, "archive.jsonl")]
        prop = rej = mv = sc = 0
        secs = 0.0
        for d in dirs[tag]:
            for x in _rows(d, "rounds.jsonl"):
                prop += x["n_proposed"]
                rej += (x["n_rejected_schema"] + x["n_rejected_static"]
                        + x["n_rejected_sandbox"] + x["n_rejected_fit"])
                mv += x["n_fit_moved"]
                sc += x["n_scored"]
                secs += x["seconds"]
        print(f"  {lab:18s} {np.median([_n_terms(e['code']) for e in arc]):4.0f} "
              f"{rej / prop:7.1%} {mv / max(1, sc):7.1%} {secs / 60:7.1f}")
        out.setdefault("common", {})[tag] = {
            "rej": rej / prop, "reach": mv / max(1, sc), "minutes": secs / 60}

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3시드는 유의성을 못 낸다 — 범위 분리로 읽는다 (원칙 27)")
    _ = weight_bounds


if __name__ == "__main__":
    main()
