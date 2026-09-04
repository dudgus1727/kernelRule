"""★ 예산 8 vs 16 — 예산이 **벽을 뚫나**. LLM 0회.

    python3 experiments/budget_report.py

실험 계획서 `docs/artifacts/budget-prereg.md`. 벽은 D-104 다 —
regret 이 낮은 규칙은 상위100 tau 가 0 근처고, tau 가 높은 규칙은
regret 이 1.6 대다.

보조 함수는 `two_stage.py` 것을 쓴다 (원칙 2).

⚠️ `runs/x-rank-b16` 은 **폐기했다** (D-107 — 출력 스키마가 8 로
굳어 있어 규칙 29개가 전부 8항이었다). 여기 16 팔은 `b16b` 다.
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from two_stage import A6000, _fit, _floor, _measure, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY
from kernelrule.rules.checks import _numeric_literals

ARMS = [("b08", "예산 8"), ("b16b", "예산 16")]
SEEDS = 3
WATCH = ("split_k_cost", "sm_idle_cost", "pipeline_warmup_frac",
         "tail_waste", "waves")


def _rows(d: Path, name: str) -> list[dict]:
    return [json.loads(x) for x in (d / name).read_text().splitlines()
            if x.strip()]


def _pick(d: Path) -> dict:
    """순위 손실 최고 — 두 팔 다 목적함수가 순위다."""
    return sorted(_rows(d, "archive.jsonl"),
                  key=lambda e: e.get("rank_loss", 1e9))[0]


def _feats(code: str) -> set:
    return {n.attr for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "f"}


def _n_terms(code: str) -> int:
    return max([n.slice.value for n in ast.walk(ast.parse(code))
                if isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id == "w"
                and isinstance(n.slice, ast.Constant)] + [-1]) + 1


def _spent(code: str, n_w: int) -> int:
    """★ 예산 단위 = 숫자 리터럴 + 가중치. **항 수가 아니다.**"""
    counted, _ = _numeric_literals(ast.parse(code))
    return len(counted) + n_w


def _configs(code, w, table, matrix, shapes) -> int:
    fn, w = compile_rule(code), np.asarray(w, dtype=np.float64)
    picks = []
    for p in shapes:
        cand = table.candidates(p)
        j = int(cand.top_k(make_score_of(fn, matrix, w)(p, cand), 1)[0])
        picks.append((str(cand.kernel_id[j]), int(cand.split_k[j]),
                      str(cand.split_k_mode[j])))
    return len(Counter(picks))


def _blk(label: str, vals: list[tuple]) -> dict:
    v = np.array(vals)
    print(f"  {label:16s} {np.median(v[:, 0]):8.4f} {np.median(v[:, 1]):12.3f} "
          f"{np.median(v[:, 2]):10.3f}   "
          f"({v[:, 1].min():+.3f}~{v[:, 1].max():+.3f})")
    return {"vals": [list(x) for x in vals],
            "med": [float(np.median(v[:, i])) for i in range(3)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/budget.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    dirs = {t: [Path("runs") / f"f1pipe-F3-{t}-s{i}" for i in range(SEEDS)]
            for t, _ in ARMS}
    best = {t: [_pick(d) for d in dirs[t]] for t, _ in ARMS}
    out: dict = {"n_holdout": len(hold)}

    # -------------------------------------------------------- §1 구속력
    print("=" * 80)
    print("§1  예산이 **구속했나** — 이것이 먼저다 (D-107 에서 세 번 틀렸다)")
    print("=" * 80)
    print(f"  {'':16s} {'예산 소비 (리터럴+가중치)':>28} {'항 수':>16}")
    for tag, label in ARMS:
        sp_, tm = [], []
        lim = json.loads((dirs[tag][0] / "config.json").read_text())
        for d in dirs[tag]:
            for e in _rows(d, "archive.jsonl"):
                sp_.append(_spent(e["code"], len(e["w"])))
                tm.append(_n_terms(e["code"]))
        # ★ D-128 개명. 옛 산출물은 변환했지만 둘 다 읽는다.
        _rc = lim["rule_constraints"]
        cap = _rc.get("parameters", _rc.get("budget"))
        n_at = sum(1 for x in sp_ if x >= cap)
        print(f"  {label:16s} 중앙 {np.median(sp_):4.1f} / 상한 {cap:2d}   "
              f"상한까지 쓴 규칙 {n_at:2d}/{len(sp_):2d}   중앙 "
              f"{np.median(tm):4.1f} ({min(tm)}~{max(tm)})")
        out.setdefault("bind", {})[tag] = {
            "spent": sp_, "terms": tm, "cap": cap, "n_at_cap": n_at}
    print("\n  ⚠️ 실험 계획서에 '16항을 쓰는 규칙이 0개면 무효' 라고 썼다.")
    print("     **단위를 잘못 썼다** — 예산은 항이 아니라 리터럴+가중치를")
    print("     센다. 항 수로는 최대 15 지만 예산 소비로는 상한에 닿는다.")

    # -------------------------------------------------------- §2 주 지표
    print("\n" + "=" * 80)
    print("§2  주 지표 — A6000 홀드아웃 20형상, 3시드 중앙")
    print("=" * 80)
    fl = _floor(T, hold)
    for obj, head in (
            ("regret", "★ regret 재적합 가중치 — 판정선이 걸린 칸"),
            ("rank", "순위 적합 가중치")):
        print(f"\n  --- {head} ---")
        print(f"  {'':16s} {'regret':>8} {'상위100 tau':>12} {'전구간':>10}"
              f"   (tau100 범위)")
        for tag, label in ARMS:
            vals = [_measure(*_fit(e["code"], e["w"], T, M, train, obj),
                             T, M, hold) for e in best[tag]]
            out.setdefault(obj, {})[tag] = _blk(label, vals)
        print(f"  {'★ 무작위 바닥':16s} {fl[0]:8.4f} {fl[1]:12.3f} "
              f"{fl[2]:10.3f}")
    out["floor"] = fl

    print("\n  판정 — 실험 계획서에 박은 선")
    r16 = out["regret"]["b16b"]["med"]
    r08 = out["regret"]["b08"]["med"]
    t16, g16 = r16[1], r16[0]
    verdict = ("★ 예산이 벽을 뚫었다" if t16 >= 0.20 and g16 <= 1.20
               else "tau 는 올랐으나 regret 이 1.20 을 넘는다 — 순위 팔로 "
                    "옮겨간 것" if t16 >= 0.20
               else "★ 벽은 예산 탓이 아니다" if t16 <= 0.10
               else "구분 불가 (0.10~0.20)")
    print(f"    예산 16 의 regret 재적합: tau {t16:+.3f} / regret {g16:.4f}")
    print(f"    예산 8  의 같은 칸:      tau {r08[1]:+.3f} / regret "
          f"{r08[0]:.4f}")
    print(f"    -> {verdict}")

    # -------------------------------------------------------- §3 과적합
    print("\n" + "=" * 80)
    print("§3  학습-홀드아웃 격차 / 적합기 / 비용")
    print("=" * 80)
    print(f"  {'':16s} {'학습':>8} {'홀드아웃':>9} {'격차':>9} "
          f"{'적합기 도달':>11} {'거부율':>7} {'config':>7} {'분':>7}")
    for tag, label in ARMS:
        tr, ho, reach = [], [], []
        prop = rej = mv = sc = 0
        secs = 0.0
        for d in dirs[tag]:
            rs = _rows(d, "rounds.jsonl")
            tr.append(rs[-1]["best_regret"])
            ho.append(rs[-1]["best_val_regret"])
            reach.append(sum(x["n_fit_moved"] for x in rs)
                         / max(1, sum(x["n_scored"] for x in rs)))
            for r in rs:
                prop += r["n_proposed"]
                rej += (r["n_rejected_schema"] + r["n_rejected_static"]
                        + r["n_rejected_sandbox"] + r["n_rejected_fit"])
                mv += r["n_fit_moved"]
                sc += r["n_scored"]
                secs += r["seconds"]
        g = [h - t for h, t in zip(ho, tr, strict=True)]
        nc = [_configs(e["code"], e["w"], T, M, hold) for e in best[tag]]
        print(f"  {label:16s} {np.median(tr):8.4f} {np.median(ho):9.4f} "
              f"{np.median(g):+9.4f} {np.mean(reach):10.1%} "
              f"{rej / prop:7.1%} {np.median(nc):7.1f} {secs / 60:7.1f}")
        print(f"  {'':16s} {'':8s} {'':9s} 시드별 "
              f"{' '.join(f'{x:+.4f}' for x in g)}")
        out.setdefault("cost", {})[tag] = {
            "train": tr, "hold": ho, "gap": g, "reach": reach,
            "rej": rej / prop, "n_config": nc, "minutes": secs / 60}

    print("\n  ★ 상한 측정의 다섯 축 (/3 시드)")
    print(f"    {'':16s} " + " ".join(f"{n[:11]:>12s}" for n in WATCH))
    for tag, label in ARMS:
        fs = [_feats(e["code"]) for e in best[tag]]
        print(f"    {label:16s} "
              + " ".join(f"{sum(n in f for f in fs):>12d}" for n in WATCH))
        out.setdefault("cost", {})[tag]["feats"] = [sorted(f) for f in fs]
    for tag, label in ARMS:
        fs = [_feats(e["code"]) for e in best[tag]]
        print(f"    {label} 축 수: {[len(f) for f in fs]}  "
              f"합집합 {len(set().union(*fs))}")

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3시드는 유의성을 못 낸다 — 범위 분리로 읽는다 (원칙 27)")


if __name__ == "__main__":
    main()
