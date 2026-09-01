"""★ 넷 이어달리기 결과 — 두 순서 + 예산 8/16. LLM 0회.

    python3 experiments/chain_report.py

## 무엇을 재나

    ord-r2g   순위 -> regret   예산 8
    ord-g2r   regret -> 순위   예산 8
    bud08     순위 고정        예산 8      ← 예산 대조의 기준선
    bud16     순위 고정        예산 16

⚠️ **전부 홀드아웃 20형상에서 잰다.** D-101 의 tau 는 학습 41형상 값이라
나란히 못 놓는다 (원칙 4).

## 두 번 재는 이유

산출물을 고를 때(선택)와 가중치를 맞출 때(적합)의 목적함수는 **다를 수
있다.** 네 팔의 최종 목적함수가 서로 달라서, 한 가지 적합으로만 재면
*순서*가 아니라 *평가 방식*을 비교하게 된다. 그래서 같은 규칙을
regret 적합과 순위 적합 **양쪽으로** 재고 두 블록을 나란히 놓는다.

보조 함수는 `two_stage.py` 것을 그대로 쓴다 — 절차가 하나여야 나란히
놓을 수 있다 (원칙 2).
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
from kernelrule.core.weights import _Problem, make_score_of
from kernelrule.features import REGISTRY

#: (태그, 라벨, 최종 목적함수) — 최종 목적함수가 **산출물 선택 기준**이다.
ARMS = [
    ("ord-r2g", "4-1 순위->regret", "regret"),
    ("ord-g2r", "4-2 regret->순위", "rank"),
    ("bud08", "예산 8  (순위 고정)", "rank"),
    ("bud16", "예산 16 (순위 고정)", "rank"),
]
SEEDS = 3
#: 상한 측정이 지목한 다섯 축 (ranking-ceiling.md §3).
WATCH = ("split_k_cost", "sm_idle_cost", "pipeline_warmup_frac",
         "tail_waste", "waves")


def _rows(d: Path, name: str) -> list[dict]:
    return [json.loads(x) for x in (d / name).read_text().splitlines()
            if x.strip()]


def _pick(d: Path, by: str) -> dict:
    """아카이브 최고 — **그 팔의 최종 목적함수 기준**."""
    key = (lambda e: e.get("rank_loss", 1e9)) if by == "rank" \
        else (lambda e: e["regret"])
    return sorted(_rows(d, "archive.jsonl"), key=key)[0]


def _feats(code: str) -> set:
    return {n.attr for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "f"}


def _switch_round(d: Path) -> int | None:
    """전환 라운드 — 실행 산출물 `config.json` 에서 읽는다.

    ★ 로그(stdout)가 아니라 산출물에서 읽는다. 로그는 세 시드가 한 파일에
    섞여 있어서 시드 귀속이 눈으로만 되고, 지워지면 되짚을 수 없다.
    """
    c = json.loads((d / "config.json").read_text())
    r = int(c.get("switch_round", -1))
    return None if r < 0 else r


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
    print(f"  {label:22s} {np.median(v[:, 0]):8.4f} {np.median(v[:, 1]):12.3f} "
          f"{np.median(v[:, 2]):10.3f}   "
          f"({v[:, 1].min():+.3f}~{v[:, 1].max():+.3f})"
          f" ({v[:, 2].min():+.3f}~{v[:, 2].max():+.3f})")
    return {"vals": vals, "med": [float(np.median(v[:, i])) for i in range(3)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/chain.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    dirs = {tag: [Path("runs") / f"f1pipe-F3-{tag}-s{i}" for i in range(SEEDS)]
            for tag, _, _ in ARMS}
    best = {tag: [_pick(d, by) for d in dirs[tag]] for tag, _, by in ARMS}
    out: dict = {"n_holdout": len(hold), "n_train": len(train)}

    # ------------------------------------------------------------ §1 지표
    print("=" * 84)
    print("§1  최종 지표 — A6000 홀드아웃 20형상")
    print("=" * 84)
    print("  선택: 각 팔의 **최종 목적함수** 기준 아카이브 최고 (3시드 중앙)")
    fl = _floor(T, hold)
    for obj, head in (("regret", "regret 적합 가중치 (정준 §10 경로)"),
                      ("rank", "순위 적합 가중치")):
        print(f"\n  --- {head} ---")
        print(f"  {'':22s} {'regret':>8} {'상위100 tau':>12} {'전구간':>10}"
              f"   (tau100 범위)")
        for tag, label, _ in ARMS:
            vals = [_measure(*_fit(e["code"], e["w"], T, M, train, obj),
                             T, M, hold) for e in best[tag]]
            out.setdefault(obj, {})[tag] = _blk(label, vals)
        print(f"  {'★ 무작위 바닥(20뽑기)':22s} {fl[0]:8.4f} {fl[1]:12.3f} "
              f"{fl[2]:10.3f}")
    out["floor"] = fl

    # ------------------------------------------------------------ §2 전환
    print("\n" + "=" * 84)
    print("§2  전환 — 언제 바뀌었고, 직후 한 라운드에서 무엇이 튀나")
    print("=" * 84)
    print("  ⚠️ 로그의 `best` 는 **그때의 목적함수 기준** 이라 전환에서")
    print("     저절로 튄다. 여기서는 두 지표를 매 라운드 다 적어 비교한다.")
    prob = _Problem(M, T, sp.train.shapes, 1)
    prob.build_pairs(T, 100)
    print(f"  순위 손실은 **여기서 다시 계산했다** — 실행 중에는 "
          f"objective=='rank' 일 때만 기록되므로\n"
          f"     전환 반대편이 NaN 이라 나란히 못 놓는다 (쌍 "
          f"{prob.n_pairs:,}개, 못 가르는 쌍 {prob.n_dropped:,}개 제외).")
    for tag, label, _ in ARMS[:2]:
        print(f"\n  {label}")
        arm = []
        for i, d in enumerate(dirs[tag]):
            rd = {r["round"]: r for r in _rows(d, "rounds.jsonl")}
            bs = _rows(d, "bests.jsonl")
            sw = _switch_round(d)
            arm.append(sw)
            print(f"    s{i}  전환 r{sw}")
            print(f"      {'라운드':>7} {'regret':>8} {'순위손실':>9} {'셀':>4}")
            seq = []
            for b in bs:
                if not (sw - 2 <= b["round"] <= sw + 2):
                    continue
                rl = float(prob.rank_loss(compile_rule(b["code"]),
                                          np.asarray(b["w"], dtype=float)))
                seq.append((b["round"], b["regret"], rl,
                            rd[b["round"]]["n_cells"]))
                print(f"      {'r' + str(b['round']):>7} {b['regret']:8.4f} "
                      f"{rl:9.4f} {rd[b['round']]['n_cells']:4d}"
                      + (" ★전환" if b["round"] == sw else ""))
            pre = [x for x in seq if x[0] == sw - 1][0]
            post = [x for x in seq if x[0] == sw][0]
            print(f"      -> regret {pre[1]:.4f}->{post[1]:.4f} "
                  f"({post[1] - pre[1]:+.4f})   순위 {pre[2]:.4f}->{post[2]:.4f} "
                  f"({post[2] - pre[2]:+.4f})   셀 {pre[3]}->{post[3]}")
            out.setdefault("switch", {}).setdefault(tag, []).append(
                {"seed": i, "round": sw, "seq": seq})

    # ------------------------------------------------------------ §3 예산
    print("\n" + "=" * 84)
    print("§3  예산 — 학습 vs 홀드아웃 격차 / 항 수 / 적합기 도달률")
    print("=" * 84)
    print("  ⚠️ 예산 16 팔은 **무효다** (D-105). 프롬프트가 8 로 렌더링돼서")
    print("     모델은 두 팔에서 같은 상한을 들었다 — 예산 대조가 아니라")
    print("     같은 조건의 반복이다. 아래 두 줄은 그렇게 읽어야 한다.\n")
    print(f"  {'':22s} {'학습 regret':>11} {'홀드아웃':>9} {'격차':>8} "
          f"{'항 수':>6} {'적합기 도달':>11}")
    for tag, label, _ in ARMS:
        g, tr, ho, reach, terms = [], [], [], [], []
        for d, e in zip(dirs[tag], best[tag], strict=True):
            rd = _rows(d, "rounds.jsonl")[-1]
            tr.append(rd["best_regret"])
            ho.append(rd["best_val_regret"])
            g.append(rd["best_val_regret"] - rd["best_regret"])
            allr = _rows(d, "rounds.jsonl")
            reach.append(sum(x["n_fit_moved"] for x in allr)
                         / max(1, sum(x["n_scored"] for x in allr)))
            terms.append(len(e["w"]))
        print(f"  {label:22s} {np.median(tr):11.4f} {np.median(ho):9.4f} "
              f"{np.median(g):+8.4f} {np.median(terms):6.1f} "
              f"{np.mean(reach):10.1%}")
        print(f"  {'':22s} {'':11s} {'':9s}   시드별 격차 "
              f"{' '.join(f'{x:+.4f}' for x in g)}   항 {terms}")
        out.setdefault("budget", {})[tag] = {
            "train": tr, "hold": ho, "gap": g, "terms": terms,
            "reach": reach}

    # ------------------------------------------------------------ §4 공통
    print("\n" + "=" * 84)
    print("§4  공통 — 살아남은 축 / config 다양성 / 거부율 / 비용")
    print("=" * 84)
    print(f"  {'':22s} {'config종류':>10} {'거부율':>8} {'채택률':>8} "
          f"{'LLM호출':>8} {'분':>7}")
    for tag, label, _ in ARMS:
        nc = [_configs(e["code"], e["w"], T, M, hold) for e in best[tag]]
        prop = rej = acc = sc = calls = secs = 0
        for d in dirs[tag]:
            for r in _rows(d, "rounds.jsonl"):
                prop += r["n_proposed"]
                rej += (r["n_rejected_schema"] + r["n_rejected_static"]
                        + r["n_rejected_sandbox"] + r["n_rejected_fit"])
                acc += r["n_accepted"]
                sc += r["n_scored"]
                calls += sum(r["llm_calls"].values())
                secs += r["seconds"]
        print(f"  {label:22s} {np.median(nc):10.1f} {rej / prop:8.1%} "
              f"{acc / max(1, sc):8.1%} {calls:8d} {secs / 60:7.1f}")
        out.setdefault("common", {})[tag] = {
            "n_config": nc, "rej": rej / prop, "acc": acc / max(1, sc),
            "llm_calls": calls, "minutes": secs / 60,
            "feats": [sorted(_feats(e["code"])) for e in best[tag]]}

    print("\n  ★ 상한 측정이 지목한 다섯 축 — 최종 규칙이 쓰는가 (/3 시드)")
    print(f"    {'':22s} " + " ".join(f"{n[:11]:>12s}" for n in WATCH))
    for tag, label, _ in ARMS:
        fs = [_feats(e["code"]) for e in best[tag]]
        print(f"    {label:22s} "
              + " ".join(f"{sum(n in f for f in fs):>12d}" for n in WATCH))
    allf = Counter(x for tag, _, _ in ARMS for e in best[tag]
                   for x in _feats(e["code"]))
    print(f"\n  쓴 축 합집합 {len(allf)}개 (괄호는 12개 규칙 중 몇 개):")
    print("    " + ", ".join(f"{k}({v})" for k, v in allf.most_common()))

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3시드는 유의성을 못 낸다 — 범위 분리로 읽는다 (원칙 27)")


if __name__ == "__main__":
    main()
