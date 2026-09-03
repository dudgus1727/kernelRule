"""★ §3 표현력 셋 — **regret 경로에서** 예산·곱·지수. 한 번에 보고한다.

    python3 experiments/expressive_report.py       # -> docs/artifacts/expressive-regret.json

사전 등록 `docs/artifacts/expressive-regret-prereg.md`. 판정선은 거기
박혀 있다 — 여기서 정하지 않는다.

```
rb08   예산 8            ★ 네 판의 내부 기준선
rb16   예산 16
rprod  예산 8 + 곱 힌트
rpow   예산 8 + 지수 힌트
```

네 판 전부 **같은 씨앗·같은 적합기**(CMA-ES, 재시작 1, 적합 300 /
다듬기 600)로 돈다 — §2 관문이 고른 팔이다 (D-123).

⚠️ **재는 쪽 적합기도 CMA 다.** Nelder-Mead 로 재적합하면 16항 규칙에서
도달률이 92% 라(D-77), "예산 16 이 나쁘다" 가 아니라 재는 쪽의 실패를
잰다. 그래서 `_fit(method="cma", n_restarts=1)` 을 준다.

⚠️ 기준선 `1.0762` 은 **Nelder-Mead 200/600** 짜리다. 참고로만 적고
판정에는 `rb08` 을 쓴다 (원칙 4).

보조 함수는 `two_stage.py` / `regret_at_k.py` / `wall_report.py` 것을
쓴다 (원칙 2).
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from power_report import _proposals
from regret_at_k import _measure as _measure_k
from two_stage import A6000, _fit, _splits
from wall_report import _n_terms, _prod_pairs

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.runset import assert_same_condition, run_condition
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.rules.checks import _numeric_literals

#: (태그, 라벨, 이 팔이 기준선과 **다른 것**)
ARMS = [("rb08", "예산 8 (기준선)", None),
        ("rb16", "예산 16", "rule_budget"),
        ("rprod", "곱 힌트", "product_hint"),
        ("rpow", "지수 힌트", "power_hint")]
SEEDS = 3
#: 사전 등록 §3. **여기서 정하지 않는다** (원칙 7).
DELTA = 0.0516
#: 옛 적합기(Nelder-Mead 200/600)의 같은 조건. **참고값이다** (원칙 4).
OLD_BASELINE = 1.0762
#: 보고할 k. 사전 등록 §1 — 1 이 주 지표, 10/50/100 이 "영역을 더
#: 정밀하게 가리키는가" 를 본다.
REPORT_KS = (1, 10, 50, 100)


def _runs(tag: str) -> list[str]:
    return [f"f1pipe-F3-{tag}-s{i}" for i in range(SEEDS)]


def _rows(run: str, name: str) -> list[dict]:
    f = Path("runs") / run / name
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()]


def _best(run: str) -> dict:
    """★ regret 최고. 네 판 다 목적함수가 regret 이다."""
    return sorted(_rows(run, "archive.jsonl"), key=lambda e: e["regret"])[0]


def _spent(code: str, n_w: int) -> int:
    """예산 단위 = 숫자 리터럴 + 가중치. **항 수가 아니다** (D-108)."""
    counted, _ = _numeric_literals(ast.parse(code))
    return len(counted) + n_w


def _n_power(code: str) -> int:
    """`np.power(f.*, w[i])` 처럼 **지수 자리에 가중치**가 든 항의 수."""
    n = 0
    for x in ast.walk(ast.parse(code)):
        if (isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                and x.func.attr == "power" and len(x.args) == 2
                and any(isinstance(m, ast.Name) and m.id == "w"
                        for m in ast.walk(x.args[1]))):
            n += 1
        if (isinstance(x, ast.BinOp) and isinstance(x.op, ast.Pow)
                and any(isinstance(m, ast.Name) and m.id == "w"
                        for m in ast.walk(x.right))):
            n += 1
    return n


def _squares_and_crosses(code: str) -> tuple[int, int]:
    """★ **제곱과 교차곱을 따로** 센다 (D-110 의 정정).

    합집합으로 접으면 `f.a * f.a` 와 `f.a * f.b` 가 같은 "쌍" 이 되고,
    최빈 쌍이 실은 `reg_pressure^3` 였다.
    """
    sq = cr = 0
    for a, b in _prod_pairs(code):
        if a == b:
            sq += 1
        else:
            cr += 1
    for x in ast.walk(ast.parse(code)):
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) \
                and x.func.attr == "square":
            sq += 1
    return sq, cr


def _condition_table(out: dict) -> None:
    """★ §0 — 팔마다 조건이 하나인가, 그리고 팔 사이에 **하나만** 다른가.

    원칙 39 를 실행 가능하게 만든 자리다 (D-120). 여기서 걸리면 아래
    숫자는 전부 무의미하다 — 먼저 멈춘다 (§26.4).
    """
    print("=" * 92)
    print("§0  조건 — 팔마다 하나인가, 팔 사이에 하나만 다른가 (원칙 39)")
    print("=" * 92)
    base = None
    for tag, label, diff in ARMS:
        cond = assert_same_condition(_runs(tag), label=f"{label} ({tag})")
        c = cond or run_condition(_runs(tag)[0])
        print(f"  {label:16s} 예산 {str(c['rule_budget']):>3s}  "
              f"목적 {c['objective']:6s}  적합기 {c['fit_method']}/"
              f"{c['fit_restarts']}  곱 {str(c['product_hint']):5s}  "
              f"지수 {str(c['power_hint']):5s}  씨앗 {str(c['seed_sha'])[:8]}")
        out.setdefault("condition", {})[tag] = c
        if base is None:
            base = c
            continue
        got = sorted(k for k in base
                     if str(base[k]) != str(c[k]))
        exp = [diff] if diff else []
        if got != exp:
            raise SystemExit(
                f"★ {label} 이 기준선과 다른 키가 {got} 다 — {exp} 여야 "
                "한다. 조건이 둘 이상 다르면 무엇을 잰 것인지 모른다.")
        print(f"  {'':16s} -> 기준선과 다른 키: {got} ✅")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/expressive-regret.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    out: dict = {"delta": DELTA, "old_baseline": OLD_BASELINE,
                 "ks": list(REPORT_KS)}
    _condition_table(out)

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    out["n_holdout"] = len(hold)
    best = {tag: [_best(r) for r in _runs(tag)] for tag, _, _ in ARMS}

    # ---------------------------------------------------------- §1 구속력
    print("\n" + "=" * 92)
    print("§1  예산이 **구속했나** — 세 번 무효였던 자리다 (D-105·106·107)")
    print("=" * 92)
    print(f"  {'':16s} {'예산 소비':>22} {'항 수 (전체)':>18} "
          f"{'★ 1라운드 항 수':>18}")
    for tag, label, _ in ARMS:
        sp_, tm, r0 = [], [], []
        cap = out["condition"][tag]["rule_budget"]
        for r in _runs(tag):
            for e in _rows(r, "archive.jsonl"):
                sp_.append(_spent(e["code"], len(e["w"])))
                tm.append(_n_terms(e["code"]))
                if e.get("round") == 0:
                    r0.append(_n_terms(e["code"]))
        n_at = sum(1 for x in sp_ if x >= cap)
        print(f"  {label:16s} 중앙 {np.median(sp_):4.1f} / 상한 {cap:2d} "
              f"({n_at:2d}/{len(sp_):2d} 상한)   "
              f"중앙 {np.median(tm):4.1f} ({min(tm)}~{max(tm)})   "
              f"중앙 {(np.median(r0) if r0 else float('nan')):4.1f} "
              f"({(min(r0) if r0 else 0)}~{(max(r0) if r0 else 0)})")
        out.setdefault("bind", {})[tag] = {
            "spent": sp_, "terms": tm, "round0_terms": r0, "cap": cap,
            "n_at_cap": n_at}
    b16 = out["bind"]["rb16"]
    if b16["terms"] and max(b16["terms"]) <= 8:
        print("\n  ★ 무효 — 예산 16 팔에 8항을 넘는 규칙이 하나도 없다 "
              "(D-107 이 그것이었다)")

    # ---------------------------------------------------------- §2 주 지표
    print("\n" + "=" * 92)
    print("§2  주 지표 — regret@1, A6000 홀드아웃 20형상, 3시드")
    print("=" * 92)
    meas: dict = {}
    for tag, _label, _ in ARMS:
        vals = []
        for e in best[tag]:
            fn, ws = _fit(e["code"], e["w"], T, M, train, "regret",
                          method="cma", n_restarts=1)
            vals.append(_measure_k(fn, ws, T, M, hold))
        meas[tag] = vals
        out.setdefault("measure", {})[tag] = [
            {"regret_at_k": {str(k): v["regret_at_k"][k] for k in REPORT_KS},
             "tau_raw": v["tau_raw"], "tau_noise": v["tau_noise"]}
            for v in vals]
    r1 = {t: [v["regret_at_k"][1] for v in meas[t]] for t, _, _ in ARMS}
    print(f"  {'':16s} {'중앙':>8} {'시드 범위':>19} {'기준선 대비':>11}  판정")
    b = float(np.median(r1["rb08"]))
    for tag, label, _ in ARMS:
        v = r1[tag]
        m = float(np.median(v))
        d = b - m                       # 양수 = 이 팔이 **좋다**
        if tag == "rb08":
            verdict = "— (기준선)"
        elif d >= DELTA:
            verdict = f"★ 표현력이 벽이었다 (+{d:.4f} >= {DELTA})"
        else:
            verdict = f"구분 불가 ({d:+.4f} < {DELTA})"
        print(f"  {label:16s} {m:8.4f}   {min(v):.4f}~{max(v):.4f}   "
              f"{d:+11.4f}  {verdict}")
    print(f"\n  {'참고: 옛 적합기':16s} {OLD_BASELINE:8.4f}   "
          "← Nelder-Mead 200/600 의 예산 8. ★ 판정에 안 쓴다 (원칙 4)")
    print("  ★ 판정은 **시드 범위**로 읽는다 — 3시드는 유의성이 안 나온다")
    print("     (판정선 0.0516 은 n=6 짜리 검정력 계산이다)")
    out["r1"] = r1

    # ---------------------------------------------------------- §3 regret@k
    print("\n" + "=" * 92)
    print("§3  regret@k — 영역을 **더 정밀하게** 가리키나, 1등만 좋아지나")
    print("=" * 92)
    print(f"  {'':16s} " + " ".join(f"{f'k={k}':>16s}"
                                    for k in REPORT_KS))
    for tag, label, _ in ARMS:
        arr = np.array([[v["regret_at_k"][k] for k in REPORT_KS]
                        for v in meas[tag]])
        med = np.median(arr, axis=0)
        print(f"  {label:16s} " + " ".join(
            f"{med[i]:6.3f} {arr[:, i].min():.3f}-{arr[:, i].max():.3f}"
            for i in range(len(REPORT_KS))))
    print()
    print(f"  {'':16s} {'상위100 tau':>12} {'노이즈 인식':>12}")
    for tag, label, _ in ARMS:
        tr = [v["tau_raw"] for v in meas[tag]]
        tn = [v["tau_noise"] for v in meas[tag]]
        print(f"  {label:16s} {np.median(tr):12.3f} {np.median(tn):12.3f}")

    # ---------------------------------------------------------- §4 격차·비용
    print("\n" + "=" * 92)
    print("§4  학습-홀드아웃 격차 / 적합기 / 거부율 / 비용")
    print("=" * 92)
    print(f"  {'':16s} {'학습':>8} {'홀드아웃':>9} {'격차':>9} "
          f"{'적합 이동':>9} {'거부율':>7} {'분':>7}")
    for tag, label, _ in ARMS:
        tr, ho, mv, sc, prop, rej = [], [], 0, 0, 0, 0
        secs = 0.0
        for r in _runs(tag):
            rs = _rows(r, "rounds.jsonl")
            tr.append(rs[-1]["best_regret"])
            ho.append(rs[-1]["best_val_regret"])
            for x in rs:
                prop += x["n_proposed"]
                rej += (x["n_rejected_schema"] + x["n_rejected_static"]
                        + x["n_rejected_sandbox"] + x["n_rejected_fit"])
                mv += x["n_fit_moved"]
                sc += x["n_scored"]
                secs += x["seconds"]
        g = [h - t for h, t in zip(ho, tr, strict=True)]
        print(f"  {label:16s} {np.median(tr):8.4f} {np.median(ho):9.4f} "
              f"{np.median(g):+9.4f} {mv / max(1, sc):9.1%} "
              f"{rej / max(1, prop):7.1%} {secs / 60:7.1f}")
        out.setdefault("cost", {})[tag] = {
            "train": tr, "hold": ho, "gap": g, "moved": mv / max(1, sc),
            "rej": rej / max(1, prop), "minutes": secs / 60,
            "n_proposed": prop}

    # ---------------------------------------------------------- §5 형태
    print("\n" + "=" * 92)
    print("§5  형태를 실제로 썼나 — ★ 제곱과 교차곱을 **따로** 센다 (D-110)")
    print("=" * 92)
    print(f"  {'':16s} {'제안':>22} {'아카이브':>22}")
    print(f"  {'':16s} {'곱쓴규칙 제곱 교차':>22} {'곱쓴규칙 제곱 교차':>22} "
          f"{'지수 제안/아카이브':>20}")
    for tag, label, _ in ARMS:
        agg = {}
        # ★ 제안은 `llm_calls/*-rule_editor.json` 에서 읽는다 —
        #   `proposals.jsonl` 은 없다 (`power_report._proposals`, 원칙 2).
        for name in ("prop", "arc"):
            n = nsq = ncr = npw = used = 0
            for r in _runs(tag):
                d = Path("runs") / r
                src = (_proposals(d) if name == "prop"
                       else _rows(r, "archive.jsonl"))
                for e in src:
                    code = e.get("code")
                    if not code:
                        continue
                    try:
                        sq, cr = _squares_and_crosses(code)
                        pw = _n_power(code)
                    except SyntaxError:
                        continue
                    n += 1
                    nsq += sq
                    ncr += cr
                    npw += pw
                    used += 1 if (sq or cr) else 0
            agg[name] = dict(n=n, used=used, sq=nsq, cr=ncr, pw=npw)
        p_, ar = agg["prop"], agg["arc"]
        print(f"  {label:16s} "
              f"{p_['used']:3d}/{p_['n']:<4d} {p_['sq']:4d} {p_['cr']:4d}   "
              f"{ar['used']:3d}/{ar['n']:<4d} {ar['sq']:4d} {ar['cr']:4d}   "
              f"{p_['pw']:4d} / {ar['pw']:<4d}")
        out.setdefault("form", {})[tag] = agg

    # ---------------------------------------------------------- §6 배분
    print("\n" + "=" * 92)
    print("§6  ★ 판정 — 사전 등록 §3 의 선, 결과 보고 정하지 않았다")
    print("=" * 92)
    hits = [label for tag, label, _ in ARMS
            if tag != "rb08" and b - float(np.median(r1[tag])) >= DELTA]
    if hits:
        print(f"  ★ 표현력이 벽이었다 — {hits}")
        print("     -> 순위 손실에서 안 됐던 것은 그 목적함수의 성질이었다")
        print("     -> conclusion.md 의 '벽' 절을 다시 쓴다 (사전 등록 §4)")
    else:
        print("  ★ 셋 다 구분 불가 — 벽은 피처 공간의 성질이다")
        print("     선형/비선형/예산이 아니다. 남는 후보는 **트리의 조건부")
        print("     분기**다 — 우리 `np.where` 는 형상 수준(`p.*`)으로만")
        print("     갈리고 GBDT 는 config 수준 피처의 문턱으로도 갈린다")
        print("     (사전 등록 §4 에 후보로 등록해 뒀다)")
    out["hits"] = hits

    n_cfg = {}
    for tag, _label, _ in ARMS:
        picks = []
        for e in best[tag]:
            fn, ws = _fit(e["code"], e["w"], T, M, train, "regret",
                          method="cma", n_restarts=1)
            for p in hold:
                cand = T.candidates(p)
                from kernelrule.core.splits import regime_of
                from kernelrule.core.weights import make_score_of
                s = make_score_of(fn, M, ws[regime_of(p, T.hw)])(p, cand)
                j = int(cand.top_k(s, 1)[0])
                picks.append((str(cand.kernel_id[j]), int(cand.split_k[j])))
        n_cfg[tag] = len(Counter(picks))
    print("\n  고른 config 종류 (3시드 x 홀드아웃 20형상): "
          + ", ".join(f"{lab}={n_cfg[t]}" for t, lab, _ in ARMS))
    out["n_config"] = n_cfg

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
