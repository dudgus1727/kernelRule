"""★ Critic (D-85) — 규칙을 **항 단위로** 심사하고, 그 판정을 검증한다.

```
python3 experiments/critic.py judge  --runs f1pipe-F3-arch24-s0 ...   # LLM
python3 experiments/critic.py ablate --judged docs/artifacts/critic.json  # LLM 0회
python3 experiments/critic.py rank   --campaign runs/f1pipe-F3-arch24     # LLM
```

## 왜 있나

"해석 가능한 규칙" 이 이 연구의 주장인데, 지금은 **사람이 코드를 직접
읽어야** 그 주장을 확인할 수 있다. 항별 물리 설명을 붙이는 것이 GBDT
와의 차이를 눈에 보이게 만든다.

## ★ 판정을 성능이 아니라 **정확도**로 잰다

`ablate` 가 관문이다. Critic 이 "설명 못 하겠다" 고 한 항을 빼고 다시
적합한다.

```
빼도 regret 이 안 나빠진다   ★ 판정이 맞았다
크게 나빠진다               틀렸거나, 설명 못 해도 유용한 항이다
```

**★ 그리고 다른 항들도 하나씩 빼서 견준다.** "설명 불가 항을 빼도
멀쩡하다" 는 **모든 항이 그렇다면** 아무 말도 아니다. 지표는
**설명 불가 항의 손상 순위**다 — 정확하면 아래쪽(덜 아픈 쪽)에 있어야
한다.

regret 의 절대값은 보고하지 않는다 (D-56 §2). 차이와 순위만 쓴다.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"


def _table_and_matrix():
    import kernelrule.features.physical  # noqa: F401
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.core.table import PerfTable
    from kernelrule.features import REGISTRY

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    return table, FeatureMatrix(table, REGISTRY), REGISTRY


def _train_groups(table):
    from kernelrule.core.splits import regime_of

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    train = [p for p in shapes if 11008 not in (p.N, p.K)]
    return {n: [p for p in train if regime_of(p, table.hw) == n]
            for n in ("short", "long")}


def _best_rule(run: str) -> dict:
    arc = [json.loads(ln) for ln in
           (Path("runs") / run / "archive.jsonl").read_text().splitlines()
           if ln.strip()]
    return min(arc, key=lambda e: e["regret"])


# ---------------------------------------------------------------- judge (LLM)
def cmd_judge(a) -> None:
    import numpy as np

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import Budget, LLMConfig, OpenAILLM
    from kernelrule.features import REGISTRY
    from kernelrule.rules.ablate import reorder_terms, term_exprs

    llm = OpenAILLM(LLMConfig(),
                    feature_names=REGISTRY.names(shape_level=False),
                    shape_values=REGISTRY.names(shape_level=True),
                    registry=REGISTRY, cache=False,
                    budget=Budget(max_calls=len(a.runs) * 3))
    # ★ 순서 섞기 (D-86). 항 순서를 바꾸고 `w` 인덱스를 다시 매긴다 —
    #   Critic 이 "마지막 항" 을 지목하는 것이 **위치 편향**인지 본다.
    #   식 자체는 그대로이므로, 같은 식을 지목하면 편향이 아니다.
    rng = np.random.default_rng(a.shuffle) if a.shuffle is not None else None
    out = []
    for run in a.runs:
        best = _best_rule(run)
        code, order = best["code"], None
        if rng is not None:
            exprs = term_exprs(code)
            order = [int(i) for i in rng.permutation(sorted(exprs))]
            code = reorder_terms(code, order)
            print(f"  {run}  순서 {order}")
        res = llm.complete("critique", "", code=code, registry=REGISTRY)
        n_un = sum(1 for t in res["terms"] if not t.get("explainable", True))
        print(f"  {run:32s} 항 {len(res['terms']):2d}  설명 불가 {n_un}")
        for t in res["terms"]:
            mark = "  " if t.get("explainable", True) else "★설명불가"
            print(f"     w[{t['index']}] {mark} {t.get('physics','')[:70]}")
        out.append({"run": run, "code": code, "critique": res,
                    "shuffle_order": order,
                    "exprs": {str(k): v for k, v in
                              term_exprs(code).items()}})
    # ★ LLM 호출은 다시 만들 수 없다 (D-33 / D-51). 반드시 남긴다.
    llm.dump(Path(a.out).with_suffix("") / "llm_calls")
    Path(a.out).write_text(json.dumps(
        {"_model": llm.cfg.model, "_note": "같은 모델이 쓰고 심사했다 — "
         "오류가 상관될 수 있다 (D-85)", "rules": out},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}   호출 {len(a.runs)}")


# --------------------------------------------------------- ablate (LLM 0회)
def cmd_ablate(a) -> None:
    import numpy as np

    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.splits import Split
    from kernelrule.core.weights import fit_weights
    from kernelrule.rules.ablate import AblateError, drop_terms, term_indices

    warnings.simplefilter("ignore")
    d = json.loads(Path(a.judged).read_text())
    table, matrix, _reg = _table_and_matrix()
    groups = _train_groups(table)

    def regret_of(code: str, w0) -> float:
        """체제별로 적합하고 결합한다 — 정준 절차와 같은 축이다."""
        fn = compile_rule(code)
        tot = n = 0.0
        for _name, g in groups.items():
            fr = fit_weights(fn, matrix, table, Split("train", tuple(g)),
                             w0, max_evals=300, warn_invariants=False)
            tot += fr.fit_regret * len(g)
            n += len(g)
        return tot / n

    rows = []
    for item in d["rules"]:
        run, code = item["run"], item["code"]
        crit = item["critique"]
        flagged = {t["index"] for t in crit["terms"]
                   if not t.get("explainable", True)}
        idx = term_indices(code)
        base = regret_of(code, [1.0] * len(idx))
        deltas: dict[int, float] = {}
        refused: dict[int, str] = {}
        for i in idx:
            try:
                cut = drop_terms(code, {i})
            except AblateError as e:
                refused[i] = str(e)
                continue
            try:
                deltas[i] = regret_of(cut, [1.0] * (len(idx) - 1)) - base
            except Exception as e:                          # noqa: BLE001
                refused[i] = f"{type(e).__name__}: {e}"[:80]

        order = sorted(deltas, key=lambda i: deltas[i])      # 덜 아픈 것부터
        ranks = {i: k for k, i in enumerate(order)}
        print(f"\n  {run}   항 {len(idx)}  설명불가 {sorted(flagged)}"
              + (f"  제거불가 {sorted(refused)}" if refused else ""))
        for i in order:
            m = "★설명불가" if i in flagged else "         "
            print(f"     w[{i}] {m} 손상 {deltas[i]:+.4f}  순위 {ranks[i]}/"
                  f"{len(order) - 1}")
        for i, why in refused.items():
            print(f"     w[{i}] — 제거 불가: {why[:60]}")
        rows.append(dict(run=run, n_terms=len(idx), flagged=sorted(flagged),
                         deltas={str(k): v for k, v in deltas.items()},
                         refused={str(k): v for k, v in refused.items()},
                         ranks={str(k): v for k, v in ranks.items()}))

    # ★ 지표: 설명 불가 항의 손상 순위가 아래쪽에 몰리는가
    fr_ranks, other_ranks = [], []
    for r in rows:
        n = len(r["ranks"])
        if n < 2:
            continue
        for k, v in r["ranks"].items():
            (fr_ranks if int(k) in r["flagged"] else other_ranks).append(
                v / (n - 1))
    print("\n" + "=" * 70)
    if fr_ranks:
        print(f"  설명 불가 항의 상대 손상 순위 (0=가장 덜 아픔)  "
              f"중앙 {np.median(fr_ranks):.2f}  n={len(fr_ranks)}")
        print(f"  나머지 항                                        "
              f"중앙 {np.median(other_ranks):.2f}  n={len(other_ranks)}")
        from scipy.stats import mannwhitneyu
        p = mannwhitneyu(fr_ranks, other_ranks, alternative="less").pvalue
        print(f"  Mann-Whitney (설명불가 < 나머지)  p = {p:.4f}")
        print("  ★ 표본 단위는 **항**이고 한 규칙의 항들은 독립이 아니다 "
              "(원칙 28) — 규칙 수가 적으면 이 p 를 믿지 마라")
    else:
        print("  ★ 설명 불가로 지목된 항이 없다 — Critic 이 전부 설명해 냈다")
    Path(a.out).write_text(json.dumps(
        {"_note": "regret 절대값은 보고 대상이 아니다 (D-56 §2). 차이만 쓴다",
         "rules": rows}, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


# ----------------------------------------------------------- rank (LLM)
def cmd_rank(a) -> None:
    """Architect 후보들을 심사해 **학습 점수와 다른 순서**가 되는지 본다.

    ★ 지금 씨앗 선택은 학습 regret 하나로 한다. 4차(D-84)가 보인 것은
    씨앗의 **피처 다양성**이 하류를 규정한다는 것인데 학습 점수는 그것을
    못 본다. **바꾸지는 않는다** — 순서가 얼마나 다른지만 기록한다.
    """
    from scipy.stats import spearmanr

    import kernelrule.features.physical  # noqa: F401
    from kernelrule.agents.openai_client import Budget, LLMConfig, OpenAILLM
    from kernelrule.features import REGISTRY

    # ★ 후보는 `summary.json` 의 `tries` 에 코드와 학습 점수가 함께 있다.
    #   `candidates/` 디렉토리는 `.py` 사본이라 점수가 없다.
    summ = Path(a.campaign) / "stage2-architect" / "summary.json"
    tries = json.loads(summ.read_text())["tries"]
    cands = [t for t in tries if t.get("ok") and t.get("code")
             and t.get("fit_regret") is not None]
    if not cands:
        raise SystemExit(f"{summ} 에 채점된 후보가 없다")

    llm = OpenAILLM(LLMConfig(),
                    feature_names=REGISTRY.names(shape_level=False),
                    shape_values=REGISTRY.names(shape_level=True),
                    registry=REGISTRY, cache=False,
                    budget=Budget(max_calls=len(cands) * 2))
    rows = []
    for c in cands:
        res = llm.complete("critique", "", code=c["code"], registry=REGISTRY)
        n_ok = sum(1 for t in res["terms"] if t.get("explainable", True))
        rows.append(dict(name=f"try{c['i']:02d}", train=c["fit_regret"],
                         n_terms=len(res["terms"]), n_explainable=n_ok,
                         frac=n_ok / max(len(res["terms"]), 1),
                         defects=len(res.get("defects", []))))
        print(f"  {rows[-1]['name']:20s} 설명 가능 {n_ok}/{len(res['terms'])}")
    llm.dump(Path(a.out).with_suffix("") / "llm_calls")     # D-33
    rho, p = spearmanr([r["train"] for r in rows], [-r["frac"] for r in rows])
    print(f"\n  학습 regret 순위 vs '설명 가능 비율' 순위  rho={rho:+.3f} "
          f"p={p:.4f}  n={len(rows)}")
    print("  ★ 기록만 한다 — 씨앗 선택 규칙은 바꾸지 않는다 (§13.4)")
    Path(a.out).write_text(json.dumps(
        {"_model": llm.cfg.model, "rows": rows,
         "spearman": {"rho": rho, "p": p}}, ensure_ascii=False, indent=1))
    print(f"  -> {a.out}")


# --------------------------------------------------------- gate (LLM 0회)
#: 학습 41 을 나눌 fold 수. SOL 오름차순 라운드로빈이라 체제가 자동으로
#: 균형 잡힌다 (fold 마다 느린 형상 3개).
N_FOLDS = 4


def _folds(table, matrix, groups):
    """★ SOL 오름차순 라운드로빈. 단일 30/11 분할은 검증에 느린 형상이
    3개뿐이라 얇다 — 4-fold 면 같은 형상이 네 번 중 한 번은 검증에 온다."""
    train = [p for g in groups.values() for p in g]
    srt = sorted(train, key=lambda p: matrix.for_shape(p)[1].log_sol_ms)
    out = [[] for _ in range(N_FOLDS)]
    for i, p in enumerate(srt):
        out[i % N_FOLDS].append(p)
    return out


def cmd_gate(a) -> None:
    """★ 관문 2 — "쓸모없음" 이 아니라 **"일반화 안 됨"** 을 잰다 (D-87).

    D-85 의 관문은 "빼도 학습 점수가 안 나빠지면 판정이 맞다" 였다.
    **틀린 정의였다** — 항 하나가 중요하면서 동시에 설명 불가일 수 있고,
    그것이야말로 우리가 걱정하는 상황이다.

    ```
    설명 가능한 항   물리라서 다른 형상에도 적용된다
                    -> 빼면 적합/검증이 비슷하게 나빠진다
    설명 불가한 항   학습 형상에 맞춰진 것
                    -> ★ 빼면 적합만 크게 나빠지고 검증은 덜 나빠진다

    지표   (적합 손상) - (검증 손상).  ★ 양수가 클수록 과적합 서명
    ```

    ⚠️ **구조 홀드아웃 20 은 안 건드린다.** 학습 41 안에서만 나눈다.
    """
    import statistics

    import numpy as np

    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.splits import Split, regime_of
    from kernelrule.core.weights import fit_weights, make_score_of
    from kernelrule.rules.ablate import AblateError, drop_terms, term_indices

    warnings.simplefilter("ignore")
    d = json.loads(Path(a.judged).read_text())
    table, matrix, _reg = _table_and_matrix()
    groups = _train_groups(table)
    folds = _folds(table, matrix, groups)
    print(f"  {N_FOLDS}-fold  " + " ".join(
        f"[{len(f)}형상, 느린 {sum(1 for p in f if regime_of(p, table.hw) == 'long')}]"
        for f in folds))

    def one_fold(code: str, k: int) -> tuple[float, float]:
        """fold k 를 검증으로 두고 (적합 regret, 검증 regret)."""
        fn = compile_rule(code)
        n_w = len(term_indices(code))
        va = folds[k]
        tr = [p for j, f in enumerate(folds) if j != k for p in f]
        fit_t = fit_n = val_t = val_n = 0.0
        for name in ("short", "long"):
            gtr = [p for p in tr if regime_of(p, table.hw) == name]
            gva = [p for p in va if regime_of(p, table.hw) == name]
            if not gtr:
                continue
            fr = fit_weights(fn, matrix, table, Split("train", tuple(gtr)),
                             [1.0] * n_w, max_evals=300,
                             warn_invariants=False)
            fit_t += fr.fit_regret * len(gtr)
            fit_n += len(gtr)
            if gva:
                sc = make_score_of(fn, matrix, fr.w)
                from kernelrule.core.scoring import evaluate_scores
                ev = evaluate_scores(sc, table, gva, ks=(1,))
                val_t += ev.at(1) * len(gva)
                val_n += len(gva)
        return fit_t / max(fit_n, 1e-9), val_t / max(val_n, 1e-9)

    rows = []
    for item in d["rules"]:
        run, code = item["run"], item["code"]
        crit = item["critique"]
        flagged = sorted({t["index"] for t in crit["terms"]
                          if not t.get("explainable", True)})
        idx = term_indices(code)
        base = [one_fold(code, k) for k in range(N_FOLDS)]
        diffs: dict[int, float] = {}
        refused: dict[int, str] = {}
        for i in idx:
            try:
                cut = drop_terms(code, {i})
            except AblateError as e:
                refused[i] = str(e)
                continue
            ds = []
            for k in range(N_FOLDS):
                cf, cv = one_fold(cut, k)
                ds.append((cf - base[k][0]) - (cv - base[k][1]))
            diffs[i] = float(np.mean(ds))
        rows.append(dict(run=run, flagged=flagged, n_terms=len(idx),
                         diffs={str(k): v for k, v in diffs.items()},
                         refused={str(k): v for k, v in refused.items()}))
        order = sorted(diffs, key=lambda i: -diffs[i])   # 양수가 큰 것부터
        print(f"\n  {run}  설명불가 {flagged}")
        for r, i in enumerate(order):
            m = "★설명불가" if i in flagged else "         "
            print(f"     w[{i}] {m} (적합-검증) {diffs[i]:+.4f}  "
                  f"순위 {1 - r / max(len(order) - 1, 1):.2f}")

    def score_flagger(pick) -> float | None:
        """규칙마다 지목 항의 상대 순위 중앙값 -> 그 중앙값 (원칙 28)."""
        per = []
        for r in rows:
            ds = {int(k): v for k, v in r["diffs"].items()}
            if len(ds) < 2:
                continue
            order = sorted(ds, key=lambda i: -ds[i])
            rank = {i: 1 - k / (len(order) - 1) for k, i in enumerate(order)}
            f = [rank[i] for i in pick(r, sorted(ds)) if i in rank]
            if f:
                per.append(statistics.median(f))
        return statistics.median(per) if per else None

    real = score_flagger(lambda r, idx: r["flagged"])
    mock1 = score_flagger(lambda r, idx: idx[-1:])
    mockk = score_flagger(lambda r, idx: idx[-len(r["flagged"]):]
                          if r["flagged"] else [])
    rng = np.random.default_rng(20260828)
    rand = statistics.median([
        score_flagger(lambda r, idx: list(rng.choice(
            idx, size=min(len(r["flagged"]), len(idx)), replace=False))
            if r["flagged"] else [])
        for _ in range(200)])
    print("\n" + "=" * 70)
    print("  지목 항의 (적합-검증) 상대 순위 — 규칙 단위 중앙값 (1=가장 양수)")
    print(f"    ★ 실제 Critic          {real if real is None else f'{real:.2f}'}")
    print(f"    목: 마지막 항 하나      {mock1 if mock1 is None else f'{mock1:.2f}'}")
    print(f"    목: 마지막 k개 (수 일치) {mockk if mockk is None else f'{mockk:.2f}'}")
    print(f"    무작위 k개 (200회 중앙)  {rand:.2f}")
    print("    기준: >=0.65 판정이 맞다 / <=0.35 틀리다 / 사이는 구분 불가")
    print("    ★ 목이 0.65 를 넘으면 실제 값이 얼마든 **판정 불가**다")
    Path(a.out).write_text(json.dumps(
        {"_note": "지표는 (적합 손상)-(검증 손상). 양수가 클수록 과적합 서명. "
                  "regret 절대값은 보고 대상이 아니다 (D-56 §2)",
         "n_folds": N_FOLDS, "rules": rows,
         "score": {"critic": real, "mock_last1": mock1,
                   "mock_lastk": mockk, "random_k": rand}},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


# ------------------------------------------------------ compare (LLM 0회)
def cmd_compare(a) -> None:
    """원래 심사와 **순서를 섞은** 심사를 견준다 (D-86 위치 편향).

    항 번호가 아니라 **식**으로 견준다 — 섞으면 번호가 달라지므로.
    """
    import statistics

    from kernelrule.rules.ablate import term_exprs

    def exprs_of(r: dict) -> dict:
        # ★ 첫 심사 산출물에는 `exprs` 가 없다 — 코드에서 다시 읽는다.
        return r.get("exprs") or {str(k): v
                                  for k, v in term_exprs(r["code"]).items()}

    base = json.loads(Path(a.base).read_text())["rules"]
    shuf = json.loads(Path(a.shuffled).read_text())["rules"]
    by = {r["run"]: r for r in shuf}
    jac, pos_hits, pos_tot = [], 0, 0
    print(f"  {'실행':4s} {'원래':>5} {'섞은뒤':>6} {'공통':>5} {'자카드':>7}")
    for b in base:
        sh = by.get(b["run"])
        if sh is None:
            continue
        be, se = exprs_of(b), exprs_of(sh)
        A = {be[str(t["index"])] for t in b["critique"]["terms"]
             if not t.get("explainable", True) and str(t["index"]) in be}
        B = {se[str(t["index"])] for t in sh["critique"]["terms"]
             if not t.get("explainable", True) and str(t["index"]) in se}
        u = A | B
        j = len(A & B) / len(u) if u else 1.0
        jac.append(j)
        n = len(se)
        for t in sh["critique"]["terms"]:
            if not t.get("explainable", True):
                pos_tot += 1
                pos_hits += t["index"] >= n - 2
        print(f"  {b['run'][-2:]:4s} {len(A):5d} {len(B):6d} "
              f"{len(A & B):5d} {j:7.2f}")
    med = statistics.median(jac)
    print(f"\n  ★ 자카드 중앙 {med:.2f}  (n={len(jac)}규칙)")
    print("     >=0.60 편향 아님 / <=0.20 편향 / 사이는 부분 편향")
    if pos_tot:
        print(f"  ★ 섞은 뒤 마지막 두 자리의 지목 비율 {pos_hits}/{pos_tot} "
              f"= {pos_hits / pos_tot:.0%}   (원래 8/13 = 62%)")
    Path(a.out).write_text(json.dumps(
        {"jaccard": jac, "jaccard_median": med,
         "late_position_rate": (pos_hits / pos_tot if pos_tot else None),
         "_note": "식으로 견준다 — 섞으면 w 번호가 달라진다"},
        ensure_ascii=False, indent=1))
    print(f"  -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--runs", nargs="+", required=True)
    j.add_argument("--out", default="docs/artifacts/critic.json")
    j.add_argument("--shuffle", type=int, metavar="SEED",
                   help="항 순서를 섞어 심사한다 (D-86 위치 편향 검사). "
                        "식은 그대로이고 w 인덱스만 다시 매긴다")
    j.set_defaults(fn=cmd_judge)
    b = sub.add_parser("ablate")
    b.add_argument("--judged", default="docs/artifacts/critic.json")
    b.add_argument("--out", default="docs/artifacts/critic-ablation.json")
    b.set_defaults(fn=cmd_ablate)
    r = sub.add_parser("rank")
    r.add_argument("--campaign", required=True)
    r.add_argument("--out", default="docs/artifacts/critic-rank.json")
    r.set_defaults(fn=cmd_rank)
    c = sub.add_parser("compare")
    c.add_argument("--base", default="docs/artifacts/critic.json")
    c.add_argument("--shuffled", default="docs/artifacts/critic-shuffled.json")
    c.add_argument("--out", default="docs/artifacts/critic-shuffle.json")
    c.set_defaults(fn=cmd_compare)
    g = sub.add_parser("gate")
    g.add_argument("--judged", default="docs/artifacts/critic.json")
    g.add_argument("--out", default="docs/artifacts/critic-gate.json")
    g.set_defaults(fn=cmd_gate)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
