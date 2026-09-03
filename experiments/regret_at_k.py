"""★ `regret@k` — 벽이 **지표**가 만든 것인가. LLM 0회.

    python3 experiments/regret_at_k.py

사전 등록 `docs/artifacts/regret-at-k-prereg.md`.

## 왜

순위 손실은 **노이즈로 못 가르는 쌍을 뺀다.** `tau` 는 안 뺀다 —
`kendalltau(variant="b")` 는 시간이 **정확히 같을 때만** 동률로 본다.
홀드아웃 상위 100 에서 못 가르는 쌍이 47.2% 인데 tau 가 동률로 보는 것은
14.8% 뿐이다. **32.4% 가 없는 순서를 채점당한다.**

```
regret@k = (규칙이 고른 상위 k 의 참 시간 평균) / (참 상위 k 의 평균)
```

평균이라 노이즈에 강하고, 38등을 36등으로 예측해도 거의 안 벌받는다.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from two_stage import A6000, _fit, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY

KS = (1, 3, 5, 10, 20, 50, 100)
TOP_N = 100
N_DRAWS = 20
CATASTROPHE = 1.15

#: (라벨, 실행들, 아카이브 선택 기준, 가중치 적합 목적함수, k, λ)
ARMS: list[tuple] = [
    ("regret 구조+regret w", [f"f1pipe-F3-arch24-s{i}" for i in range(6)],
     "regret", "regret", 100, 0.0),
    ("★ regret 구조+순위 w", [f"f1pipe-F3-arch24-s{i}" for i in range(6)],
     "regret", "rank", 100, 0.0),
    ("★ 순위 구조+regret w", [f"f1pipe-F3-rankevo-s{i}" for i in range(3)],
     "rank", "regret", 100, 0.0),
    ("순위 구조+순위 w", [f"f1pipe-F3-rankevo-s{i}" for i in range(3)],
     "rank", "rank", 100, 0.0),
    ("곱 항 (prod)", [f"f1pipe-F3-prod-s{i}" for i in range(3)],
     "rank", "rank", 100, 0.0),
    ("k=10", [f"f1pipe-F3-k010-s{i}" for i in range(3)], "rank", "rank",
     10, 0.0),
    ("k=20", [f"f1pipe-F3-k020-s{i}" for i in range(3)], "rank", "rank",
     20, 0.0),
    ("k=50", [f"f1pipe-F3-k050-s{i}" for i in range(3)], "rank", "rank",
     50, 0.0),
    ("λ=1", [f"f1pipe-F3-lam10-s{i}" for i in range(3)], "rank", "rank",
     100, 1.0),
    ("예산 16", [f"f1pipe-F3-b16b-s{i}" for i in range(3)], "rank", "rank",
     100, 0.0),
]


def _best(run: str, by: str) -> dict:
    f = Path("runs") / run / "archive.jsonl"
    arc = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    key = (lambda e: e.get("rank_loss", 1e9)) if by == "rank" \
        else (lambda e: e["regret"])
    return sorted(arc, key=key)[0]


def _noise_ranks(t: np.ndarray, noise) -> np.ndarray:
    """★ 노이즈 안을 **동률로 묶은** 참 순위.

    시간 오름차순으로 훑으며 `resolvable(그룹 첫 원소, t)` 가 True 가
    되면 새 그룹을 연다 (단일 연결). **바닥을 새로 정의하지 않는다** —
    `NoiseModel.resolvable` 을 그대로 쓴다 (원칙 2).
    """
    order = np.argsort(t, kind="stable")
    out = np.empty(len(t), dtype=np.float64)
    g, head = 0, t[order[0]]
    for i in order:
        if bool(noise.resolvable(np.array([head]), np.array([t[i]]))[0]):
            g += 1
            head = t[i]
        out[i] = g
    return out


def _measure(fn, ws, table, matrix, shapes) -> dict:
    """형상별 regret@k / tau 둘 / 파국 목록."""
    rk = {k: [] for k in KS}
    tau_raw, tau_noise, per_shape1 = [], [], {}
    for p in shapes:
        cand = table.candidates(p)
        w = ws[regime_of(p, table.hw)] if isinstance(ws, dict) else ws
        s = np.asarray(make_score_of(fn, matrix, w)(p, cand), dtype=np.float64)
        t = np.asarray(table.times_of(p), dtype=np.float64)
        ts = np.sort(t)
        for k in KS:
            pick = np.asarray(cand.top_k(s, k))
            rk[k].append(float(t[pick].mean() / ts[:k].mean()))
        per_shape1[str(p.key)] = rk[1][-1]
        top = np.argsort(t, kind="stable")[:TOP_N]
        if len(np.unique(t[top])) > 1:
            v = kendalltau(s[top], t[top], variant="b").statistic
            if np.isfinite(v):
                tau_raw.append(float(v))
        nr = _noise_ranks(t, table.noise)[top]
        if len(np.unique(nr)) > 1:
            v = kendalltau(s[top], nr, variant="b").statistic
            if np.isfinite(v):
                tau_noise.append(float(v))
    return {"regret_at_k": {k: float(np.exp(np.mean(np.log(v))))
                            for k, v in rk.items()},
            "tau_raw": float(np.median(tau_raw)) if tau_raw else float("nan"),
            "tau_noise": (float(np.median(tau_noise)) if tau_noise
                          else float("nan")),
            "per_shape_r1": per_shape1}


def _floor(table, matrix, shapes, rng) -> dict:
    """★ 무작위 바닥 (20뽑기 평균). 바닥도 표본이다 (원칙 7)."""
    acc = {k: [] for k in KS}
    for _ in range(N_DRAWS):
        one = {k: [] for k in KS}
        for p in shapes:
            cand = table.candidates(p)
            t = np.asarray(table.times_of(p), dtype=np.float64)
            s = rng.random(len(t))
            ts = np.sort(t)
            for k in KS:
                pick = np.asarray(cand.top_k(s, k))
                one[k].append(float(t[pick].mean() / ts[:k].mean()))
        for k in KS:
            acc[k].append(float(np.exp(np.mean(np.log(one[k])))))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _row(label: str, vals: list[dict]) -> None:
    a = np.array([[v["regret_at_k"][k] for k in KS] for v in vals])
    print(f"  {label:20s} " + " ".join(
        f"{m:6.3f}" for m in np.median(a, axis=0)))
    # ★ 판정은 **시드 범위**로 한다 (사전 등록 §5). 잘라 쓰면 못 읽는다.
    print(f"  {'':20s} " + " ".join(
        f"{a[:, i].min():.2f}-{a[:, i].max():.2f}" for i in range(len(KS))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/regret-at-k.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    hold, train = list(sp.val.shapes), list(sp.train.shapes)
    out: dict = {"ks": list(KS), "n_holdout": len(hold)}

    print("=" * 92)
    print("§1  지금 tau 가 노이즈를 어떻게 다루나 — 홀드아웃 20형상")
    print("=" * 92)
    tot = res = eq = 0
    for p in hold:
        t = np.sort(np.asarray(T.times_of(p)))[:TOP_N]
        iu, ju = np.triu_indices(len(t), k=1)
        tot += iu.size
        res += int(T.noise.resolvable(t[iu], t[ju]).sum())
        eq += int((t[iu] == t[ju]).sum())
    print(f"  상위 100 안의 쌍 {tot:,}")
    print(f"    노이즈로 못 가르는 쌍   {tot - res:,} ({1 - res / tot:.1%})")
    print(f"    tau-b 가 동률로 보는 쌍 {eq:,} ({eq / tot:.1%})  "
          "← 시간이 정확히 같은 것만")
    print(f"  ★ 차이 {(tot - res - eq) / tot:.1%} 가 **없는 순서를 채점당한다**")
    out["pairs"] = {"total": tot, "unresolvable": tot - res, "tied": eq}

    print("\n" + "=" * 92)
    print("§2  regret@k — 전부 홀드아웃 20형상. 위=중앙 아래=시드 범위")
    print("=" * 92)
    print(f"  {'':20s} " + " ".join(f"{'k=' + str(k):>6}" for k in KS)
          + "\n" + f"  {'':20s} " + " ".join(f"{'(범위)':>9}" for _ in KS))
    rows: dict[str, list[dict]] = {}
    for label, runs, by, obj, k, lam in ARMS:
        if not all((Path("runs") / r / "archive.jsonl").exists() for r in runs):
            print(f"  {label:20s} (실행 없음 — 건너뜀)")
            continue
        vals = []
        for r in runs:
            e = _best(r, by)
            fn, ws = _fit(e["code"], e["w"], T, M, train, obj,
                          rank_top_k=k, rank_lambda=lam if obj == "rank" else 0.0)
            vals.append(_measure(fn, ws, T, M, hold))
        rows[label] = vals
        _row(label, vals)
    fl = _floor(T, M, hold, np.random.default_rng(0))
    print(f"  {'★ 무작위 바닥':20s} " + " ".join(f"{fl[k]:6.3f}" for k in KS))
    out["floor"] = {str(k): v for k, v in fl.items()}
    out["arms"] = {lab: [v["regret_at_k"] for v in vs]
                   for lab, vs in rows.items()}

    print("\n" + "=" * 92)
    print("§3  (a) 형상별 regret@1 — 파국(>1.15)이 어디서 오나")
    print("=" * 92)
    print(f"  {'':22s} {'파국 형상 (중앙/20)':>18}  시드 범위   합집합")
    cat: dict[str, set] = {}
    for label, vs in rows.items():
        sets = [{s for s, v in x["per_shape_r1"].items() if v > CATASTROPHE}
                for x in vs]
        allc = set().union(*sets)
        cat[label] = allc
        ns = [len(s) for s in sets]
        print(f"  {label:22s} {np.median(ns):18.1f}  {min(ns):2d}~{max(ns):-2d}"
              f"      {len(allc):2d}")
    if cat:
        common = set.intersection(*[c for c in cat.values() if c]) \
            if all(cat.values()) else set()
        allu = set().union(*cat.values())
        print(f"\n  ★ 모든 팔에서 파국인 형상 {len(common)}개 / "
              f"어느 팔에서든 파국인 형상 {len(allu)}개")
        print("  -> " + ("형상의 성질에 가깝다" if len(common) >= 0.5 * len(allu)
                         else "규칙의 성질에 가깝다"))
        print("  ⚠️ 순위 팔은 20형상 중 13~17개가 파국이다 — **몇 형상에 "
              "몰린 것이 아니다.**")
        out["catastrophe"] = {"per_arm": {k: sorted(v) for k, v in cat.items()},
                              "common": sorted(common), "any": sorted(allu)}

    print("\n" + "=" * 92)
    print("§4  (b) 노이즈 인식 tau — 옛 tau 와 나란히")
    print("=" * 92)
    print(f"  {'':22s} {'옛 tau':>9} {'노이즈 인식':>12} {'차이':>8}")
    for label, vs in rows.items():
        r = float(np.median([v["tau_raw"] for v in vs]))
        n = float(np.median([v["tau_noise"] for v in vs]))
        print(f"  {label:22s} {r:9.3f} {n:12.3f} {n - r:+8.3f}")
        out.setdefault("tau", {})[label] = {"raw": r, "noise": n}

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3·6시드는 유의성을 못 낸다 — 시드 범위로 읽는다 (원칙 27)")


if __name__ == "__main__":
    main()
