"""★ (c) 재생성 사다리 — 옛 / 중간 / 새. LLM 0회.

    python3 experiments/c_ladder.py

사전 등록 `docs/artifacts/c-rerun3-prereg.md`.

```
옛 (c)    숫자 A6000 · 경고 A6000    5090sigma-s{0,1,2}
중간 (c)  숫자 5090  · 경고 A6000    5090sigma-hw-s{0,1,2}
새 (c)    숫자 5090  · 경고 5090     5090sigma-hw2-s{0,1,2}
```

각 단이 **하나씩만** 다르다. regret 은 `sigma_5090.py` 와 **같은 절차**
(정준 채점)로 낸다 — 옛 (c) 의 값이 그 절차에서 나왔다 (원칙 4).

## ⚠️ 기록된 (c) 1.0485 는 **두 씨앗을 섞은 값**이다

`transfer_29_5.TABLES["5090"]["runs"]` 가 여섯을 (c) 로 묶는데, 뒤 셋
(`-b-`)은 `human_guided` **손씨앗**이다. "5090 표에서 처음부터" 가
아니다. 사다리의 첫 단은 RuleWriter 씨앗 3시드(1.0416)를 쓴다.
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from pathlib import Path

import numpy as np
from regret_at_k import _noise_ranks
from scipy.stats import kendalltau
from sigma_5090 import _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.runset import assert_same_condition
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.splits import Split, regime_of
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY

BUNDLE, ENV = "datasets/rtx-5090-sm_120-5bb6f403", "5bb6f403"
DELTA = 0.0516          # ★ §29.5 가 이미 쓴 σ 상한 판정선
TOP_N, KS = 100, (1, 10, 100)

ARMS = [
    ("옛 (c)  숫자A6000·경고A6000", "5090sigma"),
    ("중간(c) 숫자5090 ·경고A6000", "5090sigma-hw"),
    ("새 (c)  숫자5090 ·경고5090", "5090sigma-hw2"),
]
#: ★ 기록된 (c) 에 섞여 있던 손씨앗 팔. 사다리에 안 넣고 **따로** 낸다.
MIXED = ("손씨앗 (기록된 (c) 에 섞임)", "5090sigma-b")


def _arc_best(run: str) -> dict:
    f = Path("runs") / run / "archive.jsonl"
    arc = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return sorted(arc, key=lambda e: e["regret"])[0]


def _taus(code, w0, table, matrix, sp) -> tuple[float, float, float]:
    """상위100 tau / 노이즈 인식 tau / 전구간 tau — 홀드아웃."""
    fn = compile_rule(code)
    ws = {}
    for nm in ("short", "long"):
        g = [q for q in sp.train.shapes if regime_of(q, table.hw) == nm]
        ws[nm] = fit_weights(fn, matrix, table, Split("train", tuple(g)), w0,
                             max_evals=300, objective="regret").w
    rng = np.random.default_rng(12345)
    raw, noi, allr = [], [], []
    for p in sp.val.shapes:
        cand = table.candidates(p)
        s = np.asarray(make_score_of(fn, matrix, ws[regime_of(p, table.hw)])(
            p, cand), dtype=np.float64)
        t = np.asarray(table.times_of(p), dtype=np.float64)
        top = np.argsort(t, kind="stable")[:TOP_N]
        if len(np.unique(t[top])) > 1:
            v = kendalltau(s[top], t[top], variant="b").statistic
            if np.isfinite(v):
                raw.append(float(v))
        nr = _noise_ranks(t, table.noise)[top]
        if len(np.unique(nr)) > 1:
            v = kendalltau(s[top], nr, variant="b").statistic
            if np.isfinite(v):
                noi.append(float(v))
        idx = rng.choice(len(t), size=min(4000, len(t)), replace=False)
        allr.append(float(kendalltau(s[idx], t[idx], variant="b").statistic))
    return (float(np.median(raw)), float(np.median(noi)),
            float(np.median(allr)))


def _seed_shape(tag: str) -> dict:
    d = Path("runs") / f"f1pipe-F3-{tag}" / "stage2-rule-writer"
    cs = {f.name: f.read_text() for f in sorted(d.glob("candidates/*.py"))}
    ch = json.loads((d / "chosen.json").read_text())
    sol = [n for n, c in cs.items() if "log_sol_ms" in c]
    feats = [len({m.attr for m in ast.walk(ast.parse(c))
                  if isinstance(m, ast.Attribute)
                  and isinstance(m.value, ast.Name) and m.value.id == "f"})
             for c in cs.values()]
    return {"n": len(cs), "sol": sol, "chosen_sol": "log_sol_ms" in ch["code"],
            "feats": sorted(feats), "source": ch.get("source"),
            "fit_regret": ch.get("fit_regret")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/c-ladder.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(BUNDLE, env_hash=ENV, ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    sp = _splits(T)
    out: dict = {"delta": DELTA, "n_holdout": len(sp.val.shapes)}

    print("=" * 84)
    print("§1  regret 사다리 — 정준 채점, 5090 홀드아웃 "
          f"{len(sp.val.shapes)}형상, 3시드")
    print("=" * 84)
    print(f"  {'':30s} {'중앙':>8} {'범위':>19}")
    med: dict[str, float] = {}
    for label, tag in [*ARMS, MIXED]:
        # ★ 각 단 **안에서** 조건이 하나여야 한다 (D-120)
        assert_same_condition([f"f1pipe-F3-{tag}-s{i}" for i in range(3)],
                              label=label)
        h = []
        for i in range(3):
            e = _arc_best(f"f1pipe-F3-{tag}-s{i}")
            h.append(canonical_score(e["code"], e["w"], table=T, matrix=M,
                                     splits=sp).holdout)
        h = np.array(h)
        med[tag] = float(np.median(h))
        mark = "   ← 사다리 밖" if tag == MIXED[1] else ""
        print(f"  {label:30s} {np.median(h):8.4f} "
              f"{h.min():8.4f}~{h.max():<8.4f}{mark}")
        out.setdefault("regret", {})[tag] = h.tolist()

    print(f"\n  ★ 기록된 (c) 1.0485 = 위 여섯을 합친 중앙 "
          f"({np.median([*out['regret']['5090sigma'], *out['regret']['5090sigma-b']]):.4f}) "
          "— **두 씨앗을 섞은 값이다**")

    print(f"\n  판정선 delta = {DELTA} (σ 상한, §29.5 가 이미 쓴 값)")
    for a_, b_, name in (("5090sigma", "5090sigma-hw", "옛 -> 중간 (숫자)"),
                         ("5090sigma-hw", "5090sigma-hw2",
                          "중간 -> 새 (경고 절)")):
        d = med[a_] - med[b_]          # 양수면 좋아진 것
        verdict = ("★ 쓰인다" if d >= DELTA else
                   "★ 구분 불가" if abs(d) < DELTA else "구분 불가 (나빠짐)")
        print(f"    {name:22s} {med[a_]:.4f} -> {med[b_]:.4f}  "
              f"차이 {d:+.4f}   {verdict}")
        out.setdefault("steps", {})[name] = d

    print("\n" + "=" * 84)
    print("§2  tau — 경고가 사라지면 상위권이 오르나 (홀드아웃)")
    print("=" * 84)
    print(f"  {'':30s} {'상위100 tau':>12} {'노이즈 인식':>12} {'전구간':>9}")
    for label, tag in ARMS:
        v = np.array([_taus(_arc_best(f"f1pipe-F3-{tag}-s{i}")["code"],
                            _arc_best(f"f1pipe-F3-{tag}-s{i}")["w"], T, M, sp)
                      for i in range(3)])
        print(f"  {label:30s} {np.median(v[:, 0]):12.3f} "
              f"{np.median(v[:, 1]):12.3f} {np.median(v[:, 2]):9.3f}")
        out.setdefault("tau", {})[tag] = v.tolist()

    print("\n" + "=" * 84)
    print("§3  씨앗 10개의 구조 — `p.log_sol_ms` 로 형상 길이를 가르나")
    print("=" * 84)
    for label, tag in ARMS:
        s = _seed_shape(tag)
        print(f"  {label:30s} {len(s['sol'])}/{s['n']}  "
              f"채택 씨앗에 {'있다' if s['chosen_sol'] else '없다'}   "
              f"축 수 {s['feats']}")
        out.setdefault("seed", {})[tag] = s

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")
    print("  ⚠️ 3시드는 유의성을 못 낸다 — 판정선과 범위로 읽는다 (원칙 27)")


if __name__ == "__main__":
    main()
