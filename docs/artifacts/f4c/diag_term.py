"""D-196 진단 — 한 holdout 형상에서 규칙이 왜 그 config 를 골랐나. 점수를 항별로 나눈다.
**LLM 0 · GPU 0.** 판정에 쓰지 않는다 (판정은 d196_table.py 의 inner-CV).

    python3 docs/artifacts/f4c/diag_term.py <출발 S> <대상 T> <fold> <형상,...> [<w 번호>]

```
가중치    S = T  : <T>/refit_cap1.json · refit_nocap.json (같은 절차 재적합)
          S != T : transfer/<S>_to_<T>.refit_cap1.json · nofit.json
항        w[i] 하나만 켠 점수 - 모두 끈 점수 (main = 모두 끈 점수 = 시간 본항).
          이 조건의 규칙은 가중치에 선형이라 항의 합이 점수다
비교      고른 config (점수 최소) 와 가장 빠른 config — 차이가 0.05 넘는 항만
w 번호    주면 그 항의 상한 (rules/caps.term_caps, train 형상, lam 1) 과, 그 항의 형상 안 폭
          R_s(항, w=1) — caps.py 와 같은 정의 — 를 train · holdout 형상에서 나란히
```
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402
import re  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

D = Path(__file__).resolve().parent


def _weights(s: str, t: str, fold: int) -> dict:
    if s == t:
        src = {"cap1": D / t / "refit_cap1.json", "nocap": D / t / "refit_nocap.json"}
    else:
        src = {"cap1": D / "transfer" / f"{s}_to_{t}.refit_cap1.json",
               "nofit": D / "transfer" / f"{s}_to_{t}.nofit.json"}
    return {k: np.asarray(json.loads(p.read_text())["per_fold"][str(fold)]["w"])
            for k, p in src.items() if p.exists()}


def main() -> None:
    warnings.simplefilter("ignore")
    os.chdir(ROOT)
    import kernelrule.features.physical  # noqa: F401
    from experiments import a6000_probe as P
    from experiments.f1_pipeline import _splits
    from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
    from kernelrule.core.sandbox import compile_rule

    s, t_gpu, fold, shapes = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(",")
    rules = D / "rules" if s == "a6000" else D / s / "rules"
    spec = json.loads((rules / "f4c_evolved_s0.json").read_text())["per_fold"][str(fold)]
    code = spec["code"]
    fn = compile_rule(code)
    W = _weights(s, t_gpu, fold)
    P._G["gpu"], P._G["prefix"], P._G["lib_gpu"] = t_gpu, "f4c", s
    t = P._table()
    sp = _splits(t, fold=fold, k=4, design="nkband")
    m = FeatureMatrix(t, P._registry_fold(t, fold, spec.get("extra_features", [])),
                      cache_dir=CACHE_DIR)
    nw = len(spec["w0"])
    used = sorted({int(x) for x in re.findall(r"w\[(\d+)\]", code)})

    def terms(f, info, w) -> dict:
        z = np.zeros(nw)
        base = np.asarray(fn(f, info, m.hw, z), float)
        out = {"main": base}
        for i in used:
            e = z.copy()
            e[i] = w[i]
            out[f"w{i}"] = np.asarray(fn(f, info, m.hw, e), float) - base
        return out

    print(f"규칙 {s} f{fold} -> 표 {t_gpu}")
    for sh in shapes:
        M, N, K = map(int, sh.split("x"))
        p = next(q for q in sp.val.shapes + sp.train.shapes if (q.M, q.N, q.K) == (M, N, K))
        f, info = m.for_shape(p)
        df = t.frame_for(p).reset_index(drop=True)
        tt = np.asarray(t.times_of(p))
        j = int(np.argmin(tt))
        where = "holdout" if p in sp.val.shapes else "train"

        def cfg(r: int, df=df) -> str:
            return (f"{df.iloc[r]['tile_m']}x{df.iloc[r]['tile_n']}x{df.iloc[r]['tile_k']}"
                    f" sk{df.iloc[r]['split_k']}")

        print(f"\n=== {sh} ({where})  roofline_ratio {float(info.roofline_ratio):.3g}")
        for tag, w in W.items():
            tm = terms(f, info, w)
            tot = sum(tm.values())
            i = int(np.argmin(tot))
            print(f"  [{tag}] 고름 {cfg(i)} {tt[i] * 1e3:.4g}  최적 {cfg(j)} {tt[j] * 1e3:.4g}"
                  f"  regret {tt[i] / tt[j]:.3f}")
            print(f"     {'항':6s} {'고름':>9s} {'최적':>9s} {'차이':>9s}")
            for k, v in tm.items():
                d = v[i] - v[j]
                if abs(d) > 0.05 or k == "main":
                    print(f"     {k:6s} {v[i]:9.2f} {v[j]:9.2f} {d:9.2f}")
            print(f"     {'합':6s} {tot[i]:9.2f} {tot[j]:9.2f} {tot[i] - tot[j]:9.2f}")

    if len(sys.argv) > 5:
        from kernelrule.rules.caps import main_ranges, shape_range, term_caps

        wi = int(sys.argv[5])
        w = W["cap1"]
        cap = term_caps(fn, code, w, m, sp.train.shapes, lam=1.0)[wi]
        print(f"\n w{wi}: 상한 ({cap[0]:.1f}, {cap[1]:.1f})  상한 1 재적합 {w[wi]:.1f}"
              + (f"  상한 없음 {W['nocap'][wi]:.1f}" if "nocap" in W else ""))
        on = np.zeros(nw)
        on[wi] = 1.0
        for name, shs in (("train", sp.train.shapes), ("holdout", sp.val.shapes)):
            rm = main_ranges(m, shs)
            rows = []
            for p, r in zip(shs, rm, strict=True):
                f, info = m.for_shape(p)
                x = (np.asarray(fn(f, info, m.hw, on), float)
                     - np.asarray(fn(f, info, m.hw, np.zeros(nw)), float))
                rows.append((f"{p.M}x{p.N}x{p.K}", shape_range(x), r))
            rr = np.array([a for _, a, _ in rows])
            mm = np.array([b for _, _, b in rows])
            print(f"  {name:7s} {len(rows):3d} 형상  R_s(항 w{wi}=1) 중앙 {np.median(rr):.4g}"
                  f"  최대 {rr.max():.4g}  | R_s(본항) 중앙 {np.median(mm):.1f}"
                  f"  -> |w|*R_s 중앙 {abs(w[wi]) * np.median(rr):.1f}"
                  f" 최대 {abs(w[wi]) * rr.max():.1f}")
            for sh_, a, b in rows:
                if sh_ in shapes:
                    print(f"     {sh_:18s} R_s(항) {a:.4g}  R_s(본항) {b:.1f}"
                          f"  -> |w|*R_s {abs(w[wi]) * a:.1f}")

if __name__ == "__main__":
    main()
