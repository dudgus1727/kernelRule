"""D-193 진단 — F4 시드룰을 K1a 쪽으로 한 조각씩 바꿔 loop 적합으로 잰다.
**LLM 0 · GPU 0.** 판정에 쓰지 않는다 — 미리 정한 판정은 refit.json 의 inner-CV 다.

    python3 docs/artifacts/f4-seed/ablate.py > docs/artifacts/f4-seed/ablate.txt

```
A  시드룰 그대로
B  split-K 항 splitk_excess_log -> splitk_roofline_log_time (K1a 의 것)
C  + tm_cta_warps · tm_regstaged (K1a 의 보정 둘, 없는 것만; 시작값은 K1a 의 w0)
D  B + C
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
import warnings  # noqa: E402

import numpy as np  # noqa: E402


def add_terms(code: str, w0: list, terms: list) -> tuple[str, list]:
    """`s = s + f.<x> * w[k]` before the final `return s`."""
    lines = code.rstrip().split("\n")
    assert lines[-1].strip() == "return s"
    k = len(w0)
    new = [f"    s = s + f.{x} * w[{k + i}]" for i, (x, _) in enumerate(terms)]
    return ("\n".join(lines[:-1] + new + [lines[-1]]) + "\n",
            list(w0) + [v for _, v in terms])


def variants(code: str, w0: list) -> dict:
    b = code.replace("f.splitk_excess_log", "f.splitk_roofline_log_time")
    extra = [(x, v) for x, v in (("tm_cta_warps", 1.6), ("tm_regstaged", 0.6))
             if f"f.{x}" not in code]
    return {"A 시드룰 그대로": (code, w0),
            "B split-K 항 -> splitk_roofline_log_time": (b, w0),
            "C + tm_cta_warps · tm_regstaged": add_terms(code, w0, extra),
            "D = B + C": add_terms(b, w0, extra)}


def main() -> None:
    warnings.simplefilter("ignore")
    os.chdir(ROOT)
    import kernelrule.features.physical  # noqa: F401
    from experiments import a6000_probe as P
    from experiments.f1_pipeline import _load_stage1, _splits
    from kernelrule.core.loop import _fit_rule
    from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.features import REGISTRY
    from kernelrule.features.loader import base_registry

    P._G["gpu"] = "a6000"
    t = P._table()
    rows: dict = {}
    for f in range(4):
        d = Path(f"runs/f4-a6000-f{f}")
        ch = json.loads((d / "stage2-rule-writer" / "chosen.json").read_text())
        sp = _splits(t, fold=f, k=4, design="nkband")
        reg = _load_stage1(d, base_registry("F4", human=REGISTRY), "F4", t)
        m = FeatureMatrix(t, reg, cache_dir=CACHE_DIR)
        for name, (code, w0) in variants(ch["code"], ch["w0"]).items():
            fr = _fit_rule(compile_rule(code), code, w0, None, matrix=m,
                           table=t, train=sp.train, val=sp.val,
                           objective="regret", rank_top_k=100,
                           rank_lambda=0.0, fit_space="u", fit_budget="dim")
            rows.setdefault(name, []).append(
                (len(sp.val.shapes), fr.fit_regret, fr.val_regret))
            print(f"  f{f}  {name:42s} train {fr.fit_regret:.4f}  holdout "
                  f"{fr.val_regret:.4f}", flush=True)
    print()
    for name, v in rows.items():
        n = sum(x[0] for x in v)
        tr = float(np.exp(np.mean([np.log(x[1]) for x in v])))
        ho = float(np.exp(sum(x[0] * np.log(x[2]) for x in v) / n))
        print(f"  {name:42s} train(4 fold gm) {tr:.4f}  holdout pooled "
              f"{ho:.4f}")


if __name__ == "__main__":
    main()
