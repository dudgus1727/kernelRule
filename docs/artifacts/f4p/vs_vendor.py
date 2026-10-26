"""현재 (D-199) 규칙과 벤더를 holdout 65 형상에서 견준다 — 그래프용 숫자. LLM 0 · GPU 0.

    python3 docs/artifacts/f4p/vs_vendor.py     # -> docs/artifacts/f4p/vs_vendor.json

```
GPU 별     holdout regret 기하평균 — 벤더 · K1a · D-196 (lam 1) · D-199 (lam 0.1 + 죽은 항 정리, 현재)
분류 별    D-199 와 벤더 (big · hungry · mid · small)
형상 별    D-199 와 벤더의 regret — 이김 (벤더보다 1% 넘게 빠름) · 비김 (1% 안) · 짐
```
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

D = Path(__file__).resolve().parent
F4C = D.parent / "f4c"
GPUS = ("a6000", "5090", "4090", "h100")
K1A_HOLDOUT = {"a6000": 1.0479, "5090": 1.0354, "4090": 1.0380, "h100": 1.0635}
TIE = 0.01


def main() -> None:
    warnings.simplefilter("ignore")
    os.chdir(ROOT)
    from experiments import a6000_probe as P
    from experiments.f1_pipeline import _splits

    out = {}
    for g in GPUS:
        P._G.clear()
        P._G["gpu"] = g
        t = P._table()
        splits = {f: _splits(t, fold=f, k=4, design="nkband") for f in P.FOLDS}
        refs = P._refs(t, splits)
        cur = json.loads((D / g / "refit_cap0.1.json").read_text())
        old = json.loads(((F4C / "refit_cap1.json") if g == "a6000"
                          else F4C / g / "refit_cap1.json").read_text())
        shapes = []
        for f, fr in cur["per_fold"].items():
            for s, r in fr["regret"].items():
                shapes.append({"shape": s, "fold": int(f), "vendor": refs[int(f)]["vendor"][s],
                               "current": r, "d196": old["per_fold"][f]["regret"][s]})
        win = sum(x["current"] < x["vendor"] * (1 - TIE) for x in shapes)
        loss = sum(x["current"] > x["vendor"] * (1 + TIE) for x in shapes)
        a = cur["aggregate"]
        out[g] = {"holdout": {"vendor": a["vendor_gm_65"], "k1a": K1A_HOLDOUT[g],
                              "d196": old["aggregate"]["holdout_gm_65"],
                              "current": a["holdout_gm_65"]},
                  "by_class": {"current": a["holdout_by_class"], "vendor": a["vendor_by_class"]},
                  "wins": win, "ties": len(shapes) - win - loss, "losses": loss,
                  "shapes": shapes}
        print(g, out[g]["holdout"], f"이김 {win} 비김 {len(shapes) - win - loss} 짐 {loss}")
    (D / "vs_vendor.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
