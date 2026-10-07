"""D-197 결과표 — lam 1 (D-195 · D-196) 과 0.5 · 0.25 · 0.1. 같은 절차 재적합. LLM 0 · GPU 0.

    python3 docs/artifacts/f4c/lam/table.py
"""

from __future__ import annotations

import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
GPUS = ("a6000", "5090", "4090", "h100")
LAMS = ("1", "0.5", "0.25", "0.1")


def _load(g: str, lam: str) -> dict | None:
    if lam == "1":
        p = D / "refit_cap1.json" if g == "a6000" else D / g / "refit_cap1.json"
    else:
        p = D / "lam" / f"{g}_cap{lam}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    print(f"{'GPU':6s} {'lam':>5s} {'inner-CV':>9s} {'holdout':>8s} {'>1.2':>5s} {'최악':>6s} "
          f"{'벤더':>7s}  holdout fold")
    for g in GPUS:
        for lam in LAMS:
            r = _load(g, lam)
            if r is None:
                print(f"{g:6s} {lam:>5s} -")
                continue
            a = r["aggregate"]
            reg = [v for f in r["per_fold"].values() for v in f.get("regret", {}).values()]
            hf = " ".join(f"{v:.3f}" for v in a["holdout_by_fold"].values())
            print(f"{g:6s} {lam:>5s} {a['inner_cv_gm']:9.4f} {a['holdout_gm_65']:8.4f} "
                  f"{sum(v > 1.2 for v in reg):5d} {max(reg):6.2f} {a['vendor_gm_65']:7.4f}  {hf}")


if __name__ == "__main__":
    main()
