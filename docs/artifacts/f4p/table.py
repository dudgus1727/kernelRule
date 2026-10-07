"""D-199 결과표 — 미리 정한 기준 (holdout <= K1a_G holdout + 0.02) 그대로 읽기만 한다. LLM 0 · GPU 0.

    python3 docs/artifacts/f4p/table.py > docs/artifacts/f4p/table.txt
"""

from __future__ import annotations

import json
import math
from pathlib import Path

D = Path(__file__).resolve().parent
F4C = D.parent / "f4c"
RUNS = D.parents[2] / "runs"
GPUS = ("a6000", "5090", "4090", "h100")
K1A_HOLDOUT = {"a6000": 1.0479, "5090": 1.0354, "4090": 1.0380, "h100": 1.0635}
BAR = {g: round(v + 0.02, 4) for g, v in K1A_HOLDOUT.items()}


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _ref(g: str, which: str) -> float | None:
    p = {"d196": (F4C / "refit_cap1.json") if g == "a6000" else F4C / g / "refit_cap1.json",
         "d197": F4C / "lam" / f"{g}_cap0.1.json"}[which]
    r = _load(p)
    return r["aggregate"]["holdout_gm_65"] if r else None


def _rounds(g: str) -> list[list[dict]]:
    out = []
    for f in range(4):
        p = RUNS / f"f4p-{g}-f{f}-s0" / "rounds.jsonl"
        out.append([json.loads(x) for x in p.read_text().splitlines() if x.strip()]
                   if p.exists() else [])
    return out


def main() -> None:
    print("D-199 — 같은 절차 재적합 (상한 0.1), holdout 65. 판정: holdout <= K1a holdout + 0.02")
    print(f"{'GPU':6s} {'holdout':>8s} {'기준':>7s} {'판정':4s} {'학습':>7s} {'>1.2':>5s} "
          f"{'최악':>6s} {'w (fold)':>14s} {'루프 holdout':>11s} {'D-196':>7s} {'D-197':>7s} "
          f"{'벤더':>7s} {'stage3 시간':>10s} {'지운 w/규칙':>10s}")
    for g in GPUS:
        r = _load(D / g / "refit_cap0.1.json")
        rounds = _rounds(g)
        hours = max((sum(x["seconds"] for x in rs) / 3600 for rs in rounds if rs),
                    default=float("nan"))
        sc = sum(x.get("n_scored", 0) for rs in rounds for x in rs)
        pr = sum(x.get("n_pruned", 0) for rs in rounds for x in rs)
        pruned = f"{pr / sc:.1f}" if sc else "-"
        if r is None:
            print(f"{g:6s} - (평가 전)  stage3 {hours:.1f}h  지운 w/규칙 {pruned}")
            continue
        a = r["aggregate"]
        pf = r["per_fold"]
        reg = [v for f in pf.values() for v in f.get("regret", {}).values()]
        ok = "통과" if a["holdout_gm_65"] <= BAR[g] else "실패"
        # the loop's own holdout: the archive best of the last round, pooled
        num = den = 0.0
        for f, rs in enumerate(rounds):
            if rs and str(f) in pf:
                n = pf[str(f)]["n_hold"]
                num += n * math.log(rs[-1]["best_val_regret"])
                den += n
        loop_ho = math.exp(num / den) if den else float("nan")
        ws = "/".join(str(v["check"]["n_weights"]) for v in pf.values())
        print(f"{g:6s} {a['holdout_gm_65']:8.4f} {BAR[g]:7.4f} {ok:4s} {a['train_gm']:7.4f} "
              f"{sum(v > 1.2 for v in reg):5d} {max(reg):6.2f} {ws:>14s} {loop_ho:11.4f} "
              f"{_ref(g, 'd196'):7.4f} {_ref(g, 'd197'):7.4f} {a['vendor_gm_65']:7.4f} "
              f"{hours:9.1f}h {pruned:>10s}")
    print()
    for g in GPUS:
        r = _load(D / g / "refit_cap0.1.json")
        if r:
            a = r["aggregate"]
            worst = sorted(((v, f, s) for f, fr in r["per_fold"].items()
                            for s, v in fr.get("regret", {}).items()), reverse=True)[:3]
            print(f"  {g:6s} holdout fold {a['holdout_by_fold']}  분류 {a['holdout_by_class']}")
            print(f"  {'':6s} 가장 나쁜 형상: "
                  + ", ".join(f"f{f} {s} {v:.2f}" for v, f, s in worst))


if __name__ == "__main__":
    main()
