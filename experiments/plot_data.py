"""★ 그림 데이터 세 벌 (D-186 §7). **LLM 0회 · GPU 0 · 계산 0회.**

    python3 -m experiments.plot_data

이미 있는 산출물을 긴 꼴(long form) csv 로 옮겨 적을 뿐이다. ⛔ 아무것도
다시 재지 않는다. 오토튜닝 곡선의 csv 는 `autotune_report --merge` 가
직접 쓰므로 여기서 손대지 않는다.

```
porting-cost.csv    전이 비용 곡선   dir,round,regret  (round: -2=(a) -1=(b))
porting-n.csv       N 곡선           dir,N,arm,regret  (arm: a_refit · b_loop)
autotune-curve.csv  ⛔ autotune_report 가 쓴다 — 여기서 만들지 않는다
```

⚠️ 모든 값이 ★ 재집계본이다 (D-182 · D-183 · D-186). 원주민은 `native`
열로 같이 실어 그림에서 기준선으로 쓸 수 있게 한다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

A = Path("docs/artifacts")


def _rows_cost() -> list[dict]:
    d = json.loads((A / "porting-cost.json").read_text())
    out = []
    for r in d["rows"]:
        out.append({"dir": r["dir"], "round": -2, "regret": r["a_as_is"],
                    "native": r["native"]})
        out.append({"dir": r["dir"], "round": -1, "regret": r["b_refit"],
                    "native": r["native"]})
        for i, v in enumerate(r["curve"]):
            if v is not None:
                out.append({"dir": r["dir"], "round": i, "regret": v,
                            "native": r["native"]})
    return out


def _rows_n() -> list[dict]:
    out = []
    for arm, f, key in (("a_refit_random", "porting-shapes.json",
                         "holdout_median"),
                        ("a_refit_strat", "porting-strat.json",
                         "holdout_median"),
                        ("b_loop_random", "porting-shapes-curve.json", None),
                        ("b_loop_strat", "porting-strat-curve.json", None)):
        p = A / f
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            v = r[key] if key else r["curve"][-1]
            if v is None:
                continue
            out.append({"dir": r.get("dir") or f"{r['src']}->{r['dst']}",
                        "N": r["N"], "arm": arm, "regret": round(v, 6),
                        "native": round(r["native"], 6)})
    return out


def _write(name: str, rows: list[dict], fields: list[str]) -> None:
    p = A / name
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {p}  ({len(rows)} rows)")


def main() -> None:
    print("★ 그림 데이터 (D-186 §7) — 계산 0회, 옮겨 적기만 한다")
    _write("porting-cost.csv", _rows_cost(), ["dir", "round", "regret",
                                              "native"])
    _write("porting-n.csv", _rows_n(), ["dir", "N", "arm", "regret",
                                        "native"])
    p = A / "autotune-curve.csv"
    print(f"  ★ {p} 는 `autotune_report --merge` 가 쓴다 — 손대지 않음 "
          f"({'있음' if p.exists() else '⚠️ 없음'})")


if __name__ == "__main__":
    main()
