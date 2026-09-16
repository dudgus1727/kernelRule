"""★ 형상 N개를 만드는 데 몇 시간인가 (D-177 §3-1). **0 LLM · 0 GPU.**

    python3 -m experiments.porting_shapes_time

## ⚠️ 정정 — 2026-09-16

처음 낸 값은 **150배 부풀어 있었다.** `build_seconds` 를 표의 **행마다**
더했기 때문이다.

```
⛔ 틀린 계산   형상 x 15,015 config x 빌드 16.5초  -> a6000 형상 하나 69시간
★ 사실        `kernel_id` 는 ★ 형상들이 공유한다
              a6000  행 980,915개에 ★ 서로 다른 커널은 6,285개
              같은 커널의 build_seconds 가 156개 행에 복사돼 있었다
★ 맞는 계산   컴파일 = 그 형상들이 요구하는 ★ 서로 다른 커널의 합
              측정   = 행마다 time_ms x (n_reps + warmup)
```

⛔ 틀린 값이 사실과 모순되는 것도 확인된다: a6000 표 전체 컴파일은
**29.4시간**이고 번들에 적힌 실측 기간은 **약 44시간**이다. 형상 하나가
69시간이면 66형상에 4,554시간이어야 한다.

## 그래서 §3-1 의 결론이 뒤집힌다

```
★ 컴파일이 지배하고 · 컴파일은 ★ 형상 수에 거의 안 따른다
형상 4개라도 커널 2,305개(전체의 37%)를 컴파일해야 한다
형상당 한계비용은 ★ 측정 쪽뿐이고 그건 작다 (형상당 17분)
```
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import warnings
from pathlib import Path

import numpy as np

from experiments.f1_pipeline import _splits
from experiments.porting_shapes import FOLD, GPUS, NS
from experiments.transfer_29_5 import TABLES
from kernelrule.core.table import PerfTable

OUT = Path("docs/artifacts/porting-shapes-time.json")
COUNTS = (*NS, 48)
#: ★ 뽑기마다 커널 집합이 달라 비용이 흔들린다. 5회의 중앙을 쓴다.
DRAWS = 5


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    import pandas as pd

    res: dict = {
        "counts": list(COUNTS), "draws": DRAWS,
        "correction": {
            "date": "2026-09-16",
            "wrong": ("build_seconds summed per ROW — a6000 came out 69 "
                      "hours per shape, 3,312 h for N=48"),
            "why": ("kernel_id is shared across shapes: 980,915 rows but "
                    "6,285 distinct kernels on the a6000, and the same "
                    "kernel's build_seconds is copied onto every row"),
            "contradiction": ("the a6000 bundle's whole compile cost is "
                              "29.4 h and its measured_from/to span is ~44 "
                              "h — 66 x 69 h = 4,554 h cannot be right"),
            "right": ("compile = the distinct kernels those shapes need; "
                      "measure = per row")},
        "tables": {}}
    print("=" * 96)
    print("★ 형상 N개를 만드는 데 몇 시간인가 (D-177 §3-1). 0 LLM · 0 GPU")
    print("=" * 96)
    print("  ⚠️ 정정본 — 처음 값은 build_seconds 를 행마다 더해 150배 부풀었다")
    for gpu in GPUS:
        T = TABLES[gpu]
        B = Path(T["bundle"])
        proto = json.loads((B / "BUNDLE.json").read_text())["protocol"]
        d = pd.read_parquet(B / "table.parquet",
                            columns=["M", "N", "K", "kernel_id",
                                     "build_seconds", "time_ms", "n_reps"])
        table = PerfTable.from_bundle(str(B), env_hash=T["env_hash"],
                                      ok_only=False)
        tr = [(p.M, p.N, p.K)
              for p in _splits(table, fold=FOLD, k=4,
                               design="nkband").train.shapes]
        keys = list(zip(d["M"], d["N"], d["K"], strict=True))
        whole_compile = (d.drop_duplicates("kernel_id")["build_seconds"].sum()
                         / 3600)
        rng = np.random.default_rng(0)
        rows: dict = {}
        for n in COUNTS:
            comp, meas, nk = [], [], []
            for _ in range(DRAWS):
                idx = rng.choice(len(tr), min(n, len(tr)), replace=False)
                want = {tr[int(i)] for i in idx}
                sub = d[[k in want for k in keys]]
                comp.append(float(
                    sub.drop_duplicates("kernel_id")["build_seconds"].sum()
                    / 3600))
                nr = sub["n_reps"].to_numpy(float)
                ms = sub["time_ms"].to_numpy(float) * (
                    nr + np.maximum(proto["min_warmup"],
                                    proto["warmup_frac"] * nr))
                meas.append(float(ms.sum() / 3.6e6))
                nk.append(int(sub["kernel_id"].nunique()))
            rows[str(n)] = {
                "compile_hours": round(st.median(comp), 1),
                "compile_range": [round(min(comp), 1), round(max(comp), 1)],
                "measure_hours": round(st.median(meas), 1),
                "total_hours": round(st.median(comp) + st.median(meas), 1),
                "n_kernels_median": int(st.median(nk))}
        res["tables"][gpu] = {
            "n_train": len(tr), "n_rows": int(len(d)),
            "n_kernels_total": int(d["kernel_id"].nunique()),
            "whole_table_compile_hours": round(float(whole_compile), 1),
            "warmup_frac": proto["warmup_frac"],
            "min_warmup": proto["min_warmup"],
            "by_n": rows}
        r = res["tables"][gpu]
        print(f"\n  ── {gpu}  행 {r['n_rows']:,} · ★ 서로 다른 커널 "
              f"{r['n_kernels_total']:,} · 표 전체 컴파일 "
              f"{r['whole_table_compile_hours']:.1f}h")
        for n in COUNTS:
            x = rows[str(n)]
            print(f"     N={n:2d}  커널 {x['n_kernels_median']:6,d}  "
                  f"컴파일 {x['compile_hours']:6.1f}h  "
                  f"측정 {x['measure_hours']:6.1f}h  "
                  f"★ 합 {x['total_hours']:6.1f}h")
    med = {str(n): round(float(np.median(
        [res["tables"][g]["by_n"][str(n)]["total_hours"]
         for g in res["tables"]])), 1) for n in COUNTS}
    res["median_total_hours"] = med
    print("\n  ★ 네 표의 중앙 (합)")
    for n in COUNTS:
        print(f"     N={n:2d}  {med[str(n)]:>7.1f} 시간")
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
