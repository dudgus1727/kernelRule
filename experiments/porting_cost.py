"""★ The porting-cost curve — how many rounds close the gap to native
(D-175 §2). **0 LLM calls.**

    python3 -m experiments.porting_cost

```
(a) 그대로       LLM 0회    the source rule with the source's weights
(b) 재적합       LLM 0회    ★ the seed of these runs — r-1 of the curve
★ r0..r5         LLM 7회/라운드
원주민           LLM 103회  the rule grown on that table and fold
```

⚠️ **Every column is `canonical_score`** — per-regime refit on the target's
training split, read on its holdout. That is the same procedure the transfer
table's `(a)`/`(b)`/`native` use, so the row can be read across. ⛔ The
loop's own `best_val_regret` is a **single** global fit and is not put in
this table.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import warnings
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.f1_pipeline import _load_stage1, _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry, load_generated

ROUNDS = 6
OUT = Path("docs/artifacts/porting-cost.json")
OUT_MD = Path("docs/artifacts/porting-cost.md")


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _best_upto(arc: list[dict], r: int) -> dict | None:
    """★ The best **by training score** among rules born at or before round
    `r`. ⛔ The holdout is not consulted (§10.2)."""
    cand = [e for e in arc if (e.get("round") is None or e["round"] <= r)]
    return min(cand, key=lambda e: e["regret"]) if cand else None


def _structure(code: str, shape_names: set[str]) -> dict:
    terms = {int(m.group(1)) for m in re.finditer(r"w\[(\d+)\]", code)}
    axes = sorted({m.group(1) for m in re.finditer(r"\bf\.(\w+)", code)})
    branch = sorted({m.group(1) for m in re.finditer(r"\bp\.(\w+)", code)
                     if m.group(1) in shape_names})
    return {"n_terms": len(terms), "axes": axes, "branch_axes": branch,
            "n_paths": code.count("np.where")
            + len(re.findall(r"^\s+if ", code, re.M))}


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    # ★ One direction costs 6 canonical scores (per-regime refits at 300
    #   evals). 12 of them is about 90 minutes, so they are sharded by
    #   target table and merged. ⛔ The computation per direction is
    #   unchanged.
    ap.add_argument("--dst", default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    a = ap.parse_args()

    if a.merge:
        rows = []
        for f in a.merge:
            rows.extend(json.loads(Path(f).read_text())["rows"])
        rows.sort(key=lambda r: (r["src"], r["dst"]))
        Path(a.out).write_text(json.dumps(
            {"rounds": ROUNDS,
             "note": ("⚠️ every column is canonical_score (per-regime refit "
                      "on the target's train, read on its holdout) — the "
                      "same procedure as the transfer table's "
                      "(a)/(b)/native. ⛔ the loop's own best_val_regret is "
                      "a single global fit and is not in this table."),
             "rows": rows, "summary": _summary(rows)},
            ensure_ascii=False, indent=1))
        OUT_MD.write_text(_md(rows))
        print(f"merged {len(rows)} directions -> {a.out}")
        _print_summary(_summary(rows))
        return

    rows = []
    print("=" * 112)
    print("★ the porting-cost curve (D-175 §2). 0 LLM calls · "
          "every column is canonical_score")
    print("=" * 112)
    for d in sorted(Path("runs").glob("pc-*-f0")):
        src, dst = d.name[len("pc-"):-len("-f0")].split("2", 1)
        if a.dst and dst != a.dst:
            continue
        run = Path(f"{d}-s0")
        if not (run / "archive.jsonl").exists():
            print(f"  {d.name:24s} ★ 아직 없음")
            continue
        ch = json.loads((d / "stage2-rule-writer" / "chosen.json").read_text())
        T = TABLES[dst]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        splits = _splits(table, fold=0, k=4, design="nkband")
        reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2",
                           table)
        # ★ Axes the loop built during THIS run travel with its rules.
        fp = run / "features.jsonl"
        if fp.exists():
            for f in load_generated(fp, table=table):
                if f.name not in reg._items:
                    reg.add(f)
        matrix = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
        shape_names = {n for n in reg._items if reg[n].shape_level}
        arc = _rows(run / "archive.jsonl")
        rr = _rows(run / "rounds.jsonl")
        curve = []
        for r in range(ROUNDS):
            e = _best_upto(arc, r)
            if e is None:
                curve.append(None)
                continue
            cs = canonical_score(e["code"], np.asarray(e["w"], float),
                                 table=table, matrix=matrix, splits=splits)
            curve.append(round(cs.holdout, 6))
        last = _best_upto(arc, ROUNDS - 1)
        seed_struct = _structure(ch["code"], shape_names)
        fin_struct = _structure(last["code"], shape_names) if last else {}
        kept = (len(set(seed_struct["axes"]) & set(fin_struct.get("axes", [])))
                / max(1, len(seed_struct["axes"])))
        calls = sum(sum((x.get("llm_calls") or {}).values()) for x in rr)
        rows.append({
            "src": src, "dst": dst, "dir": f"{src}->{dst}",
            "a_as_is": ch["a_as_is"], "b_refit": ch["b_refit"],
            "native": ch["native"], "curve": curve,
            "n_rounds": len(rr), "llm_calls": calls,
            "minutes": round(sum(x.get("seconds") or 0 for x in rr) / 60, 1),
            "seed_structure": seed_struct, "final_structure": fin_struct,
            "axis_overlap_with_seed": round(kept, 3),
            "axes_made_in_loop": [j["name"] for j in _rows(fp)
                                  if j.get("accepted")] if fp.exists() else [],
            "n_accepted": sum(x.get("n_accepted") or 0 for x in rr),
            "n_cells": len({tuple(e.get("cell") or []) for e in arc}),
        })
        c = curve
        print(f"  {src:6s}->{dst:6s} (a) {ch['a_as_is']:.4f}  "
              f"(b) {ch['b_refit']:.4f} | "
              + " ".join(f"{x:.4f}" if x else "  —   " for x in c)
              + f" | 원주민 {ch['native']:.4f}  호출 {calls}")
    Path(a.out).write_text(json.dumps(
        {"rounds": ROUNDS,
         "note": ("⚠️ every column is canonical_score (per-regime refit on "
                  "the target's train, read on its holdout) — the same "
                  "procedure as the transfer table's (a)/(b)/native. ⛔ the "
                  "loop's own best_val_regret is a single global fit and is "
                  "not in this table."),
         "rows": rows, "summary": _summary(rows)}, ensure_ascii=False,
        indent=1))
    OUT_MD.write_text(_md(rows))
    print()
    _print_summary(_summary(rows))
    print(f"\n  -> {a.out}\n  -> {OUT_MD}")


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    def first_beat(r):
        for i, v in enumerate(r["curve"]):
            if v is not None and v <= r["native"]:
                return i
        return None
    beats = [first_beat(r) for r in rows]
    gap0 = [r["b_refit"] - r["native"] for r in rows]
    def recov(r, i):
        g = r["b_refit"] - r["native"]
        return (r["b_refit"] - r["curve"][i]) / g if g and r["curve"][i] else None
    return {
        "n_directions": len(rows),
        "beats_native_by_round": {str(i): sum(1 for b in beats if b is not None
                                              and b <= i)
                                  for i in range(ROUNDS)},
        "never_beats": sum(1 for b in beats if b is None),
        "median_first_beat": (st.median([b for b in beats if b is not None])
                              if any(b is not None for b in beats) else None),
        "gap_b_minus_native_median": round(st.median(gap0), 6),
        "gap_recovered_by_r2_median": (
            round(st.median([x for x in (recov(r, 2) for r in rows)
                             if x is not None]), 3)),
        "gap_recovered_by_r5_median": (
            round(st.median([x for x in (recov(r, 5) for r in rows)
                             if x is not None]), 3)),
        "llm_calls_median": st.median([r["llm_calls"] for r in rows]),
        "axis_overlap_with_seed_median": st.median(
            [r["axis_overlap_with_seed"] for r in rows]),
    }


def _print_summary(s: dict) -> None:
    if not s:
        return
    print("★ 요약")
    print(f"  원주민을 넘는 방향 (라운드별 누적) {s['beats_native_by_round']}"
          f"  · 끝까지 못 넘은 것 {s['never_beats']}/{s['n_directions']}")
    print(f"  격차 (b)−원주민 중앙 {s['gap_b_minus_native_median']:+.4f}"
          f"  · r2 까지 회복 {s['gap_recovered_by_r2_median']:.0%}"
          f"  · r5 까지 {s['gap_recovered_by_r5_median']:.0%}")
    print(f"  LLM 호출 중앙 {s['llm_calls_median']:.0f}"
          f"  · 씨앗 축이 최종에 남은 비율 중앙 "
          f"{s['axis_overlap_with_seed_median']:.0%}")


def _md(rows: list[dict]) -> str:
    L = ["# 전이 비용 곡선 (D-175)", "",
         "> **재현** `python3 -m experiments.porting_cost` · LLM 0회",
         ("> ⚠️ 모든 열이 `canonical_score` — 대상의 학습 분할에서 체제별 "
          "재적합하고 홀드아웃에서 읽는다. 전이표의 (a)/(b)/원주민과 같은 "
          "절차다."),
         ("> ⛔ 루프 자신의 `best_val_regret` 은 **단일 적합**이라 이 표에 "
          "넣지 않는다."), "",
         "| 방향 | (a) 그대로 | (b)=r-1 | " +
         " | ".join(f"r{i}" for i in range(ROUNDS)) +
         " | 원주민 | LLM 호출 |",
         "|---|--:|--:|" + "--:|" * ROUNDS + "--:|--:|"]
    for r in rows:
        cur = " | ".join(f"{x:.4f}" if x else "—" for x in r["curve"])
        L.append(f"| {r['dir']} | {r['a_as_is']:.4f} | "
                 f"**{r['b_refit']:.4f}** | {cur} | "
                 f"{r['native']:.4f} | {r['llm_calls']} |")
    L += ["", "★ **굵은 열이 출발점**이고 원주민 열이 도착점이다.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
