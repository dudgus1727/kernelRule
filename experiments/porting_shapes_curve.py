"""★ The (b) loop curve — N target shapes, 3 rounds (D-177 §1-2 · §3).
**0 LLM calls** (the 36 runs already happened; this scores them).

    python3 -m experiments.porting_shapes_curve --dst <gpu>
    python3 -m experiments.porting_shapes_curve --merge a.json ...

⛔ **2026-09-18 (D-186) — 각 라운드를 ★ 적합 없이 채점한다.**

그 실행들은 이미 **N 형상으로 학습했다** (`--train-shapes`). 그러니 아카이브에
저장된 `w` 가 곧 **N 형상으로 맞춘 답**이고, 채점기가 그것을 다시 맞출 이유가
없다 (D-182 와 같은 갈래).

```
★ 각 라운드      저장된 w 를 그대로 · fold0 홀드아웃 ★ 전체에서 채점
★ 씨앗 (r−1)    옮긴 규칙 + 소스의 w — ★ 적합 없음 = (a) 와 같은 자
적합이 남는 곳   ⛔ (a) 곡선의 `_refit` 뿐이다 (porting_shapes.py)
⛔ 옛 절차       라운드마다 `_refit` 으로 다시 맞췄다 — 그 값은 기록에 남는다
```

★ `r-1` is the seed scored that way, so it must equal (a)'s rep-0 point for
that (direction, N). That identity is checked and reported (`seed_diff`).
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
from experiments.c2_ref import label, native, transfer
from experiments.f1_pipeline import _load_stage1, _splits
from experiments.porting_shapes import FOLD, GPUS, NS
from experiments.transfer_29_5 import TABLES
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.scoring import evaluate_scores, geomean
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry, load_generated

ROUNDS = 3
OUT = Path("docs/artifacts/porting-shapes-curve.json")
OUT_MD = Path("docs/artifacts/porting-shapes-curve.md")


def _score(code: str, w, table, matrix, val) -> float:
    """★ 저장된 `w` 로 홀드아웃 전체를 채점한다. ⛔ 적합하지 않는다 (D-186)."""
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import make_score_of

    e = evaluate_scores(make_score_of(compile_rule(code), matrix,
                                      np.asarray(w, float)),
                        table, val, ks=(1,))
    return float(geomean(e.regret[:, 0]))


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _best_upto(arc: list[dict], r: int) -> dict | None:
    """★ Best **by training score** among rules born at or before round `r`.
    ⛔ The holdout is not consulted."""
    c = [e for e in arc if (e.get("round") is None or e["round"] <= r)]
    return min(c, key=lambda e: e["regret"]) if c else None


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=None)
    # ★ D-178 — the same aggregation for the stratified runs (`ps2-`). The
    #   computation is identical; only which runs are read changes.
    ap.add_argument("--prefix", default="ps",
                    choices=("ps", "ps2"))
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.merge:
        rows: list[dict] = []
        for f in a.merge:
            rows += json.loads(Path(f).read_text())["rows"]
        rows.sort(key=lambda r: (r["N"], r["src"], r["dst"]))
        # ★ D-186 — 참조 칸(원주민 · (a) · (b))은 ⛔ 다시 재지 않는다.
        #   재집계된 c2 를 ★ 여기서 한 번 더 들이민다 — 옛 조각(=고치기
        #   전에 쓰인 shard)이 섞여도 같은 자리를 보게 한다.
        for r in rows:
            tr = transfer(r["src"], r["dst"], FOLD)
            r["a_as_is"], r["b_refit_full"] = tr["a_as_is"], tr["b_refit"]
            r["native"] = native(r["dst"], FOLD)
            r["a_as_is_expected"] = tr["a_as_is"]
            r["seed_diff"] = round(abs(r["seed_scored"] - tr["a_as_is"]), 9)
        Path(a.out).write_text(json.dumps(
            {"rounds": ROUNDS, "ns": list(NS), "rows": rows,
             "prefix": a.prefix, "note_d186": label(),
             "note": ("⚠️ weights refitted on that run's N shapes only, read "
                      "on the whole fold0 holdout — the same procedure as "
                      "(a). ⛔ not canonical_score, which since D-182 does "
                      "not fit at all.")},
            ensure_ascii=False, indent=1))
        # ⛔ the md follows `--out`, not a fixed name — merging the D-178
        #   stratified rows once overwrote D-177's md with them.
        md = Path(a.out).with_suffix(".md")
        md.write_text(_md(rows, prefix=a.prefix))
        print(f"merged -> {a.out}\n  -> {md}  ({len(rows)} rows)")
        _summary(rows)
        return

    rows = []
    print("=" * 112)
    print("★ (b) 루프 곡선 — N 형상 · 3라운드 (D-177). 0 LLM 호출")
    print("=" * 112)
    for d in sorted(Path("runs").glob(f"{a.prefix}-*-f0")):
        m = re.match(rf"{a.prefix}-(\w+)2(\w+)-n(\d+)-f0$", d.name)
        if not m:
            continue
        src, dst, n = m.group(1), m.group(2), int(m.group(3))
        if (a.dst and dst != a.dst) or dst not in GPUS:
            continue
        run = Path(f"{d}-s0")
        if not (run / "archive.jsonl").exists():
            print(f"  {d.name:30s} ★ 아직 없음")
            continue
        ch = json.loads((d / "stage2-rule-writer" / "chosen.json").read_text())
        T = TABLES[dst]
        table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                      ok_only=False)
        splits = _splits(table, fold=FOLD, k=4, design="nkband")
        want = {tuple(x) for x in
                json.loads((d / "train-shapes.json").read_text())}
        n_sample = sum(1 for p in splits.train.shapes
                       if (p.M, p.N, p.K) in want)
        val = list(splits.val.shapes)
        reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2", table)
        fp = run / "features.jsonl"
        if fp.exists():
            for f in load_generated(fp, table=table):
                if f.name not in reg._items:
                    reg.add(f)
        matrix = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
        # ★ D-186 — ⛔ 적합 없이. 씨앗은 옮긴 규칙 + 소스의 w 그대로다
        seed = {"holdout": _score(ch["code"], ch["w0"], table, matrix, val),
                "pooled_regimes": []}
        curve = []
        for r in range(ROUNDS):
            e = _best_upto(arc := _rows(run / "archive.jsonl"), r)
            if e is None:
                curve.append(None)
                continue
            # ★ D-186 — ⛔ 적합 없이. 저장된 w 가 그 라운드의 답이다
            curve.append(round(_score(e["code"], e["w"], table, matrix,
                                      val), 6))
        rr = _rows(run / "rounds.jsonl")
        calls = sum(sum((x.get("llm_calls") or {}).values()) for x in rr)
        # ★ D-186 — 씨앗은 이제 ⛔ 적합 없이 채점된다. 그러면 그것은 곧
        #   전이표의 **(a) 그대로** 와 같은 자다 — 그쪽과 대조한다.
        #   ⛔ 옛 대조 대상(`a_refit_holdout`, N 형상 재적합)은 다른 자다.
        #   ⛔ chosen.json 의 `a_as_is` 도 옛 값이다 — 재집계본을 본다
        tr = transfer(src, dst, FOLD)
        a_ref = tr["a_as_is"]
        rows.append({
            "src": src, "dst": dst, "N": n, "dir": f"{src}->{dst}",
            # ★ D-186 — 재집계된 c2 값. ⛔ chosen.json 의 옛 값이 아니다
            "a_as_is": tr["a_as_is"], "b_refit_full": tr["b_refit"],
            "native": native(dst, FOLD), "n_sample": n_sample,
            "a_as_is_expected": a_ref,
            "a_as_is_chosen_old": ch.get("a_as_is"),
            "b_refit_full_chosen_old": ch.get("b_refit_full"),
            "a_refit_N_old": ch.get("a_refit_holdout"),
            "seed_scored": round(seed["holdout"], 6),
            # ★ the identity check — the seed scored here must be (a)'s point
            "seed_diff": (None if a_ref is None
                          else round(abs(seed["holdout"] - a_ref), 9)),
            "curve": curve,
            "best_round": (min((v for v in curve if v is not None),
                               default=None)),
            "n_rounds": len(rr), "llm_calls": calls,
            "minutes": round(sum(x.get("seconds") or 0 for x in rr) / 60, 1),
            "n_accepted": sum(x.get("n_accepted") or 0 for x in rr),
            "n_archive": len(arc),
            "pooled_regimes": seed["pooled_regimes"]})
        r = rows[-1]
        print(f"  {src:5s}->{dst:5s} N={n:2d}  씨앗 {r['seed_scored']:.4f}"
              f" (대조 {r['seed_diff']})  | "
              + " ".join(f"{x:.4f}" if x else "  —   " for x in curve)
              + f" | 전체(b) {r['b_refit_full']:.4f}  "
                f"원주민 {r['native']:.4f}  호출 {calls}", flush=True)
    Path(a.out).write_text(json.dumps(
        {"rounds": ROUNDS, "ns": list(NS), "rows": rows},
        ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


def _summary(rows: list[dict]) -> None:
    bad = [r for r in rows if r["seed_diff"] is not None
           and r["seed_diff"] > 1e-9]
    print(f"\n  ★ 씨앗 대조 (= 전이표의 (a) 그대로) 불일치 "
          f"{len(bad)}/{len(rows)}")
    print(f"  {'N':>3} {'r2 격차중앙':>11} {'최선 격차중앙':>13} "
          f"{'원주민넘음':>10} {'호출중앙':>9}")
    for n in NS:
        rs = [r for r in rows if r["N"] == n]
        if not rs:
            continue
        g2 = [r["curve"][-1] - r["native"] for r in rs if r["curve"][-1]]
        gb = [r["best_round"] - r["native"] for r in rs if r["best_round"]]
        beat = sum(1 for r in rs if r["best_round"]
                   and r["best_round"] < r["native"])
        print(f"  {n:>3} {st.median(g2):>+11.4f} {st.median(gb):>+13.4f} "
              f"{beat:>7}/{len(rs)} {st.median([r['llm_calls'] for r in rs]):>9.0f}")


def _md(rows: list[dict], prefix: str = "ps") -> str:
    who = ("D-177 · ★ 무작위 뽑기" if prefix == "ps"
           else "D-178 · ★ 층화 뽑기")
    L = [f"# The (b) loop curve — N target shapes ({who})", "",
         ("> **reproduce** `python3 -m experiments.porting_shapes_curve "
          "--dst <gpu>` then `--merge` · 0 LLM calls"),
         ("> ⚠️ weights refitted on that run's **N shapes only**, read on "
          "the **whole** fold0 holdout"), "",
         "| dir | N | seed (a) | r0 | r1 | r2 | full (b) | native | LLM |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        c = " | ".join(f"{x:.4f}" if x else "—" for x in r["curve"])
        L.append(f"| {r['dir']} | {r['N']} | {r['seed_scored']:.4f} | {c} | "
                 f"{r['b_refit_full']:.4f} | {r['native']:.4f} | "
                 f"{r['llm_calls']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
