"""★ The (b) loop curve — N target shapes, 3 rounds (D-177 §1-2 · §3).
**0 LLM calls** (the 36 runs already happened; this scores them).

    python3 -m experiments.porting_shapes_curve --dst <gpu>
    python3 -m experiments.porting_shapes_curve --merge a.json ...

⚠️ **Every column is the same procedure as (a)**: the weights are refitted
per regime on ★ **that run's N shapes only** and read on the ★ **whole**
fold0 holdout (`porting_shapes._refit`).

```
⛔ not canonical_score with the full train — that would hand the evaluation
   the 48 shapes this experiment is trying not to build
⛔ not the loop's own best_val_regret — a single global fit
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

import kernelrule.features.physical  # noqa: F401
from experiments.f1_pipeline import _load_stage1, _splits
from experiments.porting_shapes import FOLD, GPUS, NS, _refit
from experiments.transfer_29_5 import TABLES
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry, load_generated

ROUNDS = 3
OUT = Path("docs/artifacts/porting-shapes-curve.json")
OUT_MD = Path("docs/artifacts/porting-shapes-curve.md")


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
        Path(a.out).write_text(json.dumps(
            {"rounds": ROUNDS, "ns": list(NS), "rows": rows,
             "prefix": a.prefix,
             "note": ("⚠️ weights refitted on that run's N shapes only, read "
                      "on the whole fold0 holdout — the same procedure as "
                      "(a). ⛔ not canonical_score with the full train.")},
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
        sample = [p for p in splits.train.shapes if (p.M, p.N, p.K) in want]
        val = list(splits.val.shapes)
        reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2", table)
        fp = run / "features.jsonl"
        if fp.exists():
            for f in load_generated(fp, table=table):
                if f.name not in reg._items:
                    reg.add(f)
        matrix = FeatureMatrix(table, reg, cache_dir=CACHE_DIR)
        seed = _refit(ch["code"], ch["w0"], table=table, matrix=matrix,
                      sample=sample, val=val)
        curve = []
        for r in range(ROUNDS):
            e = _best_upto(arc := _rows(run / "archive.jsonl"), r)
            if e is None:
                curve.append(None)
                continue
            curve.append(round(_refit(e["code"], e["w"], table=table,
                                      matrix=matrix, sample=sample,
                                      val=val)["holdout"], 6))
        rr = _rows(run / "rounds.jsonl")
        calls = sum(sum((x.get("llm_calls") or {}).values()) for x in rr)
        a_ref = ch.get("a_refit_holdout")
        rows.append({
            "src": src, "dst": dst, "N": n, "dir": f"{src}->{dst}",
            "a_as_is": ch["a_as_is"], "b_refit_full": ch["b_refit_full"],
            "native": ch["native"],
            "a_refit_N": a_ref,
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
    print(f"\n  ★ 씨앗 대조 불일치 {len(bad)}/{len(rows)}")
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
