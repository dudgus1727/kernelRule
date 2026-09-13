"""★ Ten stage-1 libraries, compared — and one chosen. **0 LLM calls.**

    python3 experiments/stage1_sweep.py                 # compare and choose
    python3 experiments/stage1_sweep.py --tags s1sw-0 …  # a different set

## Why

`c21-lib` — the library the 21-run campaign shared — turned out short of
**shape-level axes that are not constant**, and those are the material a
rule branches on. Four of the campaign's twenty-one best rules branch on
`roofline_ratio` and nothing else. F1 failed at exactly this spot: it had
two shape-level axes and both were constant, so branching was dead.

★ So the library is **chosen**, not taken as it comes. Ten stage-1 runs
under the same condition, and the criteria below were fixed **before the
runs finished** (D-169 §3) — choosing them after seeing the numbers is
post-hoc selection.

## The criteria, in order

```
1st  the number of axes that are shape level ★ and not constant  (more)
2nd  the number of covered physics terms                         (more)
3rd  the largest Spearman between two of its own axes            (less)
tie  -> the next criterion; all tied -> the smaller seed
```

⛔ Not used: holdout performance (a test used for selection is not a
test), anything from running the loop, the dead-term ratio (that is the
loop's doing, not the library's), the axis count itself.

★ All three are computed from the stage-1 artefacts and read no answer:
`shape_level`, "is it constant" and the between-axis Spearman are all
functions of `X = (p, hw, cfg)`.

## ⚠️ 2026-09-12 — the first criterion was measuring the wrong population

It counted "not constant" over **the table's 66 shapes**. The experiments
run on **61** (`kernelrule.core.splits.experiment_shapes`, D-167 §R), and
D-167 §O had already established that the alignment axes are constant on
those 61 and vary *only* on the 5 excluded ones. So an axis that cannot
branch anywhere an experiment looks was scoring the first criterion.

★ Corrected to `experiment_shapes(table)`. That is the shape population,
not the training split — both sides of every split live in it, so nothing
about the answer is read. The 66-shape count is kept beside it as
`n_shape_level_varying_table` so the correction is visible.

⛔ The **order** of the criteria was not touched. What was wrong was what
the first one measured, not which one comes first.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
ENV = "c63710df"
OUT_JSON = Path("docs/artifacts/stage1-sweep.json")
OUT_MD = Path("docs/artifacts/stage1-sweep.md")
CHOSEN = Path("docs/artifacts/stage1-sweep-chosen.md")


def _measure(tag: str, table) -> dict:
    import numpy as np

    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features import FeatureRegistry
    from kernelrule.features import generated as gen
    from kernelrule.features.loader import load_generated
    from kernelrule.features.validate import _spearman

    d = Path(f"runs/{tag}/stage1-features")
    props = [json.loads(x) for x in
             (d / "proposals.jsonl").read_text().splitlines() if x.strip()]
    summary = json.loads((d / "summary.json").read_text())
    ranges = json.loads((d / "observed-ranges.json").read_text())

    # ★ `table=` re-derives shape_level on this table (§30.12 / D-67).
    feats = load_generated(d / "proposals.jsonl", table=table)
    reg = FeatureRegistry(tag)
    for f in feats:
        reg.add(f)
    matrix = FeatureMatrix(table, reg)

    # -- 1st: shape level and not constant -------------------------------
    from kernelrule.core.splits import experiment_shapes
    # ★ 2026-09-13 (D-170 §1): this is **65** now, and the criterion behind
    #   it is "more than one kernel family", not alignment. Every row of the
    #   table below — the ten older libraries included — is re-measured on
    #   it, so the column is comparable across rows. The numbers recorded in
    #   `stage1-sweep.md` on 2026-09-12 were on the 61.
    used = experiment_shapes(table)
    shape_level, not_const, not_const_table = [], [], []
    #: ★ How thin a "varying" axis is: name -> (distinct values, the size of
    #: the smallest group). An axis that takes a second value on 4 of 65
    #: shapes is not the same material as one that separates 30 from 35, and
    #: a bare count of varying axes hides that.
    spread: dict = {}
    for f in feats:
        if not f.shape_level:
            continue
        shape_level.append(f.name)
        vals = np.array([float(getattr(matrix.for_shape(p)[1], f.name))
                         for p in used])
        if len(np.unique(np.round(vals, 12))) > 1:
            not_const.append(f.name)
            _, counts = np.unique(np.round(vals, 12), return_counts=True)
            spread[f.name] = [int(len(counts)), int(counts.min())]
        vt = np.array([float(getattr(matrix.for_shape(p)[1], f.name))
                       for p in table.shapes()])
        if len(np.unique(np.round(vt, 12))) > 1:
            not_const_table.append(f.name)

    # -- 3rd: the largest Spearman between two of its own axes -----------
    # ★ on the 61 too, and **constant columns are skipped** — the Spearman
    #   of two constants is not a duplicate finding, it is a division by
    #   zero wearing a 1.000.
    cols = gen._reference_columns(table, matrix, FeatureRegistry("empty"),
                                  train_shapes=used)
    names = [n for n in sorted(cols)
             if len(np.unique(np.round(np.asarray(cols[n], float), 12))) > 1]
    n_const_cols = len(cols) - len(names)
    worst = (0.0, "", "")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if len(cols[a]) != len(cols[b]):
                continue
            sp = abs(_spearman(cols[a], cols[b]))
            if sp > worst[0]:
                worst = (float(sp), a, b)

    pc = summary.get("physics_coverage") or {}
    from experiments.f1_pipeline import _physics_coverage
    cov = _physics_coverage(table, reg, FeatureRegistry("empty"))
    rej = Counter(r.get("error", "?").split(":")[0]
                  for r in props if not r.get("accepted"))
    rng = ranges.get("features") or {}
    ratios = [v.get("span_ratio") for v in rng.values()
              if isinstance(v, dict) and v.get("span_ratio")]
    return {
        "tag": tag,
        # ★ the three criteria
        "n_shape_level_varying": len(not_const),
        "n_shape_level_varying_table": len(not_const_table),
        "n_covered": pc.get("_n_covered", 0),
        # ★ D-170 §2 changed the duplication comparison set (12 shapes all
        #   aligned -> 21 covering every (M, alignment) group), so the
        #   coverage recorded in a run's own summary.json and one computed
        #   today are **different procedures**. Both are carried: `n_covered`
        #   is what that run recorded, `n_covered_now` is every library put
        #   through today's procedure.
        "n_covered_now": cov.get("_n_covered", 0),
        "n_monotone_only_now": cov.get("_n_monotone_only", 0),
        "max_spearman": round(worst[0], 4),
        # what they are, so the table can be read
        "shape_level": shape_level,
        "shape_level_varying": not_const,
        "shape_level_varying_spread": spread,
        "shape_level_varying_table_only": sorted(
            set(not_const_table) - set(not_const)),
        "n_constant_columns": n_const_cols,
        "n_monotone_only": pc.get("_n_monotone_only", 0),
        "n_terms": pc.get("_n_terms", 0),
        "max_spearman_pair": [worst[1], worst[2]],
        "n_used_shapes": len(used), "n_table_shapes": len(table.shapes()),
        # §3-2 — recorded, never used to choose
        "n_accepted": summary.get("n_accepted"),
        "n_planned": summary.get("n_planned"),
        "rejections": dict(rej),
        "n_default_range": sum(
            1 for r in props if r.get("accepted")
            and list(r.get("expected_range") or []) in ([0, 1], [0.0, 1.0])),
        "span_ratio_median": ranges.get("span_ratio_median"),
        "span_ratio_max": (max(ratios) if ratios else None),
        "n_empty_rationale": sum(1 for r in props if r.get("accepted")
                                 and not (r.get("rationale") or "").strip()),
        "needs_recheck": summary.get("shape_level_needs_recheck") or [],
        "names": sorted(f.name for f in feats),
        "seconds": summary.get("seconds"),
        "n_dup_shapes": gen.DUP_SHAPES,
        "n_spread": len(gen._spread_shapes(table, gen.DUP_SHAPES)),
    }


def _rank(rows: list[dict]) -> list[dict]:
    """★ The order was fixed before the numbers existed (D-169 §3)."""
    return sorted(rows, key=lambda r: (-r["n_shape_level_varying"],
                                       -r["n_covered"], r["max_spearman"],
                                       int(r["tag"].rsplit("-", 1)[1])))


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=[f"s1sw-{k}" for k in range(10)])
    # ★ D-170 §6: the known7 confirmation runs are **compared** against the
    #   ten, not ranked with them. The choice of D-169 §3 stands, and a
    #   ranking across two conditions would read as a new one.
    ap.add_argument("--compare-only", action="store_true",
                    help="write the table, choose nothing (two conditions "
                         "in one table cannot be ranked against each other)")
    ap.add_argument("--out", default=None,
                    help="artefact stem; default docs/artifacts/stage1-sweep")
    a = ap.parse_args()
    global OUT_JSON, OUT_MD
    if a.out:
        OUT_JSON, OUT_MD = Path(f"{a.out}.json"), Path(f"{a.out}.md")

    from kernelrule.core.table import PerfTable
    table = PerfTable.from_bundle(BUNDLE, env_hash=ENV, ok_only=False)

    rows = []
    for tag in a.tags:
        if not Path(f"runs/{tag}/stage1-features/summary.json").exists():
            raise SystemExit(f"{tag} 의 stage 1 산출물이 없다.")
        rows.append(_measure(tag, table))
        r = rows[-1]
        print(f"  {tag:8s} 형상축(비상수) {r['n_shape_level_varying']:2d}"
              f"  덮임 {r['n_covered']}/{r['n_terms']}"
              f"  최대 sp {r['max_spearman']:.4f}"
              f"  채택 {r['n_accepted']}/{r['n_planned']}")

    ranked = rows if a.compare_only else _rank(rows)
    best = ranked[0]
    names = Counter(n for r in rows for n in r["names"])
    repeated = {k: v for k, v in names.items() if v > 1}

    out = {"criteria": ["n_shape_level_varying (desc)", "n_covered (desc)",
                        "max_spearman (asc)", "seed (asc)"],
           "fixed_before_results": True,
           "bundle": BUNDLE, "split": "nk11008 train (21실행과 같다)",
           "chosen": (None if a.compare_only else best["tag"]),
           "compare_only": bool(a.compare_only),
           "ranked": [r["tag"] for r in ranked],
           "repeated_axis_names": dict(
               sorted(repeated.items(), key=lambda kv: (-kv[1], kv[0]))),
           "libraries": rows}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    if a.compare_only:
        print("\n  ★ 비교만 한다 — 고르지 않는다 (D-169 §3 의 선택이 "
              "유효하고, 조건이 둘인 표에서 순위는 의미가 없다)")
    else:
        print(f"\n  ★ 고른 것: {best['tag']}  "
              f"(형상축 {best['n_shape_level_varying']} · "
              f"덮임 {best['n_covered']} · 최대 sp {best['max_spearman']:.4f})")
    print(f"  recorded: {OUT_JSON}")
    _write_md(out, ranked, repeated)


def _write_md(out: dict, ranked: list[dict], repeated: dict) -> None:
    L = ["# ★ stage 1 열 번 — 그리고 고른 라이브러리", "",
         "> **재현** `python3 experiments/stage1_sweep.py` · LLM 0회",
         (f"> **원본** `stage1-sweep.json` · **조건** F2 · 표 "
          f"`{out['bundle']}` · 분할 21실행과 같음"),
         ("> ★ **기준은 결과가 나오기 전에 박았다** (D-169 §3) — 결과를 "
          "보고 기준을 정하면 사후 선택이다"), "",
         "## 기준 (순서대로)", "", "```",
         "1순위  형상 수준이면서 ★ 상수가 아닌 축의 수      (많을수록)",
         "2순위  물리 커버리지에서 덮인 항의 수              (많을수록)",
         "3순위  축끼리 최대 spearman                        (낮을수록)",
         "동률   -> 다음 순위 -> 그래도 동률이면 시드가 작은 것",
         "",
         "⛔ 안 쓴 것: 홀드아웃 성능 · 루프 결과 · 죽은 항 비율 · 축의 개수",
         "★ 셋 다 stage 1 산출물만으로 계산되고 답을 안 읽는다",
         "```", "",
         "## 비교표", "",
         ("| 라이브러리 | ★1 형상축(비상수) | ★2 덮임 | ★3 최대 sp | "
          "채택/제안 | range 기본값 | 범위비 중앙 | 범위비 최대 | "
          "rationale 빈 | recheck | 초 |"),
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in ranked:
        mark = " ★" if r["tag"] == out["chosen"] else ""
        sm = (f"{r['span_ratio_median']:.3g}"
              if r["span_ratio_median"] is not None else "—")
        sx = (f"{r['span_ratio_max']:.3g}"
              if r["span_ratio_max"] is not None else "—")
        L.append(
            f"| `{r['tag']}`{mark} | {r['n_shape_level_varying']} | "
            f"{r['n_covered']}/{r['n_terms']} | {r['max_spearman']:.4f} | "
            f"{r['n_accepted']}/{r['n_planned']} | {r['n_default_range']} | "
            f"{sm} | {sx} | {r['n_empty_rationale']} | "
            f"{len(r['needs_recheck'])} | {r['seconds']:.0f} |")
    best = ranked[0]
    L += ["", (f"★ 고른 것은 **`{best['tag']}`** — 형상 수준이면서 상수가 "
               f"아닌 축이 {best['n_shape_level_varying']}개로 가장 많다."), ""]
    L += ["그 축들:", "", "```"]
    for n in best["shape_level_varying"]:
        L.append(n)
    L += ["```", "",
          "## 10개 사이에 반복된 축 이름", "",
          "★ 21실행에서 독립 실행들이 같은 축을 만든 것과 같은 관찰이다.", "",
          "| 축 | 몇 개 라이브러리에 |", "|---|--:|"]
    for k, v in list(repeated.items())[:20]:
        L.append(f"| `{k}` | {v} |")
    n_all = len({n for r in out["libraries"] for n in r["names"]})
    L += ["", f"반복된 이름 {len(repeated)}개 / 전체 서로 다른 이름 {n_all}개",
          ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"  recorded: {OUT_MD}")


if __name__ == "__main__":
    main()
