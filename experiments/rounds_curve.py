"""★ Is 12 rounds right — it looks at **the recorded curves only**. 0 LLM
calls.

    python3 experiments/rounds_curve.py

The pre-registration is `docs/artifacts/rounds-prereg.md`.

"early stopping did not trigger" is not "it converged" — with
`patience=10` and 12 rounds there are only two decision windows, and it does
not stop if even one new cell appears.

★ The significance threshold is exactly the one the loop uses
(`is_significant`). No new criterion is made (principle 2).

⚠️ 2026-09-08 (D-146): **the verdict strings stay in Korean.** They are the
three verdicts written down in `docs/artifacts/rounds-prereg.md` and stored
in `rounds-curve.json`, and `docs/` is not translated.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from two_stage import A6000, _splits

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import evaluate_scores
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import make_score_of
from kernelrule.features import REGISTRY

#: (tag, the run directory prefix, the number of seeds). After the D-128
#: rename the two are the same.
GROUPS = [("F3rw-p8", "F3rw-p8", 6),
          ("F1rw-p8", "F1rw-p8", 6),
          ("F2rw-p8", "F2rw-p8", 6)]
#: The patience values to review. **They are not changed** — it only looks at
#: "when would it have stopped if it had been that".
PATIENCES = (3, 5, 7, 10)
#: The "last 3 rounds" used for the decision (the round field of the file,
#: counting from 0)
LAST3 = (9, 10, 11)


def _r(x) -> str:
    """'none' if `None`. Round numbers count from 0."""
    return "none" if x is None else f"r{x}"


def _rows(run: str) -> list[dict]:
    f = Path("runs") / run / "rounds.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()]


def _tol(run: str, T, M, hold) -> float | None:
    """The **noise threshold** measured with that run's final best rule (the
    same path as the loop).

    ★ `None` means **it could not be computed** — an F1/F2 run's rule refers
    to features made inside the loop, and the base registry does not have
    those axes. It is not filled in with another value (principle 2). The
    caller marks it as an approximation and uses it.
    """
    f = Path("runs") / run / "archive.jsonl"
    arc = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    e = sorted(arc, key=lambda z: z["regret"])[0]
    try:
        fn = compile_rule(e["code"])
        ev = evaluate_scores(make_score_of(fn, M, np.asarray(e["w"], float)),
                             T, list(hold), ks=(1,))
    except AttributeError:
        return None
    from kernelrule.core.scoring import geomean
    return float(geomean(ev.tol))


def _stop_round(vals: list[float], n: int, thr: float,
                cells: list[int]) -> int | None:
    """If `patience=n`, **at the end of which round** would it have stopped
    (the loop's formula).

    ★ The new-cell condition is used as it is too — the case where that stops
    it from stopping is the whole point.
    """
    for end in range(n, len(vals)):
        improved = vals[end - n] - vals[end]
        # Did a new cell appear within the last n rounds
        new_cell = any(cells[i] > cells[i - 1]
                       for i in range(max(1, end - n + 1), end + 1))
        if abs(improved) > thr or new_cell:
            continue
        return end
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/artifacts/rounds-curve.json")
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    T = PerfTable.from_bundle(A6000[0], env_hash=A6000[1], ok_only=False)
    M = FeatureMatrix(T, REGISTRY)
    hold = list(_splits(T).val.shapes)
    out: dict = {"last3": list(LAST3), "patiences": list(PATIENCES),
                 "groups": {}}

    for tag, prefix, n_seeds in GROUPS:
        print("=" * 96)
        print(f"{tag}  (dir {prefix})  — {n_seeds} seeds")
        print("=" * 96)
        g: dict = {}
        for i in range(n_seeds):
            run = f"{prefix}-s{i}"
            rr = _rows(run)
            vals = [x["best_val_regret"] for x in rr]
            cells = [x["n_cells"] for x in rr]
            thr = _tol(run, T, M, hold)
            approx = thr is None
            if approx:
                # ★ All 6 F3 seeds gave **the same threshold** (0.0047) — the
                #   threshold is set by the holdout shapes, not by the rule.
                #   That value is borrowed but **marked as an
                #   approximation**. The judgement is made on F3 alone
                #   (pre-registration §2).
                thr = out.get("_f3_thr", float("nan"))
            # The improvement per round (positive = it got better)
            d = [vals[k - 1] - vals[k] for k in range(1, len(vals))]
            any_imp = [k for k, x in enumerate(d, start=1) if x > 0]
            sig_imp = [k for k, x in enumerate(d, start=1) if x > thr]
            new_cell = [k for k in range(1, len(cells))
                        if cells[k] > cells[k - 1]]
            stops = {n: _stop_round(vals, n, thr, cells) for n in PATIENCES}
            if tag == "F3rw-p8":
                out["_f3_thr"] = thr
            g[run] = {"vals": vals, "cells": cells, "thr": thr,
                      "thr_is_approx": approx,
                      "last_any": (max(any_imp) if any_imp else None),
                      "last_sig": (max(sig_imp) if sig_imp else None),
                      "last_new_cell": (max(new_cell) if new_cell else None),
                      "sig_in_last3": [k for k in sig_imp if k in LAST3],
                      "stops": stops}
            print(f"  {run:28s} threshold {thr:.4f}"
                  f"{'(approx)' if approx else '        '}  "
                  f"last improvement r{g[run]['last_any']}  "
                  f"★ last **significant** improvement "
                  f"{_r(g[run]['last_sig'])}  "
                  f"last new cell r{g[run]['last_new_cell']}  "
                  f"stop(p3/p4/p10) "
                  + "/".join(str(stops[n]) if stops[n] is not None else "-"
                             for n in PATIENCES))
            print(f"  {'':28s} curve "
                  + " ".join(f"{v:.4f}" for v in vals))
        out["groups"][tag] = g

        sig = [v["last_sig"] for v in g.values()]
        in3 = [r for r, v in g.items() if v["sig_in_last3"]]
        print("\n  the last significant improvement round: "
              + ", ".join("none" if s is None else f"r{s}" for s in sig))
        print(f"  ★ seeds with a significant improvement in the last 3 "
              f"rounds (r9·r10·r11): "
              f"{len(in3)}/{len(g)}  {in3}")
        cells_end = [v["last_new_cell"] for v in g.values()]
        print("  the last new-cell round: "
              + ", ".join("none" if c is None else f"r{c}" for c in cells_end))
        print()

    # ------------------------------------------------------- the verdict
    main_g = out["groups"]["F3rw-p8"]
    in3 = [r for r, v in main_g.items() if v["sig_in_last3"]]
    late_cell = [r for r, v in main_g.items()
                 if v["last_new_cell"] is not None and v["last_new_cell"] >= 9]
    # ------------------------------------------- picking patience (D-129)
    import statistics as _st
    print("=" * 96)
    print("★ per patience — the hypothetical stop round and **the "
          "improvement that would be missed** "
          "(the pre-registration patience-prereg.md)")
    print("=" * 96)
    pat: dict = {}
    for tag, g in out["groups"].items():
        print(f"\n  {tag}")
        for n in PATIENCES:
            miss, stops = [], []
            for v in g.values():
                e = v["stops"][n]
                stops.append(e)
                # ★ How much would have been lost by stopping there. A seed
                #   that ran to the end is 0
                miss.append(0.0 if e is None else v["vals"][e] - v["vals"][-1])
            early = sum(1 for e in stops if e is not None)
            med, mx = _st.median(miss), max(miss)
            pat.setdefault(tag, {})[n] = {
                "stops": stops, "miss": miss, "median": med, "max": mx,
                "n_early": early}
            print(f"    patience {n:2d}  stop "
                  + " ".join("-" if e is None else f"r{e:<2d}" for e in stops)
                  + f"   stopped before 12: {early}/{len(stops)}"
                  + f"   ★ improvement missed, median {med:+.4f}  "
                    f"max {mx:+.4f}")
        cum = [v["vals"][6] - v["vals"][-1] for v in g.values()]
        print(f"    ★ r6 -> r11 cumulative improvement  median "
              f"{_st.median(cum):+.4f}  "
              + " ".join(f"{c:+.4f}" for c in cum))
        pat[tag]["cum_r6_r11"] = cum
    out["patience"] = pat

    SIGMA = 0.0124
    main = pat["F3rw-p8"]
    ok = [n for n in PATIENCES if main[n]["median"] < SIGMA]
    pick = min(ok) if ok else max(PATIENCES)
    print(f"\n  ★ the chosen patience = {pick}  "
          + (f"(the improvement missed, median {main[pick]['median']:+.4f} "
             f"< σ {SIGMA})"
             if ok else
             f"— ⚠️ no value is below σ {SIGMA}. It goes to the larger one"))
    out["patience_pick"] = pick

    print()
    print("=" * 96)
    print("★ the verdict — one of the three in pre-registration §2")
    print("=" * 96)
    if in3:
        verdict = "(나) 12 가 부족하다"
        print(f"  ★ (나) — seeds with a significant improvement in the last "
              f"3 rounds: {in3}")
        print("     -> round 24 has to be measured at n=6 (about 6,000 calls "
              "/ 15 hours)")
    elif late_cell:
        verdict = "(다) 애매하다 — 유의 개선은 끝났는데 새 셀이 늦게까지 생긴다"
        print("  ★ (다) — the significant improvements ended at r8 or "
              f"earlier but new cells keep appearing after r9: {late_cell}")
        print("     -> adjusting patience is **reviewed**. ⚠️ It is not "
              "changed here (that is a condition change, so it needs its own "
              "pre-registration)")
    else:
        verdict = "(가) 12 로 충분하다"
        print("  ★ (가) — for all 6 seeds the last significant improvement "
              "is at r8 or earlier and there is no new cell after r9 either")
        print("     -> rounds are not an axis. It is written in a footnote "
              "together with the curve")
    out["verdict"] = verdict
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
