"""★ The (b) transfer result, written as a stage-2 seed (D-175 §1-1).
**0 LLM calls.**

    python3 -m experiments.porting_seed

`--seed-from` reads `runs/<tag>/stage2-rule-writer/chosen.json`, so writing
the ported rule into that file lets `--stage 3` run unchanged. ⛔ No new
machinery.

```
seed      the source (table, fold)'s best-by-training rule, and the axes it
          uses — re-derived on the **target** table (D-165)
weights   ⛔ the source's `w` as they are. The loop refits on the target's
          training split every round, and `canonical_score` **then** refitted
          too — which is exactly what the transfer table calls (b).
          Verified at the time: canonical holdout == `b_refit` to 2e-7.
          ⛔ 2026-09-18 (D-182): the scorer no longer refits, so that
          identity no longer holds and (b) is defined by the loop alone.
⛔ stage 2 is not run. The seed **is** the ported rule (§1-3)
```
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

GPUS = ("a6000", "5090", "4090", "h100")
FOLD = 0
OUT_TAG = "pc"          # porting cost


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=FOLD)
    a = ap.parse_args()

    import experiments.transfer_nk4 as T
    T.PREFIX, T.SEEDS, T.DESIGN, T.LEAKED = "c2", (0, 1, 2, 3), "nkband", False
    from experiments.transfer_nk4 import _best_of_fold

    tr = json.loads(
        Path("docs/artifacts/campaign2-transfer.json").read_text())
    cells = {(c["src"], c["dst"], c["fold"]): c for c in tr["cells"]}

    made = []
    print("=" * 86)
    print(f"★ the ported rule as a seed — fold{a.fold} (D-175 §1). 0 LLM calls")
    print("=" * 86)
    for src in GPUS:
        e, seed = _best_of_fold(src, a.fold)
        if e is None:
            continue
        for dst in GPUS:
            if dst == src:
                continue
            cell = cells.get((src, dst, a.fold))
            if cell is None:
                continue
            tag = f"{OUT_TAG}-{src}2{dst}-f{a.fold}"
            d = Path("runs") / tag
            (d / "stage2-rule-writer").mkdir(parents=True, exist_ok=True)
            # ★ The stage-1 artefact the run reads: the source run's axis
            #   list, so `_load_stage1` rebuilds exactly what the rule uses.
            src_props = Path(f"runs/c2-{src}-f{a.fold}/stage1-features/"
                             f"proposals.jsonl")
            dst1 = d / "stage1-features"
            dst1.mkdir(parents=True, exist_ok=True)
            (dst1 / "proposals.jsonl").write_text(src_props.read_text())
            loop_feats = Path(f"runs/c2-{src}-f{a.fold}-s{seed}/"
                              f"features.jsonl")
            if loop_feats.exists():
                # ★ The axes the source run built in its loop go in too —
                #   the rule references them (D-165 §1).
                with (dst1 / "proposals.jsonl").open("a") as fh:
                    for ln in loop_feats.read_text().splitlines():
                        if ln.strip():
                            j = json.loads(ln)
                            # ⛔ `features.jsonl` records **refused attempts
                            #   too** (`accepted: false`). Copying them in as
                            #   accepted puts code the checker already threw
                            #   out back into the library — and it fails on
                            #   reload, which is how this was caught.
                            if not j.get("accepted"):
                                continue
                            fh.write(json.dumps(
                                {"name": j["name"], "code": j["code"],
                                 "accepted": True,
                                 "unit": j.get("unit", "dimensionless"),
                                 "direction": j.get("direction", "neutral"),
                                 "expected_range": j.get("expected_range",
                                                         [0.0, 1.0]),
                                 "rationale": j.get("rationale", "")},
                                ensure_ascii=False) + "\n")
            (dst1 / "summary.json").write_text(json.dumps({
                "condition": "F2",
                "imported_from": str(src_props),
                "note": ("★ D-175 — the SOURCE run's axis list (k7-1 + what "
                         "that run's loop built). `shape_level` is re-judged "
                         "on the target table by `_load_stage1`.")},
                ensure_ascii=False, indent=1))
            (d / "stage2-rule-writer" / "chosen.json").write_text(json.dumps({
                "source": f"ported:{src}->{dst}",
                "code": e["code"], "w0": [float(x) for x in e["w"]],
                "fit_regret": cell["b_refit"],
                "why": ("★ D-175 — the (b) transfer arm as a seed: the "
                        "source rule's structure, and the target's own "
                        "training split decides the weights. The recorded "
                        "`fit_regret` is the transfer table's `b_refit`, "
                        "which is the **target holdout** — kept here so the "
                        "seed's identity is checkable (§3 4th)."),
                "src_run": f"c2-{src}-f{a.fold}-s{seed}",
                "b_refit": cell["b_refit"], "a_as_is": cell["a_as_is"],
                "native": cell["native"]}, ensure_ascii=False, indent=1))
            made.append(tag)
            print(f"  {tag:26s} (a) {cell['a_as_is']:.4f}  "
                  f"(b) {cell['b_refit']:.4f}  원주민 {cell['native']:.4f}  "
                  f"|w0| {len(e['w'])}")
    print(f"\n  ★ {len(made)} directions")


if __name__ == "__main__":
    main()
