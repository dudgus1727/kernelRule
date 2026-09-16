"""★ Three wordings, side by side — how big a rule each one writes
(D-176 §2-4). **0 LLM calls** (the rules already exist).

    python3 -m experiments.prompt_terms

```
★ 옛 문구     c5c196a  "Add a term whenever you can say which physics it is"
★ d08a48d     "the bar is: leaving it out makes the ranking obviously worse"
★ 단일        D-176 §2-2  "Write the best rule you can" — ★ 크기 지시 없음
```

⚠️ **The label is read off the stored prompt, not assumed from the run
name.** Each stage-2 directory keeps the prompt it actually sent; this
script greps that text for the three distinguishing sentences and refuses a
run where zero or more than one match.

⛔ **This is not a controlled A/B.** The three wordings ran in three
campaigns, on different fold designs (`nkgroup` for the old one, `nkband`
for the other two), and the single-agent arm is fold 0 only. What it
supports is "how many terms does each wording produce", not a causal claim
about holdout.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import warnings
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from experiments.f1_pipeline import _load_stage1
from experiments.single_agent import _shape
from experiments.transfer_29_5 import TABLES
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry

#: ★ The sentence that identifies each wording in the stored prompt.
MARK = {
    "old_c5c196a": "Add a term whenever you can say which",
    "d08a48d": "The bar for adding a term",
    "single_d176": "Write the best rule you can",
}
PREFIX = {"nk4-": "old_c5c196a", "c2-": "d08a48d", "sa-": "single_d176"}
OUT = Path("docs/artifacts/prompt-terms.json")


def _wording(d: Path) -> str | None:
    calls = sorted((d / "stage2-rule-writer" / "llm_calls").glob("*.json"))
    if not calls:
        return None
    s = calls[0].read_text()
    hit = [k for k, m in MARK.items() if m in s]
    return hit[0] if len(hit) == 1 else None


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    tables: dict = {}
    names: dict = {}
    rows: list[dict] = []
    bad: list[str] = []
    for d in sorted(Path("runs").glob("*")):
        if not (d / "stage2-rule-writer" / "summary.json").exists():
            continue
        pre = next((p for p in PREFIX if d.name.startswith(p)), None)
        if pre is None:
            continue
        gpu = d.name.split("-")[1]
        if gpu not in TABLES:
            continue
        w = _wording(d)
        if w is None:
            bad.append(d.name)
            continue
        if w != PREFIX[pre]:
            # ★ The prompt on disk wins over the name. Recorded either way.
            bad.append(f"{d.name}: name says {PREFIX[pre]}, prompt says {w}")
        if gpu not in tables:
            T = TABLES[gpu]
            tables[gpu] = PerfTable.from_bundle(T["bundle"],
                                                env_hash=T["env_hash"],
                                                ok_only=False)
        if d.name not in names:
            reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2",
                               tables[gpu])
            names[d.name] = {n for n in reg._items if reg[n].shape_level}
        s = json.loads((d / "stage2-rule-writer" / "summary.json").read_text())
        for t in s["tries"]:
            if not t.get("ok"):
                rows.append({"run": d.name, "wording": w, "gpu": gpu,
                             "ok": False})
                continue
            rows.append({"run": d.name, "wording": w, "gpu": gpu, "ok": True,
                         "fit_regret": t["fit_regret"],
                         **_shape(t["code"], names[d.name])})

    res: dict = {"note": ("⛔ not a controlled A/B — three campaigns, two "
                          "fold designs; the single arm is fold0 only"),
                 "label_source": "the prompt stored in stage2-rule-writer/",
                 "unlabelled": bad, "wordings": {}, "rows": rows}
    print("=" * 98)
    print("★ 세 문구의 항 수 (D-176 §2-4). 0 LLM 호출 — 이미 있는 규칙을 센다")
    print("=" * 98)
    print(f"  {'문구':<14} {'규칙':>5} {'통과':>7} {'항수 중앙':>9} "
          f"{'항수 범위':>11} {'경로 중앙':>9} {'분기축 중앙':>11}")
    for w in MARK:
        rs = [r for r in rows if r["wording"] == w]
        ok = [r for r in rs if r["ok"]]
        if not ok:
            continue
        nt = sorted(r["n_terms"] for r in ok)
        npth = [r["n_paths"] for r in ok]
        nb = [len(r["branch_axes"]) for r in ok]
        e = {"n_rules": len(rs), "n_ok": len(ok),
             "pass_rate": round(len(ok) / len(rs), 4),
             "n_runs": len({r["run"] for r in rs}),
             "gpus": sorted({r["gpu"] for r in rs}),
             "n_terms_median": st.median(nt),
             "n_terms_range": [nt[0], nt[-1]],
             "n_terms_mean": round(st.mean(nt), 2),
             "n_paths_median": st.median(npth),
             "n_branch_axes_median": st.median(nb),
             "fit_regret_median": round(st.median(
                 [r["fit_regret"] for r in ok]), 6)}
        res["wordings"][w] = e
        print(f"  {w:<14} {e['n_rules']:>5} {e['pass_rate']:>7.0%} "
              f"{e['n_terms_median']:>9.1f} "
              f"{str(e['n_terms_range']):>11} {e['n_paths_median']:>9.1f} "
              f"{e['n_branch_axes_median']:>11.1f}")
    if bad:
        print(f"  ⚠️ 문구를 못 읽거나 이름과 어긋난 실행 {len(bad)}: "
              f"{bad[:3]}")
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
