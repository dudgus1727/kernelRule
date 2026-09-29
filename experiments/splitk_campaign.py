"""★ a6000 를 다시 돌린 캠페인의 집계 (D-187 §5). **LLM 0회 · GPU 0.**

    python3 -m experiments.splitk_campaign                 # 전부
    python3 -m experiments.splitk_campaign --fold 0        # 갈래

⛔ **아무것도 새로 재지 않는다.** 실행이 남긴 `rounds.jsonl`·`archive.jsonl`·
`bests.jsonl`·`hypotheses.jsonl` 과 표를 읽어 센다.

## 무엇을 c2 와 짝지어 보는가

```
★ 홀드아웃      루프가 적은 `best_val_regret` — ⛔ 적합 없음 (D-182 · D-186)
★ split_k 분포   규칙이 65형상에서 고른 top-1 의 split_k
★ sk>=3 형상     최적이 split_k>=3 인 형상에서 몇 개를 sk>=3 로 고르나
★ 경로 수        라운드마다 그 라운드의 최선 규칙이 가진 실행 경로 수
★ 중첩 깊이      `_branch_depth` — ⛔ 거부에 쓰이지 않는 값이다
★ 기존 항 수정   부모와 자식의 항을 ★ 쓰인 축의 집합으로 짝지어 센다
★ split-K 가설   `claim`/`proposed_direction` 에 split?k 가 있는 가설이
                 몇 라운드에 ★ 채택되나 (채택 = 그 id 로 아카이브에 규칙이 있다)
```

⚠️ **split_k 분포는 65형상 전부에서 잰다** — 홀드아웃을 포함한다. 그것은
서술일 뿐이며 ⛔ 어떤 선택에도 쓰지 않는다 (§10.2 는 **선택**을 막는다).

⚠️ **"기존 항 수정" 의 정의는 이 파일의 것이다.** 지시문이 인용한 2.6% /
4.4% 와 수준이 다를 수 있으므로, ⛔ 그 수치와 나란히 놓지 말고 **여기서
두 캠페인을 같은 정의로** 재서 비교한다.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics as st
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401
from experiments.f1_pipeline import _load_stage1, _splits
from experiments.transfer_29_5 import TABLES
from kernelrule.core.matrix import CACHE_DIR, FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY
from kernelrule.features.loader import base_registry, load_generated

GPU = "a6000"
FOLDS = (0, 1, 2, 3)
SEEDS = (0, 1, 2, 3)
DESIGN = "nkband"
ARMS = {"c2": "c2", "splitk": "splitk"}
OUT = Path("docs/artifacts/splitk-campaign.json")
OUT_MD = Path("docs/artifacts/splitk-campaign.md")
#: ★ 가설이 split-K 를 말하는가. ⛔ "reduction" 만으로는 너무 넓다.
_SPLITK = re.compile(r"split[\s_-]?k", re.I)


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# 경로 수 · 중첩 깊이 — ⛔ 루프에 로깅을 넣지 않고 저장된 코드에서 센다
# ---------------------------------------------------------------------------
def _paths_of(code: str) -> tuple[int, int]:
    """(실행 경로 수, `if` 중첩 깊이). 검사기의 것을 그대로 쓴다."""
    from kernelrule.rules.checks import _branch_depth, _numeric_literals, _paths

    tree = ast.parse(code.strip())
    counted, _ = _numeric_literals(tree)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        return (0, 0)
    return (len(_paths(fn.body, {id(n) for n in counted}, cap=4096)),
            _branch_depth(fn))


# ---------------------------------------------------------------------------
# 기존 항을 고쳤나 — 부모와 자식의 항을 ★ 쓰인 축의 집합으로 짝짓는다
# ---------------------------------------------------------------------------
def _terms(code: str) -> dict[frozenset, set[str]]:
    """덧셈 항을 **그 항이 쓴 축의 집합**으로 묶는다.

    ★ 같은 축을 쓰는 항이 부모와 자식에 둘 다 있으면 **같은 자리**로 보고,
    글자가 다르면 ⛔ "고쳤다" 로 센다. 이름이 아니라 축으로 짝짓는 이유는
    항에 이름이 없기 때문이다.
    """
    out: dict[frozenset, set[str]] = {}
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return out

    def axes(node) -> frozenset:
        return frozenset(
            f"{n.value.id}.{n.attr}" for n in ast.walk(node)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id in ("f", "p", "hw"))

    def add(node) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add,
                                                               ast.Sub)):
            add(node.left)
            add(node.right)
            return
        a = axes(node)
        if a:
            out.setdefault(a, set()).add(ast.unparse(node))

    for n in ast.walk(tree):
        if (isinstance(n, (ast.Assign, ast.AugAssign, ast.Return))
                and n.value is not None):
            add(n.value)
    return out


def _edit_kinds(parent: str, child: str) -> dict:
    """부모 -> 자식 한 번의 편집에서 유지 · 수정 · 제거 · 추가."""
    P, C = _terms(parent), _terms(child)
    kept = mod = 0
    for k, pv in P.items():
        if k not in C:
            continue
        (kept, mod) = ((kept + 1, mod) if pv & C[k] else (kept, mod + 1))
    return {"kept": kept, "modified": mod,
            "removed": sum(1 for k in P if k not in C),
            "added": sum(1 for k in C if k not in P),
            "n_parent_terms": len(P)}


def _edit_census(run: Path, by_id: dict) -> tuple[Counter, int]:
    """★ 이 실행에서 **제안된 모든 편집**을 부모와 견준다.

    ⚠️ 아카이브가 아니라 트레이스를 읽는다 — 아카이브는 살아남은 엘리트만
    담고 `parent_ids` 가 비어 있어서, 아카이브로 세면 ⛔ 0건이 나온다.
    (실제로 첫 판이 그렇게 나와 `수정 None` 을 찍었다.)

    ```
    proposal  부모 id · 코드 · code_sha        <- 편집 한 건
    scored    rule_id · code_sha               <- 그 코드에 붙은 이름
    ```

    ⛔ 채택된 것만이 아니라 **제안된 것 전부**를 센다. §1 의 문구가 바꾸려
    한 것은 모델이 무엇을 **쓰는가**이지 무엇이 살아남는가가 아니다.
    """
    tp = run / "trace.jsonl"
    kinds: Counter = Counter()
    if not tp.exists():
        return kinds, 0
    ev = [json.loads(x) for x in tp.read_text().splitlines() if x.strip()]
    sha2code = {d["code_sha"]: d["code"] for d in ev
                if d.get("ev") == "proposal" and d.get("code_sha")}
    id2code = {d["rule"]: sha2code.get(d.get("code_sha"))
               for d in ev if d.get("ev") == "scored"}
    id2code.update({k: v["code"] for k, v in by_id.items()})
    n = 0
    for d in ev:
        if d.get("ev") != "proposal":
            continue
        par = next((id2code.get(x) for x in (d.get("parents") or [])
                    if id2code.get(x)), None)
        if par is None or not d.get("code"):
            continue
        k = _edit_kinds(par, d["code"])
        if not k["n_parent_terms"]:
            continue
        n += 1
        for name in ("kept", "modified", "removed", "added"):
            kinds[name] += k[name]
    return kinds, n


# ---------------------------------------------------------------------------
# 규칙이 고른 config
# ---------------------------------------------------------------------------
def _picks(code, w, table, matrix, shapes) -> dict:
    """★ 65형상에서 규칙의 top-1 이 고른 `split_k` 와, 최적이 sk>=3 인
    형상에서 무엇을 골랐는지."""
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.weights import make_score_of

    so = make_score_of(compile_rule(code), matrix, np.asarray(w, float))
    dist: Counter = Counter()
    hard: list[dict] = []
    for p in shapes:
        cand = table.candidates(p)
        df = table.frame_for(p).reset_index(drop=True)
        t = np.asarray(table.times_of(p))
        i = int(cand.top_k(so(p, cand), 1)[0])
        j = int(np.argmin(t))
        sk_pick = int(df.iloc[i]["split_k"])
        sk_best = int(df.iloc[j]["split_k"])
        dist[sk_pick] += 1
        if sk_best >= 3:
            hard.append({"shape": [p.M, p.N, p.K], "sk_pick": sk_pick,
                         "sk_best": sk_best,
                         "regret": float(t[i] / t[j])})
    return {"split_k_picked": dict(sorted(dist.items())),
            "n_shapes": len(shapes),
            "n_best_sk_ge3": len(hard),
            "n_picked_sk_ge3_there": sum(1 for h in hard
                                         if h["sk_pick"] >= 3),
            "regret_on_sk_ge3": (round(float(np.exp(np.mean(np.log(
                [h["regret"] for h in hard])))), 6) if hard else None),
            "hard": hard}


# ---------------------------------------------------------------------------
def _run(arm: str, fold: int, seed: int, table, shapes) -> dict:
    d = Path(f"runs/{ARMS[arm]}-{GPU}-f{fold}")
    run = Path(f"{d}-s{seed}")
    if not (run / "rounds.jsonl").exists():
        return {"arm": arm, "fold": fold, "seed": seed, "missing": True}
    rr = _rows(run / "rounds.jsonl")
    arc = _rows(run / "archive.jsonl")
    best = _rows(run / "bests.jsonl")[-1]
    by_id = {e["rule_id"]: e for e in arc}

    # ★ 라운드별 경로 수 — 그 라운드까지의 최선 규칙(학습 점수)으로 잰다
    curve = []
    for r in range(len(rr)):
        c = [e for e in arc if (e.get("round") is None or e["round"] <= r)]
        if not c:
            curve.append(None)
            continue
        e = min(c, key=lambda x: x["regret"])
        n_paths, depth = _paths_of(e["code"])
        curve.append({"round": r, "n_paths": n_paths, "depth": depth,
                      "train": round(e["regret"], 6),
                      "holdout": round(float(rr[r]["best_val_regret"]), 6)})

    # ★ 편집의 종류 — ⛔ `archive.jsonl` 로는 못 센다. 그것은 살아남은
    #   엘리트만 담고 `parent_ids` 도 비어 있다. 부모->자식 한 쌍은
    #   `trace.jsonl` 에만 있다 (`_edit_census`).
    kinds, n_edits = _edit_census(run, by_id)
    tot_terms = kinds["kept"] + kinds["modified"] + kinds["removed"]

    # ★ split-K 가설 — 언제 나오고 언제 채택되나
    hyp = _rows(run / "hypotheses.jsonl") if (
        run / "hypotheses.jsonl").exists() else []
    used = {e.get("hypothesis_id") for e in arc}
    sk_h = [h for h in hyp
            if _SPLITK.search(str(h.get("claim", ""))
                              + str(h.get("proposed_direction", "")))]
    sk_rounds = sorted({int(h["round"]) for h in sk_h})
    sk_taken = sorted({int(h["round"]) for h in sk_h if h["id"] in used})

    fin = _paths_of(best["code"])
    return {
        "arm": arm, "fold": fold, "seed": seed, "run": run.name,
        "holdout": round(float(rr[-1]["best_val_regret"]), 6),
        "train": round(float(best["regret"]), 6),
        "n_rounds": len(rr),
        "llm_calls": sum(sum((x.get("llm_calls") or {}).values())
                         for x in rr),
        "minutes": round(sum(x.get("seconds") or 0 for x in rr) / 60, 1),
        "n_accepted": sum(x.get("n_accepted") or 0 for x in rr),
        # ⚠️ `rejections` 는 `[[사유, 메시지], ...]` 꼴이다 — dict 가 아니다.
        #   사유만 센다 (메시지는 규칙마다 달라 집계가 안 된다).
        "rejections": dict(Counter(
            x[0] for r in rr for x in (r.get("rejections") or []))),
        "final_n_paths": fin[0], "final_depth": fin[1],
        "max_n_paths": max((c["n_paths"] for c in curve if c), default=0),
        "max_depth": max((c["depth"] for c in curve if c), default=0),
        "path_curve": curve,
        "edits": {**dict(kinds), "n_edits": n_edits,
                  "modified_frac": (round(kinds["modified"] / tot_terms, 4)
                                    if tot_terms else None),
                  "removed_frac": (round(kinds["removed"] / tot_terms, 4)
                                   if tot_terms else None)},
        "splitk_hyp_rounds": sk_rounds,
        "splitk_hyp_accepted_rounds": sk_taken,
        "picks": _picks(best["code"], best["w"], table, _MATRIX[(fold, seed,
                                                                arm)],
                        shapes)}


_MATRIX: dict = {}


def _matrix_for(arm: str, fold: int, seed: int, table):
    d = Path(f"runs/{ARMS[arm]}-{GPU}-f{fold}")
    reg = _load_stage1(d, base_registry("F2", human=REGISTRY), "F2", table)
    fp = Path(f"{d}-s{seed}") / "features.jsonl"
    if fp.exists():
        for f in load_generated(fp, table=table):
            if f.name not in reg._items:
                reg.add(f)
    return FeatureMatrix(table, reg, cache_dir=CACHE_DIR)


def main() -> None:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.merge:
        rows: list[dict] = []
        for f in a.merge:
            rows += json.loads(Path(f).read_text())["rows"]
        rows.sort(key=lambda r: (r["arm"], r["fold"], r["seed"]))
        _write(rows, Path(a.out))
        return

    T = TABLES[GPU]
    table = PerfTable.from_bundle(T["bundle"], env_hash=T["env_hash"],
                                  ok_only=False)
    folds = FOLDS if a.fold is None else (a.fold,)
    rows = []
    print("=" * 104)
    print("★ a6000 재실행 집계 (D-187 §5) — c2 와 (fold, seed) 로 짝지어. "
          "LLM 0회")
    print("=" * 104)
    for fold in folds:
        sp = _splits(table, fold=fold, k=4, design=DESIGN)
        shapes = list(sp.train.shapes) + list(sp.val.shapes)
        for arm in ARMS:
            for seed in SEEDS:
                _MATRIX[(fold, seed, arm)] = _matrix_for(arm, fold, seed,
                                                         table)
                r = _run(arm, fold, seed, table, shapes)
                rows.append(r)
                if r.get("missing"):
                    print(f"  {arm:7s} f{fold} s{seed}  ★ 아직 없음")
                    continue
                pk = r["picks"]
                print(f"  {arm:7s} f{fold} s{seed}  홀드아웃 {r['holdout']:.4f}"
                      f"  경로 {r['final_n_paths']:>3d} (최대 "
                      f"{r['max_n_paths']:>3d}) 깊이 {r['max_depth']}"
                      f"  sk분포 {pk['split_k_picked']}"
                      f"  sk>=3 {pk['n_picked_sk_ge3_there']}/"
                      f"{pk['n_best_sk_ge3']}"
                      f"  수정 {r['edits']['modified_frac']}"
                      f"  splitK채택 {r['splitk_hyp_accepted_rounds']}",
                      flush=True)
    _write(rows, Path(a.out))


def _write(rows: list[dict], out: Path) -> None:
    out.write_text(json.dumps(
        {"gpu": GPU, "folds": list(FOLDS), "seeds": list(SEEDS),
         "design": DESIGN,
         "note": ("★ D-187 §5. 홀드아웃은 루프가 적은 best_val_regret — ⛔ "
                  "적합 없음 (D-182 · D-186). split_k 분포는 65형상 전부에서 "
                  "잰 서술이며 어떤 선택에도 쓰지 않는다. '기존 항 수정' 의 "
                  "정의는 이 파일의 것이다 (축 집합으로 항을 짝짓는다)."),
         "rows": rows, "summary": _summary(rows)},
        ensure_ascii=False, indent=1))
    out.with_suffix(".md").write_text(_md(rows))
    print(f"\n  -> {out}\n  -> {out.with_suffix('.md')}")
    _print(_summary(rows))


def _agg(rows: list[dict], arm: str) -> dict:
    rs = [r for r in rows if r["arm"] == arm and not r.get("missing")]
    if not rs:
        return {}
    ho = [r["holdout"] for r in rs]
    sk3 = [(r["picks"]["n_picked_sk_ge3_there"], r["picks"]["n_best_sk_ge3"])
           for r in rs]
    dist: Counter = Counter()
    for r in rs:
        for k, v in r["picks"]["split_k_picked"].items():
            dist[int(k)] += v
    mods = [r["edits"]["modified_frac"] for r in rs
            if r["edits"]["modified_frac"] is not None]
    acc = [x for r in rs for x in r["splitk_hyp_accepted_rounds"]]
    return {
        "n_runs": len(rs),
        "holdout_median": round(st.median(ho), 6),
        "holdout_geomean": round(float(np.exp(np.mean(np.log(ho)))), 6),
        "holdout_by_fold": {f: round(st.median(
            [r["holdout"] for r in rs if r["fold"] == f]), 6)
            for f in sorted({r["fold"] for r in rs})},
        "split_k_picked_total": dict(sorted(dist.items())),
        "n_runs_with_sk_ge3": sum(1 for r in rs
                                  if any(int(k) >= 3 for k in
                                         r["picks"]["split_k_picked"])),
        "sk_ge3_hit": f"{sum(a for a, _ in sk3)}/{sum(b for _, b in sk3)}",
        "regret_on_sk_ge3_median": round(st.median(
            [r["picks"]["regret_on_sk_ge3"] for r in rs
             if r["picks"]["regret_on_sk_ge3"]]), 6),
        "modified_frac_median": (round(st.median(mods), 4) if mods else None),
        "removed_frac_median": round(st.median(
            [r["edits"]["removed_frac"] for r in rs
             if r["edits"]["removed_frac"] is not None]), 4),
        "final_n_paths": sorted(r["final_n_paths"] for r in rs),
        "max_n_paths": sorted(r["max_n_paths"] for r in rs),
        "max_depth": sorted(r["max_depth"] for r in rs),
        "splitk_accepted_rounds": sorted(acc),
        "n_runs_splitk_accepted": sum(
            1 for r in rs if r["splitk_hyp_accepted_rounds"]),
        "llm_calls_median": st.median([r["llm_calls"] for r in rs]),
    }


def _summary(rows: list[dict]) -> dict:
    out = {arm: _agg(rows, arm) for arm in ARMS}
    pair = []
    idx = {(r["arm"], r["fold"], r["seed"]): r for r in rows
           if not r.get("missing")}
    for f in FOLDS:
        for s in SEEDS:
            a, b = idx.get(("c2", f, s)), idx.get(("splitk", f, s))
            if a and b:
                pair.append({"fold": f, "seed": s, "c2": a["holdout"],
                             "splitk": b["holdout"],
                             "diff": round(b["holdout"] - a["holdout"], 6)})
    if pair:
        d = [p["diff"] for p in pair]
        out["paired"] = {
            "n": len(pair), "diff_median": round(st.median(d), 6),
            "splitk_better": sum(1 for x in d if x < 0),
            "c2_better": sum(1 for x in d if x > 0),
            "rows": pair}
    return out


def _print(s: dict) -> None:
    for arm in ARMS:
        a = s.get(arm) or {}
        if not a:
            continue
        print(f"\n  ★ {arm}  실행 {a['n_runs']}  홀드아웃 중앙 "
              f"{a['holdout_median']:.4f} gm {a['holdout_geomean']:.4f}")
        print(f"     fold 별 {a['holdout_by_fold']}")
        print(f"     split_k 분포 {a['split_k_picked_total']}"
              f"  · sk>=3 고른 실행 {a['n_runs_with_sk_ge3']}/{a['n_runs']}")
        print(f"     최적 sk>=3 형상에서 sk>=3 로 고름 {a['sk_ge3_hit']}"
              f"  · 그 형상들의 regret 중앙 "
              f"{a['regret_on_sk_ge3_median']:.4f}")
        print(f"     기존 항 수정 중앙 {a['modified_frac_median']}"
              f"  제거 중앙 {a['removed_frac_median']}")
        print(f"     최종 경로 {a['final_n_paths']}  최대 경로 "
              f"{a['max_n_paths']}  최대 깊이 {a['max_depth']}")
        print(f"     split-K 가설 채택 실행 {a['n_runs_splitk_accepted']}/"
              f"{a['n_runs']}  라운드 {a['splitk_accepted_rounds']}")
    p = s.get("paired")
    if p:
        print(f"\n  ★ 짝비교 {p['n']}쌍 — 차 중앙 {p['diff_median']:+.4f}"
              f"  · splitk 가 나은 쌍 {p['splitk_better']}/{p['n']}")


def _md(rows: list[dict]) -> str:
    s = _summary(rows)
    L = ["# a6000 를 다시 돌린다 — split-K (D-187 §5)", "",
         ("> **재현** `python3 -m experiments.splitk_campaign` · LLM 0회 "
          "· GPU 0"),
         ("> ★ 홀드아웃은 루프가 적은 `best_val_regret` — ⛔ 적합 없음 "
          "(D-182 · D-186)"), "",
         "| | c2 | ★ splitk |", "|---|--:|--:|"]
    c, k = s.get("c2") or {}, s.get("splitk") or {}
    if c and k:
        def row(name, key, fmt="{}"):
            L.append(f"| {name} | {fmt.format(c.get(key))} | "
                     f"**{fmt.format(k.get(key))}** |")
        row("실행", "n_runs")
        row("홀드아웃 중앙", "holdout_median", "{:.4f}")
        row("홀드아웃 gm", "holdout_geomean", "{:.4f}")
        row("split_k 분포", "split_k_picked_total")
        row("sk>=3 고른 실행", "n_runs_with_sk_ge3")
        row("최적 sk>=3 에서 sk>=3", "sk_ge3_hit")
        row("그 형상들의 regret 중앙", "regret_on_sk_ge3_median", "{:.4f}")
        row("기존 항 수정 중앙", "modified_frac_median")
        row("기존 항 제거 중앙", "removed_frac_median")
        row("최종 경로 수", "final_n_paths")
        row("최대 경로 수", "max_n_paths")
        row("최대 중첩 깊이", "max_depth")
        row("split-K 가설 채택 실행", "n_runs_splitk_accepted")
        row("LLM 호출 중앙", "llm_calls_median")
    p = s.get("paired")
    if p:
        L += ["", "## ★ (fold, seed) 짝비교", "",
              (f"차 중앙 **{p['diff_median']:+.4f}** · splitk 가 나은 쌍 "
               f"**{p['splitk_better']}/{p['n']}**"), "",
              "| fold | seed | c2 | ★ splitk | 차 |", "|--:|--:|--:|--:|--:|"]
        for x in p["rows"]:
            L.append(f"| {x['fold']} | {x['seed']} | {x['c2']:.4f} | "
                     f"{x['splitk']:.4f} | {x['diff']:+.4f} |")
    L += ["", "## ★ 경로 수와 홀드아웃 (§3-4)", "",
          "| arm | fold | seed | 최종경로 | 최대경로 | 깊이 | 홀드아웃 |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        if r.get("missing"):
            continue
        L.append(f"| {r['arm']} | {r['fold']} | {r['seed']} | "
                 f"{r['final_n_paths']} | {r['max_n_paths']} | "
                 f"{r['max_depth']} | {r['holdout']:.4f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
