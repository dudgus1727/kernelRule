"""★ 옛↔새 대조 — 네 릴리즈를 재집계하면 무엇이 바뀌는가 (D-186 §4).

    python3 -m experiments.rescore_diff [--ref 266630c]

**LLM 0회 · GPU 0 · 계산 0회.** 이 파일은 ⛔ 아무것도 다시 재지 않는다.
이미 있는 산출물 두 벌(옛 커밋 · 지금 작업본)을 같은 자리에서 읽어 나란히
놓을 뿐이다.

```
★ 옛   `--ref` 커밋의 docs/artifacts/*.json (기본 266630c = D-185 직후)
★ 새   작업본의 같은 파일 — D-186 재집계본
```

⚠️ **무엇이 바뀌었길래 수치가 바뀌는가** (셋뿐이다)

```
1  채점기가 적합을 버렸다                     D-182
2  원주민이 캠페인 2 의 ★ 재집계 홀드아웃이다   D-183 (옛 것은 nkgroup 으로
                                              갈라져 홀드아웃이 오염됐다)
3  전이 (a)/(b) 가 재집계 전이표에서 온다      D-183 §3
```

⛔ 규칙을 다시 진화시키지 않았고, 오토튜닝 곡선도 다시 돌리지 않았다.
⛔ 옛 수치를 지우지 않는다 — 두 벌을 같이 싣는다.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
from pathlib import Path

REF = "266630c"
OUT = Path("docs/artifacts/rescore-all.json")
OUT_MD = Path("docs/artifacts/rescore-all.md")
NS = (4, 8, 16)


def _old(ref: str, path: str) -> dict | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"],
                       capture_output=True, text=True, check=False)
    return json.loads(r.stdout) if r.returncode == 0 else None


def _new(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _cost(d: dict) -> dict:
    s = d["summary"]
    return {"(b)−원주민 중앙": s["gap_b_minus_native_median"],
            "r5 까지 원주민 넘음": f"{s['beats_native_by_round']['5']}/12",
            "r2 까지 회복": s["gap_recovered_by_r2_median"],
            "끝까지 못 넘음": f"{s['never_beats']}/12",
            "LLM 호출 중앙": s["llm_calls_median"]}


def _acurve(d: dict) -> dict:
    out = {}
    for n in NS:
        rs = [r for r in d["rows"] if r["N"] == n]
        if not rs:
            continue
        g = [r["holdout_median"] - r["native"] for r in rs]
        out[f"N={n} 격차중앙"] = round(st.median(g), 6)
        out[f"N={n} 원주민넘음"] = f"{sum(1 for x in g if x < 0)}/{len(rs)}"
    return out


def _bcurve(d: dict) -> dict:
    out = {}
    for n in NS:
        rs = [r for r in d["rows"] if r["N"] == n]
        if not rs:
            continue
        g2 = [r["curve"][-1] - r["native"] for r in rs if r["curve"][-1]]
        gb = [r["best_round"] - r["native"] for r in rs if r["best_round"]]
        out[f"N={n} r2 격차중앙"] = round(st.median(g2), 6)
        out[f"N={n} 최선 격차중앙"] = round(st.median(gb), 6)
        out[f"N={n} 원주민넘음"] = (
            f"{sum(1 for x in gb if x < 0)}/{len(rs)}")
    return out


def _single(d: dict) -> dict:
    a = d["aggregate"]
    return {"단일이 루프보다 나은 표": f"{a['single_beats_loop']}/{a['n']}",
            "학습 1위의 홀드아웃 순위 중앙": a["median_rank_of_train_winner"],
            "선택 과적합 중앙": a["median_selection_overfit"]}


def _auto(d: dict) -> dict:
    out = {}
    for g, gd in d["gpus"].items():
        f = gd["folds"].get("0") or next(iter(gd["folds"].values()))
        out[f"{g} k=0 우리규칙"] = f["k0_our_rule"]
        for arm in ("random", "tpe"):
            c = (f.get("crossing") or {}).get(arm, {}).get("vs_our_rule")
            out[f"{g} {arm} 교차 k"] = (c or {}).get("k")
            ki = (c or {}).get("k_interp")
            out[f"{g} {arm} 교차 interp"] = (round(ki, 1) if ki else None)
    return out


SPECS = (
    ("전이 비용 곡선 (porting-cost)", "docs/artifacts/porting-cost.json",
     _cost),
    ("(a) 재적합 곡선 · 무작위 (porting-shapes)",
     "docs/artifacts/porting-shapes.json", _acurve),
    ("(a) 재적합 곡선 · 층화 (porting-strat)",
     "docs/artifacts/porting-strat.json", _acurve),
    ("(b) 루프 곡선 · 무작위 (porting-shapes-curve)",
     "docs/artifacts/porting-shapes-curve.json", _bcurve),
    ("(b) 루프 곡선 · 층화 (porting-strat-curve)",
     "docs/artifacts/porting-strat-curve.json", _bcurve),
    ("단일 에이전트 (single-agent)", "docs/artifacts/single-agent.json",
     _single),
    ("오토튜닝 곡선 (autotune-curve)", "docs/artifacts/autotune-curve.json",
     _auto),
)


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:+.4f}" if abs(v) < 1 else f"{v:.4f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF)
    a = ap.parse_args()
    blocks = []
    print("=" * 92)
    print(f"★ 옛↔새 대조 (D-186 §4) — 옛 = {a.ref} · 새 = 작업본. 계산 0회")
    print("=" * 92)
    for title, path, fn in SPECS:
        o, n = _old(a.ref, path), _new(path)
        if o is None or n is None:
            print(f"  ⚠️ {title} — 한쪽이 없다 (옛 {o is not None} / "
                  f"새 {n is not None})")
            continue
        try:
            ov, nv = fn(o), fn(n)
        except (KeyError, TypeError) as exc:      # 옛 산출물의 칸이 다를 때
            print(f"  ⚠️ {title} — 옛 산출물의 모양이 다르다: {exc}")
            continue
        keys = [k for k in nv if k in ov]
        blocks.append({"title": title, "path": path,
                       "old": {k: ov[k] for k in keys},
                       "new": {k: nv[k] for k in keys},
                       "changed": [k for k in keys if ov[k] != nv[k]]})
        print(f"\n  {title}")
        for k in keys:
            mark = "  " if ov[k] == nv[k] else "★ "
            print(f"    {mark}{k:28s} {_fmt(ov[k]):>10s} -> "
                  f"{_fmt(nv[k]):>10s}")
    OUT.write_text(json.dumps(
        {"ref_old": a.ref, "note": (
            "★ D-186 §4. 계산 0회 — 이미 있는 두 벌을 읽어 나란히 놓는다. "
            "바뀐 것은 셋뿐이다: 채점기의 적합이 사라졌고(D-182), 원주민이 "
            "재집계된 c2 홀드아웃이 됐고(D-183), 전이 (a)/(b) 가 재집계 "
            "전이표에서 온다. ⛔ 규칙을 다시 진화시키지 않았다."),
         "blocks": blocks}, ensure_ascii=False, indent=1))
    OUT_MD.write_text(_md(blocks, a.ref))
    print(f"\n  -> {OUT}\n  -> {OUT_MD}")


def _md(blocks: list[dict], ref: str) -> str:
    L = ["# 옛↔새 대조 — 네 릴리즈 재집계 (D-186 §4)", "",
         (f"> **재현** `python3 -m experiments.rescore_diff --ref {ref}` · "
          "LLM 0회 · GPU 0 · ★ 계산 0회"),
         (f"> 옛 = `{ref}` 의 산출물 · 새 = 재집계본. ⛔ 옛 수치를 지우지 "
          "않는다 — 두 벌을 같이 싣는다."), "",
         "바뀐 것은 셋뿐이다:", "",
         "| # | 무엇 | 어디 |", "|---|---|---|",
         "| 1 | 채점기가 적합을 버렸다 | D-182 |",
         ("| 2 | 원주민이 ★ 재집계된 c2 홀드아웃이다 | D-183 "
          "(옛 것은 `nkgroup` 으로 갈라져 오염됐다) |"),
         "| 3 | 전이 (a)/(b) 가 재집계 전이표에서 온다 | D-183 §3 |", ""]
    for b in blocks:
        L += [f"## {b['title']}", "",
              f"`{b['path']}`", "",
              "| 무엇 | 옛 | ★ 새 |", "|---|--:|--:|"]
        for k in b["old"]:
            mark = "**" if k in b["changed"] else ""
            L.append(f"| {k} | {_fmt(b['old'][k])} | "
                     f"{mark}{_fmt(b['new'][k])}{mark} |")
        if not b["changed"]:
            L.append("")
            L.append("★ 한 칸도 바뀌지 않았다.")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
