"""★ a6000 손편집 16회의 장부 (D-188). **계산 0회** — 이미 있는 결과를 읽어 표로 옮긴다.

    python3 -m experiments.a6000_ledger

읽는 것   docs/artifacts/a6000-probe/{iterations,index}.json · results/*.json
쓰는 것   docs/artifacts/a6000-probe/ledger.md

⛔ 숫자를 손으로 적지 않는다. 표의 모든 값은 결과 파일에서 읽는다.
"""

from __future__ import annotations

import json
from pathlib import Path

D = Path("docs/artifacts/a6000-probe")
VENDOR = 1.081488


def _sel(it: dict) -> dict:
    return next((r for r in it["runs"] if r["label"] == it["selected_run"]),
                it["runs"][0] if it["runs"] else {})


def main() -> None:
    it = json.loads((D / "iterations.json").read_text())
    idx = json.loads((D / "index.json").read_text())
    L = ["# a6000 규칙 손편집 — 16회 장부 (D-188)", "",
         ("> **재현** 규칙 하나: `python3 -m experiments.a6000_probe --rule "
          "docs/artifacts/a6000-probe/rules/<tag>.json --out OUT.json`"),
         "> 이 표: `python3 -m experiments.a6000_ledger` (계산 0회)", "",
         ("숫자 읽는 법: **1.00 = 가장 빠른 설정을 정확히 고름, 1.08 = 8% "
          "느림.** 작을수록 좋다."), "",
         ("- **고르는 값(inner-CV)** — 학습 형상 안에서만 잰 값. 반복 "
          "사이에서 무엇을 남길지는 이것으로 정했다."),
         ("- **보고 값(홀드아웃)** — 네 fold 의 시험 형상 65개. 벤더는 "
          f"**{VENDOR:.4f}**."), "",
         "## 기준선 — 원래 규칙(c2 루프가 만든 것)을 같은 절차로", "",
         "| 시드 | inner-CV | 홀드아웃 | 벤더 넘은 fold | 쪼개기 문제 적중 |",
         "|---|--:|--:|--:|--:|"]
    for e in idx:
        if not e["tag"].startswith("baseline"):
            continue
        g = json.loads(Path(e["result"]).read_text())["aggregate"]
        L.append(f"| {e['tag'][-2:]} | {g['inner_cv_gm']:.4f} | "
                 f"{g['holdout_gm_65']:.4f} | {g['folds_beating_vendor']}/4 | "
                 f"{g['sk_ge3_picked_on_hard']} |")
    L += ["", "## 16회", "",
          ("| # | 라운드 | 무엇을 | 대조군 | 판정 | inner-CV | 홀드아웃 | "
           "벤더 넘은 fold | 쪼개기 적중 |"),
          "|--:|--:|---|:-:|---|--:|--:|--:|--:|"]
    rnd = {}
    for n, r in enumerate(it["iterations"], 1):
        rnd[r["id"]] = (n - 1) // 4 + 1
        s = _sel(r)
        mark = "**" if s.get("holdout_gm_65", 9) < VENDOR else ""
        L.append(f"| {n} | {rnd[r['id']]} | {r['id']} {r['title'][:60]} | "
                 f"{'○' if r['is_control'] else ''} | {r['verdict']} | "
                 f"{s.get('inner_cv_gm', float('nan')):.4f} | "
                 f"{mark}{s.get('holdout_gm_65', float('nan')):.4f}{mark} | "
                 f"{s.get('folds_beating_vendor', '?')}/4 | "
                 f"{s.get('sk_ge3_picked_on_hard', '?')} |")
    L += ["", ("굵은 홀드아웃 = 벤더보다 낮음. ⚠️ 결과끼리 **0.01 안쪽은 "
               "비김**이다 (13번 대조군: 가중치 이름만 바꿔도 0.01 흔들린다)."),
          "",
          "## 한 회씩 — 무엇을 바꿨고 무엇을 알았나", ""]
    for n, r in enumerate(it["iterations"], 1):
        L += [f"### {n}. {r['id']} — {r['title']}", "",
              f"- **시험한 원인**: {r['cause_tested']}",
              f"- **바꾼 것**: {r['change_plain']}",
              "- **돌린 것**: " + " / ".join(
                  f"{x['label']}: inner {x['inner_cv_gm']:.4f} · 홀드아웃 "
                  f"{x['holdout_gm_65']:.4f}" for x in r["runs"]),
              f"- **판정**: {r['verdict']} — {r['interpretation']}",
              f"- **상수의 출처**: {r['constants_provenance']}", ""]
    L += ["## 반박 검증 (가장 좋은 둘)", ""]
    for v in it["verification"]:
        L += [f"### {v['id']} · {v['lens']} — {'버팀' if v['holds_up'] else '⛔ 못 버팀'}",
              "", v["summary"], ""]
        L += [f"- {p}" for p in v["problems"]]
        L.append("")
    (D / "ledger.md").write_text("\n".join(L) + "\n")
    print(f"  -> {D / 'ledger.md'}")


if __name__ == "__main__":
    main()
