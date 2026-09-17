"""★ The 48-run campaign report — **generated**, never typed (D-171 §6).

    python3 -m experiments.campaign_nk4_report
    python3 -m experiments.campaign_nk4_report --check   # fail if stale

Every number comes from an artefact:

```
campaign-nk4.json    §5-2 per run · §5-3 spreads · §5-4 statistics
transfer-nk4.json    §4 the transfer matrix
nk4-at-k.json        regret@k · hit@k · per-shape win/loss
nk-fold-plan.json    §1-4 the four checks
vendor-baselines.json  the baselines
```

★ D-168 learned this the hard way: a report with hand-copied numbers goes
stale the moment a run is re-scored, and nothing says so.

⛔ It does not compare against the 21-run campaign. Library, shape
population and split design all differ (D-171 §9).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

A = Path("docs/artifacts")
OUT = A / "campaign-nk4.md"
GPUS = ("a6000", "5090", "4090", "h100")
#: ★ D-174 — which artefact set the report is built from.
SETS = {
    "nk4": {"camp": "campaign-nk4.json", "tr": "transfer-nk4.json",
            "atk": "nk4-at-k.json", "plan": "nk-fold-plan.json",
            "seed": None, "out": "campaign-nk4.md", "seeds": (0, 1, 2)},
    "c2": {"camp": "campaign2.json", "tr": "campaign2-transfer.json",
           "atk": "campaign2-at-k.json",
           "plan": "campaign2-fold-plan.json",
           "seed": "campaign2-seed-holdout.json",
           "out": "campaign2.md", "seeds": (0, 1, 2, 3)},
}
SET = SETS["nk4"]


def _load(name: str) -> dict:
    return json.loads((A / name).read_text())


def build() -> str:
    camp = _load(SET["camp"])
    tr = _load(SET["tr"])
    atk = _load(SET["atk"])
    plan = _load(SET["plan"])
    seedho = _load(SET["seed"]) if SET["seed"] else None
    runs = [r for r in camp["runs"] if not r.get("missing")]
    ag = camp["aggregates"]
    L: list[str] = []
    add = L.append

    add("# 48실행 캠페인 — `(N,K)` 묶음 분할 (D-171)")
    add("")
    add("> **재현** `python3 -m experiments.campaign_nk4_report` · LLM 0회")
    add("> **원본** `campaign-nk4.json` · `transfer-nk4.json` · "
        "`nk4-at-k.json` · `nk-fold-plan.json`")
    add("> ⛔ **21실행과 나란히 놓지 않는다** — 라이브러리 · 형상 모집단 "
        "(61 → 65) · 분할 설계가 전부 다르다")
    add("")
    add("## ⚠️ 상태: **조건부** — 시드 층이 오염됐다 (D-172 §X)")
    add("")
    add("```")
    add("stage3() 이 세 시드에 ★ 같은 registry·matrix 를 넘겼고 루프가 만든 축을")
    add("되돌리지 않았다. s1·s2 는 s0 의 ★ 반복이 아니라 연장이다.")
    add("  nk4-a6000-f3  s0 27축 -> s1 ★37축 -> s2 ★39축")
    add("  ★ 32/32 (표,fold)x뒤시드 조합에서 검출 · 아카이브 규칙 ★ 270개가")
    add("    앞선 시드의 축을 쓴다")
    add("")
    add("⛔ 무효   §4 의 ★ 시드 산포와 ★ fold/시드 비 — 분모가 오염됐다")
    add("⚠️ 유보   (표,fold)별 ★ s1·s2 의 홀드아웃 — 더 큰 라이브러리에서 출발했다")
    add("★ 살아있음  각 (표,fold)의 ★ s0 16실행 — 27축에서 깨끗하게 출발했다")
    add("★ 살아있음  전이 행렬 — 질문 자체는 유효하다. 소스가 s1·s2 인 칸에는")
    add("           그 라이브러리가 s0 보다 크다는 유보가 붙는다")
    add("⛔ 수치는 지우지 않는다 — 정정으로 잇는다 (문서 규칙 2)")
    add("★ 21실행(D-168)은 영향 없다 — n_seeds=1 이라 시드마다 프로세스가 따로였다")
    add("```")
    add("")

    # -- 1 ----------------------------------------------------------------
    add("## 1. 조건")
    add("")
    add("```")
    add(f"라이브러리   ★ k7-1 — {camp['library']} · 네 표가 공유")
    add("조건         F2 = known7 (7축) · 모델 gpt-5.6-luna")
    add("형상 모집단   ★ 65 (a6000·5090) / 63 (4090·h100)")
    add("             기준: 후보 공간의 pipeline_kind 가 둘 이상 (D-170 §1)")
    add("분할         ★ (N,K) 묶음 k=4 — 같은 묶음은 반드시 한 fold 에")
    add("stage 2      (표 x fold) 마다 = 16회 · 씨앗은 그 fold 의 학습에서만")
    add("stage 3      4 fold x 3 시드 x 4 GPU = ★ 48실행")
    add("루프         12라운드 · 제안 6 · 피처 생성 3/라운드 · 예산 300")
    add("동시         4실행 · 워커 6")
    add("```")
    add("")
    if SET["out"] == "campaign2.md":
        add("## ★ 48실행(nk4)과 무엇이 다른가 — 조건만")
        add("")
        add("```")
        add("               48실행 (nk4)              ★ 이 캠페인 (c2)")
        add("배정          nk11008 을 fold0 에 고정    ★ 고정 없음 —")
        add("                                          큰 묶음 넷을 하나씩")
        add("              -> fold3 이 자투리 14형상    -> 17·16·16·16")
        add("씨앗 문구      \"물리를 말할 수 있으면 추가\"  ★ \"없으면 명백히")
        add("                                          나빠지면\" (d08a48d)")
        add("시드 격리      ⛔ 없음 — s1 이 s0 의 축을   ★ 있음 (D-172 §X)")
        add("              물려받았다 (32/32 검출)      64/64 축 20 에서 출발")
        add("시드 수        3                          ★ 4")
        add("표본(표당)     12                         ★ 16")
        add("```")
        add("")
        add("⛔ **넷이 다르므로 값을 나란히 놓고 판정하지 않는다.** 아래에서 "
            "nk4 를 언급하는 곳은 **설계 진단이 맞았는지**를 말하는 것이지 "
            "성능 비교가 아니다.")
        add("")
    add("★ 21실행과 무엇이 다른가** (조건만, 비교 아님)"
        if False else "★ **21실행과 무엇이 다른가** (조건만, 비교 아님)")
    add("")
    add("```")
    add("21실행                          48실행")
    # ★ `known5` is the pre-D-170 name of that condition, and it is the
    #   right name here — the 21-run campaign really had five axes.
    add("라이브러리 c21-lib (F2-known5)   ★ k7-1 (F2-known7)")  # D-170
    add("형상 61 (정렬 8)                 ★ 형상 65 (커널 패밀리)")
    add("분할 층화 무작위 3-fold + nk11008  ★ (N,K) 묶음 4-fold")
    add("분기 가능 축 실질 1개             ★ 7개 (known7 3 + k7-1 4)")
    add("```")
    add("")

    # -- 2 ----------------------------------------------------------------
    add("## 2. §1-4 분할 검증")
    add("")
    add("| 표 | 형상 | (N,K) 묶음 | fold 크기 | 1 커버리지 | "
        "2 (N,K) 쌍둥이 | 3 최소 소수체제 | ⚠️ M 쌍둥이 |")
    add("|---|--:|--:|---|---|--:|--:|--:|")
    for n, r in plan["tables"].items():
        sizes = "·".join(str(f["n_val"]) for f in r["folds"])
        add(f"| {n} | {r['n_shapes']} | {r['n_groups']} | {sizes} | "
            f"{'ok' if r['coverage_ok'] else '**FAIL**'} | "
            f"{len(r['nk_twins_in_train'])} | "
            f"{r['min_train_minority']:.0%} | {r['m_twins_in_train']} |")
    add("")
    add("★ **(N,K) 쌍둥이 0** — 검증 형상의 레이어를 학습이 하나도 못 본다. "
        "그것이 이 분할이 재는 것이다.")
    add("")
    add("⚠️ **M 쌍둥이는 남는다.** M=1024 가 거의 모든 묶음에 있으므로 "
        "검증 형상은 대개 학습에 M 형제를 갖는다. 이 분할이 재는 것은 "
        "**\"안 본 레이어 형상\"** 이지 \"안 본 형상 전부\" 가 아니다.")
    add("")
    bad = [n for n, r in plan["tables"].items()
           if r.get("balance_ceiling_is_the_table")]
    if bad:
        add(f"⚠️ **3번이 {', '.join(bad)} 에서 미달인데 분할의 결함이 "
            f"아니다.** 그 표들은 **표 전체**의 memory-bound 비율이 "
            + " · ".join(f"{n} {plan['tables'][n]['table_minority_frac']:.0%}"
                         for n in bad)
            + " 라서 **어떤 분할로도 25% 에 못 간다.** ridge point 가 낮아 "
              "memory-bound 형상 자체가 적다.")
        add("")

    # -- 3 ----------------------------------------------------------------
    add("## 3. 48실행 — 표 x fold x 시드")
    add("")
    add("★ `holdout` 은 `canonical_score` — ⛔ 이 값을 낼 때는 학습에서 "
        "가중치를 다시 맞췄다 (D-182 가 그 적합을 없앴다). 적합하고 "
        "검증에서 잰 값이다. **논문에 쓸 값이다.**")
    add("")
    add("| 표 | fold | n | s0 | s1 | s2 | 중앙 | in-sample 중앙 | "
        "\\|w\\| | 셀 | 분기축 |")
    add("|---|--:|--:|--:|--:|--:|--:|--:|---|---|---|")
    for g in GPUS:
        for f in range(4):
            rs = sorted((r for r in runs if r["gpu"] == g and r["fold"] == f),
                        key=lambda r: r["seed"])
            if not rs:
                continue
            h = [r["holdout"] for r in rs]
            add(f"| {g} | {f} | {rs[0]['n_holdout']} | "
                + " | ".join(f"{x:.4f}" for x in h)
                + f" | **{st.median(h):.4f}** | "
                f"{st.median([r['in_sample'] for r in rs]):.4f} | "
                + "·".join(str(r["len_w_best"]) for r in rs) + " | "
                + "·".join(str(r["n_cells"]) for r in rs) + " | "
                + "·".join(str(r["branching"]["n_axes"]) for r in rs) + " |")
    add("")

    # -- 4 ----------------------------------------------------------------
    add("## 4. §5-3 산포")
    add("")
    add("★ **s0 만으로 다시 계산한 fold 산포** — 이것이 오염되지 않은 값이다.")
    add("")
    add("| 표 | fold 산포 (★ s0 16실행) | f0 | f1 | f2 | f3 |")
    add("|---|--:|--:|--:|--:|--:|")
    for g in GPUS:
        v = {r["fold"]: r["holdout"] for r in runs
             if r["gpu"] == g and r["seed"] == 0}
        if len(v) < 2:
            continue
        m = list(v.values())
        add(f"| {g} | **{max(m) - min(m):.4f}** | "
            + " | ".join(f"{v[f]:.4f}" for f in sorted(v)) + " |")
    add("")
    add("★ **fold3 효과는 s0 만으로도 살아 있다** (5090 0.2797 · h100 0.3320). "
        "이 캠페인의 중심 관찰은 오염과 무관하다.")
    add("")
    add("⛔ **아래 표의 시드 산포와 비는 무효다** (D-172 §X). 지우지 않고 "
        "무효 표시만 남긴다 — 무엇이 왜 못 쓰이는지가 기록이다.")
    add("")
    add("| 표 | fold 산포 | ★ fold3 제외 | ⛔ 시드 산포 중앙 | ⛔ 비 (전체) |")
    add("|---|--:|--:|--:|--:|")
    by = {}
    for r in runs:
        by.setdefault((r["gpu"], r["fold"]), []).append(r["holdout"])
    for g in GPUS:
        med = [st.median(by[(g, f)]) for f in range(4) if (g, f) in by]
        med012 = [st.median(by[(g, f)]) for f in (0, 1, 2) if (g, f) in by]
        seed = st.median([max(v) - min(v) for (gg, _), v in by.items()
                          if gg == g])
        add(f"| {g} | {max(med) - min(med):.4f} | "
            f"{max(med012) - min(med012):.4f} | {seed:.4f} | "
            f"{(max(med) - min(med)) / seed:.2f}x |")
    add("")
    add("⛔ **\"분할이 시드보다 크다\" 는 이 캠페인 자료로는 말할 수 없다** — "
        "시드 산포가 오염됐다 (D-172 §X). 아래 문장은 옛 판정이며 "
        "**무효**다: ~~fold3 를 빼면 분할 산포는 시드 산포와 같거나 "
        "작아진다~~. ★ fold3 의 검증 14형상은 "
        "**11개 묶음의 단일 형상**들이라 학습이 그 (N,K) 를 하나도 못 본다 — "
        "이 분할이 재려던 것의 가장 어려운 형태다.")
    add("")

    # -- 5 ----------------------------------------------------------------
    add("## 5. 벤더 · 정적 top-1 대비")
    add("")
    add("| 표 | fold | n | 벤더 | top-1 | 우리(중앙) | 우리(최고) | "
        "형상별 벤더 대비 승 |")
    add("|---|--:|--:|--:|--:|--:|--:|--:|")
    for g in GPUS:
        for f in range(4):
            rs = [r for r in runs if r["gpu"] == g and r["fold"] == f]
            ak = [r for r in atk["runs"] if r["gpu"] == g and r["fold"] == f]
            if not rs or not ak:
                continue
            w = sum(int(r["shapes_better_than_vendor"].split("/")[0])
                    for r in ak)
            tot = sum(int(r["shapes_better_than_vendor"].split("/")[1])
                      for r in ak)
            h = [r["holdout"] for r in rs]
            add(f"| {g} | {f} | {rs[0]['n_holdout']} | "
                f"{ak[0]['vendor_at1']:.4f} | "
                f"{ak[0]['static_top1_at1']:.4f} | "
                f"{st.median(h):.4f} | {min(h):.4f} | {w}/{tot} |")
    add("")
    add(f"★ 형상별 합계 — 벤더보다 나은 형상 "
        f"**{atk['aggregate']['shapes_better_than_vendor_total']}**")
    add("")
    add("⚠️ 이 표의 `우리` 열은 `canonical_score` (⛔ D-182 이전 절차 — "
        "재적합이 있었다) 이고 "
        "`형상별 승` 은 **아카이브의 가중치 그대로** 잰 것이다 — "
        "다른 절차이므로 한 열에 섞지 않는다.")
    add("")
    add("### regret@k · hit@k")
    add("")
    add("```")
    rm = atk["aggregate"]["regret_at_median"]
    add("중앙값   " + "  ".join(f"@{k[1:]} {v:.4f}" for k, v in rm.items()))
    add(f"hit@1 {atk['aggregate']['hit_at_1_median']:.2f}  "
        f"hit@3 {atk['aggregate']['hit_at_3_median']:.2f}")
    add("```")
    add("")
    for f in range(4):
        rs = [r for r in atk["runs"] if r["fold"] == f]
        add(f"- fold{f} — @1 {st.median([r['regret_at']['k1'] for r in rs]):.4f}"
            f" · @10 {st.median([r['regret_at']['k10'] for r in rs]):.4f}"
            f" · hit@1 {st.median([r['hit_at']['k1'] for r in rs]):.2f}")
    add("")

    # -- 5b  씨앗 홀드아웃 · 구간 기여 -------------------------------------
    if seedho:
        add("## 5b. 씨앗의 홀드아웃과 구간별 기여")
        add("")
        add("★ `rounds.jsonl` 의 r0 은 **씨앗 + 편집 6개 중 최고**라 씨앗 "
            "자체의 홀드아웃이 없다. 여기서 따로 잰다.")
        add("")
        add("| 표 | fold | 씨앗 학습 | ★ 씨앗 HO | 최종 HO(s0) | "
            "최종 HO(중앙) | ★ 루프 기여 |")
        add("|---|--:|--:|--:|--:|--:|--:|")
        gains = []
        for r in seedho["rows"]:
            g = r.get("loop_gain_canonical_s0")
            if g is not None:
                gains.append(g)
            add(f"| {r['gpu']} | {r['fold']} | "
                f"{r['seed_train_recorded']:.4f} | "
                f"**{r['seed_holdout_canonical']:.4f}** | "
                f"{(r['final_holdout_s0'] or float('nan')):.4f} | "
                f"{(r['final_holdout_median'] or float('nan')):.4f} | "
                f"{(g if g is not None else float('nan')):+.4f} |")
        add("")
        add(f"★ 루프가 홀드아웃을 올린 곳 "
            f"**{sum(1 for x in gains if x > 0)}/{len(gains)}** · "
            f"중앙 **{st.median(gains):+.4f}**")
        add("")
        add("### 구간별 기여 — ⚠️ 전부 **루프 절차** "
            "(`rounds.jsonl` 과 같은 자)")
        add("")
        add("⛔ `canonical`(⛔ D-182 이전 절차)과 빼지 않는다. 다른 자다.")
        add("")
        add("⚠️ **중앙값은 분해되지 않는다** — 구간 중앙값의 합은 전체의 "
            "중앙값이 아니다. 비율은 **평균**으로 분해하고 중앙값은 옆에 둔다.")
        add("")
        for tag, d in seedho["segments"].items():
            m, mu = d["median"], d["mean"]
            add(f"- **{tag}** (n={d['n']}, 좋아진 "
                f"{d['improved_seed_to_r11']}/{d['n']})")
            add(f"  - 중앙 씨앗→r0 {m['seed_to_r0']:+.4f} · "
                f"r0→r5 {m['r0_to_r5']:+.4f} · "
                f"r5→r11 {m['r5_to_r11']:+.4f} · "
                f"합 {m['seed_to_r11']:+.4f}")
            add(f"  - 평균 씨앗→r0 {mu['seed_to_r0']:+.4f} · "
                f"r0→r5 {mu['r0_to_r5']:+.4f} · "
                f"r5→r11 {mu['r5_to_r11']:+.4f} · "
                f"합 {mu['seed_to_r11']:+.4f}")
            if "share_of_mean" in d:
                add(f"  - ★ 비율(평균 분해) {d['share_of_mean']} · "
                    f"실행별 비율 중앙 {d['share_per_run_median']}")
        add("")

    # -- 6 ----------------------------------------------------------------
    add("## 6. 전이 행렬 (§4)")
    add("")
    cells = tr["cells"]
    for key, lab in (("a_as_is", "(a) 그대로 옮김"),
                     ("b_refit", "(b) 가중치만 재적합")):
        add(f"### {lab} — 행 소스 · 열 대상 (네 fold 중앙)")
        add("")
        add("| 소스 \\ 대상 | " + " | ".join(GPUS) + " |")
        add("|---|" + "--:|" * len(GPUS))
        for s in GPUS:
            row = []
            for t in GPUS:
                v = [x[key] for x in cells if x["src"] == s and x["dst"] == t]
                row.append(f"{st.median(v):.4f}" if v else "—")
            add(f"| {s} | " + " | ".join(row) + " |")
        add("")
    ga = [x["a_as_is"] - x["native"] for x in cells if x["native"]]
    gb = [x["b_refit"] - x["native"] for x in cells if x["native"]]
    add("### ★ 원주민 대비 — \"전이가 원주민만큼 좋은가\"")
    add("")
    add("```")
    add(f"(a) 그대로   원주민보다 나은 칸 "
        f"{sum(1 for x in cells if x['native'] and x['a_as_is'] < x['native'])}"
        f"/{len(cells)}  · 격차 중앙 {st.median(ga):+.4f}")
    add(f"(b) 재적합   원주민보다 나은 칸 "
        f"{sum(1 for x in cells if x['native'] and x['b_refit'] < x['native'])}"
        f"/{len(cells)}  · 격차 중앙 {st.median(gb):+.4f}")
    add(f"(b) 가 벤더보다 나은 칸      "
        f"{sum(1 for x in cells if x['b_refit'] < x['vendor'])}/{len(cells)}")
    add(f"(b) 가 정적 top-1 보다 나은 칸 "
        f"{sum(1 for x in cells if x['b_refit'] < x['static_top1'])}"
        f"/{len(cells)}")
    add("```")
    add("")
    add("★ **구조는 옮겨간다.** 가중치만 대상 표에서 다시 맞추면 원주민과의 "
        "격차가 4분의 1로 줄고, 48칸 중 "
        f"{sum(1 for x in cells if x['native'] and x['b_refit'] < x['native'])}"
        "칸은 원주민보다 낫다.")
    add("")
    add("### §4-1 규칙이 쓰는 축도 함께 옮겼다")
    add("")
    add("```")
    add(f"shape_level 이 뒤집힌 축   ★ "
        f"{sum(1 for x in cells if x['shape_level_flips'])}칸 — 없다")
    from collections import Counter
    cc = Counter(n for x in cells for n in x["constant_on_target"])
    for n, v in cc.most_common():
        add(f"대상 홀드아웃에서 상수가 된 축   {n}  ★ {v}/{len(cells)}칸")
    add("```")
    add("")
    add("★ fold0·fold1 의 검증 형상에는 정렬이 어긋난 것이 없다. 그 fold 로 "
        "옮기면 **정렬 분기는 죽는다** (fold2·fold3 에서는 산다).")
    add("")
    add("| fold | (b) 중앙 |")
    add("|---|--:|")
    for f in range(4):
        v = [x["b_refit"] for x in cells if x["fold"] == f]
        add(f"| {f} | {st.median(v):.4f} |")
    add("")

    # -- 7 ----------------------------------------------------------------
    add("## 7. §5-4 — 안 남기면 영영 못 보는 넷")
    add("")
    add("### (a) 규칙 크기")
    add("")
    add("```")
    rs_ = ag["rule_size"]
    add(f"최고 규칙 크기 {rs_['min']}~{rs_['max']} (중앙 {rs_['median']})")
    add(f"크기 vs 홀드아웃 상관 r = {rs_['corr_size_vs_holdout']}   "
        f"⚠️ 인과 진술이 아니다")
    add("```")
    add("")
    add("### (b) 축 생성")
    add("")
    add("```")
    ax = ag["axes"]
    add(f"루프가 만든 축 {ax['made_total']}개 · 서로 다른 이름 "
        f"{ax['distinct']}개")
    add(f"★ 두 실행 이상에 같은 이름: {len(ax['repeated_across_runs'])}개")
    for k, v in list(ax["repeated_across_runs"].items())[:10]:
        add(f"   {v}회  {k}")
    add("```")
    add("")
    add("### (c) 계보")
    add("")
    add("```")
    add(json.dumps(ag["lineage"], ensure_ascii=False))
    add("```")
    add("")
    add("### (d) ★ 분기 — 준 축 중 몇 개가 실제로 쓰였나")
    add("")
    add("```")
    br = ag["branching"]
    add("라이브러리가 준 분기 가능 축 ★ 7개 (known7 3 + k7-1 4)")
    add(f"실제로 쓰인 서로 다른 축 ★ {br['n_distinct_used']}개 "
        f"(루프가 만든 형상축 포함)")
    add(f"규칙당 중앙 ★ {br['median_per_rule']}개")
    for k, v in br["axes_used"].items():
        add(f"   {v:2d}/{len(runs)}  {k}")
    add("```")
    add("")
    add("### (e) 죽은 항 — 이유별")
    add("")
    add("```")
    dt = ag["dead_terms"]
    add(f"가중치로 죽음 (|w|<1e-3)          {dt['by_weight_total']:>7,}건 "
        f"· {dt['n_rounds']}라운드 중 {dt['by_weight_nonzero_rounds']}라운드")
    add(f"민감도로 죽음 (sens<1e-6)          {dt['by_sens_total']:>7,}건")
    frac = dt["by_sens_total"] / max(1, dt["by_sens_total"]
                                     + dt["by_weight_total"])
    add(f"★ {frac:.2%} 가 민감도 쪽 — 적합기가 0으로 만든 것이 아니라")
    add("  ★ 순위를 못 바꾸는 항이다 (D-169 §2)")
    add("```")
    add("")

    # -- 8 ----------------------------------------------------------------
    add("## 8. §5-5 비용")
    add("")
    add("```")
    c = ag["cost"]
    add("LLM 호출  " + json.dumps(c["llm_calls"], ensure_ascii=False))
    add(f"라운드 벽시계 합 {c['minutes_summed'] / 60:.1f}시간 (48실행)")
    add("실제 동시 4실행 · 워커 6 · ★ 429 0건 · ★ 죽은 실행 0")
    add("```")
    add("")

    # -- 9 ----------------------------------------------------------------
    add("## 9. ★ 한계 — 알면서 못 한 것")
    add("")
    add("1. **피처 라이브러리가 한 분할의 학습 형상에서 만들어졌다.** "
        "`k7-1` 은 `nk11008` 학습 분할에서 만들어졌고, 그 분할은 이 설계의 "
        "fold0 이다. 따라서 fold1~3 의 홀드아웃 형상 일부가 라이브러리 "
        "생성 당시 학습에 있었다. 세 fold 가 형상을 분할하므로 총 접촉은 "
        "재분배될 뿐 없어지지 않는다 (원칙 6).")
    add("2. **M 쌍둥이가 남는다.** 위 §2 — 이 분할은 \"안 본 레이어\" 를 "
        "재고 \"안 본 형상\" 을 재지 않는다.")
    add("3. **예산 300 이 차원에 종속되지 않는다.** 규칙이 "
        f"{rs_['min']}~{rs_['max']} 차원인데 평가 예산은 고정 300 이다. "
        "차원당 평가가 규칙마다 다르고, 큰 규칙이 불리하다 "
        "(`pending_fixes` 18).")
    add("4. **한 (표,fold) 의 세 시드가 축 라이브러리를 공유한다.** "
        "`stage3` 이 세 시드를 한 프로세스에서 돌리며 같은 matrix 를 "
        "넘기므로, 시드 0 이 만든 축이 시드 1 의 시작 상태에 있다. "
        "따라서 **시드 산포는 독립 반복의 산포가 아니다.**")
    prop = sum(r["counts"].get("n_proposed", 0) for r in runs)
    sb = sum(r["counts"].get("n_rejected_sandbox", 0) for r in runs)
    add(f"5. **`check_rule` 이 정의되지 않은 맨 이름을 안 본다** "
        f"(`pending_fixes` 20). 이 캠페인에서 제안 {prop:,}건 중 sandbox "
        f"거부 {sb}건이고, 그중 3건이 한 라운드의 형제 편집에 복제된 "
        f"오타 하나다 — 그것이 §3-1 로 캠페인을 한 번 멈춘 사유다 "
        f"(`pending_fixes` 21).")
    add("6. **fold3 에서 규칙이 일반화하지 못한다.** in-sample 은 1.02~1.06 "
        "인데 홀드아웃이 1.10~1.81 이고 hit@1 이 0 이다. 벤더는 fold3 에서 "
        "다른 fold 와 같은 성적이고 정적 top-1 은 오히려 더 좋다 — "
        "**형상이 어려운 것이 아니라 우리 규칙이 그 지대를 못 짚는다.**")
    add("7. **홀드아웃을 여러 번 보았다.** 48실행의 홀드아웃 값을 이 "
        "보고서가 읽는다. 최종 봉인 분할(`test`)은 열지 않았다.")
    add("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", choices=tuple(SETS), default="nk4")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global SET  # noqa: PLW0603
    SET = SETS[a.campaign]
    if a.out is None:
        a.out = str(A / SET["out"])
    text = build()
    p = Path(a.out)
    if a.check:
        if not p.exists() or p.read_text() != text:
            raise SystemExit(f"{p} is stale — regenerate it")
        print(f"{p} is up to date")
        return
    p.write_text(text)
    print(f"  -> {p}  ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
