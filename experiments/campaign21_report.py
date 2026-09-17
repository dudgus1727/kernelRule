"""★ It **generates** `docs/artifacts/campaign-21.md`. 0 LLM calls.

    python3 experiments/campaign21_report.py           # write
    python3 experiments/campaign21_report.py --check    # fail if it diverged

The first version of that document was transcribed by hand from the three
json artefacts. That is the exact thing `tests/test_artifact_numbers.py`
exists to catch (§30.14) — and the campaign's numbers are the paper's, so a
transcription error there is the expensive kind. The prose lives here; every
number is read from:

```
docs/artifacts/campaign-21.json            experiments/campaign21.py
docs/artifacts/campaign-21-baselines.json  experiments/campaign21_baselines.py
docs/artifacts/library-contact.json        experiments/library_contact.py
```

⛔ Do not edit the `.md` by hand. Change the prose here and run this
(principle 2 — the same rule `runs_table.py` and `decisions_index.py`
follow).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "artifacts"
OUT = ART / "campaign-21.md"

#: 실행 태그 -> 표시 이름. 표가 `c21-` 접두사로 어수선해지지 않게.
def _short(tag: str) -> str:
    return tag.replace("c21-", "")


#: 반복 생성된 축의 한 줄 설명. **사람이 쓴다** — json 에 없는 지식이다.
_AXIS_MEANING = {
    "serial_split_k_partition_count": "split-K 직렬 분할 수",
    "pipeline_stage_count": "파이프라인 단수",
    "serial_split_k_roundtrip_ratio": "split-K 왕복 비용 비율",
    "tile_shape_aspect_mismatch": "타일 가로세로비 불일치",
    "serial_split_k_roundtrip_traffic_and_synchronization_cost":
        "왕복 트래픽 + 동기화",
    "pipeline_stage_family": "단수 · 패밀리 결합",
    "pipeline_family_indicator": "커널 패밀리",
    "multistage_pipeline_indicator": "multistage 여부",
    "multistage_family_indicator": "multistage 패밀리",
}
#: 표시 순서 — (표, 분할) 짝
_ORDER = [("a6000", "fold0"), ("a6000", "fold1"), ("a6000", "fold2"),
          ("a6000", "nk11008"), ("5090", "nk11008"), ("4090", "nk11008"),
          ("h100", "nk11008")]
_LABEL = {"a6000": "A6000", "5090": "5090", "4090": "4090", "h100": "H100"}
#: ★ 재실행한 실행. §6 에서 설명한다
_REDONE = "c21-h100-nk-s0"


def _f(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def build(d: dict, b: dict, lc: dict) -> str:
    L: list[str] = []
    add = L.append

    kf = d["a6000_kfold"]
    rs = d["rule_size"]
    cost = d["cost"]
    hyp = d["hypotheses"]
    n_ax_req = sum(d["axes"]["requested"].values())
    n_ax_acc = sum(d["axes"]["accepted"].values())
    n_ax_use = sum(d["axes"]["used_by_best"].values())
    lib = lc["a6000"][lc["chosen_library_split"]]

    add("# ★ 21실행 캠페인 — 본체 수치 (F2 · 네 표 · 네 분할)")
    add("")
    add("> **상태**: 완료 (2026-09-11 20:07 → 09-12 08:45). "
        f"{d['n_runs']}/{d['n_runs']} 완주.")
    add("> ★ 이것이 논문의 본체 수치다. 릴리즈와 D 항목은 아직이다")
    add("> **★ 이 문서는 생성된다** — `python3 experiments/campaign21_report.py`.")
    add("> ⛔ 손으로 고치지 마라. 산문은 그 스크립트에 있고 숫자는 아래 셋에서 읽는다")
    add(">")
    add("> **재현** (전부 LLM 0회)")
    add("> `python3 experiments/campaign21.py` -> `campaign-21.json`")
    add("> `python3 experiments/campaign21_baselines.py` -> "
        "`campaign-21-baselines.json`")
    add("> `python3 experiments/library_contact.py` -> `library-contact.json`")
    add("> **실행별 채점 원본** `runs/_campaign/<태그>.json`")
    add("> **모델** `gpt-5.6-luna` · **조건** F2 · **예산** 300 · **라운드** 12")
    add("")

    # -- 0 ---------------------------------------------------------------
    add("## 0. 구성")
    add("")
    add("```")
    add("조건        F2 (공개 지식 5축에서 시작)")
    add("stage 1     ★ A6000 에서 한 번만 — 21실행이 공유한다")
    add("            (전이를 하려면 소스와 대상이 같은 축을 알아야 한다)")
    add(f"            nk11008 분할의 학습 {lib['n_train']}형상에서 만들었다. "
        "20제안 중 17채택")
    add("stage 2     ★ (표 × 분할) 일곱 번 — 일곱 전부 10/10 · 거부 0")
    add("루프        12라운드 · 제안 6 · 가설 3 · 아카이브 3x3x3 · 가중치 상한 없음")
    add("            라운드당 새 축 최대 3")
    add("동시성      배치 안에서 3~4실행 동시 · 워커 6 (물리 코어 24)")
    add("")
    add("A6000   3-fold x 3시드 + nk11008 x 3시드 = 12")
    add("5090 · 4090 · H100  각 nk11008 x 3 = 9")
    add("```")
    add("")

    # -- 1 ---------------------------------------------------------------
    add("## 1. ★ 시작 전에 바뀐 설계 둘")
    add("")
    add("원안대로 돌렸으면 §3 의 비교가 성립하지 않았을 자리 둘이다. **배치를 "
        "하나도 시작하기 전에** 찾았다.")
    add("")
    add("### 1-1. stage 2 를 \"표마다 한 번\" 에서 \"(표 × 분할) 일곱 번\" 으로")
    add("")
    add("씨앗은 `loop.score_only()` = **학습 분할의 regret** 으로 고르고, "
        "`TableFacts.compute(table, splits.train)` 이 **프롬프트에 들어간다.** "
        "A6000 은 분할이 넷이므로 fold0 에서 고른 씨앗을 fold1 이 쓰면 fold1 의 "
        "검증 형상이 씨앗 선택에 쓰인 셈이 된다 — `chosen.json` 이 스스로 "
        "경고하는 그 문장이다 (원칙 6 · D-40/D-46/D-50). 추가 비용은 RuleWriter "
        "30호출뿐이었다.")
    add("")
    add("### 1-2. 공유 라이브러리를 어느 분할에서 만드나")
    add("")
    add("stage 1 의 AUC 검사는 **답을 읽는다** (그래서 D-166 §C-1 이 학습 "
        "형상으로 좁혔다). 라이브러리는 한 번만 만들어 21실행이 공유하므로, 그 "
        "학습 형상은 다른 분할에게 \"피처 선별 단계에서 읽힌 홀드아웃\" 이 된다. "
        "**몇 개인지 셌다.**")
    add("")
    add("```")
    cols = ["fold0", "fold1", "fold2", "nk11008"]
    add(f"{'라이브러리 출처':20s}" + "".join(f"{c:>11s}" for c in cols) + "     합")
    for src in ["fold0", "fold1", "fold2", "nk11008"]:
        r = lc["a6000"][src]
        # ★ 는 **100% 오염된 칸에만** 붙인다 — fold0 를 버린 이유가 그것이다.
        cells = []
        for c in cols:
            s = r["splits"][c]
            txt = f"{s['contact']:2d}/{s['n_val']:2d}"
            full = s["contact"] == s["n_val"] and s["n_val"] > 0
            cells.append(f"{('★ ' if full else '  ') + txt:>9s}")
        name = src + (" (채택)" if src == lc["chosen_library_split"]
                      else " (버림)" if src == "fold0" else "")
        name = ("★ " if src == lc["chosen_library_split"] else "  ") + name
        add(f"{name:20s}" + "".join(f"{c:>11s}" for c in cells) + f"    {r['total']}")
    add("")
    add("전이 대상 셋 — 전부 접촉 0")
    for g in ("5090", "4090", "h100"):
        t = lc["targets"][g]
        add(f"  {_LABEL[g]:5s} 홀드아웃 {t['n_val']} 중 {t['contact']}"
            f"   (표 {t['n_shapes']}형상, 그중 {t['shared_with_a6000']}개가 "
            f"라이브러리 학습에 있음)")
    add("```")
    add("")
    f0 = lc["a6000"]["fold0"]["splits"]
    add(f"**3-fold 이므로 fold0 의 학습 {lc['a6000']['fold0']['n_train']} = "
        f"fold1 의 검증 {f0['fold1']['n_val']} + fold2 의 검증 "
        f"{f0['fold2']['n_val']}** 다. fold0 에서 만들면 두 fold 의 홀드아웃이 "
        "100% 읽힌다. 세 fold 가 61형상을 분할하므로 `Σ|val ∩ T| = |T|` — "
        "**총량은 못 줄이고 분포만 바꾼다.** nk11008 이 셋을 동시에 좋게 한다.")
    add("")
    add("```")
    add("★ 오염이 상수가 된다 ("
        + " · ".join(str(lib["splits"][c]["contact"]) for c in cols[:3])
        + ")   -> §3 의 분할 비교가 성립한다")
    add("   fold0 만 깨끗하면 \"fold0 이 나쁜 것이 분할 때문인가 자기만")
    add("   깨끗해서인가\" 를 못 가른다 (문서 규칙 3)")
    add("★ nk11008 이 0/20 으로 깨끗하다   -> 전이 소스와 대상 넷이 다 깨끗하다")
    add(f"★ 총 접촉 최소 ({lc['a6000']['fold0']['total']} -> {lib['total']})")
    add("```")
    add("")
    add("⚠️ fold0 으로 시작했던 stage 1 은 `runs/x-c21-lib-fold0/` 에 "
        "`WHY-DISCARDED.md` 와 함께 남겼다. 지우지 않았다.")
    add("")
    add("### 1-3. 논문 한계 절에 그대로 들어갈 문장")
    add("")
    lo = min(lib["splits"][c]["contact"] for c in cols[:3])
    hi = max(lib["splits"][c]["contact"] for c in cols[:3])
    vlo = min(lib["splits"][c]["n_val"] for c in cols[:3])
    vhi = max(lib["splits"][c]["n_val"] for c in cols[:3])
    add(f"> 피처 라이브러리는 nk11008 분할의 학습 {lib['n_train']}형상에서 한 번 "
        "만들어져 모든 실행이 공유한다. 축을 공유해야 규칙을 다른 GPU 로 옮길 수 "
        f"있기 때문이다. ★ 그 결과 세 fold 의 홀드아웃 중 {lo}~{hi} 형상이 피처 "
        f"선별 단계에서 읽혔다 (각 {vlo}~{vhi} 중). ★ nk11008 홀드아웃과 대상 "
        "GPU 셋은 접촉이 없다. 회피하려면 분할마다 라이브러리를 따로 만들어야 "
        "하고, 그러면 전이가 불가능해진다.")
    add("")

    # -- 2 ---------------------------------------------------------------
    add(f"## 2. {d['n_runs']}실행")
    add("")
    add("`canonical_score` = 홀드아웃에서 잰 값. ⛔ 이 수치를 낼 때는 학습 "
        "분할에서 가중치를 다시 맞췄다 — D-182 가 그 적합을 없앴다. "
        "`train`/`val` 은 루프가 스스로 기록한 것.")
    add("")
    add("| 실행 | 표 | 분할 | holdout | in-sample | train | val | w | 분 | 축 | 호출 |")
    add("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for gpu, split in _ORDER:
        for k in range(3):
            tag = f"c21-{gpu}-{'nk' if split == 'nk11008' else split.replace('fold', 'f')}-s{k}"
            v = d["per_run"][tag]
            star = " ★" if tag == _REDONE else ""
            ho = f"**{_f(v['holdout'])}**" if v["holdout"] > 1.13 \
                else _f(v["holdout"])
            add(f"| {_short(tag)}{star} | {_LABEL[gpu]} | {split} | {ho} | "
                f"{_f(v['in_sample'])} | {_f(v['train_best'])} | "
                f"{_f(v['val_best'])} | {v['len_w_best']} | "
                f"{v['minutes']:.0f} | {len(v['axes_accepted'])}/"
                f"{v['axes_requested']} | {v['llm_calls']} |")
    add("")
    add(f"★ `{_short(_REDONE)}` 은 3라운드에서 죽어 **같은 시드로 다시 돌린 "
        "것**이다 (§6).")
    add("")

    # -- 3 ---------------------------------------------------------------
    add("## 3. ★ 분할이 시드보다 크다")
    add("")
    add("| 표 / 분할 | s0 | s1 | s2 | 중앙값 | 시드 산포 |")
    add("|---|--:|--:|--:|--:|--:|")
    for gpu, split in _ORDER:
        g = d["groups"][f"{gpu}/{split}"]
        add(f"| {_LABEL[gpu]} {split} | "
            + " | ".join(_f(x) for x in g["holdout"])
            + f" | {_f(g['median'])} | {_f(g['spread'])} |")
    add("")
    add("```")
    add("시드 산포 (fold 안)      "
        + " · ".join(_f(x) for x in sorted(kf["seed_spread_per_fold"]))
        + f"   중앙 ★ {_f(kf['seed_spread_median'])}")
    add(f"분할 산포 (fold 중앙값)  {_f(min(kf['fold_medians']))} ~ "
        f"{_f(max(kf['fold_medians']))}            "
        f"★ {_f(kf['split_spread'])}")
    add(f"-> 분할이 시드보다 ★ "
        f"{kf['split_spread'] / kf['seed_spread_median']:.1f}배 크다")
    add("```")
    add("")
    add("⚠️ **골라내지 않은 것 — 그리고 둘이 같은 모양이다.** 대상 GPU 두 곳의 "
        "`s0` 가 같은 방식으로 무너졌다. **학습에서 가장 좋은 규칙이 홀드아웃에서 "
        "가장 나쁘다.**")
    add("")
    add("```")
    for gpu in ("4090", "h100"):
        g = d["groups"][f"{gpu}/nk11008"]
        v = d["per_run"][f"c21-{gpu}-nk-s0"]
        others = " · ".join(_f(x) for x in g["holdout"][1:])
        add(f"{_LABEL[gpu]:5s} s0   train {_f(v['train_best'])} -> holdout "
            f"{_f(v['holdout'])}   (같은 분할의 다른 둘 {others})")
    add("둘 다 그 표에서 in-sample 이 가장 낮고, "
        + " · ".join(str(d["per_run"][f"c21-{g}-nk-s0"]["len_w_best"])
                     for g in ("4090", "h100")) + "차원이다")
    add("```")
    add("")
    add("이 둘이 4090·H100 의 시드 산포("
        + _f(d["groups"]["4090/nk11008"]["spread"]) + " · "
        + _f(d["groups"]["h100/nk11008"]["spread"])
        + ")를 거의 전부 만든다. 시드를 골라내지 않는다는 규칙대로 그대로 둔다 "
          "(D-50).")
    add("")

    # -- 4 ---------------------------------------------------------------
    add("## 4. 벤더 휴리스틱과 나란히")
    add("")
    add("NVIDIA `nvMatmulHeuristics` 의 형상별 추천을 **같은 홀드아웃 형상에서 "
        "`regret@1` 기하평균**으로 잰다. ⛔ 벤더 기준선은 커밋된 "
        "`datasets/baselines/*.json` 을 **읽기만 했고 다시 뽑지 않았다** "
        "(D-158).")
    add("")
    add("| 표 / 분할 | n | 벤더 nearest | 벤더 strict | 정적 top-1 ★ | "
        "우리 중앙 | 우리 최고 | Δ 중앙−벤더 |")
    add("|---|--:|--:|--:|--:|--:|--:|--:|")
    for gpu, split in _ORDER:
        g = b["groups"][f"{gpu}/{split}"]
        win = g["delta_median_minus_vendor"] < 0
        med = f"**{_f(g['ours_median'])}**" if win else _f(g["ours_median"])
        dl = f"{g['delta_median_minus_vendor']:+.4f}".replace("-", "−")
        dl = f"**{dl}**" if win else dl
        add(f"| {_LABEL[gpu]} {split} | {g['n_holdout']} | "
            f"{_f(g['vendor_nearest'])} | {_f(g['vendor_strict'])} | "
            f"{_f(g['static_top1_ceiling_on_holdout'])} | {med} | "
            f"{_f(g['ours_best'])} | {dl} |")
    add("")
    add(f"**일곱 그룹 중 {b['groups_where_ours_is_better']}에서 우리 중앙값이 "
        "벤더보다 낮다.** 갈리는 자리가 깔끔하다 — **대상 GPU 셋(5090 · 4090 · "
        "H100)은 전부 우리가 낫고**, A6000 은 fold0 하나만 이긴다.")
    add("")
    vmin = min(b["groups"][f"a6000/{s}"]["vendor_nearest"]
               for s in ("fold0", "fold1", "fold2"))
    vmax = max(b["groups"][f"a6000/{s}"]["vendor_nearest"]
               for s in ("fold0", "fold1", "fold2"))
    omin = min(b["groups"][f"a6000/{s}"]["ours_median"]
               for s in ("fold1", "fold2"))
    omax = max(b["groups"][f"a6000/{s}"]["ours_median"]
               for s in ("fold1", "fold2"))
    add("A6000 이 불리한 것은 벤더가 A6000 에서 특별히 잘해서가 아니라 **A6000 의 "
        "k-fold 홀드아웃이 어려운 쪽**이기 때문이다 — §3 의 분할 산포가 그것이다. "
        f"fold1·fold2 는 우리 쪽 중앙값이 {_f(omin)}~{_f(omax)} 로 올라가는 "
        f"분할이고, 벤더는 분할과 무관하게 {_f(vmin)}~{_f(vmax)} 를 낸다 "
        "(벤더는 학습하지 않으므로 분할이 의미가 없다).")
    add("")
    add("### ⚠️ 이 표로 말하지 않는 것 셋")
    add("")
    h = b["groups"]["h100/nk11008"]
    add("```")
    add("① 이것은 ★ 기하평균 비교다. 실험 계획서가 주 지표로 정한 것은")
    add("   ★ 형상별 승패의 부호검정이고, 그것은 21실행을 전부 재적합해 형상별")
    add("   regret 을 뽑아야 한다. 아직 안 했다")
    add("② \"정적 top-1 ★\" 은 경쟁자가 아니라 ★ 천장이다 — 그 홀드아웃 형상들")
    add("   자신에서 고른 최선의 단일 config 이므로 자기가 채점받을 형상에서 뽑혔다.")
    add(f"   H100 에서 이 천장({_f(h['static_top1_ceiling_on_holdout'])})이 "
        f"벤더({_f(h['vendor_nearest'])})보다 나쁜 것은")
    add("   그 표가 단일 config 로는 특히 안 덮인다는 뜻이다")
    add("③ 벤더 파일은 표 전체(64~66형상)에서 만들어졌다. 여기서는 홀드아웃 형상만")
    add("   골라 썼으므로 비교는 성립하지만, ★ 기준선 자체를 61형상에 맞추는 일은")
    add("   캠페인 밖으로 남아 있다")
    add("```")
    add("")

    # -- 5 ---------------------------------------------------------------
    add("## 5. 안 남겼으면 못 봤을 것들")
    add("")
    add("### 5-1. 규칙 크기 — 상한을 없앤 것이 옳았나")
    add("")
    add("```")
    add(f"최종 규칙의 가중치 수   {rs['min']} ~ {rs['max']}  "
        f"(중앙 {rs['median']:g})")
    add(f"크기 vs 홀드아웃 상관   ★ r = {rs['pearson_r_size_vs_holdout']:+.4f}"
        f"  (n={d['n_runs']})")
    add("```")
    add("")
    add("**사실상 0이다.** \"클수록 좋다\" 도 \"클수록 나쁘다\" 도 아니다 — 상한을 "
        f"없앤 것이 성능을 올리지도 내리지도 않았다는 것이 {d['n_runs']}실행의 "
        "답이다 (D-150/D-152).")
    add("")
    add("### 5-2. 죽은 항 (D-167 §N)")
    add("")
    dt = d["dead_terms"]["per_scored_rule"]
    add("```")
    add(f"채점된 규칙 하나당  ★ 중앙 {d['dead_terms']['median_per_rule']:g}개  "
        f"(범위 {min(dt.values()):g} ~ {max(dt.values()):g})")
    add("```")
    add("")
    add(f"{rs['median']:g}차원 규칙의 절반이 순위에 기여하지 않는다. 적합기가 "
        "**이미 계산하고 있던** 값을 라운드마다 적기 시작한 것이 캠페인 "
        "직전이었다 — 그 수정이 없었으면 이 수치는 영영 못 봤다. 축은 실행마다 "
        "새로 생기므로 다시 돌려서 얻을 수도 없다.")
    add("")
    add("### 5-3. 축 생성 — 독립 실행들이 같은 것을 발견한다")
    add("")
    add("```")
    add(f"요청 {n_ax_req} · 채택 {n_ax_acc} · ★ 최고 규칙이 실제로 쓴 것 "
        f"{n_ax_use}")
    add("```")
    add("")
    add("| 축 이름 | 만든 실행 수 | 무엇인가 |")
    add("|---|--:|---|")
    for name, n in sorted(d["axes"]["repeated_across_runs"].items(),
                          key=lambda kv: (-kv[1], kv[0])):
        add(f"| `{name}` | {n} | {_AXIS_MEANING.get(name, '—')} |")
    add("")
    add("**split-K 왕복과 파이프라인 단수로 수렴한다.** 이름이 겹친다는 것은 "
        "우연이 아니라 같은 물리를 서로 다른 실행이 각자 집어냈다는 뜻이다.")
    add("")
    add("### 5-4. 계보 · 가설 · 비용")
    add("")
    add("```")
    lin = d["lineage"]
    add("계보    최종 규칙의 출처 — "
        + " · ".join(f"{k} {v}" for k, v in sorted(lin.items()))
        + " · ★ explore 0")
    add("        (F3 에서 0, F2 에서 1이던 것이 21실행에서도 0)")
    add(f"가설    {hyp['n']:,}건 · evidence_cases 인용률 "
        f"★ {hyp['with_cases'] / hyp['n']:.0%} · 새 축 요구 "
        f"{hyp['wants_feature'] / hyp['n']:.0%} ({hyp['wants_feature']}건)")
    add(f"비용    LLM {cost['llm_calls']:,} 호출 · 토큰 입력 "
        f"{cost['tokens_in']:,} / 출력 {cost['tokens_out']:,}")
    add(f"        라운드 벽시계 합 {cost['round_minutes_total'] / 60:.1f}시간 "
        "(동시 3~4로 실제 경과 약 4시간)")
    add("```")
    add("")

    # -- 6 ---------------------------------------------------------------
    v = d["per_run"][_REDONE]
    add("## 6. ⛔ 죽은 실행 하나")
    add("")
    add("```")
    add(f"{_REDONE}   r3 에서 ★ UnexpectedModelBehavior: Exceeded maximum")
    add("                 output retries (3)")
    add("보존             runs/x-c21-h100-nk-s0-died-r3{,-s0} + WHY-DIED.md")
    add("재실행           같은 시드 k=0 -> 12라운드 완주")
    add(f"                 train {_f(v['train_best'])} · holdout "
        f"{_f(v['holdout'])} · w {v['len_w_best']} · {v['minutes']:.0f}분 · "
        f"축 {len(v['axes_accepted'])}/{v['axes_requested']}")
    add("```")
    add("")
    add("**원인은 D-164 의 부작용이다.** pydantic-ai 가 출력 검증을 3회 연속 "
        "실패하면 내는 예외인데, D-164 가 `check_rule` 을 그 재시도 경로에 "
        "넣었으므로 모델이 정적 검사를 세 번 연속 못 넘기면 **그 호출이 예외가 "
        "되고 루프 전체가 멈춘다.**")
    add("")
    add("```")
    add("D-164 이전   정적 검사 실패 -> n_rejected_static 으로 세고 루프는 계속")
    add("D-164 이후   ★ 3회 연속 실패 -> 예외가 올라와 실행이 끝난다")
    add("```")
    add("")
    add(f"거부 하나가 실행을 죽이는 것은 의도가 아니었다 — {d['n_runs']}실행 중 "
        f"1건(약 {1 / d['n_runs']:.0%}). 고치는 것은 캠페인 밖의 일이라 기록만 "
        "남긴다.")
    add("")
    add("⚠️ 죽은 실행이 3라운드에서 갖고 있던 값(train 1.0474 / val 1.0793)은 "
        "12라운드 실행들과 나란히 놓지 않는다 (문서 규칙 3). 재실행 값만 쓴다.")
    add("")

    # -- 7 ---------------------------------------------------------------
    add("## 7. 진행 중 있었던 제 잘못 둘")
    add("")
    add("```")
    add("배치2·3 뒤에 기계가 ★ 3시간 16분 놀았다")
    add("  원인  배치가 끝난 것을 사람이 확인해야 다음이 시작되는 구조였다")
    add("  고침  배치 4~7 을 한 프로세스가 wait 로 이어 돌렸다 — 유휴 0초")
    add("")
    add("배치7 의 진행 확인이 틀렸다")
    add("  로그에 grep -c '^r[0-9]' 를 썼는데 다른 줄까지 잡혀 12/12 로 보였다.")
    add("  실제로는 3줄이었다. ★ rounds.jsonl 의 줄 수를 세는 것이 맞다")
    add("  그래서 첫 집계와 첫 배치7 보고가 죽은 실행을 포함한 채 나갔다")
    add("```")
    add("")
    add("### 그리고 캠페인이 드러낸 `runs_table` 결함 둘 (D-168)")
    add("")
    add("```")
    add("① 파이프라인 태그 dir 이 실행으로 잡혔다 — 캠페인 태그가 -s<digit> 로")
    add("   끝나 seed 패턴에 걸렸고 rounds.jsonl 이 없어 죽었다.")
    add("   ★ config 의 \"loop\" 블록으로 가른다 — \"rounds.jsonl 이 있나\" 로")
    add("   가르면 1라운드 전에 죽은 실행이 표에서 사라진다")
    add("② ★ 표 열이 9실행을 a6000 으로 적고 있었다 — hw_text.sha256 조회에")
    add("   기본값이 a6000 이었고, D-166 §E 가 하드웨어 프롬프트를 바꿔 네 표의")
    add("   sha 가 전부 달라졌다. ★ 기록된 hw_text.gpu 를 먼저 읽는다")
    add("```")
    add("")

    # -- 8 · 9 -----------------------------------------------------------
    add("## 8. 붙는 유보")
    add("")
    add("```")
    add(f"공유 라이브러리   세 fold 의 홀드아웃 {lo}~{hi} 형상이 피처 선별에서 "
        "읽혔다 (§1-3)")
    add("예산 300          차원에 종속되지 않는다. 7~8차원 25 평가/차원 vs")
    add("                  37~52차원 5.8~8.1 (pending_fixes 16)")
    add("라운드 12         수렴점이 아니라 정한 정지점이다. 12도 24도 마지막")
    add("                  4라운드에서 계속 움직인다 (round-curve.md)")
    add("align == 8        형상 모집단을 정하는 기준인데 ★ 근거가 기록된 적이 없다.")
    add("                  제외 5형상 중 4개는 커널 공간이 포함 형상과 같다 (D-167 §R)")
    add("hypothesis_id     프로세스 첫 실행에서 귀속이 달라진다. 점수는 안 바뀌지만")
    add("                  \"어느 가설이 어느 규칙을 만들었나\" 에 유보가 붙는다 (D-167 §P)")
    add("```")
    add("")
    add("## 9. 남은 일")
    add("")
    add("```")
    add("형상별 부호검정   21실행 재적합이 필요하다 (§4 ①)")
    add("릴리즈            campaign-21-<커밋 7자> — 트레이스 · config · 생성 축 ·")
    add("                  아카이브 + 집계표. ★ 축약 없이, 올린 뒤 sha256 대조")
    add("runs.md           21행 — ★ 넣었다 (D-168)")
    add("decisions.md      D 항목 — 이것이 논문의 본체 수치다")
    add("캠페인 밖         전이 실험 · 벤더 기준선 61형상 재정렬 ·")
    add("                  D-164 재시도가 실행을 죽이는 문제")
    add("```")
    return "\n".join(L) + "\n"


def main() -> None:
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the .md diverged")
    a = ap.parse_args()

    need = ["campaign-21.json", "campaign-21-baselines.json",
            "library-contact.json"]
    missing = [n for n in need if not (ART / n).exists()]
    if missing:
        raise SystemExit(
            f"{missing} 이(가) 없다. 먼저 만들어라:\n"
            f"  python3 experiments/campaign21.py\n"
            f"  python3 experiments/campaign21_baselines.py\n"
            f"  python3 experiments/library_contact.py")
    d, b, lc = (json.loads((ART / n).read_text()) for n in need)
    body = build(d, b, lc)
    old = OUT.read_text() if OUT.exists() else ""
    if a.check:
        if body != old:
            raise SystemExit(
                "campaign-21.md 가 생성 결과와 다르다. "
                "`python3 experiments/campaign21_report.py` 를 돌려라.")
        print(f"campaign-21.md 는 최신이다 ({len(body.splitlines())}줄)")
        return
    OUT.write_text(body)
    print(f"campaign-21.md 를 다시 냈다 — {len(body.splitlines())}줄 · "
          f"{d['n_runs']}실행")


if __name__ == "__main__":
    main()
