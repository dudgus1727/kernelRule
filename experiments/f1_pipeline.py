"""★ F0~F3 파이프라인 — 피처부터 규칙까지 LLM 이 만든다 (§30.9).

    python3 experiments/f1_pipeline.py F1 --dry-run
    python3 experiments/f1_pipeline.py F1 --n-features 20 --n-seeds 3
    python3 experiments/f1_pipeline.py F1 --stage 2      # 1단계 산출물 재사용

## 왜 필요한가

지금까지의 모든 진화 실행은 **사람이 만든 재료를 조합**한 것이었다.

    피처   `features/physical.py` 의 24개   <- 사람이 물리 문서를 보고 씀
    씨앗   `rules/physics_seeded.py`        <- 사람이 씀. 그 24개 중 6개를 씀
    루프   24개 중 8개 고르기

`experiments/feature_writer.py` 로 "LLM 이 피처를 만들 수 있다" 는 확인했다.
그러나 그 피처들은 **사람 24개에 더해진 상태**로만 시험됐다. 아직 안 해 본
것이 근본 질문의 완성형이다.

    "F1 피처만으로 규칙을 만들 수 있는가"

## 세 단계

    1  FeatureWriter   원시 값만 -> 피처 N개            -> stage1-features/
    2  RuleWriter       1단계 피처 목록만 -> 씨앗 규칙   -> stage2-rule-writer/
    3  RoundLoop       1단계 레지스트리 + 2단계 씨앗    -> stage3-evolution/

**F1/F0 에서는 사람이 만든 24개가 세 단계 어디에도 안 들어간다.** 그것을
보장하는 것이 `features/__init__.py` 의 "레지스트리 기본값 없음" 이고,
`tests/test_features.py` 의 AST 검사가 그것을 고정한다.

## 조건

    F3  REGISTRY(24개) + physics_seeded 씨앗   = 지금까지의 모든 실행
    F2  기초 5개 + FeatureWriter 로 확장
    F1  원시 값만 -> FeatureWriter -> RuleWriter 씨앗   ★ 근본 질문
    F0  피처 없음 -> FeatureWriter 가 전부

F3 도 **이 경로로** 돌아야 한다. 다른 스크립트로 돌리면 경로 차이가
결과에 섞인다 (§26.2).
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import kernelrule.features.physical  # noqa: F401  — REGISTRY 를 채운다
from kernelrule.agents.mock import MockLLM
from kernelrule.agents.openai_client import DEFAULT_MODEL, Budget, LLMConfig
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet, check_balance
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY, FeatureRegistry
from kernelrule.features.generated import (
    SHAPE_LEVEL_REASON,
    FeatureRejected,
    register_generated,
)
from kernelrule.features.validate import alt_hw
from kernelrule.report.table_facts import TableFacts

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
OUT = Path("runs")

#: F2 의 "기초 5개". **여기 한 곳에서만 정한다** — 흩어지면 갈린다 (원칙 2).
#: 원시 값에 가장 가까운 것들이다. 파생 물리량은 하나도 없다.
F2_BASE = ("log_grid_tiles", "log_mainloop_iters", "smem_pressure",
           "reg_pressure", "is_memory_bound")

#: ★ F1-K 사전 등록. `docs/artifacts/f1k-preregistration.md` 와 **같은
#: 내용**이고 `tests/test_f1k_prereg.py` 가 갈리지 않는지 검사한다.
#: **LLM 을 한 번도 안 부른 상태에서 박았다** — 실행 직전에 쓰면 배관을
#: 만들며 생긴 감이 기준에 스며든다 (D-50).
F1K_PREREG = {
    "purpose": ("알려진 축을 주면 새 축을 더 만드나. 그리고 라이브러리가 "
                "좋아지나"),
    "expected": ("새 축 개수가 F1 보다 많다 (재발견에 예산을 안 쓰므로). "
                 "★ 진화 성능이 F3 를 따라잡을지는 모른다. 못 따라잡아도 "
                 "실패가 아니다. F1 보다 나을지도 열린 질문이다."),
    "start_library": 5,
    "areas": 7,
    "per_category": 3,
    # ⚠️ 이 키는 **동결된 기록**이다. D-93 에서 역할 이름을 바꿨지만
    #   사전 등록은 그때의 이름으로 쓰였다 — 고치면 기록을 다시 쓰는 것이다
    #   (문서 규칙 2). 살아 있는 이름은 `--n-rule-writer` 다.
    "n_architect": 10,
    "n_seeds": 6,
    "rounds": 12,
    "primary_metrics": ["새 축 개수와 사람 24개 대비 상관",
                        "진화 후 구조 홀드아웃 (F1 과 같은 정준 절차)"],
    "not_a_criterion": ("재발견 개수 — 5개를 줬으니 줄어드는 게 당연하다. "
                        "이것으로 조건을 평가하지 않는다"),
    "two_variables": ("시작 5개 + GPU 예시. **분리하지 않는다** — 둘 다 "
                      "'공개 지식을 준다' 의 일부다. D-31 의 예외이고 "
                      "'어느 쪽 덕인가' 는 이 실험으로 못 가른다"),
    "on_failure": {
        "영역 3회 연속 거부": "건너뛰고 기록. 진행",
        "채택 절반 미만": ("멈추고 ★ 거부 사유 분포부터 본다 — 중복 다수면 "
                          "정상(다섯을 줬다), §8.3 실패 다수면 검사기·필드 "
                          "문제, 스키마 실패 다수면 프롬프트 문제. "
                          "인프라 -> 검사기 -> 피험자 순서다 (원칙 8)"),
        "RuleWriter 전부 거부": "멈추고 보고",
        "3실행 연속 빈 아카이브": "멈춤"},
    "not_doing": ["나머지 19개를 넣지 않는다 — F3 조건이다",
                  "--recategorize 를 쓰지 않는다 — 고정 일곱",
                  "F1 결과를 지우지 않는다 — 비교 대상이다",
                  "모델을 바꾸지 않는다 (D-45, 원칙 25)",
                  "결과를 보고 프롬프트를 고치지 않는다 (§12.3d)"],
    "threshold_rationale": ("절반은 '실험이 성립하는 최소 요건' 이지 F1 "
                            "실측(80%) 대비로 조인 것이 아니다. F1-K 는 "
                            "다섯을 줬으니 중복 거부가 늘 수 있고 그것은 "
                            "정상 동작이다"),
    "budget_calls": 990,
    "discrimination_note": ("시드 폭 sigma=0.0274 때문에 조건 간 0.02급 "
                            "차이는 못 가린다 (D-53). 비슷하면 '구분 불가' "
                            "가 정직한 서술이다"),
}

#: LLM 이 나눌 영역 수의 범위. **사람이 영역을 정하지 않는다** — 정하면
#: 사전 지식을 건네는 것이다. 개수 범위만 준다 (§30.10).
CAT_MIN, CAT_MAX = 5, 8

#: 한 영역에서 이만큼 연속 거부되면 그 영역을 건너뛴다.
#: "이 영역은 원시 값으로 표현하기 어렵다" 가 기록으로 남는다.
CAT_GIVE_UP = 3


def _plan(cats: list[dict], n_features: int, per_cat: int) -> list[str | None]:
    """무엇을 몇 번 만들지. ★ 개수를 **영역 수에서 유도**한다 (§30.10).

    전에는 20 고정이었다. 임의적이라 채우려고 억지 피처가 나온다.
    영역이 6개면 `6 x per_cat` 이고, 영역이 없으면(자유 생성) 옛 방식대로
    `n_features` 회다.
    """
    if not cats:
        return [None] * n_features
    plan: list[str | None] = []
    for _ in range(per_cat):                # 라운드 로빈 — 한 영역에 몰리지 않게
        plan.extend(c["name"] for c in cats)
    return plan


def _task(cat: str | None, cats: list[dict], made_in: dict[str, list[str]],
          gen: FeatureRegistry, base: FeatureRegistry) -> str:
    """이번 제안의 지시. 영역이 있으면 그 영역 안에서 만들게 한다."""
    if cat is None:
        made = sorted(set(gen._items) - set(base._items))
        tail = (f"\n\n지금까지 만든 것: {made}. 이것들과 다른 축을 찾으세요."
                if made else "")
        return "## 이번에 만들 것\n\n피처 하나를 제안하세요." + tail
    desc = next(c["description"] for c in cats if c["name"] == cat)
    mine = made_in.get(cat, [])
    other = sorted(set(gen._items) - set(base._items) - set(mine))
    return (f"## 이번에 만들 것\n\n**영역: `{cat}`** — {desc}\n\n"
            f"이 영역의 물리량을 하나 제안하세요.\n\n"
            f"```\n이 영역에서 이미 만든 것: {mine or '없음'}\n"
            f"다른 영역의 것(중복 판정 참고): {other or '없음'}\n```\n\n"
            "★ **이 영역 안에서** 만드세요. 다른 영역으로 넘어가면 그 영역의\n"
            "차례에 만들 것이 없어집니다.")


#: RuleWriter 산출물이 정적 검사에 걸릴 때 몇 번까지 다시 부를까.
#: ★ 전부 실패하면 **에러다** — 씨앗 없이 조용히 진행하지 않는다 (§26.4).
ARCH_RETRIES = 3


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------


def _splits(table: PerfTable) -> SplitSet:
    """구조 분할 — 11008 레이어를 통째로 홀드아웃. 기존 실행과 같다 (§10.1)."""
    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    s = SplitSet(train=Split("train", tuple(p for p in shapes
                                            if p not in held)),
                 val=Split("val", tuple(held)), kind="nk11008")
    check_balance(s.train, table.hw)
    return s


def _base_registry(condition: str) -> FeatureRegistry:
    """조건이 정하는 **출발 레지스트리**. F0/F1 은 비어 있다."""
    if condition in ("F0", "F1"):
        return FeatureRegistry(f"{condition}-empty")
    if condition == "F1-K":
        # ★ 공개 지식 다섯 (§30.17). `physical.py` 의 원본이 아니라
        #   **표 관측을 뺀 정리본**이다 — 원본 docstring 에는 "이 표에서
        #   스필 커널은 최적 0회" 같은 측정 결과가 있다 (§12.3).
        from kernelrule.features.known5 import KNOWN5
        r = FeatureRegistry("F1-K-known5")
        for n in sorted(KNOWN5._items):
            r.add(KNOWN5[n])
        return r
    if condition == "F2":
        r = FeatureRegistry("F2-base")
        for n in F2_BASE:
            r.add(REGISTRY[n])
        return r
    if condition == "F3":
        r = FeatureRegistry("F3-human24")
        for n in sorted(REGISTRY._items):
            r.add(REGISTRY[n])
        return r
    raise ValueError(f"알 수 없는 조건: {condition!r}. F0/F1/F1-K/F2/F3")


def _make_llm(a, *, registry: FeatureRegistry, budget: Budget):
    """★ `registry` 는 필수다 — 어느 피처 목록이 프롬프트에 들어가는지가
    실험 조건 자체다 (§30.9). MockLLM 도 같은 목록을 받는다."""
    names = sorted(n for n in registry._items if not registry[n].shape_level)
    svals = sorted(n for n in registry._items if registry[n].shape_level)
    if a.dry_run:
        return MockLLM("mutate", seed=a.seed, feature_names=names,
                       shape_values=svals)
    from kernelrule.agents.openai_client import OpenAILLM
    return OpenAILLM(LLMConfig(model=a.model, concurrency=6),
                     feature_names=names, shape_values=svals,
                     registry=registry, budget=budget, cache=False)


def _dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------
# 1단계 — FeatureWriter
# ---------------------------------------------------------------------------
def stage1(a, d: Path, table, matrix, base: FeatureRegistry) -> FeatureRegistry:
    """피처를 만든다. ★ 제안마다 즉시 append 한다 (D-33).

    **영역을 먼저 나눈다** (§30.10). 전에는 `range(20)` 을 돌며 "지금까지
    만든 것과 다른 축" 이라고만 지시했는데, 방향이 없어서 처음 몇 개가
    앉은 자리 근처에서 맴돌았다 — luna 17개 중 `split_k_*` 가 3개,
    `cta_*` 가 5개였다. 그리고 20 이라는 수가 임의적이라 채우려고 억지
    피처가 나온다.

        1  영역 나누기 (LLM 1회)      -> categories.json
        2  영역별 생성 (영역당 N회)    -> 개수가 영역 수에서 유도된다

    영역을 **사람이 주지 않는다** — 그것은 사전 지식을 건네는 것이다.
    LLM 이 어떻게 나누는지 자체가 관찰 대상이다.
    """
    out = d / "stage1-features"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "proposals.jsonl"
    log.write_text("")

    gen = FeatureRegistry(f"{a.condition}-generated")
    for n in sorted(base._items):          # F2/F3 는 기초 위에 쌓는다
        gen.add(base[n])

    # ★ 기존 라이브러리를 **확장**한다 (D-63 후속). 앞선 실행의 채택분을
    #   레지스트리에 먼저 넣어야 (1) 중복 판정이 그것들과도 이뤄지고
    #   (2) 산출물이 합집합이 된다. 앞선 제안 이력을 그대로 앞에 복사하므로
    #   `load_generated` 가 읽는 형식이 유지된다.
    prior_lines: list[str] = []
    if a.extend_from:
        src = Path(a.extend_from) / "proposals.jsonl"
        if not src.exists():
            raise SystemExit(f"{src} 가 없다.")
        from kernelrule.features.loader import load_generated
        for f in load_generated(src, exclude=set(base._items), table=table):
            gen.add(f)
        prior_lines = [ln for ln in src.read_text().splitlines() if ln.strip()]
        print(f"  ★ {src} 에서 {len(gen._items) - len(base._items)}개를 "
              f"이어받았다 — 산출물은 **합집합**이다\n")
    n_base = len(gen._items)
    n_prior = n_base - len(base._items)
    if prior_lines:                       # 이어받은 이력을 앞에 둔다
        log.write_text("\n".join(prior_lines) + "\n")

    llm = _make_llm(a, registry=gen, budget=Budget(max_calls=a.n_features * 4))
    hw_alt = alt_hw(table.hw)
    rejects: dict[str, int] = {}
    t0 = time.perf_counter()

    # -- 1. 영역 나누기 ---------------------------------------------------
    cats: list[dict] = []
    cat_notes = ""
    if (a.categorize or a.categorize_only) and not a.recategorize:
        # ★ 고정 목록 (§30.18). LLM 호출 0회.
        from kernelrule.agents.openai_client import load_prompt

        block = load_prompt("areas.md")
        body = block[block.index("```") + 3:block.rindex("```")]
        # `이름 | 설명` 으로 나눈다. 공백으로 자르면 "연산 처리량" 이
        # "연산" 에서 끊긴다.
        cats = [{"name": ln.split("|", 1)[0].strip(),
                 "description": ln.split("|", 1)[1].strip()}
                for ln in body.splitlines() if "|" in ln]
        cat_notes = "고정 목록 (prompts/areas.md). --recategorize 로 다시 뽑는다"
        _dump_json(out / "categories.json",
                   {"categories": cats, "notes": cat_notes, "source": "fixed"})
        print(f"  ★ 고정 영역 {len(cats)}개 (LLM 0회):")
        for c in cats:
            print(f"     {c['name']:16s} {c['description'][:56]}")
        print()
        if a.categorize_only:
            raise SystemExit(0)
    elif a.categorize or a.categorize_only:
        try:
            res = llm.complete("categorize", "", n_min=CAT_MIN, n_max=CAT_MAX)
            cats = [dict(c) for c in res["categories"]]
            cat_notes = res.get("notes", "")
        except Exception as e:                              # noqa: BLE001
            # ★ 조용히 자유 생성으로 떨어지지 않는다 — 조건이 바뀐다 (§26.4)
            raise RuntimeError(
                f"영역 나누기가 실패했다: {type(e).__name__}: {e}. "
                "자유 생성으로 조용히 떨어지지 않는다 — 그것은 다른 "
                "조건(F1-free)이다. `--no-categorize` 로 명시하라.") from e
        _dump_json(out / "categories.json",
                   {"categories": cats, "notes": cat_notes,
                    "n_min": CAT_MIN, "n_max": CAT_MAX})
        print(f"  ★ LLM 이 {len(cats)}개 영역으로 나눴다:")
        for c in cats:
            print(f"     {c['name']:32s} {c['description'][:60]}")
        if cat_notes:
            print(f"     (뺀 것) {cat_notes[:100]}")
        print()
        if a.categorize_only:
            # ★ 진단만. 생성은 안 한다 (D-63).
            print("  ★ --categorize-only — 여기서 끝낸다. "
                  f"기록: {out / 'categories.json'}")
            raise SystemExit(0)

    #: 영역 -> 그 영역에서 만든 이름. 프롬프트에 되먹인다.
    made_in: dict[str, list[str]] = {c["name"]: [] for c in cats}
    #: 영역 -> 연속 거부 횟수. 3회면 그 영역을 건너뛴다.
    streak: dict[str, int] = dict.fromkeys(made_in, 0)
    skipped: list[str] = []

    def dump() -> None:
        made = sorted(set(gen._items) - set(base._items))
        _dump_json(out / "summary.json", {
            "shape_level": {n: SHAPE_LEVEL_REASON[n]
                            for n in sorted(gen._items)
                            if gen[n].shape_level and n in SHAPE_LEVEL_REASON},
            "shape_level_needs_recheck": sorted(
                n for n in gen._items if gen[n].shape_level
                and "재판정" in SHAPE_LEVEL_REASON.get(n, "")),
            "categories": [c["name"] for c in cats],
            "made_by_category": made_in,
            "skipped_categories": skipped,
            "condition": a.condition, "model": a.model, "dry_run": a.dry_run,
            "n_planned": n_planned, "n_base": n_base, "n_prior": n_prior,
            "extend_from": a.extend_from, "only_category": a.only_category,
            "per_category": a.per_category, "categorize": a.categorize,
            "n_accepted": len(made) - n_prior,     # ★ 이번에 새로 만든 것만
            "n_total": len(made),                  # 이어받은 것 포함
            "rejections": rejects, "seconds": round(time.perf_counter() - t0, 1),
            "feature_names": made,
            "physics_coverage": _physics_coverage(table, gen, base)})
        (out / "features.py").write_text(_features_module(gen, base))
        if hasattr(llm, "dump"):
            llm.dump(out / "llm_calls")

    if a.only_category:
        # 부분 일치. 영역 이름은 LLM 이 매번 새로 지으므로 정확히 못 박는다 —
        # **사람이 영역을 정의하지 않는다는 원칙을 지키면서** 특정 축을
        # 보강하기 위한 절충이다 (D-63).
        # `|` 로 여러 키워드. **영역 이름은 LLM 이 매번 새로 짓고 언어도
        #  바뀐다** — 1회차는 한국어(`산술_대역폭_압력`), 2회차는 영어
        #  (`roofline_pressure`) 였다. 개념 키워드 여러 개로 고른다.
        keys = [k.strip() for k in a.only_category.split("|") if k.strip()]
        hit = [c for c in cats
               if any(k in c["name"] or k in c["description"] for k in keys)]
        if not hit:
            raise SystemExit(
                f"키워드 {keys} 에 맞는 영역이 없다. "
                f"이번에 나온 것: {[c['name'] for c in cats]}")
        cats = hit[:1]
        made_in = {cats[0]["name"]: []}
        streak = {cats[0]["name"]: 0}
        print(f"  ★ 영역 {cats[0]['name']!r} 만 생성한다 — **별도 조건**이다. "
              "비교표에 섞지 마라\n")
    plan = _plan(cats, a.n_features, a.per_category)
    n_planned = len(plan)          # ★ 영역 기반이면 개수가 여기서 정해진다
    try:
        for i, cat in enumerate(plan):
            if cat is not None and cat in skipped:
                continue
            row: dict = {"i": i, "category": cat}
            try:
                res = llm.complete("feature", "", condition=a.condition,
                                   registry=gen,
                                   task=_task(cat, cats, made_in, gen, base))
                row.update({k: res.get(k) for k in
                            ("name", "code", "unit", "direction",
                             "expected_range", "rationale")})
                f = register_generated(res["code"], registry=gen, meta=res,
                                       table=table, matrix=matrix,
                                       hw_alt=hw_alt)
                row["accepted"] = True
                row["shape_level"] = f.shape_level
                if f.shape_level:
                    row["shape_level_reason"] = SHAPE_LEVEL_REASON.get(f.name)
                if cat is not None:
                    made_in[cat].append(f.name)
                    streak[cat] = 0
                print(f"  #{i:02d}  ✓ {f.name}"
                      + (f"   [{cat}]" if cat else ""))
            except FeatureRejected as e:
                row.update(accepted=False, error=str(e)[:200])
                key = str(e).split(":")[0][:40]
                rejects[key] = rejects.get(key, 0) + 1
                print(f"  #{i:02d}  ✗ {str(e)[:90]}")
                if cat is not None:
                    streak[cat] += 1
                    if streak[cat] >= CAT_GIVE_UP:
                        skipped.append(cat)
                        print(f"       ★ 영역 {cat!r} 을 건너뛴다 — "
                              f"{CAT_GIVE_UP}회 연속 거부. "
                              "'원시 값으로 표현하기 어렵다' 가 기록된다")
            except Exception as e:                          # noqa: BLE001
                row.update(accepted=False, error=f"{type(e).__name__}: {e}"[:200])
                rejects[type(e).__name__] = rejects.get(type(e).__name__, 0) + 1
                print(f"  #{i:02d}  ✗ {type(e).__name__}: {str(e)[:70]}")
            with log.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        dump()                              # ★ 중간에 죽어도 남긴다 (D-33)

    print(f"\n  이번에 채택 {len(gen._items) - n_base}/{n_planned}  "
          f"레지스트리 총 {len(gen._items)}개"
          + (f"  (영역 {len(cats)}개 x {a.per_category})" if cats else ""))
    return gen


#: `physics_seeded` 가 쓰는 여섯 항. **여기 한 곳에서만 적는다** (원칙 2).
_SEED_TERMS = ("traffic_amplification", "sm_idle_cost", "smem_pressure",
               "has_spill", "split_k_cost", "pipeline_warmup_frac")


def _physics_coverage(table, gen: FeatureRegistry,
                      base: FeatureRegistry) -> dict:
    """★ F1 라이브러리가 손씨앗의 물리를 덮는가 — **측정할 결과다.**

    고칠 문제가 아니다. `has_spill` 하나로 1.1637 -> 3.1841 이 갈렸으니
    (§8.2), 대응이 없다는 사실 자체가 결과다. LLM 0회로 계산된다.

    각 씨앗 항에 대해 생성 피처 중 스피어만·피어슨이 가장 높은 것을 찾는다.
    판정 기준은 §8.4 와 같다 — 둘 다 0.95 초과여야 "덮었다" 다.
    """
    from kernelrule.core.matrix import FeatureMatrix
    from kernelrule.features.generated import _reference_columns
    from kernelrule.features.validate import _pearson, _spearman

    made = sorted(set(gen._items) - set(base._items))
    if not made:
        return {"note": "생성된 피처가 없다"}

    human = FeatureRegistry("seed-terms")
    for n in _SEED_TERMS:
        if n in REGISTRY._items:
            human.add(REGISTRY[n])
    ref = _reference_columns(table, FeatureMatrix(table, human),
                             FeatureRegistry("empty"))
    mine: dict = {}
    for n in made:
        one = FeatureRegistry(f"c-{n}")
        one.add(gen[n])
        mine.update(_reference_columns(table, FeatureMatrix(table, one),
                                       FeatureRegistry("empty")))

    out: dict = {}
    for name, rv in ref.items():
        best = None
        for gname, gv in mine.items():
            if len(gv) != len(rv):
                continue
            sp, pe = abs(_spearman(gv, rv)), abs(_pearson(gv, rv))
            if best is None or sp > best[1]:
                best = (gname, float(sp), float(pe))
        if best is None:
            out[name] = {"covered": False, "note": "비교 불가"}
            continue
        out[name] = {"nearest": best[0], "spearman": round(best[1], 3),
                     "pearson": round(best[2], 3),
                     "covered": best[1] > 0.95 and best[2] > 0.95,
                     "monotone_only": best[1] > 0.95 and best[2] <= 0.95}
    out["_n_covered"] = sum(1 for v in out.values()
                            if isinstance(v, dict) and v.get("covered"))
    out["_n_terms"] = len(ref)
    return out


def _features_module(gen: FeatureRegistry, base: FeatureRegistry) -> str:
    """생성된 피처를 **다시 등록 가능한 형태**로 남긴다 (§11.4)."""
    made = sorted(set(gen._items) - set(base._items))
    head = ['"""1단계 FeatureWriter 산출물. 이 파일은 **기록**이다.',
            "",
            ("다시 쓰려면 `features/loader.py::load_generated` 로"
             " `proposals.jsonl` 을"),
            "읽어라 — 그쪽이 정본이고 여기는 사람이 읽기 위한 것이다.",
            '"""', "", "import numpy as np  # noqa: F401", ""]
    body = [f"# {n}\n{gen[n].source or '(소스 없음)'}\n" for n in made]
    return "\n".join(head) + "\n".join(body)


# ---------------------------------------------------------------------------
# 2단계 — RuleWriter
# ---------------------------------------------------------------------------
def stage2(a, d: Path, table, matrix, reg: FeatureRegistry, splits) -> dict:
    """씨앗 규칙을 만든다. 학습 점수 최고를 고른다 — **홀드아웃은 안 본다**."""
    from kernelrule.agents.schemas import validate_rule_proposal
    from kernelrule.core.splits import is_unsealed
    from kernelrule.rules.checks import check_rule

    out = d / "stage2-rule-writer"
    (out / "candidates").mkdir(parents=True, exist_ok=True)

    if a.condition == "F3" and a.seed_source == "physics_seeded":
        # ★ F3 는 정의상 손씨앗을 쓴다. 그래도 **같은 경로**를 밟는다.
        from kernelrule.rules.physics_seeded import CODE, W0
        chosen = {"source": "physics_seeded", "code": CODE, "w0": list(W0),
                  "fit_regret": None,
                  "why": "F3 는 손씨앗이 조건이다 (지금까지의 모든 실행)"}
        _dump_json(out / "chosen.json", chosen)
        _dump_json(out / "summary.json", {"condition": "F3", "n_tries": 0,
                                          "source": "physics_seeded"})
        return chosen

    llm = _make_llm(a, registry=reg, budget=Budget(max_calls=a.n_rule_writer * 3))
    facts = TableFacts.compute(table, splits.train)
    loop = _loop(a, table, matrix, splits, llm, run_id=f"arch-{a.condition}")
    rows: list[dict] = []
    t0 = time.perf_counter()

    def dump() -> None:
        _dump_json(out / "summary.json", {
            "condition": a.condition, "model": a.model, "dry_run": a.dry_run,
            "n_tries": a.n_rule_writer, "n_ok": sum(r["ok"] for r in rows),
            "seconds": round(time.perf_counter() - t0, 1), "tries": rows})
        if hasattr(llm, "dump"):
            llm.dump(out / "llm_calls")

    try:
        for i in range(a.n_rule_writer):
            row = {"i": i, "ok": False}
            for attempt in range(ARCH_RETRIES):
                try:
                    res = llm.complete("rule_writer", "", condition="A",
                                       table_facts=facts, registry=reg)
                    prop = validate_rule_proposal(res)
                    check_rule(prop.code, feature_names=matrix.feature_names(),
                               shape_value_names=matrix.shape_value_names(),
                               n_weights=len(prop.w0)).raise_if_bad()
                    e = loop.score_only(prop.code, prop.w0)
                    row.update(ok=True, code=prop.code, w0=list(prop.w0),
                               fit_regret=e, attempt=attempt)
                    (out / "candidates" / f"try{i:02d}.py").write_text(prop.code)
                    print(f"  arch #{i:02d}  ✓ 학습 {e:.4f}")
                    break
                except Exception as ex:                     # noqa: BLE001
                    row.update(error=f"{type(ex).__name__}: {ex}"[:200],
                               attempt=attempt)
                    print(f"  arch #{i:02d}  ✗ (시도 {attempt + 1}) "
                          f"{type(ex).__name__}: {str(ex)[:70]}")
            rows.append(row)
    finally:
        dump()

    ok = [r for r in rows if r["ok"]]
    if not ok:
        # ★ 조용히 씨앗 없이 진행하지 않는다 (§26.4). 씨앗이 없으면
        #   1라운드가 빈 부모에서 출발하고, 그러면 "리포트를 읽고 고칠 수
        #   있는가" 라는 질문 자체가 성립하지 않는다.
        raise RuntimeError(
            f"RuleWriter {a.n_rule_writer}회 x 재시도 {ARCH_RETRIES} 가 전부 "
            f"실패했다. 씨앗 없이 진행하지 않는다 — F1 피처가 규칙을 "
            f"세우기에 부족하다는 것도 **결과**이므로 여기서 멈춘다. "
            f"산출물은 {out} 에 있다.")
    best = min(ok, key=lambda r: r["fit_regret"])
    chosen = {"source": f"rule_writer-try{best['i']:02d}", "code": best["code"],
              "w0": best["w0"], "fit_regret": best["fit_regret"],
              "why": f"{len(ok)}/{a.n_rule_writer} 성공 중 학습 점수 최고",
              "all_fit_regret": sorted(r["fit_regret"] for r in ok),
              # ★ 4-3 — 선택 시점에 무엇을 봤는가를 **기록으로** 남긴다.
              #   절차로는 지켜지고 있지만 나중에 증거가 필요하다.
              "selected_on": "train_split_regret_only",
              "holdout_seen_at_selection": False,
              "unsealed": is_unsealed(),
              "_note": ("씨앗은 `RoundLoop.score_only()` 의 학습 분할 regret "
                        "으로만 골랐다. 그 함수는 홀드아웃을 돌려주지 않는다 "
                        "— 선택이 홀드아웃을 보면 그 홀드아웃은 홀드아웃이 "
                        "아니다 (원칙 6, D-40/D-46/D-50).")}
    _dump_json(out / "chosen.json", chosen)
    print(f"\n  씨앗: {chosen['source']}  학습 {best['fit_regret']:.4f}")
    return chosen


# ---------------------------------------------------------------------------
# 3단계 — RoundLoop
# ---------------------------------------------------------------------------
def _loop(a, table, matrix, splits, llm, *, run_id: str) -> RoundLoop:
    return RoundLoop(
        cfg=LoopConfig(run_id=run_id, max_rounds=a.rounds,
                       n_rules_per_round=12, seed=a.seed,
                       max_new_features_per_round=getattr(
                           a, "max_new_features", 0),
                       feature_condition=a.condition,
                       use_analyst=not getattr(a, "no_analyst", False),
                       n_workers=getattr(a, "workers", 0),
                       hypothesis_pool=tuple(
                           getattr(a, "hypothesis_pool", []) or ())),
        table=table, matrix=matrix, splits=splits, llm=llm)


def stage3(a, d: Path, table, matrix, reg, splits, seed_rule: dict) -> None:
    out = d / "stage3-evolution"
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(max_calls=3000, max_input_tokens=60_000_000,
                    max_output_tokens=8_000_000)
    for s in range(a.n_seeds):
        run_id = f"{d.name}-s{s}"
        llm = _make_llm(a, registry=reg, budget=budget)
        loop = RoundLoop(
            cfg=LoopConfig(run_id=run_id, max_rounds=a.rounds,
                           n_rules_per_round=12, seed=100 + a.seed + s,
                           max_new_features_per_round=a.max_new_features,
                           feature_condition=a.condition,
                           use_analyst=not a.no_analyst,
                           n_workers=a.workers,
                           hypothesis_pool=tuple(a.hypothesis_pool)),
            table=table, matrix=matrix, splits=splits, llm=llm)
        loop.seed(seed_rule["code"], seed_rule["w0"], changes="stage2 씨앗")
        print(f"\n  --- {run_id} ---", flush=True)
        try:
            loop.run(a.rounds)               # RoundLoop.run 이 finally 로 dump
        except Exception as e:                              # noqa: BLE001
            print(f"  ★ 중단: {type(e).__name__}: {str(e)[:100]}")
        # ★ `llm_calls/` 는 디렉토리다. `glob("*")` + `is_file()` 로만
        #   복사하면 **LLM 호출 기록이 통째로 빠진다** — 다시 만들 수 없는
        #   것이 조용히 사라지는 경로다 (D-33).
        src = OUT / run_id
        if src.exists():
            import shutil
            dst = out / f"s{s}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
class Terminated(KeyboardInterrupt):
    """SIGTERM 을 예외로 바꾼다 — 안 그러면 `finally` 가 **안 돈다**.

    ★ 기본 SIGTERM 핸들러는 스택을 풀지 않고 즉시 죽는다. `try/finally` 로
    산출물을 남기도록 짜 놨어도 그 `finally` 가 실행되지 않는다 (D-33).
    실제로 확인했다: `timeout 25` 로 죽이면 증분 append 한
    `proposals.jsonl` 만 남고 `summary.json` 은 안 써졌다.

    증분 append 가 1차 방어선이고(LLM 호출은 절대 못 되살린다), 이것은
    2차 방어선이다.
    """


def _install_signal_handlers() -> None:
    def _die(signum, _frame):
        raise Terminated(f"신호 {signal.Signals(signum).name} 로 종료한다")

    import contextlib

    # ★ SIGTERM 만 잡는다. **SIGHUP 은 잡으면 안 된다** — 백그라운드로
    #   분리될 때 오는 정상 신호인데, 종료 예외로 바꾸면 멀쩡한 실행이
    #   시작하자마자 죽는다. 실제로 F1 실행을 두 번 그렇게 잃었다.
    #   (원칙 1 의 쌍둥이 — 안전해 보이는 처리가 판정을 하고 있었다)
    with contextlib.suppress(OSError, ValueError):   # 플랫폼/스레드 제약
        signal.signal(signal.SIGTERM, _die)


def main() -> None:
    _install_signal_handlers()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("condition", choices=("F0", "F1", "F1-K", "F2", "F3"))
    ap.add_argument("--n-features", type=int, default=20,
                    help="자유 생성(--no-categorize)일 때만 쓰인다. "
                         "영역 기반이면 개수가 영역 수에서 유도된다")
    ap.add_argument("--per-category", type=int, default=3,
                    help="영역당 제안 횟수. 총 개수 = 영역 수 x 이 값")
    ap.add_argument("--categorize", dest="categorize", action="store_true",
                    help="영역을 나눠 영역별로 생성한다. ★ 기본은 꺼짐 — "
                         "편중 완화라는 목적이 새 프롬프트만으로 이미 "
                         "달성됐고 재발견은 오히려 줄었다 (D-63)")
    ap.add_argument("--recategorize", action="store_true",
                    help="★ 영역을 LLM 으로 **다시 뽑는다** (1호출). 기본은 "
                         "`prompts/areas.md` 의 고정 일곱을 쓴다 — 매번 "
                         "뽑으면 그 실행만 다른 조건이 되는데 조용히 "
                         "그렇게 된다 (§30.18). 쓴 실행은 config 에 남는다")
    ap.add_argument("--categorize-only", action="store_true",
                    help="★ 영역 나누기 **1회만** 하고 끝낸다 (진단용). "
                         "생성에는 안 쓴다. 'LLM 이 무엇을 못 만드는가' 를 "
                         "스스로 말하게 하는 값싼 장치다 — 이것이 dtype "
                         "빈틈을 찾아냈다 (D-63)")
    ap.add_argument("--only-category", metavar="NAME",
                    help="이 영역만 생성한다 (`|` 로 여러 키워드, 이름과 "
                         "설명 둘 다에서 찾는다). 기존 라이브러리를 특정 "
                         "축으로 보강할 때 쓴다. ★ 보강분은 **별도 조건**"
                         "이므로 비교표에 섞지 마라")
    ap.set_defaults(categorize=False)
    ap.add_argument("--n-rule-writer", type=int, default=10)
    ap.add_argument("--seed-source", choices=("rule_writer", "physics_seeded"),
                    default=None,
                    help="씨앗을 어디서. 기본은 F3 면 physics_seeded, "
                         "나머지는 architect. ★ F3 에 architect 를 주면 "
                         "**대조군**이 된다 — 같은 프롬프트로 사람 24개와 "
                         "F1 라이브러리를 비교할 수 있다")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=12)
    # ★ Analyst -> FeatureWriter 경로 (D-75). **0 이 기본 = 꺼짐** — 지금까지의
    #   실행과 같은 조건이다. 켜면 그 시점 이후 실행은 **별도 계열**이다.
    # ★ 다른 캠페인의 씨앗을 그대로 쓴다 (D-83). 한 캠페인의 6실행은 2단계
    #   씨앗 하나를 공유하므로, 캠페인끼리 견줄 때 씨앗이 교락이 된다 —
    #   씨앗에 대해서는 실효 표본이 1 대 1 이다 (D-82, 원칙 28).
    ap.add_argument("--seed-from", metavar="RUN_DIR",
                    help="다른 캠페인의 stage2-rule-writer/chosen.json 을 "
                         "이 캠페인의 씨앗으로 쓴다. 출처를 chosen.json 에 "
                         "기록한다 — 2단계가 만든 것으로 오인되면 안 된다")
    # ★ §16.1 ablation. 끄면 진단 리포트도 가설도 없다 (D-89).
    ap.add_argument("--no-analyst", action="store_true",
                    help="Analyst 를 끈다 (§16.1 ablation). 진단 리포트를 "
                         "만들지도 않는다")
    # ★ §16.1 대조군 C (D-91). Analyst 는 안 부르고 **남의 가설**을 넣는다.
    ap.add_argument("--hypothesis-pool", nargs="+", default=[],
                    metavar="HYPOTHESES_JSONL",
                    help="다른 실행의 hypotheses.jsonl. Analyst 없이 그 "
                         "가설을 라운드마다 빌려 쓴다 — 같은 시드 번호의 "
                         "실행은 자동으로 뺀다")
    # ★ 채점·적합 병렬화 (D-95). 0 = 순차(기본). 결과는 같아야 한다.
    ap.add_argument("--workers", type=int, default=0, metavar="N",
                    help="채점·적합을 N 프로세스로 (0=순차). 결과는 순차와 "
                         "같다 — test_parallel_matches_sequential 이 고정한다")
    ap.add_argument("--max-new-features", type=int, default=0,
                    metavar="N",
                    help="라운드당 만들 수 있는 새 축 (0=경로 없음, D-75). "
                         "1~2 를 넘기지 마라 — §21 피처 행렬 캐시가 무효화된다")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="산출물 디렉토리 접미사. ★ 같은 조건을 다른 설정으로 "
                         "두 번 돌릴 때 **반드시** 주라 — 안 주면 같은 "
                         "디렉토리에 겹쳐 써서 앞 실행이 사라진다")
    ap.add_argument("--dry-run", action="store_true",
                    help="MockLLM 으로 배관만 확인한다. LLM 호출 0회")
    ap.add_argument("--stage", type=int, choices=(1, 2, 3),
                    help="이 단계만 실행. 앞 단계 산출물을 읽는다")
    ap.add_argument("--extend-from", metavar="STAGE1_DIR",
                    help="기존 1단계 산출물을 이어받아 **확장**한다. "
                         "중복 판정이 그것들과도 이뤄지고 산출물은 합집합이 "
                         "된다. ★ 확장분은 별도 조건이므로 비교표에 섞지 마라")
    ap.add_argument("--import-featwriter", metavar="RUN_DIR",
                    help="`experiments/feature_writer.py` 산출물을 1단계로 "
                         "가져온다. 형식이 같아서 복사면 된다 — 20호출을 "
                         "아낀다. 가져온 뒤 --stage 2 로 시작하라")
    a = ap.parse_args()
    if a.seed_source is None:
        a.seed_source = "physics_seeded" if a.condition == "F3" else "rule_writer"

    tag = a.tag or ("mock" if a.dry_run else a.model)
    d = OUT / f"f1pipe-{a.condition}-{tag}"
    # ★ 겹쳐 쓰기를 막는다. LLM 호출은 다시 만들 수 없다 (D-33).
    if d.exists() and any(d.iterdir()) and a.stage in (None, 1):
        raise SystemExit(
            f"{d} 가 이미 있고 비어 있지 않다. 겹쳐 쓰면 앞 실행의 LLM "
            f"호출이 사라진다 (D-33). `--tag` 로 다른 이름을 주거나 "
            f"지우고 다시 돌려라.")
    d.mkdir(parents=True, exist_ok=True)

    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    splits = _splits(table)
    base = _base_registry(a.condition)

    print("=" * 78)
    print(f"F0~F3 파이프라인 — 조건 {a.condition}  [{tag}]"
          + ("  ★ DRY RUN (LLM 0회)" if a.dry_run else ""))
    print("=" * 78)
    print(f"  출발 레지스트리 {base.name!r}: {len(base._items)}개")
    print(f"  학습 {len(splits.train.shapes)} / 구조 홀드아웃 "
          f"{len(splits.val.shapes)}")
    print(f"  산출물 {d}\n")

    if a.import_featwriter:
        src = Path(a.import_featwriter) / "proposals.jsonl"
        if not src.exists():
            raise SystemExit(f"{src} 가 없다.")
        dst = d / "stage1-features"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "proposals.jsonl").write_text(src.read_text())
        _dump_json(dst / "summary.json", {
            "condition": a.condition, "imported_from": str(src),
            "note": ("`feature_writer.py` 산출물을 그대로 가져왔다. 형식이 "
                     "같다 — `load_generated` 가 읽는 키가 동일하다. "
                     "physics_coverage 는 계산되지 않았다 (1단계를 안 돌렸다)")})
        n = sum(1 for ln in src.open()
                if ln.strip() and json.loads(ln).get("accepted"))
        print(f"  ★ {src} 에서 채택 {n}개를 1단계로 가져왔다. "
              f"--stage 2 로 시작하라.\n")
        return

    stages = (a.stage,) if a.stage else (1, 2, 3)

    # ★ F3 는 **정의상 사람 24개 그대로**다 (지금까지의 모든 실행). 여기서
    #   FeatureWriter 를 돌리면 레지스트리가 27개가 되어 "기존 조건" 이
    #   아니게 된다. 조용히 건너뛰지 않고 **말한다** (§26.4).
    if a.condition == "F3" and 1 in stages:
        stages = tuple(x for x in stages if x != 1)
        print("  ★ F3 는 1단계(FeatureWriter)를 돌리지 않는다 — 조건이 "
              "'사람 24개 그대로' 이기 때문이다. 새 축을 더하려면 F2 를 "
              "쓰거나 `--stage 1` 을 명시하라.\n")
    if a.stage == 1 and a.condition == "F3":
        raise SystemExit(
            "F3 에 --stage 1 은 조건과 모순이다. F3 는 사람 24개 그대로가 "
            "조건이고, 거기에 피처를 더하면 그것은 F2 다.")

    # 1단계
    if 1 in stages:
        # ★ FeatureWriter 는 **기초 레지스트리로 만든 행렬**을 본다.
        #   F1 이면 빈 행렬이라 사람 피처 값이 어디에도 안 나온다.
        m0 = FeatureMatrix(table, base)
        print("--- 1단계 FeatureWriter ---")
        reg = stage1(a, d, table, m0, base)
    elif a.condition == "F3":
        reg = base                       # 사람 24개 그대로
        print(f"--- 1단계 없음 (F3) — 사람 피처 {len(reg._items)}개 ---")
    else:
        reg = _load_stage1(d, base, a.condition, table)
        n_sh = sum(1 for n in reg._items if reg[n].shape_level)
        print(f"--- 1단계 건너뜀 — 저장된 피처 {len(reg._items)}개 "
              f"(형상 수준 {n_sh}, config 수준 {len(reg._items) - n_sh}) ---")

    matrix = FeatureMatrix(table, reg)
    from kernelrule.core.splits import is_unsealed

    _dump_json(d / "config.json", {
        "condition": a.condition, "model": a.model, "dry_run": a.dry_run,
        "seed_source": a.seed_source,
        # ★ 영역을 LLM 으로 다시 뽑았는가 (§30.18). 뽑았으면 그 실행은
        #   고정 목록을 쓴 실행과 **다른 조건**이다.
        "recategorize": a.recategorize,
        # ★ 최종 분할이 열린 채로 돈 실행인가 (§30.15)
        "unsealed": is_unsealed(),
        "seed": a.seed, "rounds": a.rounds, "n_seeds": a.n_seeds,
        "n_features": a.n_features, "n_rule_writer": a.n_rule_writer,
        "bundle": BUNDLE, "split_kind": splits.kind,
        "registry": {"name": reg.name, "n": len(reg._items),
                     "names": sorted(reg._items)},
        "human_features_present": sorted(set(reg._items) & set(REGISTRY._items))
        if a.condition in ("F0", "F1") else "N/A (F2/F3 는 의도적으로 포함)"})

    # 2단계 — ★ 뒤 단계를 안 돌 거면 **읽지도 않는다.** 전에는 `--stage 1`
    #   인데도 `chosen.json` 을 읽어서 FileNotFoundError 로 죽었다. 1단계
    #   산출물은 멀쩡했지만 종료 코드가 1 이라 실패로 보인다.
    # ★ 씨앗을 다른 캠페인에서 가져온다 (D-83). 2단계보다 **먼저** 처리해
    #   `--stage 3` 만으로도 성립하게 한다.
    if a.seed_from:
        src = Path(a.seed_from) / "stage2-rule-writer" / "chosen.json"
        if not src.exists():
            raise SystemExit(f"{src} 가 없다 — 가져올 씨앗이 없다")
        got = json.loads(src.read_text())
        got["copied_from"] = str(src)
        got["source"] = f"{got.get('source', '?')} (복사: {a.seed_from})"
        dst = d / "stage2-rule-writer"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "chosen.json").write_text(
            json.dumps(got, ensure_ascii=False, indent=1))
        print(f"  ★ 씨앗을 {src} 에서 가져왔다 — 이 캠페인의 2단계가 만든 "
              "것이 아니다")

    if 3 in stages or 2 in stages:
        if 2 in stages:
            print("\n--- 2단계 RuleWriter ---")
            chosen = stage2(a, d, table, matrix, reg, splits)
        else:
            chosen = json.loads(
                (d / "stage2-rule-writer" / "chosen.json").read_text())
            print(f"\n--- 2단계 건너뜀 — 저장된 씨앗 {chosen['source']} ---")

        # 3단계
        if 3 in stages:
            print("\n--- 3단계 진화 ---")
            stage3(a, d, table, matrix, reg, splits, chosen)

    print(f"\n완료. 산출물 {d}")


def _load_stage1(d: Path, base: FeatureRegistry, condition: str, table):
    """저장된 1단계 피처를 되살린다. ★ 없으면 조용히 기초만 쓰지 않는다.

    ★ `table` 을 반드시 넘긴다 — `shape_level` 을 **다시 판정**하기
    위해서다 (§30.12). 안 넘기면 기록된 값(대부분 없음 = False)을 쓰고,
    그러면 **형상 수준 피처가 0개인 채로** 2·3단계가 돈다. 실제로 F1
    2단계를 그 상태로 한 번 돌렸다 — `p.*` 분기가 0/10 이었는데 그것은
    LLM 에 대한 관찰이 아니라 이 결함이었다 (D-67).
    """
    from kernelrule.features.loader import extended_registry, load_generated

    path = d / "stage1-features" / "proposals.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없다. --stage 2/3 는 1단계 산출물이 필요하다. "
            "기초 레지스트리로 조용히 떨어지지 않는다 — 그러면 조건이 "
            "바뀐 채로 돌게 된다 (§26.4).")
    made = load_generated(path, exclude=set(base._items), table=table)
    # ★ 검사기 결함으로 버려졌다가 **재검사로 되살아난 것**을 함께 읽는다
    #   (`experiments/revalidate.py`). 원본 `proposals.jsonl` 은 안 고친다 —
    #   "그때 무엇이 거부됐는지" 가 사라지면 안 된다 (문서 규칙 2).
    revive = path.parent / "revalidated.jsonl"
    if revive.exists():
        extra = load_generated(revive, exclude=set(base._items) |
                               {f.name for f in made}, table=table)
        if extra:
            print(f"  ★ 재검사로 되살아난 {len(extra)}개를 더한다: "
                  f"{[f.name for f in extra]}")
        made = [*made, *extra]
    return extended_registry(base, made, name=f"{condition}-loaded")


if __name__ == "__main__":
    main()
