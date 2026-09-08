"""진단 리포트 — **이 시스템의 엔진** (§12).

LLM 이 기존 최적화 알고리즘을 이길 수 있는 유일한 지점이다. 스칼라 점수만
주면 성능 나쁜 수치 최적화기가 된다.

## 다섯 블록

    1    하드웨어 사실        매번 고정 주입. 기억에서 꺼내게 하면 틀린다
    2    현재 규칙 코드 전문
    3    체제별 regret 분해   ★ 크기 층화를 먼저 (§30.5)
    3.5  표 구조 관찰        개별 사례로는 절대 안 보이는 패턴
    4    사례 10~15개        ★ 핵심. 선택 vs 최적을 나란히
    5    실패 이력           없으면 같은 아이디어를 3라운드마다 반복한다

## ★ 검증 방법 (§12.4)

**프롬프트를 완성하기 전에 사람이 먼저 읽는다.** 이 다섯 블록만 보고
"나라면 뭘 고칠까" 가 떠오르면 리포트가 잘 만들어진 것이다.
안 떠오르면 **프롬프트가 아니라 리포트를 고친다.**

## ★ 리포트는 결론을 미리 쓰지 않는다. 전부 계산한다

    금지  "X 가 Y 보다 크게 달라진다", "A 가 지배적이다" 를 **템플릿에 박는 것**
    허용  계산 결과를 서술로 렌더링하는 것 (어느 쪽이 큰지 재서 문장을 만든다)

이 규칙이 왜 필요한지는 실제로 밟았기 때문이다. 블록 3 에 "크기 층화가
난이도 층화보다 크게 달라진다" 를 미리 적어 뒀는데, **학습 분할(M<=2048)의
실제 숫자는 반대였다** (0.0007 vs 0.1651). 그 분할에는 긴 형상이 9개뿐이라
크기 격차가 사라진다. **리포트가 자기 데이터와 모순되면 LLM 은 데이터가
아니라 문장을 믿는다.**

★ 이 규칙은 **하드웨어 사실 블록에는 적용되지 않는다.** 그것은 물리 상수이지
이 분할의 관측이 아니다. `hardware_block()` 만 예외다.

`tests/test_diagnostic.py` 가 렌더된 텍스트에 계산되지 않은 비교어가 있는지
검사한다.

## 절대 넣지 말 것 (§12.3)

    표 전체              토큰도 안 되고 넣는 순간 암기한다
    홀드아웃 점수         넣으면 홀드아웃에 맞춰 최적화한다
    모든 형상의 최적 목록   전부 주면 조건문으로 옮겨 쓴다

**구조적 강제:** `DiagnosticReport` 는 `train_shapes` 만 받는다. 검증/최종
분할이 들어오는 경로를 만들지 않는다 (§10.2).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

import numpy as np

from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.numerics import approx_equal
from kernelrule.core.scoring import Evaluation, evaluate_scores, geomean
from kernelrule.core.splits import Split, SplitError
from kernelrule.core.table import PerfTable
from kernelrule.core.types import Problem
from kernelrule.report.table_facts import TableFacts

__all__ = ["DiagnosticReport", "Case", "Regime", "build_report"]

#: 체제 정의. `(이름, 형상 술어)`.
#:
#: ★ 2026-09-08 (D-145): **memory/compute 를 앞으로**. 아카이브가 그 축으로
#: 보존하는데(D-144) Analyst 는 크기(SOL<0.5ms)로 진단하고 있었다 — 진단과
#: 보존이 다른 축을 보면 가설이 아카이브가 지키는 것을 못 짚는다.
#: 여덟 구간을 다 보여주는 것 자체는 정보이므로 유지한다.
REGIMES: tuple[tuple[str, str], ...] = (
    ("memory-bound", "mem"),
    ("compute-bound", "comp"),
    ("t_sol < 0.5ms (short)", "small"),
    ("t_sol >= 0.5ms (long)", "large"),
    ("waves < 1", "wlt1"),
    ("waves 1~4", "w14"),
    ("waves > 8", "wgt8"),
    ("K <= 1024 (short mainloop)", "smallk"),
)


@dataclass
class Regime:
    name: str
    n_shapes: int
    regret: float
    worst_shape: str = ""
    worst_regret: float = float("nan")


@dataclass
class Case:
    """사례 하나. **선택 vs 최적을 나란히 보여주는 것이 핵심이다** (§12.1)."""

    shape: tuple
    regime: str
    kind: str                 # "worst" | "best"
    regret: float
    difficulty: float
    best_ms: float
    noise_floor: float
    n_answers: int
    n_candidates: int
    picked: dict              # config 축 + 피처값
    optimum: dict
    #: 상위 5개 실측 (config 요약, ms, 최적 대비 몇 σ)
    neighbors: list[tuple]
    #: (피처명, 선택값, 최적값, 규칙에서 사용 중인가)
    feature_rows: list[tuple]
    #: ★ 선택과 최적의 격차가 **노이즈 바닥의 몇 배**인가.
    #: 1 미만이면 그 형상에서는 순위가 측정으로 존재하지 않는다 (§30.2).
    gap_sigma: float = float("nan")


@dataclass
class DiagnosticReport:
    run_id: str
    hw_block: str
    rule_code: str
    rule_weights: list[float]
    overall: dict
    regimes: list[Regime]
    table_facts: TableFacts | None
    cases: list[Case]
    failures: list[dict] = field(default_factory=list)
    hypotheses_applied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        return _render(self)

    def token_estimate(self) -> int:
        """대략적 토큰 수. §12.2 의 예산은 ~5,500 이다."""
        return len(self.render()) // 3


# ---------------------------------------------------------------------------
# 블록 1 — 하드웨어 사실 (§12.1). **기억에서 꺼내게 하면 틀린다**
# ---------------------------------------------------------------------------
def hardware_block(hw, noise) -> str:
    return textwrap.dedent(f"""\
        GPU: {hw.name} ({hw.arch})
          {hw.sm_count} SMs / smem {hw.smem_per_block}B per block
          up to {hw.max_threads_per_sm} threads / {hw.regs_per_sm} registers per SM
          L2 {hw.l2_bytes // (1 << 20)}MB
          effective {hw.peak_tflops_f16} TFLOP/s / {hw.bandwidth_gbps} GB/s
          ridge point {hw.ridge_point:.1f} FLOP/byte

        Execution model:
          CTAs are distributed across SMs; on the last wave some SMs idle.
          A tile computes everything it covers, even outside the shape — a
            128-row tile on M=1 wastes 99.2% of the work.
          split-K divides K to create more tiles, adding a reduction cost.
          serial split-K round-trips D in fp16 per partition (precision loss).
          parallel split-K writes M*N*sk partials to DRAM and reads them back.
          stages=2 (MmaPipelined) and stages>=3 (multistage) are different
            kernel families.
          If alignment does not reach 16 bytes, cp.async is unavailable and
            only 2 stages are possible.

        Limits of measurement:
          Time is only recorded in units of the CUDA event timer's tick
            ({noise.tick_ms * 1000:.3f}us). Smaller differences **cannot be
            distinguished by measurement.**
          The shorter the kernel, the larger that tick is relatively —
            one tick is 7.3% at 14us, 0.08% at 1.3ms.""")


# ---------------------------------------------------------------------------
# 블록 3 — 체제별 분해. ★ 크기가 먼저다
# ---------------------------------------------------------------------------
def _regime_masks(table: PerfTable, matrix: FeatureMatrix,
                  shapes) -> dict[str, np.ndarray]:
    import math

    # ★ 체제는 (형상, 하드웨어)의 성질이지 **피처 목록의 성질이 아니다.**
    #   전에는 `info.log_sol_ms` / `info.is_memory_bound` 를 읽었는데,
    #   그것은 레지스트리에 그 두 피처가 있다는 가정이었다. F0/F1
    #   레지스트리에는 없어서 리포트가 통째로 죽는다 (§30.9).
    #   `regime_of` 가 `physical.py` 의 **함수**를 직접 부르므로 조건과
    #   무관하고, 판정이 한 곳에 모인다 (원칙 2).
    from kernelrule.core.splits import regime_of

    small, mem, waves = [], [], []
    for p in shapes:
        small.append(regime_of(p, table.hw, axis="size") == "short")
        mem.append(regime_of(p, table.hw, axis="roofline") == "mem")
        gm = math.ceil(p.M / 128) * math.ceil(p.N / 128)
        waves.append(gm / table.hw.sm_count)
    small = np.asarray(small)
    mem = np.asarray(mem)
    w = np.asarray(waves, dtype=np.float64)
    k = np.asarray([p.K for p in shapes])
    return {"small": small, "large": ~small, "mem": mem, "comp": ~mem,
            "wlt1": w < 1.0, "w14": (w >= 1.0) & (w < 4.0), "wgt8": w > 8.0,
            "smallk": k <= 1024}


def _regimes(ev: Evaluation, masks: dict, shapes) -> list[Regime]:
    out = []
    r1 = ev.regret[:, 0]
    for name, key in REGIMES:
        m = masks[key]
        if not m.any():
            continue
        i = int(np.argmax(np.where(m, r1, -np.inf)))
        p = shapes[i]
        out.append(Regime(name=name, n_shapes=int(m.sum()),
                          regret=geomean(r1[m]),
                          worst_shape=f"{p.M}x{p.N}x{p.K}",
                          worst_regret=float(r1[i])))
    return sorted(out, key=lambda r: -r.regret)


# ---------------------------------------------------------------------------
# 블록 4 — 사례. **다양성을 강제한다** (§12.1)
# ---------------------------------------------------------------------------
_CASE_AXES = ("tile_m", "tile_n", "tile_k", "split_k", "split_k_mode",
              "ext_warp_m", "ext_warp_n", "ext_stages", "ext_swizzle_n",
              "ext_swizzle_type", "kernel_id")


def _cfg_summary(row: dict) -> str:
    sw = ("id" if row.get("ext_swizzle_type") == "identity" else "hz") + \
        str(row.get("ext_swizzle_n", "?"))
    return (f"tb{row['tile_m']}x{row['tile_n']}x{row['tile_k']} "
            f"w{row.get('ext_warp_m','?')}x{row.get('ext_warp_n','?')} "
            f"st{row.get('ext_stages','?')} sw{sw} "
            f"sk{row['split_k']}{row['split_k_mode'][:3]}")


def _sigma(t_pick: float, t_opt: float, noise) -> float:
    """격차가 **노이즈 바닥의 몇 배**인가.

    한 숫자가 두 문제를 다 해결한다 — 눈금 동점도, 일반적인 해상도도.
    LLM 이 "이 사례를 신경 써야 하나" 를 스스로 판단할 수 있다.

    ⚠️ **사례 선정에는 쓰지 않는다.** 짧은 형상은 σ 가 작게 나오기 쉬운데
    여지가 몰려 있는 곳이 거기다 (§30.5). 표시만 한다.
    """
    denom = t_opt * float(noise.floor(t_opt))
    return (t_pick - t_opt) / denom if denom > 0 else float("inf")


def _make_case(table: PerfTable, matrix: FeatureMatrix, p: Problem,
               order: np.ndarray, regret: float, regime: str,
               kind: str, used_features: frozenset = frozenset(),
               n_feats: int = 8) -> Case:
    df = table.frame_for(p).reset_index(drop=True)
    t = np.asarray(table.times_of(p))
    st = table.stats(p)
    pick = int(order[0])
    # ★ "최적" 은 tie-break 의존이므로 **동점 중 결정론적 대표**를 쓰고,
    #   주변 config 를 함께 보여 뾰족한지 넓은지 알게 한다 (§12.1).
    cand = table.candidates(p)
    best_mask = t <= st.best_ms * (1.0 + 1e-12)
    opt = int(np.flatnonzero(best_mask)[
        np.argmin(cand.tiebreak[best_mask])])

    f, _ = matrix.for_shape(p)
    rows = []
    for name in matrix.feature_names():
        v = getattr(f, name)
        if not approx_equal(float(v[pick]), float(v[opt])):
            rows.append((name, float(v[pick]), float(v[opt])))
    rows.sort(key=lambda r: -abs(r[1] - r[2]) / (abs(r[2]) + 1e-9))
    # ★ "항이 없다" 와 "항은 있는데 가중치가 틀렸다" 는 **다른 수정**이다.
    #   전자는 추가, 후자는 조정. 규칙 소스의 AST 에서 판정한다.
    rows = [(n, a, b, n in used_features) for n, a, b in rows[:n_feats]]

    top5 = np.argsort(t, kind="mergesort")[:5]
    seen, neigh = set(), []
    for i in top5:
        r = df.iloc[int(i)].to_dict()
        cs = _cfg_summary(r)
        if cs in seen:
            continue
        seen.add(cs)
        neigh.append((cs, float(t[i]),
                      _sigma(float(t[i]), st.best_ms, table.noise)))

    return Case(
        shape=(p.M, p.N, p.K), regime=regime, kind=kind, regret=regret,
        difficulty=st.difficulty, best_ms=st.best_ms,
        noise_floor=st.noise_floor, n_answers=st.n_answers,
        n_candidates=st.n_candidates,
        picked={**{k: df.iloc[pick][k] for k in _CASE_AXES if k in df},
                "ms": float(t[pick])},
        optimum={**{k: df.iloc[opt][k] for k in _CASE_AXES if k in df},
                 "ms": float(t[opt])},
        neighbors=neigh, feature_rows=rows,
        gap_sigma=_sigma(float(t[pick]), float(t[opt]), table.noise))


def _select_cases(table, matrix, order_of, ev, masks, shapes,
                  used_features: frozenset = frozenset(),
                  per_regime: int = 2, n_best: int = 2) -> list[Case]:
    """체제마다 최악 n개 + **잘 맞춘 사례** 2개 (§12.1).

    "regret 상위 15개" 는 나쁜 선택이다 — 같은 실패 모드가 15번 반복되면
    정보가 하나뿐이다. 그리고 **실패만 보여주면 잘 되던 것까지 망가뜨린다.**
    """
    r1 = ev.regret[:, 0]
    cases, used = [], set()
    # ★ 같은 (선택, 최적) 쌍이 반복되면 사례가 여러 개여도 정보는 하나다.
    #   §12.1 의 "다양성 강제" 는 체제만이 아니라 **실패 모드**에도 적용된다.
    seen_modes: set[tuple[str, str]] = set()

    def _mode(c: Case) -> tuple[str, str]:
        return (_cfg_summary(c.picked), _cfg_summary(c.optimum))

    for name, key in REGIMES:
        m = masks[key]
        if not m.any():
            continue
        idx = np.argsort(np.where(m, -r1, np.inf))[:per_regime]
        for raw_i in idx:
            i = int(raw_i)
            if not m[i] or i in used or r1[i] <= 1.0 + 1e-9:
                continue
            used.add(i)
            c = _make_case(table, matrix, shapes[i], order_of(shapes[i]),
                           float(r1[i]), name, "worst", used_features)
            sig = _mode(c)          # `m` 은 위에서 마스크다. 섀도잉 금지
            if sig in seen_modes:
                continue      # 같은 실패 모드다. 정보가 늘지 않는다
            seen_modes.add(sig)
            cases.append(c)
    n_added = 0
    for raw_i in np.argsort(r1):
        if n_added >= n_best:
            break
        i = int(raw_i)
        if i in used:
            continue
        c = _make_case(table, matrix, shapes[i], order_of(shapes[i]),
                       float(r1[i]), "잘 맞춘 사례", "best", used_features)
        sig = _mode(c)
        if sig in seen_modes:
            continue
        used.add(i)
        seen_modes.add(sig)
        cases.append(c)
        n_added += 1
    return cases


# ---------------------------------------------------------------------------
# 조립
# ---------------------------------------------------------------------------
def build_report(*, run_id: str, table: PerfTable, matrix: FeatureMatrix,
                 score_fn, weights, code: str, train: Split,
                 table_facts: TableFacts | None = None,
                 failures: list[dict] | None = None,
                 hypotheses_applied: list[str] | None = None,
                 notes: list[str] | None = None) -> DiagnosticReport:
    """★ `train` 은 **학습 분할만** 받는다 (§10.2, §12.3).

    검증/최종 분할이 리포트에 들어가는 경로를 만들지 않는다. 프롬프트에
    홀드아웃 점수를 넣을 수 있으면 결국 거기에 맞춰 튜닝하게 된다.
    """
    if not isinstance(train, Split) or train.role != "train":
        raise SplitError(
            "진단 리포트는 학습 분할만 받는다 (§10.2). 홀드아웃 점수가 "
            "프롬프트에 들어가는 경로를 만들지 않는다.")
    # ★ 전에는 여기가 자유 문자열 리스트였고 위 검사를 **완전히 우회했다**
    #   — 첫 실제 실행의 블록 3.5 가 전수 표에서 계산됐다. §12.3 은 "점수"
    #   만 막았고 집계가 빠져나갔다 (D-28).
    if table_facts is not None and not isinstance(table_facts, TableFacts):
        raise SplitError(
            "table_facts 는 TableFacts.compute(table, train) 로만 만든다 "
            "(§12.3). 자유 문자열을 받으면 학습 분할 검사를 우회한다 — "
            "집계도 홀드아웃을 넘지 않는다.")

    shapes = list(train.shapes)
    # ★ 규칙 소스에서 `f.<이름>` 을 AST 로 뽑는다 (A-1 의 검사기 재사용).
    #   "항이 없다" 와 "가중치가 틀렸다" 를 LLM 이 유추하지 않아도 되게 한다.
    from kernelrule.rules.checks import check_rule

    used = frozenset(check_rule(
        code, feature_names=matrix.feature_names(),
        shape_value_names=matrix.shape_value_names(),
        n_weights=len(weights)).features_used)

    from kernelrule.core.weights import make_score_of

    so = make_score_of(score_fn, matrix, weights)
    ev = evaluate_scores(so, table, shapes, ks=(1, 3, 5, 10), label=run_id)
    masks = _regime_masks(table, matrix, shapes)

    def order_of(p):
        cand = table.candidates(p)
        return cand.top_k(so(p, cand), 5)

    overall = {f"regret@{k}": ev.at(k) for k in ev.ks}
    overall.update({f"hit@{k}": ev.hit_rate(k) for k in ev.ks})
    overall.update(ev.stratified(1))
    overall["size_gap@1"] = ev.size_gap(1)
    overall["difficulty_gap@1"] = ev.difficulty_gap(1)
    # ★ 아카이브 축(roofline)의 집계 (D-145). 요약이 이것부터 보여준다.
    import numpy as _np
    _m = _np.asarray(masks["mem"], dtype=bool)
    if _m.any() and not _m.all():
        overall["mem"] = ev.at(1, mask=_m)
        overall["comp"] = ev.at(1, mask=~_m)
        overall["n_mem"] = int(_m.sum())

    return DiagnosticReport(
        run_id=run_id,
        hw_block=hardware_block(table.hw, table.noise),
        rule_code=code.strip(), rule_weights=[float(x) for x in weights],
        overall=overall, regimes=_regimes(ev, masks, shapes),
        table_facts=table_facts,
        cases=_select_cases(table, matrix, order_of, ev, masks, shapes,
                            used_features=used),
        failures=list(failures or []),
        hypotheses_applied=list(hypotheses_applied or []),
        notes=list(notes or []))


def _render(r: DiagnosticReport) -> str:
    L: list[str] = []
    add = L.append
    add(f"# Diagnostic report — {r.run_id}")
    if r.notes:
        add("")
        for n in r.notes:
            add(f"> {n}")
    add("")
    add("## Block 1 — hardware facts")
    add("```")
    add(r.hw_block)
    add("```")

    add("")
    add("## Block 2 — current rule")
    add(f"The weights were fitted by the numerical optimiser: "
        f"{np.round(r.rule_weights, 3).tolist()}")
    add("```python")
    add(r.rule_code)
    add("```")
    if r.hypotheses_applied:
        add("")
        add("The current rule already reflects these hypotheses:")
        for h in r.hypotheses_applied:
            add(f"  {h}")

    add("")
    add("## Block 3 — regret broken down by regime")
    o = r.overall
    add("```")
    add(f"overall regret@1 {o['regret@1']:.4f}  (@3 {o['regret@3']:.4f}  "
        f"@5 {o['regret@5']:.4f}  @10 {o['regret@10']:.4f})")
    # ★ Look at regret and hit side by side. When they disagree the nature of
    #   the error is different.
    add(f"hit rate hit@1 {o['hit@1']:.3f}  hit@3 {o['hit@3']:.3f}  "
        f"(a hit = within 2 sigma of the noise floor from the optimum)")
    add("  Low regret with hit 0 means **not a near miss but structurally")
    add("  pointing elsewhere** — that needs a term, not a weight change.")
    add("")
    # ★ Do not pre-write the conclusion. The numbers in this report may say
    #   the opposite, and then the LLM believes the sentence, not the data.
    #   **Measure it and write that.**
    sg, dg = abs(o["size_gap@1"]), abs(o["difficulty_gap@1"])
    which = ("size" if sg > dg else "difficulty")
    ratio = (max(sg, dg) / max(min(sg, dg), 1e-9))
    add(f"stratification — on this split, **{which} separates more** "
        f"({max(sg,dg):.4f} vs {min(sg,dg):.4f})")
    # ★ Show the archive axis (roofline) **first** (D-145).
    if "mem" in o and "comp" in o:
        add(f"  ★ memory-bound   {o['mem']:.4f}   "
            f"({int(o.get('n_mem', 0))} shapes)")
        add(f"  ★ compute-bound  {o['comp']:.4f}   "
            f"({int(o['n_shapes']) - int(o.get('n_mem', 0))} shapes)   "
            f"gap {o['comp'] - o['mem']:+.4f}")
    add(f"  t_sol >= 0.5ms   {o['large(>=0.5ms)']:.4f}   "
        f"({int(o['n_shapes']) - int(o['n_small'])} shapes)")
    add(f"  t_sol <  0.5ms   {o['small(<0.5ms)']:.4f}   "
        f"({int(o['n_small'])} shapes)   gap {o['size_gap@1']:+.4f}")
    add(f"  difficulty hi/lo {o['hard']:.4f} / {o['easy']:.4f}   "
        f"gap {o['difficulty_gap@1']:+.4f}")
    if ratio < 2.0:
        add("  (the two axes separate similarly. It may be the shape mix "
            "of this split)")
    add("")
    add(f"{'regime':26s} {'shapes':>6} {'regret':>8}   worst shape")
    for g in r.regimes:
        add(f"{g.name:26s} {g.n_shapes:6d} {g.regret:8.4f}   "
            f"{g.worst_shape} ({g.worst_regret:.3f})")
    add("```")

    if r.table_facts is not None:
        add("")
        add("## Block 3.5 — table structure observations")
        add("Patterns that a single case never shows. "
            "**Computed on the training split only** (§12.3).")
        add("```")
        for f in r.table_facts.lines:
            add(f)
        add("```")

    add("")
    add("## Block 4 — cases")
    add("**Picked vs optimal, side by side.** The neighbouring configs tell")
    add("you whether the optimum is sharp or wide — wide does not mean you")
    add("must hit it exactly.")
    for i, c in enumerate(r.cases, 1):
        add("")
        add(f"### Case #{i}  {c.shape[0]}x{c.shape[1]}x{c.shape[2]}  "
            f"[{c.regime}] {'★ good match' if c.kind == 'best' else ''}")
        add("```")
        add(f"rule picked: {_cfg_summary(c.picked):46s} -> "
            f"{c.picked['ms']*1000:9.2f}us  (regret {c.regret:.3f})")
        add(f"actual best: {_cfg_summary(c.optimum):46s} -> "
            f"{c.optimum['ms']*1000:9.2f}us")
        add("")
        add(f"gap = **{c.gap_sigma:.1f}x** the noise floor"
            + ("   <- inside the noise. This shape has no ordering"
               if c.gap_sigma < 1.0 else ""))
        add("")
        if c.feature_rows:
            add(f"{'feature (largest diff first)':28s} {'picked':>12} "
                f"{'best':>12}  in rule")
            for name, a, b, in_rule in c.feature_rows:
                add(f"{name:28s} {a:12.4f} {b:12.4f}  "
                    + ("in use" if in_rule else "★ unused"))
            add("")
        add("top 5 measured for this shape (multiples of the noise floor "
            "from the optimum):")
        for cs, ms, sg in c.neighbors:
            tag = "(optimal)" if sg <= 1e-9 else f"+{sg:.1f} sigma"
            add(f"  {ms*1000:9.2f}us  {tag:>12s}  {cs}")
        add("")
        add(f"difficulty {c.difficulty:.2f}   noise floor "
            f"{c.noise_floor*100:.3f}%   "
            f"indistinguishable answers {c.n_answers}/{c.n_candidates}")
        add("```")

    if r.failures:
        add("")
        add("## Block 5 — failure history")
        add("**Do not repeat the same idea.**")
        add("```")
        for f in r.failures:
            add(f"r{f.get('round','?'):<4} {f.get('verdict','?'):12s} "
                f"{f.get('regret_before','?')} -> {f.get('regret_after','?')}"
                f"   {f.get('idea','')}")
        add("```")
    return "\n".join(L)
