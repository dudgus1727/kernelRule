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

    금지  "X 가 Y 보다 크게 갈린다", "A 가 지배적이다" 를 **템플릿에 박는 것**
    허용  계산 결과를 서술로 렌더링하는 것 (어느 쪽이 큰지 재서 문장을 만든다)

이 규칙이 왜 필요한지는 실제로 밟았기 때문이다. 블록 3 에 "크기 층화가
난이도 층화보다 크게 갈린다" 를 미리 적어 뒀는데, **학습 분할(M<=2048)의
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
from kernelrule.core.scoring import Evaluation, evaluate_scores, geomean
from kernelrule.core.splits import Split, SplitError
from kernelrule.core.table import PerfTable
from kernelrule.core.types import Problem
from kernelrule.report.table_facts import TableFacts

__all__ = ["DiagnosticReport", "Case", "Regime", "build_report"]

#: 체제 정의. `(이름, 형상 술어)`. 크기가 먼저다 (§30.5).
REGIMES: tuple[tuple[str, str], ...] = (
    ("t_sol < 0.5ms (짧음)", "small"),
    ("t_sol >= 0.5ms (김)", "large"),
    ("memory-bound", "mem"),
    ("compute-bound", "comp"),
    ("waves < 1", "wlt1"),
    ("waves 1~4", "w14"),
    ("waves > 8", "wgt8"),
    ("K <= 1024 (짧은 mainloop)", "smallk"),
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
          SM {hw.sm_count}개 / 블록당 smem {hw.smem_per_block}B
          SM당 최대 {hw.max_threads_per_sm} 스레드 / 레지스터 {hw.regs_per_sm}
          L2 {hw.l2_bytes // (1 << 20)}MB
          실효 {hw.peak_tflops_f16} TFLOP/s / {hw.bandwidth_gbps} GB/s
          ridge point {hw.ridge_point:.1f} FLOP/byte

        실행 모델:
          CTA가 SM에 배분되며 마지막 wave에서 SM 일부가 유휴.
          타일은 형상 경계를 넘어도 그 부분을 전부 계산한다 — M=1에 128행
            타일이면 일의 99.2%가 버려진다.
          split-K는 K를 나눠 타일 수를 늘리되 리덕션 비용이 추가된다.
          serial split-K는 파티션마다 fp16으로 D를 왕복한다 (정밀도 손실).
          parallel split-K는 부분합 M*N*sk개를 DRAM에 쓰고 다시 읽는다.
          stages=2(MmaPipelined)와 stages>=3(multistage)은 다른 커널 계열이다.
          alignment가 16바이트를 못 맞추면 cp.async를 못 써서 2단만 가능하다.

        측정의 한계:
          시간은 CUDA 이벤트 타이머의 눈금({noise.tick_ms * 1000:.3f}us) 단위로만
            기록된다. 그보다 작은 차이는 **측정으로 구분할 수 없다.**
          짧은 커널일수록 그 눈금이 상대적으로 크다 —
            14us에서 한 눈금이 7.3%, 1.3ms에서 0.08%.""")


# ---------------------------------------------------------------------------
# 블록 3 — 체제별 분해. ★ 크기가 먼저다
# ---------------------------------------------------------------------------
def _regime_masks(table: PerfTable, matrix: FeatureMatrix,
                  shapes) -> dict[str, np.ndarray]:
    import math

    small, mem, waves = [], [], []
    for p in shapes:
        _, info = matrix.for_shape(p)
        small.append(info.log_sol_ms < math.log2(0.5))
        mem.append(bool(info.is_memory_bound))
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
        if abs(float(v[pick]) - float(v[opt])) > 1e-9:
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
    add(f"# 진단 리포트 — {r.run_id}")
    if r.notes:
        add("")
        for n in r.notes:
            add(f"> {n}")
    add("")
    add("## 블록 1 — 하드웨어 사실")
    add("```")
    add(r.hw_block)
    add("```")

    add("")
    add("## 블록 2 — 현재 규칙")
    add(f"가중치는 수치 최적화기가 맞춘 값이다: "
        f"{np.round(r.rule_weights, 3).tolist()}")
    add("```python")
    add(r.rule_code)
    add("```")
    if r.hypotheses_applied:
        add("")
        add("현재 규칙에는 다음 가설들이 반영되어 있다:")
        for h in r.hypotheses_applied:
            add(f"  {h}")

    add("")
    add("## 블록 3 — 체제별 regret 분해")
    o = r.overall
    add("```")
    add(f"전체 regret@1 {o['regret@1']:.4f}  (@3 {o['regret@3']:.4f}  "
        f"@5 {o['regret@5']:.4f}  @10 {o['regret@10']:.4f})")
    # ★ regret 과 hit 을 나란히 본다. 둘이 갈리면 오류의 성격이 다르다.
    add(f"정답 적중 hit@1 {o['hit@1']:.3f}  hit@3 {o['hit@3']:.3f}  "
        f"(정답 = 최적 대비 노이즈 바닥 2시그마 이내)")
    add("  regret 은 낮은데 hit 이 0 이면 **아깝게 빗나가는 것이 아니라")
    add("  구조적으로 다른 곳을 짚는 것**이다 — 가중치 조정이 아니라 항이 필요하다.")
    add("")
    # ★ 결론을 미리 적어 두면 안 된다. 이 리포트의 숫자가 반대일 수 있고,
    #   그러면 LLM 이 데이터가 아니라 문장을 믿는다. **재서 쓴다.**
    sg, dg = abs(o["size_gap@1"]), abs(o["difficulty_gap@1"])
    which = ("크기" if sg > dg else "난이도")
    ratio = (max(sg, dg) / max(min(sg, dg), 1e-9))
    add(f"층화 — 이 분할에서는 **{which} 층화가 더 크게 갈린다** "
        f"({max(sg,dg):.4f} vs {min(sg,dg):.4f})")
    add(f"  t_sol >= 0.5ms   {o['large(>=0.5ms)']:.4f}   "
        f"({int(o['n_shapes']) - int(o['n_small'])}형상)")
    add(f"  t_sol <  0.5ms   {o['small(<0.5ms)']:.4f}   "
        f"({int(o['n_small'])}형상)   격차 {o['size_gap@1']:+.4f}")
    add(f"  난이도 상 / 하    {o['hard']:.4f} / {o['easy']:.4f}   "
        f"격차 {o['difficulty_gap@1']:+.4f}")
    if ratio < 2.0:
        add("  (두 축의 격차가 비슷하다. 이 분할의 형상 구성 때문일 수 있다)")
    add("")
    add(f"{'체제':26s} {'형상':>4} {'regret':>8}   최악 형상")
    for g in r.regimes:
        add(f"{g.name:26s} {g.n_shapes:4d} {g.regret:8.4f}   "
            f"{g.worst_shape} ({g.worst_regret:.3f})")
    add("```")

    if r.table_facts is not None:
        add("")
        add("## 블록 3.5 — 표 구조 관찰")
        add("개별 사례로는 보이지 않는 패턴이다. "
            "**학습 분할에서만 계산했다** (§12.3).")
        add("```")
        for f in r.table_facts.lines:
            add(f)
        add("```")

    add("")
    add("## 블록 4 — 사례")
    add("**선택 vs 최적을 나란히 본다.** 주변 config 는 최적이 뾰족한지")
    add("넓은지 알려준다 — 넓으면 정확히 맞추라는 뜻이 아니다.")
    for i, c in enumerate(r.cases, 1):
        add("")
        add(f"### 사례 #{i}  {c.shape[0]}x{c.shape[1]}x{c.shape[2]}  "
            f"[{c.regime}] {'★ 잘 맞춤' if c.kind == 'best' else ''}")
        add("```")
        add(f"규칙 선택: {_cfg_summary(c.picked):46s} -> {c.picked['ms']*1000:9.2f}us"
            f"  (regret {c.regret:.3f})")
        add(f"실제 최적: {_cfg_summary(c.optimum):46s} -> {c.optimum['ms']*1000:9.2f}us")
        add("")
        add(f"격차 = 노이즈 바닥의 **{c.gap_sigma:.1f}배**"
            + ("   <- 노이즈 안이다. 이 형상에는 순위가 없다"
               if c.gap_sigma < 1.0 else ""))
        add("")
        if c.feature_rows:
            add(f"{'피처(차이 큰 순)':28s} {'선택':>12} {'최적':>12}  규칙에서")
            for name, a, b, in_rule in c.feature_rows:
                add(f"{name:28s} {a:12.4f} {b:12.4f}  "
                    + ("사용 중" if in_rule else "★ 미사용"))
            add("")
        add("같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):")
        for cs, ms, sg in c.neighbors:
            tag = "(최적)" if sg <= 1e-9 else f"+{sg:.1f}시그마"
            add(f"  {ms*1000:9.2f}us  {tag:>12s}  {cs}")
        add("")
        add(f"난이도 {c.difficulty:.2f}   노이즈 바닥 {c.noise_floor*100:.3f}%   "
            f"구분 불가능한 정답 {c.n_answers}/{c.n_candidates}개")
        add("```")

    if r.failures:
        add("")
        add("## 블록 5 — 실패 이력")
        add("**같은 아이디어를 반복하지 마라.**")
        add("```")
        for f in r.failures:
            add(f"r{f.get('round','?'):<4} {f.get('verdict','?'):12s} "
                f"{f.get('regret_before','?')} -> {f.get('regret_after','?')}"
                f"   {f.get('idea','')}")
        add("```")
    return "\n".join(L)
