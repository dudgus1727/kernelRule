"""라운드 루프 (§14) — 코드가 순서를 통제한다.

**에이전트는 서로 대화하지 않는다** (§11.1). 진짜 멀티 에이전트로 만들면
어느 단계에서 망가졌는지 추적이 불가능해진다.

## 한 라운드 (§14.1, 부록 ★수정 1)

    1. 아카이브 최고 규칙 채점 -> 진단 리포트 생성           [코드]
    2. 가설 3~5개                                          [LLM 1회]
    3. 필요시 피처 생성 + 심사 + 자동 검증                   [LLM 0~4회]
    4. 규칙 12개 생성 (부모/가설 조합)                       [LLM 12회]
    5. 정적 검사 -> 샌드박스 -> ★ 가중치 최적화 -> 채점       [코드]
    6. 아카이브 갱신
    7. 가설 이력 / 실패 목록 기록
    8. 종료 조건 확인

**5번의 가중치 최적화가 채점 앞에 오는 것이 핵심이다** (§29.3). 안 하면
좋은 구조가 나쁜 초기값 때문에 버려지고, 진화가 구조가 아니라 **가중치
운**을 선택한다.

## 종료 조건 (§14.3)

    10라운드 연속 개선이 노이즈 바닥 이하
      AND
    아카이브에 새 셀이 채워지지 않음

점수가 멈춰도 다양성이 늘고 있으면 아직 탐색 중이다. **두 조건을 모두** 쓴다.
조기 종료는 **검증 분할**로 판정한다 — 학습 regret 으로 하면 이미 과적합이
시작된 뒤에도 계속 돈다 (§10.2).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kernelrule.agents.schemas import SchemaViolation, validate_rule_proposal
from kernelrule.core.archive import Archive, Elite
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import SandboxError, compile_rule, run_isolated
from kernelrule.core.scoring import evaluate_scores, is_significant
from kernelrule.core.splits import SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import FitError, fit_weights, make_score_of
from kernelrule.report.diagnostic import build_report
from kernelrule.rules.checks import PARAMETERS as _PARAMETERS
from kernelrule.rules.checks import check_rule, limits_for, weight_bounds

__all__ = ["RoundLoop", "RoundResult", "LoopConfig", "LLMUnreachable"]


@dataclass
class LoopConfig:
    run_id: str
    n_rules_per_round: int = 12
    max_rounds: int = 20
    max_evals: int = 200
    #: ★ 실행 트레이스 (D-133). `trace.jsonl` 에 사건을 시간순으로 남긴다.
    #: **조건이 아니다** — 로깅만 하고 계산 경로를 안 건드린다. 그래서
    #: `runset.KEYS` 에 넣지 않는다 (`tests/test_trace.py` 가 확인한다).
    trace: bool = True
    #: ★ 조기 종료 **끔** (D-132). `0` 이면 라운드 상한까지 다 돈다.
    #:
    #: 왜 껐나 — 문제가 값이 아니라 **판단 기준**이었다:
    #: ```
    #: 루프 내부 점수(best_val_regret)가 평평해도
    #: ★ 최종 채점(체제별 재적합 -> 홀드아웃)은 계속 좋아진다
    #: -> 내부 점수로 멈추면 원리적으로 최종 성능을 잃는다 (D-131: +0.0187)
    #: patience 를 5 로 올려도 크기만 줄고 같은 문제다
    #: ```
    #: 0 보다 크면 옛 규칙이 그대로 돈다 — 옛 실행 재현용으로 남긴다.
    patience: int = 0
    seed: int = 0
    sandbox_first_seen: bool = True
    out_dir: str = "runs"
    #: ★ Analyst -> FeatureWriter 경로 (D-75). 라운드당 만들 수 있는 새 축의
    #: 상한. **0 이면 경로가 없다** — 2026-08-28 이전의 동작이고 기본값이다.
    #:
    #: 상한이 필요한 이유는 §21 이다: 피처 행렬이 새 축마다 전 형상을 다시
    #: 계산하고, 캐시 키가 레지스트리 해시라 라운드마다 바뀌면 캐시가 안
    #: 듣는다. 1~2 를 넘기지 마라.
    max_new_features_per_round: int = 0
    #: FeatureWriter 조건 (F1/F2/F3). 루프 밖 1단계와 **같은 조건**을
    #: 줘야 한다 — 다르면 라운드 안에서 조건이 바뀐다.
    feature_condition: str = "F3"
    #: ★ §16.1 대조군 C (D-91) — Analyst 를 끄되 **다른 실행·다른 라운드의
    #: 가설**을 그 자리에 넣는다. `hypotheses.jsonl` 경로들.
    #:
    #: 무엇을 가르나:
    #:   B 에 가까우면  가설이 **현재 상태에 맞을 필요는 없다**
    #:                  = 진단 리포트의 기여가 다양성 주입이다
    #:   A 에 가까우면  ★ 이 라운드의 진단에 맞아야 한다 (§16.1 의 강한 형태)
    #:
    #: Analyst 를 안 부르므로 **A 와 호출 수가 같다** — 비용 비교가 깨끗하다.
    hypothesis_pool: tuple[str, ...] = ()
    #: ★ 가중치 목적함수 (D-101). 기본 `"regret"` — **지금까지의 동작**이다.
    #: `"rank"` 면 참 상위 `rank_top_k` 안의 가중 쌍 손실로 맞추고,
    #: **아카이브 채택도 그것으로 한다** (셀 축은 안 바뀐다).
    #: `regret` 은 그 경우에도 계속 계산해서 기록한다 (실험 계획서 §4 —
    #: 판정에는 안 쓰고 두 진화를 나란히 놓을 때 쓴다).
    #: ★ 2026-09-04 (D-128): 기본이 다시 `"regret"` 이고, **`"rank"` 는
    #: 거부한다** — 순위 손실은 틀린 목적함수로 결론났다 (D-118·D-121).
    #: 필드 자체는 남긴다: 옛 `config.json` 을 읽어야 한다.
    #: 순위 손실·tau 는 **지표로는** 그대로 쓴다 (`weights.rank_loss` 등).
    objective: str = "regret"
    rank_top_k: int = 100
    #: ★ 두 손실의 합 (D-109). `L = rank_loss(k) + lambda * rank_loss_top1(k)`
    #: `rank_loss_top1` 은 **참 1등이 낀 쌍만** — `regret` 의 부드러운
    #: 대리다. `regret` 을 직접 넣지 않는다 (계단이라 L-BFGS 를 못 쓴다).
    rank_lambda: float = 0.0
    #: ★ 적합기 (D-123). 기본 `"nelder-mead"` — **지금까지의 모든 실행**이다.
    #: `"cma"` 는 §2 통과 조건이 고른 팔이다 (16차원 도달률 100%, `cma` 패키지 필요).
    #: ★ 조건이므로 `config.json` 에 남기고 묶음 검사가 본다 (원칙 39).
    fit_method: str = "nelder-mead"
    #: 적합 재시작 수. CMA-ES 는 1 이다 — 예산을 넷으로 쪼개면 세대가
    #: 여섯 번뿐이라 CMA 가 아니다 (`fitter-regret-prereg.md` §2).
    fit_restarts: int = 4
    #: ★ 두 순서 실험 (D-104). `"rank->regret"` 또는 `"regret->rank"`.
    #: `None` 이면 목적함수가 안 바뀐다 — 지금까지의 동작이다.
    #:
    #: **전환 시점은 하이퍼파라미터가 아니라 중단 조건이다** — 직전
    #: `switch_window` 라운드의 개선이 `switch_min_improve` 미만이면
    #: 바꾼다. 근거: s2 가 r5~r9 에서 0.2945 -> 0.2915 (1.0%) 로 평평했다.
    #: ★ 파라미터 상한 (D-104). `None` 이면 `checks.PARAMETERS`(8) — 지금까지의 동작.
    #: 검사기·프롬프트가 **같은 값**을 봐야 한다 (원칙 2).
    parameters: int | None = None
    objective_switch: str | None = None
    switch_min_improve: float = 0.01
    switch_window: int = 3
    #: ★ 채점·적합 병렬화 (D-95). 0 이면 순차 — **지금까지의 동작**이다.
    #: 결과는 같아야 한다 (`test_parallel_matches_sequential`).
    n_workers: int = 0
    #: ★ §16.1 ablation — Analyst 를 끄면 진단 리포트도 가설도 없다.
    #: RuleEditor 는 부모 규칙과 피처 목록만 보고 고친다.
    #: **기본은 켬**이다 (지금까지의 모든 실행이 그렇다).
    use_analyst: bool = True


class LLMUnreachable(RuntimeError):
    """LLM 에 닿지 못했다. **모델의 실패가 아니라 우리 문제다** (D-43)."""


#: LLM 에 닿지 못한 것을 알아보는 이름들. 크레딧·인증·네트워크 문제이지
#: 모델의 실패가 아니다 (D-43).
_TRANSPORT_HINTS = ("HTTPError", "APIError", "APIConnection", "APIStatus",
                    "Timeout", "RateLimit", "Authentication", "Permission",
                    "ConnectError", "ReadError", "ServiceUnavailable")
#: 본문에 이것이 있으면 확실하다.
_TRANSPORT_BODY = ("no credits", "insufficient_quota", "invalid_api_key",
                   "401", "402", "403", "429", "500", "502", "503")


def _is_transport_error(exc: BaseException) -> bool:
    """LLM 에 **닿지 못한** 것인가, 모델이 스키마를 못 맞춘 것인가."""
    name = type(exc).__name__
    if any(h in name for h in _TRANSPORT_HINTS):
        return True
    text = str(exc).lower()
    return any(h in text for h in _TRANSPORT_BODY)


#: 한 라운드의 제안이 **전부** 전송 실패면 멈춘다. 크레딧이나 인증 문제는
#: 저절로 낫지 않고, 남은 라운드를 태워도 빈 아카이브만 남는다.
#: 실제로 12라운드 x 48초를 그렇게 썼다.
STOP_ON_TOTAL_LLM_FAILURE = True


#: 검증 격차가 이보다 크면 **체제 전이 실패**로 본다.
#: 과적합과 다르다 — 항이 3개뿐인 규칙도 이 값을 넘는다 (실측 +4.99).
VAL_GAP_ALARM = 0.5


#: ★ 워커가 보는 것 (D-95). `fork` 로 만들어진 자식이 **부모의 메모리를
#: 그대로 물려받는다** — 표 + 피처 행렬이 4.2GB 라 워커마다 다시 로드하면
#: 12개에 50GB 다. 복사-후-쓰기라 실제로는 거의 안 는다.
#:
#: ⚠️ 풀을 만들기 **전에** 채워야 한다. fork 시점의 스냅숏이 전부다.
_WORKER: dict = {}


def _fit_and_score(job: tuple) -> dict:
    """★ 한 후보를 적합하고 채점한다. **워커에서 돈다** (D-95).

    라운드 벽시계의 96% 가 `fit_weights` 다 (후보당 5.4초 x 12).
    GIL 때문에 스레드로는 안 되고 프로세스여야 한다.

    ⚠️ **정적 검사·샌드박스는 부모가 한다.** 여기서 하면 워커가 또 프로세스를
    띄우고(`run_isolated`), 중첩 spawn 이 된다. 그 둘은 합쳐도 3% 다.

    돌려주는 것은 **순수 자료**다 — `Elite` 의 id 와 순서는 부모가 정한다
    (결정론).
    """
    idx, code, w0 = job
    from kernelrule.core.sandbox import compile_rule
    from kernelrule.core.scoring import evaluate_scores
    from kernelrule.core.weights import fit_weights, make_score_of

    c = _WORKER
    try:
        fn = compile_rule(code)
        fr = fit_weights(fn, c["matrix"], c["table"], c["train"], w0,
                         max_evals=c["max_evals"], val_split=c["val"],
                         objective=c.get("objective", "regret"),
                         rank_top_k=c.get("rank_top_k", 100),
                         rank_lambda=c.get("rank_lambda", 0.0),
                         method=c.get("fit_method", "nelder-mead"),
                         n_restarts=c.get("fit_restarts", 4),
                         bounds=weight_bounds(code, len(w0)))
    except (FitError, SchemaViolation) as e:
        return {"i": idx, "err": ("fit", str(e)[:90])}
    except Exception as e:                                  # noqa: BLE001
        return {"i": idx, "err": ("run", f"{type(e).__name__}: {e}"[:90])}
    ev = evaluate_scores(make_score_of(fn, c["matrix"], fr.w), c["table"],
                         list(c["train"].shapes), ks=(1, 3))
    return {"i": idx, "w": [float(x) for x in fr.w],
            "regret": fr.fit_regret, "moved": bool(fr.moved),
            "rank_loss": _rank_loss_of(fn, c, fr.w),
            "val_regret": fr.val_regret,
            "short": ev.at(1, mask=c["short_mask"]),
            "long": ev.at(1, mask=c["long_mask"])}


def _rank_loss_of(fn, c: dict, w) -> float:
    """★ 채택에 쓸 순위 손실. `objective="regret"` 이면 NaN 을 남긴다.

    `regret` 조건에서 억지로 계산하지 않는다 — 그러면 쌍을 만드는 비용이
    모든 기존 실행에 붙는다.
    """
    import math

    if c.get("objective") != "rank":
        return float("nan")
    from kernelrule.core.weights import _Problem

    k = c.get("rank_top_k", 100)
    lam = float(c.get("rank_lambda", 0.0) or 0.0)
    pr = _Problem(c["matrix"], c["table"], c["train"].shapes, 1)
    pr.build_pairs(c["table"], k)
    v = pr.rank_loss(fn, w)
    # ★ 채택도 **적합과 같은 목적함수**로 해야 한다 (원칙 37). 람다를
    #   빼고 고르면 최적화한 것과 다른 것으로 뽑는다.
    if lam and math.isfinite(v):
        pr.build_top1_pairs(c["table"], k)
        v += lam * pr.rank_loss_top1(fn, w)
    return float(v) if math.isfinite(v) else float("nan")


def _requirement_of(h: dict) -> str:
    """가설이 요구한 물리량 문장. **두 이름을 다 읽는다.**

    필드는 `needs_new_feature` 다. 2026-08-28 에 잠깐 `physical_requirement`
    로 바꿔 실행 3개를 돌렸다가 **되돌렸다** (D-81) — 기준선이 옛 이름으로
    측정됐기 때문이다. 그 사이 실행들이 조용히 0건이 되면 안 되므로 둘 다
    읽는다.
    """
    v = h.get("physical_requirement") or h.get("needs_new_feature")
    return str(v).strip() if v else ""


def _feature_task(text: str) -> str:
    """FeatureWriter 에게 가는 **전부**. ★ 진단 리포트는 안 간다 (D-75).

    사례 번호도 점수도 형상 목록도 없다. 물리 요구 한 문장뿐이다 —
    루프 안에서 만든 피처가 학습 형상에 맞춰지는 통로를 막는다.
    """
    return ("## 이번에 만들 것\n\n"
            "아래 물리량을 재는 피처 하나를 만드세요.\n\n"
            f"> {text}\n\n"
            "이 문장 말고 다른 맥락은 없습니다. 표도 사례도 보지 않고, "
            "**물리에서 유도**하세요.")


@dataclass
class RoundResult:
    round: int
    n_proposed: int = 0
    n_rejected_static: int = 0
    n_rejected_sandbox: int = 0
    n_rejected_schema: int = 0
    #: ★ LLM 에 **닿지 못한** 횟수. 스키마 거부와 섞으면 안 된다 —
    #: 크레딧 소진·인증 실패·네트워크 오류가 "모델이 나쁜 규칙을 냈다" 로
    #: 보인다 (D-43). 실제로 429 를 12라운드 동안 "스키마 거부 144건" 으로
    #: 읽었다.
    n_llm_error: int = 0
    n_rejected_fit: int = 0
    #: ★ 가중치 적합기가 실제로 움직인 후보 수 (D-54). 낮으면 그 라운드는
    #: **초기값으로 채점된 것**이고, 진화가 구조가 아니라 가중치 운을
    #: 고르게 된다 (§29.3). 화면에 뜨면 첫 라운드에서 눈치챈다.
    n_fit_moved: int = 0
    n_scored: int = 0
    n_accepted: int = 0
    best_regret: float = float("nan")
    best_val_regret: float = float("nan")
    #: ★ 채택 기준이 `rank` 일 때 아카이브 최고의 순위 손실 (D-101).
    #: `regret` 조건에서는 NaN 이다 — 억지로 계산하지 않는다.
    best_rank_loss: float = float("nan")
    n_cells: int = 0
    val_gap: float = float("nan")
    n_val_blowups: int = 0
    #: ★ 부모 종류별 (exploit / explore / cross) 제안·중복·채점 수 (D-94).
    #: `{"exploit": {"n": 6, "dup": 1, "scored": 5}, ...}`
    #:
    #: 왜 여기 남기나: 부모 종류가 **프롬프트 문자열에만** 있어서
    #: `llm_calls` 의 `prompt` 가 빈 실행에서는 못 읽었다 — "가설-부모
    #: 불일치" 를 옛 자료로 재려다 실패했다.
    by_parent_kind: dict = field(default_factory=dict)
    #: ★ 이 라운드에 Analyst 가 요구한 새 축 / 실제로 만들어진 축 (D-75).
    n_feature_requests: int = 0
    n_features_made: int = 0
    #: 라운드당 상한에 걸려 **만들지 않은** 요구. 조용히 버리지 않는다.
    n_feature_over_cap: int = 0
    seconds: float = 0.0
    llm_calls: dict = field(default_factory=dict)
    rejections: list[tuple] = field(default_factory=list)

    def line(self) -> str:
        err = f"★LLM오류 {self.n_llm_error} " if self.n_llm_error else ""
        mv = f"적합이동 {self.n_fit_moved}/{self.n_scored} | " if self.n_scored else ""
        gap = f"{self.val_gap:+.3f}"
        alarm = "!" if self.val_gap > VAL_GAP_ALARM else " "
        over = f"(상한초과 {self.n_feature_over_cap}) " \
            if self.n_feature_over_cap else ""
        feat = (f"새축 {self.n_features_made}/{self.n_feature_requests} "
                f"{over}| " if self.n_feature_requests else "")
        return (
            f"r{self.round:<3d} 제안 {self.n_proposed:2d} | {err}{feat}"
            f"거부 스키마 {self.n_rejected_schema} 정적 "
            f"{self.n_rejected_static} 샌드박스 {self.n_rejected_sandbox} "
            f"적합 {self.n_rejected_fit} | 채점 {self.n_scored:2d} "
            f"채택 {self.n_accepted:2d} | {mv}best {self.best_regret:.4f} "
            + (f"rank {self.best_rank_loss:.4f} "
               if self.best_rank_loss == self.best_rank_loss else "")
            + f"val {self.best_val_regret:.4f}({gap}{alarm})"
            f"| 셀 {self.n_cells:2d} 폭발 {self.n_val_blowups} | "
            f"{self.seconds:.1f}s")


def _git_commit() -> str:
    """지금 커밋. 트레이스 첫 줄이 자족하려면 코드 판이 있어야 한다."""
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=False,
                              timeout=5).stdout.strip() or "?"
    except Exception:                                     # noqa: BLE001
        return "?"


def _sha(code: str) -> str:
    """코드의 짧은 해시. 트레이스에서 같은 제안을 잇는 데 쓴다 (D-133)."""
    import hashlib

    return hashlib.sha256(code.encode()).hexdigest()[:12]


class RoundLoop:
    def __init__(self, *, cfg: LoopConfig, table: PerfTable,
                 matrix: FeatureMatrix, splits: SplitSet, llm) -> None:
        # ★ 진화 경로에서 순위 손실을 뺀다 (D-128). **맨 먼저** 본다 —
        #   뒤에서 걸리면 표를 다 읽은 뒤에 죽는다 (원칙 1).
        if cfg.objective != "regret" or cfg.objective_switch is not None:
            raise ValueError(
                f"진화 목적함수는 regret 뿐이다 (objective={cfg.objective!r}, "
                f"objective_switch={cfg.objective_switch!r}). 순위 손실은 "
                "틀린 목적함수로 결론났다 (D-118·D-121) — 지표로만 쓴다. "
                "옛 실행을 재현하려면 그 커밋으로 돌아가라 (D-128).")
        self.cfg = cfg
        # ★ 트레이스 (D-133). `dump()` 와 같은 곳에 쌓는다.
        from kernelrule.core.trace import Tracer
        self.trace = Tracer(
            Path(cfg.out_dir) / cfg.run_id / "trace.jsonl"
            if cfg.trace else None)
        self.table = table
        self.matrix = matrix
        self.splits = splits
        self.llm = llm
        # ★ 주입받지 않고 **여기서 학습 분할로부터 계산한다** (§12.3 / D-28).
        #   호출자가 전수 표에서 계산한 문장을 넘길 수 있으면 리포트의 분할
        #   검사가 아무 일도 하지 않는다 — 첫 실제 실행이 그렇게 오염됐다.
        from kernelrule.report.table_facts import TableFacts
        self.table_facts = TableFacts.compute(table, splits.train)
        self.rng = np.random.default_rng(cfg.seed)
        self.archive = Archive(
            noise_tol=0.0,
            select_by=("rank" if cfg.objective == "rank" else "regret"),
            # ★ 칸도 목적함수를 따라간다 (D-101). 절대 경계는 regret
            #   규모에 맞춰 정한 값이라 순위 손실에서는 전부 첫 칸에
            #   몰린다. 순위 4분위는 경계값이 필요 없다.
            cell_mode=("quantile" if cfg.objective == "rank"
                       else "absolute"))
        self.rounds: list[RoundResult] = []
        self.failures: list[dict] = []
        self.hypotheses: list[dict] = []
        #: ★ 라운드 안에서 만든 피처들 (D-75). 산출물에 그대로 남긴다.
        self.features_made: list[dict] = []
        #: ★ cross 자식이 두 부모의 항을 섞었는가 (D-96 관찰 1).
        #: 부모 코드는 남지 않으므로 그 자리에서 계산해 둔다.
        self.cross_lineage: list[dict] = []
        #: ★ 라운드별 아카이브 최고 (코드째). D-101 관찰용.
        self.bests: list[dict] = []
        #: ★ 목적함수 전환 상태 (D-104). `switch_round` 는 안 바뀌면 -1.
        #: ★ 예산을 검사기에 흘린다. `None` 이면 기본(8) 그대로다.
        self._budget = (cfg.parameters if cfg.parameters is not None
                        else _PARAMETERS)
        # ★ 예산만 올리면 **AST 노드 상한에서 막힌다** (D-106). 상한들을
        #   한 곳에서 같이 움직인다.
        self._limits = (limits_for(cfg.parameters)
                        if cfg.parameters is not None else None)
        self._objective = cfg.objective
        self._switched = False
        self.switch_round = -1
        #: 가설 id 의 **유일한 출처**. 모델이 붙인 id 는 응답 안에서만
        #: 유일해서 라운드/패스를 넘으면 겹친다.
        self._hyp_seq = 0
        #: 빌려온 가설 묶음 (대조군 C). 처음 쓸 때 한 번만 읽는다.
        self._pool: list[list[dict]] | None = None
        #: 병렬 채점용 프로세스 풀 (D-95). 라운드마다 새로 만들지 않는다.
        self._pool_exec = None
        self._rule_seq = 0
        self._seen_code: dict[str, float] = {}      # 캐시 (§15.4)
        self._feats = matrix.feature_names()
        #: ★ 지수 자리 가드용 (D-112). 축이 늘면 같이 갱신한다.
        self._fmins = matrix.feature_mins()
        self._shape_vals = matrix.shape_value_names()
        self._short_mask, self._long_mask = self._regime_masks()
        # ★ 풀은 **생성자에서** 만든다 (D-95). `fork` 는 스레드가 도는 중이면
        #   위험하고(CPython 이 DeprecationWarning 을 낸다) asyncio LLM 호출이
        #   스레드를 만든다 — 생성자가 이 객체가 통제할 수 있는 가장 이른
        #   시점이다. 마스크가 `_WORKER` 에 들어가므로 그 뒤여야 한다.
        #   ⚠️ 완전하지는 않다 — 같은 프로세스에서 앞 단계가 이미 LLM 을
        #      불렀으면 스레드가 있을 수 있다. 그때는 `n_workers=0` 으로
        #      떨어뜨려라 (기본값이다).
        self._pool_or_none()

    # -- ★ 대조군 C — 남의 가설을 빌려 온다 (§16.1, D-91) ------------------
    def _pool_round(self, r: int) -> list[dict]:
        """다른 실행의 **한 라운드 전체**를 통째로 빌린다.

        가설 하나씩 섞지 않고 (실행, 라운드) 단위로 가져오는 이유는 두
        가지다.

        ```
        1  Analyst 한 번의 출력은 서로를 보완하는 집합이다.
           낱개로 섞으면 "맞지 않는 가설" 이 아니라 "앞뒤가 안 맞는 묶음" 이
           되어 다른 것을 재게 된다
        2  라운드당 개수 분포가 B 와 자동으로 같아진다 (평균 4.4개)
        ```

        ⚠️ **같은 시드 번호의 실행은 뺀다** — `abl-B-s1` 의 가설을
        `abl-C-s1` 에 주면 "다른 실행" 이 아니다.
        """
        if self._pool is None:
            import json as _json
            groups: dict[tuple, list[dict]] = {}
            mine = self.cfg.run_id.rsplit("-s", 1)[-1]
            for path in self.cfg.hypothesis_pool:
                pp = Path(path)
                src = pp.parent.name
                if src.rsplit("-s", 1)[-1] == mine:
                    continue                    # 같은 시드 번호는 뺀다
                for ln in pp.read_text().splitlines():
                    if not ln.strip():
                        continue
                    h = _json.loads(ln)
                    if h.get("analyst_pass", 1) != 1:
                        continue
                    groups.setdefault((src, h.get("round")), []).append(h)
            self._pool = [groups[k] for k in sorted(groups)]
            if not self._pool:
                raise ValueError(
                    "가설 풀이 비었다. 같은 시드 번호만 줬거나 경로가 "
                    "틀렸다 — 조용히 가설 없이 돌지 않는다 (§26.4).")
            self.rng.shuffle(self._pool)
        block = self._pool[r % len(self._pool)]
        out = []
        for h in block:
            g = dict(h)
            # ★ 출처를 남긴다. 나중에 "이 가설이 어디서 왔나" 를 못 물으면
            #   이 팔의 결과를 해석할 수 없다.
            g["borrowed_from"] = f"{h.get('id')}@r{h.get('round')}"
            g.pop("analyst_pass", None)
            out.append(g)
        return out

    # -- ★ Analyst -> FeatureWriter (D-75) --------------------------------
    def _write_features(self, hyps: list[dict], r: int,
                        res: RoundResult) -> list[str]:
        """가설이 요구한 물리량을 **피처로 만든다.** 만든 이름들을 돌려준다.

        ## 왜 있나

        `needs_new_feature` 는 33실행에서 303번
        채워졌고 **`loop.py` 에 그것을 읽는 코드가 없었다.** 다섯 번에 한
        번꼴로 "이걸 재려면 새 축이 필요하다" 고 말한 것이 전부 버려졌다.

        ## 조건 셋 — 이 함수가 지킨다

        ```
        1  ★ FeatureWriter 에게 **진단 리포트를 주지 않는다.**
           요구 문장 하나만 넘긴다. 루프 안에서 만든 피처가 학습 형상에
           맞춰지는 통로를 막는다 — 그러면 F1 조건이 루프 안에서도 유지된다
        2  라운드당 상한 (`cfg.max_new_features_per_round`)
           §21 피처 행렬이 새 축마다 전 형상을 다시 계산한다
        3  판정은 성능이 아니라 관찰로 — 빈도 / 사용률 / 요구 내용
        ```

        ⚠️ 검증(§8.3)에 걸린 것은 **조용히 버리지 않는다.** 거부 사유를
        `features_made` 에 남긴다 — "무엇을 만들려다 실패했나" 가 관찰이다.
        """
        from kernelrule.features.generated import (
            FeatureRejected,
            register_generated,
        )
        from kernelrule.features.validate import alt_hw

        reqs = [(h.get("id", "?"), _requirement_of(h)) for h in hyps]
        reqs = [(hid, t) for hid, t in reqs if t]
        res.n_feature_requests = len(reqs)
        if not reqs:
            return []

        made: list[str] = []
        reg = self.matrix.registry
        cap = self.cfg.max_new_features_per_round
        # ★ 상한은 **코드에서** 건다. 프롬프트로 억제하면 재려는 것(요구
        #   빈도)을 직접 눌러 버린다 — 기준선 17.9% 는 그 문구가 없는
        #   조건에서 측정됐다. 넘친 요구는 **버리고 기록한다**: "몇 건이
        #   상한에 걸렸나" 자체가 관찰이다.
        for hid, text in reqs[cap:]:
            self.features_made.append(
                {"round": r, "hypothesis_id": hid, "requirement": text,
                 "accepted": False, "over_cap": True,
                 "error": f"라운드당 상한 {cap}개를 넘겼다 — 만들지 않았다"})
        res.n_feature_over_cap = max(0, len(reqs) - cap)
        for hid, text in reqs[:cap]:
            row = {"round": r, "hypothesis_id": hid, "requirement": text}
            try:
                out = self.llm.complete(
                    "feature", "", condition=self.cfg.feature_condition,
                    registry=reg, task=_feature_task(text))
                row["name"] = out.get("name")
                row["code"] = out.get("code")
                f = register_generated(out["code"], registry=reg, meta=out,
                                       table=self.table, matrix=self.matrix,
                                       hw_alt=alt_hw(self.table.hw))
                # ★ 열을 지금 만든다. 안 하면 규칙이 이름을 써도 KeyError 다.
                self.matrix.invalidate(f.name)
                self._feats = self.matrix.feature_names()
                self._fmins = self.matrix.feature_mins()
                self._shape_vals = self.matrix.shape_value_names()
                # ★ 워커는 fork 시점의 행렬을 들고 있다 — 새 열을 못 본다.
                #   풀을 버리고 다시 만든다. 안 하면 **워커가 낡은 행렬로
                #   채점하고** 그것이 조용히 다른 점수가 된다 (D-95).
                self._restart_pool()
                row.update(accepted=True, shape_level=f.shape_level)
                made.append(f.name)
            except FeatureRejected as e:
                row.update(accepted=False, error=str(e)[:200])
            except Exception as e:                          # noqa: BLE001
                row.update(accepted=False,
                           error=f"{type(e).__name__}: {e}"[:200])
            self.features_made.append(row)
        res.n_features_made = len(made)
        return made

    # -- 체제 마스크 (셀 축) — ★ 크기로 가른다 (§10.1, §30.5) -------------
    def _regime_masks(self):
        """학습 분할 안에서 짧은/긴 형상을 가른다.

        ⚠️ 경계를 `best_ms`(정답)가 아니라 **roofline 하한**으로 잡는다.
        ⚠️ **학습 분할 안에서만** 가른다 — 검증을 셀 축에 쓰면 홀드아웃이
        오염된다 (§10.2).
        """
        # ★ `regime_of` 를 쓴다 — 같은 판정이 두 곳에 있으면 달라진다 (원칙 2).
        #   전에는 여기서 `info.log_sol_ms < log2(0.5)` 를 직접 계산했는데,
        #   그것은 **레지스트리에 `log_sol_ms` 가 있다는 가정**이었다.
        #   F0/F1 레지스트리에는 없어서 루프가 통째로 죽는다 (§30.9).
        #   체제는 (형상, 하드웨어)의 성질이지 피처 목록의 성질이 아니다.
        from kernelrule.core.splits import regime_of

        short = np.asarray([regime_of(p, self.table.hw) == "short"
                            for p in self.splits.train.shapes])
        if not short.any() or short.all():
            import warnings
            warnings.warn(
                f"학습 분할이 한 크기 체제만 담고 있다 "
                f"(짧은 {int(short.sum())} / 긴 {int((~short).sum())}). "
                "셀 축이 무의미해지고, 진화가 다른 체제를 희생해도 안 보인다 "
                "(§10.1).", stacklevel=3)
        return short, ~short

    # -- 채점 -------------------------------------------------------------
    # -- 병렬 채점 (D-95) --------------------------------------------------
    def _pool_or_none(self):
        """`fork` 로 만든 프로세스 풀. **표와 행렬을 복사하지 않는다.**

        표 + 피처 행렬이 4.2GB 다. 워커마다 다시 로드하면 12개에 50GB 이고
        로드에만 3초씩 든다. `fork` 는 부모 메모리를 복사-후-쓰기로
        물려주므로 둘 다 안 든다 — 대신 **풀을 만들기 전에 `_WORKER` 를
        채워야 한다.**

        ⚠️ `fork` 는 스레드가 도는 중이면 위험하다. 여기는 `_call_optimizers`
        (asyncio)가 끝난 **뒤**에만 불린다.
        """
        if self.cfg.n_workers <= 0:
            return None
        if self._pool_exec is None:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context
            _WORKER.update(
                table=self.table, matrix=self.matrix,
                train=self.splits.train, val=self.splits.val,
                max_evals=self.cfg.max_evals,
                objective=self._objective,
                rank_top_k=self.cfg.rank_top_k,
                rank_lambda=self.cfg.rank_lambda,
                fit_method=self.cfg.fit_method,
                fit_restarts=self.cfg.fit_restarts,
                short_mask=self._short_mask, long_mask=self._long_mask)
            self._pool_exec = ProcessPoolExecutor(
                max_workers=self.cfg.n_workers, mp_context=get_context("fork"))
        return self._pool_exec

    def _restart_pool(self) -> None:
        """풀을 버리고 다시 만든다. **행렬이 바뀌면 반드시** (D-95)."""
        if self._pool_exec is not None:
            self._pool_exec.shutdown(wait=True)
            self._pool_exec = None
        self._pool_or_none()

    def _evaluate_batch(self, props: list, res: RoundResult) -> list:
        """여러 후보를 한 번에. 순차와 **결과가 같아야 한다** (D-95).

        ★ 결정론을 지키는 방법:
        ```
        제출 순서대로 결과를 조립한다 (`sorted(by i)`)
        rule_id 와 카운터는 부모가 그 순서로 매긴다
        fit_weights 는 시드가 고정이라 워커 순서와 무관하다
        ```
        """
        pool = self._pool_or_none()
        admitted = []
        for prop in props:
            got = self._admit(prop, res)
            if got is not None:
                admitted.append((prop, got[1]))
        if not admitted:
            return []
        if pool is None:                    # 순차 — 지금까지의 경로
            out = []
            for prop, _rep in admitted:
                e = self._evaluate_candidate(prop, res)
                if e is not None:
                    out.append(e)
            return out

        jobs = [(i, prop.code, list(prop.w0))
                for i, (prop, _r) in enumerate(admitted)]
        got = sorted(pool.map(_fit_and_score, jobs), key=lambda d: d["i"])
        elites = []
        for d in got:
            prop, rep = admitted[d["i"]]
            if "err" in d:
                res.n_rejected_fit += 1
                res.rejections.append(d["err"])
                continue
            if d["moved"]:
                res.n_fit_moved += 1
            res.n_scored += 1
            elites.append(self._elite_from(prop, rep, d))
        return elites

    def _score(self, score_fn, w, shapes):
        return evaluate_scores(make_score_of(score_fn, self.matrix, w),
                               self.table, list(shapes), ks=(1, 3))

    def _admit(self, prop, res: RoundResult):
        """정적 검사 -> 컴파일 -> 샌드박스. **fail-closed.** 부모에서 돈다.

        ★ 이 셋은 라운드의 3% 다. 병렬로 보내지 않는 이유는 비용이 아니라
        `run_isolated` 가 프로세스를 띄우기 때문이다 — 워커 안에서 하면
        중첩 spawn 이 된다 (D-95).
        """
        rep = check_rule(prop.code, limits=self._limits,
                         feature_names=self._feats,
                         feature_mins=self._fmins,
                         shape_value_names=self._shape_vals,
                         n_weights=len(prop.w0))
        if not rep.ok:
            res.n_rejected_static += 1
            res.rejections.append(("static", rep.violations[0][:90]))
            return None

        try:
            fn = compile_rule(prop.code)
        except SandboxError as e:
            res.n_rejected_sandbox += 1
            res.rejections.append(("compile", str(e)[:90]))
            return None

        if self.cfg.sandbox_first_seen:
            p0 = self.splits.train.shapes[0]
            f, info = self.matrix.for_shape(p0)
            out = run_isolated(prop.code, (f, info, self.table.hw,
                                           np.asarray(prop.w0)), timeout=5.0)
            if not out.ok:
                res.n_rejected_sandbox += 1
                res.rejections.append(("sandbox", str(out)[:90]))
                return None
        return fn, rep

    def _elite_from(self, prop, rep, out: dict) -> Elite:
        """워커 결과 -> `Elite`. ★ id 와 순서는 **부모가** 정한다 (결정론)."""
        self._rule_seq += 1
        return Elite(
            rule_id=f"r{self._rule_seq:04d}", code=prop.code,
            w=out["w"], regret=out["regret"],
            short_objective=out["short"], long_objective=out["long"],
            code_len=rep.n_nodes, round=len(self.rounds),
            changes=prop.changes, hypothesis_id=prop.hypothesis_id,
            val_regret=out["val_regret"],
            rank_loss=float(out.get("rank_loss", float("nan"))))

    def _evaluate_candidate(self, prop, res: RoundResult):
        """정적 검사 -> 샌드박스 -> 가중치 최적화 -> 채점. **전부 fail-closed.**"""
        got = self._admit(prop, res)
        if got is None:
            return None
        fn, rep = got

        try:
            fr = fit_weights(fn, self.matrix, self.table, self.splits.train,
                             prop.w0, max_evals=self.cfg.max_evals,
                             val_split=self.splits.val,
                             objective=self._objective,
                             rank_top_k=self.cfg.rank_top_k,
                             rank_lambda=self.cfg.rank_lambda,
                             method=self.cfg.fit_method,
                             n_restarts=self.cfg.fit_restarts,
                             bounds=weight_bounds(prop.code, len(prop.w0)))
        except (FitError, SchemaViolation) as e:
            res.n_rejected_fit += 1
            res.rejections.append(("fit", str(e)[:90]))
            return None
        except Exception as e:                            # noqa: BLE001
            # 규칙이 채점 중 터지면 **기각**이다. 삼키지 않는다 (§26.4).
            res.n_rejected_fit += 1
            res.rejections.append(("run", f"{type(e).__name__}: {e}"[:90]))
            return None

        if fr.moved:
            res.n_fit_moved += 1
        ev = self._score(fn, fr.w, self.splits.train.shapes)
        res.n_scored += 1
        return self._elite_from(prop, rep, {
            "w": [float(x) for x in fr.w], "regret": fr.fit_regret,
            "rank_loss": _rank_loss_of(fn, {
                "objective": self._objective,
                "rank_top_k": self.cfg.rank_top_k,
                "rank_lambda": self.cfg.rank_lambda,
                "matrix": self.matrix, "table": self.table,
                "train": self.splits.train}, fr.w),
            "val_regret": fr.val_regret, "moved": fr.moved,
            "short": ev.at(1, mask=self._short_mask),
            "long": ev.at(1, mask=self._long_mask)})

    def score_only(self, code: str, w0) -> float:
        """규칙 하나를 **학습 분할에서만** 채점한다. 아카이브에 안 넣는다.

        RuleWriter 후보를 줄 세우는 데 쓴다 (§30.9 2단계). 홀드아웃은
        `fit_weights` 가 보고용으로만 계산하고 여기서는 돌려주지 않는다 —
        씨앗 선택이 홀드아웃을 보면 그 홀드아웃은 홀드아웃이 아니다
        (§26.4, 원칙 6).
        """
        from kernelrule.agents.schemas import RuleProposal

        res = RoundResult(round=-2)
        e = self._evaluate_candidate(
            RuleProposal(code=code, w0=list(w0), changes="score_only"), res)
        if e is None:
            raise ValueError(f"후보가 거부됐다: {res.rejections}")
        return float(e.regret)

    def seed(self, code: str, w0, *, changes: str = "seed") -> Elite:
        """초기 규칙을 아카이브에 넣는다.

        **없으면 1라운드가 빈 부모에서 출발한다.** 손규칙을 기준선으로 삼아
        "리포트를 읽고 그것을 고칠 수 있는가" 를 시험하려면 여기서 시작해야
        한다 — 그러지 않으면 루프가 전혀 다른 규칙의 리포트를 본다.
        """
        from kernelrule.agents.schemas import RuleProposal

        res = RoundResult(round=-1)
        e = self._evaluate_candidate(
            RuleProposal(code=code, w0=list(w0), changes=changes), res)
        if e is None:
            raise ValueError(f"초기 규칙이 거부됐다: {res.rejections}")
        e.round = -1
        self.archive.consider(e)
        self._seen_code[code.strip()] = e.regret
        return e

    # -- 한 라운드 --------------------------------------------------------
    def _record_cross(self, r: int, codes: list[str], child: str) -> None:
        """★ 자식이 **두 부모 각각에만 있던 피처**를 둘 다 썼는가 (D-96).

        `cross` 가 두 부모를 받고도 한쪽만 베끼면 `explore` 와 다를 것이
        없다 — 그것이 실험 계획서의 관찰 1이다. **섞을 것이 없는 경우**
        (A 고유 또는 B 고유가 비었을 때) 를 따로 세지 않으면 분모가
        틀린다.
        """
        def feats(code: str) -> set:
            # ★ `except Exception` 이었다. 2026-09-03 에 `self._fmins` 를
            #   추가하면서 그 `AttributeError` 를 삼켜 피처 집합이 **빈 채로**
            #   지나갔다 — 관찰 장치가 조용히 0 이 되는 자리다. 규칙이
            #   못 읽히는 경우만 삼킨다.
            try:
                return check_rule(code, limits=self._limits,
                                  feature_names=self._feats,
                                  feature_mins=self._fmins,
                                  shape_value_names=self._shape_vals,
                                  n_weights=self._budget).features_used
            except SyntaxError:
                return set()
        a, b, c = feats(codes[0]), feats(codes[1]), feats(child)
        only_a, only_b = a - b, b - a
        self.cross_lineage.append({
            "round": r,
            "a": sorted(a), "b": sorted(b), "child": sorted(c),
            "only_a": sorted(only_a), "only_b": sorted(only_b),
            # 섞을 것이 있었는가 — 없으면 분모에서 뺀다
            "mixable": bool(only_a and only_b),
            "took_a": sorted(c & only_a), "took_b": sorted(c & only_b),
            "mixed": bool(c & only_a) and bool(c & only_b),
            # 한쪽을 통째로 베꼈는가 (부모 코드와 문자 단위로 같다)
            "copied": child.strip() in (codes[0].strip(), codes[1].strip())})

    def run_round(self) -> RoundResult:
        t0 = time.perf_counter()
        r = len(self.rounds)
        res = RoundResult(round=r)
        calls = {"analyze": 0, "rule_editor": 0, "feature": 0}
        self.trace.ev("round_start", round=r,
                      archive_best=(self.archive.best.regret
                                    if self.archive.best else None),
                      cells=self.archive.n_cells)

        # 1~2. 진단 리포트 -> 가설
        #  ★ `use_analyst=False` 면 이 블록을 통째로 건너뛴다 (§16.1).
        #    리포트를 만들지도 않는다 — 만들어 놓고 안 주면 "진단이 있는데
        #    안 쓴다" 가 되어 다른 조건이 된다.
        hyps: list[dict] = []
        if self.cfg.use_analyst and self.archive.best is not None:
            def analyze() -> list[dict]:
                """리포트를 **다시 만들고** 가설을 받는다.

                ★ 리포트를 재사용하지 않는다 — 3단계에서 축이 생기면
                피처 목록이 바뀌고, 옛 리포트를 다시 주면 Analyst 는 방금
                만든 축을 못 본다.
                """
                fn = compile_rule(self.archive.best.code)
                rep = build_report(
                    run_id=f"{self.cfg.run_id}-r{r:03d}", table=self.table,
                    matrix=self.matrix, score_fn=fn,
                    weights=self.archive.best.w,
                    code=self.archive.best.code, train=self.splits.train,
                    table_facts=self.table_facts,
                    failures=self.failures[-20:],
                    hypotheses_applied=[h["claim"][:70]
                                        for h in self.hypotheses[-3:]])
                out = self.llm.complete("analyze", rep.render())
                calls["analyze"] += 1
                return list((out or {}).get("hypotheses", []))

            first = analyze()
            for h in first:
                h["analyst_pass"] = 1
            hyps = first
            # ★ 되돌아가면 **두 응답을 다 남긴다.** 전에는 `hyps` 를 덮어써서
            #   첫 응답이 기록에서 사라졌다 — 요구가 담긴 쪽이 그쪽이라
            #   "요구 빈도" 의 **분모가 통째로 없어진다.**
            replaced: list[dict] = []

            # 3. ★ 없는 축을 요구했으면 만든다 (D-75)
            if self.cfg.max_new_features_per_round > 0 and first:
                made = self._write_features(first, r, res)
                calls["feature"] += min(res.n_feature_requests,
                                        self.cfg.max_new_features_per_round)
                if made:
                    # ★ **Analyst 로 되돌아간다.** 축을 만들어 놓고 그 라운드에
                    #   못 쓰면 반쪽이다 — 다음 라운드까지 기다리면 그 축을
                    #   요구한 가설과 이어지지 않는다.
                    second = analyze()
                    if second:
                        for h in second:
                            h["analyst_pass"] = 2
                        replaced, hyps = first, second

            # 가설에 id 를 붙인다. RuleEditor 프롬프트와 계보 추적에 쓰인다.
            # ⚠️ 요구 빈도를 옛 실행과 견줄 때는 `analyst_pass == 1` 만 센다 —
            #    옛 실행은 라운드당 Analyst 가 한 번이었다 (원칙 4).
            # ★ id 는 **우리가 정한다.** 모델이 자기 응답 안에서 `H0..H4` 를
            #   붙이므로 그것을 그대로 두면 라운드마다, 그리고 되돌아간
            #   두 응답 사이에서 겹친다. 겹치면 계보 추적이 조용히 어긋난다.
            for h in replaced + hyps:
                h["id"] = f"H{self._hyp_seq}"
                self._hyp_seq += 1
                h["round"] = r
                h.setdefault("analyst_pass", 1)
            self.hypotheses.extend(replaced)
            self.hypotheses.extend(hyps)
            self.trace.llm_calls(self.llm, round=r)
            self.trace.ev("hypotheses", round=r,
                          ids=[h.get("id") for h in hyps],
                          claims=[h.get("claim", "") for h in hyps],
                          needs_feature=[h.get("needs_feature")
                                         for h in hyps],
                          n_replaced=len(replaced))
        elif self.cfg.hypothesis_pool and self.archive.best is not None:
            # ★ 대조군 C — Analyst 는 안 부르고 남의 가설을 넣는다 (D-91)
            hyps = self._pool_round(r)
            for h in hyps:
                h["id"] = f"H{self._hyp_seq}"
                self._hyp_seq += 1
                h["round"] = r
                h["analyst_pass"] = 0        # 0 = 빌려옴
            self.hypotheses.extend(hyps)

        # 4. 규칙 생성 — ★ 병렬 호출 (§4-0). 12개나 1개나 벽시계가 비슷하다
        parents = self.archive.parents(self.cfg.n_rules_per_round, self.rng)
        applied = [f"{h.get('id','?')}: {h.get('claim','')[:80]}"
                   for h in self.hypotheses[-4:]]
        self.trace.ev("parents", round=r, picks=[
            {"kind": k, "rules": [x.rule_id for x in ps],
             "cells": [list(x.cell) if x.cell else None for x in ps]}
            for k, ps in parents])
        reqs = []
        for kind, ps in parents:
            parent, parent2, n_terms = None, None, 0
            if ps:
                from kernelrule.agents.schemas import RuleProposal
                parent = RuleProposal(code=ps[0].code, w0=ps[0].w)
                # ★ 두 번째 부모를 **실제로 넘긴다** (D-96). `archive.parents`
                #   는 `cross` 에 Elite 둘을 주는데 여기서 `ps[0]` 만 써서
                #   **§13 의 교차가 구현된 적이 없었다** — `cross` 가
                #   `explore` 와 같았다.
                if len(ps) > 1:
                    parent2 = RuleProposal(code=ps[1].code, w0=ps[1].w)
                # 부모의 항 수를 세어 프롬프트에 넣는다 (교체 프레임)
                pr = check_rule(ps[0].code, limits=self._limits,
                                feature_names=self._feats,
                                feature_mins=self._fmins,
                                shape_value_names=self._shape_vals,
                                n_weights=len(ps[0].w))
                n_terms = pr.n_terms
            # ★ 가설 배정을 **무작위**로 (D-94). `hyps[i % len(hyps)]` 는
            #   앞쪽 가설을 더 자주 쓴다 — 가설 5개면 3 3 2 2 2, 7개면
            #   2 2 2 2 2 1 1 이다. **설계가 아니라 12 % n 이고**, 부모
            #   종류(i=0~5 exploit / 6~8 explore / 9~11 cross)와도 상관된다.
            #   `self.rng` 를 쓰므로 시드로 재현된다.
            hyp = (hyps[int(self.rng.integers(len(hyps)))] if hyps else None)
            reqs.append({"prompt": f"round={r} parent={kind}",
                         # ★ 부모 종류를 **별도 필드**로 남긴다. 프롬프트
                         #   문자열에만 있으면 `llm_calls` 의 `prompt` 가 빈
                         #   실행에서 못 읽는다 — 실제로 "가설-부모 불일치"
                         #   를 옛 자료로 못 쟀다 (D-94).
                         "parent_kind": kind,
                         "parent": parent, "parent2": parent2,
                         # ★ 관찰용 부모 코드 (D-96 관찰 1). 프롬프트에는
                         #   안 들어간다 — `_user_prompt` 가 안 읽는 키다.
                         #   자식이 **두 부모 각각에만 있던 피처**를 둘 다
                         #   쓰는지 세려면 부모 피처 집합이 필요하다.
                         "_codes": [x.code for x in ps[:2]],
                         # ★ 트레이스용 부모 id (D-133). `_user_prompt` 가
                         #   안 읽는 키다 — 프롬프트에는 안 들어간다.
                         "_parent_ids": [x.rule_id for x in ps[:2]],
                         "parent_n_terms": n_terms,
                         "hypothesis": hyp,
                         "hypotheses_applied": applied,
                         # ★ 가설 절을 만들지 말지 (§16.1). `hypothesis=None`
                         #   으로 추측하면 안 된다 — 그것은 "가설이 없는
                         #   라운드" 와 "Analyst 자체가 없음" 을 섞는다
                         # 가설 절을 만들지 말지. 대조군 C 는 Analyst 를
                         # 안 부르지만 **가설은 받으므로** 절이 있어야 한다
                         "analyst": bool(self.cfg.use_analyst
                                         or self.cfg.hypothesis_pool)})
        raws = self._call_optimizers(reqs)
        calls["rule_editor"] += len(reqs)
        self.trace.llm_calls(self.llm, round=r)

        elites: list[Elite] = []
        #: 병렬로 보낼 후보 — 부모 종류를 함께 들고 간다 (D-94 계수용).
        batch: list = []

        def bump(kind: str, key: str) -> None:
            res.by_parent_kind.setdefault(
                kind, {"n": 0, "dup": 0, "scored": 0})[key] += 1

        for i, (req, raw) in enumerate(zip(reqs, raws, strict=True)):
            hyp = req["hypothesis"]
            kind = req.get("parent_kind", "?")
            bump(kind, "n")
            res.n_proposed += 1
            if isinstance(raw, BaseException):
                # ★ 두 가지를 가른다 (D-43).
                #   전송 실패   크레딧·인증·네트워크. **우리 문제**다
                #   재시도 소진  모델이 스키마를 못 맞춘 것. 폐기다 (§26.4)
                #   섞으면 "모델이 나쁜 규칙을 냈다" 로 읽힌다.
                if _is_transport_error(raw):
                    res.n_llm_error += 1
                    res.rejections.append(("llm-transport", (
                        f"{type(raw).__name__}: {str(raw)[:70]}")))
                    self.trace.ev("reject", round=r, i=i, kind=kind,
                                  why="llm-transport",
                                  detail=f"{type(raw).__name__}: {raw}"[:300])
                else:
                    res.n_rejected_schema += 1
                    res.rejections.append(("llm", (
                        f"{type(raw).__name__}: {str(raw)[:70]}")))
                    self.trace.ev("reject", round=r, i=i, kind=kind,
                                  why="llm",
                                  detail=f"{type(raw).__name__}: {raw}"[:300])
                continue
            try:
                # ★ 예산을 넘긴다 (D-107). 안 넘기면 MockLLM 경로와
                #   구조화 출력을 안 쓰는 경로에서 16항이 조용히 거부된다.
                prop = validate_rule_proposal(raw,
                                              parameters=self.cfg.parameters)
            except SchemaViolation as e:
                res.n_rejected_schema += 1
                res.rejections.append(("schema", str(e)[:90]))
                self.trace.ev("reject", round=r, i=i, kind=kind,
                              why="schema", detail=str(e)[:300])
                continue
            if hyp:
                prop.hypothesis_id = hyp.get("id", "")
            if kind == "cross" and len(req.get("_codes") or ()) > 1:
                self._record_cross(r, req["_codes"], prop.code)
            self.trace.ev("proposal", round=r, i=i, kind=kind,
                           parents=req.get("_parent_ids") or [],
                           hyp=(hyp or {}).get("id"),
                           changes=prop.changes, n_weights=len(prop.w0),
                           code=prop.code, code_sha=_sha(prop.code))
            key = prop.code.strip()
            if key in self._seen_code:      # 재채점하지 않는다 (§15.4)
                bump(kind, "dup")
                self.trace.ev("duplicate", round=r, i=i, kind=kind,
                              code_sha=_sha(prop.code))
                continue
            # ★ 중복 제거를 **병렬 진입 전에** 끝낸다 — 같은 라운드 안의
            #   중복도 여기서 잡아야 워커가 같은 일을 두 번 안 한다.
            self._seen_code[key] = float("nan")
            batch.append((prop, kind))

        # ★ 채점·적합은 한 번에 (D-95). `n_workers=0` 이면 순차 그대로다.
        got = self._evaluate_batch([b for b, _k in batch], res)
        by_code = {e.code.strip(): e for e in got}
        for prop, kind in batch:
            e2 = by_code.get(prop.code.strip())
            if e2 is None:
                self._seen_code.pop(prop.code.strip(), None)
                # ★ 채점까지 못 간 것 — 지금까지 어디에도 안 남았다 (D-133)
                self.trace.ev("reject", round=r, kind=kind, why="fit_or_run",
                              code_sha=_sha(prop.code))
                continue
            bump(kind, "scored")
            self.trace.ev("scored", round=r, kind=kind, rule=e2.rule_id,
                          code_sha=_sha(e2.code), fit=e2.regret,
                          val=e2.val_regret,
                          # ★ 적합기가 안 움직였는가 — "조용히 아무것도 안
                          #   함" 의 자리다 (D-54)
                          moved=bool(getattr(e2, "moved", True)))
            self._seen_code[prop.code.strip()] = e2.regret
            elites.append(e2)

        # 6~7. 아카이브 갱신 + 실패 기록
        before = self.archive.best.regret if self.archive.best else float("inf")
        for e in elites:
            won = self.archive.consider(e)
            self.trace.ev("archive", round=r, rule=e.rule_id,
                          accepted=bool(won),
                          cell=list(e.cell) if e.cell else None,
                          regret=e.regret)
            if won:
                res.n_accepted += 1
            else:
                self.failures.append({
                    "round": r, "idea": e.changes,
                    "regret_before": round(before, 4),
                    "regret_after": round(e.regret, 4),
                    "verdict": "made_worse" if e.regret > before
                               else "no_effect"})
        res.n_cells = self.archive.n_cells
        if self.archive.best:
            res.best_regret = self.archive.best.regret
            res.best_rank_loss = self.archive.best.rank_loss
            res.best_val_regret = self.archive.best.val_regret
            # ★ 라운드마다 **그때의 최고**를 코드째 남긴다 (D-101 관찰).
            #   `archive.jsonl` 은 마지막 상태뿐이라 "라운드마다 tau 가
            #   오르는가" 를 되짚을 수 없다.
            self.bests.append({
                "round": len(self.rounds), "rule_id": self.archive.best.rule_id,
                "code": self.archive.best.code, "w": self.archive.best.w,
                "regret": self.archive.best.regret,
                "rank_loss": self.archive.best.rank_loss})
            res.val_gap = res.best_val_regret - res.best_regret
        # ★ 아카이브는 **학습** 점수로 고른다 (검증을 쓰면 홀드아웃이 오염된다).
        #   그래서 검증에서 무너지는 규칙이 "최고" 가 될 수 있다 — 실제로 났다
        #   (train 1.164 / val 6.085). 선택은 그대로 두되 **경보를 낸다.**
        res.n_val_blowups = sum(
            1 for e in self.archive.cells.values()
            if np.isfinite(e.val_regret) and e.val_regret - e.regret
            > VAL_GAP_ALARM)
        res.llm_calls = calls
        res.seconds = time.perf_counter() - t0
        self.rounds.append(res)
        self.trace.ev("round_end", round=r, best=res.best_regret,
                      val=res.best_val_regret, cells=res.n_cells,
                      calls=dict(calls), proposed=res.n_proposed,
                      scored=res.n_scored, accepted=res.n_accepted,
                      rejected=(res.n_rejected_schema + res.n_rejected_static
                                + res.n_rejected_sandbox + res.n_rejected_fit),
                      seconds=round(res.seconds, 1))
        return res

    def _call_optimizers(self, reqs: list[dict]) -> list:
        """규칙 12개를 부른다. 클라이언트가 지원하면 **병렬**로.

        `MockLLM` 은 동기이고 `OpenAILLM` 은 `many()` 를 제공한다. 루프는
        둘을 구분하지 않는다 — `LLMClient` Protocol 뒤에 있다.
        """
        # ★ `_` 로 시작하는 키는 **관찰용**이고 프롬프트에 안 간다 (D-96).
        #   `_user_prompt` 가 안 읽으므로 넘겨도 지금은 무해하지만, 그것은
        #   **우연이다** — 나중에 누가 `kw` 를 훑으면 조용히 새어 들어간다.
        reqs = [{k: v for k, v in q.items() if not k.startswith("_")}
                for q in reqs]
        many = getattr(self.llm, "many", None)
        if many is None:
            out = []
            for req in reqs:
                q = dict(req)
                prompt = q.pop("prompt", "")
                try:
                    out.append(self.llm.complete("rule_editor", prompt, **q))
                except Exception as e:                    # noqa: BLE001
                    out.append(e)
            return out
        import asyncio
        return asyncio.run(many("rule_editor", [dict(q) for q in reqs]))

    # -- 목적함수 전환 (D-104) ---------------------------------------------
    def _maybe_switch(self) -> bool:
        """★ 직전 `switch_window` 라운드의 개선이 문턱 미만이면 바꾼다.

        전환은 **한 번뿐이다.** 두 번 바꾸면 "언제 바꾸나" 가 두 개가 되고
        그것이 곧 하이퍼파라미터다.
        """
        sw = self.cfg.objective_switch
        if not sw or self._switched:
            return False
        src, dst = sw.split("->")
        if self._objective != src:
            raise ValueError(
                f"objective_switch={sw!r} 인데 시작 목적함수가 "
                f"{self._objective!r} 다. 앞쪽과 같아야 한다.")
        n = self.cfg.switch_window
        if len(self.rounds) < n + 1:
            return False
        key = ("best_rank_loss" if self._objective == "rank"
               else "best_regret")
        vals = [getattr(x, key) for x in self.rounds[-(n + 1):]]
        if not np.all(np.isfinite(vals)) or vals[0] <= 0:
            return False
        if (vals[0] - vals[-1]) / vals[0] >= self.cfg.switch_min_improve:
            return False
        self._switch_to(dst)
        return True

    def _switch_to(self, dst: str) -> None:
        """목적함수를 바꾸고 **아카이브를 새 기준으로 재정렬한다.**"""
        from kernelrule.core.archive import Archive

        self._objective = dst
        self._switched = True
        self.switch_round = len(self.rounds)
        old = list(self.archive.cells.values())
        if self.archive.best is not None and self.archive.best not in old:
            old.append(self.archive.best)
        # ★ 새 기준값이 없으면 채운다. 조용히 NaN 으로 두면 아카이브가
        #   거부한다 (그것이 맞는 동작이다).
        if dst == "rank":
            for e in old:
                if not np.isfinite(e.rank_loss):
                    e.rank_loss = _rank_loss_of(
                        compile_rule(e.code),
                        {"objective": "rank",
                         "rank_top_k": self.cfg.rank_top_k,
                         "rank_lambda": self.cfg.rank_lambda,
                         "matrix": self.matrix, "table": self.table,
                         "train": self.splits.train}, np.asarray(e.w))
        self.archive = Archive(
            noise_tol=0.0,
            select_by=("rank" if dst == "rank" else "regret"),
            cell_mode=("quantile" if dst == "rank" else "absolute"))
        for e in sorted(old, key=lambda x: (x.rank_loss if dst == "rank"
                                            else x.regret)):
            self.archive.consider(e)
        # ★ 프롬프트의 목표 정의도 바꾼다. 캐시된 에이전트를 버린다 —
        #   안 버리면 옛 지시가 계속 간다 (원칙 1).
        if hasattr(self.llm, "objective"):
            self.llm.objective = dst
            if hasattr(self.llm, "_agents"):
                self.llm._agents.clear()
        self._restart_pool()
        print(f"  ★ 목적함수 전환 r{self.switch_round}: -> {dst} "
              f"(아카이브 {len(old)}개 재정렬 -> 셀 {len(self.archive.cells)})")

    # -- 종료 판정 (§14.3) -------------------------------------------------
    def should_stop(self) -> tuple[bool, str]:
        """★ **검증 분할**로 판정한다. 두 조건을 모두 쓴다."""
        n = self.cfg.patience
        # ★ 0 이면 조기 종료를 안 한다 (D-132). 라운드 상한이 유일한 종료다.
        if n <= 0:
            return False, ""
        if len(self.rounds) < n + 1:
            return False, ""
        vals = [x.best_val_regret for x in self.rounds[-(n + 1):]]
        if not np.all(np.isfinite(vals)):
            return False, ""
        improved = vals[0] - vals[-1]
        ev = self._score(compile_rule(self.archive.best.code),
                         self.archive.best.w, self.splits.val.shapes)
        significant = is_significant(improved, ev)
        new_cell_recent = (self.archive.last_new_cell_round
                           > len(self.rounds) - 1 - n)
        if significant or new_cell_recent:
            return False, ""
        return True, (f"{n}라운드 연속 검증 개선이 노이즈 바닥 이하"
                      f"({improved:+.5f}) 이고 새 셀도 없다")

    def run(self, n_rounds: int | None = None, *, verbose: bool = True,
            dump_each_round: bool = True):
        """라운드를 돌린다. ★ **끝날 때 반드시 저장한다** (D-33).

        전에는 `dump()` 를 호출자가 불러야 했고, 부르지 않은 러너가 78분
        1400호출의 결과를 통째로 잃었다. 규칙 코드가 메모리에만 있었으므로
        재채점이 불가능했다 — 표준출력의 요약만 남았다.

        `finally` 로 감싼 이유는 **중간에 죽어도 거기까지는 남아야** 하기
        때문이다. 예산 초과·rate limit·Ctrl-C 가 전부 여기 걸린다.
        `dump_each_round` 는 라운드마다 덮어써 장시간 실행의 보험이 된다
        (아카이브가 작아 비용이 무시할 만하다).
        """
        n = n_rounds or self.cfg.max_rounds
        # ★ 첫 줄이 자족해야 한다 (D-133 §3-3) — 트레이스 하나만 있어도
        #   조건을 알 수 있게 config 전체와 커밋 해시를 담는다.
        self.trace.ev("run_start", run_id=self.cfg.run_id, n_rounds=n,
                      commit=_git_commit(), config=self._config_dict(),
                      table=str(getattr(self.table, "bundle", "")),
                      split=self.splits.kind,
                      n_train=len(self.splits.train.shapes),
                      n_val=len(self.splits.val.shapes),
                      features=sorted(self.matrix.feature_names()))
        try:
            for _ in range(n):
                res = self.run_round()
                if verbose:
                    print(res.line(), flush=True)
                # ★ 목적함수 전환 (D-104). 라운드가 끝난 **뒤에** 본다 —
                #   그래야 그 라운드까지의 개선으로 판정한다.
                self._maybe_switch()
                # ★ 제안이 **전부** 전송 실패면 멈춘다 (D-43). 크레딧이나
                #   인증 문제는 저절로 낫지 않는다 — 남은 라운드를 태워도
                #   빈 아카이브만 남는다. 실제로 12라운드를 그렇게 썼다.
                if (STOP_ON_TOTAL_LLM_FAILURE and res.n_proposed
                        and res.n_llm_error == res.n_proposed):
                    raise LLMUnreachable(
                        f"r{res.round}: 제안 {res.n_proposed}건이 전부 LLM "
                        f"전송 실패다. 마지막 사유: "
                        + next((m for k, m in reversed(res.rejections)
                                if k == "llm-transport"), "?"))
                if dump_each_round:
                    self.dump()
                stop, why = self.should_stop()
                if stop:
                    if verbose:
                        print(f"조기 종료: {why}")
                    break
        finally:
            path = self.dump()
            # ★ 워커를 남기지 않는다. 12개가 4.2GB 를 공유한 채 떠 있으면
            #   다음 실행이 fork 할 때 메모리가 는다.
            if self._pool_exec is not None:
                self._pool_exec.shutdown(wait=True)
                self._pool_exec = None
            if verbose:
                print(f"  -> {path}", flush=True)
        return self.rounds

    def _config_dict(self) -> dict:
        """`config.json` 의 내용. ★ 트레이스 첫 줄도 **이것을** 쓴다 —
        두 곳에서 따로 만들면 어긋난다 (원칙 2, D-133)."""
        from kernelrule.core.splits import is_unsealed

        cfg: dict = {"loop": dict(self.cfg.__dict__),
                     "split": {"kind": self.splits.kind,
                               "n_train": len(self.splits.train.shapes),
                               "n_val": len(self.splits.val.shapes),
                               # ★ 최종 분할이 열린 채로 돈 실행인가 (§30.15).
                               #   열렸으면 그 수치는 **오염 가능**이다.
                               "unsealed": is_unsealed()},
                     "n_features": len(self.matrix.feature_names()),
                     # ★ 루프 **안에서** 만든 축 (D-75). 밖에서 받은 것과
                     #   섞이면 "라이브러리가 몇 개였나" 를 못 되짚는다.
                     "n_features_made_in_loop": sum(
                         1 for x in self.features_made if x.get("accepted")),
                     # ★ 규칙 제약. **조건이므로 실행마다 남긴다** (D-78).
                     #   분기 비교 상수 면제 전후는 같은 계열이 아니다.
                     # ★ 목적함수 전환 (D-104). **조건이므로 남긴다.**
                     "objective": self.cfg.objective,
                     "fit_method": self.cfg.fit_method,
                     "fit_restarts": self.cfg.fit_restarts,
                     "objective_switch": self.cfg.objective_switch,
                     "switch_round": self.switch_round,
                     "final_objective": self._objective,
                     "rule_constraints": {
                         "parameters": (self.cfg.parameters
                                        if self.cfg.parameters is not None
                                        else _PARAMETERS),
                         "branch_constants_exempt": True}}
        llm_cfg = getattr(self.llm, "cfg", None)
        if llm_cfg is not None and hasattr(llm_cfg, "to_dict"):
            cfg["llm"] = llm_cfg.to_dict()
        else:                                   # MockLLM 등
            cfg["llm"] = {"class": type(self.llm).__name__}
        return cfg

    def dump(self, out: str | Path | None = None) -> Path:
        # ★ **무엇으로 돌렸는지**를 남긴다 (D-31, D-45, D-51). 이것이 없으면
        #   나중에 어느 실행이 어느 모델/엔드포인트/추론강도였는지 알 수
        #   없고, 그러면 나란히 놓을 수 없다.
        d = Path(out or (Path(self.cfg.out_dir) / self.cfg.run_id))
        d.mkdir(parents=True, exist_ok=True)
        cfg = self._config_dict()
        (d / "config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=1, default=str))
        self.archive.dump(d / "archive.jsonl")
        (d / "rounds.jsonl").write_text("\n".join(
            json.dumps(x.__dict__, ensure_ascii=False, default=str)
            for x in self.rounds))
        (d / "failures.jsonl").write_text("\n".join(
            json.dumps(x, ensure_ascii=False) for x in self.failures))
        (d / "hypotheses.jsonl").write_text("\n".join(
            json.dumps(h, ensure_ascii=False) for h in self.hypotheses))
        # ★ 라운드 안에서 만든 축 (D-75). **거부된 것도 남긴다** — "무엇을
        #   만들려다 실패했나" 가 관찰이다.
        # ★ cross 계보 (D-96 관찰 1). 부모 코드는 어디에도 안 남으므로
        #   여기서만 되짚을 수 있다.
        if self.bests:
            (d / "bests.jsonl").write_text("\n".join(
                json.dumps(x, ensure_ascii=False) for x in self.bests))
        if self.cross_lineage:
            (d / "cross.jsonl").write_text("\n".join(
                json.dumps(x, ensure_ascii=False) for x in self.cross_lineage))
        if self.features_made:
            (d / "features.jsonl").write_text("\n".join(
                json.dumps(x, ensure_ascii=False) for x in self.features_made))
        if hasattr(self.llm, "dump"):
            self.llm.dump(d / "llm_calls")
        return d
