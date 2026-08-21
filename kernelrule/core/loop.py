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
from kernelrule.rules.checks import check_rule

__all__ = ["RoundLoop", "RoundResult", "LoopConfig"]


@dataclass
class LoopConfig:
    run_id: str
    n_rules_per_round: int = 12
    max_rounds: int = 20
    max_evals: int = 200
    #: 종료: 이만큼 연속으로 유의한 개선이 없고 새 셀도 없으면 멈춘다
    patience: int = 10
    seed: int = 0
    sandbox_first_seen: bool = True
    out_dir: str = "runs"


#: 검증 격차가 이보다 크면 **체제 전이 실패**로 본다.
#: 과적합과 다르다 — 항이 3개뿐인 규칙도 이 값을 넘는다 (실측 +4.99).
VAL_GAP_ALARM = 0.5


@dataclass
class RoundResult:
    round: int
    n_proposed: int = 0
    n_rejected_static: int = 0
    n_rejected_sandbox: int = 0
    n_rejected_schema: int = 0
    n_rejected_fit: int = 0
    n_scored: int = 0
    n_accepted: int = 0
    best_regret: float = float("nan")
    best_val_regret: float = float("nan")
    n_cells: int = 0
    val_gap: float = float("nan")
    n_val_blowups: int = 0
    seconds: float = 0.0
    llm_calls: dict = field(default_factory=dict)
    rejections: list[tuple] = field(default_factory=list)

    def line(self) -> str:
        return (f"r{self.round:<3d} 제안 {self.n_proposed:2d} | "
                f"거부 스키마 {self.n_rejected_schema} 정적 "
                f"{self.n_rejected_static} 샌드박스 {self.n_rejected_sandbox} "
                f"적합 {self.n_rejected_fit} | 채점 {self.n_scored:2d} "
                f"채택 {self.n_accepted:2d} | best {self.best_regret:.4f} "
                f"val {self.best_val_regret:.4f}"
                f"({self.val_gap:+.3f}{'!' if self.val_gap > VAL_GAP_ALARM else ' '})"
                f"| 셀 {self.n_cells:2d} 폭발 {self.n_val_blowups} | "
                f"{self.seconds:.1f}s")


class RoundLoop:
    def __init__(self, *, cfg: LoopConfig, table: PerfTable,
                 matrix: FeatureMatrix, splits: SplitSet, llm,
                 table_facts: list[str] | None = None) -> None:
        self.cfg = cfg
        self.table = table
        self.matrix = matrix
        self.splits = splits
        self.llm = llm
        self.table_facts = list(table_facts or [])
        self.rng = np.random.default_rng(cfg.seed)
        self.archive = Archive(noise_tol=0.0)
        self.rounds: list[RoundResult] = []
        self.failures: list[dict] = []
        self.hypotheses: list[dict] = []
        self._rule_seq = 0
        self._seen_code: dict[str, float] = {}      # 캐시 (§15.4)
        self._feats = matrix.feature_names()
        self._shape_vals = matrix.shape_value_names()
        self._short_mask, self._long_mask = self._regime_masks()

    # -- 체제 마스크 (셀 축) — ★ 크기로 가른다 (§10.1, §30.5) -------------
    def _regime_masks(self):
        """학습 분할 안에서 짧은/긴 형상을 가른다.

        ⚠️ 경계를 `best_ms`(정답)가 아니라 **roofline 하한**으로 잡는다.
        ⚠️ **학습 분할 안에서만** 가른다 — 검증을 셀 축에 쓰면 홀드아웃이
        오염된다 (§10.2).
        """
        import math

        short = []
        for p in self.splits.train.shapes:
            _, info = self.matrix.for_shape(p)
            short.append(info.log_sol_ms < math.log2(0.5))
        short = np.asarray(short)
        if not short.any() or short.all():
            import warnings
            warnings.warn(
                f"학습 분할이 한 크기 체제만 담고 있다 "
                f"(짧은 {int(short.sum())} / 긴 {int((~short).sum())}). "
                "셀 축이 무의미해지고, 진화가 다른 체제를 희생해도 안 보인다 "
                "(§10.1).", stacklevel=3)
        return short, ~short

    # -- 채점 -------------------------------------------------------------
    def _score(self, score_fn, w, shapes):
        return evaluate_scores(make_score_of(score_fn, self.matrix, w),
                               self.table, list(shapes), ks=(1, 3))

    def _evaluate_candidate(self, prop, res: RoundResult):
        """정적 검사 -> 샌드박스 -> 가중치 최적화 -> 채점. **전부 fail-closed.**"""
        rep = check_rule(prop.code, feature_names=self._feats,
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

        try:
            fr = fit_weights(fn, self.matrix, self.table, self.splits.train,
                             prop.w0, max_evals=self.cfg.max_evals,
                             val_split=self.splits.val)
        except (FitError, SchemaViolation) as e:
            res.n_rejected_fit += 1
            res.rejections.append(("fit", str(e)[:90]))
            return None
        except Exception as e:                            # noqa: BLE001
            # 규칙이 채점 중 터지면 **기각**이다. 삼키지 않는다 (§26.4).
            res.n_rejected_fit += 1
            res.rejections.append(("run", f"{type(e).__name__}: {e}"[:90]))
            return None

        ev = self._score(fn, fr.w, self.splits.train.shapes)
        res.n_scored += 1
        self._rule_seq += 1
        return Elite(
            rule_id=f"r{self._rule_seq:04d}", code=prop.code,
            w=[float(x) for x in fr.w], regret=fr.fit_regret,
            short_regret=ev.at(1, mask=self._short_mask),
            long_regret=ev.at(1, mask=self._long_mask),
            code_len=rep.n_nodes, round=len(self.rounds),
            changes=prop.changes, hypothesis_id=prop.hypothesis_id,
            val_regret=fr.val_regret)

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
    def run_round(self) -> RoundResult:
        t0 = time.perf_counter()
        r = len(self.rounds)
        res = RoundResult(round=r)
        calls = {"diagnose": 0, "optimize": 0, "instrument": 0, "audit": 0}

        # 1~2. 진단 리포트 -> 가설
        hyps: list[dict] = []
        if self.archive.best is not None:
            fn = compile_rule(self.archive.best.code)
            rep = build_report(
                run_id=f"{self.cfg.run_id}-r{r:03d}", table=self.table,
                matrix=self.matrix, score_fn=fn, weights=self.archive.best.w,
                code=self.archive.best.code, train=self.splits.train,
                table_facts=self.table_facts, failures=self.failures[-20:],
                hypotheses_applied=[h["claim"][:70]
                                    for h in self.hypotheses[-3:]])
            out = self.llm.complete("diagnose", rep.render())
            calls["diagnose"] += 1
            hyps = list((out or {}).get("hypotheses", []))
            # 가설에 id 를 붙인다. Optimizer 프롬프트와 계보 추적에 쓰인다.
            for j, h in enumerate(hyps):
                h.setdefault("id", f"H{len(self.hypotheses) + j}")
                h["round"] = r
            self.hypotheses.extend(hyps)

        # 4. 규칙 생성 — ★ 병렬 호출 (§4-0). 12개나 1개나 벽시계가 비슷하다
        parents = self.archive.parents(self.cfg.n_rules_per_round, self.rng)
        applied = [f"{h.get('id','?')}: {h.get('claim','')[:80]}"
                   for h in self.hypotheses[-4:]]
        reqs = []
        for i, (kind, ps) in enumerate(parents):
            parent, n_terms = None, 0
            if ps:
                from kernelrule.agents.schemas import RuleProposal
                parent = RuleProposal(code=ps[0].code, w0=ps[0].w)
                # 부모의 항 수를 세어 프롬프트에 넣는다 (교체 프레임)
                pr = check_rule(ps[0].code, feature_names=self._feats,
                                shape_value_names=self._shape_vals,
                                n_weights=len(ps[0].w))
                n_terms = pr.n_terms
            reqs.append({"prompt": f"round={r} parent={kind}",
                         "parent": parent, "parent_n_terms": n_terms,
                         "hypothesis": hyps[i % len(hyps)] if hyps else None,
                         "hypotheses_applied": applied})
        raws = self._call_optimizers(reqs)
        calls["optimize"] += len(reqs)

        elites: list[Elite] = []
        for req, raw in zip(reqs, raws, strict=True):
            hyp = req["hypothesis"]
            res.n_proposed += 1
            if isinstance(raw, BaseException):
                # ★ 재시도 상한을 넘은 것은 **폐기**다. 부분 수용 금지 (§26.4)
                res.n_rejected_schema += 1
                res.rejections.append(("llm", (f"{type(raw).__name__}: "
                                              f"{str(raw)[:70]}")))
                continue
            try:
                prop = validate_rule_proposal(raw)
            except SchemaViolation as e:
                res.n_rejected_schema += 1
                res.rejections.append(("schema", str(e)[:90]))
                continue
            if hyp:
                prop.hypothesis_id = hyp.get("id", "")
            key = prop.code.strip()
            if key in self._seen_code:      # 재채점하지 않는다 (§15.4)
                continue
            e2 = self._evaluate_candidate(prop, res)
            if e2 is not None:
                self._seen_code[key] = e2.regret
                elites.append(e2)

        # 6~7. 아카이브 갱신 + 실패 기록
        before = self.archive.best.regret if self.archive.best else float("inf")
        for e in elites:
            won = self.archive.consider(e)
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
            res.best_val_regret = self.archive.best.val_regret
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
        return res

    def _call_optimizers(self, reqs: list[dict]) -> list:
        """규칙 12개를 부른다. 클라이언트가 지원하면 **병렬**로.

        `MockLLM` 은 동기이고 `OpenAILLM` 은 `many()` 를 제공한다. 루프는
        둘을 구분하지 않는다 — `LLMClient` Protocol 뒤에 있다.
        """
        many = getattr(self.llm, "many", None)
        if many is None:
            out = []
            for req in reqs:
                q = dict(req)
                prompt = q.pop("prompt", "")
                try:
                    out.append(self.llm.complete("optimize", prompt, **q))
                except Exception as e:                    # noqa: BLE001
                    out.append(e)
            return out
        import asyncio
        return asyncio.run(many("optimize", [dict(q) for q in reqs]))

    # -- 종료 판정 (§14.3) -------------------------------------------------
    def should_stop(self) -> tuple[bool, str]:
        """★ **검증 분할**로 판정한다. 두 조건을 모두 쓴다."""
        n = self.cfg.patience
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

    def run(self, n_rounds: int | None = None, *, verbose: bool = True):
        n = n_rounds or self.cfg.max_rounds
        for _ in range(n):
            res = self.run_round()
            if verbose:
                print(res.line(), flush=True)
            stop, why = self.should_stop()
            if stop:
                if verbose:
                    print(f"조기 종료: {why}")
                break
        return self.rounds

    def dump(self, out: str | Path | None = None) -> Path:
        d = Path(out or (Path(self.cfg.out_dir) / self.cfg.run_id))
        d.mkdir(parents=True, exist_ok=True)
        self.archive.dump(d / "archive.jsonl")
        (d / "rounds.jsonl").write_text("\n".join(
            json.dumps(x.__dict__, ensure_ascii=False, default=str)
            for x in self.rounds))
        (d / "failures.jsonl").write_text("\n".join(
            json.dumps(x, ensure_ascii=False) for x in self.failures))
        (d / "hypotheses.jsonl").write_text("\n".join(
            json.dumps(h, ensure_ascii=False) for h in self.hypotheses))
        if hasattr(self.llm, "dump"):
            self.llm.dump(d / "llm_calls")
        return d
