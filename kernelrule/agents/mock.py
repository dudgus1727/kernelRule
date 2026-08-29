"""MockLLM (§24) — API 비용 없이 루프 전체를 개발·디버깅한다.

## 네 모드

    canned        미리 준비한 규칙을 순환        루프 배관 / 아카이브 / 채점
    mutate        부모의 **구조**를 섭동          진화 동역학 / 수렴 곡선
    adversarial   일부러 나쁜 코드를 낸다         ★ 정적 검사와 샌드박스
    replay        이전 run 의 응답을 재생         결정론적 재현

## ★ `mutate` 는 가중치가 아니라 **구조**를 섭동한다

§24.2 는 "부모 규칙의 가중치를 무작위 섭동" 이라고 했다. 그러면 아무것도
시험하지 못한다 — 가중치는 어차피 `fit_weights` 가 맞추므로(§29.3) 초기값
섭동은 채점에 거의 영향이 없다.

**항을 넣고 빼고 바꾸는 것**이 진화의 단위다. 그래서 이 목은 피처 항을
추가/제거/교체하고, 형상 수준 분기를 붙였다 뗀다. 그래야 "파이프라인이
빠진 항을 찾아내는가" 를 실제로 시험할 수 있다.

## 결정론

모든 모드가 시드로 결정론적이다. `replay` 가 성립하려면 필수다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from kernelrule.agents.schemas import (
    Hypothesis,
    RuleProposal,
    SchemaViolation,
)

__all__ = ["LLMClient", "MockLLM", "LLMCall", "ADVERSARIAL_CASES",
           "CANNED_RULES"]


class LLMClient(Protocol):
    def complete(self, role: str, prompt: str, **kw) -> Any: ...


@dataclass
class LLMCall:
    role: str
    prompt_hash: str
    response: Any
    seq: int
    mode: str = ""


# ---------------------------------------------------------------------------
# canned — 배관 확인용
# ---------------------------------------------------------------------------
CANNED_RULES: tuple[tuple[str, list[float]], ...] = (
    ("""def score(f, p, hw, w):
    return np.log2(f.traffic_amplification) * w[0]
""", [1.0]),
    ("""def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    return s + f.has_spill * w[1]
""", [1.0, 3.0]),
    ("""def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.has_spill * w[1]
    return s + f.is_two_stage * w[2]
""", [1.0, 3.0, 0.5]),
    ("""def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.has_spill * w[1]
    s = s + f.sm_idle_cost * w[2]
    return s + f.log_workspace_bytes * w[3]
""", [1.0, 3.0, 0.5, 0.1]),
    ("""def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.has_spill * w[1]
    s = s + f.smem_pressure * w[2]
    return s + f.split_k_cost * w[3]
""", [1.0, 3.0, 0.4, 0.3]),
)

# ---------------------------------------------------------------------------
# adversarial — ★ 하나라도 통과하면 방어에 구멍이 있다 (§24.3)
# ---------------------------------------------------------------------------
ADVERSARIAL_CASES: tuple[tuple[str, str, list[float]], ...] = (
    ("암기", """def score(f, p, hw, w):
    if p.M == 4096:
        return f.waves * w[0]
    return f.waves * w[0]
""", [1.0]),
    ("정답 누출", """def score(f, p, hw, w):
    return f.waves * w[0] + time_ms
""", [1.0]),
    ("난이도 참조", """def score(f, p, hw, w):
    return f.waves * w[0] * difficulty
""", [1.0]),
    ("무한 루프", """def score(f, p, hw, w):
    while True:
        pass
    return f.waves * w[0]
""", [1.0]),
    ("샌드박스 탈출", """def score(f, p, hw, w):
    import os
    os.system('echo PWNED')
    return f.waves * w[0]
""", [1.0]),
    ("오타", """def score(f, p, hw, w):
    return f.tail_wast * w[0]
""", [1.0]),
    ("배열에 if", """def score(f, p, hw, w):
    if f.waves < 1:
        return f.waves * w[0]
    return f.waves * w[0]
""", [1.0]),
    ("리터럴 초과", """def score(f, p, hw, w):
    return (f.waves*1.1 + f.tail_waste*2.2 + f.smem_pressure*3.3
            + f.has_spill*4.4 + f.edge_waste*5.5 + 6.6 + 7.7 + 8.8) * w[0]
""", [1.0]),
    ("비결정론", """def score(f, p, hw, w):
    return np.random.rand(3) * w[0]
""", [1.0]),
    ("구문 오류", """def score(f, p, hw, w)
    return 1
""", [1.0]),
    ("w 슬라이싱", """def score(f, p, hw, w):
    return f.waves * w[0] + sum(w[1:])
""", [1.0, 2.0]),
    ("던더 우회", """def score(f, p, hw, w):
    return f.waves * w[0] + score.__globals__['x']
""", [1.0]),
)


# ---------------------------------------------------------------------------
# mutate — ★ 구조를 섭동한다
# ---------------------------------------------------------------------------
_TEMPLATE_HEAD = "def score(f, p, hw, w):\n"


def _render_rule(terms: list[str], branch: tuple[str, str] | None) -> tuple:
    """항 목록 -> 코드 + `w0`. 가중치는 순서대로 `w[i]` 다."""
    lines, i = [], 0
    for t in terms:
        op = "s = " if i == 0 else "s = s + "
        lines.append(f"    {op}{t} * w[{i}]")
        i += 1
    if branch is not None:
        cond, term = branch
        lines.append(f"    if p.{cond}:")
        lines.append(f"        s = s + {term} * w[{i}]")
        i += 1
    lines.append("    return s")
    w0 = [1.0] * i
    return _TEMPLATE_HEAD + "\n".join(lines) + "\n", w0


def _parse_terms(code: str) -> tuple[list[str], tuple[str, str] | None]:
    """`_render_rule` 이 만든 코드를 되읽는다. 목 전용 파서다."""
    terms, branch, cond = [], None, None
    for ln in code.split("\n"):
        t = ln.strip()
        if t.startswith("if p."):
            cond = t[5:].rstrip(":")
        elif t.startswith(("s = ", "s = s + ")) and " * w[" in t:
            expr = t.split(" * w[")[0]
            expr = expr.replace("s = s + ", "").replace("s = ", "")
            if cond is not None:
                branch = (cond, expr)
            else:
                terms.append(expr)
    return terms, branch


class MockLLM:
    """API 없이 도는 LLM 대역. **결정론적이다.**"""

    def __init__(self, mode: str = "canned", *, seed: int = 0,
                 feature_names: list[str] | None = None,
                 shape_values: list[str] | None = None,
                 replay_dir: str | Path | None = None) -> None:
        if mode not in ("canned", "mutate", "adversarial", "replay"):
            raise ValueError(f"알 수 없는 모드: {mode!r}")
        self.mode = mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.features = list(feature_names or [])
        self.shape_values = list(shape_values or ["is_memory_bound"])
        self.calls: list[LLMCall] = []
        self._seq = 0
        self._n_features = 0
        self.replay_dir = Path(replay_dir) if replay_dir else None
        self._replay: list[LLMCall] = []
        if mode == "replay":
            self._load_replay()

    # -- 기록/재생 --------------------------------------------------------
    def _load_replay(self) -> None:
        if self.replay_dir is None or not self.replay_dir.exists():
            raise FileNotFoundError(
                f"replay 모드인데 {self.replay_dir} 가 없다. "
                "조용히 canned 로 떨어지지 않는다 (§26.4).")
        for f in sorted(self.replay_dir.glob("*.json")):
            d = json.loads(f.read_text())
            self._replay.append(LLMCall(role=d["role"],
                                        prompt_hash=d["prompt_hash"],
                                        response=d["response"],
                                        seq=d["seq"], mode=d.get("mode", "")))

    def dump(self, out: str | Path) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for c in self.calls:
            (out / f"{c.seq:05d}-{c.role}.json").write_text(json.dumps(
                {"role": c.role, "prompt_hash": c.prompt_hash, "seq": c.seq,
                 "mode": c.mode, "response": c.response},
                ensure_ascii=False, indent=1))

    # -- 진입점 -----------------------------------------------------------
    def complete(self, role: str, prompt: str, **kw) -> Any:
        h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        seq = self._seq
        self._seq += 1
        if self.mode == "replay":
            if seq >= len(self._replay):
                raise SchemaViolation(
                    f"replay 에 {seq}번 호출이 없다 ({len(self._replay)}개뿐). "
                    "루프가 달라졌다 — 조용히 새로 만들지 않는다.")
            rec = self._replay[seq]
            if rec.role != role:
                raise SchemaViolation(
                    f"replay 불일치: {seq}번이 {rec.role!r} 인데 {role!r} 요청")
            return rec.response
        resp = self._generate(role, prompt, **kw)
        self.calls.append(LLMCall(role=role, prompt_hash=h, response=resp,
                                  seq=seq, mode=self.mode))
        return resp

    def _generate(self, role: str, prompt: str, **kw) -> Any:
        if role == "analyze":
            return self._diagnose(prompt)
        if role == "optimize":
            return self._optimize(prompt, **kw)
        if role == "feature":
            return self._feature()
        if role == "architect":
            return self._architect(**kw)
        if role == "categorize":
            return {"categories": [
                {"name": f"mock_area_{i}", "description": f"목 영역 {i}"}
                for i in range(5)], "notes": "목이 나눈 영역이다"}
        if role == "critique":
            return self._critique(**kw)
        raise ValueError(f"알 수 없는 역할: {role!r}")

    # -- FeatureWriter / Architect — ★ 배관 확인용 (§30.9) ----------------
    #
    #   실제 LLM 없이 F0~F3 파이프라인이 **끝까지 도는지** 보려면 이 두
    #   역할이 있어야 한다. 목이 만드는 피처는 물리적으로 의미 없다 —
    #   `--dry-run` 의 목적은 성능이 아니라 배관이다.

    #: 원시 값만으로 만드는 피처 틀. `RAW_FIELDS` 안의 이름만 쓴다.
    _FEATURE_FORMS = (
        ("mock_tile_area_ratio", "타일 면적 / 문제 면적",
         "float(cfg.tile_m * cfg.tile_n) / max(1.0, float(p.M) * p.N)"),
        ("mock_k_depth", "K 방향 반복 깊이",
         "float(p.K) / max(1.0, float(cfg.tile_k))"),
        ("mock_thread_load", "스레드당 출력 원소",
         "float(cfg.tile_m * cfg.tile_n) / max(1.0, float(cfg.threads))"),
        ("mock_smem_share", "SM 공유메모리 점유율",
         "float(cfg.smem_bytes) / max(1.0, float(hw.smem_per_block))"),
        ("mock_grid_per_sm", "SM 당 타일 수",
         ("float(p.M) * p.N / max(1.0, float(cfg.tile_m * cfg.tile_n))"
          " / max(1.0, float(hw.sm_count))")),
    )

    def _feature(self) -> dict:
        """제안마다 **다른** 피처를 낸다. 같은 것을 반복하면 중복 판정에
        전부 걸려서 배관 확인이 안 된다."""
        i = self._n_features % len(self._FEATURE_FORMS)
        self._n_features += 1
        name, doc, expr = self._FEATURE_FORMS[i]
        suffix = "" if self._n_features <= len(self._FEATURE_FORMS) else \
            f"_{self._n_features}"
        code = (f"def {name}{suffix}(p, hw, cfg) -> float:\n"
                f'    """{doc}."""\n'
                f"    return {expr}\n")
        return {"name": name + suffix, "code": code, "unit": "dimensionless",
                "direction": "higher_is_worse", "expected_range": [0.0, 1e6],
                "rationale": "목이 만든 피처다 — 배관 확인용이다"}

    def _critique(self, *, code: str = "", **_kw) -> dict:
        """★ 항 단위 심사의 배관 확인용 (D-85).

        **마지막 항을 항상 "설명 불가" 로 낸다** — 절제 검증(`--ablate`)이
        실제로 도는지 보려면 설명 불가가 하나는 나와야 한다. 물리적으로
        의미 있는 판정이 아니다.
        """
        import re

        idx = sorted({int(i) for i in re.findall(r"\bw\[(\d+)\]", code)})
        terms = []
        for k, i in enumerate(idx):
            last = k == len(idx) - 1
            terms.append({
                "index": i, "expression": f"(mock 항 {i})",
                "physics": "목이 만든 설명이다 — 배관 확인용이다",
                "explainable": not last,
                "why_not": "목이 마지막 항을 항상 설명 불가로 낸다" if last
                           else "",
                "regime_dependent": False, "regime": ""})
        return {"terms": terms, "overall": "목이 만든 심사다",
                "defects": ["(mock) 결함 없음"]}

    def _architect(self, **kw) -> dict:
        """씨앗 규칙. **주어진 피처 이름만** 쓴다 (F1 이면 F1 피처).

        `self.features` 가 비어 있으면 조용히 사람 피처로 떨어지지 않고
        예외를 낸다 — 그것이 §30.9 가 막으려는 경로다.
        """
        if not self.features:
            raise ValueError(
                "MockLLM(feature_names=...) 이 비었다. 씨앗을 만들 피처가 "
                "없다 — 조용히 사람이 쓴 24개로 떨어지지 않는다 (§26.4).")
        n = min(4, len(self.features))
        pick = [self.features[int(i)] for i in
                self.rng.choice(len(self.features), size=n, replace=False)]
        code, w0 = _render_rule([f"f.{x}" for x in pick], None)
        return {"code": code, "w0": w0,
                "changes": "목 Architect 씨앗 — 주어진 피처에서 골랐다"}

    def _diagnose(self, prompt: str) -> dict:
        """진단 — 리포트에서 **미사용 피처**를 읽어 가설로 만든다.

        이 목이 하는 유일한 '지능' 이다. 리포트가 `★ 미사용` 열을 내므로
        그것을 읽는다. **루프 배관이 그 정보를 전달하는지** 시험하는 것이
        목적이다.
        """
        missing = []
        for ln in prompt.split("\n"):
            if "★ 미사용" in ln:
                name = ln.split()[0]
                if name in self.features and name not in missing:
                    missing.append(name)
        hyps = [Hypothesis(
            id=f"H{i}", claim=f"규칙이 {n} 를 쓰지 않는다. 사례에서 선택과 "
                              f"최적의 값이 크게 다르다",
            measurable_with=[n], proposed_direction=f"{n} 항을 추가한다",
            risk="다른 항의 효과를 희석할 수 있다")
            for i, n in enumerate(missing[:5])]
        # ★ 첫 가설만 없는 축을 요구한다 (D-75 경로 배관 확인용).
        #   루프는 `max_new_features_per_round > 0` 일 때만 이것을 읽으므로
        #   기존 dry-run 은 영향을 받지 않는다.
        if hyps:
            hyps[0].needs_new_feature = (
                "타일 하나가 L2 에 남아 다음 타일이 재사용하는 양")
        return {"hypotheses": [h.__dict__ for h in hyps]}

    def _optimize(self, prompt: str, *, parent: RuleProposal | None = None,
                  hypothesis: dict | None = None, **kw) -> dict:
        if self.mode == "adversarial":
            name, code, w0 = ADVERSARIAL_CASES[
                self._seq % len(ADVERSARIAL_CASES)]
            return {"code": code, "w0": w0, "changes": f"[adversarial] {name}"}
        if self.mode == "canned":
            code, w0 = CANNED_RULES[self._seq % len(CANNED_RULES)]
            return {"code": code, "w0": w0, "changes": "[canned]"}
        return self._mutate(parent, hypothesis)

    def _mutate(self, parent: RuleProposal | None,
                hypothesis: dict | None) -> dict:
        """★ 구조를 섭동한다. 가중치가 아니다 (모듈 docstring 참조)."""
        base = (parent.code if parent is not None
                else "def score(f, p, hw, w):\n"
                     "    s = f.traffic_amplification * w[0]\n    return s\n")
        terms, branch = _parse_terms(base)
        if not terms:
            terms = ["f.traffic_amplification"]
        pool = [f"f.{n}" for n in self.features]
        unused = [t for t in pool if t not in terms
                  and (branch is None or t != branch[1])]

        # 가설이 특정 피처를 지목하면 그것을 우선 추가한다
        want = None
        if hypothesis:
            for n in hypothesis.get("measurable_with", []):
                if f"f.{n}" in unused:
                    want = f"f.{n}"
                    break

        r = self.rng.random()
        changes = ""
        if want is not None and r < 0.65:
            terms.append(want)
            changes = f"가설이 지목한 {want} 항 추가"
        elif unused and r < 0.55:
            t = unused[int(self.rng.integers(len(unused)))]
            terms.append(t)
            changes = f"{t} 항 추가 (무작위 탐색)"
        elif len(terms) > 1 and r < 0.72:
            i = int(self.rng.integers(len(terms)))
            changes = f"{terms[i]} 항 제거"
            terms.pop(i)
        elif unused and len(terms) > 1 and r < 0.86:
            i = int(self.rng.integers(len(terms)))
            t = unused[int(self.rng.integers(len(unused)))]
            changes = f"{terms[i]} -> {t} 교체"
            terms[i] = t
        elif branch is None and unused:
            cond = self.shape_values[int(self.rng.integers(
                len(self.shape_values)))]
            t = unused[int(self.rng.integers(len(unused)))]
            branch = (cond, t)
            changes = f"형상 수준 분기 추가: if p.{cond} -> {t} 재가중"
        else:
            branch = None
            changes = "형상 수준 분기 제거"

        # 리터럴 예산(8) 안으로 자른다. 넘으면 정적 검사가 거부한다.
        n_w = len(terms) + (1 if branch else 0)
        while n_w > 8 and len(terms) > 1:
            terms.pop()
            n_w = len(terms) + (1 if branch else 0)
        code, w0 = _render_rule(terms, branch)
        return {"code": code, "w0": w0, "changes": changes}
