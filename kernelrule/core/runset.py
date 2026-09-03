"""★ 실행 묶음의 **조건 동일성**을 검사한다 (D-120).

## 왜

`transfer_29_5.TABLES["5090"]["runs"]` 가 여섯 실행을 (c) 로 묶었는데
뒤 셋이 `physics_seeded` **손씨앗**이었다. "5090 표에서 처음부터" 가
아닌 것이 (c) 에 섞여 있었고, 그 사실은 `chosen.json` 의 `source` 에
**적혀 있었다.** 아무도 안 읽었다.

```
D-113   arch_prompt 가 config.json 에 있었고 안 읽었다
D-119   씨앗 source 가 chosen.json 에 있었고 안 읽었다
★ 원칙 39 가 생긴 지 하루 만에 두 번째다
```

**그래서 검사로 만든다.** 묶음을 만드는 자리에서 부르면, 조건이 갈린
묶음은 **실패한다** (§26.4).

## 무엇을 보나

```
씨앗       stage2 `chosen.json` 의 source + 코드 해시
목적함수   objective / rank_top_k / rank_lambda
형태       rule_budget / product_hint / power_hint
하드웨어   arch_prompt 또는 hw_text 해시   ← D-113 이 여기였다
분할·조건  split.kind / feature_condition / 모델
```

값이 **없는 것**(옛 실행)은 갈림으로 안 센다 — 없는 것과 다른 것은
다르다. 다만 한쪽에만 있으면 그것은 갈림이다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = ["RunSetError", "run_condition", "assert_same_condition",
           "condition_report"]

ROOT = Path("runs")

#: 비교할 조건 키. **여기 없는 것은 안 본다** — 늘릴 때 시험도 같이 는다.
KEYS = ("seed_source", "seed_sha", "objective", "rank_top_k", "rank_lambda",
        "rule_budget", "product_hint", "power_hint", "hw", "split_kind",
        "feature_condition", "model", "fit_method", "fit_restarts")

#: ★ 새로 생긴 키의 **옛 기본값** (D-123). 옛 실행의 `config.json` 에는
#: 이 키가 없는데, 그때 코드가 하던 것이 이 값이다 — 그러므로 "없음" 을
#: 이 값으로 메우는 것은 봐주기가 아니라 **사실을 채우는 것**이다.
#: 새 키를 KEYS 에 넣을 때만 여기에 적는다. 값이 있는 실행끼리는
#: 그대로 견준다.
_OLD_DEFAULTS = {"fit_method": "nelder-mead", "fit_restarts": 4}


class RunSetError(ValueError):
    """묶음 안에서 조건이 갈렸다. **조용히 진행하지 않는다** (§26.4)."""


def _campaign(run: str) -> str:
    """`f1pipe-F3-tag-s0` -> `f1pipe-F3-tag`. 씨앗은 캠페인 단위다."""
    parts = run.rsplit("-s", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else run


def run_condition(run: str, root: Path | None = None) -> dict:
    """한 실행의 조건. 없는 값은 `None` 이다."""
    r = (root or ROOT)
    cfg_p = r / run / "config.json"
    if not cfg_p.exists():
        raise RunSetError(f"{cfg_p} 가 없다. 조건을 읽을 수 없다.")
    c = json.loads(cfg_p.read_text())
    loop, llm = c.get("loop", {}), c.get("llm", {})
    hw = llm.get("hw_text")
    hw_id = (hw.get("sha256") if isinstance(hw, dict)
             else llm.get("arch_prompt"))
    out = {
        "objective": loop.get("objective"),
        "rank_top_k": loop.get("rank_top_k"),
        "rank_lambda": loop.get("rank_lambda"),
        "rule_budget": (c.get("rule_constraints") or {}).get("budget"),
        "product_hint": llm.get("product_hint"),
        "power_hint": llm.get("power_hint"),
        "hw": hw_id,
        "split_kind": (c.get("split") or {}).get("kind"),
        "feature_condition": loop.get("feature_condition"),
        "model": llm.get("model") or llm.get("class"),
        # ★ 적합기 (D-123). 옛 실행에는 키가 없고, 그때는 Nelder-Mead
        #   4재시작뿐이었다 — `_OLD_DEFAULTS` 로 메운다.
        "fit_method": loop.get(
            "fit_method", _OLD_DEFAULTS["fit_method"]),
        "fit_restarts": loop.get(
            "fit_restarts", _OLD_DEFAULTS["fit_restarts"]),
        "seed_source": None, "seed_sha": None,
    }
    ch = r / _campaign(run) / "stage2-rule-writer" / "chosen.json"
    if ch.exists():
        d = json.loads(ch.read_text())
        out["seed_source"] = d.get("source")
        out["seed_sha"] = hashlib.sha256(
            (d.get("code") or "").encode()).hexdigest()[:12]
    return out


def condition_report(runs, root: Path | None = None) -> dict:
    """키별로 **관측된 값들**. 하나면 같은 조건이다."""
    conds = {r: run_condition(r, root) for r in runs}
    return {k: sorted({str(c[k]) for c in conds.values()}) for k in KEYS}


def assert_same_condition(runs, *, keys=KEYS, label: str = "묶음",
                          root: Path | None = None) -> dict:
    """묶음 안에서 조건이 하나인가. 아니면 **예외** (§26.4).

    ★ 묶음을 만드는 자리에서 불러라. 나중에 보면 "있었는데 안 봤다" 가
    된다 (원칙 39).
    """
    runs = list(runs)
    if len(runs) < 2:
        return {}
    conds = {r: run_condition(r, root) for r in runs}
    bad = {}
    for k in keys:
        vals = {r: c[k] for r, c in conds.items()}
        uniq = {str(v) for v in vals.values()}
        if len(uniq) > 1:
            bad[k] = vals
    if bad:
        lines = [f"{label} 안에서 조건이 갈렸다 ({len(bad)}개 키):"]
        for k, vals in bad.items():
            lines.append(f"  {k}:")
            for r, v in vals.items():
                lines.append(f"    {r:34s} {v}")
        lines.append("같은 조건이 아닌 실행을 한 묶음으로 집계하면 "
                     "그 수치는 두 조건의 평균이다 (D-119).")
        raise RunSetError("\n".join(lines))
    return {k: conds[runs[0]][k] for k in keys}
