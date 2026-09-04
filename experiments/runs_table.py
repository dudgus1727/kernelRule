"""★ `docs/artifacts/runs.md` 를 **만든다** — 손으로 쓰지 않는다 (D-128).

    python3 experiments/runs_table.py            # 생성
    python3 experiments/runs_table.py --check    # 갈렸는지 검사 (시험이 부른다)

조건은 `config.json` 에서, 정준값은 **artifacts json 에서** 읽는다.
숫자를 이 파일에 적지 않는다 — 적으면 갈린다 (`decisions_index.py` 와 같은
방식이다).

## 태그 규칙 (D-128 §1-7)

```
<피처><씨앗>-p<파라미터>[-<표현력>][-<실험명>]
  F3rw-p8   F3rw-p16   F3rw-p8-prod   F3hg-p8-d75-a
★ 표(GPU)·계승·코드 판은 태그에 안 넣는다 — config.json 이 갖는다
★ `x-` 로 시작하는 것은 **폐기**다 (순위 손실 계열 등). 표에 안 넣는다
```
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from kernelrule.core.runset import run_condition
from kernelrule.rules.checks import fitter_for

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "artifacts" / "runs.md"
BEGIN = "<!-- RUNS:BEGIN — experiments/runs_table.py 가 만든다 -->"
END = "<!-- RUNS:END -->"

#: 정준값이 **어느 산출물의 어디에** 있나. 값은 여기 안 적는다 (원칙 2).
#: `(파일, 키 경로, 집계)` — 집계 `med` 는 리스트의 중앙값.
CANON: dict[str, tuple[str, tuple, str]] = {
    # ★ 옛 정본. 새 정본(F3rw-p8)은 재측정이 끝나면 산출물이 생긴다
    "F3rw-p8-old": ("conclusion.json", ("f1_vs_human", "human_median"),
                    "one"),
    "F1rw-p8": ("conclusion.json", ("f1_vs_human", "f1_median"), "one"),
    # `f1k` 는 conclusion.json 안의 **옛 키**다 (개명 전 이름, D-128)
    "F2rw-p8": ("conclusion.json", ("f1k", "median"), "one"),  # D-128
    "F3rw-p8-cma": ("expressive-regret.json", ("r1", "rb08"), "med"),
    "F3rw-p16": ("expressive-regret.json", ("r1", "rb16"), "med"),
    "F3rw-p8-prod": ("expressive-regret.json", ("r1", "rprod"), "med"),
    "F3rw-p8-pow": ("expressive-regret.json", ("r1", "rpow"), "med"),
    "F3rw-p8-5090": ("c-ladder.json", ("regret", "5090sigma-hw2"), "med"),
    "F3rw-p8-4090": ("sigma-4090.json", ("holdout_regret",), "med"),
}
#: 표에서 빼는 접두. **지우지 않는다** — 이름으로 표시만 한다.
DROP = ("x-",)
#: ★ 폐기(재측정 대상) 태그와 이유. 표에 **상태로 남긴다** (D-129 §3-2).
RETIRED = {
    "F3rw-p8-cma": "p8 인데 CMA — 지금 규칙(fitter_for)으로는 안 나온다",
    "F3rw-p8-prod": "p8 인데 CMA. 재측정 대상",
    "F3rw-p8-pow": "p8 인데 CMA. 재측정 대상",
    "F3rw-p8-old": "옛 정본 — 옛 프롬프트·라운드12·patience10 (D-129)",
}


def _canon(tag: str) -> tuple[str, str]:
    """(값 문자열, 출처). 없으면 빈칸."""
    spec = CANON.get(tag)
    if spec is None:
        return "", ""
    f, path, how = spec
    p = ROOT / "docs" / "artifacts" / f
    if not p.exists():
        return "", ""
    v = json.loads(p.read_text())
    for k in path:
        if not isinstance(v, (dict, list)) or (
                isinstance(v, dict) and k not in v):
            return "", ""
        v = v[k]
    if how == "med":
        if not isinstance(v, list) or not v:
            return "", ""
        import statistics as st
        v = st.median(v)
    return f"{float(v):.4f}", f


def _groups() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in sorted(RUNS.iterdir()):
        m = re.match(r"^(.*)-s(\d+)$", d.name) if d.is_dir() else None
        if not m or not (d / "config.json").exists():
            continue
        if m.group(1).startswith(DROP):
            continue
        out.setdefault(m.group(1), []).append(d.name)
    return out


def _rows() -> list[dict]:
    rows = []
    for tag, runs in sorted(_groups().items()):
        conds = [run_condition(r) for r in runs]

        def one(key: str, _c=conds) -> str:
            """시드 전부가 같은 값인가. 갈리면 **표에 그렇게 적는다**."""
            vals = {str(c[key]) for c in _c}
            if len(vals) > 1:
                return "★갈림"
            v = vals.pop()
            # ★ 없는 것은 `?` 다. 추정해 채우지 않는다 (원칙 39).
            return "?" if v in ("None", "") else v

        cfg = json.loads((RUNS / runs[0] / "config.json").read_text())
        nr = sorted({sum(1 for _ in (RUNS / r / "rounds.jsonl").open())
                     for r in runs})
        hw = one("hw")
        gpu = {"37762692c36f5ba4": "4090", "cc392ddd72b4902d": "5090"}.get(
            hw, "a6000" if hw and hw != "None" else "?")
        val, src = _canon(tag)
        rows.append({
            "tag": tag, "n": len(runs),
            "features": str(cfg.get("n_features", "?")),
            "condition": one("feature_condition"),
            "seed": one("seed_source").split(" (")[0],
            "parameters": one("parameters"),
            "hint": ("곱" if one("product_hint") == "True" else "")
                    + ("지수" if one("power_hint") == "True" else "") or "기본",
            "fitter": f"{one('fit_method')}/{one('fit_restarts')}/"
                      f"{cfg['loop'].get('max_evals', '?')}",
            # ★ §1-6 의 규칙과 다른 실행인가 (p<=8 이면 nelder-mead)
            "off_rule": (one("parameters") not in ("?", "★갈림")
                         and one("fit_method") != fitter_for(
                             int(one("parameters")))["fit_method"]),
            "rounds": "~".join(map(str, nr)),
            "gpu": gpu, "canon": val, "canon_src": src,
            "retired": RETIRED.get(tag, ""),
            "objective": one("objective")})
    return rows


def render() -> str:
    rows = _rows()
    head = ("| 태그 | 시드 | 피처 | 씨앗 | 파라미터 | 표현력 | 적합기 | "
            "라운드 | 표 | 정준값 | 출처 | 상태 |")
    L = [BEGIN, "", head,
         "|---|--:|---|---|--:|---|---|---|---|--:|---|---|"]
    for r in rows:
        L.append(
            f"| `{r['tag']}` | {r['n']} | {r['features']}/{r['condition']} |"
            f" {r['seed']} | {r['parameters']} | {r['hint']} | {r['fitter']} |"
            f" {r['rounds']} | {r['gpu']} | {r['canon'] or '—'} |"
            f" {r['canon_src'] or '—'} |"
            + (f" ⛔ 폐기 — {r['retired']} |" if r["retired"]
               else " ⚠️ 적합기 규칙 밖 |" if r["off_rule"] else " |"))
    L += ["", END]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="갈렸으면 0 이 아닌 코드로 끝난다 (시험이 부른다)")
    a = ap.parse_args()
    body = render()
    txt = OUT.read_text() if OUT.exists() else ""
    if BEGIN in txt and END in txt:
        new = txt[:txt.index(BEGIN)] + body + txt[txt.index(END) + len(END):]
    else:
        new = txt.rstrip() + "\n\n" + body + "\n"
    if a.check:
        if new != txt:
            print("★ runs.md 가 실행 산출물과 갈렸다. "
                  "`python3 experiments/runs_table.py` 로 다시 만들어라.")
            sys.exit(1)
        print(f"runs.md 최신 ({len(_rows())}개 태그)")
        return
    OUT.write_text(new)
    print(f"runs.md 갱신 — {len(_rows())}개 태그")


if __name__ == "__main__":
    main()
