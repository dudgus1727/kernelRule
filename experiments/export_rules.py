"""최종 규칙과 **적합된** 가중치를 저장소로 내보낸다.

    python3 experiments/export_rules.py

## 왜 필요한가

`runs/` 는 `.gitignore` 다. 그래서 **규칙 파일이 저장소에 없고**, 문서의
숫자를 대조할 방법이 없었다 — 의도적 조작이 아니라 **전사 오류나 반올림이
섞여도 아무도 모른다.**

```
LLM 실행    재현 불가 (난수 통제 안 됨 — §24.4b)
채점·재채점  ★ 완전히 결정론적. 몇 초
```

**후자는 검증 가능하다.** 규칙과 가중치를 커밋하면 누구나
`experiments/verify_rules.py` 로 문서의 숫자를 몇 초에 확인한다.

## 무엇을 쓰는가

```
docs/artifacts/rules/<run>.py      score() + W_FITTED (체제별)
docs/artifacts/rules/index.json    기계가 쓴 점수. 문서가 참조한다
```

⚠️ `W_FITTED` 는 **적합된** 값이어야 한다. 초기값을 적으면 파일이
거짓말을 한다 — 처음에 실제로 그렇게 썼다가 잡았다.
"""

from __future__ import annotations

import json
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.canonical import canonical_score
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet
from kernelrule.core.table import PerfTable
from kernelrule.features import REGISTRY

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
OUT = Path("docs/artifacts/rules")
#: 내보낼 실행. **지시된 모델의 것만** 넣는다 (D-52).
PREFIXES = ("luna-", "lunaNAMES-")


def setup():
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    shapes = [p for p in table.shapes() if aligned(p)]
    held = [p for p in shapes if 11008 in (p.N, p.K)]
    splits = SplitSet(
        train=Split("train", tuple(p for p in shapes if p not in held)),
        val=Split("val", tuple(held)), kind="nk11008")
    return table, FeatureMatrix(table, REGISTRY), splits


def main() -> None:
    table, matrix, splits = setup()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(d.name for d in Path("runs").iterdir()
                  if d.is_dir() and d.name.startswith(PREFIXES))
    if not runs:
        raise SystemExit(f"내보낼 실행이 없다 (접두사 {PREFIXES}).")

    index = []
    for run in runs:
        d = Path("runs") / run
        with (d / "archive.jsonl").open() as fh:
            arc = [json.loads(ln) for ln in fh if ln.strip()]
        best = min(arc, key=lambda e: e["regret"])
        w0 = [round(float(x), 6) for x in best["w"]]
        r = canonical_score(best["code"], best["w"], table=table,
                            matrix=matrix, splits=splits)
        cfg = {}
        if (d / "config.json").exists():
            cfg = json.loads((d / "config.json").read_text()).get("llm", {})

        w = "\n".join(f"    {k!r}: {[round(x, 6) for x in v]},"
                      for k, v in sorted(r.weights.items()))
        (OUT / f"{run}.py").write_text(
            f'"""{run} 의 최종 규칙 — 아카이브에서 **학습 점수 최소**.\n\n'
            f'모델      {cfg.get("model", "?")} / {cfg.get("endpoint", "?")}\n'
            f'추론      {cfg.get("reasoning_effort", "?")}\n'
            f'피처 표시  {cfg.get("feature_detail", "?")}\n\n'
            f'구조 홀드아웃 {r.holdout:.4f}  (표본내 {r.in_sample:.4f})\n\n'
            f'★ `W_FITTED` 는 **체제별로 적합된** 값이다. 초기값이 아니다.\n'
            f'재현:  python3 experiments/verify_rules.py\n"""\n\n'
            "import numpy as np  # noqa: F401\n\n"
            + best["code"].strip() + "\n\n\n"
            + "W_FITTED = {\n" + w + "\n}\n")

        index.append({
            "run": run, "in_sample": round(r.in_sample, 6),
            "holdout": round(r.holdout, 6),
            "by_regime": {k: round(v, 6) for k, v in r.by_regime.items()},
            "n_holdout": r.n_holdout, "llm": cfg,
            "weights": {k: [round(x, 6) for x in v]
                        for k, v in r.weights.items()},
            # ★ 적합기가 실제로 움직였는가 (D-54). 안 움직이면 §29 의
            #   "구조를 공정하게 비교한다" 가 성립하지 않는다.
            "w_moved": {k: [round(x, 6) for x in v] != w0
                        for k, v in r.weights.items()}})
        moved = sum(1 for k, v in r.weights.items()
                    if [round(x, 6) for x in v] != w0)
        print(f"  {run:16s} 표본내 {r.in_sample:.4f}  구조HO {r.holdout:.4f}"
              f"  적합기 움직임 {moved}/2")

    (OUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    print(f"\n  {len(index)}개 -> {OUT}/  (+ index.json)")
    print("  검증:  python3 experiments/verify_rules.py")


if __name__ == "__main__":
    main()
