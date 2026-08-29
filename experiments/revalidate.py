"""★ 거부된 제안을 **고친 검사기로 다시 검사한다**. LLM 0회.

    python3 experiments/revalidate.py runs/f1pipe-F1-K-k1/stage1-features

## 왜 재제안이 아니라 재검사인가

```
재제안 (LLM N회)   그 영역만 제안 횟수가 늘어 **조건이 달라진다**
★ 재검사 (LLM 0회)  같은 코드를 고친 검사기로. 조건이 안 바뀐다
```

**검사기가 버린 것은 LLM 이 못 만든 것이 아니다.** D-37/D-38 때도
그렇게 했다 — `inspect.getsource` 결함으로 버려진 것을 고치고 다시
통과시켰다.

**되살아난 것과 여전히 거부되는 것을 분리해 기록한다.**
"검사기 결함으로 버려졌다" 와 "정말 안 된다" 는 다른 사실이다.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.table import PerfTable
from kernelrule.features import FeatureRegistry
from kernelrule.features.generated import FeatureRejected, register_generated
from kernelrule.features.loader import load_generated
from kernelrule.features.validate import alt_hw

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"




def main(stage1: str) -> None:
    warnings.simplefilter("ignore")
    d = Path(stage1)
    path = d / "proposals.jsonl"
    rows = [json.loads(ln) for ln in path.open() if ln.strip()]
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)

    # ★ 채택분을 먼저 넣는다 — 중복 판정이 그것들과도 이뤄져야 한다.
    gen = FeatureRegistry("revalidate")
    for f in load_generated(path, table=table, exclude=set()):
        gen.add(f)
    n_before = len(gen._items)
    matrix = FeatureMatrix(table, gen) if gen._items else None
    hw_alt = alt_hw(table.hw)

    print("=" * 78)
    print(f"재검사 — {stage1}   (LLM 0회)")
    print("=" * 78)
    print(f"  기존 채택 {n_before}개.  거부 "
          f"{sum(1 for r in rows if not r.get('accepted'))}건을 다시 본다\n")

    revived, still = [], []
    for r in rows:
        if r.get("accepted") or not r.get("code"):
            continue
        name = r.get("name") or "?"
        try:
            f = register_generated(
                r["code"], registry=gen,
                meta={k: r.get(k) for k in
                      ("name", "unit", "direction", "expected_range")},
                table=table,
                matrix=matrix or FeatureMatrix(table, FeatureRegistry("e")),
                hw_alt=hw_alt)
            revived.append(f.name)
            print(f"  ★ 되살아남  {f.name}")
            print(f"       전: {str(r.get('error'))[:80]}")
        except FeatureRejected as e:
            still.append((name, str(e)[:90]))
            print(f"     여전히 거부  {name}")
            print(f"       {str(e)[:88]}")
        except Exception as e:                              # noqa: BLE001
            still.append((name, f"{type(e).__name__}: {e}"[:90]))
            print(f"     여전히 거부  {name}  {type(e).__name__}")

    print()
    print(f"  ★ 되살아난 것 {len(revived)}: {revived}")
    print(f"     여전히 거부 {len(still)}")
    print(f"     라이브러리 {n_before} -> {len(gen._items)}")

    if revived:
        # ★ 되살아난 것을 **별도 파일**로 남긴다. `proposals.jsonl` 을
        #   덮어쓰면 "그때 무엇이 거부됐는지" 가 사라진다 (문서 규칙 2).
        out = d / "revalidated.jsonl"
        with out.open("w") as fh:
            for r in rows:
                if r.get("name") in revived:
                    fh.write(json.dumps({**r, "accepted": True,
                                         "revalidated": True,
                                         "original_error": r.get("error")},
                                        ensure_ascii=False) + "\n")
        print(f"\n  기록: {out}  (원본 proposals.jsonl 은 그대로 둔다)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "runs/f1pipe-F1-K-k1/stage1-features")
