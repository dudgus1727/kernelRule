"""★ 옛 산출물의 역할 이름을 한 번에 옮긴다 (D-93). LLM 0회.

    python3 experiments/rename_roles.py --check     # 세기만
    python3 experiments/rename_roles.py --apply     # 옮긴다

## 왜 alias 를 안 두나

옛 이름을 읽는 호환 경로를 만들면 **두 이름이 공존하고 그것이 달라진다.**
`is_reference` / `top_k` / `DEFAULT_MODEL` / `REGISTRY` /
`load_generated` / `approx_equal` / 예산 상수에 이은 여덟 번째가 된다
(원칙 2). 그래서 **읽는 쪽을 고치지 않고 자료를 옮긴다.**

## ★ 옮긴 뒤 집계를 대조한다

역할별 호출 수의 **합**은 이름을 바꿔도 안 변한다. 안 맞으면 못 옮긴
자리가 있다는 뜻이다 — 조용히 지나가면 `cost.md` 가 틀린다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

#: 옛 이름 -> 새 이름. **한 곳에서만 적는다.**
#: ⚠️ 이 표는 **옛 이름 -> 새 이름**이다. 한 번 이 파일이 자기 자신의
#: 일괄 치환에 걸려 항등 사상이 된 적이 있다 (`{"rule_writer":
#: "rule_writer"}`). 그러면 "남은 옛 이름 0" 이 거짓이 된다 —
#: `test_rename_map_is_not_identity` 가 그것을 고정한다.
RENAME = {"architect": "rule_writer", "optimize": "rule_editor"}


def _role_counts(root: Path) -> Counter:
    """`llm_calls/*.json` 의 역할별 호출 수. 이름을 정규화해서 센다."""
    c: Counter = Counter()
    for f in root.glob("*/llm_calls/*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        r = str(d.get("role", ""))
        c[RENAME.get(r, r)] += 1
    return c


#: 산출물 디렉토리 이름. 역할 이름이 경로에 박혀 있어 새 실행도 옛 이름을
#: 쓰게 된다 — 그것이 바로 "두 이름 공존" 이다.
DIR_RENAME = {"stage2-architect": "stage2-rule-writer"}


def _move_dirs(root: Path, apply: bool) -> int:
    n = 0
    for old, new in DIR_RENAME.items():
        for d in sorted(root.glob(f"*/{old}")):
            n += 1
            if apply:
                d.rename(d.parent / new)
    return n


def _walk(root: Path):
    yield from root.glob("*/llm_calls/*.json")
    yield from root.glob("*/config.json")
    yield from root.glob("*/rounds.jsonl")
    yield from root.glob("*/stage2-*/summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)

    before = _role_counts(root)
    print("=" * 68)
    print(f"역할 이름 이전  {root}   {'(세기만)' if a.check else '(적용)'}")
    print("=" * 68)
    print(f"  이전 집계(정규화): {dict(before)}")

    n_files = 0
    hits: Counter = Counter()
    for f in _walk(root):
        txt = f.read_text()
        found = {k: txt.count(f'"{k}"') for k in RENAME
                 if f'"{k}"' in txt}
        if not found:
            continue
        n_files += 1
        hits.update(found)
        if a.apply:
            for old, new in RENAME.items():
                txt = txt.replace(f'"{old}"', f'"{new}"')
            f.write_text(txt)
    print(f"  옛 이름이 든 파일 {n_files}개   {dict(hits)}")
    n_dirs = _move_dirs(root, a.apply)
    print(f"  옛 이름 디렉토리 {n_dirs}개 {DIR_RENAME}")

    if a.apply:
        after = _role_counts(root)
        # ★ 집계 대조. 이름만 바뀌었으므로 **합이 같아야 한다.**
        ok = before == after
        print(f"  이후 집계:         {dict(after)}")
        print("  ★ 집계 대조: " + ("일치" if ok else "★불일치 — 못 옮긴 자리가 있다"))
        if not ok:
            raise SystemExit(1)
        left = sum(1 for f in _walk(root)
                   for k in RENAME if f'"{k}"' in f.read_text())
        print(f"  남은 옛 이름: {left}")
        if left:
            raise SystemExit(1)
    print("\n  ★ alias 를 두지 않는다 — 두 이름이 공존하면 달라진다 (원칙 2)")


if __name__ == "__main__":
    main()
