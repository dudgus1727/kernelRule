"""★ 새 번들을 **쓰기 전에** 확인한다 (D-89). LLM 0회.

    python3 experiments/bundle_guard.py datasets/<번들> --env-hash <해시>

## 왜 있나

kernelTab 이 캠페인 중에 노이즈 계수를 고치기로 했다. **번들 생성 전에
안 고쳐지면 A6000 계수가 5090 번들에 실린다.**

```
SIGMA_ABS_MS  5090 0.000016   ⛔ A6000 0.000374  (23배)
EVENT_TICK_MS 5090 0.000032   ⛔ A6000 0.001024  (32배)
```

**A6000 계수가 실리면 83us 형상에서 "구분 불가" 로 묶이는 후보가 크게
늘어난다** — 정답 집합이 넓어지고 순위 정보가 사라진다. 그리고 그것이
**조용히** 일어난다: 값이 있으므로 경고가 안 난다.

**받자마자 이것부터 돌리고, 틀렸으면 쓰지 않는다.**
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: A6000 값. **새 GPU 번들에 이 값이 있으면 잘못 실린 것이다.**
A6000_SIGMA_ABS_MS = 0.000374
A6000_TICK_MS = 0.001024
#: 허용 오차. JSON 의 십진 리터럴을 그대로 읽으므로 아주 작아도 된다.
#: ★ `abs(a-b) < tol` 을 직접 쓰지 않는다 — `inf` 에서 무너진다 (D-71).
#:   시험 `test_no_raw_float_comparison_outside_numerics` 가 이것을 잡았다.
_TOL = 1e-12


def _close(a: float, b: float) -> bool:
    from kernelrule.core.numerics import approx_equal

    return approx_equal(a, b, tol=_TOL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--env-hash", required=True)
    ap.add_argument("--expect-rows", type=int)
    ap.add_argument("--expect-tick-ms", type=float,
                    help="예고된 눈금 (5090 이면 0.000032)")
    ap.add_argument("--expect-sigma-abs-ms", type=float,
                    help="예고된 절대 노이즈 (5090 이면 0.000016)")
    a = ap.parse_args()

    root = Path(a.bundle)
    meta = json.loads((root / "BUNDLE.json").read_text())
    nf = meta.get("noise_floor") or {}
    bad: list[str] = []
    warn: list[str] = []

    print("=" * 72)
    print(f"번들 검사  {root.name}")
    print("=" * 72)
    print(f"  gpu       {meta.get('gpu_name')}  ({meta.get('arch')})")
    print(f"  env_hash  {str(meta.get('env_hash'))[:16]}")
    print(f"  행/형상/커널  {meta.get('n_rows')} / {meta.get('n_shapes')} / "
          f"{meta.get('n_kernels')}")
    print(f"  ridge     {meta.get('ridge_point')}")
    print(f"  noise     {json.dumps(nf, ensure_ascii=False)}")

    # 1) env_hash 가 요청과 맞는가 (§3.4)
    if not str(meta.get("env_hash", "")).startswith(a.env_hash):
        bad.append(f"env_hash 불일치: 요청 {a.env_hash!r}")

    # 2) ★ A6000 계수가 실렸는가
    sig = nf.get("sigma_abs_ms")
    tick = nf.get("tick_ms")
    if sig is not None and _close(float(sig), A6000_SIGMA_ABS_MS):
        bad.append(f"★ sigma_abs_ms 가 A6000 값 {A6000_SIGMA_ABS_MS} 다 — "
                   "노이즈 계수가 안 고쳐진 채 실렸다")
    if tick is not None and _close(float(tick), A6000_TICK_MS):
        bad.append(f"★ tick_ms 가 A6000 값 {A6000_TICK_MS} 다 — "
                   "눈금이 안 고쳐진 채 실렸다")
    if tick is None:
        warn.append("tick_ms 가 번들에 없다 (schema_version 1) — "
                    "대체값이 쓰이고 그것은 A6000 눈금이다")

    # 3) 예고값과 맞는가
    if a.expect_tick_ms is not None and tick is not None \
            and not _close(float(tick), a.expect_tick_ms):
        bad.append(f"tick_ms {tick} != 예고 {a.expect_tick_ms}")
    if a.expect_sigma_abs_ms is not None and sig is not None \
            and not _close(float(sig), a.expect_sigma_abs_ms):
        bad.append(f"sigma_abs_ms {sig} != 예고 {a.expect_sigma_abs_ms}")
    if a.expect_rows is not None and meta.get("n_rows") != a.expect_rows:
        warn.append(f"행 수 {meta.get('n_rows')} != 예고 {a.expect_rows}")

    # 4) 노이즈 출처가 이 GPU 인가
    src_gpu = str(nf.get("gpu", ""))
    if src_gpu and meta.get("gpu_name") and src_gpu != meta["gpu_name"]:
        bad.append(f"★ 노이즈 출처 GPU 가 다르다: {src_gpu!r} vs "
                   f"{meta['gpu_name']!r}")

    print()
    for w in warn:
        print(f"  ⚠️  {w}")
    for b in bad:
        print(f"  ⛔ {b}")
    if bad:
        print("\n  ★ 이 번들을 쓰지 마라. kernelTab 에 보고하고 다시 받아라.")
        sys.exit(1)
    print("\n  ✅ 통과 — 써도 된다")


if __name__ == "__main__":
    main()
