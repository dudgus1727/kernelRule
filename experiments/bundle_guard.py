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


def _close_at(value: float, expect_str: str) -> tuple[bool, float]:
    """★ **예고값의 자릿수까지** 맞는가 (D-125).

    릴리즈 노트는 반올림한 값을 적는다 — 4090 은 `sigma_abs 0.000743` 인데
    번들은 `0.0007433368963633708` 이다. `tol=1e-12` 로 견주면 **멀쩡한
    번들이 거부된다** (실제로 그랬다).

    허용오차를 예고 문자열의 소수 자릿수에서 만든다: `0.000743` 이면
    반올림 폭 5e-7. **느슨하게 푸는 것이 아니라 예고된 정밀도까지만
    본다** — 자릿수를 더 주면 더 엄격해진다. A6000 계수 검사(`_close`)는
    그대로 엄격하다.
    """
    t = expect_str.strip().lower()
    if "e" in t:
        mant, _, exp = t.partition("e")
        dec = len(mant.partition(".")[2]) - int(exp)
    else:
        dec = len(t.partition(".")[2])
    tol = 0.5 * 10.0 ** (-dec) if dec > 0 else 0.5
    from kernelrule.core.numerics import approx_equal

    return approx_equal(value, float(t), tol=tol), tol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--env-hash", required=True)
    ap.add_argument("--expect-rows", type=int)
    # ★ `type=float` 로 받지 않는다 — **적어 준 자릿수**가 허용오차다
    #   (D-125). 문자열로 받아야 "0.000743" 의 정밀도를 알 수 있다.
    ap.add_argument("--expect-tick-ms",
                    help="예고된 눈금 (5090 이면 0.000032). ★ 적어 준 "
                         "자릿수까지 본다 — 릴리즈 노트는 반올림값이다")
    ap.add_argument("--expect-sigma-abs-ms",
                    help="예고된 절대 노이즈. ★ 적어 준 자릿수까지 본다")
    # ★ schema_version 3 에서 새로 생긴 것들. 예고값이 있으면 맞춘다.
    ap.add_argument("--expect-env-hash-v2",
                    help="예고된 env_hash_v2 (접두 일치)")
    ap.add_argument("--expect-schema-version", type=int)
    ap.add_argument("--expect-aggregate-status",
                    help="'all' 이면 status 필터 없이 전부 집계됐다는 뜻")
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
    print(f"  schema    v{meta.get('schema_version')}  "
          f"aggregate_status={meta.get('aggregate_status')!r}  "
          f"env_hash_v2={str(meta.get('env_hash_v2'))[:16]}")
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
    for name, got, want in (("tick_ms", tick, a.expect_tick_ms),
                            ("sigma_abs_ms", sig, a.expect_sigma_abs_ms)):
        if want is None or got is None:
            continue
        ok, tol = _close_at(float(got), want)
        if ok:
            print(f"  ✓ {name} {got} = 예고 {want} (자릿수 허용 ±{tol:g})")
        else:
            bad.append(f"{name} {got} != 예고 {want} "
                       f"(자릿수 허용 ±{tol:g})")
    if a.expect_rows is not None and meta.get("n_rows") != a.expect_rows:
        warn.append(f"행 수 {meta.get('n_rows')} != 예고 {a.expect_rows}")

    # 3-b) schema_version 3 의 새 항목들
    if a.expect_env_hash_v2 and not str(
            meta.get("env_hash_v2", "")).startswith(a.expect_env_hash_v2):
        bad.append(f"env_hash_v2 불일치: {str(meta.get('env_hash_v2'))[:16]!r}"
                   f" != 예고 {a.expect_env_hash_v2[:16]!r}")
    if a.expect_schema_version is not None \
            and meta.get("schema_version") != a.expect_schema_version:
        bad.append(f"schema_version {meta.get('schema_version')} != 예고 "
                   f"{a.expect_schema_version}")
    if a.expect_aggregate_status is not None \
            and meta.get("aggregate_status") != a.expect_aggregate_status:
        # ★ 거부다. `aggregate_status` 가 'ok' 면 **정답 집합이 다르다** —
        #   5090 은 status != ok 가 22.21% 라 A6000(10.65%) 보다 훨씬 크고,
        #   그것을 버린 표와 전부 담은 표는 나란히 못 놓는다 (D-91 계열).
        bad.append(f"★ aggregate_status {meta.get('aggregate_status')!r} != "
                   f"예고 {a.expect_aggregate_status!r} — 정답 집합이 다르다")

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
