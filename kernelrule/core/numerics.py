"""부동소수 비교 — **한 곳에서만** 한다 (§30.13).

## 왜 모으나

`abs(a - b) < tol` 은 `inf` 와 `nan` 에서 무너진다.

```python
abs(inf - inf)          # nan
nan < 1e-9              # False  -> "다르다" 로 판정된다
```

실제로 `expected_range` 대조가 그 함정에 빠졌다 —
`global_tile_traffic_amplification` 의 선언·레지스트리·프롬프트가 전부
`[1, inf]` 인데 "다름" 으로 찍혔다 (D-71). **`expected_range` 에 `inf` 가
실제로 들어 있으므로 우연이 아니다.**

그리고 이 저장소에는 같은 비교가 여럿이다.

```
verify_rules       기록값 vs 재계산값
physics_coverage   상관 임계
expected_range     선언 대조
기여도             0 판정
정답 집합          노이즈 바닥
shape_level        상대 분산
```

**같은 판정이 여러 곳에 있으면 갈린다** (원칙 2). `is_reference` /
`top_k` / `DEFAULT_MODEL` / `REGISTRY` / `load_generated` 에 이은
여섯 번째다.
"""

from __future__ import annotations

import math

__all__ = ["approx_equal", "approx_zero", "DEFAULT_TOL"]

#: 기본 허용 오차. 자리별로 다르면 호출부가 명시한다.
DEFAULT_TOL = 1e-9


def approx_equal(a: float, b: float, tol: float = DEFAULT_TOL) -> bool:
    """`a == b` 를 허용 오차 안에서. **`inf`/`nan` 을 명시적으로 다룬다.**

    ```
    approx_equal(inf, inf)    True    ★ abs(inf-inf) 는 nan 이다
    approx_equal(inf, -inf)   False
    approx_equal(nan, nan)    False   ★ nan 은 자기 자신과도 다르다
    ```

    마지막이 의도다 — `nan` 이 나왔다는 것은 계산이 깨졌다는 뜻이고,
    "같다" 로 넘기면 그 사실이 사라진다 (§26.4).
    """
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return False
    if a == b:                  # inf == inf, -inf == -inf 를 여기서 잡는다
        return True
    if math.isinf(a) or math.isinf(b):
        return False            # 하나만 무한대면 다르다
    return abs(a - b) <= tol


def approx_zero(x: float, tol: float = DEFAULT_TOL) -> bool:
    """`x == 0` 을 허용 오차 안에서. `nan` 은 0 이 아니다."""
    return approx_equal(float(x), 0.0, tol)
