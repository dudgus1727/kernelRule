"""노이즈 바닥 — kernelTab 에 위임하되 **번들 계수를 강제한다** (§30, §4-3).

## 재구현하지 않는다

모델(`max(a/t + b, tick/t)`)은 `kerneltab.core.noise` 에 있다. 여기서 다시
쓰지 않는다. 이 파일이 하는 일은 **어느 계수가 쓰이는지를 붙잡아 두는 것**뿐이다.

## 왜 래퍼가 필요한가 — 실제로 발견한 fail-open 경로

kernelTab 에는 노이즈 진입점이 **둘**이고 성격이 다르다.

    kerneltab.core.noise.noise_floor(t)   모듈 전역 상수 (A6000 값이 박혀 있다)
    Bundle.noise_floor                    번들 계수로 만든 함수 + tick 부재 시 경고

그리고 `kerneltab.core.table.answer_set()` 은 **전자를 호출한다.** 지금은 개발용
번들이 A6000 이라 값이 같아 문제가 없지만, 4090/H100 번들이 오면 `answer_set()`
이 A6000 눈금을 조용히 쓴다. 경고도 안 난다 — `Bundle.tick_ms` 를 안 거치니까.

이건 §30.8 이 말한 "조용히 아무것도 안 하는 안전장치" 와 같은 클래스다.
kernelTab 을 고치지 않고(우리 범위 밖) 이쪽에서 **불일치를 에러로 만든다.**

    NoiseModel.from_bundle(b)  ->  번들 계수를 읽고
                                   모듈 전역 상수와 대조하고
                                   다르면 에러  (조용한 진행 금지)

## ★ 2026-08-31 — kernelTab 이 그 구멍을 막았다. 검사를 바꾼다

**위 서술은 정정 이력으로 남긴다.** 5090 번들(`5bb6f403`)을 처음 열 때
위 대조가 걸려서 확인해 보니 `kerneltab.core.table.answer_set()` 이
더는 모듈 전역을 안 쓴다 — `noise` 주입이 **필수**이고 없으면
`NoiseCoefRequired` 를 던진다. 모듈 전역 `noise_floor()` 함수 자체가
없어졌다. kernelTab 쪽 docstring 이 우리가 예측한 실패를 그대로 적고
있다 ("4090/H100 번들을 채점할 때도 A6000 눈금을 경고 없이 쓴다").

그러면 **계수 불일치는 더 이상 위험이 아니다.** 다른 GPU 번들은
정의상 A6000 상수와 다르므로, 옛 검사를 그대로 두면 **A6000 이 아닌
모든 번들을 영구히 막는다** — 안전장치가 아니라 통행 금지다.

```
옛 검사   번들 계수 == 모듈 전역 ?          -> 다른 GPU 는 영원히 실패
새 검사   ★ answer_set 이 주입을 요구하나 ?  -> 구멍이 살아 있으면 실패
          계수 불일치는 **사실로 기록**한다 (경고 + source 에 남긴다)
```

★ 검사를 **능력 검사**로 바꾼다. "값이 같은가" 가 아니라 "위험한 경로가
막혀 있는가" 다. 구멍이 되살아나면(누가 기본값을 다시 넣으면) 새 검사가
잡는다 — 옛 검사는 못 잡았다 (값이 같은 A6000 에서는 통과했으니까).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["NoiseModel", "NoiseMismatchError"]


class NoiseMismatchError(RuntimeError):
    """번들 계수와 kernelTab 모듈 전역 상수가 다르다.

    그 상태에서는 `kerneltab.core.table.answer_set()` 이 **틀린 계수로**
    돈다 (모듈 전역을 쓰기 때문). 우리 쪽 계산만 고쳐도 소용없으므로 멈춘다.
    """


def _require_injected_noise() -> None:
    """★ `answer_set`/`answer_tolerance` 가 계수 주입을 **요구하는가**.

    요구하지 않으면 기본값(= A6000 눈금)으로 조용히 돈다. 그것이 §30.8 이
    말한 "조용히 아무것도 안 하는 안전장치" 다.

    옛 검사(번들 계수 == 모듈 전역)는 이것을 **못 잡았다** — 개발용
    A6000 번들에서는 값이 같아서 통과했고, 정작 다른 GPU 에서는 위험이
    없는데도 막았다. 방향이 둘 다 틀렸다.
    """
    from kerneltab.core.noise import NoiseCoefRequired
    from kerneltab.core.table import answer_tolerance

    try:
        answer_tolerance(1.0)
    except NoiseCoefRequired:
        return
    except Exception as e:                                  # noqa: BLE001
        raise NoiseMismatchError(
            "kerneltab.core.table.answer_tolerance 가 예상 밖으로 실패한다: "
            f"{type(e).__name__}: {e}") from e
    raise NoiseMismatchError(
        "★ kerneltab.core.table.answer_tolerance() 가 노이즈 계수 주입 "
        "없이도 값을 돌려준다 — 기본값이 되살아났다.\n"
        "  그 기본값은 A6000 눈금이고, 다른 GPU 번들을 채점하면 정답 "
        "집합이 조용히 틀린다 (5090 눈금은 A6000 의 1/64 이다).\n"
        "  kernelTab 쪽을 확인하라. 우리 채점은 "
        "PerfTable.answer_mask() 로 안전하지만, 이 상태에서는 "
        "kernelTab 헬퍼를 쓰는 다른 코드가 조용히 틀린다.")


@dataclass(frozen=True, slots=True)
class NoiseModel:
    """이 측정 조건의 노이즈 바닥. **계수의 출처를 들고 다닌다.**

    `sigma_rel` 과 `noise_floor` 를 **구분해서** 노출한다 (§30.3).
    한 함수로 뭉개면 정답 집합(분해능 필요)과 노이즈 모델 검증(통계만)이
    같은 값을 쓰게 된다.
    """

    sigma_abs_ms: float
    sigma_rel_coef: float
    tick_ms: float
    #: 계수가 어디서 왔는가. 리포트와 재현에 필요하다.
    source: str
    #: `tick_ms` 가 번들에 없어 대체값이 쓰였는가 (schema_version 1).
    tick_is_fallback: bool = False

    # -- 두 항을 따로 노출한다 (§30.3) ------------------------------------
    def sigma(self, t):
        """**통계적** 노이즈만. `a/t + b`. 관측 산포와 비교할 때 쓴다.

        반복 측정하면 평균으로 줄어드는 성분이다.
        """
        t = np.asarray(t, dtype=np.float64)
        out = np.where(t > 0, self.sigma_abs_ms / np.where(t > 0, t, 1.0)
                       + self.sigma_rel_coef, self.sigma_rel_coef)
        return float(out) if out.ndim == 0 else out

    def tick_pct(self, t):
        """**분해 한계**. `tick/t`. 반복 측정해도 줄지 않는다.

        같은 눈금에 떨어진 두 config 는 시간이 문자 그대로 동일하게 기록된다.
        """
        t = np.asarray(t, dtype=np.float64)
        out = np.where(t > 0, self.tick_ms / np.where(t > 0, t, 1.0), np.inf)
        return float(out) if out.ndim == 0 else out

    def floor(self, t):
        """두 측정값을 **구분할 수 있는 최소 상대 차이**. `max(통계, 분해능)`.

        ⚠️ 계산할 수 없으면 **보수적으로 큰 값**을 낸다 (§26.4). 0 이 아니다 —
        0 을 내면 "모든 차이가 유의하다" 가 되어 노이즈를 전부 신호로 배운다.
        """
        t = np.asarray(t, dtype=np.float64)
        bad = ~(np.isfinite(t) & (t > 0))
        safe = np.where(bad, 1.0, t)
        out = np.maximum(self.sigma_abs_ms / safe + self.sigma_rel_coef,
                         self.tick_ms / safe)
        # 시간을 모르면 분해 불가로 본다. 1.0 = 100%, 즉 "아무것도 구분 못 함".
        out = np.where(bad, 1.0, out)
        return float(out) if out.ndim == 0 else out

    def answer_tol(self, best_ms) -> float:
        """정답 집합 허용치. `2 * floor(best)` (§30.3 의 2σ 근거).

        1σ 면 노이즈로 정답이 오답이 되는 경우가 32%, 2σ 면 5% 다.
        허용치는 **과대평가가 과소평가보다 안전**하다 — 과대평가하면 규칙에
        관대해질 뿐이지만, 과소평가하면 노이즈를 신호로 배운다.
        """
        return 2.0 * float(self.floor(best_ms))

    def resolvable(self, t_a, t_b, k: float = 2.0):
        """두 측정값의 차이가 노이즈로 설명되지 않는가."""
        a = np.asarray(t_a, dtype=np.float64)
        b = np.asarray(t_b, dtype=np.float64)
        s = k * np.sqrt((self.floor(a) * a) ** 2 + (self.floor(b) * b) ** 2)
        out = np.abs(a - b) > s
        return bool(out) if out.ndim == 0 else out

    # -- 생성 -------------------------------------------------------------
    @classmethod
    def from_bundle(cls, bundle, *, strict: bool = True) -> NoiseModel:
        """번들의 `noise_floor` 계수로 만든다. **모듈 전역과 대조한다.**

        `strict=True` (기본) 면 불일치 시 에러다. 위 docstring 의 fail-open
        경로 때문이다 — `answer_set()` 이 모듈 전역을 쓰므로, 불일치는
        "우리 계산은 맞고 kernelTab 헬퍼는 틀린" 상태를 뜻한다.
        """
        from kerneltab.core import noise as kt

        info = getattr(bundle, "info", None) or {}
        coefs = dict(info.get("noise_floor") or {})
        if not coefs:
            raise NoiseMismatchError(
                f"번들 {info.get('bundle_id')} 에 noise_floor 계수가 없다. "
                "기본값으로 때우지 않는다 (§26.4).")

        missing = [k for k in ("sigma_abs_ms", "sigma_rel") if k not in coefs]
        if missing:
            raise NoiseMismatchError(
                f"번들 noise_floor 에 {missing} 가 없다. 계수를 추정하지 않는다.")

        # tick_ms 는 schema_version 1 에 없다. Bundle.tick_ms 가 대체하며
        # **경고한다** — 그 경고를 삼키지 않고 여기서 사실로 기록한다 (§30.3b).
        tick_present = bool(coefs.get("tick_ms"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tick = float(bundle.tick_ms)
        if not tick_present:
            for w in caught:
                warnings.warn(
                    f"[kernelRule] 번들 tick_ms 대체값 사용 — {w.message}",
                    stacklevel=2)

        model = cls(
            sigma_abs_ms=float(coefs["sigma_abs_ms"]),
            sigma_rel_coef=float(coefs["sigma_rel"]),
            tick_ms=tick,
            source=(f"BUNDLE.json:{info.get('bundle_id')}"
                    f"{'' if tick_present else ' (tick_ms=대체값)'}"),
            tick_is_fallback=not tick_present,
        )

        # ★ **능력 검사** — 위험한 경로가 막혀 있는가 (2026-08-31).
        #   값 대조가 아니다. 값은 다른 GPU 면 당연히 다르다.
        if strict:
            _require_injected_noise()

        # 계수 불일치는 **사실로 기록**한다. 막지는 않는다 — 다른 GPU
        # 번들은 정의상 A6000 상수와 다르다.
        drift = {
            "sigma_abs_ms": (model.sigma_abs_ms, kt.SIGMA_ABS_MS),
            "sigma_rel": (model.sigma_rel_coef, kt.SIGMA_REL),
            "tick_ms": (model.tick_ms, kt.EVENT_TICK_MS),
        }
        bad = {k: v for k, v in drift.items()
               if abs(v[0] - v[1]) > 1e-12 * max(1.0, abs(v[1]))}
        if bad:
            warnings.warn(
                f"[kernelRule] 번들 계수가 kerneltab.core.noise 의 A6000 "
                f"참조값과 다르다 (정상 — 다른 GPU다): "
                f"{ {k: v[0] for k, v in bad.items()} }. "
                "채점은 번들 계수를 쓴다 (PerfTable.answer_mask).",
                stacklevel=2)
        return model

    @classmethod
    def a6000_reference(cls) -> NoiseModel:
        """A6000 c63710df 실측 계수. **합성 표와 테스트 전용.**

        실제 번들에는 `from_bundle` 을 써라 — 다른 GPU 에 이 값을 쓰면 틀린다.
        """
        from kerneltab.core import noise as kt

        return cls(sigma_abs_ms=kt.SIGMA_ABS_MS, sigma_rel_coef=kt.SIGMA_REL,
                   tick_ms=kt.EVENT_TICK_MS,
                   source="kerneltab.core.noise (A6000 c63710df 실측)")
