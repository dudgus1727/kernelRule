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

        # ★ 모듈 전역과 대조. answer_set() 이 그쪽을 쓰기 때문이다.
        drift = {
            "sigma_abs_ms": (model.sigma_abs_ms, kt.SIGMA_ABS_MS),
            "sigma_rel": (model.sigma_rel_coef, kt.SIGMA_REL),
            "tick_ms": (model.tick_ms, kt.EVENT_TICK_MS),
        }
        bad = {k: v for k, v in drift.items()
               if abs(v[0] - v[1]) > 1e-12 * max(1.0, abs(v[1]))}
        if bad and strict:
            raise NoiseMismatchError(
                f"번들 계수와 kerneltab.core.noise 모듈 상수가 다르다: {bad}\n"
                "  kerneltab.core.table.answer_set() 은 **모듈 전역**을 쓰므로 "
                "이 상태에서 채점하면 정답 집합이 틀린 계수로 만들어진다.\n"
                "  이 번들의 계수로 answer_set 을 계산하려면 "
                "kernelrule.core.table.PerfTable.answer_mask() 를 써라 — "
                "그쪽은 이 NoiseModel 을 쓴다.")
        if bad:
            warnings.warn(f"노이즈 계수 불일치 (strict=False): {bad}",
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
