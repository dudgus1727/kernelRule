"""피처 자동 검증 (§8.3). **예외를 삼키고 승인하지 않는다** (§26.4).

일곱 가지를 본다.

    1. 실행되는가          표본에서 유한한 float
    2. 범위가 선언대로인가
    3. 벡터화가 스칼라와 같은가   ★ 학습(행렬)과 배포(스칼라)가 갈리는 지점
    4. 상수인가            std < 1e-9 -> 기각 (설명력이 0이다)
    5. 중복인가            기존 피처와 스피어만 > 0.95 -> 폐기 후보
    6. 스케일 불변성       ★ hw 를 바꿨는데 값이 안 변하면 하드웨어를 안 쓴다
    7. 유용한가            단독 랭킹 AUC — **기각은 안 하고 표시만**

**6번이 핵심이다.** 하드웨어 상수를 하드코딩한 피처를 잡는 유일한 자동 검사이고,
아키텍처 전이가 이 프로젝트의 주 지표이므로 여기서 새면 결론이 무너진다.

검사 결과의 처리:

    fail   -> 기각. 레지스트리에 넣지 않는다
    warn   -> 통과시키되 표시. 사람이 본다 (30개 중 3~5개)
    info   -> 기록만
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kernelrule.core.types import Hardware, config_from_row
from kernelrule.features import Feature, FeatureRegistry

__all__ = ["Check", "ValidationReport", "validate_feature", "validate_registry"]

#: 스피어만 상관이 이보다 크면 중복 후보 (§8.4)
DUP_RHO = 0.95
#: 표준편차가 이보다 작으면 상수
CONST_STD = 1e-9
#: `hw` 를 바꿨을 때 값이 이보다 덜 변하면 하드웨어를 안 쓰는 것
SCALE_MIN_CHANGE = 1e-9


@dataclass
class Check:
    name: str
    level: str          # "fail" | "warn" | "info" | "ok"
    detail: str = ""

    def __str__(self) -> str:
        mark = {"fail": "✗", "warn": "!", "info": "·", "ok": "✓"}[self.level]
        return f"{mark} {self.name}" + (f" — {self.detail}" if self.detail else "")


@dataclass
class ValidationReport:
    feature: str
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(c.level == "fail" for c in self.checks)

    @property
    def warned(self) -> bool:
        return any(c.level == "warn" for c in self.checks)

    def fails(self) -> list[Check]:
        return [c for c in self.checks if c.level == "fail"]

    def __str__(self) -> str:
        head = f"[{'기각' if self.failed else '통과'}] {self.feature}"
        body = "\n".join("    " + str(c) for c in self.checks
                         if c.level != "ok")
        return head + ("\n" + body if body else "")


def _sample(table, n_shapes: int, rng) -> list:
    shapes = table.shapes()
    idx = rng.choice(len(shapes), size=min(n_shapes, len(shapes)),
                     replace=False)
    return [shapes[int(i)] for i in sorted(idx)]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """순위 상관. 단조 관계를 잡으므로 스케일이 달라도 중복을 찾는다."""
    if a.size < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _rank_auc(vals: np.ndarray, good: np.ndarray) -> float:
    """이 피처 하나로 정답 집합을 얼마나 가려내는가 (§8.3 7번).

    낮을수록 좋다는 규약이므로 좋은 config 가 작은 값을 가지면 AUC > 0.5.
    """
    n_pos = int(good.sum())
    n_neg = int((~good).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # ★ 동점을 평균 순위로 처리한다. `argsort(argsort())` 는 동점을 임의로
    #   갈라서, 이진 피처(has_spill 등)의 AUC 를 0.5 근처로 뭉개 버린다.
    order = np.argsort(-vals, kind="mergesort")
    r = np.empty(vals.size, dtype=np.float64)
    sv = -vals[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return float((r[good].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def validate_feature(f: Feature, table, matrix, *, hw_alt: Hardware,
                     others: dict[str, np.ndarray] | None = None,
                     n_shapes: int = 6, n_rows: int = 512,
                     seed: int = 0) -> ValidationReport:
    """피처 하나를 검증한다. **예외는 잡아서 `fail` 로 만든다.**"""
    rep = ValidationReport(f.name)
    rng = np.random.default_rng(seed)
    # ★ 형상 수준 피처는 **전 형상**을 본다. 표본을 쓰면 `is_memory_bound`
    #   같은 이진 피처가 우연히 한 종류만 뽑혀 "상수" 로 기각된다 — 피처
    #   문제가 아니라 표본 문제다. 형상은 수십 개뿐이라 비용도 없다.
    shapes = (list(table.shapes()) if f.shape_level
              else _sample(table, n_shapes, rng))

    # -- 1. 실행되는가 ----------------------------------------------------
    vals: list[np.ndarray] = []
    try:
        for p in shapes:
            fe, info = matrix.for_shape(p)
            if f.shape_level:
                # 형상 수준은 스칼라다. 후보 수만큼 펼쳐 통계를 맞춘다.
                n = int(info.n_candidates)
                v = np.full(n, float(getattr(info, f.name)))
            else:
                v = np.asarray(getattr(fe, f.name), dtype=np.float64)
            vals.append(v)
    except Exception as e:                       # noqa: BLE001
        # ⚠️ 삼키지 않는다. 예외는 승인이 아니라 기각이다 (§26.4).
        rep.checks.append(Check("실행", "fail", f"{type(e).__name__}: {e}"))
        return rep
    all_v = np.concatenate(vals)
    if not np.all(np.isfinite(all_v)):
        rep.checks.append(Check("유한", "fail",
                                f"{int((~np.isfinite(all_v)).sum())}개 비유한"))
        return rep
    rep.checks.append(Check("실행", "ok"))

    # -- 2. 범위 ----------------------------------------------------------
    lo, hi = f.expected_range
    out = int(((all_v < lo - 1e-9) | (all_v > hi + 1e-9)).sum())
    if out:
        rep.checks.append(Check(
            "범위", "warn",
            f"선언 [{lo}, {hi}] 밖 {out}/{all_v.size}개 "
            f"(실측 [{all_v.min():.4g}, {all_v.max():.4g}])"))
    else:
        rep.checks.append(Check("범위", "ok"))

    # -- 3. 벡터화 == 스칼라 ----------------------------------------------
    if f.vec is None:
        rep.checks.append(Check("벡터화", "info", "스칼라만 있다 (느리다)"))
    else:
        try:
            from kernelrule.features import verify_vectorized
            for p in shapes[:2]:
                _, info = matrix.for_shape(p)
                verify_vectorized(f, table.frame_for(p), matrix.hw, info,
                                  n=min(n_rows, 128))
            rep.checks.append(Check("벡터화", "ok"))
        except Exception as e:                   # noqa: BLE001
            rep.checks.append(Check("벡터화", "fail", str(e)))
            return rep

    # -- 4. 상수 ----------------------------------------------------------
    if float(all_v.std()) < CONST_STD:
        rep.checks.append(Check(
            "상수", "fail",
            f"std={all_v.std():.3g} — 설명력이 0이다. 표에 필요한 컬럼이 "
            "없어서 0 으로 떨어졌을 수 있다"))
        return rep
    rep.checks.append(Check("상수", "ok"))

    # -- 5. 중복 ----------------------------------------------------------
    if others:
        # ★ 스피어만만으로 중복을 판정하면 안 된다.
        #
        #   `sm_idle_cost = 1/(1-tail_waste) - 1` 은 `tail_waste` 의 **단조
        #   변환**이라 스피어만이 0.999 다. 그런데 규칙은 **선형 가중합**이라
        #   둘의 효과가 전혀 다르다 — 512³ 에서 선형 항으로는 못 내는 5.3배
        #   벌점을 비선형 항이 낸다. 실제로 손규칙이 1.221 -> 1.192 로 내려간
        #   이유가 그것이다 (kernelTab baselines.md).
        #
        #   즉 **단조 변환은 중복이 아니다.** 선형 중복까지 겹칠 때만 폐기
        #   후보로 본다.
        worst = ("", 0.0, 0.0)
        for name, ov in others.items():
            if name == f.name or ov.size != all_v.size:
                continue
            rho = abs(_spearman(all_v, ov))
            r = abs(_pearson(all_v, ov))
            if min(rho, r) > min(worst[1], worst[2]):
                worst = (name, rho, r)
        if worst[1] > DUP_RHO and worst[2] > DUP_RHO:
            rep.checks.append(Check(
                "중복", "warn",
                f"{worst[0]} 와 스피어만 {worst[1]:.3f} / 피어슨 {worst[2]:.3f} "
                f"— 둘 다 > {DUP_RHO} 이므로 폐기 후보 (§8.4)"))
        elif worst[1] > DUP_RHO:
            rep.checks.append(Check(
                "중복", "info",
                f"{worst[0]} 의 **단조 변환** (스피어만 {worst[1]:.3f}, "
                f"피어슨 {worst[2]:.3f}). 선형 가중합에서는 다른 항이다"))
        else:
            rep.checks.append(Check("중복", "ok",
                                    f"최대 min(rho,r) {min(worst[1], worst[2]):.3f}"))

    # -- 6. ★ 스케일 불변성 ------------------------------------------------
    if f.shape_level:
        alt = np.asarray([float(f.fn(p, hw_alt, table.configs(p)[0]))
                          for p in shapes], dtype=np.float64)
        base = np.asarray([float(f.fn(p, matrix.hw, table.configs(p)[0]))
                           for p in shapes], dtype=np.float64)
    else:
        alt_list, base_list = [], []
        for p in shapes[:2]:
            df = table.frame_for(p).iloc[:n_rows]
            cfgs = [config_from_row(r) for r in df.to_dict("records")]
            alt_list.append([float(f.fn(p, hw_alt, c)) for c in cfgs])
            base_list.append([float(f.fn(p, matrix.hw, c)) for c in cfgs])
        alt = np.concatenate([np.asarray(x) for x in alt_list])
        base = np.concatenate([np.asarray(x) for x in base_list])
    changed = float(np.max(np.abs(alt - base))) if alt.size else 0.0
    uses_hw = _uses_hardware(f)
    if uses_hw and changed <= SCALE_MIN_CHANGE:
        rep.checks.append(Check(
            "스케일 불변성", "fail",
            "hw 를 바꿨는데 값이 안 변한다 — hw.* 를 읽는 것처럼 보이지만 "
            "실제로는 하드웨어 상수가 하드코딩돼 있을 수 있다 (§8.3 6번)"))
    elif not uses_hw:
        rep.checks.append(Check(
            "스케일 불변성", "info",
            "hw 를 안 쓰는 피처다 (형상/config 만의 함수). 정상일 수 있다"))
    else:
        rep.checks.append(Check("스케일 불변성", "ok",
                                f"최대 변화 {changed:.4g}"))

    # -- 7. 유용한가 (표시만) ---------------------------------------------
    aucs = []
    for p, v in zip(shapes, vals, strict=True):
        good = table.answer_mask(p)
        a = _rank_auc(v, good)
        if np.isfinite(a):
            aucs.append(a)
    if f.shape_level:
        # 형상 수준 피처는 형상 안에서 상수라 후보를 가를 수 없다.
        # AUC 가 0.5 인 것이 **정상**이다 — 이 피처들은 랭킹이 아니라
        # `if p.is_memory_bound:` 같은 체제 분기에 쓰인다 (§8.1 대체본).
        rep.checks.append(Check("단독 AUC", "info",
                                "형상 수준이라 해당 없음 (분기용 피처)"))
    elif aucs:
        m = float(np.mean(aucs))
        if m < 0.45:
            # ★ 방향이 선언과 반대다. 기각하지는 않는다 — 다른 항과 조합되면
            #   의미가 있을 수 있다. 실제로 `tail_waste` 가 여기 걸리는데,
            #   그것만 최소화하면 작은 타일을 고르게 되어 손규칙이 1.776 까지
            #   나빠졌다 (kernelTab baselines.md). 교락 변수(타일 크기)를
            #   같이 넣어야 의미가 산다.
            rep.checks.append(Check(
                "단독 AUC", "warn",
                f"{m:.3f} < 0.5 — 선언한 방향({f.direction})과 **반대**로 "
                "예측한다. 다른 항과 교락됐을 수 있다"))
        else:
            rep.checks.append(Check(
                "단독 AUC", "info",
                f"{m:.3f} ({'유의미' if abs(m - 0.5) > 0.05 else '거의 무정보'})"))
    return rep


def _uses_hardware(f: Feature) -> bool:
    """소스에 `hw.` 참조가 있는가. 스케일 검사의 해석에 쓴다.

    ★ `f.source` 를 먼저 본다. `exec` 로 만든 피처는 `inspect.getsource` 가
    `OSError` 를 내는데, 그때 `True` 로 떨어지면 **하드웨어를 안 쓰는 정상
    피처가 전부 기각된다** — 스케일 검사는 `uses_hw` 일 때만 fail 이기
    때문이다. F1 첫 실행에서 실제로 그렇게 버려졌다 (D-37).
    """
    if f.source:
        return "hw." in f.source
    import inspect
    try:
        src = inspect.getsource(f.fn)
    except (OSError, TypeError):
        # 소스를 못 읽으면 **판단하지 않는다**. `True` 는 "hw 를 쓴다" 는
        # 주장이고, 그 주장이 틀리면 정상 피처를 기각한다 (§26.4).
        raise ValueError(
            f"{f.name}: 소스를 읽을 수 없어 하드웨어 사용 여부를 판정할 수 "
            "없다. `Feature(source=...)` 에 코드를 넣어라 — 추측하면 "
            "하드웨어 무관 피처를 기각한다 (D-37)") from None
    return "hw." in src


def validate_registry(reg: FeatureRegistry, table, matrix, *,
                      hw_alt: Hardware, **kw) -> dict[str, ValidationReport]:
    """레지스트리 전체를 검증한다. 중복 검사를 위해 값들을 모아 둔다."""
    rng = np.random.default_rng(kw.get("seed", 0))
    shapes = _sample(table, kw.get("n_shapes", 6), rng)
    pool: dict[str, np.ndarray] = {}
    for name in reg.names(shape_level=False):
        try:
            pool[name] = np.concatenate(
                [np.asarray(getattr(matrix.for_shape(p)[0], name))
                 for p in shapes])
        except Exception:                        # noqa: BLE001, S112
            continue      # 실행 실패는 개별 검증에서 fail 로 잡힌다
    return {f.name: validate_feature(f, table, matrix, hw_alt=hw_alt,
                                     others=pool, **kw)
            for f in reg.items()}
