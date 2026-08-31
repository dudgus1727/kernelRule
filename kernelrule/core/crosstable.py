"""두 표(다른 GPU)를 견주기 위한 공통 부분집합 (D-88).

5090 표가 오면 "구조가 전이되는가" 를 재야 한다. 그런데 **두 표는 같은
격자가 아니다.**

```
형상 격자   66 vs 66 인데 값이 다르다 (층 B 의 M 상향, 층 E 사다리 이동)
config 축   split_k 8종 vs 10종
ridge       159.1 vs 117.9  -> ★ 같은 형상의 바운드 분류가 뒤집힌다
눈금        1.024us vs 32ns
```

**교집합을 명시적으로 만들고, 통제하지 못하는 것을 세어 남긴다.**
조용히 한쪽에만 있는 것을 버리면 "전이가 됐다" 가 표본 선택의 결과일 수
있다 (§26.4).

## config 동일성 — 아키텍처 독립 축으로만 정한다

`kernel_id` 는 아키텍처마다 다르게 컴파일되므로 조인 키가 될 수 없다.
`regs_per_thread` / `smem_bytes` / `spill_bytes` 도 **빌드 결과**라 GPU 가
바뀌면 달라진다. 남는 것은 **사람이 고르는 축**뿐이다.

```
tile_m, tile_n, tile_k, split_k, split_k_mode, align_a/b/c
★ pipeline_kind 는 넣지 않는다 — 세대마다 이름이 달라질 수 있다
```

⚠️ 이 키가 같아도 **다른 커널**일 수 있다. 같은 타일 축이라도 세대별로
다른 명령어를 쓴다. "같은 config" 가 아니라 **"같은 축 좌표"** 다.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernelrule.core.types import Hardware, Problem

__all__ = ["AXIS_FIELDS", "CrossReport", "axis_key", "common_shapes",
           "common_axis_keys", "bound_flipped", "cross_report"]

#: config 동일성을 정하는 **아키텍처 독립 축**. 빌드 결과(레지스터·smem·
#: 스필)와 `kernel_id` 는 GPU 가 바뀌면 달라지므로 넣지 않는다.
AXIS_FIELDS = ("tile_m", "tile_n", "tile_k", "split_k", "split_k_mode",
               "align_a", "align_b", "align_c")


def axis_key(row) -> tuple:
    """행 하나의 축 좌표. `row` 는 dict 또는 `Config`."""
    get = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    return tuple(get(f) for f in AXIS_FIELDS)


def common_shapes(a, b) -> list[Problem]:
    """두 표에 다 있는 형상. **순서는 `a` 를 따른다** (결정론)."""
    bk = {(p.M, p.N, p.K, p.dtype) for p in b.shapes()}
    return [p for p in a.shapes() if (p.M, p.N, p.K, p.dtype) in bk]


def common_axis_keys(a, b, p: Problem) -> set[tuple]:
    """형상 `p` 에서 두 표에 다 있는 축 좌표."""
    return {axis_key(r) for r in a.frame_for(p).to_dict("records")} & \
           {axis_key(r) for r in b.frame_for(p).to_dict("records")}


def _arith_intensity(p: Problem) -> float:
    """FLOP / 이동 바이트. **형상만의 함수다** — 하드웨어가 안 들어간다.

    ## ★ 정정 (2026-08-31) — 정의를 여기서 다시 쓰지 않는다

    원래 이 함수는 출력 항에 `acc_bytes_per_element`(f32, 4바이트)를
    곱했다. 그런데 누산기는 레지스터에 있고 **DRAM 으로 나가는 C 는
    f16** 이다. kernelTab 의 `arith_intensity` 컬럼과 등록 피처
    `features.physical.arith_intensity` 는 둘 다 셋 다 원소 바이트로
    센다.

    ```
    128x4096x4096   여기 117.03   표/피처 120.471   5090 ridge 117.855
    ```

    **경계가 그 사이에 있어서 `bound_flipped` 가 이 형상을 놓쳤다** —
    5090 전이에서 뒤집힘 4개를 3개로 셌다. 53개 공통 형상 **전부**에서
    두 정의가 달랐다.

    ★ 세 번째 정의를 만든 것이 잘못이다 (원칙 2). 등록 피처에 위임한다.
    """
    from kernelrule.features.physical import arith_intensity

    return arith_intensity(p, None, None)


def _ridge(hw: Hardware) -> float:
    """★ `hw.ridge_point` 를 쓴다 — 여기서 다시 나누지 않는다.

    실효값/스펙값 중 무엇을 쓰는지가 26% 어긋나고 경계 형상의 분류를
    뒤집는다 (§6.2). 그 판단은 `Hardware` 한 곳에만 있어야 한다.
    """
    return float(hw.ridge_point)


def bound_flipped(a, b, shapes=None) -> list[tuple[Problem, bool, bool]]:
    """★ ridge 차이로 **바운드 분류가 뒤집히는** 형상 (D-88).

    `(형상, a 에서 메모리 바운드인가, b 에서 메모리 바운드인가)`.

    체제별로 가중치를 따로 적합하는데(§10) 그 체제 판정이 표마다 다르면
    **두 표에서 다른 것을 재게 된다.** 조용히 넘어가면 "전이가 안 됐다" 가
    사실은 "다른 것을 비교했다" 일 수 있다.
    """
    sh = shapes if shapes is not None else common_shapes(a, b)
    ra, rb = _ridge(a.hw), _ridge(b.hw)
    out = []
    for p in sh:
        ai = _arith_intensity(p)
        ma, mb = ai < ra, ai < rb
        if ma != mb:
            out.append((p, ma, mb))
    return out


@dataclass(frozen=True, slots=True)
class CrossReport:
    """두 표의 겹침. **버린 것을 센다.**"""

    n_shapes_a: int
    n_shapes_b: int
    n_shapes_common: int
    n_axis_a: int
    n_axis_b: int
    n_axis_common: int
    n_bound_flipped: int
    ridge_a: float
    ridge_b: float

    def render(self) -> str:
        def frac(k: int, n: int) -> str:
            return f"{k}/{n} = {k / n:.0%}" if n else f"{k}/0"
        drop_a = frac(self.n_shapes_a - self.n_shapes_common, self.n_shapes_a)
        drop_b = frac(self.n_shapes_b - self.n_shapes_common, self.n_shapes_b)
        flip = ("  — 체제별 적합이 두 표에서 다른 것을 잰다"
                if self.n_bound_flipped else "")
        return "\n".join([
            (f"  형상      A {self.n_shapes_a}  B {self.n_shapes_b}  "
             f"공통 {self.n_shapes_common}"),
            f"            A 에서 버림 {drop_a}   B 에서 버림 {drop_b}",
            (f"  축 좌표    A {self.n_axis_a}  B {self.n_axis_b}  "
             f"공통 {self.n_axis_common}"),
            (f"  ridge     A {self.ridge_a:.1f}  B {self.ridge_b:.1f}  "
             f"({self.ridge_b / self.ridge_a:.2f}배)"),
            f"  ★ 바운드 뒤집힘  {self.n_bound_flipped} 형상{flip}",
        ])


def cross_report(a, b) -> CrossReport:
    sh = common_shapes(a, b)
    ka: set[tuple] = set()
    kb: set[tuple] = set()
    for p in a.shapes():
        ka |= {axis_key(r) for r in a.frame_for(p).to_dict("records")}
    for p in b.shapes():
        kb |= {axis_key(r) for r in b.frame_for(p).to_dict("records")}
    return CrossReport(
        n_shapes_a=len(a.shapes()), n_shapes_b=len(b.shapes()),
        n_shapes_common=len(sh), n_axis_a=len(ka), n_axis_b=len(kb),
        n_axis_common=len(ka & kb),
        n_bound_flipped=len(bound_flipped(a, b, sh)),
        ridge_a=_ridge(a.hw), ridge_b=_ridge(b.hw))
