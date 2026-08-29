"""LLM 이 쓴 피처를 검사하고 등록한다 (§11.4 — FeatureWriter).

## 왜 별도 레지스트리인가

`REGISTRY` 는 사람이 쓴 24개다. 생성된 피처를 거기 섞으면 **F0~F3 조건을
구분할 수 없게 된다** — "24개를 조합만 했나" 와 "스스로 만들었나" 가
이 프로젝트의 근본 질문이다.

    F0   피처 없음        물리를 처음부터 코드로 옮길 수 있나
    F1   원시 값만        파생 물리량을 만들 수 있나       ★ 가장 흥미롭다
    F2   기초 5개         그 위에 쌓을 수 있나
    F3   24개 전부        조합만 (= 지금까지의 모든 실행)

## 검사 순서 — 하나라도 실패하면 등록하지 않는다 (§26.4)

    1. AST      금지 이름 / import / cfg.ext / 하드코딩 상수
    2. 샌드박스  격리 실행. 무한 루프와 예외를 여기서 잡는다
    3. §8.3     실행·범위·벡터화·상수·중복·스케일 불변성·유용성

**3번의 "스케일 불변성" 이 핵심이다.** `hw` 를 바꿨는데 값이 안 변하면
하드웨어 상수를 하드코딩한 것이고, 그러면 아키텍처 전이가 무너진다.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, replace

import numpy as np

from kernelrule.core.matrix import FeatureMatrix
from kernelrule.features import Feature, FeatureRegistry

__all__ = ["FeatureRejected", "check_feature_code", "compile_feature",
           "register_generated", "RAW_FIELDS", "field_block"]


class FeatureRejected(ValueError):
    """피처가 검사를 통과하지 못했다. **고쳐서 쓰지 않는다** (§26.4)."""


#: 프롬프트에 노출하는 원시 필드. `Config.ext` 는 **의도적으로 뺀다** (§4.3).
RAW_FIELDS: dict[str, tuple[str, ...]] = {
    # ★ `bytes_per_element` / `acc_bytes_per_element` 는 `dtype` 에서
    #   유도된 값이라 **새 정보가 아니다.** 노출하는 이유는 §30.11 —
    #   `p.dtype` 은 문자열인데 샌드박스에 `np.dtype(...).itemsize` 가
    #   없어서, roofline 을 만들려던 LLM 이 세 번 연속 거부됐다 (D-63).
    "p": ("M", "N", "K", "dtype", "acc_dtype",
          "bytes_per_element", "acc_bytes_per_element",
          "layout_a", "layout_b", "layout_c"),
    "hw": ("sm_count", "smem_per_block", "max_threads_per_sm", "regs_per_sm",
           "peak_tflops_f16", "bandwidth_gbps", "l2_bytes", "ridge_point"),
    "cfg": ("tile_m", "tile_n", "tile_k", "align_a", "align_b", "align_c",
            "split_k", "split_k_mode", "regs_per_thread", "threads",
            "smem_bytes", "spill_bytes", "max_blocks_per_sm",
            "pipeline_kind", "inst_total"),
}

_ALLOWED_NP = frozenset({
    "where", "clip", "minimum", "maximum", "log", "log2", "sqrt", "abs",
    "exp", "power", "sign", "floor", "ceil", "round", "isfinite",
    "nan_to_num", "square", "fmin", "fmax"})
_ALLOWED_MATH = frozenset({
    "ceil", "floor", "log", "log2", "sqrt", "exp", "fabs", "pow", "inf"})

#: 이 이름이 나오면 정답을 보려는 것이다 (§3).
_BANNED = ("time_ms", "cublas_ms", "difficulty", "tflops", "best_ms",
           "distinct_time_frac", "TABLE", "__", "open", "eval", "exec",
           "import", "random")

#: 하드웨어 상수로 보이는 리터럴. `hw.*` 에서 읽어야 한다.
_HW_LITERALS = re.compile(r"\b(84|101376|99328|65536|1536|116\.1|729\.7|"
                          r"159\.1|154\.8|768|6291456)\b")

MAX_LINES = 12


def field_block() -> str:
    """프롬프트에 넣을 원시 필드 목록."""
    out = []
    for base, names in RAW_FIELDS.items():
        out.append(f"### `{base}` — "
                   + {"p": "GEMM 형상", "hw": "하드웨어",
                      "cfg": "커널 config"}[base])
        out.append("  " + "  ".join(f"`{base}.{n}`" for n in names))
    out.append("")
    out.append("`hw.ridge_point` 는 `peak_flops / bandwidth` 입니다 — "
               "roofline 의 무릎이고 계산해 두었습니다.")
    out.append("")
    out.append("★ `p.dtype` / `p.acc_dtype` 은 **문자열**입니다 "
               "(`\"f16\"` 같은). 바이트가 필요하면 "
               "`p.bytes_per_element` / `p.acc_bytes_per_element` 를 "
               "쓰세요 — dtype 에서 유도해 둔 float 입니다.")
    return "\n".join(out)


def uses_cfg(code: str) -> bool:
    """이 피처 함수가 `cfg.*` 를 참조하는가 (§30.12).

    `shape_level` 판정의 **AST 겹**이다. 참조하지 않으면 값이 config 와
    무관하다는 것이 코드에서 확정된다 — 표에 의존하지 않는 판정이다.
    """
    tree = ast.parse(code.strip())
    return any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
               and n.value.id == "cfg" for n in ast.walk(tree))


#: 형상 안 상대 분산이 이보다 작으면 "상수" 로 본다. 부동소수 비교라
#: 정확 일치를 쓰면 마지막 자리 차이에 걸린다.
_CONST_RTOL = 1e-12

#: 피처 이름 -> 형상 수준으로 판정한 근거. ★ "cfg 를 참조하는데 이 표에서
#: 상수" 인 것은 **번들이 바뀌면 다시 판정해야 한다** (§30.12).
SHAPE_LEVEL_REASON: dict[str, str] = {}


def detect_shape_level(f: Feature, table, *, n_shapes: int = 8
                       ) -> tuple[bool, str]:
    """★ 두 겹 판정 — `(형상 수준인가, 근거)` (§30.12).

    ```
    1. 데이터   전 형상에서 config 간 상대 분산이 0 인가
    2. AST      함수가 cfg.* 를 참조하는가
                안 함        -> 확실히 형상 수준 (코드가 보증한다)
                하는데 상수  -> ★ 이 표에서만 그럴 수 있다. 경고를 단다
    ```

    두 번째가 왜 필요한가: `alignment_guarantee_deficit` 은 이 번들의
    61형상에서 alignment 가 config 와 무관해 상수일 수 있다. **다른 표
    에서는 config 의존일 수도 있다.** 그런 것은 형상 수준으로 등록하되
    그 사실을 기록해서, 번들이 바뀔 때 다시 판정하게 한다.
    """
    from kernelrule.core.matrix import FeatureMatrix

    one = FeatureRegistry(f"probe-shape-{f.name}")
    one.add(f)
    mat = FeatureMatrix(table, one)
    for p in list(table.shapes())[:n_shapes]:
        fe, _ = mat.for_shape(p)
        v = np.asarray(getattr(fe, f.name), dtype=np.float64)
        scale = max(float(np.nanmax(np.abs(v))), 1.0)
        if float(np.nanstd(v)) > _CONST_RTOL * scale:
            return False, "config 마다 값이 다르다"
    if f.source and not uses_cfg(f.source):
        return True, "cfg 를 참조하지 않는다 — 코드가 보증한다"
    return True, ("★ cfg 를 참조하는데 이 표에서 상수다 — "
                  "다른 번들에서는 config 의존일 수 있다. 재판정 필요")


@dataclass(frozen=True, slots=True)
class _Spec:
    name: str
    code: str


def check_feature_code(code: str, *, known: frozenset[str]) -> str:
    """AST 정적 검사. 통과하면 함수 이름을, 아니면 예외.

    ★ 문자열 검사가 아니라 AST 다 — 주석에 `import` 를 쓴다고 거부하면
    재시도만 소진한다 (D-27).
    """
    try:
        tree = ast.parse(code.strip())
    except SyntaxError as e:
        raise FeatureRejected(f"문법 오류: {e}") from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise FeatureRejected("함수 정의 **하나**여야 한다")
    fn = tree.body[0]
    args = [a.arg for a in fn.args.args]
    if args != ["p", "hw", "cfg"]:
        raise FeatureRejected(f"시그니처가 (p, hw, cfg) 여야 한다. 받은 것: {args}")
    if fn.name in known:
        raise FeatureRejected(f"이미 있는 이름이다: {fn.name}")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", fn.name):
        raise FeatureRejected(f"이름은 소문자+밑줄이어야 한다: {fn.name}")
    n_lines = len(code.strip().splitlines())
    if n_lines > MAX_LINES:
        raise FeatureRejected(f"{n_lines}줄 > {MAX_LINES}. 더 짧게")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise FeatureRejected("import 금지. np 와 math 는 이미 있다")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base, attr = node.value.id, node.attr
            if base in RAW_FIELDS and attr not in RAW_FIELDS[base]:
                raise FeatureRejected(
                    f"쓸 수 없는 필드: {base}.{attr}. "
                    f"허용: {sorted(RAW_FIELDS[base])}")
            if base == "np" and attr not in _ALLOWED_NP:
                raise FeatureRejected(f"허용되지 않은 numpy 함수: np.{attr}")
            if base == "math" and attr not in _ALLOWED_MATH:
                raise FeatureRejected(f"허용되지 않은 math 함수: math.{attr}")
            if base == "cfg" and attr == "ext":
                raise FeatureRejected("cfg.ext 는 아키텍처 전용이다 (§4.3)")
        if isinstance(node, ast.Name) and node.id not in (
                "p", "hw", "cfg", "np", "math", "float", "int", "max", "min",
                "abs", "sum", "len", "bool", "round", "pow", *(
                    t.id for t in ast.walk(tree)
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store))):
            raise FeatureRejected(f"알 수 없는 이름: {node.id}")

    src = ast.unparse(tree)
    # ★ **허용 필드를 먼저 지운 뒤** 금지어를 찾는다 (D-73).
    #
    #   `hw.peak_tflops_f16` 은 `RAW_FIELDS` 가 명시적으로 허용한 필드인데
    #   금지어 `"tflops"` 가 부분 문자열로 걸렸다. roofline 을 만들려던
    #   제안이 그렇게 거부됐다 — **검사기가 자기가 허용한 것을 금지했다.**
    #   D-37(`inspect.getsource` 실패를 "hw 를 쓴다" 로 떨어뜨림)과 같은
    #   부류다: 검사기의 결함이 LLM 의 실패로 보인다 (원칙 8).
    masked = src
    for base, names in RAW_FIELDS.items():
        for n in names:
            masked = masked.replace(f"{base}.{n}", f"{base}.<ok>")
    for b in _BANNED:
        if b in masked:
            raise FeatureRejected(f"금지된 참조: {b!r} (§3)")
    if (m := _HW_LITERALS.search(src)):
        raise FeatureRejected(
            f"하드웨어 상수를 하드코딩했다: {m.group()}. `hw.*` 에서 읽어라 "
            "— 그러지 않으면 다른 GPU 에서 틀린 값이 된다 (§4.3)")
    return fn.name


def compile_feature(code: str, *, known: frozenset[str]):
    """검사 후 호출 가능한 함수로. **`np`/`math` 만 준다.**"""
    name = check_feature_code(code, known=known)
    env: dict = {"np": np, "math": math, "__builtins__": {
        "max": max, "min": min, "abs": abs, "float": float, "int": int,
        "sum": sum, "len": len, "bool": bool, "round": round, "pow": pow}}
    exec(compile(code.strip(), f"<feature:{name}>", "exec"), env)  # noqa: S102
    return name, env[name]


def _reference_columns(table, matrix, extra: FeatureRegistry,
                       n_shapes: int = 4) -> dict[str, np.ndarray]:
    """중복 판정용 기준 열. **사람이 쓴 것 + 이미 만든 것** 둘 다 본다.

    새 축을 찾으라고 해 놓고 이미 만든 것과의 중복을 안 보면, 같은 것을
    이름만 바꿔 반복하게 된다.
    """
    out: dict[str, list] = {}
    shapes = list(table.shapes())[:n_shapes]
    for reg, mat in ((matrix.registry, matrix),
                     (extra, FeatureMatrix(table, extra) if extra._items
                      else None)):
        if mat is None:
            continue
        for p in shapes:
            fe, info = mat.for_shape(p)
            for n in reg._items:
                f = reg[n]
                v = (np.full(int(info.n_candidates), float(getattr(info, n)))
                     if f.shape_level else np.asarray(getattr(fe, n), float))
                out.setdefault(n, []).append(v)
    return {n: np.concatenate(v) for n, v in out.items()}


def register_generated(code: str, *, registry: FeatureRegistry, meta: dict,
                       table, matrix, hw_alt,
                       others: dict | None = None) -> Feature:
    """검사 -> 샌드박스 -> §8.3 검증 -> 등록. 하나라도 실패하면 예외.

    `others` 는 중복 판정용 기준 열이다. 안 주면 매번 다시 계산하는데,
    사람이 쓴 24개는 바뀌지 않으므로 호출자가 한 번 만들어 넘기는 편이
    낫다 (제안 20회면 30초 x 20 이다).
    """
    from kernelrule.features.validate import validate_feature

    name, fn = compile_feature(code, known=frozenset(registry._items))
    rng = tuple(meta.get("expected_range", (0.0, 1.0)))
    f = Feature(name=name, fn=fn, unit=str(meta.get("unit", "dimensionless")),
                expected_range=(float(rng[0]), float(rng[1])),
                direction=str(meta.get("direction", "higher_is_worse")),
                doc=str(meta.get("rationale", ""))[:200],
                physical_meaning=str(meta.get("rationale", "")),
                code_hash=str(abs(hash(code.strip()))),
                source=code.strip())
    # ★ 후보를 담은 **임시 행렬**로 검증한다. `validate_feature` 는 값을
    #   `matrix.for_shape()` 에서 읽으므로, 아직 등록되지 않은 피처는
    #   호출자의 행렬에 없다. 한 열만 계산하므로 비용은 작다 (§21.2).
    tmp = FeatureRegistry(f"probe-{name}")
    tmp.add(f)
    probe = FeatureMatrix(table, tmp)
    if others is None:
        others = _reference_columns(table, matrix, registry)
    rep = validate_feature(f, table, probe, hw_alt=hw_alt, others=others)
    if rep.failed:
        raise FeatureRejected(
            f"{name}: §8.3 검증 실패 — "
            + "; ".join(f"{c.name}: {c.detail}" for c in rep.fails()))
    # ★ 형상 수준 판정 (§30.12). 생성 경로에는 `shape_feature` 데코레이터가
    #   없어서 **전부 config 수준으로 등록되고 있었다.** 그러면 규칙이
    #   `if p.<x>:` 로 분기할 수 없고, 형상 안에서 상수인 항은 순위를 하나도
    #   못 바꾼다 (절대 규칙 2). F1 21개 중 5개가 그 상태였다 (D-65).
    is_shape, why = detect_shape_level(f, table)
    if is_shape:
        f = replace(f, shape_level=True)
        # ★ 근거를 남긴다. 특히 "cfg 를 참조하는데 이 표에서 상수" 인 것은
        #   **번들이 바뀌면 다시 판정해야 한다.** frozen dataclass 라
        #   모듈 수준 표에 담는다 — 호출자가 summary.json 에 적는다.
        SHAPE_LEVEL_REASON[name] = why
    registry.add(f)
    return f
