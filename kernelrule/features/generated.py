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
from dataclasses import dataclass

import numpy as np

from kernelrule.core.matrix import FeatureMatrix
from kernelrule.features import Feature, FeatureRegistry

__all__ = ["FeatureRejected", "check_feature_code", "compile_feature",
           "register_generated", "RAW_FIELDS", "field_block"]


class FeatureRejected(ValueError):
    """피처가 검사를 통과하지 못했다. **고쳐서 쓰지 않는다** (§26.4)."""


#: 프롬프트에 노출하는 원시 필드. `Config.ext` 는 **의도적으로 뺀다** (§4.3).
RAW_FIELDS: dict[str, tuple[str, ...]] = {
    "p": ("M", "N", "K", "dtype", "acc_dtype",
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
    return "\n".join(out)


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
    for b in _BANNED:
        if b in src:
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
    registry.add(f)
    return f
