"""LLM 생성 규칙의 정적 검사 (§8.3 + 부록 §8.1 갱신본).

**파싱에 실패하면 거부다. 통과가 아니다** (§26.4). 이 파일의 모든 경로가
실패 쪽으로 기운다.

## 무엇을 막는가

    암기          `if problem.M == 4096: ...`
    조건부 특수화  `if f.waves < 1: s += 5.0`   <- config 수준 분기
    정답 참조      `time_ms`, `difficulty`, `TABLE`
    탈출          import, open, exec, 속성 우회(`__globals__`)
    과적합        숫자 리터럴 + 가중치 개수 <= 8

## 허용하는 것

    형상 수준 분기  `if p.is_memory_bound:`      <- 스칼라라 일반화된다
    np 연산        where/clip/minimum/log/sqrt/... 와 사칙연산
    가중치         `w[0]`, `w[1]` — **인덱스 접근만**

`f.*` 는 배열이라 `if` 가 런타임에 `ValueError` 를 내지만, 그때는 이미 규칙을
실행한 뒤다. AST 로 **실행 전에** 잡는다.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

__all__ = ["CheckReport", "RuleCheckError", "check_rule", "LIMITS"]

LIMITS = {
    #: 숫자 리터럴 + len(W0). 가중치를 예산에 넣는 이유는 §29.4 —
    #: 가중치가 많으면 어떤 구조든 비슷한 regret 에 도달해 구조 비교가
    #: 무의미해진다.
    "literal_budget": 8,
    "ast_nodes": 400,
    "max_lines": 60,
}

#: 규칙 소스에 나타나면 안 되는 이름. 정답이거나 탈출 경로다.
_BANNED_NAMES = frozenset({
    # 정답 (ANSWER_COLS)
    "time_ms", "time_std_ms", "time_min_ms", "time_max_ms", "n_reps",
    "outlier_frac", "cublas_ms", "tflops", "frac_of_peak", "vs_cublas",
    "difficulty", "distinct_time_frac", "n_distinct_times",
    # 측정해야 아는 값 (OUTCOME_COLS)
    "status", "max_rel_error", "actual_split_k", "drift_ratio",
    "sm_clock_mhz", "mem_clock_mhz", "gpu_temp_c", "power_w", "timestamp",
    # 표 접근
    "TABLE", "PerfTable", "table", "times_of", "best_time", "answer_mask",
    "load_for_scoring", "load_for_ranking", "load_bundle", "read_parquet",
    # 탈출
    "eval", "exec", "compile", "open", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "breakpoint",
})

#: 속성 이름에 나타나면 안 되는 것 (`x.__globals__` 같은 우회).
_BANNED_ATTR_PREFIX = ("__",)

#: 허용된 numpy 함수. 이 밖은 거부한다 — `np.random` 이 비결정론을 만든다.
_ALLOWED_NP = frozenset({
    "where", "clip", "minimum", "maximum", "log", "log2", "log10", "sqrt",
    "abs", "exp", "power", "sign", "floor", "ceil", "round", "isfinite",
    "nan_to_num", "square", "reciprocal", "logical_and", "logical_or",
    "logical_not", "greater", "less", "equal", "asarray", "zeros_like",
    "ones_like", "full_like", "fmin", "fmax", "hypot", "cbrt",
})

_ALLOWED_BUILTINS = frozenset({"min", "max", "abs", "float", "int", "len",
                               "sum", "sorted", "range", "enumerate", "zip"})


class RuleCheckError(RuntimeError):
    """규칙이 정적 검사를 통과하지 못했다."""


@dataclass
class CheckReport:
    ok: bool
    violations: list[str] = field(default_factory=list)
    #: 거부는 아니지만 사람이 봐야 하는 것. **`ok` 에 영향을 주지 않는다.**
    warnings: list[str] = field(default_factory=list)
    n_literals: int = 0
    n_weights: int = 0
    n_nodes: int = 0
    max_w_index: int = -1
    #: 가중치가 곱해진 항의 개수. `n_weights` 를 넘으면 재사용이 있다는 뜻이다.
    n_terms: int = 0
    features_used: set[str] = field(default_factory=set)
    shape_values_used: set[str] = field(default_factory=set)

    @property
    def budget_used(self) -> int:
        return self.n_literals + self.n_weights

    def raise_if_bad(self) -> CheckReport:
        if not self.ok:
            raise RuleCheckError(
                "규칙이 정적 검사를 통과하지 못했다:\n  "
                + "\n  ".join(self.violations))
        return self

    def __str__(self) -> str:
        head = "통과" if self.ok else "거부"
        return (f"[{head}] 리터럴 {self.n_literals} + 가중치 {self.n_weights} "
                f"= {self.budget_used}/{LIMITS['literal_budget']}, "
                f"항 {self.n_terms}, "
                f"노드 {self.n_nodes}/{LIMITS['ast_nodes']}, "
                f"피처 {sorted(self.features_used)}"
                + "".join("\n    ✗ " + v for v in self.violations)
                + "".join("\n    ! " + v for v in self.warnings))


def check_rule(code: str, *, feature_names, shape_value_names,
               n_weights: int, limits: dict | None = None) -> CheckReport:
    """`score(f, p, hw, w)` 소스를 검사한다.

    `n_weights` 는 LLM 이 제시한 `W0` 의 길이다. **리터럴 예산에 합산된다.**
    """
    lim = {**LIMITS, **(limits or {})}
    rep = CheckReport(ok=True, n_weights=int(n_weights))
    feature_names = set(feature_names)
    shape_value_names = set(shape_value_names)

    def bad(msg: str) -> None:
        rep.ok = False
        rep.violations.append(msg)

    def warn(msg: str) -> None:
        """거부는 아니지만 사람이 봐야 하는 것."""
        rep.warnings.append(msg)

    # -- 파싱. 실패는 **거부**다 (§26.4) ----------------------------------
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        rep.ok = False
        rep.violations.append(f"파싱 실패: {e}. 통과가 아니라 거부다.")
        return rep

    if code.count("\n") + 1 > lim["max_lines"]:
        bad(f"줄 수 {code.count(chr(10)) + 1} > {lim['max_lines']}")

    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fns) != 1 or fns[0].name != "score":
        bad("최상위에 `def score(f, p, hw, w)` 하나만 있어야 한다 "
            f"(찾은 것: {[getattr(n, 'name', type(n).__name__) for n in tree.body]})")
        return rep
    fn = fns[0]
    args = [a.arg for a in fn.args.args]
    if args[:4] != ["f", "p", "hw", "w"]:
        bad(f"시그니처가 `score(f, p, hw, w)` 여야 한다 (받은 것: {args})")

    rep.n_nodes = sum(1 for _ in ast.walk(tree))
    if rep.n_nodes > lim["ast_nodes"]:
        bad(f"AST 노드 {rep.n_nodes} > {lim['ast_nodes']}")

    # -- 순회 -------------------------------------------------------------
    array_names: set[str] = set()      # f.* 에서 유래한 지역 변수
    #: w[i] 가 몇 번 쓰였는가. 재사용은 예산 우회다.
    w_index_uses: dict[int, int] = {}
    #: 가중치가 곱해진 식의 서명. 중복 항 검출용.
    term_sigs: list[str] = []

    # `w[0]` 의 `0` 은 리터럴 예산에서 뺀다 — 가중치는 `n_weights` 로 이미
    # 예산에 들어가 있다. 안 빼면 항 하나마다 두 번 세어 예산이 반토막 난다.
    weight_index_nodes = {
        id(n.slice) for n in ast.walk(tree)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
        and n.value.id == "w" and isinstance(n.slice, ast.Constant)}

    for node in ast.walk(tree):
        # import 금지
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bad("import 금지. `np` 는 이미 주어진다")

        # 금지 이름
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            bad(f"금지된 이름 참조: {node.id!r}")
        if isinstance(node, ast.Attribute):
            if node.attr in _BANNED_NAMES:
                bad(f"금지된 속성 참조: .{node.attr}")
            if node.attr.startswith(_BANNED_ATTR_PREFIX):
                bad(f"던더 속성 접근 금지: .{node.attr}")

        # 숫자 리터럴
        if isinstance(node, ast.Constant) and isinstance(node.value,
                                                         (int, float)):
            if not isinstance(node.value, bool) \
                    and id(node) not in weight_index_nodes:
                rep.n_literals += 1

        # f.<name> / p.<name>
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base, attr = node.value.id, node.attr
            if base == "f":
                rep.features_used.add(attr)
                if attr not in feature_names:
                    bad(f"등록되지 않은 피처: f.{attr}. "
                        "오타를 조용히 통과시키지 않는다")
            elif base == "p":
                rep.shape_values_used.add(attr)
                if attr not in shape_value_names:
                    bad(f"등록되지 않은 형상 수준 값: p.{attr}")
            elif base == "hw":
                pass
            elif base == "np":
                if attr not in _ALLOWED_NP:
                    bad(f"허용되지 않은 numpy 함수: np.{attr}. "
                        f"허용: {sorted(_ALLOWED_NP)[:8]} ...")

        # problem.M / p.M 직접 비교 -> 암기 경로
        if isinstance(node, ast.Compare):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id in ("p", "problem")
                        and sub.attr in ("M", "N", "K")):
                    bad(f"형상 크기를 직접 비교하는 분기 금지: "
                        f"{sub.value.id}.{sub.attr}. 피처를 거쳐라")

        # w 접근은 인덱스만
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id == "w":
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, int):
                rep.max_w_index = max(rep.max_w_index, sl.value)
                w_index_uses[sl.value] = w_index_uses.get(sl.value, 0) + 1
            else:
                bad("w 는 상수 인덱스로만 접근한다 (w[0], w[1] ...). "
                    "슬라이싱/변수 인덱스 금지")
        if isinstance(node, ast.Name) and node.id == "w":
            parent_ok = True   # 아래 두 번째 순회에서 검사
        # 지역 변수 추적 — f.* 가 대입되면 그 변수도 배열이다
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            if _touches_features(node.value):
                array_names.add(node.targets[0].id)

    # 가중치가 곱해진 식의 서명을 모은다 (중복 항 검출)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for a, b_ in ((node.left, node.right), (node.right, node.left)):
                if (isinstance(a, ast.Subscript)
                        and isinstance(a.value, ast.Name) and a.value.id == "w"):
                    try:
                        term_sigs.append(ast.dump(b_))
                    except Exception:                    # noqa: BLE001, S110
                        pass

    # `w` 를 인덱스 없이 통째로 쓰는 것 금지 (w.sum(), w * f 등)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "w":
            if not _is_subscript_base(tree, node):
                bad("w 를 통째로 쓰지 마라. 인덱스 접근만 허용된다 (§8.1)")
                break

    # -- ★ 가중치 재사용 금지 — 리터럴 예산 우회를 막는다 -------------------
    #
    #   실제 실행에서 LLM 이 이렇게 뚫었다:
    #
    #       s = s + f.log_workspace_bytes * w[0]
    #       s = s + f.log_dram_traffic    * w[0]      <- w[0] 재사용
    #       s = s + f.is_two_stage        * w[0]      <- 또
    #
    #   가중치 8개로 **항 17개**를 만들었다. 예산(§29.4)의 목적은 "파라미터가
    #   많으면 어떤 구조든 비슷한 regret 에 도달해 구조 비교가 무의미해진다"
    #   를 막는 것인데, 항 수를 무제한으로 늘리면 그 목적이 무너진다.
    #   `len(W0) == max_index + 1` 검사만으로는 못 잡는다.
    dup_w = sorted(i for i, c in w_index_uses.items() if c > 1)
    if dup_w:
        bad(f"가중치를 여러 항에 재사용했다: "
            f"{[f'w[{i}]x{w_index_uses[i]}' for i in dup_w]}. "
            f"항 {sum(w_index_uses.values())}개에 가중치 {rep.n_weights}개다 — "
            "리터럴 예산(§29.4)을 우회한다. 항마다 다른 가중치를 써라")
    rep.n_terms = sum(w_index_uses.values())

    # ⚠️ "같은 식이 두 번 나오면 거부" 는 넣지 않는다. **정당한 재가중을
    #    오탐한다** — 형상 수준 분기에서 같은 피처에 다른 가중치를 주는 것이
    #    바로 우리가 원하는 패턴이다 (§A-1):
    #
    #        s = f.traffic * w[0]
    #        if p.is_memory_bound:
    #            s = s + f.traffic * w[6]      # 재가중. 정당하다
    #
    #    실제로 관측된 병리(같은 식 + **같은** 가중치가 두 번)는 위의
    #    재사용 검사가 이미 잡는다.

    if rep.max_w_index >= 0 and rep.max_w_index + 1 != rep.n_weights:
        bad(f"W0 길이 {rep.n_weights} != 참조한 최대 인덱스 + 1 "
            f"({rep.max_w_index + 1}). 안 쓰는 가중치는 예산 낭비다")

    if rep.budget_used > lim["literal_budget"]:
        bad(f"리터럴 {rep.n_literals} + 가중치 {rep.n_weights} = "
            f"{rep.budget_used} > {lim['literal_budget']} (§29.4)")

    # -- ★ config 수준 분기 금지 -------------------------------------------
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            if _touches_features(node.test, array_names):
                bad("config 수준 피처로 분기하지 마라 — `f.*` 는 배열이다. "
                    "`np.where(...)` 를 써라 (§2.3 의 '나쁜 수정')")
        if isinstance(node, ast.comprehension):
            bad("컴프리헨션 금지. 벡터 연산으로 써라")

    # -- ★ 형상 수준 분기가 순위를 못 바꾸는 경우 (경고) --------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and not _touches_features(node.test,
                                                              array_names):
            noop = _noop_shape_branch(node, array_names)
            if noop:
                warn(f"형상 수준 분기가 순위를 바꾸지 못한다: {noop}. "
                     "점수 전체에 형상 상수를 곱하거나 더하는 것은 그 형상 "
                     "안에서 **단조 변환**이라 정렬 결과가 같다. config 수준 "
                     "항의 **가중치를 바꿔야** 의미가 있다")

    return rep


def _noop_shape_branch(node: ast.If, array_names: set) -> str | None:
    """형상 수준 `if` 블록이 순위에 영향이 없는가 (흔한 형태만).

    ## 왜 필요한가 — 실제로 밟았다

        if p.is_memory_bound:
            s = s * w[2]        # ⛔ 순위가 **하나도 안 바뀐다**

    규칙은 형상마다 독립적으로 정렬되므로, 누적 점수 전체에 곱하거나 더한
    **형상 상수는 소거된다.** 문법적으로는 완전히 합법이라 다른 검사에
    안 걸린다. 손규칙 첫 판이 이것 때문에 죽은 항을 들고 있었다.

    ## 무엇을 잡는가

    블록 안의 **모든** 문장이 `s = s <op> <형상 상수>` 형태이고 `op` 가
    `*` 또는 `+`(또는 `-`, `/`)이면 no-op 이다. 형상 상수는
    `p.*` / `hw.*` / `w[i]` / 숫자 리터럴과 그들의 조합이다.

    ## 무엇을 못 잡는가

    일반적인 경우는 못 잡는다 (중간 변수를 거치거나 조건부로 항을 재정의하는
    형태). **가장 흔한 실수만 잡고, 거부가 아니라 경고다.** 진짜 판정은
    채점기가 한다 — §12 진단 리포트의 "형상 수준 분기가 순위를 바꾸는가".
    """
    stmts = list(node.body)
    if not stmts:
        return None
    target = None
    for st in stmts:
        if not isinstance(st, (ast.Assign, ast.AugAssign)):
            return None          # return / if 중첩 등은 판정하지 않는다
        if isinstance(st, ast.Assign):
            if len(st.targets) != 1 or not isinstance(st.targets[0], ast.Name):
                return None
            name = st.targets[0].id
            val = st.value
            # `s = s <op> <스칼라>` 형태여야 한다
            if not (isinstance(val, ast.BinOp)
                    and isinstance(val.op, (ast.Mult, ast.Add, ast.Sub,
                                            ast.Div))):
                return None
            left, right = val.left, val.right
            if isinstance(left, ast.Name) and left.id == name:
                other = right
            elif isinstance(right, ast.Name) and right.id == name:
                other = left
            else:
                return None
        else:                     # AugAssign: s *= ...
            if not isinstance(st.target, ast.Name):
                return None
            if not isinstance(st.op, (ast.Mult, ast.Add, ast.Sub, ast.Div)):
                return None
            name, other = st.target.id, st.value
        if target is None:
            target = name
        elif target != name:
            return None
        if not _is_shape_scalar(other, array_names):
            return None
    return f"`{target}` 에 형상 상수만 {len(stmts)}회 적용" if target else None


def _is_shape_scalar(node, array_names: set) -> bool:
    """이 식이 형상 수준 스칼라뿐인가 (`p.*` / `hw.*` / `w[i]` / 리터럴)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            if isinstance(sub.value, ast.Name) and sub.value.id in ("p", "hw"):
                continue
            return False
        if isinstance(sub, ast.Name):
            if sub.id in ("p", "hw", "w", "np"):
                continue
            if sub.id in array_names:
                return False
            return False
        if isinstance(sub, (ast.Constant, ast.BinOp, ast.UnaryOp,
                            ast.Subscript, ast.Load, ast.Mult, ast.Add,
                            ast.Sub, ast.Div, ast.Pow, ast.USub, ast.Index)):
            continue
        if isinstance(sub, ast.Call):
            return False
    return True


def _touches_features(node, extra: set | None = None) -> bool:
    """이 식이 config 수준 피처(배열)를 건드리는가."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
                and sub.value.id == "f":
            return True
        if extra and isinstance(sub, ast.Name) and sub.id in extra:
            return True
    return False


def _is_subscript_base(tree, target: ast.Name) -> bool:
    """이 `Name` 노드가 `w[...]` 의 밑변인가."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and node.value is target:
            return True
    return False
