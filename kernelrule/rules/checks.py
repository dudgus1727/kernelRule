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

__all__ = ["BUDGET", "CheckReport", "RuleCheckError", "check_rule", "LIMITS",
           "weight_reuse_message", "literal_budget_message",
           "identity_transform_message", "noop_term_message"]


def noop_term_message(code: str) -> str | None:
    """누적 점수에 형상 상수만 더하는 항을 찾는다. 순위가 안 바뀐다.

    ★ 규칙은 **형상마다 독립적으로 정렬**되므로 형상 수준 스칼라를 점수
    전체에 더하거나 곱하는 것은 단조 변환이고 순서를 하나도 바꾸지 않는다.
    그런 항은 예산 하나를 그냥 버린다.

    문법적으로 합법이라 실행도 되고 예외도 없다 — **조용히 아무 일도
    하지 않는다.** 그래서 이 검사가 필요하다 (§26.4). RuleWriter A 조건
    첫 성공 규칙이 `p.log_sol_ms * w[0]` 으로 항 하나를 버렸다.

    형상 수준 값이 **`f.*` 와 곱해지면** 의미가 있다 — 그때는 config 수준
    항의 가중치를 형상에 따라 바꾸는 것이므로 순위가 바뀐다.
    """
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None
    bad_terms: list[str] = []

    def walk(body):
        for st in body:
            if isinstance(st, ast.If):
                walk(st.body)
                walk(st.orelse)
                continue
            if not isinstance(st, (ast.Assign, ast.AugAssign)):
                continue
            val = st.value
            # `s = s + <expr>` 또는 `s += <expr>` 의 <expr> 만 본다
            if isinstance(st, ast.Assign) and isinstance(val, ast.BinOp) \
                    and isinstance(val.op, ast.Add):
                expr = val.right
            elif isinstance(st, ast.AugAssign) and isinstance(st.op, ast.Add):
                expr = val
            else:
                continue
            if not _uses_weight(expr):
                continue
            if not _touches_features(expr):
                bad_terms.append(ast.unparse(expr))

    walk(tree.body[0].body)
    if not bad_terms:
        return None
    return (f"순위에 아무 효과가 없는 항이 있다: {bad_terms}. 규칙은 형상마다 "
            "독립적으로 정렬되므로 **형상 수준 값을 점수 전체에 더하면 "
            "순서가 하나도 바뀌지 않는다** — 가중치 하나를 버리는 것이다. "
            "형상 수준 값은 `if p.<이름>:` 으로 분기해 config 수준 항의 "
            "가중치를 바꾸는 데 쓰거나, `f.*` 와 곱해서 써라")


def _uses_weight(node) -> bool:
    return any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
               and n.value.id == "w" for n in ast.walk(node))


def _numeric_literals(tree: ast.AST) -> tuple[list[ast.Constant],
                                              list[ast.Constant]]:
    """숫자 리터럴을 **예산에 드는 것 / 안 드는 것**으로 나눈다 (D-78).

    반환은 `(예산에 드는 것, 비교 상수)`.

    ## 왜 나누나

    §29.4 가 두 가지를 한 예산에 묶고 있었다.

    ```
    가중치 제한   파라미터 수를 막는다 — 구조 비교를 위해서다
    리터럴 제한   "상수를 하드코딩해 이 표에 맞추는 것" 을 막으려던 것
    ```

    `p.roofline_ratio < 1` 은 뒤엣것이 아니다 — roofline 의 무릎이라는
    **물리 상수**다. 그런데 합산 예산에 걸려 진화가 `1` 을 안 쓰고
    우회했다:

    ```
    np.square(x) < x        x < np.sqrt(x)        x < np.sign(x)
    np.isfinite(x)          <- 한 규칙 안에 9번. 오로지 상수 1 을 쓰려고
    ```

    **넷 다 `x < 1` 과 같고, 사람이 읽기 어렵다.** "해석 가능한 규칙" 이
    이 연구의 주장인데 그 주장을 예산이 갉아먹고 있었다.

    ## 무엇이 면제인가

    `ast.Compare` 의 **직접 피연산자**인 숫자 리터럴만이다. 중첩된 식
    안의 상수는 면제가 아니다 — `(f.x - 3) < 1` 에서 `3` 은 든다.

    ## 무엇이 여전히 금지인가

    형상 크기와의 직접 비교(`p.M > 1024`)는 **면제와 무관하게 거부**다.
    그 검사는 `check_rule` 이 따로 한다. 가르는 기준은 **그 상수가
    하드웨어/물리에서 나오는가, 이 표의 형상 분포에서 나오는가** 이고,
    정적으로는 못 가르므로 **면제된 상수를 전부 기록**해 사람이 본다
    (`CheckReport.branch_constants`).
    """
    skip = {id(n.slice) for n in ast.walk(tree)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
            and n.value.id == "w"}
    exempt: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            for side in [n.left, *n.comparators]:
                if (isinstance(side, ast.Constant)
                        and isinstance(side.value, (int, float))
                        and not isinstance(side.value, bool)):
                    exempt.add(id(side))
    counted: list[ast.Constant] = []
    branch: list[ast.Constant] = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Constant)
                and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool)):
            continue
        if id(n) in skip:
            continue                    # w[0] 의 0 은 가중치 쪽에서 센다
        (branch if id(n) in exempt else counted).append(n)
    return counted, branch


#: 인자가 유한하면 **언제나 1** 인 호출. 상수를 만드는 데밖에 못 쓴다.
#: ⚠️ "이 표에서 유한하다" 는 **표 의존 사실**이다 (f 피처 19개 전부에서
#: 확인, D-78). 다른 표에서는 실제 판별 기능을 가질 수 있으므로 메시지를
#: "언제나 1" 이 아니라 **"상수 취급 위험"** 으로 쓴다.
_CONST_CALLS = ("isfinite",)


def identity_transform_message(code: str) -> str | None:
    """★ 항등 변환으로 상수를 만드는가 (D-92).

    ```
    np.isfinite(x)                    유한하면 언제나 1
    np.sign(x)                        x > 0 이면 언제나 1
    x < np.sqrt(x) / np.square(x) < x   둘 다 x < 1 과 같다
    ```

    **설명 가능성이 아니라 결함이다.** 넷 다 수학적으로 상수/단순 비교와
    같은데 사람이 읽기 어렵고, "해석 가능한 규칙" 이라는 주장을 갉아먹는다.

    ## 왜 지금 막나

    합산 예산이 리터럴을 막던 동안에는 **우회할 이유가 있었다** — 항 8개를
    쓰려면 리터럴이 0개여야 했다. D-78 로 분기 비교 상수를 예산에서 빼서
    그 이유를 없앴고, 실제로 리터럴 비교가 6/6 실행에 나왔다 (D-84).
    **이유를 없앤 뒤에 막는다** — 순서를 바꾸면 다른 우회를 찾는다.

    거부 메시지에 **대안**을 함께 적는다. 무엇이 금지인지만 말하면 모델은
    또 다른 우회를 만든다.
    """
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None
    bad: list[str] = []
    for n in ast.walk(tree):
        # np.isfinite(x) — 상수 1 을 만드는 호출
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "np"
                and n.func.attr in _CONST_CALLS):
            bad.append(f"np.{n.func.attr}(...) — 인자가 유한하면 상수 1 로 "
                       "취급될 위험이 있다")
        # np.sign(x) 이 비교의 한쪽에 오면 사실상 리터럴 1 이다
        if isinstance(n, ast.Compare):
            for side in [n.left, *n.comparators]:
                if (isinstance(side, ast.Call)
                        and isinstance(side.func, ast.Attribute)
                        and isinstance(side.func.value, ast.Name)
                        and side.func.value.id == "np"
                        and side.func.attr == "sign"):
                    bad.append("np.sign(...) 을 비교에 썼다 — 양수 입력에서 "
                               "`< 1` 과 같다")
            # x < np.sqrt(x) / np.square(x) < x  — 둘 다 x < 1 이다
            for lo, hi in zip([n.left, *n.comparators][:-1],
                              [n.left, *n.comparators][1:], strict=False):
                if _same_arg_transform(lo, hi, "sqrt") \
                        or _same_arg_transform(hi, lo, "square"):
                    bad.append("`x < np.sqrt(x)` / `np.square(x) < x` — "
                               "둘 다 `x < 1` 과 같다")
    if not bad:
        return None
    uniq = sorted(set(bad))
    return ("항등 변환으로 상수를 만들고 있다 (D-92): " + " / ".join(uniq)
            + ". ★ 그럴 필요가 없다 — **분기 조건의 비교 상수는 예산에서 "
              "면제된다** (D-78). `p.<형상값> < 1` 처럼 숫자를 그대로 써라. "
              "읽는 사람이 물리적 경계를 볼 수 있어야 한다")


def _same_arg_transform(a, b, fn: str) -> bool:
    """`a` 와 `np.<fn>(a)` 가 같은 식인가."""
    if not (isinstance(b, ast.Call) and isinstance(b.func, ast.Attribute)
            and isinstance(b.func.value, ast.Name) and b.func.value.id == "np"
            and b.func.attr == fn and len(b.args) == 1):
        return False
    try:
        return ast.unparse(a) == ast.unparse(b.args[0])
    except Exception:                                       # noqa: BLE001
        return False


def literal_budget_message(code: str, n_weights: int,
                           *, budget: int | None = None) -> str | None:
    """숫자 리터럴 + 가중치가 예산을 넘으면 메시지를, 아니면 `None`.

    ★ `weight_reuse_message` 와 같은 이유로 따로 뺐다 — LLM 경계에서
    재시도를 걸어야 모델이 무엇이 틀렸는지 듣는다. 이 검사가 정적 단계에만
    있었을 때 RuleWriter 제안 3개가 연속으로 같은 이유로 폐기됐고, 모델은
    "가중치 8개면 리터럴을 쓸 수 없다" 를 끝내 알지 못했다.

    ★ 세는 일은 `_numeric_literals` 하나가 한다 — `check_rule` 과 여기가
    **따로 세면 갈린다** (D-37 계열). 갈리면 LLM 경계는 통과시키고 정적
    검사가 조용히 버린다.
    """
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None
    counted, branch = _numeric_literals(tree)
    n_lit = len(counted)
    total = n_lit + n_weights
    b = int(budget if budget is not None else LIMITS["budget"])
    if total <= b:
        return None
    hint = ""
    if branch:
        hint = (f" (분기 비교 상수 {len(branch)}개는 예산에서 빠졌다 — "
                "그것은 계속 써도 된다)")
    return (f"숫자 리터럴 {n_lit}개 + 가중치 {n_weights}개 = {total} > "
            f"{b} (§29.4).{hint} 가중치와 **분기 비교가 "
            f"아닌** 숫자 리터럴이 같은 예산을 쓴다 — 상수를 하나 쓰면 "
            f"가중치를 하나 줄여야 한다. 항을 줄이거나, 그 상수를 분기 "
            f"조건의 비교로 옮겨라")


def weight_reuse_message(code: str) -> str | None:
    """`w[i]` 를 여러 항에 재사용했으면 그 메시지를, 아니면 `None`.

    ★ 이것만 따로 뺀 이유는 **LLM 경계에서 재시도를 걸기 위해서**다.
    전체 검사는 AST 순회가 무겁고 등록된 피처 목록이 필요한데, 재사용은
    코드만 보면 알 수 있다. 스키마 validator 가 이걸 부르면 Pydantic AI 가
    메시지를 모델에 되먹여 **고쳐서 다시 내게** 한다.

    이 검사가 `checks.py` 에만 있었을 때는 RuleWriter 제안이 조용히 폐기됐다
    — 모델은 무엇이 틀렸는지 듣지 못하고 같은 실수를 반복했다.
    """
    uses: dict[int, int] = {}
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None                     # 문법 오류는 다른 검사가 잡는다
    for n in ast.walk(tree):
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id == "w" and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, int)):
            uses[n.slice.value] = uses.get(n.slice.value, 0) + 1
    dup = sorted(i for i, c in uses.items() if c > 1)
    if not dup:
        return None
    return (f"가중치를 여러 항에 재사용했다: "
            f"{[f'w[{i}]x{uses[i]}' for i in dup]}. 항 {sum(uses.values())}개를 "
            f"가중치 {len(uses)}개로 만들었다 — 리터럴 예산(§29.4)을 우회한다. "
            "항마다 **다른** 가중치를 써라. 항이 예산을 넘으면 항을 지워라")

#: ★ 예산의 **유일한 출처**. 프롬프트·스키마·검사기가 각자 8 을 적으면
#: 하나를 빠뜨린다 (`is_reference` / `top_k` / `DEFAULT_MODEL` /
#: `REGISTRY` / `load_generated` 에 이은 여섯 번째가 된다).
#:
#: ⚠️ **8 은 임의로 정한 숫자이고 검증하지 않았다** (§29.4). 8 vs 16 을
#: 재려 했으나 적합기가 16차원에서 버티지 못해 멈췄다 (D-77).
BUDGET = 8

LIMITS = {
    #: 가중치 개수 + **분기 비교가 아닌** 숫자 리터럴 (D-78).
    #: 가중치를 예산에 넣는 이유는 §29.4 — 가중치가 많으면 어떤 구조든
    #: 비슷한 regret 에 도달해 구조 비교가 무의미해진다.
    "budget": BUDGET,
    "ast_nodes": 400,
    "max_lines": 60,
}


def limits_for(budget: int | None) -> dict:
    """★ 예산에 **따라 움직이는 상한들**. 예산만 올리면 다른 벽에 막힌다.

    실측: 8항 규칙의 AST 노드가 중앙 271 / 최대 383 이다. 상한 400 을
    그대로 두고 예산만 16 으로 올리면 **16항 규칙은 노드 상한에서
    거부된다.** 그러면 "예산 16 이 효과가 없다" 가 아니라 "16항을 쓸 수
    없었다" 를 재게 된다 (D-105 와 같은 자리).

    비례로 올린다 — 8항에 400 이면 16항에 800 이다.
    """
    b = int(budget if budget is not None else BUDGET)
    if b < 1:
        raise ValueError(f"예산은 1 이상이어야 한다: {b}")
    k = b / BUDGET
    return {"budget": b,
            "ast_nodes": int(round(LIMITS["ast_nodes"] * k)),
            "max_lines": int(round(LIMITS["max_lines"] * k))}

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
    #: ★ 예산에서 **면제된** 분기 비교 상수들 (D-78). 거부하지 않지만
    #: 기록한다 — "물리 상수인가, 이 표의 형상 분포인가" 는 정적으로 못
    #: 가르므로 사람이 본다.
    branch_constants: list[float] = field(default_factory=list)

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
        bc = (f", 분기상수 {self.branch_constants}(면제)"
              if self.branch_constants else "")
        return (f"[{head}] 리터럴 {self.n_literals} + 가중치 {self.n_weights} "
                f"= {self.budget_used}/{LIMITS['budget']}{bc}, "
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

    # ★ 리터럴은 `_numeric_literals` **하나**가 센다 — LLM 경계
    #   (`literal_budget_message`) 와 여기가 따로 세면 갈린다 (D-37 계열).
    #   `w[0]` 의 `0` 은 거기서 빠진다: 가중치는 `n_weights` 로 이미 예산에
    #   들어가 있어서, 안 빼면 항마다 두 번 세어 예산이 반토막 난다.
    _counted, _branch = _numeric_literals(tree)
    rep.n_literals = len(_counted)
    rep.branch_constants = [n.value for n in _branch]

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
                    import contextlib

                    with contextlib.suppress(Exception):
                        term_sigs.append(ast.dump(b_))

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

    if (m := identity_transform_message(code)):
        bad(m)

    if rep.budget_used > lim["budget"]:
        bad(f"리터럴 {rep.n_literals} + 가중치 {rep.n_weights} = "
            f"{rep.budget_used} > {lim['budget']} (§29.4). "
            + (f"분기 비교 상수 {rep.branch_constants} 는 이미 면제됐다"
               if rep.branch_constants else
               "분기 조건의 비교 상수는 예산에서 빠진다 (D-78)"))

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
