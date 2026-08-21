"""스키마 계약과 어댑터 (§23).

kernelTab 은 지금도 바뀌고 있다 (`migration_plan.md` 11건, `env_hash` 재정의,
`schema_version` 2 로의 이행). 오늘 스키마에 맞춰 짜면 본 캠페인 표가 넘어올
때 깨진다. 그래서 표와 우리 타입 사이에 얇은 계층을 하나 둔다.

원칙 (§26.4 — 전부 실패 쪽으로 기운다):

    필수 컬럼이 없다        -> 에러       (기본값으로 때우지 않는다)
    새 컬럼이 생겼다        -> 경고 후 진행
    별칭으로 찾았다         -> 기록하고 진행
    유도 컬럼을 만들 수 없다 -> 에러
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pandas as pd

__all__ = [
    "ALIASES",
    "DERIVED",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "SchemaError",
    "SchemaReport",
    "check_schema",
    "normalize",
]

#: 우리 쪽 계약 버전. kernelTab 의 `BUNDLE.json` schema_version 과 다른 축이다.
SCHEMA_VERSION = "1.0"


class SchemaError(RuntimeError):
    """표가 계약을 만족하지 않는다. **조용히 진행하지 않는다.**"""


#: 규칙 입력(`load_for_ranking`)에 반드시 있어야 하는 컬럼.
#: 여기 없으면 `Config` 를 만들 수 없거나 물리 피처를 계산할 수 없다.
REQUIRED_COLUMNS: frozenset[str] = frozenset({
    # 형상
    "M", "N", "K", "dtype",
    # config (아키텍처 공통) — 전이 피처가 이것만으로 계산되어야 한다 (§4.3)
    "tile_m", "tile_n", "tile_k",
    "align_a", "align_b", "align_c",
    "split_k", "split_k_mode",
    "kernel_id", "arch",
    # 빌드 시점에 알 수 있는 커널 속성 (§3.2 — 실행 불필요하므로 허용)
    "regs_per_thread", "threads", "max_blocks_per_sm", "pipeline_kind",
})

#: 있으면 쓰고 없으면 넘어가는 것. **없다고 에러를 내지 않는다.**
OPTIONAL_COLUMNS: frozenset[str] = frozenset({
    "acc_dtype", "layout_a", "layout_b", "layout_c",
    "ext_warp_m", "ext_warp_n", "ext_warp_k", "ext_stages",
    "ext_swizzle_type", "ext_swizzle_n",
    "hmma_count", "inst_total", "ldg_count", "lds_count", "sts_count",
    "ldsm_count", "cpasync_count",
    "workspace_bytes", "workspace_dtype", "partials_dtype",
    "theoretical_occupancy", "regs_total_per_block", "launchable",
    "local_bytes", "res_regs", "res_local", "build_seconds",
    "smem_matches", "hmma_matches", "expected_hmma", "cutlass_max_blocks",
    "env_hash", "bundle_id", "gpu_name", "sm_count", "clock_locked",
    "cutlass_commit", "nvcc_arch", "locked_mhz",
})

#: 표의 이름과 우리가 쓰는 이름이 1:1 일 필요는 없다.
#: 값: (우리 이름) -> (표에서 찾아볼 후보들, 앞에서부터)
ALIASES: dict[str, tuple[str, ...]] = {
    "smem_bytes": ("smem_bytes", "smem_dynamic", "smem_computed",
                   "smem_static_bytes"),
}

#: 유도 컬럼 — 어댑터가 만든다 (§23.3). 규칙이 쓰는 이름과 표의 이름을
#: 분리해 두면 kernelTab 이 컬럼을 쪼개거나 합쳐도 여기만 고치면 된다.
DERIVED: dict[str, tuple[str, ...]] = {
    # 스필은 읽기/쓰기가 따로 기록된다. 규칙은 합계만 알면 된다.
    "spill_bytes": ("spill_stores", "spill_loads"),
}


@dataclass
class SchemaReport:
    ok: bool
    n_rows: int
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    aliased: dict[str, str] = field(default_factory=dict)
    derived: list[str] = field(default_factory=list)

    def raise_if_bad(self) -> SchemaReport:
        if not self.ok:
            raise SchemaError(
                f"표가 계약을 만족하지 않는다. 누락된 필수 컬럼: {self.missing}\n"
                f"  core/adapter.py 의 REQUIRED_COLUMNS / ALIASES / DERIVED 를 "
                f"확인하라. 기본값으로 때우지 않는다 (§26.4).")
        return self

    def __str__(self) -> str:
        parts = [f"rows={self.n_rows}", f"ok={self.ok}"]
        if self.missing:
            parts.append(f"missing={self.missing}")
        if self.aliased:
            parts.append(f"aliased={self.aliased}")
        if self.derived:
            parts.append(f"derived={self.derived}")
        if self.unexpected:
            parts.append(f"new={len(self.unexpected)}개")
        return "SchemaReport(" + ", ".join(parts) + ")"


def _resolve(df: pd.DataFrame, name: str) -> str | None:
    """`name` 을 표의 실제 컬럼 이름으로 푼다. 별칭을 훑는다."""
    if name in df.columns:
        return name
    for cand in ALIASES.get(name, ()):
        if cand in df.columns:
            return cand
    return None


def check_schema(df: pd.DataFrame, *, unexpected: str = "warn") -> SchemaReport:
    """필수 컬럼 존재 / 별칭 / 유도 가능성 / 새 컬럼을 보고한다.

    `unexpected="warn"` 이 기본이다 — kernelTab 이 컬럼을 추가하는 것은
    정상이고, 그때마다 터지면 표를 못 쓴다. `"raise"` 로 올릴 수 있다.
    """
    cols = set(df.columns)
    missing: list[str] = []
    aliased: dict[str, str] = {}
    derived: list[str] = []

    for name in sorted(REQUIRED_COLUMNS):
        got = _resolve(df, name)
        if got is None:
            missing.append(name)
        elif got != name:
            aliased[name] = got

    for name in sorted(ALIASES):
        if name in REQUIRED_COLUMNS:
            continue
        got = _resolve(df, name)
        if got is None:
            missing.append(name)
        elif got != name:
            aliased[name] = got

    for name, sources in DERIVED.items():
        if name in cols:
            continue
        if all(s in cols for s in sources):
            derived.append(name)
        else:
            have = [s for s in sources if s in cols]
            missing.append(f"{name}(유도 불가: {sources} 중 {have} 만 있음)")

    known = (REQUIRED_COLUMNS | OPTIONAL_COLUMNS | set(ALIASES)
             | set(DERIVED) | {s for v in DERIVED.values() for s in v}
             | {c for v in ALIASES.values() for c in v})
    unexpected_cols = sorted(cols - known)

    rep = SchemaReport(ok=not missing, n_rows=len(df), missing=missing,
                       unexpected=unexpected_cols, aliased=aliased,
                       derived=derived)
    if unexpected_cols:
        msg = (f"표에 계약에 없는 컬럼 {len(unexpected_cols)}개: "
               f"{unexpected_cols[:12]}{' ...' if len(unexpected_cols) > 12 else ''}\n"
               "  kernelTab 이 컬럼을 추가한 것일 수 있다. 정답에서 유도된 값이면 "
               "kernelTab 의 ANSWER_COLS 에 들어가야 하고, 피처면 여기 "
               "OPTIONAL_COLUMNS 에 넣어라. 그전까지는 무시된다.")
        if unexpected == "raise":
            raise SchemaError(msg)
        if unexpected == "warn":
            warnings.warn(msg, stacklevel=2)
    return rep


def normalize(df: pd.DataFrame, *, unexpected: str = "warn") -> pd.DataFrame:
    """표를 우리 이름으로 정규화한다. 별칭을 풀고 유도 컬럼을 만든다.

    ⚠️ **정답 컬럼을 만들지도 옮기지도 않는다.** 입력은 `load_for_ranking`
    결과여야 한다. `time_ms` 가 섞여 있으면 여기서 터진다 — 어댑터가 정답을
    통과시키는 경로가 되면 §3 의 격리가 무의미해진다.
    """
    from kerneltab.core.table import ANSWER_COLS

    leaked = sorted(set(df.columns) & set(ANSWER_COLS))
    if leaked:
        raise SchemaError(
            f"normalize() 에 정답 컬럼이 들어왔다: {leaked}\n"
            "  load_for_ranking() 결과를 넘겨라. 어댑터는 정답을 통과시키지 "
            "않는다 (§3.2).")

    check_schema(df, unexpected=unexpected).raise_if_bad()

    out = df
    renames = {}
    for name in sorted(set(REQUIRED_COLUMNS) | set(ALIASES)):
        got = _resolve(df, name)
        if got is not None and got != name:
            renames[got] = name
    if renames:
        out = out.rename(columns=renames)

    for name, sources in DERIVED.items():
        if name in out.columns:
            continue
        if name == "spill_bytes":
            out = out.assign(spill_bytes=(out["spill_stores"].fillna(0)
                                          + out["spill_loads"].fillna(0)
                                          ).astype("int64"))
        else:  # pragma: no cover - DERIVED 에 항목을 추가하면 여기도 채워라
            raise SchemaError(f"유도 규칙이 구현되지 않았다: {name}")

    for name in ("acc_dtype", "layout_a", "layout_b", "layout_c"):
        if name not in out.columns:
            default = {"acc_dtype": "f32", "layout_a": "row",
                       "layout_b": "col", "layout_c": "row"}[name]
            out = out.assign(**{name: default})
    return out
