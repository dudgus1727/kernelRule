"""★ 체제 전이 — 구조를 고정하고 다른 체제에 재적합한다. LLM 호출 0회.

    python3 experiments/regime_transfer.py

## 무엇을 가르는 실험인가

"LLM 구조는 전이되지 않고 손규칙만 전이된다" 는 주장이 나왔고, 그 원인으로
"LLM 구조에 체제 특화 항이 있다" 가 지목됐다. **그것은 관찰이지 인과가
아니다.** 제거 실험으로 가른다.

    1-A  LLM 구조에서 체제 특화 항을 빼고 긴 형상에 재적합
         회복하면 -> 그 항들이 원인
    1-B  손규칙에 체제 특화 항을 넣고 짧은 적합 -> 긴 재적합
         나빠지면 -> 인과가 양방향으로 확인

블록 0 이 먼저 "누가 실제로 그 항을 쓰는가" 를 센다. 전제부터 확인하지
않으면 제거 실험이 무엇을 잰 것인지 알 수 없다.

## 결과 (2026-08-21)

**양방향 모두 인과를 부정했다.** `docs/artifacts/structure-transfer.md`.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np

import kernelrule.features.physical  # noqa: F401  — REGISTRY 를 채운다
from kernelrule.baselines.vendor import load_vendor, vendor_order_fn
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.sandbox import compile_rule
from kernelrule.core.scoring import compare, evaluate, evaluate_scores, geomean
from kernelrule.core.splits import _DUMMY_CFG, Split
from kernelrule.core.table import PerfTable
from kernelrule.core.weights import fit_weights, make_score_of
from kernelrule.features import REGISTRY
from kernelrule.features.physical import log_sol_ms
from kernelrule.rules.physics_seeded import CODE as HW
from kernelrule.rules.physics_seeded import W0 as HW_W0

#: 지목된 세 항. `K/tile_k` 가 작을 때만 의미가 있다는 것이 주장의 근거였다.
REGIME_SPECIFIC = frozenset({"is_two_stage", "pipeline_warmup_frac",
                             "log_mainloop_iters"})

BUNDLE = "datasets/rtx-a6000-sm_86-c63710df"
VENDOR = "datasets/baselines/vendor-a6000-c63710df.json"
RUN_REAL = Path("runs/real-gpt-5.4-mini-2026-03-17/archive.jsonl")
RUN_SMOKE = Path("runs/smoke2-gpt-5.4-mini-2026-03-17/archive.jsonl")

#: 긴/짧은 경계 (§30). 0.5 ms 아래에서 눈금 해상도가 지배한다.
SIZE_THRESHOLD_MS = 0.5


# ---------------------------------------------------------------------------
# AST — 항을 세고, 지우고, 가중치 인덱스를 다시 매긴다
# ---------------------------------------------------------------------------

def features_used(node: ast.AST) -> set[str]:
    """`f.<name>` 으로 참조된 피처 이름."""
    return {n.attr for n in ast.walk(node)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "f"}


def weight_indices(node: ast.AST) -> set[int]:
    """`w[i]` 의 상수 인덱스."""
    return {n.slice.value for n in ast.walk(node)
            if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name) and n.value.id == "w"
            and isinstance(n.slice, ast.Constant)}


def _drop_terms(body: list[ast.stmt], banned: frozenset[str]) -> list[ast.stmt]:
    """금지 피처를 참조하는 문장을 지운다. 빈 `if` 는 통째로 사라진다."""
    keep: list[ast.stmt] = []
    for st in body:
        if isinstance(st, ast.If):
            st.body = _drop_terms(st.body, banned)
            if st.body:
                keep.append(st)
        elif isinstance(st, ast.Return) or not (features_used(st) & banned):
            keep.append(st)
    return keep


def ablate(code: str, w0: list[float],
           banned: frozenset[str] = REGIME_SPECIFIC) -> tuple[str, list[float]]:
    """항을 지우고 남은 `w` 인덱스를 0..n-1 로 다시 매긴다.

    ★ 인덱스를 다시 매기지 않으면 `w0` 와 어긋나 **다른 규칙을 재는 것**이
    된다. 그것이 이 함수가 존재하는 이유다.
    """
    tree = ast.parse(code.strip())
    fn = tree.body[0]
    fn.body = _drop_terms(fn.body, banned)
    used = sorted(weight_indices(fn))
    remap = {old: i for i, old in enumerate(used)}

    class _Renumber(ast.NodeTransformer):
        def visit_Subscript(self, n):                       # noqa: N802
            self.generic_visit(n)
            if (isinstance(n.value, ast.Name) and n.value.id == "w"
                    and isinstance(n.slice, ast.Constant)
                    and n.slice.value in remap):
                n.slice = ast.Constant(remap[n.slice.value])
            return n

    tree = _Renumber().visit(tree)
    ast.fix_missing_locations(tree)
    new_w0 = [float(w0[o]) if o < len(w0) else 1.0 for o in used]
    return ast.unparse(tree), new_w0


# ---------------------------------------------------------------------------

def _load_archive(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


#: 손규칙에 지목된 항 둘을 더한 것. `pipeline_warmup_frac` 은 **이미 있다.**
#: 가중치 9개로 §29.4 예산(8)을 넘지만, 이것은 후보가 아니라 **진단용
#: 구성물**이다 — 예산을 지키면 다른 항을 빼야 해서 무엇을 잰 것인지
#: 흐려진다.
HW_PLUS = """
def score(f, p, hw, w):
    s  = np.log2(f.traffic_amplification) * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + f.smem_pressure * w[2]
    s = s + f.has_spill * w[3]
    s = s + f.split_k_cost * w[4]
    s = s + f.pipeline_warmup_frac * w[5]
    s = s + f.is_two_stage * w[7]
    s = s + f.log_mainloop_iters * w[8]
    if p.is_memory_bound:
        s = s + np.log2(f.traffic_amplification) * w[6]
    return s
"""


def main() -> None:                                          # noqa: PLR0915
    table = PerfTable.from_bundle(BUNDLE, env_hash="c63710df", ok_only=False)
    matrix = FeatureMatrix(table, REGISTRY)

    def aligned(p) -> bool:
        d = table.frame_for(p)
        return bool((d.align_a == 8).all() and (d.align_b == 8).all()
                    and (d.align_c == 8).all())

    all_shapes = [p for p in table.shapes() if aligned(p)]
    thr = math.log2(SIZE_THRESHOLD_MS)
    short = [p for p in all_shapes
             if log_sol_ms(p, table.hw, _DUMMY_CFG) < thr]
    long_ = [p for p in all_shapes if p not in short]
    vendor = load_vendor(VENDOR)

    real = _load_archive(RUN_REAL)
    llm_train = min(real, key=lambda e: e["regret"])
    llm_val = min((e for e in real if np.isfinite(e["val_regret"])),
                  key=lambda e: e["val_regret"])
    llm_smoke = min(_load_archive(RUN_SMOKE), key=lambda e: e["regret"])

    def refit(code, w0, shapes):
        return fit_weights(compile_rule(code), matrix, table,
                           Split("train", tuple(shapes)), w0, max_evals=300)

    def score(code, w, shapes, label=""):
        return evaluate_scores(make_score_of(compile_rule(code), matrix, w),
                               table, shapes, ks=(1,), label=label)

    def vend(shapes):
        return evaluate(vendor_order_fn(table, vendor, mapping="nearest"),
                        table, shapes, ks=(1,), label="벤더")

    v_long, v_short, v_all = vend(long_), vend(short), vend(all_shapes)

    cands = [("손규칙(7항)", HW, HW_W0),
             ("LLM 스모크(8항)", llm_smoke["code"], llm_smoke["w"]),
             ("LLM 검증최고(16항)", llm_val["code"], llm_val["w"]),
             ("LLM 학습최고(19항)", llm_train["code"], llm_train["w"])]

    # -- 0. 전제 확인 -------------------------------------------------------
    print("=" * 76)
    print("0. 어떤 규칙이 체제 특화 항을 실제로 쓰는가")
    print("=" * 76)
    for name, code, _ in cands:
        used = features_used(ast.parse(code.strip())) & REGIME_SPECIFIC
        print(f"  {name:22s} 체제특화 {sorted(used)}")

    # -- 1-A. 제거가 회복시키는가 ------------------------------------------
    print(f"\n{'=' * 76}")
    print("1-A. 체제 특화 항 제거 -> 긴 20 재적합   (제거가 회복시키는가)")
    print("=" * 76)
    print(f"  {'구조':22s} {'원본 긴20':>10} {'제거후':>10} {'항수':>9}")
    for name, code, w0 in cands:
        fit = refit(code, w0, long_)
        base = score(code, fit.w, long_).at(1)
        abl_code, abl_w0 = ablate(code, list(fit.w))
        if not abl_w0:
            print(f"  {name:22s} {base:10.4f}   (전부 제거되어 규칙이 빈다)")
            continue
        fit2 = refit(abl_code, abl_w0, long_)
        after = score(abl_code, fit2.w, long_).at(1)
        print(f"  {name:22s} {base:10.4f} {after:10.4f} "
              f"{len(w0):3d}→{len(abl_w0):<3d}")
    print(f"  {'벤더':22s} {v_long.at(1):10.4f}")

    # -- 1-B. 추가가 망가뜨리는가 (인과의 반대 방향) ------------------------
    print(f"\n{'=' * 76}")
    print("1-B. 손규칙에 체제 특화 항 추가 -> 짧은 적합 -> 긴 재적합")
    print("=" * 76)
    # ★ 짧은 형상에 적합한 뒤 **그 가중치에서 출발해** 긴 형상에 재적합한다.
    #   §29.5 (b) 가 말하는 절차 그대로다 — 구조는 그대로, 가중치만 새 체제로.
    fit_s = refit(HW, HW_W0, short)
    hw_s = score(HW, fit_s.w, short)
    hw_l = score(HW, refit(HW, list(fit_s.w), long_).w, long_, "손규칙")
    plus_fit_s = refit(HW_PLUS, [*HW_W0, 0.3, 0.3], short)
    hp_s = score(HW_PLUS, plus_fit_s.w, short)
    hp_l = score(HW_PLUS, refit(HW_PLUS, list(plus_fit_s.w), long_).w,
                 long_, "손규칙+3")
    print(f"  {'손규칙 (7항, 원본)':26s} 짧은41 {hw_s.at(1):.4f}"
          f"   →  긴20 재적합 {hw_l.at(1):.4f}")
    print(f"  {'손규칙+체제특화 (9항)':26s} 짧은41 {hp_s.at(1):.4f}"
          f"   →  긴20 재적합 {hp_l.at(1):.4f}")
    print(f"  {'벤더':26s} 짧은41 {v_short.at(1):.4f}"
          f"   →  긴20      {v_long.at(1):.4f}")

    # -- 2. 체제별 재적합의 전체 61 성능 ------------------------------------
    print(f"\n{'=' * 76}")
    print("2. 손규칙 체제별 재적합의 전체 61 성능")
    print("=" * 76)
    i_s = {p: i for i, p in enumerate(hw_s.shapes)}
    i_l = {p: i for i, p in enumerate(hw_l.shapes)}
    per_shape = np.array([hw_s.regret[i_s[p], 0] if p in i_s
                          else hw_l.regret[i_l[p], 0] for p in all_shapes])
    tol = np.array([hw_s.tol[i_s[p]] if p in i_s else hw_l.tol[i_l[p]]
                    for p in all_shapes])
    is_short = np.array([p in i_s for p in all_shapes])
    hw_one = score(HW, refit(HW, HW_W0, all_shapes).w, all_shapes, "손규칙단일")
    print(f"  {'':28s} {'짧은41':>8} {'긴20':>8} {'전체61':>8}")
    print(f"  {'손규칙 (체제별 재적합)':28s} {hw_s.at(1):8.4f} "
          f"{hw_l.at(1):8.4f} {geomean(per_shape):8.4f}")
    print(f"  {'손규칙 (전체 단일 적합)':28s} "
          f"{hw_one.at(1, mask=is_short):8.4f} "
          f"{hw_one.at(1, mask=~is_short):8.4f} {hw_one.at(1):8.4f}")
    print(f"  {'벤더':28s} {v_short.at(1):8.4f} {v_long.at(1):8.4f} "
          f"{v_all.at(1):8.4f}")
    print("  ★ 관문 1.080")

    # -- 3. 유의성 판정 (§30.6) --------------------------------------------
    print(f"\n{'=' * 76}")
    print("3. 유의성 판정 — 형상별 (t_A - t_B) / (t_최적 x noise_floor)")
    print("=" * 76)
    from dataclasses import replace
    hw_mix = replace(hw_one, regret=per_shape.reshape(-1, 1), tol=tol,
                     label="손규칙(체제별)")
    pairs = [
        ("손규칙(체제별 재적합)  vs 벤더  전체61", hw_mix, v_all),
        ("손규칙(긴 재적합)      vs 벤더  긴20", hw_l, v_long),
        ("손규칙(짧은 적합)      vs 벤더  짧은41", hw_s, v_short),
        ("LLM 학습최고           vs 벤더  짧은41",
         score(llm_train["code"], llm_train["w"], short), v_short),
        ("LLM 학습최고           vs 벤더  전체61",
         score(llm_train["code"], llm_train["w"], all_shapes), v_all),
    ]
    for label, a, b in pairs:
        c = compare(a, b, table, name_a="A", name_b="벤더")
        print(f"  {label:38s} {c.geo_a:.4f} vs {c.geo_b:.4f}  "
              f"이김 {int(c.a_wins.sum()):2d} / 짐 {int(c.a_loses.sum()):2d}"
              f" / 구분불가 {int(c.tied.sum()):2d}")


if __name__ == "__main__":
    main()
