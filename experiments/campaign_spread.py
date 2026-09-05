"""★ 캠페인 산포 — 같은 조건을 두 번 돌리면 얼마나 벌어지나. LLM 0회 (D-136).

    python3 experiments/campaign_spread.py

`round_curve.py` 가 이미 만든 곡선 원자료만 읽는다 (표도 안 읽는다).

## 왜 필요한가

지금 판정선 `delta = 0.0516` 은 **한 캠페인 안의 시드 폭** σ 로 만들었다.
그런데 우리가 판정에 쓰는 비교의 상당수는 **다른 캠페인끼리**다. 캠페인
효과가 있으면 그 성분은 시드를 늘려도 안 줄어든다 — 판정선의 **바닥**이다.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

ART = Path("docs/artifacts")
# (이름, 곡선 파일, 그 파일 안의 그룹 키)
CAMPAIGNS = [
    ("옛", "round-curve.json", None),
    ("p3", "round-curve-p3.json", None),
    ("새24", "round-curve-new.json", None),
]
# 검정력 계수: 양측 0.05 + 검정력 0.8 -> z(0.975) + z(0.8)
ZSUM = 2.8016


def _curves(fn: str, key: str | None) -> dict[str, list[float]]:
    j = json.loads((ART / fn).read_text())
    g = j["groups"]
    k = key or next(iter(g))
    return g[k]["curves"]


def _at(c: list[float], r: int) -> float:
    return c[min(r, len(c) - 1)]


def _live(run: str) -> int:
    """이 시드가 **실제로 돈** 마지막 라운드.

    ⚠️ 곡선 길이로 재면 안 된다 — `archive.jsonl` 은 **개선될 때만** 줄을
    쓰므로, 마지막 라운드에 개선이 없던 시드가 "일찍 멈춘 것" 으로 보인다.
    라운드를 세는 곳은 `rounds.jsonl` 하나뿐이다 (원칙 2).
    """
    rs = [json.loads(x) for x
          in (Path("runs") / run / "rounds.jsonl").read_text().splitlines()
          if x.strip()]
    return max(r["round"] for r in rs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "campaign-spread.json"))
    a = ap.parse_args()

    cur = {name: _curves(fn, k) for name, fn, k in CAMPAIGNS}
    print("=" * 88)
    print("1. 캠페인마다 몇 라운드까지 **모든 시드가 살아 있나**")
    print("=" * 88)
    live = {}
    for name, cs in cur.items():
        ends = {s: _live(s) for s in cs}
        live[name] = min(ends.values())
        print(f"  {name:5s} 시드별 마지막 라운드 "
              + " ".join(f"{s[-2:]}:r{e}" for s, e in sorted(ends.items()))
              + f"   -> 전원 생존 r{live[name]}")

    # ★ 같은 프롬프트·같은 조건인 두 캠페인은 p3 와 새24 뿐이다
    r = min(live["p3"], live["새24"])
    print(f"\n  ★ p3 와 새24 는 **patience 말고 조건이 같다**. 둘 다 살아 있는 r{r}"
          " 에서 비교한다")

    print("\n" + "=" * 88)
    print(f"2. r{r} 에서 — 캠페인 안 산포와 캠페인 사이 차이")
    print("=" * 88)
    vals = {}
    for name in ("p3", "새24"):
        v = sorted(_at(c, r) for c in cur[name].values())
        vals[name] = v
        print(f"  {name:5s} n={len(v)}  " + " ".join(f"{x:.4f}" for x in v))
        print(f"        평균 {st.mean(v):.4f}  중앙 {st.median(v):.4f}"
              f"  σ(시드) {st.stdev(v):.4f}")
    d_mean = st.mean(vals["새24"]) - st.mean(vals["p3"])
    d_med = st.median(vals["새24"]) - st.median(vals["p3"])
    sw = math.sqrt(sum(st.variance(v) for v in vals.values()) / 2)
    n = len(vals["p3"])
    se_within = sw * math.sqrt(2.0 / n)
    print(f"\n  캠페인 차 (평균)   {d_mean:+.4f}")
    print(f"  캠페인 차 (중앙)   {d_med:+.4f}")
    print(f"  σ(시드) 통합       {sw:.4f}")
    print(f"  ★ 시드 산포만으로 기대되는 차의 표준편차  {se_within:.4f}")
    print(f"     실측 차 / 그 값 = {abs(d_mean) / se_within:.1f}배")

    print("\n" + "=" * 88)
    print("2-2. ★ 세 캠페인을 **전원 생존 라운드**에서 나란히")
    print("=" * 88)
    print(f"  {'라운드':>6s} " + " ".join(f"{k:>22s}" for k in cur)
          + "   (평균 · 중앙 · σ)")
    for rr in (4, 5, 11, 23):
        row = []
        for name, cs in cur.items():
            if rr > min(_live(s) for s in cs):
                row.append(f"{'— (일부 종료)':>22s}")
                continue
            v = [_at(c, rr) for c in cs.values()]
            row.append(f"{st.mean(v):7.4f} {st.median(v):7.4f} {st.stdev(v):6.4f}")
        print(f"  r{rr:<5d} " + " ".join(row))
    print("\n  ⚠️ 일부 시드가 멈춘 캠페인을 그 뒤 라운드에서 비교하면 **멈춘 쪽은")
    print("     값이 고정되고 도는 쪽만 나아진다** — 그 차이는 캠페인 산포가 아니라")
    print("     '한쪽이 그만뒀다' 는 사실이다 (D-136 이 D-135 를 정정하는 자리)")

    print("\n" + "=" * 88)
    print("3. 분산 분해 — 캠페인 성분")
    print("=" * 88)
    # Var(캠페인 평균) = sw^2/n + sc^2 ;  Var(차) = 2(sw^2/n + sc^2)
    # 관측이 **차 하나**뿐이므로 점추정만 가능하다 (자유도 1).
    var_diff = d_mean ** 2
    sc2 = var_diff / 2.0 - sw ** 2 / n
    sc = math.sqrt(sc2) if sc2 > 0 else 0.0
    print(f"  σ(시드)  = {sw:.4f}   -> 평균의 표준오차 {sw / math.sqrt(n):.4f}")
    print(f"  ★ σ(캠페인) 점추정 = {sc:.4f}   (자유도 1 — 구간은 못 준다)")
    print(f"  캠페인 평균의 표준편차 = sqrt(σw²/n + σc²)"
          f" = {math.sqrt(sw ** 2 / n + sc2):.4f}")

    print("\n" + "=" * 88)
    print("4. 판정선 — 무엇이 바뀌나")
    print("=" * 88)
    SIG_HI = 0.0319   # §29.5 가 쓴 σ 95% 상한
    cur_line = ZSUM * SIG_HI * math.sqrt(2.0 / n)
    print(f"  지금 판정선 (시드 σ 상한 {SIG_HI} · n={n} · 비대응)  {cur_line:.4f}")
    new_line = ZSUM * math.sqrt(2.0 * (SIG_HI ** 2 / n + sc2))
    print(f"  ★ 캠페인 성분을 넣으면 (점추정)                    {new_line:.4f}")
    floor = ZSUM * math.sqrt(2.0) * sc
    print(f"  ★ 시드를 무한히 늘려도 남는 **바닥**                {floor:.4f}")
    print("\n  시드 수를 늘렸을 때 판정선:")
    print(f"  {'n':>4s} {'시드만':>9s} {'캠페인 성분 포함':>16s}")
    for m in (3, 6, 12, 24, 96):
        print(f"  {m:4d} {ZSUM * SIG_HI * math.sqrt(2.0 / m):9.4f}"
              f" {ZSUM * math.sqrt(2.0 * (SIG_HI ** 2 / m + sc2)):16.4f}")

    print("\n" + "=" * 88)
    print("5. ⚠️ 이 추정의 한계 — 캠페인 **쌍이 하나**다")
    print("=" * 88)
    # d ~ N(0, tau^2) 을 **한 번** 본 것이므로 tau 의 95% 상한은
    # |d| / sqrt(chi2_{0.05,1}) = |d| / 0.0627 -> 16배. 사실상 정보가 없다.
    print(f"  같은 조건 캠페인 쌍  1개  ->  자유도 1")
    print(f"  σ(캠페인) 95% 상한   {abs(d_mean) / 0.0627:.4f}"
          "   ★ 실측의 16배 — 상한으로 못 쓴다")
    print("  ★ 말할 수 있는 것: **실측 차가 시드 산포만으로 기대되는 폭과 같다**")
    print("     말할 수 없는 것: 캠페인 성분이 작다는 **보장**")
    print("  이것을 재려면 같은 조건 캠페인이 3개 이상 필요하다 (자유도 2+)")

    out = {
        "round": r,
        "seed_values": vals,
        "diff_mean": d_mean, "diff_median": d_med,
        "sigma_within": sw, "sigma_campaign_point": sc,
        "line_current": cur_line, "line_with_campaign": new_line,
        "line_floor": floor, "n": n, "sigma_hi_used": SIG_HI,
        "note": "σ(캠페인) 은 캠페인 쌍 하나에서 나온 점추정이다 (자유도 1)",
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
