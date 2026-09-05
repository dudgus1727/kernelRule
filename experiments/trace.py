"""★ 실행 트레이스를 **사람이 읽게** 편다 (D-133). LLM 0회.

    python3 experiments/trace.py <run> --round 7
    python3 experiments/trace.py <run> --round 7 --full   # 프롬프트 전문까지
    python3 experiments/trace.py <run> --summary

`trace.jsonl` 은 한 줄이 한 사건이고 시간순이다. 이 도구는 **읽기만**
한다 — 세는 것은 사람이 읽어보고 정한다 (지시문 §5).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load(run: str) -> list[dict]:
    p = Path("runs") / run / "trace.jsonl"
    if not p.exists():
        raise SystemExit(
            f"{p} 가 없다. 트레이스는 D-133 이후 실행에만 있다 — "
            "옛 실행은 llm_calls/ 와 rounds.jsonl 을 봐라.")
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _fold(txt: str, n: int = 240) -> str:
    txt = (txt or "").strip()
    return txt if len(txt) <= n else txt[:n] + f" … (+{len(txt) - n}자)"


def show_round(evs: list[dict], r: int, *, full: bool) -> None:
    sub = [e for e in evs if e.get("round") == r]
    if not sub:
        raise SystemExit(f"라운드 {r} 의 사건이 없다.")
    start = next((e for e in sub if e["ev"] == "round_start"), {})
    end = next((e for e in sub if e["ev"] == "round_end"), {})
    print("=" * 92)
    print(f"라운드 {r}   시작 best {start.get('archive_best')}  "
          f"셀 {start.get('cells')}   ->   끝 best {end.get('best')}  "
          f"셀 {end.get('cells')}   {end.get('seconds')}초")
    print("=" * 92)

    for e in sub:
        if e["ev"] == "hypotheses":
            print(f"\n  가설 {len(e.get('ids') or [])}개"
                  + (f" (되돌아가 교체 {e['n_replaced']}개)"
                     if e.get("n_replaced") else ""))
            for hid, claim in zip(e.get("ids") or [],
                                  e.get("claims") or [], strict=False):
                print(f"    {hid}  {_fold(claim, 400 if full else 160)}")
        elif e["ev"] == "parents":
            print("\n  부모 배정")
            for pk in e.get("picks") or []:
                print(f"    {pk['kind']:8s} {pk.get('rules')}")

    if full:
        for e in sub:
            if e["ev"] == "llm_call":
                print(f"\n  --- LLM {e['role']} (seq {e['seq']}, "
                      f"{e.get('n_in')}->{e.get('n_out')} 토큰, "
                      f"{e.get('ms')}ms) ---")
                print("  [사용자 프롬프트]")
                print("  " + (e.get("user_prompt") or "").replace("\n", "\n  "))
                print("  [응답]")
                print("  " + json.dumps(e.get("response"), ensure_ascii=False,
                                        indent=1).replace("\n", "\n  "))

    # 제안 -> 결과를 한 줄씩 잇는다
    props = {e["i"]: e for e in sub if e["ev"] == "proposal"}
    rej = {e.get("i"): e for e in sub if e["ev"] == "reject"}
    dup = {e.get("i"): e for e in sub if e["ev"] == "duplicate"}
    scored = {e["code_sha"]: e for e in sub if e["ev"] == "scored"}
    arch = {e["rule"]: e for e in sub if e["ev"] == "archive"}
    print(f"\n  제안 {len(props)}개")
    print(f"  {'i':>2s} {'부모종류':8s} {'가설':5s} {'결과':10s} "
          f"{'학습':>8s} {'검증':>8s} {'셀':>10s}  changes")
    for i in sorted(set(props) | set(rej) | set(dup)):
        p = props.get(i, {})
        sha = p.get("code_sha")
        sc = scored.get(sha)
        a = arch.get(sc["rule"]) if sc else None
        if i in rej:
            out, fit, val, cell = f"거부:{rej[i].get('why')}", "", "", ""
        elif i in dup:
            out, fit, val, cell = "중복", "", "", ""
        elif sc is None:
            out, fit, val, cell = "채점못함", "", "", ""
        else:
            out = "★ 채택" if (a and a.get("accepted")) else "탈락"
            fit = f"{sc['fit']:.4f}"
            val = ("" if sc.get("val") is None
                   else f"{sc['val']:.4f}")
            cell = str(a.get("cell")) if a else ""
        print(f"  {i:2d} {p.get('kind', '?'):8s} {str(p.get('hyp') or '-'):5s} "
              f"{out:10s} {fit:>8s} {val:>8s} {cell:>10s}  "
              f"{_fold(p.get('changes'), 400 if full else 70)}")
        if i in rej and rej[i].get("detail"):
            print(f"     ⛔ {_fold(rej[i]['detail'], 300)}")
    print(f"\n  호출 {end.get('calls')}   제안 {end.get('proposed')} / "
          f"채점 {end.get('scored')} / 채택 {end.get('accepted')} / "
          f"거부 {end.get('rejected')}")
    if not full:
        print("  ★ 프롬프트 전문은 `--full` 로")


def summary(evs: list[dict]) -> None:
    start = evs[0]
    print("=" * 92)
    print(f"{start.get('run_id')}   커밋 {start.get('commit')}   "
          f"라운드 {start.get('n_rounds')}   분할 {start.get('split')} "
          f"({start.get('n_train')}/{start.get('n_val')})   "
          f"피처 {len(start.get('features') or [])}개")
    print("=" * 92)
    print(f"  {'r':>2s} {'호출':>4s} {'제안':>4s} {'채점':>4s} {'채택':>4s} "
          f"{'거부':>4s} {'중복':>4s} {'best':>9s} {'셀':>3s}")
    for e in evs:
        if e["ev"] != "round_end":
            continue
        r = e["round"]
        dups = sum(1 for x in evs
                   if x["ev"] == "duplicate" and x.get("round") == r)
        calls = sum((e.get("calls") or {}).values())
        print(f"  {r:2d} {calls:4d} {e.get('proposed', 0):4d} "
              f"{e.get('scored', 0):4d} {e.get('accepted', 0):4d} "
              f"{e.get('rejected', 0):4d} {dups:4d} "
              f"{(e.get('best') or float('nan')):9.4f} {e.get('cells', 0):3d}")
    print()
    why = Counter(e.get("why") for e in evs if e["ev"] == "reject")
    if why:
        print("  거부 사유:", dict(why))
    kinds = Counter(e.get("kind") for e in evs if e["ev"] == "duplicate")
    if kinds:
        print("  중복이 나온 부모 종류:", dict(kinds))
    print("  ★ 무엇을 더 셀지는 읽어보고 정한다 (D-133 §6)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--round", type=int)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="프롬프트·응답 전문까지 편다")
    a = ap.parse_args()
    evs = _load(a.run)
    if a.round is not None:
        show_round(evs, a.round, full=a.full)
    else:
        summary(evs)


if __name__ == "__main__":
    main()
