"""★ It unfolds a run trace **for a human to read** (D-133). 0 LLM calls.

    python3 experiments/trace.py <run> --round 7
    python3 experiments/trace.py <run> --round 7 --full   # the full prompts too
    python3 experiments/trace.py <run> --summary

In `trace.jsonl` one line is one event, in time order. This tool **only
reads** — what to count is decided by a human after reading (the instruction
§5).
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
            f"{p} does not exist. A trace only exists for runs after D-133 — "
            "for an old run, look at llm_calls/ and rounds.jsonl.")
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _fold(txt: str, n: int = 240) -> str:
    txt = (txt or "").strip()
    return txt if len(txt) <= n else txt[:n] + f" … (+{len(txt) - n} chars)"


def show_round(evs: list[dict], r: int, *, full: bool) -> None:
    sub = [e for e in evs if e.get("round") == r]
    if not sub:
        raise SystemExit(f"there is no event for round {r}.")
    start = next((e for e in sub if e["ev"] == "round_start"), {})
    end = next((e for e in sub if e["ev"] == "round_end"), {})
    print("=" * 92)
    print(f"round {r}   start best {start.get('archive_best')}  "
          f"cells {start.get('cells')}   ->   end best {end.get('best')}  "
          f"cells {end.get('cells')}   {end.get('seconds')}s")
    print("=" * 92)

    for e in sub:
        if e["ev"] == "hypotheses":
            print(f"\n  {len(e.get('ids') or [])} hypotheses"
                  + (f" ({e['n_replaced']} replaced on re-entry)"
                     if e.get("n_replaced") else ""))
            for hid, claim in zip(e.get("ids") or [],
                                  e.get("claims") or [], strict=False):
                print(f"    {hid}  {_fold(claim, 400 if full else 160)}")
        elif e["ev"] == "parents":
            print("\n  parent assignment")
            for pk in e.get("picks") or []:
                print(f"    {pk['kind']:8s} {pk.get('rules')}")

    if full:
        for e in sub:
            if e["ev"] == "llm_call":
                print(f"\n  --- LLM {e['role']} (seq {e['seq']}, "
                      f"{e.get('n_in')}->{e.get('n_out')} tokens, "
                      f"{e.get('ms')}ms) ---")
                print("  [user prompt]")
                print("  " + (e.get("user_prompt") or "").replace("\n", "\n  "))
                print("  [response]")
                print("  " + json.dumps(e.get("response"), ensure_ascii=False,
                                        indent=1).replace("\n", "\n  "))

    # The proposals -> the outcomes, joined one line each
    props = {e["i"]: e for e in sub if e["ev"] == "proposal"}
    rej = {e.get("i"): e for e in sub if e["ev"] == "reject"}
    dup = {e.get("i"): e for e in sub if e["ev"] == "duplicate"}
    scored = {e["code_sha"]: e for e in sub if e["ev"] == "scored"}
    arch = {e["rule"]: e for e in sub if e["ev"] == "archive"}
    print(f"\n  {len(props)} proposals")
    print(f"  {'i':>2s} {'parent':8s} {'hyp':5s} {'outcome':10s} "
          f"{'train':>8s} {'val':>8s} {'cell':>10s}  changes")
    for i in sorted(set(props) | set(rej) | set(dup)):
        p = props.get(i, {})
        sha = p.get("code_sha")
        sc = scored.get(sha)
        a = arch.get(sc["rule"]) if sc else None
        if i in rej:
            out, fit, val, cell = f"refused:{rej[i].get('why')}", "", "", ""
        elif i in dup:
            out, fit, val, cell = "duplicate", "", "", ""
        elif sc is None:
            out, fit, val, cell = "not scored", "", "", ""
        else:
            out = "★ accepted" if (a and a.get("accepted")) else "dropped"
            fit = f"{sc['fit']:.4f}"
            val = ("" if sc.get("val") is None
                   else f"{sc['val']:.4f}")
            cell = str(a.get("cell")) if a else ""
        print(f"  {i:2d} {p.get('kind', '?'):8s} {str(p.get('hyp') or '-'):5s} "
              f"{out:10s} {fit:>8s} {val:>8s} {cell:>10s}  "
              f"{_fold(p.get('changes'), 400 if full else 70)}")
        if i in rej and rej[i].get("detail"):
            print(f"     ⛔ {_fold(rej[i]['detail'], 300)}")
    print(f"\n  calls {end.get('calls')}   proposed {end.get('proposed')} / "
          f"scored {end.get('scored')} / accepted {end.get('accepted')} / "
          f"refused {end.get('rejected')}")
    if not full:
        print("  ★ the full prompts are behind `--full`")


def summary(evs: list[dict]) -> None:
    start = evs[0]
    print("=" * 92)
    print(f"{start.get('run_id')}   commit {start.get('commit')}   "
          f"rounds {start.get('n_rounds')}   split {start.get('split')} "
          f"({start.get('n_train')}/{start.get('n_val')})   "
          f"features {len(start.get('features') or [])}")
    print("=" * 92)
    print(f"  {'r':>2s} {'call':>4s} {'prop':>4s} {'scor':>4s} {'acpt':>4s} "
          f"{'refu':>4s} {'dupl':>4s} {'best':>9s} {'cel':>3s}")
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
        print("  refusal reasons:", dict(why))
    kinds = Counter(e.get("kind") for e in evs if e["ev"] == "duplicate")
    if kinds:
        print("  the parent kinds duplicates came from:", dict(kinds))
    print("  ★ what else to count is decided after reading (D-133 §6)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--round", type=int)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="unfold the full prompts and responses too")
    a = ap.parse_args()
    evs = _load(a.run)
    if a.round is not None:
        show_round(evs, a.round, full=a.full)
    else:
        summary(evs)


if __name__ == "__main__":
    main()
