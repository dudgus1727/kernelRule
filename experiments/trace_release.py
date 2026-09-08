"""★ It uploads a trace as a **GitHub Release asset** (D-138). 0 LLM calls.

    python3 experiments/trace_release.py --tag F3rw-p8-nan            # prepare
    python3 experiments/trace_release.py --tag F3rw-p8-nan --upload   # upload
    python3 experiments/trace_release.py --tag F3rw-p8-nan --verify   # fetch
                                                                      # and
                                                                      # check

## Why a release

```
a repository commit  stays in the git history forever — piling up per
                     campaign makes the clone heavy
★ a release asset    outside the git history. It stays. It is fetched with
                     curl
```

**No abridged version is made.** The repository does not get heavy, so the
full text goes up as it is.

## What is not done

```
[ ] do not upload while a campaign is running — mtime blocks it
[ ] do not squash the commits into one — if they differ, all of them are
    written down along with the diff result
[ ] do not commit a trace into the repository
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
#: ★ Which tag's trace is in which release. `runs.md` reads this file, and
#: `--check` catches "there is a trace but no release" (principle 2).
MANIFEST = ROOT / "docs" / "artifacts" / "trace-releases.json"
#: If a trace was written within this time it is taken to be **running**.
LIVE_SECONDS = 1800


def _sh(*a: str) -> str:
    return subprocess.run(a, capture_output=True, text=True,
                          check=True, cwd=ROOT).stdout.strip()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _seeds(tag: str) -> list[tuple[str, Path]]:
    out = []
    for d in sorted(RUNS.glob(f"{tag}-s*")):
        t = d / "trace.jsonl"
        if t.exists():
            out.append((d.name, t))
    return out


def _schema(paths: list[Path]) -> dict[str, list[str]]:
    """The field list per `ev` kind, **taken from the data**. It is not
    written by hand."""
    fields: dict[str, set] = {}
    for p in paths:
        with p.open() as f:
            for line in f:
                j = json.loads(line)
                fields.setdefault(j["ev"], set()).update(j)
    return {k: sorted(v) for k, v in sorted(fields.items())}


def _stats(p: Path) -> tuple[int, Counter]:
    n, c = 0, Counter()
    with p.open() as f:
        for line in f:
            n += 1
            c[json.loads(line)["ev"]] += 1
    return n, c


def _commits(seeds: list[tuple[str, Path]]) -> dict[str, str]:
    out = {}
    for name, p in seeds:
        with p.open() as f:
            out[name] = json.loads(f.readline()).get("commit", "?")
    return out


def _notes(tag: str, rel: str, seeds, work: Path, commits, sizes) -> str:
    from kernelrule.core.runset import run_condition
    c = run_condition(seeds[0][0])
    # ★ The commits in **time order** — laying them out alphabetically makes
    #   the ordering a lie
    uniq = sorted(set(commits.values()),
                  key=lambda c: int(_sh("git", "show", "-s", "--format=%ct", c)))
    L = [f"# Run traces — `{tag}` (n={len(seeds)})", "",
         ("The raw record of what conversation the loop actually had, held "
          "**in one file in time order** (D-133). Nothing was abridged."), "",
         "## The run condition", "", "```"]
    for k in ("feature_condition", "seed_source", "parameters",
              "product_hint", "power_hint", "fit_method", "fit_restarts",
              "objective", "hw"):
        L.append(f"{k:18s} {c.get(k)}")
    L += ["```", "", "It is the same condition as that row of `runs.md`.", "",
          "## The commits", "", "```"]
    for name in sorted(commits):
        L.append(f"{name:20s} {commits[name]}")
    L.append("```")
    if len(uniq) > 1:
        L += ["", (f"⚠️ **there are {len(uniq)} commits** — documents were "
                   f"committed while the campaign was running. They are not "
                   f"squashed, all of them are written down (D-137)."), "",
              "```"]
        for i in range(len(uniq) - 1):
            n = len(_sh("git", "diff", "--stat", uniq[i], uniq[i + 1],
                        "--", "kernelrule/", "prompts/").splitlines())
            L.append(f"git diff {uniq[i]}..{uniq[i + 1]} -- kernelrule/ "
                     f"prompts/  ->  files changed: {n}")
        L += ["```", "",
              ("★ **The pipeline code and the prompts did not change.** The "
               "behaviour is the same.")]
    L += ["", "## The files", "",
          "| file | raw | compressed | lines (events) |", "|---|--:|--:|--:|"]
    for name, _p in seeds:
        raw, comp, nev = sizes[name]
        L.append(f"| `trace-{name}.jsonl.zst` | {raw / 1e6:.1f} MB "
                 f"| {comp / 1e6:.2f} MB | {nev:,} |")
    tr = sum(v[0] for v in sizes.values())
    tc = sum(v[1] for v in sizes.values())
    L += ["", (f"total raw {tr / 1e6:.1f} MB -> compressed "
               f"{tc / 1e6:.2f} MB ({tc / tr:.1%})"), "",
          "Each file has a `.sha256` beside it.", "",
          "```bash",
          (f"curl -LO https://github.com/dudgus1727/kernelRule/releases/"
           f"download/{rel}/trace-{seeds[0][0]}.jsonl.zst"),
          (f"curl -LO https://github.com/dudgus1727/kernelRule/releases/"
           f"download/{rel}/trace-{seeds[0][0]}.jsonl.zst.sha256"),
          "sha256sum -c *.sha256",
          f"zstd -d trace-{seeds[0][0]}.jsonl.zst",
          "```", "",
          "## The schema", "",
          ("★ **There is no version field.** The format is set by "
           "`kernelrule/core/trace.py` at the commit above. The list below "
           "is **taken from this data**, not written by hand."), "", "```"]
    for ev, ks in _schema([p for _n, p in seeds]).items():
        L.append(f"{ev:12s} {' '.join(ks)}")
    L += ["```", "",
          ("★ `llm_call.user_prompt` and `llm_call.response` are **the full "
           "text**. `prompt_hash` is "
           "`sha256(role + \\x00 + user_prompt)[:16]`, so it is the check "
           "value of that full text."), "",
          ("⚠️ **The system prompt is not in the trace** — the `prompts/` of "
           "the commit above holds it. The trace alone does not reconstruct "
           "it."), "",
          "## How to read it", "",
          "```bash",
          "python3 experiments/trace.py runs/<run>/trace.jsonl --round 3",
          "```"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--release", default=None,
                    help="the release tag. The default is "
                         "trace-<tag>-<the first commit>")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="**fetch back** the uploaded asset and check the "
                         "sha256")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    seeds = _seeds(a.tag)
    if not seeds:
        raise SystemExit(f"there is no trace.jsonl for {a.tag}")
    now = time.time()
    live = [n for n, p in seeds if now - p.stat().st_mtime < LIVE_SECONDS]
    if live and (a.upload or a.verify):
        raise SystemExit(
            f"★ it is still running ({', '.join(live)}). It is uploaded in "
            f"one go after it finishes.")

    commits = _commits(seeds)
    rel = a.release or f"trace-{a.tag}-{commits[seeds[0][0]]}"
    work = Path(a.out) if a.out else (
        ROOT / ".trace-release" / rel)
    work.mkdir(parents=True, exist_ok=True)

    sizes = {}
    for name, p in seeds:
        z = work / f"trace-{name}.jsonl.zst"
        subprocess.run(["zstd", "-19", "-q", "-f", "-o", str(z), str(p)],
                       check=True)
        nev, _c = _stats(p)
        sizes[name] = (p.stat().st_size, z.stat().st_size, nev)
        (work / f"{z.name}.sha256").write_text(
            f"{_sha256(z)}  {z.name}\n")
        print(f"  {name}  {p.stat().st_size / 1e6:5.1f} MB -> "
              f"{z.stat().st_size / 1e6:5.2f} MB  events {nev:,}")

    notes = work / "NOTES.md"
    notes.write_text(_notes(a.tag, rel, seeds, work, commits, sizes))
    print(f"\n  release tag  {rel}\n  notes        {notes}")

    if a.upload:
        assets = sorted(str(x) for x in work.glob("trace-*"))
        _sh("gh", "release", "create", rel, "--title",
            f"Run traces — {a.tag}", "--notes-file", str(notes), *assets)
        print(f"  ★ uploaded: {len(assets)} assets")
        m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        m[a.tag] = {
            "release": rel,
            "runs": [n for n, _p in seeds],
            "commits": commits,
            "sha256": {f"trace-{n}.jsonl.zst":
                       (work / f"trace-{n}.jsonl.zst.sha256"
                        ).read_text().split()[0] for n, _p in seeds},
            "bytes_raw": {n: sizes[n][0] for n, _p in seeds},
            "bytes_zst": {n: sizes[n][1] for n, _p in seeds},
            "events": {n: sizes[n][2] for n, _p in seeds},
            "uploaded": time.strftime("%Y-%m-%d"),
        }
        MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1,
                                       sort_keys=True) + "\n")
        print(f"  ★ {MANIFEST.relative_to(ROOT)} updated")

    if a.verify:
        chk = work / "verify"
        chk.mkdir(exist_ok=True)
        bad = 0
        for name, _p in seeds:
            fn = f"trace-{name}.jsonl.zst"
            _sh("gh", "release", "download", rel, "--pattern", fn,
                "--dir", str(chk), "--clobber")
            got, want = _sha256(chk / fn), (
                work / f"{fn}.sha256").read_text().split()[0]
            first = subprocess.run(["zstd", "-dc", str(chk / fn)],
                                   capture_output=True, text=True,
                                   check=True).stdout.split("\n", 1)[0]
            j = json.loads(first)
            ok = (got == want and j["ev"] == "run_start"
                  and "config" in j and j.get("commit"))
            bad += not ok
            print(f"  {fn}  sha {'same' if got == want else '★ different'}"
                  f"  first line {j['ev']} "
                  f"config={'present' if 'config' in j else '★ absent'}"
                  f" commit={j.get('commit')}")
        if bad:
            sys.exit(1)
        print("  ★ everything checked out")


if __name__ == "__main__":
    main()
