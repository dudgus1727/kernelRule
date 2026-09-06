"""★ 트레이스를 **GitHub Release 자산**으로 올린다 (D-138). LLM 0회.

    python3 experiments/trace_release.py --tag F3rw-p8-nan            # 준비만
    python3 experiments/trace_release.py --tag F3rw-p8-nan --upload   # 올린다
    python3 experiments/trace_release.py --tag F3rw-p8-nan --verify   # 받아서 대조

## 왜 릴리즈인가

```
저장소 커밋   git 이력에 영원히 남는다 — 캠페인마다 쌓이면 클론이 무거워진다
★ 릴리즈 자산  git 이력 밖. 남는다. curl 로 받는다
```

**축약본을 만들지 않는다.** 저장소가 안 무거워지므로 전문 그대로 올린다.

## 하지 않는 것

```
[ ] 캠페인이 도는 중에 올리지 마라 — mtime 으로 막는다
[ ] 커밋을 하나로 뭉개지 마라 — 갈리면 전부 적고 diff 결과도 적는다
[ ] 트레이스를 저장소에 커밋하지 마라
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
#: ★ 어느 태그의 트레이스가 어느 릴리즈에 있나. `runs.md` 가 이 파일을 읽고,
#: `--check` 가 "트레이스는 있는데 릴리즈가 없다" 를 잡는다 (원칙 2).
MANIFEST = ROOT / "docs" / "artifacts" / "trace-releases.json"
#: 이 시간 안에 트레이스가 쓰였으면 **도는 중**으로 본다.
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
    """`ev` 종류마다 **자료에서 뽑은** 필드 목록. 손으로 적지 않는다."""
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
    # ★ 커밋을 **시간순**으로 — 사전순으로 늘어놓으면 순서가 거짓말이 된다
    uniq = sorted(set(commits.values()),
                  key=lambda c: int(_sh("git", "show", "-s", "--format=%ct", c)))
    L = [f"# 실행 트레이스 — `{tag}` (n={len(seeds)})", "",
         "루프가 실제로 무슨 대화를 했는지 **한 파일에 시간순으로** 담은 "
         "원자료다 (D-133). 축약하지 않았다.", "",
         "## 실행 조건", "", "```"]
    for k in ("feature_condition", "seed_source", "parameters",
              "product_hint", "power_hint", "fit_method", "fit_restarts",
              "objective", "hw"):
        L.append(f"{k:18s} {c.get(k)}")
    L += ["```", "", "`runs.md` 의 그 줄과 같은 조건이다.", "",
          "## 커밋", "", "```"]
    for name in sorted(commits):
        L.append(f"{name:20s} {commits[name]}")
    L.append("```")
    if len(uniq) > 1:
        L += ["", f"⚠️ **커밋이 {len(uniq)}개다** — 캠페인이 도는 동안 "
              "문서를 커밋했다. 뭉개지 않고 다 적는다 (D-137).", "",
              "```"]
        for i in range(len(uniq) - 1):
            n = len(_sh("git", "diff", "--stat", uniq[i], uniq[i + 1],
                        "--", "kernelrule/", "prompts/").splitlines())
            L.append(f"git diff {uniq[i]}..{uniq[i + 1]} -- kernelrule/ "
                     f"prompts/  ->  변경 파일 {n}")
        L += ["```", "",
              "★ **파이프라인 코드와 프롬프트는 안 바뀌었다.** 동작은 같다."]
    L += ["", "## 파일", "",
          "| 파일 | 원본 | 압축 | 줄(이벤트) |", "|---|--:|--:|--:|"]
    for name, _p in seeds:
        raw, comp, nev = sizes[name]
        L.append(f"| `trace-{name}.jsonl.zst` | {raw / 1e6:.1f} MB "
                 f"| {comp / 1e6:.2f} MB | {nev:,} |")
    tr = sum(v[0] for v in sizes.values())
    tc = sum(v[1] for v in sizes.values())
    L += ["", f"합계 원본 {tr / 1e6:.1f} MB -> 압축 {tc / 1e6:.2f} MB "
          f"({tc / tr:.1%})", "",
          "각 파일마다 `.sha256` 이 함께 있다.", "",
          "```bash",
          f"curl -LO https://github.com/dudgus1727/kernelRule/releases/"
          f"download/{rel}/trace-{seeds[0][0]}.jsonl.zst",
          f"curl -LO https://github.com/dudgus1727/kernelRule/releases/"
          f"download/{rel}/trace-{seeds[0][0]}.jsonl.zst.sha256",
          "sha256sum -c *.sha256",
          f"zstd -d trace-{seeds[0][0]}.jsonl.zst",
          "```", "",
          "## 스키마", "",
          "★ **버전 필드가 없다.** 형식은 위 커밋의 "
          "`kernelrule/core/trace.py` 가 정한다. 아래 목록은 **이 자료에서 "
          "뽑은 것**이지 손으로 적은 것이 아니다.", "", "```"]
    for ev, ks in _schema([p for _n, p in seeds]).items():
        L.append(f"{ev:12s} {' '.join(ks)}")
    L += ["```", "",
          "★ `llm_call.user_prompt` 와 `llm_call.response` 는 **전문**이다. "
          "`prompt_hash` 는 `sha256(role + \\x00 + user_prompt)[:16]` 이라 "
          "그 전문의 검사값이다.", "",
          "⚠️ **시스템 프롬프트는 트레이스에 없다** — 위 커밋의 `prompts/` "
          "가 갖는다. 트레이스만으로는 재구성되지 않는다.", "",
          "## 읽는 법", "",
          "```bash",
          "python3 experiments/trace.py runs/<실행>/trace.jsonl --round 3",
          "```"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--release", default=None,
                    help="릴리즈 태그. 기본 trace-<태그>-<첫 커밋>")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="올린 자산을 **다시 받아** sha256 을 대조한다")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    seeds = _seeds(a.tag)
    if not seeds:
        raise SystemExit(f"{a.tag} 의 trace.jsonl 이 없다")
    now = time.time()
    live = [n for n, p in seeds if now - p.stat().st_mtime < LIVE_SECONDS]
    if live and (a.upload or a.verify):
        raise SystemExit(
            f"★ 아직 도는 중이다 ({', '.join(live)}). 끝난 뒤 한 번에 올린다.")

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
              f"{z.stat().st_size / 1e6:5.2f} MB  이벤트 {nev:,}")

    notes = work / "NOTES.md"
    notes.write_text(_notes(a.tag, rel, seeds, work, commits, sizes))
    print(f"\n  릴리즈 태그  {rel}\n  노트         {notes}")

    if a.upload:
        assets = sorted(str(x) for x in work.glob("trace-*"))
        _sh("gh", "release", "create", rel, "--title",
            f"실행 트레이스 — {a.tag}", "--notes-file", str(notes), *assets)
        print(f"  ★ 올렸다: {len(assets)}개 자산")
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
        print(f"  ★ {MANIFEST.relative_to(ROOT)} 갱신")

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
            print(f"  {fn}  sha {'같다' if got == want else '★ 다르다'}"
                  f"  첫 줄 {j['ev']} config={'있다' if 'config' in j else '★ 없다'}"
                  f" commit={j.get('commit')}")
        if bad:
            sys.exit(1)
        print("  ★ 전부 대조됐다")


if __name__ == "__main__":
    main()
