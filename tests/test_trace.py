"""★ 실행 트레이스 (D-133) — **로깅이 계산을 안 건드리는가**.

지시문 §4 그대로다: 트레이스는 조건이 아니다. 켜고 끄고 돌려서 산출물이
**같아야** 한다. 다르면 로깅이 무언가를 건드린 것이다.

MockLLM 으로 검사한다 — 결정론이라 **정확히 같아야** 하고 LLM 0회다.
"""
from __future__ import annotations

import json
from pathlib import Path

import kernelrule.features.physical  # noqa: F401
from kernelrule.agents.mock import MockLLM
from kernelrule.core.loop import LoopConfig, RoundLoop
from kernelrule.core.matrix import FeatureMatrix
from kernelrule.core.splits import Split, SplitSet
from kernelrule.features import REGISTRY


def _run(table, tmp_path: Path, *, trace: bool, rounds: int = 2):
    """★ 같은 씨앗·같은 MockLLM 으로 켜고/끄고 (`synth_table` 은 conftest)."""
    m = FeatureMatrix(table, REGISTRY)
    sh = table.shapes()
    sp = SplitSet(train=Split("train", tuple(sh[:-2])),
                  val=Split("val", tuple(sh[-2:])))
    llm = MockLLM("mutate", seed=1, feature_names=m.feature_names(),
                  shape_values=["is_memory_bound"])
    cfg = LoopConfig(run_id=f"t{int(trace)}", max_rounds=rounds,
                     n_rules_per_round=4, max_evals=40, seed=0,
                     sandbox_first_seen=False, out_dir=str(tmp_path),
                     objective="regret", trace=trace)
    loop = RoundLoop(cfg=cfg, table=table, matrix=m, splits=sp, llm=llm)
    loop.run(rounds, verbose=False)
    loop.dump()
    return Path(tmp_path) / cfg.run_id


def test_trace_does_not_change_the_result(synth_table, tmp_path):
    """★ 켜고/끄고 돌려 **아카이브와 라운드 기록이 같은가**.

    같은 씨앗·같은 RNG·MockLLM 이면 정확히 같아야 한다. 다르면 로깅이
    계산 경로에 부수 효과를 냈다는 뜻이다 (지시문 §4).
    """
    off = _run(synth_table, tmp_path, trace=False)
    on = _run(synth_table, tmp_path, trace=True)
    assert (off / "archive.jsonl").read_text() == \
        (on / "archive.jsonl").read_text()

    # ★ `seconds` 는 벽시계라 두 실행이 다를 수밖에 없다 — 계산 결과가
    #   아니다. 그것만 빼고 **나머지 전부**를 견준다.
    def rounds(d):
        return [{k: v for k, v in json.loads(x).items() if k != "seconds"}
                for x in (d / "rounds.jsonl").read_text().splitlines()]

    assert rounds(off) == rounds(on)
    assert not (off / "trace.jsonl").exists()
    assert (on / "trace.jsonl").exists()


def test_trace_first_line_is_self_sufficient(synth_table, tmp_path):
    """★ 첫 줄만 읽어도 조건을 알 수 있는가 (지시문 §3-3)."""
    d = _run(synth_table, tmp_path, trace=True)
    first = json.loads((d / "trace.jsonl").read_text().splitlines()[0])
    assert first["ev"] == "run_start"
    for k in ("commit", "config", "split", "n_train", "n_val", "features"):
        assert k in first, k
    # config 전체 — `config.json` 과 **같은 것**이어야 한다 (원칙 2)
    saved = json.loads((d / "config.json").read_text())
    assert first["config"]["loop"]["seed"] == saved["loop"]["seed"]
    assert first["config"]["rule_constraints"] == saved["rule_constraints"]


def test_trace_records_the_order_and_the_failures(synth_table, tmp_path):
    """★ 순서와 **실패**가 남는가 — 흩어진 파일이 못 하던 것이다."""
    d = _run(synth_table, tmp_path, trace=True)
    evs = [json.loads(x)["ev"]
           for x in (d / "trace.jsonl").read_text().splitlines()]
    assert evs[0] == "run_start"
    assert evs.count("round_start") == evs.count("round_end") == 2
    assert "proposal" in evs and "scored" in evs and "archive" in evs
    # 시간이 단조인가 — 한 파일에 시간순이라는 것이 이 형식의 요점이다
    ts = [json.loads(x)["t"]
          for x in (d / "trace.jsonl").read_text().splitlines()]
    assert ts == sorted(ts)


def test_trace_is_not_a_condition():
    """★ `runset.KEYS` 에 들어가면 안 된다 (지시문 §8)."""
    from kernelrule.core.runset import KEYS

    assert "trace" not in KEYS
