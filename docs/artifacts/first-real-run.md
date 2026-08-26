# 첫 실제 LLM 실행 — dev-cu124 (2026-08-21)

> **상태**: 조건부
> **조건**: gpt-5.4-mini + **블록 3.5 오염 상태** (D-28) + 예산 우회 (19항/8가중치)
> **후속**: 이후 실행 전부


**재현** 이 실행에는 전용 스크립트가 없다 (`experiments/` 도입 전). `runs/real-gpt-5.4-mini-2026-03-17/` 의 `config.json` · `llm_calls/` 가 남아 있다. 채점은 `experiments/rescore_canonical.py`.

> ## ⚠️ 이 실행의 블록 3.5 는 오염된 상태였다 (2026-08-21 확인, D-28)
>
> 진단 리포트에 주입한 `table_facts` 가 **66형상 전수 / a888 61형상**에서
> 계산된 문장이었다. 검증·최종 분할의 집계가 프롬프트에 들어갔다.
> §12.3 이 "홀드아웃 점수" 만 막았고 집계가 빠져나갔다.
>
> **재실행하지 않는다.** LLM 이 그 정보를 얼마나 썼는지는 알 수 없으므로
> **이 실행의 검증 점수 해석에는 유보가 붙는다.** 학습 점수와 구조 관련
> 관찰은 영향이 적고, 검증/최종 분할 대비 일반화 주장은 약해진다.
>
> 우회 경로는 `report/table_facts.py` 로 막았다 — 이후 실행은 학습
> 분할에서만 계산한다.


> ⚠️ **이 실행은 관문 판정에 쓸 수 없다.** 최종 규칙이 가중치 8개로
> 항 19개를 만들어 리터럴 예산을 우회했다 (§15.1b). 정적 검사를
> 보강한 뒤 재실행해야 한다.

> 개발용 표(CUDA 12.4 / 호스트)다. 성능 수치를 인용하지 마라.

```
{
 "llm": {
  "model": "gpt-5.4-mini-2026-03-17",
  "temperature": 0.7,
  "seed": 20260821,
  "max_retries": 2,
  "concurrency": 6,
  "arch_prompt": "hw/sm_86.md"
 },
 "loop": {
  "run_id": "real-gpt-5.4-mini-2026-03-17",
  "n_rules_per_round": 12,
  "max_rounds": 20,
  "max_evals": 200,
  "patience": 10,
  "seed": 7,
  "sandbox_first_seen": true,
  "out_dir": "runs"
 },
 "split": "nk11008",
 "seed_rule": "handwritten",
 "agents": [
  "diagnose",
  "optimize"
 ],
 "data_source": "real",
 "bundle": "rtx-a6000-sm_86-c63710df"
}
```

## 라운드

```
r0   제안 12 채점  9 채택  5 | train 1.1219 val 1.2733 (+0.151) | 셀  2
r1   제안 12 채점  4 채택  3 | train 1.1083 val 1.1635 (+0.055) | 셀  3
r2   제안 12 채점  3 채택  2 | train 1.0896 val 1.2820 (+0.192) | 셀  5
r3   제안 12 채점  5 채택  2 | train 1.0709 val 1.0927 (+0.022) | 셀  6
r4   제안 12 채점  3 채택  1 | train 1.0709 val 1.0927 (+0.022) | 셀  6
r5   제안 12 채점  3 채택  1 | train 1.0700 val 1.0783 (+0.008) | 셀  6
r6   제안 12 채점  2 채택  1 | train 1.0690 val 1.0960 (+0.027) | 셀  6
r7   제안 12 채점  6 채택  1 | train 1.0690 val 1.0960 (+0.027) | 셀  7
r8   제안 12 채점  4 채택  1 | train 1.0690 val 1.0960 (+0.027) | 셀  7
r9   제안 12 채점  4 채택  2 | train 1.0690 val 1.0960 (+0.027) | 셀  8
r10  제안 12 채점  4 채택  1 | train 1.0682 val 1.0960 (+0.028) | 셀  8
r11  제안 12 채점  3 채택  0 | train 1.0682 val 1.0960 (+0.028) | 셀  8
r12  제안 12 채점  5 채택  0 | train 1.0682 val 1.0960 (+0.028) | 셀  8
r13  제안 12 채점  4 채택  0 | train 1.0682 val 1.0960 (+0.028) | 셀  8
r14  제안 12 채점  3 채택  1 | train 1.0679 val 1.0848 (+0.017) | 셀  8
r15  제안 12 채점  4 채택  1 | train 1.0661 val 1.0849 (+0.019) | 셀  8
r16  제안 12 채점  4 채택  2 | train 1.0655 val 1.1335 (+0.068) | 셀  8
r17  제안 12 채점  3 채택  1 | train 1.0640 val 1.1335 (+0.070) | 셀  8
r18  제안 12 채점  4 채택  1 | train 1.0632 val 1.2035 (+0.140) | 셀  8
r19  제안 12 채점  0 채택  0 | train 1.0632 val 1.2035 (+0.140) | 셀  8
```

## 관문과 같은 기준(a888 61형상)에서 재채점

| | 전체 61 | 짧은 41 | 긴 20 | 홀드아웃 10 |
|---|---:|---:|---:|---:|
| **벤더 (nearest) ★관문** | **1.0797** | 1.0977 | 1.0439 | 1.0700 |
| 손규칙 (같은 분할 적합) | 1.1891 | 1.2180 | 1.1318 | 1.2121 |
| **LLM 최고** | **1.0850** | **1.0695** | 1.1174 | 1.2035 |

**관문 미달.** 그러나 짧은 형상(여지의 87%)에서는 벤더를 이겼다.
긴 형상에서 잃는다 — 학습 분할이 짧은 형상 69% 라 §10.1 이 예측한 그대로다.

## 최종 규칙 (★ 예산 우회. 현재 정적 검사는 이것을 거부한다)

```python
def score(f, p, hw, w):
    s = np.log2(f.traffic_amplification) * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + f.smem_pressure * w[2]
    s = s + f.has_spill * w[3]
    s = s + f.split_k_cost * w[4]
    s = s + f.pipeline_warmup_frac * w[5]
    s = s + f.reg_pressure * w[6]
    s = s + f.log_inst_total * w[7]
    s = s + f.log_workspace_bytes * w[0]
    s = s + f.log_dram_traffic * w[0]
    s = s + f.occupancy_deficit * w[2]
    s = s + f.log_dram_traffic * w[4]
    s = s + f.is_two_stage * w[0]
    s = s + f.tile_aspect_imbalance * w[1]
    if p.is_memory_bound:
        s = s + (f.split_k_cost + f.log_mainloop_iters + f.log_dram_traffic) * w[7]
        s = s + f.tile_aspect_imbalance * w[5]
    s = s + (f.split_k_cost + f.log_mainloop_iters + f.log_dram_traffic) * w[7]
    s = s + f.tail_waste * w[7]
    s = s + f.is_two_stage * f.pipeline_warmup_frac * w[1]
    return s
```

w = [26.09, 95.113, 17.774, 108.838, 1.565, 115.261, 122.965, -3.235]

train 1.0632 / val 1.2035 / short 1.0394 / long 1.1170

## 거부 사유

```
llm     132건  UnexpectedModelBehavior: Exceeded maximum output retries (2)
               (+ asyncio.Semaphore 이벤트 루프 버그로 인한 손실)
static   20건  리터럴+가중치 9~10 > 8 / AST 노드 413 > 400
```
