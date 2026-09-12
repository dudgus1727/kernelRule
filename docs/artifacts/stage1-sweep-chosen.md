# ★ 고른 라이브러리 — `s1sw-3`

> **재현** `python3 experiments/stage1_sweep.py` · LLM 0회
> **비교표** `stage1-sweep.md` · **원본** `stage1-sweep.json`

## 근거 — §3 의 순서대로

```
1순위  형상 수준이면서 상수가 아닌 축   ★ 열 개 전부 0 — 동률
2순위  물리 커버리지에서 덮인 항        s1sw-3 과 s1sw-8 이 2/6, 나머지 1/6
3순위  축끼리 최대 spearman             s1sw-3 0.9558 vs s1sw-8 1.0000  -> ★ s1sw-3
```

동률이 1순위에서 전부 났으므로 2순위로 넘어갔고, 거기서 둘이 남아 3순위가
갈랐다. ⛔ 기준의 순서는 결과를 보기 전에 박은 그대로다.

## ⛔ 그런데 1순위가 0/10 이다 — 이것이 이번 sweep 의 진짜 결과다

```
열 라이브러리 전부 ★ 형상 수준이면서 61형상에서 변하는 축이 0개다
각 라이브러리의 형상 수준 축은 1개씩 있고, 그 축은 표의 66형상에서는
변하지만 ★ 실험이 쓰는 61형상에서는 값이 하나다 (D-167 §O 의 그 축들)
  s1sw-0·1·4·5·6·7·8·9  alignment_deficit
  s1sw-2                 unaligned_access_count
  s1sw-3                 vector_alignment_deficit
c21-lib 도 같다 — unaligned_operand_count · alignment_traffic_deficit 둘 다
```

**즉 분기 재료 부족은 시드 운이 아니라 구조다.** stage 1 을 열 번 돌려도
안 고쳐진다. 21실행의 최고 규칙들이 `roofline_ratio` 로만 분기한 이유가
여기 있다 — 그 축은 생성된 것이 아니라 **known5(기초)** 의 축이다.

★ 그러므로 `s1sw-3` 은 "분기 문제를 푼 라이브러리" 가 아니라
**"주어진 기준에서 가장 나은 것"** 이다. 분기 재료를 늘리려면 stage 1 을
다시 돌리는 것이 아니라 FeatureWriter 가 형상 수준 축을 만들게 하는
조건 자체를 바꿔야 한다 — 이 문서의 범위 밖이다.

## 고른 라이브러리의 내용

```
채택            16/20
덮임            2/6  (monotone_only 1)
최대 spearman   0.9558  cta_dispatch_amortization ~ cta_global_memory_bytes
range 기본값    7개
rationale 빈    0개
recheck 걸림    1개
stage 1 시간    736초
```

축:

```
contiguous_access_shortfall
cta_dispatch_amortization
cta_global_memory_bytes
cta_memory_intensity_deficit
instruction_overhead_per_flop
k_tail_waste
l2_tile_working_set_ratio
local_spill_traffic_fraction
mainloop_barrier_overhead
parallel_split_reduction_traffic
pipeline_fill_drain_fraction
register_file_pressure
serial_split_reduction_work
shared_memory_pressure
tiled_memory_redundancy
vector_alignment_deficit
```

## 순위 전체

```
 1. s1sw-3   형상축 0 · 덮임 2 · sp 0.9558
 2. s1sw-8   형상축 0 · 덮임 2 · sp 1.0000
 3. s1sw-2   형상축 0 · 덮임 1 · sp 0.8641
 4. s1sw-5   형상축 0 · 덮임 1 · sp 0.8753
 5. s1sw-1   형상축 0 · 덮임 1 · sp 0.8769
 6. s1sw-6   형상축 0 · 덮임 1 · sp 0.8956
 7. s1sw-4   형상축 0 · 덮임 1 · sp 0.9096
 8. s1sw-7   형상축 0 · 덮임 1 · sp 0.9096
 9. s1sw-9   형상축 0 · 덮임 1 · sp 0.9096
10. s1sw-0   형상축 0 · 덮임 1 · sp 1.0000
```
