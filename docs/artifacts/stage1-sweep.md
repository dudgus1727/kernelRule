# ★ stage 1 열 번 — 그리고 고른 라이브러리

> **재현** `python3 experiments/stage1_sweep.py` · LLM 0회
> **원본** `stage1-sweep.json` · **조건** F2 · 표 `datasets/rtx-a6000-sm_86-c63710df` · 분할 21실행과 같음
> ★ **기준은 결과가 나오기 전에 박았다** (D-169 §3) — 결과를 보고 기준을 정하면 사후 선택이다

## 기준 (순서대로)

```
1순위  형상 수준이면서 ★ 상수가 아닌 축의 수      (많을수록)
2순위  물리 커버리지에서 덮인 항의 수              (많을수록)
3순위  축끼리 최대 spearman                        (낮을수록)
동률   -> 다음 순위 -> 그래도 동률이면 시드가 작은 것

⛔ 안 쓴 것: 홀드아웃 성능 · 루프 결과 · 죽은 항 비율 · 축의 개수
★ 셋 다 stage 1 산출물만으로 계산되고 답을 안 읽는다
```

## 비교표

| 라이브러리 | ★1 형상축(비상수) | ★2 덮임 | ★3 최대 sp | 채택/제안 | range 기본값 | 범위비 중앙 | 범위비 최대 | rationale 빈 | recheck | 초 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `s1sw-3` ★ | 0 | 2/6 | 0.9558 | 16/20 | 7 | 68.3 | 4.27e+13 | 0 | 1 | 736 |
| `s1sw-8` | 0 | 2/6 | 1.0000 | 17/20 | 8 | 126 | 5.12e+11 | 0 | 1 | 739 |
| `s1sw-2` | 0 | 1/6 | 0.8641 | 20/20 | 14 | 1.51 | 1.64e+04 | 0 | 1 | 783 |
| `s1sw-5` | 0 | 1/6 | 0.8753 | 15/20 | 11 | 4 | 8.66e+11 | 0 | 1 | 747 |
| `s1sw-1` | 0 | 1/6 | 0.8769 | 17/20 | 16 | 1.24 | 5.24e+05 | 0 | 1 | 765 |
| `s1sw-6` | 0 | 1/6 | 0.8956 | 17/20 | 15 | 4.2 | 6.67e+07 | 0 | 1 | 741 |
| `s1sw-4` | 0 | 1/6 | 0.9096 | 16/20 | 13 | 1.25 | 1.24e+11 | 0 | 1 | 808 |
| `s1sw-7` | 0 | 1/6 | 0.9096 | 13/20 | 8 | 1.48 | 2.49e+08 | 0 | 1 | 631 |
| `s1sw-9` | 0 | 1/6 | 0.9096 | 14/20 | 13 | 1.25 | 583 | 0 | 1 | 734 |
| `s1sw-0` | 0 | 1/6 | 1.0000 | 15/20 | 12 | 3.07 | 1.23e+11 | 0 | 1 | 719 |

★ 고른 것은 **`s1sw-3`** — 형상 수준이면서 상수가 아닌 축이 0개로 가장 많다.

그 축들:

```
```

## 10개 사이에 반복된 축 이름

★ 21실행에서 독립 실행들이 같은 축을 만든 것과 같은 관찰이다.

| 축 | 몇 개 라이브러리에 |
|---|--:|
| `alignment_deficit` | 8 |
| `grid_underfill_deficit` | 2 |
| `parallel_reduction_traffic_fraction` | 6 |
| `pipeline_fill_drain_fraction` | 6 |
| `register_file_pressure` | 9 |
| `shared_memory_pressure` | 10 |
| `spill_traffic_fraction` | 4 |
| `tile_roofline_deficit` | 4 |
| `instruction_overhead_fraction` | 2 |
| `k_tail_waste` | 2 |
| `pipeline_depth_deficit` | 2 |
| `l2_working_set_pressure` | 3 |
| `parallel_reduction_traffic` | 2 |
| `instruction_overhead_per_flop` | 2 |
| `spill_traffic_ratio` | 3 |
| `instruction_density` | 3 |
| `mainloop_iteration_count` | 2 |
| `execution_wave_count` | 2 |

반복된 이름 18개 / 전체 서로 다른 이름 106개

