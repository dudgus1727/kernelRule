# F1 — LLM 이 물리량을 만들 수 있는가

**측정일** 2026-08-22 · `gpt-5.4` · 제안 20회 · **재현**
`python3 experiments/feature_writer.py 20 F1`

**조건 F1** 하드웨어·형상·config 의 **원시 값만** 준다. 기존 피처 24개는
이름조차 프롬프트에 없다 (테스트가 고정). 파생 물리량을 스스로 만들어야 한다.

---

## 결과

```
제안 20   채택 16   호출 20   입력 91,400 / 출력 6,443 토큰   52분
```

거부 4건은 전부 §8.3 자동 검증에서 걸렸다 (상수 / 스케일 불변성).
AST 검사(하드코딩 상수·금지 필드·정답 참조)에 걸린 것은 **0건**이다.

### 재발견 — 사람이 쓴 24개 중

**엄격** (스피어만 **과** 피어슨 둘 다 > 0.95):

| 사람이 쓴 것 | 생성된 것 | sp | pe |
|---|---|---:|---:|
| `occupancy_deficit` | `occupancy_headroom_loss` | 1.000 | 1.000 |
| `smem_pressure` | `cta_smem_capacity_pressure` | 1.000 | 1.000 |
| `waves` | `cta_launch_rounds` | 1.000 | 1.000 |
| `is_memory_bound` | `predicated_tile_waste` | 0.964 | 1.000 |

**단조 동등** (스피어만만 > 0.95 — 같은 물리, 다른 함수형):

| 사람이 쓴 것 | 생성된 것 | sp | pe |
|---|---|---:|---:|
| `edge_waste` | `predicated_tile_waste` | 1.000 | 0.817 |
| `log_grid_tiles` | `cta_tile_count_per_sm` | 1.000 | 0.672 |
| `tail_waste` | `cta_launch_rounds` | 0.980 | 0.308 |
| `sm_idle_cost` | `cta_launch_rounds` | 0.980 | 0.179 |
| `can_up_cp_async` | `k_tail_iteration_ratio` | 1.000 | 0.000 |

```
★ 4/24 엄격 + 5/24 단조 = 9/24 (38%)
  어느 것과도 안 맞는 새 축 10개
```

> ⚠️ **두 건은 우연으로 보인다.** `can_use_cp_async ← k_tail_iteration_ratio`
> 는 피어슨 0.000 이고, `is_memory_bound ← predicated_tile_waste` 는 이진
> 피처에 피어슨 1.000 이 나온 것이 부자연스럽다. 12형상 표본에서 이진
> 피처의 상관은 신뢰하기 어렵다. **보수적으로 읽으면 7/24 (29%) 다.**

### 정확히 맞은 셋이 의미하는 것

`occupancy_headroom_loss` 는 **스레드·레지스터·smem 한계의 최소**로
활성 CTA 수를 구한다 — 표준 CUDA occupancy 계산이다. 원시 값
(`cfg.threads`, `cfg.regs_per_thread`, `cfg.smem_bytes`, `hw.*`)만 보고
유도했고, 사람이 쓴 `occupancy_deficit` 와 **완전히 일치**한다.

`cta_launch_rounds` 는 `waves` 를 그대로 재현했고, 그 단조 변환인
`tail_waste` · `sm_idle_cost` 와도 스피어만 0.980 이다 — **wave
quantization 을 이해했다는 증거**다.

### 새 축 10개

```
cta_k_passes_per_output   cta_l2_residency_pressure   cta_lane_waste_ratio
cta_payload_per_thread    k_tail_iteration_ratio      roofline_excess_ratio
split_k_granularity_loss  split_k_io_amplification    split_k_tail_fraction
wave_tile_imbalance
```

`cta_l2_residency_pressure` 는 이전에 "빠진 피처 후보" 로 적어 둔 **L2
재사용** 축이다. `split_k_*` 셋은 split-K 를 세 방향으로 나눠 본다.

**단, 이 10개가 유용한지는 아직 모른다.** 규칙에 넣어 regret 이 개선되는지는
별도 실험이다 (§12.3d 대로 구조 홀드아웃은 그때 한 번만 본다).

---

## ★ 검증기의 결함이 연달아 둘 드러났다

**F1 은 검증기가 LLM 생성물을 처음 마주하는 자리다.** 사람이 쓴 24개는
검증기를 통과하도록 만들어졌으므로 이 결함들이 안 보였다.

### D-37 — `inspect.getsource` 실패를 "hw 를 쓴다" 로 떨어뜨렸다

`exec` 로 만든 함수는 항상 `OSError` 다. 그래서 생성 피처가 전부
`uses_hw=True` 로 판정됐고 **하드웨어를 안 쓰는 정상 피처가 기각**됐다.
첫 스모크에서 `tile_compute_waste`(= `edge_waste` 와 sp/pe 1.000)가 그렇게
버려졌다.

### D-38 — 검사용 가짜 하드웨어가 필드 하나를 안 바꿨다

`_alt_hw` 가 `max_threads_per_sm` 을 그대로 뒀다. occupancy 피처는
`min(by_threads, by_regs, by_smem)` 이라 그 항에 물리면 값이 안 변한다.

```
수정 전   채택 3/20   ← 같은 제안이 16회 반복됐다
수정 후   채택 16/20
```

**16회 반복은 되먹임이 없어서다.** 거부 사유를 안 주니 모델이 같은 것을
계속 냈다 — 132/240 재시도 소진과 같은 병이다. 최근 5건의 거부 사유를
프롬프트에 넣자 사라졌다.

**둘 다 "LLM 이 피처를 못 만든다" 로 읽힐 뻔했다.** 도구의 결함을
피험자의 실패로 오독하는 것이 이 실험의 가장 큰 위험이다.

---

## 판정

**"피처 정의는 사람이, 조합은 LLM 이" 는 성립하지 않는다.**

원시 값만으로 24개 중 최소 7개(보수적)~9개를 재발견했고, 그중 셋은
**수치까지 정확히 일치**한다. 그리고 기존에 없던 축 10개를 냈다.

**부분 성공이다.** 절반을 못 넘었으므로 "전부 자동화된다" 는 아니다.
남는 질문 둘:

```
1. 못 만든 15개는 왜인가 — 20회로는 부족한가, 유도가 어려운가
2. 새 축 10개가 regret 을 개선하는가 — 만드는 것과 쓸모는 다른 문제다
```

**전이 주장에 층이 하나 추가된다.** 피처가 물리 계산이면 아키텍처
무관하지만, 그 피처를 사람이 만들어야 하면 "새 GPU 마다 사람이 붙어야
한다" 가 된다. F1 은 **그 단계도 일부 자동화된다**는 첫 증거다.

---

## 알려진 비용 문제

제안당 시간이 뒤로 갈수록 늘어난다 (초반 40초 → 후반 4분). 중복 판정용
기준 열을 **생성 레지스트리가 커질 때마다 다시 계산**하기 때문이다
(O(n²)). 20회에 52분이 걸렸다. 누적 열을 캐시하면 된다. **미수정.**
