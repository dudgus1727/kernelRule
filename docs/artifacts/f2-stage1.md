# F1-K 1단계 결과 — 공개 지식 다섯으로 시작 (2026-08-27)

> **상태**: 1단계 완료. 2·3단계 지시 대기
> **재현**: `python3 experiments/f1_pipeline.py F1-K --stage 1 --categorize --per-category 3 --tag k1`
> **사전 등록**: `f1k-preregistration.md` (LLM 호출 0회 상태에서 작성)
> **모델**: `gpt-5.6-luna` / responses / medium
> **표**: `datasets/rtx-a6000-sm_86-c63710df` (dev, 수치 대외 보고 금지)

## 채택 12/21 (57%)

중단선(절반)은 넘었다. 그래도 사전 등록대로 **거부 사유 분포부터** 봤다.

```
§8.3 검증 실패 (상수 std=0)   7건   ★ 압도적
쓸 수 없는 필드                1건
금지된 참조 'tflops'           1건   ★ 검사기 결함이었다 (아래)
중복                          0건   ← 예상과 다르다
```

**중복이 0 이다.** 다섯을 줬으니 중복 거부가 늘 것으로 예상했는데
하나도 안 났다.

## ★ 원칙 8 순서로 갈랐다 — 셋이 다른 원인이었다

### (1) 검사기 결함 — `hw.peak_tflops_f16` 이 금지어에 걸렸다

```
#16 roofline_lower_bound_time   금지된 참조: 'tflops' (§3)
    참조: [..., peak_tflops_f16, bandwidth_gbps, ...]
```

`_BANNED` 의 `"tflops"` 가 **`RAW_FIELDS` 가 명시적으로 허용한 필드**에
부분 문자열로 걸렸다. **검사기가 자기가 허용한 것을 금지했다.**

roofline 하한 시간을 만들려던 제안이 그렇게 버려졌다 — D-37
(`inspect.getsource` 실패를 "hw 를 쓴다" 로 떨어뜨림)과 같은 부류다.
**허용 필드를 먼저 가린 뒤 금지어를 찾도록** 고쳤다 (D-73).

### (2) 표의 성질 — 정렬 축이 상수다

`정렬·경계` 영역이 3회 연속 거부로 건너뛰어졌다. **표를 봤다.**

```
align_a  고유값 1  [8]   ★ 상수
align_b  고유값 1  [8]   ★ 상수
align_c  고유값 1  [8]   ★ 상수
```

**이 표에서 정렬은 축이 아니다.** LLM 이 만든 `alignment_*` 셋은
논리적으로 옳지만 값이 전부 같게 나온다.

D-64(dtype 빈틈)와 겉모습이 같지만 **다르다** — 그때는 필드를 노출하면
풀렸고, 이번에는 **표에 그 축이 없다.** 고칠 수 없고 고치지 않는다.
`skipped_categories` 에 남는 것이 정확한 기록이다.

### (3) 진짜 상수 — 나머지 넷

`pipeline_overlap_deficit` / `compute_lane_underfill` /
`resident_cta_quantization_loss` / `pipeline_sync_exposure` 를 **전
형상 전 config 에서** 다시 계산했다 (검증은 6형상 표본을 쓴다).

```
일곱 건 전부 고유값 1, std 0.000e+00   ★ 표본 탓이 아니다
```

`pipeline_kind` 는 2값, `threads` 3값, `max_blocks_per_sm` 4값으로
상수가 아닌데도 그렇다 — **함수가 그 축들을 상수로 접었다.**
검사기가 옳게 잡았다.

## 영역별 산출

```
연산 처리량        2   instruction_overhead_fraction, compute_k_underfill_fraction
메모리 트래픽      3   tiled_global_traffic_ratio, l2_residency_pressure, spill_traffic_ratio
연산·트래픽 비율   1   roofline_memory_gap
자원 압력          2   sm_resource_pressure, resident_block_scarcity
파이프라인         1   pipeline_fill_drain_fraction
리덕션            3   reduction_work_fraction, reduction_tree_depth, reduction_merge_tasks_per_sm
정렬·경계          0   ★ 건너뜀 (표의 축이 상수)
```

**여섯 영역에 1~3개씩 고르게 퍼졌다.** 한 영역에 몰리지 않았다.

## ★ 새 축 9 / 유효 12

사람 24개와의 상관. 엄격 = 스피어만·피어슨 둘 다 > 0.95.

| F1-K 축 | 영역 | 최근접 사람 피처 | sp | pe | 판정 |
|---|---|---|---:|---:|---|
| `reduction_tree_depth` | 리덕션 | `split_k_cost` | **1.000** | **1.000** | 중복 |
| `spill_traffic_ratio` | 트래픽 | `spill_magnitude` | 0.959 | 0.777 | 단조중복 |
| `reduction_merge_tasks_per_sm` | 리덕션 | `log_grid_tiles` | 0.975 | 0.817 | 단조중복 |
| `sm_resource_pressure` | 자원 | `smem_pressure` | 0.916 | 0.930 | ★새 축 |
| `roofline_memory_gap` | 비율 | `traffic_amplification` | 0.948 | 0.873 | ★새 축 |
| `reduction_work_fraction` | 리덕션 | `log_mainloop_iters` | 0.908 | 0.824 | ★새 축 |
| `l2_residency_pressure` | 트래픽 | `split_k_cost` | 0.751 | 0.607 | ★새 축 |
| `tiled_global_traffic_ratio` | 트래픽 | `waves` | 0.726 | 0.657 | ★새 축 |
| `instruction_overhead_fraction` | 연산 | `log_inst_total` | 0.676 | 0.635 | ★새 축 |
| `resident_block_scarcity` | 자원 | `smem_pressure` | 0.579 | 0.760 | ★새 축 |
| `pipeline_fill_drain_fraction` | 파이프라인 | `log_mainloop_iters` | 0.487 | 0.528 | ★새 축 |
| `compute_k_underfill_fraction` | 연산 | `waves` | 0.176 | 0.021 | ★새 축 |

```
중복 1 / 단조중복 2 / ★ 새 축 9
```

**사전 등록의 예상("새 축이 F1 보다 많다")과 비교:**

```
F1     16 채택 -> 새 축 7   (엄격 재발견 6, 단조 2)
F1-K   12 채택 -> 새 축 9   (중복 1, 단조중복 2)
```

**개수로도 비율로도 더 많다** (7/16 = 44% vs 9/12 = 75%).
다만 F1-K 는 **다섯을 줬으니 재발견할 것이 줄어드는 게 당연**하고,
사전 등록에 "재발견 개수는 판정 기준이 아니다" 라고 적어 뒀다.

## ★ 트래픽 영역과 roofline 영역은 겹치지 않는다 (§30.18 확인)

영역을 고정할 때 LLM 의 세 판을 제안 일곱에 매핑하면서 **"메모리
트래픽" 을 roofline 영역에 접은 적이 있었다.** 그것이 정당했는지
확인했다.

```
tiled_global_traffic_ratio <-> roofline_memory_gap   sp 0.700
l2_residency_pressure      <-> roofline_memory_gap   sp 0.300
spill_traffic_ratio        <-> roofline_memory_gap   sp 0.110
```

**셋 다 0.95 미만이다. 다른 축이 맞다.** 매핑에서 접은 것은 부정확했고,
일곱으로 나눈 것이 옳았다.

## 관찰 — 출처는 잘 안 붙었다

```
rationale 에 출처 표기가 있는 것   2/12
```

예시 셋이 전부 `출처:` 를 달고 있는데도 **여섯 중 하나만 따라 했다.**
"출처를 적는 습관이 전달된다" 는 기대는 이번에 약하게 나타났다.
`role/feature.md` 의 출력 명세에 `rationale` 설명은 있지만 출처를
요구하지는 않는다 — **고치지 않는다.** 결과를 보고 프롬프트를 바꾸면
그 표에 맞추는 것이다 (§12.3d).

## `shape_level` — 0개

생성된 12개 중 형상 수준으로 판정된 것이 없다. 시작 라이브러리의
`roofline_ratio` 는 형상 수준이므로, 진화에서 `if p.roofline_ratio:`
분기는 여전히 가능하다.

## `physics_coverage` — 1/6

`physics_seeded` 의 여섯 항 중 하나만 덮는다 (F1 은 3/6). 다만 이
지표는 **사람 24개 중 여섯을 얼마나 재현하나**를 재는 것이라, 다섯을
이미 준 조건에서는 해석이 다르다 — 시작 라이브러리에 그 여섯 중
`has_spill` 하나만 들어 있다.

## 비용

```
호출 21   입력 103,168   출력 28,053   46분
```

---

# ★ 재검사 (2026-08-28) — 검사기 결함으로 버려진 것 하나를 되살렸다

> **재현**: `python3 experiments/revalidate.py runs/f1pipe-F1-K-k1/stage1-features`
> **LLM 0회.**

```
재제안 (LLM N회)   그 영역만 제안 횟수가 늘어 **조건이 달라진다**
★ 재검사 (LLM 0회)  같은 코드를 고친 검사기로. 조건이 안 바뀐다
```

거부 9건 전부를 고친 검사기로 다시 봤다.

```
★ 되살아남 1   roofline_lower_bound_time   (전: 금지된 참조 'tflops')
   여전히 거부 8
```

**`hw.peak_tflops_f16` 이 금지어에 걸린 그 하나만 되살아났다** — 진단이
정확했다. 나머지 8은 검사기 결함이 아니라 **표의 성질**(정렬 축 상수)과
**함수 자체가 상수**인 경우다.

`revalidated.jsonl` 로 **따로** 남긴다 — `proposals.jsonl` 을 덮어쓰면
"그때 무엇이 거부됐는지" 가 사라진다 (문서 규칙 2).

```
라이브러리   시작 5 + 생성 13 = 18개   (형상 수준 1, config 수준 17)
```

⚠️ **F1 과 크기가 다르다.**

```
F1     0 시작 + 21 생성 = 21
F1-K   5 시작 + 13 생성 = 18
```

진화 결과를 비교할 때 이 사실을 함께 적는다 — **"라이브러리가 작아서
진 것인가" 라는 반론이 가능하다.**

## ★ 영역 고정이 표의 성질을 드러냈다

`정렬·경계` 가 통째로 건너뛰어진 것은 **F1 에서도 마찬가지였을 것**이다.
그때는 영역이 없어서 LLM 이 그냥 다른 축을 만들고 넘어갔고, **그래서
안 드러났다.**

```
영역을 고정하면 "이 영역에서 만들 수 없다" 가 기록으로 남는다.
자유 생성은 그 사실을 조용히 지나간다.
```

D-64(dtype 빈틈)를 찾아낸 것도 같은 장치였다.

---

# 2단계 — Architect (2026-08-28)

> **재현**: `python3 experiments/f1_pipeline.py F1-K --stage 2 --tag k1 --n-architect 10`

## 10/10 성공, 거부 0

| 시도 | 학습 | 항 | `p.*` 분기 | 새 축 | known5 |
|---:|---:|---:|---|---:|---:|
| 1 | **1.1794** | 7 | `where` 1 | 4 | 4 |
| 9 | 1.1806 | 7 | `if` 1 | 3 | 5 |
| 6 | 1.2093 | 8 | `where` 1 | 5 | 4 |
| 2 | 1.2174 | 7 | `where` 1 | 5 | 5 |
| 3 | 1.2325 | 7 | `where` 1 | 5 | 4 |
| 5 | 1.2353 | 7 | `where` 1 | 4 | 4 |
| 4 | 1.3134 | 6 | `if` 1 | 3 | 3 |
| 0 | 1.3171 | 7 | `if` 1 | 3 | 4 |
| 8 | 1.3288 | 7 | `where` 1 | 5 | 3 |
| 7 | 1.3801 | 7 | `if` 1 | 4 | 3 |

```
학습 중앙 1.2339  최소 1.1794  최대 1.3801
```

### ★ 10/10 이 `p.*` 분기를 쓴다

시작 라이브러리의 `roofline_ratio` 가 형상 수준이라 가능했다.
**F1 은 2단계에서 형상 수준 피처가 0개였던 적이 있고** (D-67), 고친
뒤에도 8/10 이었다. **여기는 10/10 이다.**

### ★ 예산이 안 찼다 — 7항이 여덟 개

```
F1        8항 x9, 7항 x1
사람 24개  8항 x9, 7항 x1
F1-K      ★ 7항 x8, 8항 x1, 6항 x1
```

**두 팔은 전부 예산(8)을 채웠는데 F1-K 는 안 채웠다.** 진화가 교체만
가능한 상태로 시작하지 않는다 (D-35 계열).

### 새 축은 실제로 쓰인다 — 9개 중 7개

```
10회 시도에서 한 번이라도 쓰인 새 축   7/9
안 쓰인 것   l2_residency_pressure, resident_block_scarcity
```

**"만들었지만 쓸모없다" 가 아니다.** 씨앗도 새 축 넷을 쓴다.

## 씨앗 — `architect-try01`, 학습 1.1794, **7항**

```python
def score(f, p, hw, w):
    s = f.edge_waste * w[0]
    s = s + f.tiled_global_traffic_ratio * w[1]
    s = s + f.spill_traffic_ratio * w[2]
    s = s + f.occupancy_deficit * w[3]
    s = s + f.tail_waste * w[4]
    s = s + f.reduction_work_fraction * w[5]
    s = s + np.where(p.roofline_ratio < 1,
                     f.roofline_memory_gap,
                     f.instruction_overhead_fraction) * w[6]
    return s
```

```
known5    edge_waste / occupancy_deficit / tail_waste / roofline_ratio (분기)
새 축      tiled_global_traffic_ratio / reduction_work_fraction /
          roofline_memory_gap / instruction_overhead_fraction
```

**주어진 다섯 중 넷과 새로 만든 넷을 섞는다.** 그리고 `roofline_ratio`
를 **경계값 1 로 읽어** 메모리 바운드면 트래픽 축을, 아니면 명령
오버헤드 축을 본다 — 시작 라이브러리에 문턱형을 넣은 의도대로다.

### ★ `physics_seeded` 에 없던 형태다 — 재가중이 아니라 **선택**

```python
# physics_seeded — 재가중
if p.is_memory_bound:
    s = s + f.traffic_amplification * w[6]

# F1-K 씨앗 — 선택
s = s + np.where(p.roofline_ratio < 1,
                 f.roofline_memory_gap,           # 메모리 바운드면 이것
                 f.instruction_overhead_fraction  # 아니면 저것
                 ) * w[6]
```

**재가중은 "같은 물리를 더/덜 본다" 이고, 선택은 "체제에 따라 다른
물리를 본다" 이다.** 표현력이 한 단계 높다.

사람 팔(`architect-try05`)도 `np.where(p.is_memory_bound, A, B)` 를
세 번 쓴다 — 그러니 이 형태 자체가 F1-K 만의 것은 아니다. 다만
**F1-K 는 시작 라이브러리에 문턱형(`roofline_ratio`)이 있어서 경계를
연속값으로 읽는다** (`< 1` 이 roofline 의 무릎이다). 이진 피처
(`is_memory_bound`)로 나누는 것과 달리 경계를 옮길 수 있다.

`p.*` 분기가 **10/10** 인 것이 그 효과로 보인다.

```
F1 (D-67 수정 전)   0/10   형상 수준 피처가 0개였다
F1 (수정 후)        8/10
사람 24개           6/10
★ F1-K              10/10
```

## 비용

```
2단계 호출 10   289초
```

---

# 3단계 진화 — 결과 (2026-08-28)

> **재현**: `python3 experiments/f1_pipeline.py F1-K --stage 3 --tag k1 --n-seeds 6 --rounds 12`
> 관찰: `python3 experiments/f1k_observe.py`

## 구조 홀드아웃 (정준 절차: 실행마다 geomean -> 6실행 중앙)

```
s0 1.1426   s1 1.1482   s2 1.1255   s3 1.1321   s4 1.1101   s5 1.1217

중앙 1.1288   사분위 [1.1227, 1.1400]
★ 시드 폭 σ 0.0140   범위 0.0381 (1.1101~1.1482)
```

## ★ 세 조건 비교

| | 시작 | 생성 | **라이브러리** | 새 축 | **씨앗 항** | 홀드아웃 중앙 | 시드 폭 σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **F1** | 0 | 21 | **21** | 7 | 8 | 1.1195 | 0.0177 |
| **F1-K** | 5 | 13 | **18** | 9 | **7** | 1.1288 | 0.0140 |
| **사람 24개** | 24 | 0 | **24** | — | 8 | **1.0762** | 0.0124 |

```
F1-K vs F1       중앙차 +0.0093   p = 0.589   ★ 구분 불가
F1-K vs 사람24    중앙차 +0.0526   p = 0.002
F1   vs 사람24    중앙차 +0.0433   p = 0.002
```

**F1-K 와 F1 은 구분 불가다.** 차이 0.0093 은 시드 폭(σ 0.0274, D-53)
안이고 p=0.589 다. 사전 등록에 적은 그대로다.

> ★ 진화 성능이 F3 를 따라잡을지는 모른다. 못 따라잡아도 실패가 아니다.
> F1 보다 나을지도 열린 질문이다.

**답: F3(사람 24개)는 못 따라잡았고 F1 과는 구분되지 않는다.**

⚠️ **라이브러리 크기가 다르다** (18 vs 21 vs 24). "F1-K 가 작아서
진 것인가" 라는 반론이 가능하다 — 이 자료로는 못 가른다.

## 관찰 1 — ★ 7항 씨앗이 즉시 8항이 된다

```
8항이 나온 실행   6/6
첫 등장 라운드     중앙 r1 (r0 셋, r2 하나, r5 하나, r6 하나)
6실행 전부 최종 최고 규칙이 8항
```

**D-54 조사에서 본 것이 반복됐다** — F1 의 Architect #9 가 7항으로
시작해 r1 에 스스로 8항을 채웠다. **여유가 있어도 채운다.**

```
"7항 시작이 이점이다" 는 성립하지 않는다.
예산 8은 진화가 즉시 채우는 값이고, 씨앗의 항 수는 최종 결과를 정하지 않는다.
```

D-54 때 "7항 대안을 쓰지 마라" 던 판단이 이 자료로 다시 확인된다.

## 관찰 2 — `np.where` 선택 구조는 4/6 에서 살아남는다

```
s0  where-선택 0   "occupancy_deficit 항을 제거하고, sm_resource_press..."
s1  where-선택 1
s2  where-선택 2   ★ 늘었다
s3  where-선택 0   "H25를 반영하기 위해 sm_resource_pressure와 중복되고..."
s4  where-선택 1
s5  where-선택 1
```

**s2 는 오히려 하나를 더 만들었다.** 사라진 둘(s0, s3)은 `changes` 를
보면 **그 항을 지운 것이 아니라 중복·약한 항을 정리하다 함께 없어진**
것으로 읽힌다.

`if` 재가중은 6실행 어디에도 없다 — **전부 `np.where` 선택이다.**

## 관찰 3 — ★ 새 축 8/9 가 진화 후에도 쓰인다

```
실행당 새 축   5~7개
6실행 최고 규칙 전체에서 쓰인 새 축   8/9
안 쓰인 것     reduction_work_fraction  (씨앗에는 있었다)
```

**전 실행이 `instruction_overhead_fraction` / `roofline_memory_gap` /
`resident_block_scarcity` / `tiled_global_traffic_ratio` 를 쓴다.**

2단계에서 안 쓰이던 `l2_residency_pressure` 와
`resident_block_scarcity` 가 **진화 중에 들어왔다** — Architect 가 안
골랐지만 Optimizer 가 찾았다.

```
"만들었지만 쓸모없다" 는 아니다.
다만 "쓰인다" 와 "성능을 올린다" 는 다른 말이다 — 홀드아웃은 F1 과 같다.
```

## 비용

```
1단계   호출  21   입력   103,168   출력    28,053   46분
재검사   LLM 0회
2단계   호출  10   입력    90,533   출력    20,411    5분
3단계   호출 936   입력 8,110,673   출력 1,267,434   약 3시간
합계    호출 967   (사전 등록 상한 990 안)
```

---

# ★ 경계값은 안 옮겨졌다 — 리터럴을 우회했다 (2026-08-28)

`np.where(p.roofline_ratio < 1, ...)` 의 **경계 1 이 진화 중에 옮겨지나**
를 봤다. 안 옮겨졌다. **대신 `1` 을 안 쓰는 표현으로 바뀌었다.**

```
씨앗      p.roofline_ratio < 1

s1       np.square(p.roofline_ratio) < p.roofline_ratio     x² < x  ≡ x < 1
s2       p.roofline_ratio < np.sqrt(p.roofline_ratio)       x < √x  ≡ x < 1
s5       p.roofline_ratio <= np.sqrt(p.roofline_ratio)      같다
s4       p.roofline_ratio < np.sign(p.roofline_ratio)       x < 1   (x>0 이므로)
s0, s3   분기 자체가 사라짐
```

**넷 다 수학적으로 `x < 1` 과 같다.** 그리고 전 아카이브에서 리터럴
`1` 을 쓴 비교는 **5회뿐**이다.

## ★ 다섯째 형태 — `np.isfinite(x)` (2026-08-28 추가)

사람 24개 팔(`f1pipe-F3-arch24-s0`)의 최종 규칙에서 나왔다. 이것이
가장 노골적이다.

```python
np.nan_to_num(np.isfinite(f.tail_waste) / (np.isfinite(f.tail_waste) - f.tail_waste))
#              ~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~
#              = 1 (유한하므로)              = 1
# 즉  1 / (1 - f.tail_waste)
np.log2(np.maximum(f.traffic_amplification, np.isfinite(f.traffic_amplification)))
# 즉  log2(max(x, 1))
np.minimum(f.waves, np.isfinite(f.waves))     # 즉 min(x, 1)
```

**한 규칙 안에서 아홉 번 쓴다.** `np.sqrt`/`np.square`/`np.sign` 은
적어도 `x<1` 이라는 관계를 표현하지만, `np.isfinite(x)` 는 **오로지
상수 1 을 쓰기 위한 것**이다 — 인자가 무엇이든 값이 1 이다.

전 아카이브 330규칙에서:

```
np.sqrt(      61 규칙
np.square(    46
np.sign(      19
np.isfinite(   4      <- 드물지만 한 규칙 안에서 여러 번
```

앞의 셋은 **정당한 용도가 섞여 있어** 세는 것만으로는 우회인지 못
가린다. `np.isfinite` 는 다르다 — 이 표에서 **f 피처 19개 전부가 모든
(형상, config) 에서 유한**하다(확인: `np.isfinite(m.column(n))` 이
19/19 전부 참). 그러므로 `np.isfinite(f.X)` 는 언제나 1 이고,
**정당한 용도가 없다.**

⚠️ "이 표에서 유한하다" 는 **표 의존 사실**이다. 다른 표에서 비유한
값이 나오면 `np.isfinite` 는 실제 판별 기능을 갖는다. 검사기로 막을
때는 "언제나 1" 이 아니라 "상수 취급 위험" 으로 적어야 한다.

★ 이것은 (나) "항등 변환을 정적으로 잡는다" 가 일반적으로는 불가능해도
**이 한 형태는 잡을 수 있다**는 뜻이기도 하다. 다만 지금은 안 고친다 —
예산 실험 전에 표현을 바꾸면 전후 비교가 깨진다.

## 왜 그러나 — 예산 규칙이다

```
숫자 리터럴 + 가중치 개수 <= 8
```

항 8개를 쓰려면 **리터럴이 0개**여야 한다. 그래서 `1` 을 `np.sign(x)`
로 쓴다 — 함수 호출은 리터럴이 아니다.

## ★ 두 가지가 걸린다

```
1  예산 우회다
   D-35 계열의 "가중치 재사용으로 항 늘리기" 와 같은 종류다.
   그때는 막았고 이번에는 안 막혀 있다

2  ★ 해석 가능성을 해친다
   "해석 가능한 규칙" 이 이 프로젝트의 주장인데
   `p.roofline_ratio < np.sign(p.roofline_ratio)` 는 해석이 어렵다
   GBDT 와의 차이가 여기서 줄어든다
```

**아직 안 고쳤다.** 고치는 방법이 둘인데 둘 다 부작용이 있다.

```
(가) 리터럴 예산을 늘린다        구조 비교가 흐려진다 (§29.4 의 목적)
(나) 항등 변환을 정적으로 잡는다   일반적으로 불가능하다 (임의 표현식)
```

**경계가 안 옮겨진 것 자체는 두 해석이 가능하다** — `1` 이 물리적으로
맞는 자리(roofline 의 무릎)라서일 수도 있고, **옮길 수단이 예산에
막혀서**일 수도 있다. 이 자료로는 못 가른다.
