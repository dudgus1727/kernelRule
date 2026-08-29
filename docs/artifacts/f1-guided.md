# F1 영역 기반 생성 — ★ 판정 기준 (실험 **전**에 박는다)

> **상태**: 완료 (2026-08-26) — 결론 D-63
> **재현**: `python3 experiments/f1_pipeline.py F1 --stage 1` (guided)
>          `python3 experiments/f1_pipeline.py F1 --stage 1 --no-categorize` (free)
> **모델**: `gpt-5.6-luna` / responses / medium
> **표**: `datasets/rtx-a6000-sm_86-c63710df` (dev, 수치 대외 보고 금지)


> ⚠️ **조건 기록 정정 (2026-08-27).** 이 실행의 프롬프트에는 **내부 메모가
> 포함돼 있었다.** `role/_rules_common.md` / `_rules_edit.md` / `areas.md`
> 의 `<!-- ... -->` 주석이 걷히지 않고 그대로 렌더링됐고, 거기에는
> `§30.10`, `D-45`, `D-47`, `원칙 2` 같은 **내부 결정 번호**가 있었다.
>
> 즉 "하드웨어 사실 + 실행 모델 + 피처 목록만 준다" 는 조건 서술이
> 부정확했다 — LLM 이 우리 내부 규율 문서의 일부를 함께 읽었다.
> **성능에 어떤 영향이 있었는지는 알 수 없다.** 재실행하지 않고
> 조건 기록만 정정한다.
>
> `load_prompt` 가 이제 HTML 주석을 걷어내고, 렌더링 전문에 `<!--` 나
> `§30.` / `D-NN` 이 남으면 실패하는 테스트가 있다 (§30.19).

**결과를 보고 기준을 정하면 오염이다** (D-50). 여기 먼저 적는다.

---

## ★ 교락을 먼저 없앤다

```
F1-free    luna 17개    옛 프롬프트 (10,255자, 예시 하나 — 나쁜 예시)
F1-guided  이제 돌릴 것   새 프롬프트 (3,280자, 예시 셋 — 다른 도메인)
```

**두 변수가 동시에 다르다** — 카테고리화와 프롬프트 개편 (D-31).
그래서 **F1-free 를 새 프롬프트로 다시 돌린다.**

```
(가) luna 17개    옛 프롬프트 + 자유       이미 있다
(나) F1-free      새 프롬프트 + 자유       20호출
(다) F1-guided    새 프롬프트 + 영역별      ~19호출

(나) vs (다)   ★ 카테고리화의 순수 효과
(가) vs (나)   프롬프트 개편의 효과
```

합계 39호출. 싸다.

## 주 관찰 — ★ 성능이 아니다

시드 폭과 무관한 것들만 본다.

```
★ categories.json      LLM 이 나눈 영역과 설명. **이 실험의 핵심 관찰**
카테고리 분포          (가)에서 split_k_* 3개, cta_* 5개로 편중됐다
재발견 수와 대상        엄격(sp·pe 둘 다 >0.95) / 단조(sp만)
새 축 수
physics_coverage       `physics_seeded` 여섯 항을 얼마나 덮나
거부율과 사유
skipped_categories     3회 연속 거부로 건너뛴 영역
```

## ★ 예상 결과

```
카테고리화가 편중을 줄인다 — 그것이 목적이므로

★ 재발견 수가 늘어날지는 모른다. 줄어도 실패가 아니다.
  영역을 고르게 도느라 쉬운 축(edge_waste 류)을 덜 팔 수 있다.
```

**"재발견이 줄면 실패" 로 읽지 않는다.** 목적은 편중 완화이지 재발견
극대화가 아니다. 예상을 미리 적는 것이 사후 합리화를 막는다.

## 판정

```
편중        카테고리별 산출 개수의 최대/최소 비. (가)는 5:0 이었다
            (cta_* 5개 vs 만들지 않은 영역)
빈틈        skipped_categories 가 있으면 "원시 값으로 표현하기 어려운
            물리" 의 목록이다 — 그것도 결과다
중복        영역이 겹치게 나뉘었으면 그것도 결과다. 고치지 않는다
```

**LLM 이 나눈 영역이 부실해도 고치지 않는다.** 사람이 카테고리를 주면
사전 지식을 건네는 것이고, 부실함 자체가 관찰이다.

## 안 하는 것

```
구조 홀드아웃 채점    §12.3d — F1 이 끝난 뒤 한 번만 본다
프롬프트 재튜닝       결과를 보고 프롬프트를 고치면 그 표에 맞추는 것이다
카테고리 수 조정      5~8 은 실행 전에 정했다
```

---

## 결과 (2026-08-26)

세 조건. **(나) 를 새로 돌려 교락을 없앴다** — (가)와 (다)는 프롬프트와
생성 전략이 동시에 달랐다 (D-31).

```
(가) luna 17개    옛 프롬프트(10,255자, 나쁜 예시 하나) + 자유
(나) F1-free      새 프롬프트(3,280자, 다른 도메인 예시 셋) + 자유
(다) F1-guided    새 프롬프트 + 영역별
```

### ★ LLM 이 나눈 영역 (`categories.json` 전문)

```
산술_대역폭_압력       형상의 연산량과 전역 메모리 이동량이 하드웨어의
                    ridge point 에 비해 어느 쪽 병목을 강제하는가
타일_경계_낭비        M·N·K 가 타일 크기로 나누어지지 않아 가장자리에서
                    계산·적재된 작업 중 얼마나 유효하지 않게 되는가
메모리_접근_정렬       행렬 레이아웃과 각 입력·출력의 정렬 조건이 연속적이고
                    병합된 메모리 접근을 얼마나 제한하는가
자원_점유율          스레드 수, 레지스터, shared memory, spill 이 SM 당
                    동시 블록 수와 잠재 병렬성을 얼마나 제약하는가
작업량_파동_충전       타일 수와 split-K 로 만들어진 작업량이 SM 수의 파동
                    단위에 맞지 않아 유휴 실행 슬롯을 얼마나 남기는가
분할_k_결합          K 를 분할할 때 중간 결과 저장과 부분합 결합에 필요한
                    추가 작업 및 동기화가 얼마나 발생하는가
파이프라인_명령_효율     타일 K 단계, 파이프라인 종류, 명령 총량이 데이터
                    이동과 연산의 겹침 및 발행 효율을 얼마나 떨어뜨리는가
```

**뺀 것 (LLM 이 스스로 적었다):**

> 정확한 캐시 적중률, 실제 메모리 트랜잭션 수, 동기화 지연, 레이턴시와
> dtype 별 실제 처리율은 주어진 원시 값만으로 직접 측정할 수 없어 별도
> 영역으로 만들지 않았습니다.

**이것이 이 실험의 핵심 관찰이다.** 사람이 `§30.9` 의 Architect 프롬프트에
적어 둔 "느려지는 경로들"(일을 얼마나 하는가 / 기계를 얼마나 채우는가 /
메모리를 얼마나 움직이는가 / 자원이 모자라지 않는가 / 파이프라인이 도는가 /
나눈 대가를 치르는가)과 **여섯이 대응하고**, LLM 은 거기에
`메모리_접근_정렬` 을 더했다. 그리고 **표현할 수 없는 것을 스스로
식별했다** — 캐시 적중률·트랜잭션 수·동기화 지연은 실제로 원시 값에 없다.

### 재발견과 새 축

| 조건 | 채택 | 유효 | 엄격 재발견 | 단조 재발견 | 새 축 |
|---|---:|---:|---:|---:|---:|
| (가) 옛프롬 free | 17 | 16 | **0** | 3 | 13 |
| (나) 새프롬 free | 16 | 15 | **6** | 2 | 7 |
| (다) 새프롬 guided | 15 | 14 | **3** | 2 | 9 |

"유효" 는 이 4형상에서 상수/비유한이 아닌 것. 엄격 = 스피어만·피어슨
둘 다 0.95 초과, 단조 = 스피어만만.

**★ 프롬프트 개편의 효과가 압도적이다 — 엄격 재발견 0 → 6.**
(나)에서 `occupancy_deficit` · `reg_pressure` · `smem_pressure` 가
**스피어만·피어슨 둘 다 1.000** 으로 나왔다. 옛 프롬프트로는 하나도
못 맞췄던 것이다.

```
(가) -> (나)   프롬프트 개편   엄격 0 -> 6
(나) -> (다)   카테고리화      엄격 6 -> 3
```

### physics_coverage — `physics_seeded` 여섯 항

```
(나) free     3/6   traffic_amplification / smem_pressure / split_k_cost
(다) guided   1/6   split_k_cost
```

**둘 다 `has_spill` 을 못 덮었다** (최근접 피어슨 0.000). `has_spill`
하나로 1.1637 → 3.1841 이 갈렸으므로 (§8.2), **F1 라이브러리만으로는
그 물리가 표현되지 않는다는 것이 결과다.** 고치지 않는다.

### 편중 — 목적을 달성하지 못했다

| 조건 | 최대 집중 |
|---|---|
| (가) 옛프롬 free | 접두사 최대 1개 (`split_k_*` 3, `cta_*` 5 는 **첫 토큰** 기준) |
| (나) 새프롬 free | 접두사 최대 1개 |
| (다) 새프롬 guided | **`split_k_*` 3개** |

영역별 배분 (다):

```
산술_대역폭_압력       0개   ★ 3회 연속 거부로 건너뜀
타일_경계_낭비        3개
메모리_접근_정렬       1개
자원_점유율          3개
작업량_파동_충전       2개
분할_k_결합          3개
파이프라인_명령_효율     3개
```

**★ 예상과 달랐다.** 사전 등록에 "카테고리화가 편중을 줄인다 — 그것이
목적이므로" 라고 적었는데, **(나)가 이미 고르다.** 새 프롬프트만으로
편중이 사라졌고, 카테고리화는 오히려 **영역 안에서** `split_k_*` 를
세 개 만들게 했다 — 영역을 균등하게 돌라고 지시했으니 당연한 결과다.

즉 **(가)의 편중은 생성 전략이 아니라 프롬프트 탓이었다.**

### 건너뛴 영역 — 그것도 결과다

`산술_대역폭_압력` 이 3회 연속 거부로 건너뛰어졌다. 거부 사유는 전부
`허용되지 않은 numpy 함수: np.dtype` 이다.

**"원시 값으로 표현하기 어렵다" 가 아니라 "dtype 을 보려 했다" 이다.**
roofline 판정에는 바이트/원소가 필요한데 `cfg`/`p` 에 dtype 이 없다.
LLM 이 `np.dtype` 으로 우회하려다 정적 검사에 걸렸다. **원시 필드의
빈틈이지 LLM 의 실패가 아니다.**

거부 사유 분포:

```
(나) free     np.dtype 3건, §8.3 상수 1건                  (4/20)
(다) guided   np.dtype 3건, 알 수 없는 이름 1건,
              §8.3 상수 1건, 길이 초과 1건                  (6/21)
```

### 비용

```
(가) 옛프롬 free    호출 20   입력 88,974   출력 26,492
(나) 새프롬 free    호출 20   입력 70,959   출력 27,939   45분
(다) 새프롬 guided  호출 22   입력 77,613   출력 28,892   41분
```

입력이 (가) 대비 20% 줄었다 — 프롬프트가 짧아진 만큼이다. 출력은
비슷하다.

### 판정

```
프롬프트 개편   ★ 크게 유효. 엄격 재발견 0 -> 6
카테고리화      편중 완화에는 불필요했다 (새 프롬프트가 이미 고르다)
               재발견은 6 -> 3 으로 줄었다
영역 나누기 자체 ★ 관찰로서 유효. LLM 이 사람과 거의 같게 나누고
               표현 불가능한 것을 스스로 식별했다
```

**"재발견이 줄면 실패" 로 읽지 않는다** — 사전 등록에 적어 뒀다. 다만
편중 완화라는 **목적 자체가 이미 달성돼 있었으므로**, 카테고리화를
기본으로 켤 근거가 약하다.

### 다음에 쓸 라이브러리

**(나) 새프롬 free 15개**를 쓴다. 엄격 재발견이 가장 많고
physics_coverage 도 가장 높다. `--import-featwriter` 로 2단계부터
시작할 수 있다.

---

# ★ 후속 — dtype 빈틈을 메우고 roofline 을 보강했다 (2026-08-26)

> **이것은 별도 조건이다.** 위 세 조건 비교표(가/나/다)에 섞지 않는다 —
> 원시 필드가 달라졌다 (원칙 4).
> **재현**: `python3 experiments/f1_pipeline.py F1 --stage 1 --tag free-roofline
> --extend-from runs/f1pipe-F1-free/stage1-features --categorize
> --only-category "산술|roofline|arith|대역폭|bandwidth" --per-category 5`

## 원인은 "표현하기 어렵다" 가 아니었다

`산술_대역폭_압력` 영역이 3회 연속 거부로 건너뛰어졌고 사유는 전부
`허용되지 않은 numpy 함수: np.dtype` 이었다.

```
필요   바이트/원소 (roofline 은 FLOP/byte 다)
있음   p.dtype — 그런데 **문자열**("f16")이다
막힘   np.dtype(...).itemsize 가 샌드박스 허용 목록에 없다
```

**필드 노출의 빈틈이지 LLM 의 실패가 아니었다.** `p.bytes_per_element` /
`p.acc_bytes_per_element` 를 노출했다 — `dtype` 에서 계산되므로 **새 정보가
아니다.** `field_block` 에 "dtype 은 문자열이고 바이트가 필요하면 이걸
쓰라" 를 명시했다.

## 결과 — 5/5, 거부 0건

```
roofline_bandwidth_pressure
roofline_compute_time_fraction
roofline_output_store_fraction
roofline_ridge_mismatch
roofline_arithmetic_intensity_deficit
```

**거부가 하나도 없었다.** 전에는 3연속 거부로 영역이 통째로 버려졌다.
빈틈이 원인이었다는 것이 확인된다.

그리고 이번 영역 나누기의 첫 영역 설명에 **"형상과 원소 바이트 수가
만드는 연산 대 메모리 작업량"** 이 들어갔다 — 새 필드를 본 것이다.

## 라이브러리 확정

```
F1-free + roofline 보강   16 + 5 = 21개
runs/f1pipe-F1-free-roofline/stage1-features/proposals.jsonl
```

`physics_coverage` 는 **3/6 그대로**다. roofline 축은 `physics_seeded` 의
여섯 항에 없으므로 당연하다 — `traffic_amplification` /
`smem_pressure` / `split_k_cost` 를 덮고 `has_spill` 은 여전히 못 덮는다.

## ★ 영역 나누기가 세 번 다 비슷했다

같은 프롬프트로 세 번 나눴고 매번 이름과 언어가 달랐는데 **내용은 거의
같다.**

| | 1회 | 2회 | 3회 |
|---|---|---|---|
| 영역 수 | 7 | 7 | 8 |
| 언어 | 한국어 | 영어 | 영어 |
| roofline | 산술_대역폭_압력 | roofline_pressure | roofline_arithmetic_intensity |
| 타일 경계 | 타일_경계_낭비 | boundary_utilization | tile_boundary_utilization |
| 정렬 | 메모리_접근_정렬 | global_access_alignment | access_alignment_layout |
| 점유율 | 자원_점유율 | resource_occupancy | occupancy_resource_pressure |
| 병렬성 | 작업량_파동_충전 | parallel_work_distribution | grid_parallelism |
| split-K | 분할_k_결합 | reduction_and_execution_overhead | reduction_decomposition |
| 파이프라인 | 파이프라인_명령_효율 | tile_reuse_locality* | instruction_pipeline_work |
| (3회만) | — | — | cache_residency_pressure |

**여섯 축이 세 번 다 나왔다.** 이름은 매번 새로 짓지만 물리의 분해는
안정적이다. 그래서 `--only-category` 는 **여러 키워드 부분 일치**로
고른다 — 이름을 못 박으면 사람이 영역을 정의하는 것이 된다.

"뺀 것" 도 세 번 다 같은 계열이었다 — 캐시 적중률, 실제 트랜잭션/지연,
동기화. 3회차는 "각 영역의 근사 변수로만 다룬다" 고 덧붙였다.

## ★ 관찰 — LLM 이 이진형을 덜 만든다

`has_spill` 은 세 조건 어디서도 안 나왔다 (최근접 피어슨 0.000).
**그런데 `cfg.spill_bytes` 는 원시 필드에 있고**, 연속량인
`spill_magnitude` 계열(`spill_burden_ratio`, `spill_traffic_burden`)은
만들었다.

```
연속량 (로그 압축, 비율)   잘 만든다
이진 판정 ("켜지면 자릿수가 달라진다")   덜 만든다
```

프롬프트 예시 (3) `page_fault_present` 가 **정확히 그 형태**인데도 그렇다.
`has_spill` 은 그 하나로 1.1637 → 3.1841 이 갈린 항이므로 (§8.2), 이
경향은 실질적인 한계다.

**확인이 필요하면 그 영역만 따로 요청하면 된다** — 지금은 관찰로만 남긴다.

---

# ★ 2단계 전 렌더링 확인에서 둘이 나왔다 (LLM 0회)

Architect 프롬프트를 **실제로 렌더링해서** 읽었다. 둘 다 파일만 봐서는
안 보인다 (원칙 16).

## (1) 프롬프트가 사람 피처 이름을 박아 두고 있었다

```
role/architect.md   s = np.log2(f.traffic_amplification) * w[0]
                    s = s + f.tail_waste * w[1]
                    if p.is_memory_bound:
                    s = s + p.log_sol_ms * w[0]      ⛔ 예시
_base.md            물리량을 거치세요 — 예: `if p.is_memory_bound:`
role/_rules_common  if p.is_memory_bound: / f.tile_bytes / f.waves
role/optimize.md    f.waves / f.tail_waste
role/analyze.md     is_two_stage / split_k_cost
role/_rules_edit.md log_workspace_bytes / log_dram_traffic / is_two_stage
```

**F1 조건에서 이것은 답을 건네주는 것이다** (D-35). 레지스트리에 없는
이름인데 물리를 지목한다 — "traffic amplification 이라는 것이 있다",
"memory bound 로 나누면 된다".

전부 `f.<이름>` / `p.<형상값>` 같은 자리표시자로 바꿨다. 그리고
**레지스트리 이름이 프롬프트 어디에도 없어야** 실패하는 테스트를 붙였다.

## (2) ★ F1 은 형상 수준 피처를 만들 수 없다

```
F1 라이브러리 21개 중 형상 수준: 없음
```

`register_generated` 가 `shape_level` 을 **한 번도 설정하지 않는다.**
생성된 피처는 전부 config 수준(`f.*`, 배열)으로 등록된다.

**그러면 `if p.<x>:` 분기가 불가능하다.** 그런데 §30.9 와 `architect.md`
가 말하는 규칙의 핵심 표현 수단이 바로 그것이다 — "어느 물리가 언제
지배하는지" 를 형상 수준 분기로 나타낸다. F1 은 그것을 못 한다.

**실제로 형상 수준인 것이 있다.** 21개 각각을 한 형상의 모든 후보에서
계산해 보니 **5개가 형상 안에서 상수**다.

```
roofline_arithmetic_intensity_deficit   ★ 상수
roofline_bandwidth_pressure             ★ 상수
roofline_compute_time_fraction          ★ 상수
roofline_output_store_fraction          ★ 상수
alignment_guarantee_deficit             ★ 상수
```

넷이 roofline 계열이다 — **당연하다.** roofline 은 형상과 하드웨어로
정해지고 config 와 무관하다. 그런데 파이프라인이 그것을 config 수준으로
등록해서, 규칙이 `if p.roofline_...:` 로 분기할 수 없고 `f.` 로만 쓸 수
있다. **배열 안의 값이 전부 같으므로 그 항은 순위를 하나도 바꾸지
못한다** — `_rules_common.md` 의 절대 규칙 2 가 말하는 바로 그 상태다.

```
즉 F1 라이브러리 21개 중 5개는 규칙에 넣어도 아무 효과가 없다.
```

**이것은 F1 조건의 성질이 아니라 파이프라인의 결함이다.** 사람이 쓴 24개는
`shape_feature` 데코레이터로 손수 표시했지만, 생성 경로에는 그 표시가 없다.

**고치는 법은 기계적이다** — 값이 형상 안에서 상수면 형상 수준이다.
사람이 아무것도 안 알려줘도 되고, LLM 이 선언할 필요도 없다.
**아직 안 고쳤다.** 고치면 F1 조건이 또 바뀌므로 지시를 기다린다.

---

# 2단계 — Architect 두 팔 (2026-08-26)

> **재현**: `python3 experiments/f1_pipeline.py F1 --stage 2 --tag free-roofline --n-architect 10`
>          `python3 experiments/f1_pipeline.py F3 --stage 2 --seed-source architect --tag arch24 --n-architect 10`
> 두 팔이 **같은 프롬프트**를 쓴다. `_rules_edit.md` 모순을 제거한 뒤다 —
> `architect-gate.md` 의 1.1942 와 비교하지 않는다 (원칙 4).

## ★ 첫 실행은 버렸다 — 형상 수준 피처가 0개인 채로 돌았다

`_load_stage1` 이 `load_generated` 에 `table` 을 안 넘겨서 `shape_level`
을 **재판정하지 않았다.** 기록된 값은 대부분 없음(=False)이라, F1 팔이
**형상 수준 피처 0개**로 돌았다 — D-65 가 고치려던 바로 그 상태다.

```
버린 실행   p.* 분기 0/10, 전부 8항, 출력 6,049 토큰
고친 실행   p.* 분기 8/10
```

**"LLM 이 형상 수준 피처를 안 쓴다" 로 읽을 뻔했다.** 파이프라인 결함이
LLM 의 성질로 보이는 전형적인 경우다 (원칙 1). 버린 실행은
`stage2-architect-BROKEN-noshapelevel/` 에 정정 이력으로 남긴다.

## 비교 — 구조로

| | **F1 21개** | **사람 24개** |
|---|---|---|
| 성공 | 10/10 (정적 검사 재시도 1) | 10/10 |
| 학습 regret 중앙 | 1.2776 | **1.1550** |
| 학습 regret 최소 | 1.1551 | **1.1077** |
| 학습 regret 폭 | 1.1551~1.5032 | 1.1077~1.1866 |
| **`p.*` 분기** | **8/10** | 6/10 |
| 항 수 | 8항 x9, 7항 x1 | 8항 x9, 7항 x1 |
| 비용 (입력/출력) | 144,528 / 14,575 (11호출) | 118,404 / 26,978 (10호출) |

**점수는 사람 24개가 낫다** — 중앙 0.12, 최소 0.05 차이. 시드 폭
(0.0274) 을 크게 넘으므로 이것은 실재하는 차이로 보인다. 다만 이 값은
**학습 분할 regret** 이고 구조 홀드아웃은 §12.3d 대로 아직 안 본다.

**폭이 크게 다르다.** F1 은 1.15~1.50 으로 벌어지고 사람 24개는
1.11~1.19 에 모인다. 사람 피처는 어떻게 조합해도 비슷한 데 도달하고,
F1 라이브러리는 조합에 따라 크게 갈린다.

## 씨앗

### F1 21개 — `architect-try09`, 학습 1.1551, 8항, ★ 분기 있음

```python
def score(f, p, hw, w):
    s = np.log2(f.global_tile_traffic_amplification) * w[0]
    s = s + f.boundary_predication_fraction * w[1]
    s = s + f.tail_wave_imbalance * w[2]
    s = s + f.global_grid_underfill * w[3]
    s = s + np.maximum(f.shared_memory_capacity_pressure,
                       f.register_file_footprint_fraction) * w[4]
    s = s + f.spill_burden_ratio * w[5]
    s = s + f.split_k_reduction_fraction * w[6]
    if p.roofline_arithmetic_intensity_deficit > p.roofline_compute_time_fraction:
        s = s + np.log2(f.global_tile_traffic_amplification) * w[7]
    return s
```

**보강한 roofline 축 둘로 체제를 나눈다** — "대역폭 결핍이 계산 시간
비중보다 크면 트래픽 항을 한 번 더 가중한다". dtype 빈틈을 메우지
않았으면 이 분기가 존재할 수 없었다 (D-64).

⚠️ **8항으로 예산이 찼다.** 7항 이하 대안: `try04` 학습 1.2530 (7항).

### 사람 24개 — `architect-try05`, 학습 1.1077, 8항, 분기 없음

```python
def score(f, p, hw, w):
    s = np.where(f.has_spill, f.spill_magnitude + f.has_spill, f.has_spill) * w[0]
    s = s + f.edge_waste * w[1]
    s = s + f.reg_pressure * w[2]
    s = s + f.sm_idle_cost * w[3]
    s = s + np.where(p.is_memory_bound, f.log_dram_traffic, f.log_inst_total) * w[4]
    s = s + np.where(p.is_memory_bound, f.traffic_amplification, f.occupancy_deficit) * w[5]
    s = s + f.split_k_cost * w[6]
    s = s + np.where(p.can_use_cp_async, f.smem_pressure, f.pipeline_warmup_frac) * w[7]
    return s
```

`if` 문은 없지만 `np.where(p.<형상값>, A, B)` 로 **세 번 체제 전환**을
한다 — 형상 수준 값을 스칼라 조건으로 써서 항의 **내용**을 바꾼다.
F1 씨앗의 `if` 는 항의 **가중치**를 바꾼다. 같은 목적의 다른 문법이다.

⚠️ 8항. 7항 이하 대안: `try06` 학습 1.1587.

`w0` 이 `[1.0, 0.003, 0.25, 0.01, 0.03, 0.5, 1.0, 0.5]` 로 **범위에 맞춰
자릿수를 조정했다** — F1 씨앗은 `[0.25, 1, 1, 1, 1, 1, 1, 0.25]` 로 거의
균일하다. F1 피처가 대부분 [0,1] 로 정규화돼 있어서 그럴 필요가 적다.

## 같은 물리를 다른 이름으로

두 씨앗이 겹치는 물리:

```
트래픽 증폭    traffic_amplification      global_tile_traffic_amplification
경계 낭비      edge_waste                 boundary_predication_fraction
레지스터 압력   reg_pressure               register_file_footprint_fraction
smem 압력      smem_pressure              shared_memory_capacity_pressure
스필           spill_magnitude/has_spill  spill_burden_ratio
split-K        split_k_cost               split_k_reduction_fraction
wave 유휴      sm_idle_cost               tail_wave_imbalance
```

**일곱 축이 대응한다.** 이름만 다르다.

F1 에만 있는 것: `global_grid_underfill`, roofline 계열.
사람 24개에만 있는 것: `log_inst_total`(SASS 명령어 수),
`occupancy_deficit`, `pipeline_warmup_frac`, `can_use_cp_async`.

## 항 수 — 둘 다 예산이 찼다

```
F1        8항 x9, 7항 x1
사람 24개  8항 x9, 7항 x1
```

**진화가 교체만 가능한 상태다** (D-35 계열). 두 팔 다 7항 이하 대안을
`chosen.json` 옆에 기록해 뒀다 — 진화가 갇히면 그것으로 다시 본다.

---

# 4-4 — `expected_range` 의 출처 (2026-08-27)

> **재현**: `python3 -m pytest tests/test_features.py -k expected_range`
> LLM 0회.

## 질문

```
FeatureWriter 가 expected_range 를 선언한다
파이프라인이 그것을 **실측으로 덮어쓰는가**?

덮어쓴다  ->  ★ 표 정보가 프롬프트에 들어간다. F1 21개 결과가 오염
안 덮는다 ->  LLM 이 식에서 유도한 것. 문제 없음
```

**LLM 은 표를 못 보므로 스스로 측정할 수 없다.** 누출이 있다면
파이프라인이 만든 것이다.

## 답 — 덮어쓰지 않는다. 21/21 일치

```
LLM 선언  ->  register_generated  ->  Feature.expected_range
          ->  render_features     ->  프롬프트
```

세 지점의 값이 **21개 전부 같다.** 코드 경로에도 덮어쓰는 곳이 없다.

```
validate_feature   선언 범위 밖이면 **경고만** 한다. 다시 쓰지 않는다
annotate()         `physical.py`(사람 24개)에서만 불린다. 생성 경로에는 없다
```

## 실측 값이 새는 경로도 없다

`validate_feature` 의 범위 경고 문구에는 실측 min/max 가 들어간다.

```python
f"선언 [{lo}, {hi}] 밖 {out}/{n}개 (실측 [{min:.4g}, {max:.4g}])"
```

**그런데 그 문구는 LLM 에 안 간다.**

```
범위 검사의 level 은 "warn" 이다
FeatureRejected 는 `rep.fails()` — level == "fail" 만 — 로 만들어진다
따라서 실측 값이 거부 메시지에 들어가지 않는다

그리고 `f1_pipeline.py` 는 거부를 **되먹이지 않는다** (기록만 한다).
구 스크립트 `feature_writer.py` 는 되먹이지만, 위 이유로 실측 값은
그 문자열에 없다.
```

## 방향도 맞는다 — 선언이 실측보다 **넓다**

누출이면 선언이 실측을 바짝 감쌀 것이다. 반대다.

| 피처 | 선언 | 실측 |
|---|---|---|
| `boundary_predication_fraction` | [0, 1] | [0.9688, 0.9961] |
| `instruction_work_per_flop` | [0, 1] | [1.1e-05, 2.1e-04] |
| `l2_tile_residency_pressure` | [0, 1] | [0.0013, 0.0078] |
| `alignment_guarantee_deficit` | [0, 3] | [0.3333, 0.3333] |

**전부 식에서 유도할 수 있는 경계다** — 비율이면 [0,1], 세 행렬 합이면
[0,3]. 실측을 봤다면 이렇게 넓게 쓸 이유가 없다.

사람이 쓴 24개도 과대 선언을 그대로 뒀다 — `waves` [0, 1e5],
`roofline_ratio` [0, 1e4]. **좁히지 않은 판단이 코드에서도 지켜진다.**

## ★ 내 비교가 오탐을 하나 냈다

`global_tile_traffic_amplification` 이 "★ 다름" 으로 찍혔다.
선언·레지스트리·프롬프트가 전부 `[1, inf]` 인데도 그랬다.

```python
abs(got[1] - hi) < 1e-9      # abs(inf - inf) = nan, nan < x 는 False
```

**부동소수 비교가 무한대에서 무너진다.** 실제 누출이 아니다 —
계측이 만든 오탐이 또 나왔다 (원칙 14).

## 결론

```
✅ 누출 없음. F1 21개 라이브러리를 다시 만들 필요가 없다
✅ F1 실험 결과(D-63, D-68)의 유효성이 이 경로로는 훼손되지 않는다
```
