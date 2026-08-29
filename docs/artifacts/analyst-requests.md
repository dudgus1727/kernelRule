# Analyst 가 요구한 피처 303건 — 전부 버려졌다 (2026-08-28)

> **상태**: 관찰 확정. **경로는 2026-08-28 에 구현했다** (D-75, 아래)
> **재현**: `runs/*/hypotheses.jsonl` 의 `needs_new_feature` 집계. LLM 0회
> **원문 303건**: `docs/artifacts/analyst-requests.json` — 요약이 아니라 **전문**이다
> **표**: `datasets/rtx-a6000-sm_86-c63710df` (dev)

## 경로가 없다

```
schemas.py    needs_new_feature 필드가 있다
analyze.md    "목록에 없는 물리량이 필요하면 needs_new_feature 에 쓰세요"
loop.py       ★ 그 필드를 읽는 코드가 없다
              calls["feature"] 가 0 으로 초기화되고 한 번도 안 늘어난다
```

라운드 흐름:

```
1~2.  진단 리포트 -> Analyst -> out["hypotheses"] 만 꺼낸다
      ★ needs_new_feature 는 안 본다
3.    (FeatureWriter 자리 — 비어 있음)
4.    Optimizer x 12
```

## 얼마나 요구했나

```
가설 총 1,655건   needs_new_feature 채워진 것 303건 (18.3%)
가설을 남긴 34실행 중 33실행에서 나온다 (실행당 1~20건)
안 나온 하나는 smoke2 — 연기 시험이라 라운드가 1회다
고유 문장 293개, 길이 중앙값 49자 (최소 16, 최대 102)
```

> **정정** (2026-08-28): 처음 이 문서와 보고에 "33개 실행 **전부**" 라고
> 썼다. 가설을 남긴 실행은 34개이고 그중 33개다. 분모를 잘못 적었다.

**길이 중앙값 49자가 설계를 정한다** — 요구는 거의 전부 한 구절이다.
`physical_requirement` 를 짧은 문자열 한 개로 두면 충분하고, 진단
리포트를 통째로 넘길 이유가 없다(아래 조건 1).

**"라이브러리로 충분했다" 가 아니다.** 다섯 번에 한 번꼴로 "이걸
재려면 새 축이 필요하다" 고 말했고, **그 말이 전부 버려졌다.**

## ★ 무엇을 요구했나 — 이것이 관찰이다

주제별(중복 허용):

| 건수 | 주제 |
|---:|---|
| 82 | wave / CTA **절대량** — 비율 말고 몇 개인가 |
| 64 | L2 재사용 **이득** — 압력만 있고 이득이 없다 |
| 51 | 파이프라인 계열 — `stages=2` vs multistage, cp.async 가능성 |
| 48 | split-K 의 **이득** — 비용만 있고 병렬성 이득이 없다 |
| 21 | launch / CTA 고정 비용 — 짧은 커널에서 비중이 크다 |
| 15 | warp 수준 세분 — accumulator 타일, warp 간 분할 |
| 13 | M/N 비대칭 — 방향성 트래픽 |
| 5 | 정밀도 / dtype |

### 원문 예시 — 손대지 않은 그대로

```
절대 CTA 수 또는 총 wave 수
split-K가 제공하는 CTA 병렬성·wave 충전 이득 또는 split-K 적용 전후의
  유효 wave 부족 완화량
pipeline implementation family 또는 cp.async eligibility
커널 실행 계열별 steady-state issue 효율 또는 MMA 파이프라인 효율
swizzle별 global-memory coalescing 및 shared-memory bank-conflict 비용   (4회)
```

**★ 가장 많이 반복된 문장(4회)이 하필 금지 축이다.** `swizzle` 은
`cfg.ext` 라 프롬프트가 명시적으로 막는다(§4.3, 아키텍처 전용이라 다른
GPU 로 안 옮겨진다).

```
LLM 이 가장 자주 원하는 축을 우리가 **설계상 막고 있다.**
```

이것은 결함이 아니라 **비용이다.** 전이 가능성을 지키려고 표현력을
내주고 있고, 그 값이 얼마인지는 모른다 — `cfg.ext` 를 여는 조건을
따로 돌려 보기 전에는 못 잰다. 자주 나온다고 만들 수 있는 것이 아니다.

**둘째·넷째가 같은 형태다** — "우리 피처는 **비용만** 재고 **이득**을
안 잰다". split-K 는 리덕션 비용이 있지만 CTA 병렬성 이득이 없고,
L2 는 압력이 있지만 재사용 이득이 없다.

## ★ 대부분 이미 있는 필드로 표현 가능하다

| 요구 | 필요한 것 | `RAW_FIELDS` 에 |
|---|---|---|
| 파이프라인 계열 | `cfg.pipeline_kind` | ✅ 있다 |
| split-K 이득 | `cfg.split_k` | ✅ 있다 |
| split-K 모드 | `cfg.split_k_mode` | ✅ 있다 |
| wave/CTA 절대량 | `cfg.max_blocks_per_sm`, `hw.sm_count` | ✅ 있다 |
| L2 | `hw.l2_bytes` | ✅ 있다 |
| M/N 비대칭 | `cfg.tile_m` / `tile_n` | ✅ 있다 |
| **warp 세분** | `ext_warp_m` / `ext_warp_n` | ★ `cfg.ext` — **의도적 금지** (§4.3) |
| **swizzle** | `ext_swizzle_n` | ★ 같음 |

**문자열 필드도 쓸 수 있다** — dtype 함정이 아니다.

```python
cfg.pipeline_kind == "multistage"   # 정적 검사 통과, 실제로 0/1 로 나뉜다
cfg.split_k_mode == "parallel"      # 같다
```

즉 상위 네 주제(245건)는 **표현할 수단이 있는데 안 만든 것**이다.
FeatureWriter 는 1단계에서만 돌고, Analyst 의 요구가 그쪽으로 가는
경로가 없다.

## 하위 두 주제는 조건상 못 만든다

`warp_*` / `swizzle` 은 `cfg.ext` 이고 프롬프트가 명시적으로 금지한다
— **아키텍처 전용 필드**라 다른 GPU 로 안 옮겨진다 (§4.3). 28건이
거기 걸린다. **이것은 결함이 아니라 설계다.**

## 다음 — 경로를 만든다면

```
Analyst -> 가설
  needs_new_feature 가 채워짐?
    -> FeatureWriter 호출 -> 검증 -> 레지스트리 추가
    -> ★ Analyst 로 되돌아가 다시 가설
Optimizer x 12
```

**조건 셋:**

```
1  FeatureWriter 에게 진단 리포트를 주지 않는다
   ✅ "타일이 L2 에 얼마나 남는지"        물리적 요구
   ❌ "사례 #7 에서 이 형상이 느렸다"      표 정보
   -> AnalysisOutput 에 physical_requirement 필드를 두고 그것만 넘긴다
   -> 루프 안에서 만든 피처가 학습 형상에 맞춰지는 것을 막는다

2  라운드당 1~2개 상한
   §21 피처 행렬이 새 피처마다 약 10초. 캐시 키가 레지스트리 해시라
   라운드마다 바뀌면 캐시가 안 듣는다

3  판정은 성능이 아니라 관찰로
   빈도 / 만든 것의 사용률 / ★ 요구 내용
   성능은 0.02급이라 어차피 못 가린다 (D-53)
```

**2026-08-28 에 구현했다.** `LoopConfig.max_new_features_per_round`
(기본 **0 = 꺼짐**), `RoundLoop._write_features`, `loop._feature_task`.
조건 셋을 코드가 지키고 시험이 고정한다:

```
test_feature_path_is_off_by_default             기본은 꺼짐 — 옛 조건 유지
test_analyst_request_reaches_the_feature_writer 요구 -> 호출 -> 등록 -> 열 생성
test_feature_writer_never_sees_the_diagnostic_report  ★ 조건 1
test_requirement_reads_the_old_field_name       옛 필드명도 읽는다
```

⚠️ 필드 이름을 `needs_new_feature` -> `physical_requirement` 로 바꿨다.
뜻은 같고 `_requirement_of` 가 둘 다 읽는다 — 옛 33실행을 다시 읽을 때
조용히 0건이 되면 안 된다.

**아직 돌리지는 않았다** (LLM 호출 0회).
