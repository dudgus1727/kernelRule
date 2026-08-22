# (가)가 갇힌 이유 — 예산 포화가 아니라 항 하나

**측정일** 2026-08-22 · **LLM 호출 0회** · 정준 절차(체제별 적합, 61형상 결합)

---

## 예산 포화 가설은 기각됐다 — 기존 데이터로

가설은 *"`physics_seeded` 는 7항이고 Architect A 최고는 8항이라 예산이 차서
탐색이 교체로만 가능해졌다"* 였다.

**Architect A 산출물 10개 중 항이 8개 미만인 것은 `#9` 하나뿐이고, 그것이
train 최소값(1.1942)이라 이미 (가)의 씨앗이었다 — 7항이다.**

그리고 (가) s0 의 아카이브를 보면 씨앗은 r1 에 8항으로 채워졌다. 여유가
있었는데 채우고 갇혔다. 예산이 원인이 아니다.

---

## 원인 — `has_spill` 항 하나

두 구조의 피처 차이:

```
Architect 만   is_two_stage, log_dram_traffic, occupancy_deficit, tail_waste
physics 만     has_spill, sm_idle_cost, smem_pressure
공통          pipeline_warmup_frac, split_k_cost, traffic_amplification
```

**항을 하나씩 빼 보면 `has_spill` 만 자릿수가 다르다** (AST 로 지우고 `w`
인덱스를 다시 매긴 뒤 재적합):

| `physics_seeded` 에서 뺀 항 | 정준 61 |
|---|---:|
| (원본, 7항) | 1.1637 |
| − `f.has_spill` | **3.1841** |
| − `f.sm_idle_cost` | 1.3269 |
| − `f.pipeline_warmup_frac` | 1.1706 |
| − `f.smem_pressure` | 1.1675 |
| − `f.split_k_cost` | 1.1555 |

**반대 방향도 맞는다.** Architect A `#9` 에 physics 전용 피처를 하나씩
더해 보면:

| Architect A `#9` (7항) 에 더한 항 | 정준 61 |
|---|---:|
| (원본) | 1.1780 |
| + `f.has_spill` | **1.1456** ← `physics_seeded` 1.1637 보다 낫다 |
| + `f.sm_idle_cost` | 1.1705 |
| + `f.smem_pressure` | 1.1700 |

**Architect 구조는 나쁘지 않다. 항 하나가 없었을 뿐이다.**

### 왜 그렇게 큰가

```
스필 커널은 후보의 7.4% 인데 정답 집합에 든 형상이 0/61 이다
```

**비용 없는 가지치기다.** 후보의 13분의 1을 공짜로 버릴 수 있는데, 그
항이 없으면 규칙이 그것들을 계속 1등으로 고른다.

---

## ★ 왜 Architect 는 그것을 못 썼나 — 프롬프트의 비대칭

`has_spill` 은 피처 목록에 있었고 설명은 이랬다.

```
f.has_spill    [0, 1]    레지스터 스필이 있는가. 0 또는 1.
```

**무엇인지는 말하지만 왜 나쁜지는 말하지 않는다.** 범위 `[0,1]` 은
`tail_waste` 와 같아서 "둘 다 비슷한 크기의 벌점" 으로 읽힌다.

`physics_seeded` 를 쓴 쪽은 설계 문서를 통째로 읽었고, 레지스터 스필이
로컬 메모리(= DRAM)로 나가는 것이라 **수십 배** 느리다는 것을 알았다.
`w0` 도 그 항만 `3.0` 으로 줬다 (나머지는 0.3~1.0).

**이것은 표 관측이 아니라 하드웨어 물리다** — 어느 GPU 에서도 참이고,
§12.3b 가 허용하는 종류다. `physical_meaning` 이 "무엇을 재는가" 만 담고
"왜 그것이 성능을 좌우하는가" 를 안 담은 것이 문제다.

### 그리고 Optimizer 는 설명을 아예 못 본다

```
Architect   render_features()  ->  이름 + 범위 + 물리적 정의
Optimizer   _feature_block()   ->  ★ 이름만
```

`optimize.md` 의 `{feature_list}` 는 `- \`has_spill\`` 같은 이름 목록이다.
진화 루프의 LLM 은 **어느 피처가 무엇을 재는지 모르는 채로** 항을 고른다.
Architect 에서 범위를 안 줬을 때 regret 8.4 가 나온 것과 같은 종류의
구멍이다 (`artifacts/architect-gate.md`).

**미적용.** (나') 실행 중이라 프롬프트를 건드리지 않았다. 고칠 것 둘:

1. `Feature.physical_meaning` 에 **왜 그것이 성능을 좌우하는가**를 넣는다.
   표 관측이 아니라 하드웨어 물리로 (§12.3b).
2. `optimize.md` 도 `render_features()` 를 쓰게 한다 — 두 역할이 같은
   피처 설명을 봐야 한다.

---

## 남는 질문

이 결과는 **"A 조건이 표 없이 좋은 구조를 만든다"** 를 약화시키지 않는다.
Architect 는 항 하나만 더 있었으면 `physics_seeded` 를 이겼다(1.1456).
그 항을 못 고른 이유가 표 접근이 아니라 **피처 설명의 부실**이라면,
프롬프트를 고쳐서 해결되는 문제다.

**시험:** `physical_meaning` 을 고친 뒤 A 조건 10회를 다시 돌려
`has_spill` 을 쓰는 비율과 최고 regret 을 본다.
