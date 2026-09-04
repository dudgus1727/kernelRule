# kernelRule

성능 표에서 **GEMM config 선택 규칙을 자동으로 만들어내는 파이프라인**.
LLM 이 규칙 코드를 쓰고, 미리 측정된 표가 채점하고, 진화 루프가 반복한다.

배포되는 산출물은 마이크로초 단위로 도는 순수 파이썬 함수와, 그 함수가 왜
그렇게 생겼는지의 기록이다. **LLM 은 규칙을 만드는 도구이지 최종 시스템의
일부가 아니다.**

```
입력   kernelTab 번들 (table.parquet + env.json + kernels.jsonl + BUNDLE.json)
출력   score(f, p, hw, w) -> 점수 배열   +  적합된 가중치 W_FITTED
       + features/ (검증된 물리 피처 라이브러리)
       + 가설 이력 (규칙이 그렇게 생긴 이유)
```

## ★ 읽는 순서

새 세션은 **이 순서로 세 개만** 읽으면 된다. 그것으로 "지금 무엇이
확정됐고 다음에 무엇을 해야 하는가" 에 답할 수 있어야 한다.

```
1. docs/principles.md            반복해서 밟은 것 38가지 (1,084줄). 여기부터
2. docs/artifacts/conclusion.md  ★ 대표값 수치와 현재 결론 (605줄)
3. docs/design.md                설계 전체 (5,170줄 — 필요한 절만)
```

⚠️ `design.md` 는 정정 상자가 옛 서술 위에 쌓여 5,170줄이다. **현재
구현만 추린 판으로 나누는 것이 밀려 있다** (`pending_fixes` 13).

나머지는 필요할 때만 본다.

```
docs/glossary.md            용어 — 이름이 틀려서 결론이 두 번 뒤집혔다
docs/decisions.md           D-1~D-114 시간순 기록. ★ 머리에 색인이 있다
docs/artifacts/*.md         실험별 결과. **최상단 상태 배지를 먼저 보라**
docs/artifacts/cost.md      비용 집계 — 탐색 vs 재현
docs/examples/              리포트 샘플
experiments/README.md       스크립트 ↔ artifact 매핑
```

## 현재 상태 (2026-09-03, D-114 / 원칙 38)

★ **수치는 여기 안 적는다.** 대표값은
[docs/artifacts/conclusion.md](docs/artifacts/conclusion.md) 의
"★ 대표값 성능 수치" 절 하나뿐이다 — 두 곳에 두면 달라진다 (원칙 2).

```
확정   벤더와 **구분 불가** (형상별 9/11, p=0.824)
      배포 가능한 산출물 둘 — 규칙 함수 + 검증된 피처 라이브러리
      체제 경계를 대리 지표(SOL)로 판정 가능 — t_best 와 61/61 일치
      전이(§29.5): 구조는 옮겨가고 **가중치는 하드웨어 상수**다
      ★ 순위 축의 벽 — 여섯 방향에서 밀었고 안 움직였다 (D-104~D-112)

미확정 5090 (c) 재생성 — ★ A6000 하드웨어 사실을 받고 돌았다 (D-113).
      사다리 세 팔로 다시 잰다 (숫자 / 경고 절을 하나씩)
      ★ regret@k 로 벽을 다시 재는 중 (regret-at-k-prereg.md)
      design.md 를 현재/역사로 못 나눴다 (pending_fixes 13)
      셀 축 재설계로 D-42 를 다시 시험 (pending_fixes 7)
      가중치를 형상마다 하나로 두는 것 (pending_fixes 11 — 유보 붙음)
```

**통과 조건은 단일 임계값이 아니라 비용-성능 표다** (§9.2c). regret 한 칸만 보고
"넘었다/못 넘었다" 를 말하지 않는다.

### 순위 축은 닫혔다 (2026-09-03)

```
regret 이 낮은 규칙은 상위권 순서를 못 매기고, 그 반대도 마찬가지다.
구조 / 예산 / 순서 / 표현력 / 목표 정의 / 형태 — 여섯 방향 전부 음성.
남은 진술: 이 피처 공간의 **선형 결합으로는** 상위권 순서를 못 맞춘다.
```

자세한 것은 `conclusion.md` 의 "★ 벽 — 여섯 방향에서 밀었고 안 움직였다".

## 문서 갱신 규칙 세 가지

1. 모든 통과 조건 숫자에 **재현 절차**를 붙인다 (status 필터 / 덮개 정의 / 형상
   집합 / 후보 집합 / 홀드아웃 분할 / 계산한 스크립트)
2. 틀린 값을 **지우지 말고 정정 이력으로** 남긴다
3. 두 숫자를 나란히 쓰기 전에 **같은 절차 / 같은 분모 / 같은 집계 /
   같은 데이터 / 같은 모델**에서 나온 것인지 확인한다 (D-31)

## 설치

```bash
pip install -e ../kernelTab       # 표 로더 / 정답 격리 / 노이즈 모델
pip install -e '.[test,llm]'      # ★ llm 없이는 실제 실행 경로가 죽는다
python3 -m pytest tests/ -q
```

`[llm]` 은 `pydantic` + `pydantic-ai-slim[openai]` + `openai` 다.
**빼면 `tests/test_openai_client.py` 8건이 스킵되고**, 그 모듈은
`CRITICAL_MODULES` 라 세션이 실패한다 — 즉 그 실행 결과로는 아무것도
보증하지 못한다 (§26.3).

채점만 할 거면 `[test]` 만으로도 된다. 다만 LLM 실험은 못 돌린다.

```bash
export OPENAI_API_KEY=...          # 없으면 MockLLM 으로 조용히 떨어지지 않고 중단한다
```

## 절대 규칙

**규칙 함수는 표를 볼 수 없다.** 네 겹으로 막는다.

| 겹 | 장치 |
|---|---|
| 자료구조 | `Problem`/`Config`/`CandidateSet` 어디에도 시간이 없다 |
| 로더 | `load_for_ranking` / `load_for_scoring` 이원화 + 화이트리스트 |
| 정적 검사 | AST — 금지 이름, import, `M/N/K` 직접 비교, `f.*` 에 `if` |
| 행동 검사 | `null` 프리셋, 상수 점수 테스트 |

**동점은 config 정체성으로만 가른다.** 이 표는 66형상 중 29개가 최적시간에
정확한 동점이고 최대 84중 동점이다 — "그 형상의 최적 config" 는 tie-break
규칙의 함수이지 물리적 사실이 아니다. 그래서 `PerfTable` 은 `best_config` 를
제공하지 않는다.

**노이즈 바닥은 형상마다 다르다.** 고정 1% 금지. 이 표에서 11.3us 형상의
바닥은 9.1%, 9.7ms 형상은 0.048% 다.

## 표

```
dev     CUDA 12.4 / 호스트 / schema_version 1   — 개발용
본      CUDA 13.3 / 컨테이너                    — 대표값 수치는 이쪽
5090    rtx-5090-sm_120-5bb6f403                — 전이 (§29.5)
```

★ **판정은 분할별로 한다.** 분할 x 방법 표를 항상 병기한다 (§30.6c).
개발용 표로 얻은 성능 수치는 보고하지 않는다.

## 구조

```
kernelrule/
  core/     types adapter noise table scoring weights matrix splits
            sandbox archive loop
  features/ physical (25개) validate
  rules/    checks handwritten
  baselines/ static_topk vendor gbdt
  agents/   schemas mock
  report/   diagnostic
tools/synth.py    합성 표 생성기 (4 프리셋)
```

## 실행

```bash
# 진단 리포트 (사람이 먼저 읽는다 — §12.4)
python3 -c "..."   # 예시는 docs/decisions.md 참조

# MockLLM 루프 — ★ __main__ 가드가 필요하다 (샌드박스가 자식을 띄운다)
def main(): ...
if __name__ == "__main__": main()
```
