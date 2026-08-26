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
1. docs/principles.md            반복해서 밟은 것 10가지. 여기부터
2. docs/design.md                설계 전체 (§0~§31, 부록 A/B/C)
3. docs/artifacts/conclusion.md  현재 결론 — 무엇이 확정, 무엇이 미확정
```

나머지는 필요할 때만 본다.

```
docs/glossary.md            용어 — 이름이 틀려서 결론이 두 번 뒤집혔다
docs/decisions.md           D-1~D-53 시간순 기록. principles.md 의 상세
docs/artifacts/*.md         실험별 결과. **최상단 상태 배지를 먼저 보라**
docs/artifacts/cost.md      비용 집계 — 탐색 vs 재현
docs/examples/              리포트 샘플
experiments/README.md       스크립트 ↔ artifact 매핑
```

## 현재 상태 (2026-08-26)

```
확정   벤더 구조 홀드아웃 1.0737 / 정적 top-1 1.115 / GBDT 1.019
      체제 경계를 대리 지표(SOL)로 판정 가능 — t_best 와 61/61 일치
      has_spill 항 하나가 regret 1.1637 -> 3.1841 (결정론적)
      판별 한계 0.03 — 시드 표준편차 0.0274 (D-53)

미확정 ★ 성능 주장이 없다. luna 6시드 구조HO 중앙 1.1019 > 벤더 1.0737
      피처 설명 효과가 luna 에서 재현 안 됨 (0.016, p=1.000)
      F1 / Architect / 새 축 결과는 지시 없이 도입된 모델의 것 (D-52)
```

**관문은 단일 임계값이 아니라 비용-성능 표다** (§9.2c). regret 한 칸만 보고
"넘었다/못 넘었다" 를 말하지 않는다.

## 문서 갱신 규칙 세 가지

1. 모든 관문 숫자에 **재현 절차**를 붙인다 (status 필터 / 덮개 정의 / 형상
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

## 현재 상태 (dev-cu124)

```
관문     벤더 nearest  1.080   (strict 1.102 병기)
GBDT 상한  1.044       (짧은 형상 블록 8분할 기하평균 / 최악 1.085)
손규칙     1.172
```

★ **판정은 분할별로 한다.** 분할 x 방법 표를 항상 병기한다 (§30.6c).

> 개발용 표(CUDA 12.4 / 호스트, `schema_version 1`)의 값이다.
> 본 캠페인(CUDA 13.3 / 컨테이너) 표가 오면 재계산한다.

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
