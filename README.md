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

## 문서

```
docs/kernelrule_design.md            본 설계 (§0~§20, 부록 A/B)
docs/kernelrule_design_addendum.md   보완 (§21~§30, 부록 C)  ★ 이 저장소가 유지
docs/decisions.md                    결정과 발견 기록
```

**갱신 규칙 세 가지** (addendum 머리말):

1. 모든 관문 숫자에 **재현 절차**를 붙인다 (status 필터 / 덮개 정의 / 형상
   집합 / 후보 집합 / 홀드아웃 분할 / 계산한 스크립트)
2. 틀린 값을 **지우지 말고 정정 이력으로** 남긴다
3. 두 숫자를 나란히 쓰기 전에 **같은 절차 / 같은 분모 / 같은 집계 방식 /
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
