# 실행 좌표 — **생성물이다**

> ★ 이 표는 `experiments/runs_table.py` 가 만든다. **손으로 고치지 마라** —
> `runs/*/config.json` 과 산출물 json 이 원본이고, 달라지면
> `tests/test_docs.py` 가 잡는다 (`--check`).
>
> ```
> python3 experiments/runs_table.py            # 다시 만든다
> python3 experiments/runs_table.py --check    # 달라졌는지만 본다
> ```
>
> ⚠️ **캠페인이 도는 중이면 `--check` 가 빨간 것이 정상이다** — 라운드마다
> `config.json` 이 갱신되므로 표가 뒤처진다. 끝나고 다시 만들면 된다.
> 이 검사가 잡으려는 것은 **손으로 고친 표**다.

## 태그 규칙 (D-128)

```
<피처><씨앗>-p<파라미터>[-<표현력>][-<실험명>]

F3rw-p8        F3 라이브러리 · RuleWriter 씨앗 · 파라미터 8 · 기본
F3rw-p16       파라미터 16
F3rw-p8-prod   곱 힌트          F3rw-p8-pow   지수 힌트
F3hg-p8-d75-a  human_guided 씨앗
★ 표(GPU)·계승·코드 판은 태그에 안 넣는다 — config.json 이 갖는다
★ `x-` 로 시작하는 디렉토리는 **폐기**다 (순위 손실 계열 · 조건 오류).
   지우지 않고 이름으로 표시했고, 이 표에는 안 들어간다
```

<!-- RUNS:BEGIN — experiments/runs_table.py 가 만든다 -->

| 태그 | 시드 | 피처 | 씨앗 | 파라미터 | 표현력 | 적합기 | 라운드 | 표 | 최종 점수 | 출처 | 상태 |
|---|--:|---|---|--:|---|---|---|---|--:|---|---|
| `F1rw-p8` | 6 | 16/? | rule_writer-try09 | ? | 기본 | nelder-mead/4/200 | 12 | a6000 | 1.1195 | conclusion.json | |
| `F2rw-p8` | 6 | 17/? | rule_writer-try01 | ? | 기본 | nelder-mead/4/200 | 12 | a6000 | 1.1288 | conclusion.json | |
| `F3hg-p8-d75-a` | 3 | 20/F3 | human_guided | 8 | 기본 | nelder-mead/4/200 | 4 | a6000 | — | — | |
| `F3hg-p8-d75-b` | 6 | 19/F3 | human_guided | 8 | 기본 | nelder-mead/4/200 | 4 | a6000 | — | — | |
| `F3rw-p16` | 3 | 19/F3 | rule_writer-try05 | 16 | 기본 | cma/1/300 | 12 | a6000 | 1.0906 | expressive-regret.json | |
| `F3rw-p8` | 6 | 19/F3 | rule_writer-try05 | 8 | 기본 | nelder-mead/4/200 | 24 | a6000 | 1.0787 | canon-p8.json | |
| `F3rw-p8-4090` | 3 | 19/F3 | rule_writer-try00 | 8 | 기본 | nelder-mead/4/200 | 12 | 4090 | 1.0493 | sigma-4090.json | |
| `F3rw-p8-5090` | 3 | 19/F3 | rule_writer-try05 | 8 | 기본 | nelder-mead/4/200 | 12 | 5090 | 1.0611 | c-ladder.json | |
| `F3rw-p8-abl-analyst` | 3 | 19/F3 | rule_writer-try05 | 8 | 기본 | nelder-mead/4/200 | 12 | a6000 | — | — | |
| `F3rw-p8-abl-noanalyst` | 3 | 19/F3 | rule_writer-try05 | 8 | 기본 | nelder-mead/4/200 | 12 | a6000 | — | — | |
| `F3rw-p8-abl-shuffled` | 3 | 19/F3 | rule_writer-try05 | 8 | 기본 | nelder-mead/4/200 | 12 | a6000 | — | — | |
| `F3rw-p8-cma` | 3 | 19/F3 | rule_writer-try05 | 8 | 기본 | cma/1/300 | 12 | a6000 | 1.0987 | expressive-regret.json | ⛔ 폐기 — p8 인데 CMA — 지금 규칙(fitter_for)으로는 안 나온다 |
| `F3rw-p8-cross` | 3 | 19/F3 | rule_writer-try05 | 8 | 기본 | nelder-mead/4/200 | 12 | a6000 | — | — | |
| `F3rw-p8-d75` | 6 | 21/F3 | rule_writer-try05 | 8 | 기본 | nelder-mead/4/200 | 4 | a6000 | — | — | |
| `F3rw-p8-old` | 6 | 19/? | rule_writer-try05 | ? | 기본 | nelder-mead/4/200 | 12 | a6000 | 1.0762 | conclusion.json | ⛔ 폐기 — 옛 대표값 — 옛 프롬프트·라운드12·patience10 (D-129) |
| `F3rw-p8-p3` | 6 | 19/F3 | rule_writer-try05 | 8 | 기본 | nelder-mead/4/200 | 5~6~7 | a6000 | — | — | |
| `F3rw-p8-pow` | 3 | 19/F3 | rule_writer-try05 | 8 | 지수 | cma/1/300 | 12 | a6000 | 1.0839 | expressive-regret.json | ⛔ 폐기 — p8 인데 CMA. 재측정 대상 |
| `F3rw-p8-prod` | 3 | 19/F3 | rule_writer-try05 | 8 | 곱 | cma/1/300 | 12 | a6000 | 1.0840 | expressive-regret.json | ⛔ 폐기 — p8 인데 CMA. 재측정 대상 |
| `luna` | 3 | 19/? | ? | ? | 기본 | nelder-mead/4/200 | 12 | a6000 | — | — | |
| `lunaNAMES` | 6 | 19/? | ? | ? | 기본 | nelder-mead/4/200 | 12 | a6000 | — | — | |
| `verify` | 2 | 19/? | ? | ? | 기본 | nelder-mead/4/200 | 6 | a6000 | — | — | |

<!-- RUNS:END -->

## 각주

```
★ n=6 이 표준이다. n=3 인 실행은 원리적으로 판정이 약하다
  (3대3 에서 "안 겹침" 의 최소 p = 0.10) — 재측정 대상
목적함수      regret@1 (D-128 이후 진화 경로는 이것뿐이다)
분할          nk11008 — 학습/홀드아웃 20. 표마다 학습 수가 다르다
모델          gpt-5.6-luna 고정
적합기        파라미터 8 -> nelder-mead/4/200,  16 -> cma/1/300 (D-128)
              ⚠️ 표시가 붙은 줄은 **그 규칙과 다르게 돈 옛 실행**이다
최종 점수        체제별 재적합 -> 홀드아웃. **출처 열의 json 에서 읽는다**
★ 라운드 12 는 검증 안 된 값이다 — D-127 판정은 "(나) 부족하다" 이고,
  부족한 양(0.0055~0.0060)이 시드 폭 σ(0.0124)보다 작다
피처 열       `n_features`/`feature_condition`. `?` 는 그 키가 없던 시절이다
씨앗 열       `chosen.json` 의 source. `?` 는 파일이 없는 옛 실행
```

## 아직 최종 점수가 빈 줄

`—` 는 **그 태그의 최종 점수를 담은 산출물 json 이 아직 없다**는 뜻이다.
숫자를 여기 손으로 적지 않는다 — 재측정하고 산출물을 만들면 채워진다.
