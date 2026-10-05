# 시간 피처 16개 — 물리 설명만 남긴 판 (2026-10-05)

`time_features.jsonl` 은 k1a 실행이 쓴 라이브러리(`runs/k7-1+k1a/stage1-features/proposals.jsonl`,
원래 출처 `docs/artifacts/a6000-rules/features/pool_current.jsonl` · `pool_K1.jsonl`)의 시간 피처
16개다. **바꾼 것은 설명(`rationale`)뿐이다.** 앞으로 이 피처를 프롬프트에 넣을 때는 이 판을 쓴다.

```
지운 것   측정 결과 · a6000 관찰 ("2단 128x128 이 3단보다 약 4.5% 느리다", "6 MB L2 에서
          작은 타일이 25-30% 잃는다", train Spearman, 형상별 수치)
          개발 메모 "Pool note: ..." (서로게이트 수치, fold 적합 관찰, 쓰는 법 권고)
          다른 규칙 · 피처 이름 (J1e, K1, R1/R2, splitk_gain_log, roof_time_hraster ...)
          식을 고른 이유가 측정이었다는 말 ("측정한 split-K 곡선에 더 잘 맞았다")
          imported_from (a6000-rules 경로) — 출처는 이 파일에 적는다
남긴 것   무엇을 재는지, 어떤 효과가 어느 쪽으로 움직이는지, 단위
코드      그대로. 하나만 docstring 을 고쳤다 — log_roofline_time_mraster 가 라이브러리에
          없는 피처(roof_time_hraster)를 가리키고 있었다. 값은 바뀌지 않는다
          (a6000 · 5090 각 6형상 x 400 config x 16축 = 38,400 값, 최대 차이 0)
```

⚠️ **원본 파일은 지우지 않는다** — D-189 · D-191 이 실제로 쓴 기록이다.
⚠️ 원본 설명은 k1a 실행(16실행, 4 GPU)의 프롬프트에 그대로 들어갔다 — LLM 호출 1,440번
(RuleEditor 1,148 · Analyst 205 · FeatureWriter 87). D-191 정정 2 를 볼 것.

다시 만들기: 이 판은 스크립트 한 번으로 만들었다 — 설명 문장은 사람이 썼으므로 원본에서 자동으로
다시 생기지 않는다. 바꿀 때는 이 파일을 직접 고치고 값 비교를 다시 한다.
