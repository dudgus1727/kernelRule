# experiments/

**전부 저장소 뿌리에서 실행한다.** `runs/` 와 `datasets/` 를 상대 경로로 읽는다.

각 스크립트는 결과를 `docs/artifacts/` 의 한 문서로 남긴다. **문서가 없는
스크립트는 지운다** — 커밋 메시지에만 있는 결과는 새 세션이 못 찾는다.

| 스크립트 | LLM | 만든 artifact | 상태 |
|---|---|---|---|
| `rerun.py` | ✅ | `rerun-preregistration.md` | ★ **재실행 대표값 러너** — 기준이 `PREREG` 에 |
| `f1_pipeline.py` | ✅ | `f1-guided.md` · `f1k-preregistration.md` · (§30.9) | ★ **F0~F3 대표값 러너** — `--dry-run` 확인 완료 |
| `seed_selection.py` | ✅ | `luna-baseline.md` · `conclusion.md` | F3 전용 (구 대표값) |
| `export_rules.py` | — | `docs/artifacts/rules/` | **규칙 내보내기** |
| `verify_rules.py` | — | (검증만) | ★ **검증 경로** |

| `new_axes.py` | ✅ | `new-axes.md` | ⚠️ 원본 삭제 |
| `feature_writer.py` | ✅ | `feature-writer-f1.md` | 유효 — luna 재측정 완료 |
| `rule_writer_gate.py` | ✅ | `rule-writer-gate.md` | ⚠️ 원본 삭제 (mini 실행만 남음) |

| `score_new_axes.py` | — | `new-axes.md` 채점 | ⚠️ 원본 삭제 |
| `rescore_canonical.py` | — | `conclusion.md` 의 최종 채점 표 | 유효 |
| `regime_transfer.py` | — | `structure-transfer.md` · `regime-diagnosis.md` | 유효 |
| `proxy_dispatch.py` | — | `regime-diagnosis.md` · `glossary.md` | 유효 |
| `regime_count.py` | — | `regime-count.md` | ⚠️ evolved 팔 원본 삭제 |
| `seed_spread.py` | — | `conclusion.md` 의 시드 폭 | ⚠️ 원본 삭제 — 목록을 채워야 돈다 |
| `selection_spread.py` | — | `decisions.md` D-40 / D-42 | ⚠️ 원본 삭제 — 목록을 채워야 돈다 |
| `fitter_sweep.py` | — | `fitter-sweep.md` · D-55 | 유효 — 커밋된 규칙만 읽는다 |
| `fitter_polish.py` | — | `fitter-sweep.md` · D-55 | 유효 — 커밋된 규칙만 읽는다 |
| `fitter_movement.py` | — | D-56 · D-57 | ★ **적합기 통과 조건** — 도달률 90% |
| `polish_ranking.py` | — | D-57 | ★ **재실행 판정** — 순위가 바뀌는가 |

## ★ 숫자를 검증하는 법

```bash
python3 experiments/verify_rules.py
```

`runs/` 는 `.gitignore` 라 저장소에 없다. 이 명령은 그것을 **안 읽고**
`docs/artifacts/rules/*.py` 의 규칙과 적합된 가중치만으로 구조 홀드아웃을
다시 계산해 `index.json` 과 대조한다. 어긋나면 실패한다.

```
LLM 실행    재현 불가 (난수 통제 안 됨 — §24.4b)
채점·재채점  ★ 결정론적. 몇 초
```

문서의 성능 숫자는 이 경로로 검증된다. 새 실행을 만들면
`export_rules.py` 를 다시 돌려라 — 안 돌리면 문서가 조용히 낡는다
(테스트가 `rules/*.py` 와 `index.json` 의 짝은 본다).

## 실행 조건은 코드가 아니라 설정에 있다

모델·엔드포인트·추론 강도는 **`kernelrule/agents/openai_client.py` 의
`DEFAULT_MODEL` 과 `LLMConfig`** 한 곳에서 온다. 스크립트가 직접 박으면
테스트가 실패한다 (D-45).

실험 조건(피처 설명 유무 등)은 `LLMConfig.feature_detail` 처럼 **플래그로**
둔다 — 코드를 되돌렸다 돌렸다 하면 어느 실행이 어느 조건이었는지 알 수 없다.
모든 실행은 `runs/{id}/config.json` 에 설정을 남긴다 (D-51).

## 채점은 `core/canonical.py` 하나로

`canonical_score(code, w0, table=, matrix=, splits=)` 는 **루프의 `SplitSet`
을 받아야만** 돌고, 형상을 따로 뽑는 경로가 없다. 그러지 않으면 홀드아웃이
학습 형상과 겹친다 — 실제로 19 중 11이 겹쳤다 (D-36).

## ★ F0~F3 를 돌리는 법

```bash
python3 experiments/f1_pipeline.py F1 --dry-run      # 배관 확인, LLM 0회
python3 experiments/f1_pipeline.py F1                # 실제
python3 experiments/f1_pipeline.py F1 --stage 2      # 1단계 산출물 재사용
```

조건이 정하는 것은 **어느 레지스트리가 세 단계 전부에 들어가는가** 하나다.

```
F3  사람 24개 + human_guided 씨앗   1단계를 건너뛴다 (조건이 그렇다)
F2  기초 5개 + FeatureWriter
F1  원시 값만 -> FeatureWriter -> RuleWriter 씨앗
F0  피처 없음
```

돌린 뒤 **가장 먼저 볼 것:**

```bash
jq .human_features_present runs/f1pipe-F1-*/config.json   # F0/F1 이면 [] 여야 한다
jq .physics_coverage runs/f1pipe-F1-*/stage1-features/summary.json
```
