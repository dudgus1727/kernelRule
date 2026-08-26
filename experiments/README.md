# experiments/

**전부 저장소 뿌리에서 실행한다.** `runs/` 와 `datasets/` 를 상대 경로로 읽는다.

각 스크립트는 결과를 `docs/artifacts/` 의 한 문서로 남긴다. **문서가 없는
스크립트는 지운다** — 커밋 메시지에만 있는 결과는 새 세션이 못 찾는다.

| 스크립트 | LLM | 만든 artifact | 상태 |
|---|---|---|---|
| `seed_selection.py` | ✅ | `luna-baseline.md` · `conclusion.md` | **정본 러너** |

| `new_axes.py` | ✅ | `new-axes.md` | ⚠️ 원본 삭제 |
| `feature_writer.py` | ✅ | `feature-writer-f1.md` | ⚠️ 원본 삭제 |
| `architect_gate.py` | ✅ | `architect-gate.md` | ⚠️ 원본 삭제 (mini 실행만 남음) |

| `score_new_axes.py` | — | `new-axes.md` 채점 | ⚠️ 원본 삭제 |
| `rescore_canonical.py` | — | `conclusion.md` 의 정준 표 | 유효 |
| `regime_transfer.py` | — | `structure-transfer.md` · `regime-diagnosis.md` | 유효 |
| `proxy_dispatch.py` | — | `regime-diagnosis.md` · `glossary.md` | 유효 |
| `regime_count.py` | — | `regime-count.md` | ⚠️ evolved 팔 원본 삭제 |
| `seed_spread.py` | — | `conclusion.md` 의 시드 폭 | ⚠️ 원본 삭제 — 목록을 채워야 돈다 |
| `selection_spread.py` | — | `decisions.md` D-40 / D-42 | ⚠️ 원본 삭제 — 목록을 채워야 돈다 |

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
