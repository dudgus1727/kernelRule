# 개선안 2번 — FeatureWriter 의 시간 축을 train 만으로 채점해 돌려준다 (2026-10-04)

`kernelrule/features/time_gate.py`. 단위에 "time" 이 들어간 config 단위 축만: 형상 안에서
log(시간) 과의 순위 상관이 0.3 이상인가, 그리고 split_k · stages · 래스터 폭 중 하나만
바꿔 가며 볼 때 시간이 가운데에서 최소인데 축은 끝에서 최소인 형상이 절반을 넘지 않는가.
train 형상만 읽는다. stage 1 은 거부 사유를 숫자와 함께 모델에게 돌려줘 3번까지 다시
만들게 하고, 루프 안에서는 거부만 한다. 조건 `time_gate` (예전 실행은 꺼짐).

보정 (`calib.json`, fold 0 train): 손으로 만든 시간 축 5개 중 4개 통과, d190 의
FeatureWriter 시간 축 5개 중 4개 걸림. 기준값(0.3 · 1/4)은 이 10개 축을 본 **뒤에** 정했다.

## tgate 실행 — d190 과 같은 조건에 `--time-gate` 하나만

`campaign.sh 1 2 3` (태그 `tgate-a6000-f{0..3}`, seed 0). 실행 전에 정한 판정:

```
1  stage 1 의 시간 축이 채점을 통과한 채로 남는가 (fold 마다 1개 이상)
2  그 축들의 품질: calib 와 같은 진단 (순위 상관, split_k 끝 놓침) — d190 의 축과 견준다
3  루프 결과: 같은 절차 재적합의 inner-CV gm 이 d190 seed 0 (1.0746) 보다 낮으면 '나아짐'
   (판정은 inner-CV. a6000 holdout 은 이미 여러 번 썼으므로 적기만 한다)
⚠️ seed 하나 대 seed 하나 — d190 은 seed 사이에 pooled holdout 이 1.080~1.131 로 흔들렸다
```
