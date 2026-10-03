#!/bin/bash
# 개선안 1번 진단 — 검증된 시간 모델 피처를 라이브러리에 넣고, K1a 에서 루프를 시작한다.
# d190 과 같은 루프 코드 · 설정 (F2 · nkband k=4 · 12 라운드 · gpt-5.6-luna).
# 다른 점: stage 1 = k7-1 + 시간 모델 16축 (runs/k7-1+k1a), stage 2 = K1a (runs/k1a-seed).
# 사용: campaign.sh   (fold 4개 x seed 0)
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=${LOG:-/tmp/k1a-logs}
mkdir -p "$LOG"
COMMON=(F2 --split-design nkband --folds 4 --n-seeds 4 --rounds 12
        --n-rule-writer 10 --seed 0 --model gpt-5.6-luna
        --bundle datasets/rtx-a6000-sm_86-c63710df --env-hash c63710df
        --workers 2)
for f in ${FOLDS:-0 1 2 3}; do
  python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag k1a-a6000-f$f --fold $f \
    --import-featwriter runs/k7-1+k1a/stage1-features > "$LOG/s1-f$f.log" 2>&1
done
for f in ${FOLDS:-0 1 2 3}; do
  python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag k1a-a6000-f$f --fold $f \
    --seed-from runs/k1a-seed --stage 3 --seed-index 0 > "$LOG/s3-f$f-s0.log" 2>&1 &
  sleep 20
done
wait
echo "k1a done $(date +%T)"
