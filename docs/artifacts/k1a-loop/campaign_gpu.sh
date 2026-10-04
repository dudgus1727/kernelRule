#!/bin/bash
# 개선안 1번 진단을 다른 GPU 에서 — 같은 라이브러리(k7-1 + 시간 모델 16축) · 같은 씨앗(K1a).
# 사용: campaign_gpu.sh 5090 4090 h100   (GPU 마다 fold 4개 x seed 0, GPU 는 차례로)
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=${LOG:-/tmp/k1a-logs}
mkdir -p "$LOG"
declare -A BUNDLE=([a6000]=datasets/rtx-a6000-sm_86-c63710df [5090]=datasets/rtx-5090-sm_120-5bb6f403
                   [4090]=datasets/rtx-4090-sm_89-ad95d455 [h100]=datasets/h100-nvl-sm_90-63684546)
declare -A HASH=([a6000]=c63710df [5090]=5bb6f403 [4090]=ad95d455 [h100]=63684546)
for g in "$@"; do
  COMMON=(F2 --split-design nkband --folds 4 --n-seeds 4 --rounds 12
          --n-rule-writer 10 --seed 0 --model gpt-5.6-luna
          --bundle "${BUNDLE[$g]}" --env-hash "${HASH[$g]}" --workers 2)
  for f in 0 1 2 3; do
    python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag k1a-$g-f$f --fold $f \
      --import-featwriter runs/k7-1+k1a/stage1-features > "$LOG/$g-s1-f$f.log" 2>&1
  done
  for f in 0 1 2 3; do
    python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag k1a-$g-f$f --fold $f \
      --seed-from runs/k1a-seed --stage 3 --seed-index 0 > "$LOG/$g-s3-f$f-s0.log" 2>&1 &
    sleep 20
  done
  wait
  echo "k1a $g done $(date +%T)"
done
