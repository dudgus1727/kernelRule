#!/bin/bash
# D-193 — 조건 F4 (known4 + 시간 피처 15) + 실제 범위 + 시간 틀 예시로 시드룰을 만든다.
# a6000 nkband k=4, stage 1 (FeatureWriter, 새로) · stage 2 (RuleWriter 10). 태그 f4-a6000-f{0..3}.
# 사용: campaign.sh <stage...>   예) campaign.sh 1 2
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=${LOG:-/tmp/f4-logs}
mkdir -p "$LOG"
COMMON=(F4 --split-design nkband --folds 4 --n-seeds 4 --rounds 12
        --n-rule-writer 10 --n-features 20 --seed 0 --model gpt-5.6-luna
        --bundle datasets/rtx-a6000-sm_86-c63710df --env-hash c63710df
        --workers 2 --time-gate --observed-ranges
        --size-guidance role/_size_time.md)
for st in "$@"; do
  for f in ${FOLDS:-0 1 2 3}; do
    python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag f4-a6000-f$f \
      --fold $f --stage "$st" > "$LOG/s$st-f$f.log" 2>&1 &
    sleep 20
  done
  wait
  echo "f4 stage $st done $(date +%T)"
done
