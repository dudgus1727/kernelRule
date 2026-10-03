#!/bin/bash
# D-190 a6000 캠페인 — c2 와 같은 설정 (F2 · nkband k=4 · 4 seed · 12 라운드 ·
# RuleWriter 10 · gpt-5.6-luna). 다른 점: stage 1 을 실제로 돌린다 —
# k7-1 을 이어받아(--extend-from) fold 의 train 에서 새 축 8개.
# 사용: campaign.sh <stage...>   예) campaign.sh 1 2 3
set -u
cd /home/piai/workspace/kernelRule
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=${LOG:-/tmp/d190-logs}
mkdir -p "$LOG"
COMMON=(F2 --split-design nkband --folds 4 --n-seeds 4 --rounds 12
        --n-rule-writer 10 --n-features 8 --seed 0 --model gpt-5.6-luna
        --bundle datasets/rtx-a6000-sm_86-c63710df --env-hash c63710df
        --extend-from runs/k7-1/stage1-features --workers 2)
for st in "$@"; do
  if [ "$st" = 3 ]; then
    for s in ${SEEDS:-0 1 2 3}; do
      for f in ${FOLDS:-0 1 2 3}; do
        python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag d190-a6000-f$f \
          --fold $f --stage 3 --seed-index $s > "$LOG/s3-f$f-s$s.log" 2>&1 &
        sleep 20
      done
      wait
      echo "wave s$s done $(date +%T)"
    done
  else
    for f in ${FOLDS:-0 1 2 3}; do
      python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag d190-a6000-f$f \
        --fold $f --stage "$st" > "$LOG/s$st-f$f.log" 2>&1 &
      sleep 20
    done
    wait
    echo "stage $st done $(date +%T)"
  fi
done
