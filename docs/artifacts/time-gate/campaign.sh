#!/bin/bash
# 개선안 2번 — FeatureWriter 의 시간 축을 train 만으로 채점해 돌려준다 (--time-gate).
# d190 과 같은 조건 (F2 · nkband k=4 · stage 1 = k7-1 + 새 축 8개 · RuleWriter 10 · 12 라운드)
# 에 --time-gate 하나만 더한다. 태그 tgate-a6000-f{0..3}.
# 사용: campaign.sh <stage...>   예) campaign.sh 1   /   SEEDS=0 campaign.sh 2 3
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=${LOG:-/tmp/tgate-logs}
mkdir -p "$LOG"
COMMON=(F2 --split-design nkband --folds 4 --n-seeds 4 --rounds 12
        --n-rule-writer 10 --n-features 8 --seed 0 --model gpt-5.6-luna
        --bundle datasets/rtx-a6000-sm_86-c63710df --env-hash c63710df
        --extend-from runs/k7-1/stage1-features --workers 2 --time-gate)
for st in "$@"; do
  if [ "$st" = 3 ]; then
    for s in ${SEEDS:-0}; do
      for f in 0 1 2 3; do
        python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag tgate-a6000-f$f \
          --fold $f --stage 3 --seed-index $s > "$LOG/s3-f$f-s$s.log" 2>&1 &
        sleep 20
      done
      wait
      echo "tgate wave s$s done $(date +%T)"
    done
  else
    for f in 0 1 2 3; do
      python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag tgate-a6000-f$f \
        --fold $f --stage "$st" > "$LOG/s$st-f$f.log" 2>&1 &
      sleep 20
    done
    wait
    echo "tgate stage $st done $(date +%T)"
  fi
done
