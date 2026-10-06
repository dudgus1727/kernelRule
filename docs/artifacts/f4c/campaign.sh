#!/bin/bash
# D-195 단계 2 — F4 (has_spill 포함, D-194) + 보정 항 상한 lam=1 로 stage 2 · 3 을 다시.
# stage 1 은 D-193 의 라이브러리를 그대로 가져온다 (runs/f4-a6000-f*/stage1-features).
# 태그 f4c-a6000-f{0..3}. 사용: campaign.sh import 2   /   SEEDS=0 campaign.sh 3
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=${LOG:-/tmp/f4c-logs}
mkdir -p "$LOG"
COMMON=(F4 --split-design nkband --folds 4 --n-seeds 4 --rounds 12
        --n-rule-writer 10 --n-features 20 --seed 0 --model gpt-5.6-luna
        --bundle datasets/rtx-a6000-sm_86-c63710df --env-hash c63710df
        --workers 2 --time-gate --observed-ranges --term-cap 1
        --size-guidance role/_size_time.md)
for st in "$@"; do
  if [ "$st" = import ]; then
    for f in ${FOLDS:-0 1 2 3}; do
      python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag f4c-a6000-f$f --fold $f \
        --import-featwriter runs/f4-a6000-f$f/stage1-features > "$LOG/import-f$f.log" 2>&1
    done
    echo "f4c import done $(date +%T)"
  elif [ "$st" = 3 ]; then
    for s in ${SEEDS:-0}; do
      for f in ${FOLDS:-0 1 2 3}; do
        python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag f4c-a6000-f$f \
          --fold $f --stage 3 --seed-index $s > "$LOG/s3-f$f-s$s.log" 2>&1 &
        sleep 20
      done
      wait
      echo "f4c stage 3 wave s$s done $(date +%T)"
    done
  else
    for f in ${FOLDS:-0 1 2 3}; do
      python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag f4c-a6000-f$f \
        --fold $f --stage "$st" > "$LOG/s$st-f$f.log" 2>&1 &
      sleep 20
    done
    wait
    echo "f4c stage $st done $(date +%T)"
  fi
done
