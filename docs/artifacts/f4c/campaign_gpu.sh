#!/bin/bash
# D-196 — a6000 단계 2 (D-195) 와 같은 조건을 다른 GPU 에서. stage 1 은 새로 (FeatureWriter 20).
# 조건: F4 (20축, has_spill 포함) · observed_ranges · time_gate · term_cap 1 · _size_time · seed 0.
# 태그 f4c-<gpu>-f{0..3}.
# 사용: campaign_gpu.sh "<gpus>" <stage...>
#   예) campaign_gpu.sh "5090 4090 h100" 1 2     (GPU 들을 같이, fold 4개씩)
#       WORKERS=1 campaign_gpu.sh "5090 4090" 3  (stage 3 는 GPU 몇 개씩 묶어서)
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=${LOG:-/tmp/f4c-gpu-logs}
mkdir -p "$LOG"
declare -A BUNDLE=([a6000]=datasets/rtx-a6000-sm_86-c63710df [5090]=datasets/rtx-5090-sm_120-5bb6f403
                   [4090]=datasets/rtx-4090-sm_89-ad95d455 [h100]=datasets/h100-nvl-sm_90-63684546)
declare -A HASH=([a6000]=c63710df [5090]=5bb6f403 [4090]=ad95d455 [h100]=63684546)
GPUS=$1; shift
for st in "$@"; do
  for g in $GPUS; do
    COMMON=(F4 --split-design nkband --folds 4 --n-seeds 4 --rounds 12
            --n-rule-writer 10 --n-features 20 --seed 0 --model gpt-5.6-luna
            --bundle "${BUNDLE[$g]}" --env-hash "${HASH[$g]}" --workers ${WORKERS:-2}
            --time-gate --observed-ranges --term-cap 1 --size-guidance role/_size_time.md)
    for f in 0 1 2 3; do
      if [ "$st" = 3 ]; then
        python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag f4c-$g-f$f --fold $f \
          --stage 3 --seed-index 0 > "$LOG/$g-s3-f$f-s0.log" 2>&1 &
      else
        python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag f4c-$g-f$f --fold $f \
          --stage "$st" > "$LOG/$g-s$st-f$f.log" 2>&1 &
      fi
      sleep 15
    done
  done
  wait
  echo "f4c [$GPUS] stage $st done $(date +%T)"
done
