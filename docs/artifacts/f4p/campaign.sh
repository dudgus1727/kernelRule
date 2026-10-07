#!/bin/bash
# D-199 — F4 + 상한 lam 0.1 + 죽은 항 정리 (기본 켬) 로 네 GPU 를 다시. 태그 f4p-<gpu>-f{0..3}, seed 0.
#   stage 1  runs/f4c-<gpu>-f<f> 의 라이브러리를 그대로 가져온다 (--import-featwriter)
#   stage 2  runs/f4c-<gpu>-f<f> 의 RuleWriter 후보 10 개를 이 조건으로 다시 채점해 시드룰을 고른다
#            (--import-rule-writer, LLM 0) — 시드룰은 적합으로 고르고, 바뀐 것이 적합이다
#   stage 3  12 라운드 (LLM)
# 사용: campaign.sh "<gpus>" <step...>     step = import | 2 | 3
#   예) campaign.sh "a6000 5090 4090 h100" import 2     (빠르다)
#       WORKERS=1 campaign.sh "a6000 5090 4090 h100" 3  (16 실행을 같이)
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=${LOG:-/tmp/f4p-logs}
mkdir -p "$LOG"
declare -A BUNDLE=([a6000]=datasets/rtx-a6000-sm_86-c63710df [5090]=datasets/rtx-5090-sm_120-5bb6f403
                   [4090]=datasets/rtx-4090-sm_89-ad95d455 [h100]=datasets/h100-nvl-sm_90-63684546)
declare -A HASH=([a6000]=c63710df [5090]=5bb6f403 [4090]=ad95d455 [h100]=63684546)
GPUS=$1; shift
for st in "$@"; do
  for g in $GPUS; do
    COMMON=(F4 --split-design nkband --folds 4 --n-seeds 4 --rounds 12
            --n-rule-writer 10 --n-features 20 --seed 0 --model gpt-5.6-luna
            --bundle "${BUNDLE[$g]}" --env-hash "${HASH[$g]}" --workers ${WORKERS:-1}
            --time-gate --observed-ranges --term-cap 0.1 --prune-dead
            --size-guidance role/_size_time.md)
    for f in 0 1 2 3; do
      case $st in
        import)
          python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag f4p-$g-f$f --fold $f \
            --import-featwriter runs/f4c-$g-f$f/stage1-features > "$LOG/$g-import-f$f.log" 2>&1 ;;
        2)
          python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag f4p-$g-f$f --fold $f \
            --stage 2 --import-rule-writer runs/f4c-$g-f$f > "$LOG/$g-s2-f$f.log" 2>&1 & ;;
        3)
          python3 experiments/f1_pipeline.py "${COMMON[@]}" --tag f4p-$g-f$f --fold $f \
            --stage 3 --seed-index 0 > "$LOG/$g-s3-f$f-s0.log" 2>&1 &
          sleep 30 ;;
      esac
    done
  done
  wait
  echo "f4p [$GPUS] step $st done $(date +%T)"
done
