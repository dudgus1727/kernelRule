#!/bin/bash
# 시드룰 (RuleWriter 가 stage 2 에서 고른 seed 규칙) 을 K1a 와 같은 두 자로 잰다. LLM 0 · GPU 0.
#   loop 적합     k1a_start        — 루프의 seed() 와 같은 적합 (출발점과 같은 자)
#   같은 절차     a6000_probe      — CMA 2x2000, inner-CV + holdout (K1a 결과와 같은 자)
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=docs/artifacts/seedrule-baseline
for x in d190:a6000 tgate:a6000 c2:a6000 c2:5090 c2:4090 c2:h100; do
  pre=${x%%:*}; g=${x##*:}
  python3 -m experiments.k1a_start --prefix $pre --gpu $g --out $D/results/${pre}_${g}.loopfit.json
  python3 -m experiments.a6000_probe --rule $D/rules/seedrule_${pre}_${g}.json --prefix $pre --gpu $g \
    --workers 4 --out $D/results/${pre}_${g}.refit.json
done
