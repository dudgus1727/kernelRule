#!/bin/bash
# D-193 stage 3 — loop 의 최종 규칙을 미리 정한 기준으로 판정한다. LLM 0 · GPU 0.
#   관찰      seed_structure --source final (tm_cta_warps · split-K 항 · 가중치)
#   같은 절차  a6000_probe (판정은 inner-CV, holdout 은 같이 적는다)
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=docs/artifacts/f4-seed
python3 -m experiments.seed_structure --prefix f4 --source final --out $D/stage3-structure.json
# make-baselines writes s0, then stops at the missing s1 — expected (seed 0 only)
python3 -m experiments.a6000_probe --make-baselines $D/stage3-rules --prefix f4 --gpu a6000 || true
python3 -m experiments.a6000_probe --rule $D/stage3-rules/f4_evolved_s0.json --prefix f4 \
  --gpu a6000 --workers 4 --out $D/stage3-refit.json
