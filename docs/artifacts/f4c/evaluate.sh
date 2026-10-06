#!/bin/bash
# D-195 단계 2 — loop 최종 규칙을 미리 정한 기준으로 판정한다. LLM 0 · GPU 0.
#   구조       seed_structure (시드룰 · 최종)
#   같은 절차  a6000_probe --cap 1 (주, 판정은 inner-CV) · 상한 없이 (같이 적는다)
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=docs/artifacts/f4c
python3 -m experiments.seed_structure --prefix f4c --out $D/seed-structure.json
python3 -m experiments.seed_structure --prefix f4c --source final --out $D/final-structure.json
# make-baselines writes s0, then stops at the missing s1 — expected (seed 0 only)
python3 -m experiments.a6000_probe --make-baselines $D/rules --prefix f4c --gpu a6000 || true
python3 -m experiments.a6000_probe --rule $D/rules/f4c_evolved_s0.json --prefix f4c \
  --gpu a6000 --workers 4 --cap 1 --out $D/refit_cap1.json
python3 -m experiments.a6000_probe --rule $D/rules/f4c_evolved_s0.json --prefix f4c \
  --gpu a6000 --workers 4 --out $D/refit_nocap.json
