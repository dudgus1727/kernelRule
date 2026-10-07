#!/bin/bash
# D-199 — f4p 최종 규칙을 미리 정한 기준으로 판정한다. LLM 0 · GPU 0.
#   구조       seed_structure (시드룰 · 최종)
#   같은 절차  a6000_probe --cap 0.1 — train 에 한 번 맞추고 holdout 을 읽는다 (inner-CV 없음, D-199)
# 사용: evaluate.sh "<gpus>"     GPU 마다 한 줄씩 같이 돈다 (캐시 키가 GPU 마다 다르다)
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=docs/artifacts/f4p
for g in $1; do
  (
    mkdir -p $D/$g
    python3 -m experiments.seed_structure --prefix f4p --gpu $g --out $D/$g/seed-structure.json
    python3 -m experiments.seed_structure --prefix f4p --gpu $g --source final --out $D/$g/final-structure.json
    # make-baselines writes s0, then stops at the missing s1 — expected (seed 0 only)
    python3 -m experiments.a6000_probe --make-baselines $D/$g/rules --prefix f4p --gpu $g >/dev/null 2>&1 || true
    python3 -m experiments.a6000_probe --rule $D/$g/rules/f4p_evolved_s0.json --prefix f4p --gpu $g \
      --workers 4 --cap 0.1 --out $D/$g/refit_cap0.1.json
  ) > "${LOG:-/tmp}/f4p-eval-$g.log" 2>&1 &
done
wait
echo "f4p evaluate done $(date +%T)"
