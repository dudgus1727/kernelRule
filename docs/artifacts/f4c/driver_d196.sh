#!/bin/bash
# D-196 — stage 1 · 2 (campaign_gpu.sh "5090 4090 h100" 1 2) 가 끝난 뒤의 모든 단계. LLM 은 stage 3 만.
#   stage 3: 5090 · 4090 함께 (workers 1) -> H100 (스케줄, 조건 아님)
#   1단계 판정: GPU 마다 구조 · 같은 절차 재적합 (상한 1 / 상한 없음)
#   2단계 전이: 출발 4 x 대상 3 = 12 쌍, (a) --no-fit · (b) --cap 1 재적합
# 사용: driver_d196.sh <stage-1·2 의 PID (없으면 0)>
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WAIT=${1:-0}
while [ "$WAIT" != 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 30; done
echo "[stage 3 5090+4090] $(date +%T)"
WORKERS=1 bash docs/artifacts/f4c/campaign_gpu.sh "5090 4090" 3
echo "[stage 3 h100] $(date +%T)"
WORKERS=2 bash docs/artifacts/f4c/campaign_gpu.sh "h100" 3
D=docs/artifacts/f4c
for g in 5090 4090 h100; do
  echo "[native $g] $(date +%T)"
  mkdir -p $D/$g
  python3 -m experiments.seed_structure --prefix f4c --gpu $g --out $D/$g/seed-structure.json
  python3 -m experiments.seed_structure --prefix f4c --gpu $g --source final --out $D/$g/final-structure.json
  python3 -m experiments.a6000_probe --make-baselines $D/$g/rules --prefix f4c --gpu $g >/dev/null 2>&1 || true
  python3 -m experiments.a6000_probe --rule $D/$g/rules/f4c_evolved_s0.json --prefix f4c --gpu $g \
    --workers 4 --cap 1 --out $D/$g/refit_cap1.json 2>&1 | grep -E "★"
  python3 -m experiments.a6000_probe --rule $D/$g/rules/f4c_evolved_s0.json --prefix f4c --gpu $g \
    --workers 4 --out $D/$g/refit_nocap.json 2>&1 | grep -E "★"
done
mkdir -p $D/transfer
for s in a6000 5090 4090 h100; do
  if [ $s = a6000 ]; then spec=$D/rules/f4c_evolved_s0.json; else spec=$D/$s/rules/f4c_evolved_s0.json; fi
  for t in a6000 5090 4090 h100; do
    [ $s = $t ] && continue
    echo "[transfer $s -> $t] $(date +%T)"
    python3 -m experiments.a6000_probe --rule $spec --prefix f4c --lib-gpu $s --gpu $t \
      --workers 4 --no-fit --out $D/transfer/${s}_to_${t}.nofit.json 2>&1 | grep -E "★"
    python3 -m experiments.a6000_probe --rule $spec --prefix f4c --lib-gpu $s --gpu $t \
      --workers 4 --cap 1 --out $D/transfer/${s}_to_${t}.refit_cap1.json 2>&1 | grep -E "★"
  done
done
echo "D196 ALL DONE $(date +%T)"
