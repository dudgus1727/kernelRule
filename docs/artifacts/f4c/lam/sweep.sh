#!/bin/bash
# D-197 — f4c 최종 규칙 (네 GPU) 을 같은 절차로 lam 0.5 · 0.25 · 0.1 로 재적합. LLM 0 · GPU 0.
#   GPU 마다 한 줄 (lam 은 차례로). 줄마다 (대상 표, 라이브러리 GPU) 가 달라 캐시 키가 겹치지 않는다
#   (D-196 의 같은 칸 실행이 캐시를 이미 만들었다 — 읽기만 한다).
# 사용: LOG=<로그 디렉터리> sweep.sh
set -u
cd "$(dirname "$0")/../../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=${LOG:-/tmp/f4c-lam-logs}
D=docs/artifacts/f4c
mkdir -p "$LOG"
for g in a6000 5090 4090 h100; do
  if [ $g = a6000 ]; then spec=$D/rules/f4c_evolved_s0.json; else spec=$D/$g/rules/f4c_evolved_s0.json; fi
  (
    for lam in 0.5 0.25 0.1; do
      python3 -m experiments.a6000_probe --rule $spec --prefix f4c --gpu $g --workers 4 \
        --cap $lam --out $D/lam/${g}_cap${lam}.json > "$LOG/lam-$g-$lam.log" 2>&1
      echo "[$g lam $lam] exit $? $(date +%T)"
      grep -E "★" "$LOG/lam-$g-$lam.log"
    done
  ) > "$LOG/lam-$g.out" 2>&1 &
done
wait
echo "D197 LAM DONE $(date +%T)"
