#!/bin/bash
# D-196 — driver_d196.sh 의 stage 3 다음 단계를 **순서만 바꿔** 돌린다 (2026-10-06 16시).
#   명령 · 옵션 · 출력 경로는 driver_d196.sh 와 같다. 바뀐 것은 언제 무엇을 같이 돌리느냐뿐.
#   왜: H100 stage 3 fold 0 이 라운드마다 25~30 분이라, 그 동안 CPU 48 개 중 3 개만 돌았다.
#       H100 과 무관한 칸(5090 · 4090 판정, a6000 · 5090 · 4090 사이 전이 6 쌍)을 먼저 돌린다.
#   동시에 도는 칸은 (대상 표, 라이브러리 GPU) 가 서로 다르다 — 특징 행렬 캐시 키가 겹치지 않는다
#   (`FeatureMatrix._save_cache` 는 원자적으로 쓰지 않는다). 한 칸의 두 실행은 차례로.
# 사용: driver_d196b.sh <H100 stage 3 campaign_gpu.sh 의 PID>
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
H100=${1:?H100 campaign PID}
LOG=${LOG:-/tmp/f4c-gpu-logs}
D=docs/artifacts/f4c
mkdir -p "$LOG" $D/transfer

prep() {   # 구조 · 규칙 파일 (driver_d196.sh 의 [native g] 앞부분)
  g=$1
  echo "[prep $g] $(date +%T)"
  mkdir -p $D/$g
  python3 -m experiments.seed_structure --prefix f4c --gpu $g --out $D/$g/seed-structure.json
  python3 -m experiments.seed_structure --prefix f4c --gpu $g --source final --out $D/$g/final-structure.json
  python3 -m experiments.a6000_probe --make-baselines $D/$g/rules --prefix f4c --gpu $g >/dev/null 2>&1 || true
}

probe() {  # 실행 하나 — 전체 출력은 $LOG/probe-<이름>.log, ★ 줄만 여기에
  name=$1; shift
  python3 -m experiments.a6000_probe "$@" > "$LOG/probe-$name.log" 2>&1
  echo "  [$name] exit $? $(date +%T)"
  grep -E "★" "$LOG/probe-$name.log"
}

cell() {   # (출발 s, 대상 t) 한 칸. s = t 이면 1단계 판정, 아니면 2단계 전이
  s=$1; t=$2
  if [ $s = $t ]; then
    echo "[native $t] $(date +%T)"
    probe native-$t-cap1 --rule $D/$t/rules/f4c_evolved_s0.json --prefix f4c --gpu $t \
      --workers 4 --cap 1 --out $D/$t/refit_cap1.json
    probe native-$t-nocap --rule $D/$t/rules/f4c_evolved_s0.json --prefix f4c --gpu $t \
      --workers 4 --out $D/$t/refit_nocap.json
  else
    if [ $s = a6000 ]; then spec=$D/rules/f4c_evolved_s0.json; else spec=$D/$s/rules/f4c_evolved_s0.json; fi
    echo "[transfer $s -> $t] $(date +%T)"
    probe ${s}_to_${t}.nofit --rule $spec --prefix f4c --lib-gpu $s --gpu $t \
      --workers 4 --no-fit --out $D/transfer/${s}_to_${t}.nofit.json
    probe ${s}_to_${t}.refit_cap1 --rule $spec --prefix f4c --lib-gpu $s --gpu $t \
      --workers 4 --cap 1 --out $D/transfer/${s}_to_${t}.refit_cap1.json
  fi
}

# A — H100 과 무관한 8 칸, 대상 GPU 마다 한 줄
prep 5090
prep 4090
( cell 5090 5090; cell a6000 5090; cell 4090 5090 ) > "$LOG/d196b-to5090.out" 2>&1 &
( cell 4090 4090; cell a6000 4090; cell 5090 4090 ) > "$LOG/d196b-to4090.out" 2>&1 &
( cell 5090 a6000; cell 4090 a6000 ) > "$LOG/d196b-toa6000.out" 2>&1 &
echo "[A started] $(date +%T)"

# B — H100 stage 3 가 끝나면, H100 이 낀 7 칸 (판정 1 + 전이 6)
while kill -0 "$H100" 2>/dev/null; do sleep 30; done
echo "[H100 stage 3 ended] $(date +%T)"
prep h100
( cell h100 h100; cell a6000 h100 ) > "$LOG/d196b-toh100-1.out" 2>&1 &
( cell 5090 h100; cell 4090 h100 ) > "$LOG/d196b-toh100-2.out" 2>&1 &
( cell h100 5090 ) > "$LOG/d196b-h100to5090.out" 2>&1 &
( cell h100 4090 ) > "$LOG/d196b-h100to4090.out" 2>&1 &
( cell h100 a6000 ) > "$LOG/d196b-h100toa6000.out" 2>&1 &
echo "[B started] $(date +%T)"
wait
echo "D196 ALL DONE $(date +%T)"
