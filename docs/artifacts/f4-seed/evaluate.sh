#!/bin/bash
# D-193 — stage 2 가 끝난 뒤 시드룰을 미리 정한 기준으로 판정한다. LLM 0 · GPU 0.
#   구조      seed_structure (시간 본항이 있는가)
#   loop 적합  k1a_start      (기준선 표의 loop 적합 holdout 과 같은 자)
#   같은 절차  a6000_probe    (판정은 inner-CV, holdout 은 같이 적는다)
set -u
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=docs/artifacts/f4-seed
python3 -m experiments.seed_structure --prefix f4 --out $D/structure.json
python3 - <<'PY'
import json
from pathlib import Path
pf = {}
for f in range(4):
    c = json.loads(Path(f"runs/f4-a6000-f{f}/stage2-rule-writer/chosen.json")
                   .read_text())
    pf[str(f)] = {"code": c["code"], "w0": c["w0"], "extra_features": []}
Path("docs/artifacts/f4-seed/seedrule_f4_a6000.json").write_text(json.dumps(
    {"name": "f4 a6000 시드룰 (RuleWriter stage 2 chosen, per fold)",
     "per_fold": pf}, ensure_ascii=False, indent=1))
PY
python3 -m experiments.k1a_start --prefix f4 --gpu a6000 --out $D/loopfit.json
python3 -m experiments.a6000_probe --rule $D/seedrule_f4_a6000.json --prefix f4 \
  --gpu a6000 --workers 4 --out $D/refit.json
