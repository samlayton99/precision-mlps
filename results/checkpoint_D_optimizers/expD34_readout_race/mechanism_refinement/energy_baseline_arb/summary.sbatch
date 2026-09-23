#!/bin/bash
set -euo pipefail
srun env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,shutil
from flint import arb,ctx
ctx.prec=100
root=Path('/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/evidence/energy_baseline_arb')
data=json.loads((root/'conditions.json').read_text()); assert data['complete'] and len(data['rows'])==18
assert all(r['norm_condition'] and r['loss_condition'] and r['eta_condition'] for r in data['rows'])
summary={}
for n in (128,512,1024):
    part=[r for r in data['rows'] if r['nref']==n]
    summary[n]=dict(cases=len(part),max_M0_upper=str(max(arb(r['M0_upper']).upper() for r in part)),max_L0_upper=str(max(arb(r['L0_upper']).upper() for r in part)),all_conditions_verified=True)
(root/'summary.json').write_text(json.dumps(dict(widths=summary,conditions_sha256=hashlib.sha256((root/'conditions.json').read_bytes()).hexdigest(),verification_job=1295,one_cpu_wall_seconds=103,scope='Only initial norm/loss/eta hypotheses of the separate generic energy theorem; all18 verified with outward rounding'),indent=2))
inputs=root/'inputs'; inputs.mkdir(exist_ok=False)
for name in data['input_sha256']:
    shutil.copy2(name,inputs/Path(name).name)
print((root/'summary.json').read_text())
PY
