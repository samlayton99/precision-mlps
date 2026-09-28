#!/bin/bash
set -euo pipefail
cd /workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/code
srun env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
import numpy as np
from flint import arb
import d34_arb_probe as cert
root=Path('/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/evidence')
source_hash=hashlib.sha256(Path(cert.__file__).read_bytes()).hexdigest()
assert source_hash=='8ce86d75c5bc57035a8402d00e2add13b4d4faf6444f1e07a5968d925094788d'
out=root/'energy_baseline_arb'; out.mkdir(exist_ok=False)
rows=[]; hashes={}
for n in (128,512,1024):
    source=root/'moving_tube_inputs'/f'N{n}_seed30_20k.npz'
    hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source) as data:
        pp=data['p'].copy(); x=data['x'].copy(); yy=data['y'].copy(); cases=json.loads(str(data['cases']))
    assert max(abs(x))<=1 and len(cases)==6
    for i,(p,y,case) in enumerate(zip(pp,yy,cases)):
        assert case['seed']==30 and case['start']==20000
        state=cert.point(p,x,y)
        M0=cert.upper(cert.norm(cert.exact(p[:-1])))
        L0=cert.upper(state['residual']**2/2)
        row=dict(case,input_row=i,M0_upper=str(M0),L0_upper=str(L0),residual_rms_upper=str(state['residual']),
            norm_condition=bool(M0<=arb(3)),loss_condition=bool(L0<=arb(1)),
            eta_condition=bool(arb(float(.002))<=arb(1)/490),point_seconds=state['seconds'])
        rows.append(row); print(json.dumps(row),flush=True)
        (out/'conditions.json').write_text(json.dumps(dict(certificate_source_sha256=source_hash,precision=cert.ctx.prec,
            input_sha256=hashes,rows=rows,complete=len(rows)==18,
            scope='Outward-rounded initial norm/loss conditions only for generic energy theorem; no finite-width fine-force constants evaluated'),indent=2))
PY
