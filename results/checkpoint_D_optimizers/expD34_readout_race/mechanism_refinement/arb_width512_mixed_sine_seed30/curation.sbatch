#!/bin/bash
set -euo pipefail
srun env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,shutil
from flint import arb,ctx
ctx.prec=100
base=Path('/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923')
out=base/'evidence/arb_width512_mixed_sine_seed30'
summary=[]
for chunk,job,elapsed in [(1000,1291,23),(100,1293,697)]:
    log=base/'logs'/f'arb512-c{chunk}-{job}.out'
    text=log.read_text()
    rows=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
    assert rows and 'Vacuous growing radius' in text
    assert all(r['start']==i*chunk and r['end']==(i+1)*chunk for i,r in enumerate(rows))
    assert all(r['all_excluded'] for r in rows)
    actual_prefix=(arb(rows[-1]['prefix'])/4).upper()
    record=dict(job=job,chunk=chunk,one_cpu_wall_seconds=elapsed,requested_updates=20000,
        completed_certified_prefix=rows[-1]['end'],first_uncompleted_block=[rows[-1]['end']+1,rows[-1]['end']+chunk],
        full_requested_horizon_certified=False,termination='original helper raised Vacuous growing radius',
        original_helper_h='1/64',actual_h='1/256',actual_normalized_prefix_upper=str(actual_prefix),
        actual_lambda_below_0_002=bool(actual_prefix < arb(2)/1000),
        scope='Exact real-arithmetic GD on archived binary64 inputs and initial state, with exact binary64 eta; extracted original stdout bounds before failure',
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),rows=rows)
    (out/f'partial_chunk{chunk}.json').write_text(json.dumps(record,indent=2))
    shutil.copy2(log,out/log.name)
    summary.append({k:v for k,v in record.items() if k!='rows'})
for name in ['arb512-preparation.sbatch','arb512-c100.sbatch']:
    shutil.copy2(base/'logs'/name,out/name)
(out/'outcome.json').write_text(json.dumps(dict(attempts=summary,certificate_source_sha256=hashlib.sha256((base/'code/d34_arb_probe.py').read_bytes()).hexdigest(),source_unchanged=True,postprocessing='Only parse original completed stdout rows and rescale their interval bounds by exact factor1/4; no new enclosure algorithm'),indent=2))
print((out/'outcome.json').read_text())
PY
