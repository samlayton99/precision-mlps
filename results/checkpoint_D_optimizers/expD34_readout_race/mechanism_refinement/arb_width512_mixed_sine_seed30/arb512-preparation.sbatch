#!/bin/bash
set -euo pipefail
cd /workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/code
srun env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/junmiaoh/experiments/precision-mlps/.venv-feedback/bin/python - <<'PY'
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from experiments.expD34_readout_race import effective_feedback as ef
from experiments.expD34_readout_race.mechanism_splitting_baselines import issue
root=Path('/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/evidence')
source=root/'moving_tube_inputs/N512_seed30_20k.npz'
out=root/'arb_width512_mixed_sine_seed30'
out.mkdir(exist_ok=False)
p,x,y,cases=ef.load_inputs(source)
selected=[i for i,c in enumerate(cases) if c['target']=='mixed_sine' and c['seed']==30]
assert len(selected)==1
row=selected[0]
inputs=out/'input.npz'
np.savez_compressed(inputs,p=p[[row]],x=x,y=y[[row]],cases=np.array(json.dumps([cases[row]])))
issue(SimpleNamespace(inputs=inputs,snapshot=None,output=out/'reference_forecast',all_policies=False,horizons=','.join(str(n) for n in range(100,20001,100)),eta=.002,role='prospective'))
with np.load(out/'reference_forecast/predictions.npz') as archive:
    centers=np.zeros((20000,p.shape[1]))
    horizons=archive['horizons'].copy()
    centers[horizons-1]=archive['effective_pure_p'][0,0]
    p0=archive['p0'][0].copy()
np.savez_compressed(out/'chord_input.npz',p0=p0,p=centers)
(out/'preparation.json').write_text(json.dumps(dict(original_input=str(source),original_input_sha256=ef.digest(source),selected_row=row,case=cases[row],selected_input_sha256=ef.digest(inputs),reference_sha256=ef.digest(out/'reference_forecast/predictions.npz'),chord_input_sha256=ef.digest(out/'chord_input.npz'),used_rows='99,199,...,19999; all other rows are zero and must not be read',supported_chunks=[100,1000],reference='new archived binary64 frozen-effective forecast endpoints; exact chords between them, not the earlier FP64 moving-tube reference',certificate_source_sha256=ef.digest(Path.cwd()/'d34_arb_probe.py'),helper_h='1/64',actual_h='1/256',headline_rescaling='divide helper prefix by4; helper exclusion at1/4 implies actual normalized lambda below1/16'),indent=2))
print((out/'preparation.json').read_text())
PY
