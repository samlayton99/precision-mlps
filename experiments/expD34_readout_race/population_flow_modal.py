"""One GPU-hour cap; six small initial states, scalar logs, final vectors only."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/work')
BASE=Path('experiments/expD34_readout_race')
INPUT=Path('results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/energy_baseline_arb/inputs/N512_seed30_20k.npz')
STRESS=Path('results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/balance_archive_20260924/stress_inputs.npz')
FILES=tuple(BASE/name for name in ('population_feedback_flow.py','population_flow_modal.py',
    'mechanism_persistence_kernel.py','population_feedback_budget.py','population_accumulated_audit.py',
    'population_balance_dynamics.py','population_balance_flow.py'))+(INPUT,Path('tests/test_population_feedback_flow.py'),
    Path('tests/test_population_balance_dynamics.py'))
app=modal.App('d34-feedback-flow-verification')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('jax[cuda12]==0.11.1','numpy==2.5.1','scipy==1.18.0','pytest==8.4.0')
       .env({'PYTHONPATH':'/work','JAX_ENABLE_X64':'true','JAX_PLATFORMS':'cuda',
             'XLA_PYTHON_CLIENT_PREALLOCATE':'false','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}))
if modal.is_local():
    if (ROOT/STRESS).exists(): FILES=FILES+(STRESS,)
    for path in FILES: image=image.add_local_file(ROOT/path,str(Path('/work')/path))


@app.function(image=image,gpu=['H100','H200'],cpu=4,memory=(2048,8192),
              timeout=3600,max_containers=1,retries=0)
def run(horizon:float=40.,sample_interval:float=.2,study:str='feedback',source:str='baseline')->bytes:
    import resource
    import importlib.metadata
    started=time.time(); output=Path('/tmp/feedback-flow')
    if source not in ('baseline','stress'): raise ValueError('Unknown input capsule')
    selected=INPUT if source=='baseline' else STRESS
    commands=[[sys.executable,'-m','pytest','tests/test_population_feedback_flow.py','-q','-p','no:cacheprovider'],
              [sys.executable,'-m','experiments.expD34_readout_race.population_feedback_flow',
               '--inputs',str(selected),'--output',str(output),
               '--horizon',str(horizon),'--sample-interval',str(sample_interval)]]
    if study == 'balance':
        commands[0].insert(4,'tests/test_population_balance_dynamics.py')
        commands[1][2]='experiments.expD34_readout_race.population_balance_flow'
    elif study != 'feedback':
        raise ValueError('Unknown study')
    logs=[]
    for command in commands:
        with subprocess.Popen(command,cwd='/work',text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as process:
            lines=[]
            for line in process.stdout:
                print(line,end='',flush=True); lines.append(line)
            status=process.wait()
        logs.append(dict(command=command,returncode=status,stdout=''.join(lines)))
        if status: raise RuntimeError('Remote flow verification failed')
    record=dict(platform='Modal GPU',study=study,source=source,started_unix=started,finished_unix=time.time(),
                gpu_seconds=time.time()-started,memory_hard_limit_mib=8192,
                child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
                files={str(p):hashlib.sha256((Path('/work')/p).read_bytes()).hexdigest() for p in (*FILES,selected)},
                versions={p:importlib.metadata.version(p) for p in ('jax','numpy','scipy','pytest')},commands=logs)
    (output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    if sum(p.stat().st_size for p in output.iterdir()) > 16*1024**2: raise RuntimeError('Output cap exceeded')
    packed=io.BytesIO()
    with zipfile.ZipFile(packed,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output.iterdir(): archive.write(path,path.name)
    return packed.getvalue()


@app.local_entrypoint()
def main(output:str,horizon:float=40.,sample_interval:float=.2,study:str='feedback',source:str='baseline'):
    destination=Path(output)
    if destination.exists(): raise ValueError('Use a new output directory')
    if not 0<horizon<=200 or sample_interval<=0: raise ValueError('Bound the continuation to flow time 200')
    data=run.remote(horizon,sample_interval,study,source)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 16*1024**2: raise ValueError('Download cap exceeded')
        if any(Path(item.filename).name!=item.filename for item in archive.infolist()): raise ValueError('Flat files only')
        destination.mkdir(parents=True); archive.extractall(destination)
    print(f'Downloaded {len(data)} bytes to {destination}')
