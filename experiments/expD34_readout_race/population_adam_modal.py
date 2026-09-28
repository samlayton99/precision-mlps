"""Bounded Modal execution for the Adam population audit; no local numerics."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/work')
CODE=Path('experiments/expD34_readout_race')
BASE=Path('results/checkpoint_D_optimizers/expD34_readout_race')
NAMES=('population_adam.py','population_adam_modal.py','population_adam_run.py',
       'population_adam_analysis.py','adam_forces.py','targets.py','effective_feedback_holdout.py')
FILES=[CODE/n for n in NAMES if (ROOT/CODE/n).exists()]+[Path('tests/test_population_adam.py')]
app=modal.App('d34-adam-population')
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda12]==0.11.1','numpy==2.5.1','scipy==1.18.0','matplotlib==3.10.3','pytest==8.4.0')
    .env({'PYTHONPATH':'/work','JAX_ENABLE_X64':'true','JAX_PLATFORMS':'cpu',
          'XLA_PYTHON_CLIENT_PREALLOCATE':'false','MPLBACKEND':'Agg',
          'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}))
if modal.is_local():
    for path in FILES:image=image.add_local_file(ROOT/path,str(Path('/work')/path))
    for folder in sorted((ROOT/BASE/'adam_force_extension/raw').glob('*')):
        for name in ('snapshots.npz','manifest.json'):
            path=folder/name
            if path.exists():image=image.add_local_file(path,str(Path('/work')/path.relative_to(ROOT)))
    for path in sorted((ROOT/BASE/'mechanism_refinement/adam/runs').glob('*/snapshots/*.npz')):
        image=image.add_local_file(path,str(Path('/work')/path.relative_to(ROOT)))
    for path in sorted((ROOT/BASE/'mechanism_refinement/adam/runs').glob('*/manifest.json')):
        image=image.add_local_file(path,str(Path('/work')/path.relative_to(ROOT)))
    spectrum=ROOT.parent/'precision-mlps-frozen-gamma-probe'
    relative=Path('results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/section34_long_horizon_20260924/main')
    for name in ('joint_adam/parameter_checkpoints.npy','joint_adam/checkpoint_steps.npy',
                 'joint_adam/metadata.json','base/joint_input.npz','base/manifest.json'):
        path=spectrum/relative/name
        if path.exists():image=image.add_local_file(path,str(Path('/spectrum')/name))
    for path in sorted((ROOT/BASE/'population_output/evidence').glob('window_*20260925/states.csv')):
        image=image.add_local_file(path,str(Path('/work')/path.relative_to(ROOT)))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def execute(stage,seconds,payload=b'',gpu=False):
    import resource
    started=time.monotonic();output=Path('/tmp/adam-population');output.mkdir()
    environment=dict(os.environ,JAX_PLATFORMS='cuda' if gpu else 'cpu')
    tests=[sys.executable,'-m','pytest','tests/test_population_adam.py','-q','-p','no:cacheprovider']
    checked=subprocess.run(tests,cwd='/work',env=environment,text=True,capture_output=True,timeout=300)
    print(checked.stdout,flush=True)
    if checked.returncode:raise RuntimeError(checked.stdout+checked.stderr)
    command=tests;logs=checked.stdout
    if stage!='verify':
        module='population_adam_analysis' if stage=='analyze' else 'population_adam_run'
        command=[sys.executable,'-m',f'experiments.expD34_readout_race.{module}',
                 '--output',str(output),'--stage',stage,
                 '--seconds',str(max(1,seconds-(time.monotonic()-started)-30))]
        if payload:
            source=Path('/tmp/adam-evidence-inputs');source.mkdir()
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                if sum(i.file_size for i in z.infolist())>128*1024**2:raise ValueError('Input cap')
                if any(Path(i.filename).is_absolute() or '..' in Path(i.filename).parts for i in z.infolist()):raise ValueError('Input path')
                z.extractall(source)
            command+=['--inputs',str(source)]
        with subprocess.Popen(command,cwd='/work',env=environment,text=True,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            for line in process.stdout:
                print(line,end='',flush=True);logs+=line
            if process.wait():raise RuntimeError(logs[-12000:])
    record=dict(platform='Modal GPU' if gpu else 'Modal CPU',stage=stage,
        seconds=time.monotonic()-started,memory_hard_limit_mib=8192 if gpu else 4096,
        child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
        command=command,tests=checked.stdout,log=logs,
        source_hashes={str(p):digest(Path('/work')/p) for p in FILES})
    (output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    paths=[p for p in output.rglob('*') if p.is_file()]
    if sum(p.stat().st_size for p in paths)>64*1024**2:raise ValueError('Artifact cap')
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(path,str(path.relative_to(output)))
    return data.getvalue()


@app.function(image=image,cpu=2,memory=(1024,4096),timeout=7200,max_containers=1,retries=0)
def cpu(stage:str,seconds:float,payload:bytes=b''):
    return execute(stage,seconds,payload)


@app.function(image=image,gpu=['H100','H200'],cpu=4,memory=(2048,8192),timeout=10800,max_containers=1,retries=0)
def gpu(stage:str,seconds:float):
    return execute(stage,seconds,gpu=True)


@app.local_entrypoint()
def main(output:str,stage:str='verify',seconds:float=300,sources:str='',recover:str=''):
    destination=Path(output)
    if destination.exists():raise ValueError('Use a fresh output path')
    if not 0<seconds<=10800:raise ValueError('Three GPU-hour campaign cap')
    if recover:data=modal.FunctionCall.from_id(recover).get()
    else:
        payload=io.BytesIO()
        if sources:
            with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
                for i,folder in enumerate(sources.split(',')):
                    for p in sorted(Path(folder).iterdir()):
                        if p.suffix in ('.csv','.json'):
                            if p.stat().st_size>32*1024**2:raise ValueError('Scalar input cap')
                            z.write(p,f'part{i}/{p.name}')
        call=(cpu.spawn(stage,seconds,payload.getvalue()) if stage in ('verify','archive','analyze')
              else gpu.spawn(stage,seconds))
        print(f'Recoverable Modal call: {call.object_id}',flush=True)
        data=call.get()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('Artifact cap')
        if any(Path(i.filename).is_absolute() or '..' in Path(i.filename).parts for i in z.infolist()):raise ValueError('Artifact path')
        destination.mkdir(parents=True);z.extractall(destination)
    print(f'Downloaded {len(data)} bytes to {destination}',flush=True)
