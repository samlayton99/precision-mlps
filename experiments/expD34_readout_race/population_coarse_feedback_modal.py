"""Bounded Modal execution; one small checkpoint at a time, no local numerics."""
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import hashlib
import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/work')
CODE=Path('experiments/expD34_readout_race')
EVIDENCE=Path('results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence')
NAMES=('adam_forces.py','targets.py','effective_feedback_holdout.py','population_adam.py',
       'population_adam_run.py','population_coarse_feedback.py','population_coarse_feedback_run.py',
       'population_coarse_feedback_analysis.py','population_coarse_feedback_modal.py')
FILES=[CODE/n for n in NAMES if (ROOT/CODE/n).exists()]+[Path('tests/test_population_coarse_feedback.py')]
app=modal.App('d34-coarse-feedback')
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda12]==0.11.1','numpy==2.5.1','scipy==1.18.0','matplotlib==3.10.3','pytest==8.4.0')
    .env({'PYTHONPATH':'/work','JAX_ENABLE_X64':'true','JAX_PLATFORMS':'cpu',
          'XLA_PYTHON_CLIENT_PREALLOCATE':'false','MPLBACKEND':'Agg',
          'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}))
if modal.is_local():
    for p in FILES:image=image.add_local_file(ROOT/p,str(Path('/work')/p))
    inputs=[]
    for folder in sorted((ROOT/EVIDENCE).glob('adam_population_wide*20260926')):
        inputs += [folder/'cases.csv',*folder.glob('*state25000.npz'),*folder.glob('*state125000.npz')]
    for name in ('window_development_20260925','window_panel31_20260925','window_widths_20260925'):
        inputs += [(ROOT/EVIDENCE/name)/n for n in ('facts.json','sparse_states.npz')]
    folder=ROOT/EVIDENCE/'adam_variance_runs_20260926'
    inputs += [folder/'facts.json',*folder.glob('*fork130000.npz')]
    for p in inputs:
        if p.stat().st_size>8*1024**2:raise ValueError(f'Unexpected large checkpoint: {p}')
        image=image.add_local_file(p,str(Path('/inputs')/p.parent.name/p.name))


def execute(stage,seconds,cohort,payload):
    import os
    import resource
    start=time.monotonic();output=Path('/tmp/feedback-output');output.mkdir()
    gpu=stage in ('audit','gd','adam','followup')
    env=dict(os.environ,JAX_PLATFORMS='cuda' if gpu else 'cpu')
    logs='';returncode=0;command=[]
    if stage in ('verify','audit'):
        command=[sys.executable,'-m','pytest','tests/test_population_coarse_feedback.py','-q','-p','no:cacheprovider']
        result=subprocess.run(command,cwd='/work',env=env,text=True,capture_output=True,timeout=300)
        logs=result.stdout+result.stderr;print(logs,flush=True)
        if result.returncode:raise RuntimeError(logs)
    if stage!='verify':
        source='/inputs'
        if payload:
            source='/tmp/feedback-scalars';Path(source).mkdir()
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                if sum(p.file_size for p in z.infolist())>192*1024**2:raise ValueError('Scalar input cap')
                z.extractall(source)
        module='population_coarse_feedback_analysis' if stage=='analyze' else 'population_coarse_feedback_run'
        remaining=max(1,seconds-(time.monotonic()-start)-45)
        command=['timeout','--signal=TERM','--kill-after=5',str(int(remaining)+15),sys.executable,
            '-m',f'experiments.expD34_readout_race.{module}','--inputs',source,'--output',str(output),
            '--stage',stage,'--cohort',cohort,'--seconds',str(remaining)]
        with subprocess.Popen(command,cwd='/work',env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as p:
            for line in p.stdout:print(line,end='',flush=True);logs+=line
            returncode=p.wait()
    receipt=dict(platform='Modal GPU' if gpu else 'Modal CPU',stage=stage,cohort=cohort,
        seconds=time.monotonic()-start,returncode=returncode,memory_hard_limit_mib=8192 if gpu else 4096,
        peak_child_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
        source_hashes={str(p):hashlib.sha256((Path('/work')/p).read_bytes()).hexdigest() for p in FILES},
        command=command,log=logs)
    (output/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths=[p for p in output.iterdir() if p.is_file()]
    if sum(p.stat().st_size for p in paths)>96*1024**2:raise ValueError('96 MiB output cap')
    result=io.BytesIO()
    with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.name)
    return result.getvalue()


@app.function(image=image,cpu=2,memory=(1024,4096),timeout=1800,retries=0,max_containers=1)
def cpu(stage:str,seconds:float,cohort:str,payload:bytes=b''):
    return execute(stage,seconds,cohort,payload)


@app.function(image=image,gpu=['H100','H200'],cpu=4,memory=(2048,8192),timeout=10800,retries=0,max_containers=1)
def gpu(stage:str,seconds:float,cohort:str):
    return execute(stage,seconds,cohort,b'')


@app.local_entrypoint()
def main(output:str,stage:str='verify',seconds:float=300,cohort:str='all',source:str='',recover:str=''):
    destination=Path(output)
    if destination.exists():raise ValueError('Use a fresh destination')
    spent=0.
    if destination.parent.exists():
        for p in destination.parent.glob('*/execution.json'):
            d=json.loads(p.read_text())
            if d['platform']=='Modal GPU':spent+=d['seconds']+15 # Include receipt/packaging reserve.
    if stage in ('audit','gd','adam','followup') and not recover and not 0<seconds<=10800-spent-60:
        raise ValueError(f'Aggregate 3 GPU-hour cap: {spent:.1f}s already allocated')
    if recover:data=modal.FunctionCall.from_id(recover).get()
    else:
        payload=io.BytesIO()
        if source:
            with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
                for folder in sorted(Path(source).iterdir()):
                    if folder.is_dir() and folder.name!='analysis':
                        for p in folder.iterdir():
                            if p.suffix in ('.csv','.json'):z.write(p,str(Path(folder.name)/p.name))
        call=gpu.spawn(stage,seconds,cohort) if stage in ('audit','gd','adam','followup') else cpu.spawn(stage,seconds,cohort,payload.getvalue())
        print(f'Recoverable Modal call: {call.object_id}',flush=True);data=call.get()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if sum(p.file_size for p in z.infolist())>96*1024**2:raise ValueError('Artifact cap')
        if any(Path(p.filename).is_absolute() or '..' in Path(p.filename).parts for p in z.infolist()):raise ValueError('Artifact path')
        destination.mkdir(parents=True);z.extractall(destination)
    receipt=json.loads((destination/'execution.json').read_text())
    print(json.dumps({k:receipt[k] for k in ('stage','cohort','seconds','returncode','peak_child_rss_mib')}),flush=True)
    if receipt['returncode']:raise RuntimeError('Remote stage failed; partial artifacts retained')
