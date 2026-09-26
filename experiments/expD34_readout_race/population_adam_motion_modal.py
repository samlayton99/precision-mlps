"""Small checkpoint mounts and bounded remote execution for the motion campaign."""
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
EVIDENCE=Path('results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence')
NAMES=('population_adam_motion.py','population_adam_motion_run.py','population_adam_motion_modal.py',
       'population_adam_motion_analysis.py','population_adam.py','population_adam_run.py',
       'adam_forces.py','targets.py','effective_feedback_holdout.py')
FILES=[CODE/n for n in NAMES if (ROOT/CODE/n).exists()]+[
    Path('tests/test_population_adam.py'),Path('tests/test_population_adam_motion.py')]
app=modal.App('d34-adam-population-motion')
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda12]==0.11.1','numpy==2.5.1','scipy==1.18.0','matplotlib==3.10.3','pytest==8.4.0')
    .env({'PYTHONPATH':'/work','JAX_ENABLE_X64':'true','JAX_PLATFORMS':'cpu',
          'XLA_PYTHON_CLIENT_PREALLOCATE':'false','MPLBACKEND':'Agg',
          'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}))
if modal.is_local():
    for path in FILES:image=image.add_local_file(ROOT/path,str(Path('/work')/path))
    for folder in sorted((ROOT/EVIDENCE).glob('adam_population_wide*20260926')):
        paths=[folder/'cases.csv',*folder.glob('*state25000.npz'),*folder.glob('*state125000.npz')]
        for path in paths:
            if path.stat().st_size>2*1024**2:raise ValueError('Expected sparse checkpoints')
            image=image.add_local_file(path,str(Path('/inputs')/folder.name/path.name))


def execute(stage,seconds,payload=b''):
    import resource
    start=time.monotonic();output=Path('/tmp/motion-output');output.mkdir()
    env=dict(os.environ,JAX_PLATFORMS='cuda' if stage=='run' else 'cpu')
    command=[sys.executable,'-m','pytest','tests/test_population_adam.py',
             'tests/test_population_adam_motion.py','-q','-p','no:cacheprovider']
    checked=subprocess.run(command,cwd='/work',env=env,text=True,capture_output=True,timeout=300)
    print(checked.stdout,flush=True)
    if checked.returncode:raise RuntimeError(checked.stdout+checked.stderr)
    logs=checked.stdout
    if stage!='verify':
        source='/inputs'
        if payload:
            source='/tmp/scalar-inputs';Path(source).mkdir()
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('Scalar input cap')
                z.extractall(source)
        name='population_adam_motion_run' if stage=='run' else 'population_adam_motion_analysis'
        command=[sys.executable,'-m',f'experiments.expD34_readout_race.{name}',
                 '--inputs',source,'--output',str(output),'--seconds',str(max(1,seconds-(time.monotonic()-start)-30))]
        with subprocess.Popen(command,cwd='/work',env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as p:
            for line in p.stdout:print(line,end='',flush=True);logs+=line
            if p.wait():raise RuntimeError(logs[-10000:])
    receipt=dict(platform='Modal GPU' if stage=='run' else 'Modal CPU',stage=stage,
        seconds=time.monotonic()-start,memory_hard_limit_mib=8192 if stage=='run' else 4096,
        peak_child_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
        tests=checked.stdout,command=command,log=logs,
        source_hashes={str(p):hashlib.sha256((Path('/work')/p).read_bytes()).hexdigest() for p in FILES})
    (output/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths=[p for p in output.iterdir() if p.is_file()]
    if sum(p.stat().st_size for p in paths)>96*1024**2:raise ValueError('Output cap')
    result=io.BytesIO()
    with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(path,path.name)
    return result.getvalue()


@app.function(image=image,cpu=2,memory=(1024,4096),timeout=600,max_containers=1,retries=0)
def cpu(stage:str,seconds:float,payload:bytes=b''):
    return execute(stage,seconds,payload)


@app.function(image=image,gpu=['H100','H200'],cpu=4,memory=(2048,8192),timeout=6480,max_containers=1,retries=0)
def gpu(seconds:float):
    return execute('run',seconds)


@app.local_entrypoint()
def main(output:str,stage:str='verify',seconds:float=300,source:str='',recover:str=''):
    destination=Path(output)
    if destination.exists():raise ValueError('Use a fresh output path')
    if stage=='run' and not 0<seconds<=6300:raise ValueError('1.8 GPU-hour total cap; allow teardown reserve')
    if recover:data=modal.FunctionCall.from_id(recover).get()
    else:
        payload=io.BytesIO()
        if source:
            with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
                for p in Path(source).iterdir():
                    if p.suffix in ('.csv','.json'):z.write(p,p.name)
        call=gpu.spawn(seconds) if stage=='run' else cpu.spawn(stage,seconds,payload.getvalue())
        print(f'Recoverable Modal call: {call.object_id}',flush=True)
        data=call.get()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if sum(i.file_size for i in z.infolist())>96*1024**2:raise ValueError('Artifact cap')
        if any(Path(i.filename).is_absolute() or '..' in Path(i.filename).parts for i in z.infolist()):raise ValueError('Artifact path')
        destination.mkdir(parents=True);z.extractall(destination)
    print(f'Downloaded {len(data)} bytes to {destination}')
