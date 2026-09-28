"""Memory-bounded Modal CPU verification of six small initial-state capsules."""
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
HELPER=Path('experiments/expD34_readout_race/population_initial_feedback.py')
TEST=Path('tests/test_population_initial_feedback.py')
INPUT=Path('results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/energy_baseline_arb/inputs/N512_seed30_20k.npz')
FILES=(HELPER,TEST,INPUT,Path('experiments/expD34_readout_race/population_initial_modal.py'))
app=modal.App('d34-initial-feedback-certificate')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('numpy==2.2.6','scipy==1.15.3','python-flint==0.8.0','pytest==8.4.0')
       .env({'PYTHONPATH':'/work','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}))
if modal.is_local():
    for path in FILES:
        image=image.add_local_file(ROOT/path,str(Path('/work')/path))


@app.function(image=image,cpu=2,memory=(1024,4096),timeout=1800,max_containers=1,retries=0)
def run(certify:bool=True)->bytes:
    import resource
    import importlib.metadata
    started=time.time(); output=Path('/tmp/initial-feedback')
    commands=[[sys.executable,'-m','pytest',str(TEST),'-q','-p','no:cacheprovider'],
              [sys.executable,'-m','experiments.expD34_readout_race.population_initial_feedback',
               '--inputs',str(INPUT),'--output',str(output)]]
    if certify: commands[-1].append('--certify')
    logs=[]
    for command in commands:
        process=subprocess.run(command,cwd='/work',text=True,stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,timeout=1650,check=False)
        print(process.stdout,flush=True)
        logs.append(dict(command=command,returncode=process.returncode,stdout=process.stdout))
        if process.returncode: raise RuntimeError('Remote verification failed')
    record=dict(platform='Modal CPU',started_unix=started,finished_unix=time.time(),
                memory_hard_limit_mib=4096,
                child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
                files={str(p):hashlib.sha256((Path('/work')/p).read_bytes()).hexdigest() for p in FILES},
                versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','python-flint','pytest')},
                commands=logs)
    (output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    packed=io.BytesIO()
    with zipfile.ZipFile(packed,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output.iterdir(): archive.write(path,path.name)
    if len(packed.getvalue()) > 4*1024**2: raise RuntimeError('Unexpected artifact size')
    return packed.getvalue()


@app.local_entrypoint()
def main(output:str,certify:bool=True):
    destination=Path(output)
    if destination.exists(): raise ValueError('Use a new evidence directory')
    data=run.remote(certify)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 8*1024**2:
            raise ValueError('Artifact cap exceeded')
        if any(Path(item.filename).name != item.filename for item in archive.infolist()):
            raise ValueError('Only flat evidence artifacts expected')
        destination.mkdir(parents=True)
        archive.extractall(destination)
    print(f'Downloaded {len(data)} bytes to {destination}')
