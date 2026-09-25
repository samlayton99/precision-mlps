"""Explicit small-archive allowlist; remote CPU, 4 GiB memory, no local arrays."""
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
BASE=Path('results/checkpoint_D_optimizers/expD34_readout_race/mechanism_dilation/evidence')
CODE=Path('experiments/expD34_readout_race')
FILES=tuple(CODE/name for name in ('population_balance_archive.py','population_balance_archive_modal.py',
    'population_balance_dynamics.py','mechanism_persistence_kernel.py'))+tuple(
    BASE/name for panel in ('late_development','late_confirmation','wide_N512','wide_N1024')
    for name in (panel+'.npz',panel+'_run/manifest.json',panel+'_run/000000000.npz',panel+'_run/000020000.npz'))
app=modal.App('d34-balance-archive-audit')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('jax==0.11.1','numpy==2.5.1','matplotlib==3.10.3')
       .env({'PYTHONPATH':'/work','JAX_ENABLE_X64':'true','JAX_PLATFORMS':'cpu',
             'MPLBACKEND':'Agg','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}))
if modal.is_local():
    for path in FILES: image=image.add_local_file(ROOT/path,str(Path('/work')/path))


@app.function(image=image,cpu=2,memory=(1024,4096),timeout=2400,max_containers=1,retries=0)
def run():
    import resource
    started=time.time(); output=Path('/tmp/balance-archive')
    command=[sys.executable,'-m','experiments.expD34_readout_race.population_balance_archive',
             '--base',str(BASE),'--output',str(output)]
    with subprocess.Popen(command,cwd='/work',text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as process:
        lines=[]
        for line in process.stdout: print(line,end='',flush=True); lines.append(line)
        if process.wait(): raise RuntimeError('Archive analysis failed')
    record=dict(platform='Modal CPU',seconds=time.time()-started,memory_hard_limit_mib=4096,
                child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
                files={str(p):hashlib.sha256((Path('/work')/p).read_bytes()).hexdigest() for p in FILES},
                command=command,stdout=''.join(lines))
    (output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    packed=io.BytesIO()
    with zipfile.ZipFile(packed,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in output.iterdir(): archive.write(p,p.name)
    return packed.getvalue()


@app.local_entrypoint()
def main(output:str):
    destination=Path(output)
    if destination.exists(): raise ValueError('Use a new destination')
    data=run.remote()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(p.file_size for p in archive.infolist())>16*1024**2: raise ValueError('Artifact cap')
        if any(Path(p.filename).name!=p.filename for p in archive.infolist()): raise ValueError('Flat artifacts only')
        destination.mkdir(parents=True); archive.extractall(destination)
    print(f'Downloaded {len(data)} bytes to {destination}')
