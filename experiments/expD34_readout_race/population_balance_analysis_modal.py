"""Memory-bounded remote CPU analysis of the small balance replay artifacts."""
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
BASE=Path('results/checkpoint_D_optimizers/expD34_readout_race')
SOURCE=BASE/'population_output/evidence/balance_flow_20260924/states.csv'
CAPSULE=BASE/'mechanism_refinement/energy_baseline_arb/inputs/N512_seed30_20k.npz'
STRESS_SOURCE=BASE/'population_output/evidence/balance_stress_flow_20260924/states.csv'
STRESS_CAPSULE=BASE/'population_output/evidence/balance_archive_20260924/stress_inputs.npz'
OLD_ENDPOINTS=BASE/'population_output/evidence/feedback_flow_100k/endpoints.npz'
NEW_ENDPOINTS=BASE/'population_output/evidence/balance_flow_20260924/endpoints.npz'
CODE=Path('experiments/expD34_readout_race')
FILES=(SOURCE,CAPSULE,OLD_ENDPOINTS,NEW_ENDPOINTS)+tuple(CODE/name for name in (
    'population_balance_analysis.py','population_balance_analysis_modal.py',
    'population_balance_dynamics.py','mechanism_persistence_kernel.py'))+(Path('tests/test_population_balance_comparison.py'),)
app=modal.App('d34-balance-budget-analysis')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('jax==0.11.1','numpy==2.5.1','matplotlib==3.10.3','pytest==8.4.0')
       .env({'PYTHONPATH':'/work','JAX_ENABLE_X64':'true','JAX_PLATFORMS':'cpu',
             'MPLBACKEND':'Agg','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}))
if modal.is_local():
    if (ROOT/STRESS_SOURCE).exists(): FILES=FILES+(STRESS_SOURCE,STRESS_CAPSULE)
    for path in FILES: image=image.add_local_file(ROOT/path,str(Path('/work')/path))


@app.function(image=image,cpu=2,memory=(1024,4096),timeout=600,max_containers=1,retries=0)
def run(source:str='baseline'):
    import resource
    started=time.time(); output=Path('/tmp/balance-analysis')
    test=subprocess.run([sys.executable,'-m','pytest','tests/test_population_balance_comparison.py','-q','-p','no:cacheprovider'],
                        cwd='/work',text=True,capture_output=True,timeout=60)
    print(test.stdout,flush=True)
    if test.returncode: raise RuntimeError(test.stdout+test.stderr)
    if source not in ('baseline','stress'): raise ValueError('Unknown source')
    src,cap=(SOURCE,CAPSULE) if source=='baseline' else (STRESS_SOURCE,STRESS_CAPSULE)
    command=[sys.executable,'-m','experiments.expD34_readout_race.population_balance_analysis',
             '--source',str(src),'--capsule',str(cap),'--output',str(output)]
    result=subprocess.run(command,cwd='/work',text=True,capture_output=True,timeout=550)
    print(result.stdout,flush=True); print(result.stderr,flush=True)
    if result.returncode: raise RuntimeError('Analysis failed')
    if source=='baseline':
        import numpy as np
        with np.load(Path('/work')/OLD_ENDPOINTS,allow_pickle=False) as old, np.load(Path('/work')/NEW_ENDPOINTS,allow_pickle=False) as new:
            np.testing.assert_array_equal(old['p0'],new['p0'])
            reference=dict(zip(map(str,old['labels']),old['endpoints'],strict=True))
            changes=[dict(label=str(label),max_coordinate_difference=float(np.max(np.abs(p-reference[str(label)]))),
                          parameter_distance=float(np.linalg.norm(p-reference[str(label)])))
                     for label,p in zip(new['labels'],new['endpoints'],strict=True)]
        (output/'endpoint_reproduction.json').write_text(json.dumps(changes,indent=2)+'\n')
    record=dict(platform='Modal CPU',seconds=time.time()-started,memory_hard_limit_mib=4096,
                child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
                files={str(p):hashlib.sha256((Path('/work')/p).read_bytes()).hexdigest()
                       for p in (*FILES,src,cap)},
                command=command,stdout=result.stdout,stderr=result.stderr,tests=test.stdout)
    (output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    packed=io.BytesIO()
    with zipfile.ZipFile(packed,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in output.iterdir(): archive.write(p,p.name)
    return packed.getvalue()


@app.local_entrypoint()
def main(output:str,source:str='baseline'):
    destination=Path(output)
    if destination.exists(): raise ValueError('Use a new destination')
    data=run.remote(source)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(p.file_size for p in archive.infolist())>16*1024**2: raise ValueError('Artifact cap')
        if any(Path(p.filename).name!=p.filename for p in archive.infolist()): raise ValueError('Flat artifacts only')
        destination.mkdir(parents=True); archive.extractall(destination)
    print(f'Downloaded {len(data)} bytes to {destination}')
