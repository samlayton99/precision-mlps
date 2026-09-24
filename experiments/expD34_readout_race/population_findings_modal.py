"""CPU-only scalar postprocessing and plotting with bounded downloads."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/work')
EVIDENCE=Path('results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence')
SOURCES=tuple(EVIDENCE/path for path in ('feedback_budget_final/trajectories.csv',
    'initial_feedback/initial.json','feedback_flow/facts.json','feedback_flow/states.csv',
    'feedback_flow_100k/facts.json','feedback_flow_100k/states.csv'))
FILES=(*SOURCES,Path('experiments/expD34_readout_race/population_feedback_findings.py'),
       Path('experiments/expD34_readout_race/population_accumulated_audit.py'),
       Path('experiments/expD34_readout_race/population_findings_modal.py'))
app=modal.App('d34-feedback-findings')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('numpy==2.2.6','scipy==1.15.3','matplotlib==3.10.3')
       .env({'PYTHONPATH':'/work','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}))
if modal.is_local():
    for path in FILES: image=image.add_local_file(ROOT/path,str(Path('/work')/path))


@app.function(image=image,cpu=2,memory=(1024,4096),timeout=300,max_containers=1,retries=0)
def run()->bytes:
    import resource
    output=Path('/tmp/findings')
    subprocess.run([sys.executable,'-m','experiments.expD34_readout_race.population_feedback_findings',
                    '--root',str(EVIDENCE),'--output',str(output)],cwd='/work',check=True,timeout=240)
    record=dict(platform='Modal CPU',memory_hard_limit_mib=4096,
                child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
                hashes={str(p):hashlib.sha256((Path('/work')/p).read_bytes()).hexdigest() for p in FILES})
    (output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    packed=io.BytesIO()
    with zipfile.ZipFile(packed,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output.iterdir(): archive.write(path,path.name)
    return packed.getvalue()


@app.local_entrypoint()
def main(output:str):
    destination=Path(output)
    if destination.exists(): raise ValueError('Use a new output directory')
    packed=run.remote()
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        if sum(item.file_size for item in archive.infolist())>8*1024**2: raise ValueError('Download cap exceeded')
        if any(Path(item.filename).name!=item.filename for item in archive.infolist()): raise ValueError('Flat artifacts only')
        destination.mkdir(parents=True); archive.extractall(destination)
    print(f'Downloaded {len(packed)} bytes to {destination}')
