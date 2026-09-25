"""Memory-bounded Modal jobs for the short-window persistence experiment."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/work')
CODE = Path('experiments/expD34_readout_race')
BASE = Path('results/checkpoint_D_optimizers/expD34_readout_race')
INPUT = BASE/'mechanism_refinement/widths/inputs'
FILES = tuple(CODE/name for name in (
    'population_window.py', 'population_window_modal.py',
    'population_feedback_flow.py', 'population_feedback_budget.py',
    'population_accumulated_audit.py', 'mechanism_persistence_kernel.py',
    'population_balance_dynamics.py', 'targets.py', 'adam_forces.py',
    'effective_feedback_holdout.py'))+(Path('tests/test_population_window.py'),)
# Discover the same mounted files remotely so execution hashes also include
# the runner, comparison tests, analysis, and checkpoint inputs.
FILES += tuple(p.relative_to(ROOT) for p in sorted((ROOT/INPUT).glob('*.npz')))
for name in ('population_window_run.py', 'population_window_analysis.py',
             'population_concentration.py', 'population_concentration_analysis.py'):
    if (ROOT/CODE/name).exists(): FILES += (CODE/name,)
if (ROOT/'tests/test_population_window_comparison.py').exists():
    FILES += (Path('tests/test_population_window_comparison.py'),)
if (ROOT/'tests/test_population_concentration.py').exists():
    FILES += (Path('tests/test_population_concentration.py'),)

app = modal.App('d34-temporal-window-persistence')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('jax[cuda12]==0.11.1', 'numpy==2.5.1', 'scipy==1.18.0',
                      'matplotlib==3.10.3', 'pytest==8.4.0')
         .env({'PYTHONPATH': '/work', 'JAX_ENABLE_X64': 'true',
               'JAX_PLATFORMS': 'cuda', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
               'MPLBACKEND': 'Agg', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}))
if modal.is_local():
    for path in FILES: image = image.add_local_file(ROOT/path, str(Path('/work')/path))


@app.function(image=image, gpu=['H100', 'H200'], cpu=4, memory=(2048,8192),
              timeout=10800, max_containers=1, retries=0)
def run(study: str, seconds: float):
    import resource
    started = time.monotonic()
    output = Path('/tmp/window-evidence')
    output.mkdir()
    command = [sys.executable, '-m', 'pytest', 'tests/test_population_window.py', '-q', '-p', 'no:cacheprovider']
    checked = subprocess.run(command, cwd='/work', text=True, capture_output=True, timeout=300)
    print(checked.stdout, flush=True)
    if checked.returncode: raise RuntimeError(checked.stdout+checked.stderr)
    logs = checked.stdout
    if study != 'verify':
        command = [sys.executable, '-m', 'experiments.expD34_readout_race.population_window_run',
                   '--study', study, '--output', str(output),
                   '--max-seconds', str(max(1, seconds-(time.monotonic()-started)-30))]
        with subprocess.Popen(command, cwd='/work', text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                logs += line
            if process.wait(): raise RuntimeError('Window experiment failed; see remote log')
    execution = dict(platform='Modal GPU', study=study, seconds=time.monotonic()-started,
                     memory_hard_limit_mib=8192,
                     child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
                     inputs_and_code={str(p): hashlib.sha256((Path('/work')/p).read_bytes()).hexdigest()
                                      for p in FILES}, command=command, log=logs)
    (output/'execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    paths = list(output.rglob('*'))
    if sum(p.stat().st_size for p in paths if p.is_file()) > 16*1024**2:
        raise RuntimeError('16 MiB artifact cap exceeded')
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file(): archive.write(path, str(path.relative_to(output)))
    return data.getvalue()


@app.function(image=image,cpu=2,memory=(1024,4096),timeout=1800,max_containers=1,retries=0)
def analyze(payload: bytes, study: str='analysis'):
    import os
    import resource
    started=time.monotonic()
    source=Path('/tmp/window-inputs'); source.mkdir()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(p.file_size for p in archive.infolist())>64*1024**2: raise ValueError('Input cap')
        archive.extractall(source)
    output=Path('/tmp/window-analysis')
    concentration=study.startswith('concentration')
    environment=dict(os.environ, JAX_PLATFORMS='cpu') if concentration else None
    test_file='test_population_concentration.py' if concentration else 'test_population_window_comparison.py'
    test=subprocess.run([sys.executable,'-m','pytest',f'tests/{test_file}',
                         '-q','-p','no:cacheprovider'],cwd='/work',env=environment,
                         text=True,capture_output=True,timeout=180)
    if test.returncode: raise RuntimeError(test.stdout+test.stderr)
    print(test.stdout,flush=True)
    module='population_concentration_analysis' if concentration else 'population_window_analysis'
    command=[sys.executable,'-m',f'experiments.expD34_readout_race.{module}',
             '--inputs',*map(str,sorted(source.glob('*.csv'))),'--output',str(output)]
    if study=='concentration_verify':
        output.mkdir()
        command=test.args
    else:
        checked=subprocess.run(command,cwd='/work',env=environment,text=True,capture_output=True,timeout=1500)
        if checked.returncode: raise RuntimeError(checked.stdout+checked.stderr)
        print(checked.stdout,flush=True)
    record=dict(platform='Modal CPU',seconds=time.monotonic()-started,memory_hard_limit_mib=4096,
                child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
                command=command,tests=test.stdout,input_sha256=hashlib.sha256(payload).hexdigest(),
                code={str(p):hashlib.sha256((Path('/work')/p).read_bytes()).hexdigest()
                      for p in FILES if p.suffix=='.py'})
    (output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    if (source/'inputs_manifest.json').exists():
        (output/'inputs_manifest.json').write_bytes((source/'inputs_manifest.json').read_bytes())
    result=io.BytesIO()
    with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in output.iterdir(): archive.write(path,path.name)
    return result.getvalue()


@app.local_entrypoint()
def main(output: str, study: str='verify', seconds: float=300, sources: str='', recover: str=''):
    destination = Path(output)
    if destination.exists(): raise ValueError('Use a new output directory')
    if not 0 < seconds <= 10800: raise ValueError('Three GPU-hour campaign cap')
    if recover:
        data=modal.FunctionCall.from_id(recover).get()
    elif study in ('analysis','concentration','concentration_verify'):
        payload=io.BytesIO()
        input_manifest=[]
        with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as archive:
            for i,name in enumerate(sources.split(',') if sources else ()):
                path=Path(name)/'states.csv'
                if path.stat().st_size>16*1024**2: raise ValueError('Scalar input cap')
                archive.write(path,f'part{i}.csv')
                input_manifest.append(dict(member=f'part{i}.csv',source=str(path),
                                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                if study=='concentration':
                    sparse=Path(name)/'sparse_states.npz'
                    if sparse.stat().st_size>16*1024**2: raise ValueError('Sparse input cap')
                    archive.write(sparse,f'part{i}.npz')
                    input_manifest.append(dict(member=f'part{i}.npz',source=str(sparse),
                                               sha256=hashlib.sha256(sparse.read_bytes()).hexdigest()))
            archive.writestr('inputs_manifest.json',json.dumps(input_manifest,indent=2)+'\n')
        call=analyze.spawn(payload.getvalue(),study)
        print(f'Recoverable Modal call: {call.object_id}',flush=True)
        data=call.get()
    else:
        call=run.spawn(study,seconds)
        print(f'Recoverable Modal call: {call.object_id}',flush=True)
        data=call.get()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(p.file_size for p in archive.infolist()) > 16*1024**2:
            raise ValueError('Artifact cap exceeded')
        if any(Path(p.filename).is_absolute() or '..' in Path(p.filename).parts for p in archive.infolist()):
            raise ValueError('Invalid artifact path')
        destination.mkdir(parents=True)
        archive.extractall(destination)
    print(f'Downloaded {len(data)} bytes to {destination}')
