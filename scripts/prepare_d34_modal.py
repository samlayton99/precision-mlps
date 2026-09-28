"""Package queued source and transferred inputs; performs no numerical work."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, inputs, archive, expected):
    repo = Path(__file__).resolve().parents[1]
    if digest(archive) != expected:
        raise ValueError('Transferred input archive does not match Runpod SHA-256')
    source.mkdir(parents=True, exist_ok=False)
    inputs.mkdir(parents=True, exist_ok=False)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '0f3385d'], cwd=repo, text=True).splitlines()
    names = [n for n in names if Path(n).suffix in ('.py', '.yaml', '.yml', '.toml')
             and (n.startswith(('src/', 'experiments/', 'tests/')) or n == 'pyproject.toml')]
    packed = subprocess.check_output(['git', 'archive', '0f3385d', '--', *names], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(packed)) as tar:
        tar.extractall(source, filter='data')
    original = {n: digest(source / n) for n in names}
    overlay = ['experiments/expD34_readout_race/' + n for n in (
        'plateau_runtime.py', 'plateau_run.py', 'plateau_probes.py', 'modal_campaign.py', 'modal_pilot.py')]
    overlay += ['tests/' + n for n in ('test_expD34_plateau_runtime.py',
                'test_expD34_plateau_outputs.py', 'test_expD34_modal_campaign.py')]
    for name in overlay:
        shutil.copy2(repo / name, source / name)
    files = {n: digest(source / n) for n in sorted(set(names + overlay))}
    tests = sorted(n for n in files if n.startswith('tests/test_expD34'))
    lock = dict(numerical_commit='0f3385d', runtime_base_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), files=files,
        archived_files=original, runtime_overlays=overlay, tests=tests)
    (source / 'source_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    with tarfile.open(archive) as tar:
        tar.extractall(inputs, filter='data')
    hashes = {str(p.relative_to(inputs)): digest(p) for p in sorted(inputs.rglob('*')) if p.is_file()}
    (inputs / 'hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(dict(source=str(source), inputs=str(inputs), input_archive_sha256=expected,
                          input_files=len(hashes), source_files=len(files), tests=len(tests))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    prepare(args.source, args.inputs, args.archive, args.sha256)
