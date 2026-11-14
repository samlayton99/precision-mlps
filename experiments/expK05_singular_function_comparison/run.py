"""Save both full width curves; resume from already constructed networks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expK05_singular_function_comparison.construction import (
    TARGETS, TargetExtension, Network, build, metrics, prepare_stencil,
)

OUT = ROOT/'results/checkpoint_K_practical_recipes/expK05_singular_function_comparison'
SIZES = [1, 2, 4, 8, 16, 24, 32, 48, *range(64, 273, 16)]


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def run_one(job):
    name, n = job
    folder = OUT/name
    folder.mkdir(exist_ok=True)
    model = folder/f'N{n}_R32.npz'
    row_path = folder/f'N{n}_R32.json'
    if row_path.exists():
        return json.loads(row_path.read_text())
    start = time.monotonic()
    net = Network.load(model) if model.exists() else build(TargetExtension(name=name), n, OUT/'cache')
    net.save(model)
    row = dict(target=name, N=n, halo=32, blocks=15, width=net.width, parameters=3*net.width+1,
               relative_l2=metrics(net, name, 256), relative_l2_order512=metrics(net, name, 512),
               model=str(model.relative_to(ROOT)), seconds=time.monotonic()-start)
    save(row_path, row)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', nargs='+', type=int, default=SIZES)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source = [Path(__file__), Path(__file__).with_name('construction.py'),
              ROOT/'src/construction/qi_mpmath.py',
              ROOT/'experiments/expK02_inverse_sqrt_extension/construction.py',
              ROOT/'experiments/expK04_npu_elementary_suite/problem.py']
    spec = dict(targets=TARGETS, domain=[.0001, 1.], N=SIZES, halo=32, blocks=15,
                cutoff_order=16, extension='C17; flat outside [0.00005,3.2768]',
                lambda_star=.25, Kc=160, construction_decimal_digits=30, inference='float64',
                metric='physical-dx relative L2', training=False, readout_refit=False,
                sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source})
    path = OUT/'specification.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(spec)):
        raise RuntimeError('Source/specification changed: preserve previous results and audit before resuming')
    save(path, spec)
    for n in args.sizes:
        prepare_stencil(n, OUT/'cache')
    jobs = [(name, n) for n in args.sizes for name in TARGETS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_one, jobs):
            print(row['target'], row['N'], row['width'], row['relative_l2'], flush=True)
    rows = [json.loads(p.read_text()) for name in TARGETS for p in (OUT/name).glob('N*_R32.json')]
    save(OUT/'scaling.json', sorted(rows, key=lambda r: (r['target'], r['N'])))


if __name__ == '__main__':
    main()
