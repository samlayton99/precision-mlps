"""Launch Figure 4 widths using the completed W=512 protocol.

Prepare with GPUs disabled. Run each lane under an external 10,790-second
timeout (five-second kill grace), exposing exactly one authorized GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from adam_feature_probe_prepare import geometry, grid, initial_parameters, target


GROUPS = ('joint_adam_cosine', 'joint_adam_constant', 'joint_gd')
LANES = {0: (128, 256), 1: (1024,)}
RERUN_LANES = {0: (1024,), 1: (512, 128, 256)}
HORIZON = 5_000_000


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(root, reference, rerun_all=False):
    """Reuse target arrays and recipe grids; change only the width geometry."""
    if (root / 'manifest.json').exists():
        raise FileExistsError('Use a fresh sweep directory')
    with np.load(reference / 'base/joint_input.npz') as saved:
        x, y = saved['x'].copy(), saved['target'].copy()
        reference_parameters = saved['initial_parameters'].copy()
        _, info512 = geometry(512)
        expected = np.array([initial_parameters(512, info512['interior_intervals'], s)
                             for s in range(5)])
        # CPU/library changes can alter the last bit of uniform scaling.
        np.testing.assert_allclose(saved['initial_parameters'], expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(x, grid(2048))
    scale = np.sqrt(np.mean(target(x)**2))
    np.testing.assert_allclose(y, target(x) / scale, rtol=0, atol=2e-14)
    with np.load(reference / 'base/input.npz') as saved:
        evaluation = {k: saved[k].copy() for k in
                      ('train_x', 'target', 'validation_x', 'validation_target',
                       'eval_x', 'eval_target')}
    configs = {}
    for name in GROUPS:
        old = reference / 'h5m' / name
        config = json.loads((old / 'metadata.json').read_text())['config']
        assert config['horizon'] == HORIZON
        assert json.loads((old / 'summary.json').read_text())['completed_updates'] == HORIZON
        configs[name] = config
    manifests = []
    widths = (128, 256, 512, 1024) if rerun_all else (128, 256, 1024)
    for width in widths:
        _, info = geometry(width)
        base = root / f'w{width}' / 'base'
        base.mkdir(parents=True, exist_ok=False)
        parameters = np.array([initial_parameters(width, info['interior_intervals'], s)
                               for s in range(5)])
        if width == 512:
            parameters = reference_parameters.copy()
        np.savez_compressed(base / 'joint_input.npz', x=x, target=y,
                            initial_parameters=parameters)
        np.savez_compressed(base / 'input.npz', **evaluation)
        manifest = dict(**info, seed_ids=list(range(5)), samples=len(x),
                        validation_samples=4096, eval_samples=8192,
                        target_name='mixed_sines', target_normalizer=float(scale),
                        target_formula='sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(14*pi*x)',
                        input_sha256=hashlib.sha256((base / 'joint_input.npz').read_bytes()).hexdigest(),
                        initialization='D24 affine Xavier, RNG keyed by seed and budget-derived intervals')
        write_json(base / 'manifest.json', manifest)
        manifests.append(manifest)
        for name, config in configs.items():
            write_json(root / f'w{width}' / f'{name}.json', config)
    write_json(root / 'manifest.json', dict(
        horizon=HORIZON, widths=[128, 256, 512, 1024], new_widths=list(widths),
        reused_width512=None if rerun_all else str(reference),
        reference_protocol=str(reference), lanes=RERUN_LANES if rerun_all else LANES,
        configs=configs, durable_exports=rerun_all,
        selection='One recipe per width and optimizer by median final validation error across five seeds; all endpoints finite.',
        recipe_scope='Same final five-million-update candidate grid as W=512; no new per-width LR screening.',
        raw_traces='Every update error/RMS; full parameters every 10000 updates.',
        total_gpu_hour_limit=6, lane_timeout_seconds=10400 if rerun_all else 10790,
        kill_grace_seconds=5,
        geometries=manifests))
    print(json.dumps(dict(prepared=str(root), widths=list(widths))), flush=True)


def run_lane(root, lane):
    import jax
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not mask or len(mask.split(',')) != 1:
        raise RuntimeError('Expose exactly one authorized GPU per lane')
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != 'gpu':
        raise RuntimeError(f'Expected one GPU, got {devices}')
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['lane_timeout_seconds'] in (10400, 10790)
    print(json.dumps(dict(lane=lane, gpu_mask=mask, devices=[str(d) for d in devices],
                          pid=os.getpid(), started_unix=time.time())), flush=True)
    runner = Path(__file__).with_name('adam_joint_probe_run.py')
    records = []
    for width in manifest['lanes'][str(lane)]:
        for name in GROUPS:
            case = root / f'w{width}'
            output = case / name
            record = dict(width=width, group=name, output=str(output), started_unix=time.time())
            print(json.dumps(dict(starting=record)), flush=True)
            command = [sys.executable, '-u', str(runner), '--input', str(case / 'base/joint_input.npz'),
                       '--config', str(case / f'{name}.json'), '--output', str(output), '--self-test']
            if manifest.get('durable_exports'):
                command += ['--export-dir', str(root / 'durable' / f'w{width}' / name)]
            with (case / f'{name}.log').open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            record.update(exit_code=result.returncode, ended_unix=time.time())
            if result.returncode == 0:
                record['completed_updates'] = json.loads((output / 'summary.json').read_text())['completed_updates']
                assert record['completed_updates'] == HORIZON
            records.append(record)
            write_json(root / f'lane{lane}_completed.json', records)
            print(json.dumps(dict(finished=record)), flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
    print(json.dumps(dict(lane=lane, complete=True, ended_unix=time.time())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--lane', type=int, choices=LANES)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--rerun-all', action='store_true',
                        help='Rerun all four widths with acknowledged off-pod exports.')
    args = parser.parse_args()
    if args.prepare:
        if args.reference is None:
            parser.error('--prepare needs --reference')
        prepare(args.root, args.reference, args.rerun_all)
    else:
        run_lane(args.root, args.lane)
