"""Audit retained degree-9 states and run bounded modal-loss continuations."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import stagnation as st, targets, transport as tr
from .recovery import clean

STEPS = (20000, 100000, 600000)


def write_table(path, rows):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


def extract(path, seed, steps):
    with np.load(path) as f:
        cases = json.loads(str(f['cases']))
        indexes = [i for i, c in enumerate(cases) if c['seed'] == seed
                   and c['target'] == 'moment9' and c['arm'] == 'joint']
        if len(indexes) != 1:
            raise ValueError(f'Expected one unchanged-GD degree-9 case in {path}')
        i = indexes[0]
        if cases[i]['training_eta'] != .002:
            raise ValueError('Input must be the original equal-rate GD trajectory')
        j = [int(np.flatnonzero(f['steps'] == step)[0]) for step in steps]
        return f['z'][i, j], f['d'][i, j], f['x'], f['y'][i]


def audit(curated, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, vectors, probes, provenance, zz, dd, yy = [], [], [], [], [], [], []
    refinement = []; duplicate_errors = []
    for seed in range(5):
        folder = f'seed{seed}_fork20000' if seed < 3 else f'dense_seed{seed}'
        path = curated/'states'/folder/'compact_states.npz'
        z, d, x, y = extract(path, seed, STEPS)
        np.testing.assert_array_equal(x, targets.grid(2048))
        np.testing.assert_allclose(y, targets.data(2048, 'moment9')['y'], atol=2e-13, rtol=2e-13)
        provenance.append(dict(seed=seed, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if seed < 3:
            alternate = curated/'states'/f'seed{seed}_fork100000'/'compact_states.npz'
            az, ad, ax, ay = extract(alternate, seed, (100000,))
            np.testing.assert_array_equal(ax, x); np.testing.assert_array_equal(ay, y)
            np.testing.assert_allclose(az[0], z[1], atol=2e-13, rtol=2e-13)
            np.testing.assert_allclose(ad[0], d[1], atol=2e-13, rtol=2e-13)
            duplicate_errors.append(float(max(np.max(abs(az[0]-z[1])), abs(ad[0]-d[1]))))
        zz.append(z); dd.append(d); yy.append(y)
        q = tr.basis(x, 65); qfine = tr.basis(x, 129)
        for j, step in enumerate(STEPS):
            for arm in st.ARMS:
                row, arr = st.diagnostics(z[j], d[j], x, y, arm, basis=q)
                rows.append(dict(seed=seed, step=step, arm=arm, **row))
                vectors.append({k: arr[k] for k in ('gradient', 'effective_a', 'tracking_a', 'generated_a', 'hard_a', 'mode_outward')})
                fine, _ = st.diagnostics(z[j], d[j], x, y, arm, degree=129, basis=qfine)
                refinement.append(dict(seed=seed, step=step, arm=arm,
                    effective_outward_error=abs(fine['effective_outward']-row['effective_outward']),
                    tracking_outward_error=abs(fine['tracking_outward']-row['tracking_outward']),
                    effective_force_norm_error=abs(fine['effective_a_norm']-row['effective_a_norm'])))
            for probe in st.force_probes(z[j], d[j], x, y, basis=q):
                probes.append(dict(seed=seed, step=step, **probe))
    np.savez_compressed(output/'inputs.npz', z=np.stack(zz), d=np.stack(dd), y=np.stack(yy), x=x,
                        seeds=np.arange(5), steps=np.array(STEPS))
    np.savez_compressed(output/'audit_vectors.npz', **{k: np.stack([a[k] for a in vectors]) for k in vectors[0]})
    write_table(output/'audit.csv.gz', rows)
    write_table(output/'modal_refinement.csv', refinement)
    write_table(output/'force_probes.csv', probes)
    write_json(output/'protocol.json', dict(target='moment9', seeds=list(range(5)),
        audit_steps=list(STEPS), starts=list(STEPS[:2]), horizon=20000, eta=.002,
        half_step_eta=.001, half_step_seeds=[0], width=177, m=2048, dtype='float64',
        arms={arm: dict(modes=list(st.MODES[arm]), remove=arm == 'remove_lower') for arm in st.ARMS},
        diagnostic_stride=200, extra_early_offsets=[1, 2, 5, 10, 20, 50, 100],
        integrals='Exact per-update travel and full-complement coarse-projection tracking; scalar modal curves sampled',
        evidence_role='Training-grid optimization mechanism; no held-out generalization evaluation',
        selection='Five-mode choice informed by prior retained-state diagnostics; fixed before continuations',
        source_archives=provenance, duplicate_state_max_error=max(duplicate_errors),
        inputs_sha256=hashlib.sha256((output/'inputs.npz').read_bytes()).hexdigest()))
    print(json.dumps(dict(audited_states=15, arm_comparisons=len(rows), duplicate_max=max(duplicate_errors))), flush=True)


def run(inputs, output, arm, eta, horizon=20000, require_gpu=True):
    if not jax.config.x64_enabled:
        raise ValueError('Set JAX_ENABLE_X64=true')
    output.mkdir(parents=True, exist_ok=True)
    if require_gpu:
        from .run import verify_gpu
        verify_gpu(output)
    with np.load(inputs) as f:
        seeds = [0] if eta == .001 else list(range(5))
        cases = [dict(seed=s, fork_step=step, arm=arm, training_eta=eta)
                 for s in seeds for step in STEPS[:2]]
        z = np.stack([f['z'][s, j] for s in seeds for j in range(2)])
        d = np.array([f['d'][s, j] for s in seeds for j in range(2)])
        y = np.stack([f['y'][s] for s in seeds for j in range(2)])
        x = f['x']
    factor = round(.002/eta)
    if eta not in (.002, .001):
        raise ValueError('This protocol uses eta=0.002 or its matched half step')
    archive = output/'compact_states.npz'
    if archive.exists():
        raise FileExistsError(f'Preserve the existing run at {archive}')
    manifest = dict(cases=cases, input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        source_commit=os.environ.get('RACE_SOURCE_COMMIT'), jax=jax.__version__, numpy=np.__version__,
        horizon=horizon, reference_eta=.002, diagnostic_degree=65,
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__), Path(st.__file__), Path(tr.__file__))})
    write_json(output/'manifest.json', manifest)
    state = st.initial(z, d)
    offsets = sorted(set(range(0, horizon+1, 200)) | {0, horizon}
                     | {k for k in (1, 2, 5, 10, 20, 50, 100) if k <= horizon})
    snapshots, rows = [], []
    q = tr.basis(x, 65)
    advance = st.advance_factory(len(x), z.shape[-1], arm, eta)
    started = time.monotonic(); previous = 0
    for offset in offsets:
        if offset:
            state = advance(state, jnp.asarray(y), (offset-previous)*factor)
        host = jax.device_get(state)
        if not all(np.all(np.isfinite(a)) for a in host.values()):
            raise FloatingPointError(f'Nonfinite state at offset {offset}')
        if np.min(host['min_coarse_eigenvalue']) <= 64*np.finfo(float).eps:
            raise ValueError('Coarse projection lost numerical resolution during GD')
        snapshots.append(host)
        for i, case in enumerate(cases):
            row, _ = st.diagnostics(host['z'][i], host['d'][i], x, y[i], arm, basis=q)
            row.update(positive_travel=host['positive'][i].mean(), negative_travel=host['negative'][i].mean(),
                path=host['path'][i], tracking_travel=host['tracking_travel'][i].mean(),
                crossing=host['crossing'][i], net_outward_fraction=np.mean(abs(host['z'][i, 0]) > abs(z[i, 0])),
                max_positive_travel=host['positive'][i].max())
            for k, name in enumerate('abcd'):
                row[f'tracking_path_{name}'] = host['tracking_path'][i, k]
            rows.append(dict(**case, offset=offset, step=case['fork_step']+offset, **row))
        previous = offset
        if offset % 2000 == 0:
            print(json.dumps(dict(arm=arm, eta=eta, offset=offset, cases=len(cases), seconds=time.monotonic()-started)), flush=True)
    packed = {k: np.stack([s[k] for s in snapshots], axis=1) for k in state}
    np.savez_compressed(archive, offsets=np.array(offsets), cases=np.array(json.dumps(cases)), x=x, y=y, **packed)
    write_table(output/'metrics.csv.gz', rows)
    final = snapshots[-1]
    gamma_change = abs(final['z'][:, 0])-abs(z[:, 0])
    balance = lambda v: np.sum(v[:, 0]**2+v[:, 1]**2-v[:, 2]**2, axis=1)
    status = dict(complete=True, cases=len(cases), reference_updates=horizon, actual_updates=horizon*factor,
        seconds=time.monotonic()-started,
        motion_identity_error=np.max(abs(final['positive']-final['negative']-gamma_change)),
        balance_identity_error=np.max(abs(balance(final['z'])-balance(z)-final['balance_flow']-final['balance_discrete'])),
        slope_energy_identity_error=np.max(abs(.5*np.sum(final['z'][:, 0]**2-z[:, 0]**2, axis=1)
            -final['slope_energy_flow']-final['slope_energy_discrete'])),
        min_coarse_eigenvalue=np.min(final['min_coarse_eigenvalue']))
    write_json(output/'status.json', status)
    print(json.dumps(clean(status)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'run'))
    parser.add_argument('--curated', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=st.ARMS)
    parser.add_argument('--eta', type=float, default=.002)
    parser.add_argument('--horizon', type=int, default=20000)
    args = parser.parse_args()
    if args.action == 'audit':
        if args.curated is None:
            parser.error('audit requires --curated')
        audit(args.curated, args.output)
    else:
        if args.inputs is None or args.arm is None:
            parser.error('run requires --inputs and --arm')
        run(args.inputs, args.output, args.arm, args.eta, args.horizon)


if __name__ == '__main__':
    main()
