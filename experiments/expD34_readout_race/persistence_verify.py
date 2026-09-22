"""Independent numerical comparisons for the completed persistence campaign."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import persistence as pe, persistence_analyze as pa, persistence_quadrature as pq
from . import persistence_theory as pt, stagnation, targets, transport


def verify(source, root):
    x, y, _, _ = pe.af.data('moment9'); mapping = targets.polynomial_map(x)
    q = np.polynomial.legendre.legvander(x, 9) @ mapping
    rules = {n: pq.empirical_rule(x, n) for n in (64, 128)}
    field = jax.jit(pe.field, static_argnums=(5,))
    pairs = []; checks = []; force_rows = []; refinements = []; bounds = []
    archives = {}; hashes = {}
    for folder in sorted((root/'runs').iterdir()):
        if not (folder/'snapshots.npz').exists(): continue
        manifest = json.loads((folder/'manifest.json').read_text())
        with np.load(folder/'snapshots.npz') as f: arrays = {k: f[k] for k in f.files}
        archives[folder.name] = (manifest, arrays)
        hashes[str(folder/'snapshots.npz')] = hashlib.sha256((folder/'snapshots.npz').read_bytes()).hexdigest()
        p0, _, _, _ = pe.load_inputs(source, manifest['start'], manifest['seeds'])
        checks.append(dict(run=folder.name, complete=int(arrays['steps'][-1]) == manifest['end'],
            starting_parameter_error=float(np.max(abs(p0-arrays['p0']))),
            motion_identity=float(np.max(abs(arrays['positive']-arrays['negative']-(abs(arrays['p'][:, :, :177])-abs(p0[:, None, :177]))))),
            minimum_coarse_eigenvalue=float(arrays['min_coarse'][:, 1:].min())))
    reference = archives.get('full_600k_q64')
    if reference:
        rm, ref = reference
        for name in ('full_600k_q128', 'full_600k_half', 'ten_mode_600k_half', 'linear_features_600k_half', 'full_600k_direct',
                     'five_mode_100k_half', 'ten_mode_100k_half', 'linear_features_100k_half'):
            if name not in archives: continue
            cm, comparison = archives[name]
            reference_name = 'ten_mode_600k_q64' if name.startswith('ten_mode') else 'linear_features_600k' if name.startswith('linear_features') else 'full_600k_q64'
            if name == 'full_600k_direct': reference_name = 'full_600k_q64_control'
            if cm['start'] == 100000: reference_name = cm['model']+'_100k'
            if reference_name not in archives: continue
            mm, rr = archives[reference_name]
            common = np.intersect1d(rr['steps'], comparison['steps'])
            for i, seed in enumerate(cm['seeds']):
                ri = mm['seeds'].index(seed)
                for step in common:
                    if step == cm['start']: continue
                    a = rr['p'][ri, np.flatnonzero(rr['steps'] == step)[0]]
                    b = comparison['p'][i, np.flatnonzero(comparison['steps'] == step)[0]]
                    pairs.append(dict(comparison=name, reference=reference_name, seed=seed, step=step,
                        parameter_error=np.linalg.norm(a-b), slope_error=np.linalg.norm(a[:177]-b[:177]),
                        mean_gamma_error=abs(np.mean(abs(a[:177])-abs(b[:177]))),
                        reference_slope_displacement=np.linalg.norm(a[:177]-rr['p0'][ri, :177])))
        # Test the same actual-state force with 64 nodes, 128 nodes, and the
        # original sample sum; evaluation error is distinct from trajectory error.
        for step in (600000, 1000000, 2000000, 4000000, 6000000):
            matches = np.flatnonzero(ref['steps'] == step)
            if not len(matches): continue
            for seed in range(5):
                p = ref['p'][seed, matches[0]]
                for model in ('full', 'five_mode', 'ten_mode'):
                    columns = [0, 1, 2, 3, 9] if model == 'five_mode' else list(range(10))
                    qq = q[:, columns] if model != 'full' else np.empty((len(x), 0))
                    g, _, jc = jax.device_get(field(p, p, x, y, qq, model, np.ones_like(x)/len(x)))
                    effective = g-jc.T @ np.linalg.solve(jc @ jc.T, jc @ g)
                    for nodes, (z, mass) in rules.items():
                        yz = targets.values('moment9', z, mapping)
                        qz = np.polynomial.legendre.legvander(z, 9) @ mapping
                        qz = qz[:, columns] if model != 'full' else np.empty((nodes, 0))
                        gn, rn, jn = jax.device_get(field(p, p, z, yz, qz, model, mass))
                        en = gn-jn.T @ np.linalg.solve(jn @ jn.T, jn @ gn)
                        force_rows.append(dict(seed=seed, step=step, model=model, nodes=nodes,
                            gradient_error=np.linalg.norm(gn-g), slope_error=np.linalg.norm(gn[:177]-g[:177]),
                            effective_slope_error=np.linalg.norm(en[:177]-effective[:177]),
                            reference_effective_slope_norm=np.linalg.norm(effective[:177]),
                            analytic_truncation_bound=pq.analytic_force_remainder(p, x, nodes, model, qz.T @ (mass*rn))))
        for seed in range(5):
            p = ref['p'][seed, -1]
            low, vl = stagnation.diagnostics(p[:-1].reshape(3, 177), p[-1], x, y, degree=65)
            high, vh = stagnation.diagnostics(p[:-1].reshape(3, 177), p[-1], x, y, degree=129)
            refinements.append(dict(seed=seed, step=int(ref['steps'][-1]),
                signed_effective_error=abs(low['effective_outward']-high['effective_outward']),
                vector_effective_error=np.linalg.norm(vl['effective_a']-vh['effective_a'])))
    # Check the radius against independent true-state samples, without feeding
    # these errors into its constants or horizon selection.
    with gzip.open(root/'enclosures.csv.gz', 'rt') as f: enclosure = list(csv.DictReader(f))
    for start in (100000, 600000):
        pp, _, _, _ = pe.load_inputs(source, start, range(5))
        for seed, p in enumerate(pp):
            actual = pa.original_states(source, seed)
            if reference:
                actual.update({int(t): reference[1]['p'][seed, i] for i, t in enumerate(reference[1]['steps'])})
            spectrum = pt.frozen_spectrum(p, x, y)
            for row in enclosure:
                if int(row['seed']) != seed or int(row['start']) != start or row['closed'] != 'True': continue
                step = start+int(row['end'])
                if step not in actual: continue
                prediction = pt.frozen_at(spectrum, int(row['end']))[0]
                error = np.linalg.norm(actual[step]-prediction); radius = float(row['end_error'])
                bounds.append(dict(seed=seed, start=start, step=step, parameter_error=error,
                    radius=radius, ratio=error/radius, contained=bool(error <= radius)))
    pa.table(root/'run_verification.csv', checks)
    if pairs: pa.table(root/'numerical_pairs.csv', pairs)
    if force_rows: pa.table(root/'quadrature_verification.csv', force_rows)
    if refinements: pa.table(root/'modal_verification.csv', refinements)
    if bounds: pa.table(root/'bound_validation.csv', bounds)
    record = dict(run_hashes=hashes, backend='cpu', float_precision='float64',
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        maximum_starting_error=max(r['starting_parameter_error'] for r in checks),
        maximum_motion_identity=max(r['motion_identity'] for r in checks),
        incomplete_runs=[r['run'] for r in checks if not r['complete']],
        bound_validation_failures=sum(not r['contained'] for r in bounds),
        claim='Analytical inequalities evaluated in FP64; no directed-rounding interval certification')
    (root/'verification.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'run_hashes'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); verify(args.source, args.root)
