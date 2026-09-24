"""Direct frozen-GD residual witnesses; SVD proposes vectors but proves nothing.

The bound requires only norms/inner products of the saved witness and design.
This helper evaluates them in FP64, not interval arithmetic.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .mechanism_dilation_analysis import clean, digest, write_csv
from .population_frozen_readout import setup, propagate


ORIGINAL_CUTOFFS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1)
CUTOFFS = (1e-14, 1e-12, 1e-10, 1e-8, *ORIGINAL_CUTOFFS)


def witness_bound(design, residual, witness, eta, updates, L):
    norm = float(np.linalg.norm(witness))
    margin = 2-eta*L
    if norm == 0 or margin <= 0:
        return dict(status='zero_witness_or_unresolved_step', lower_norm=0., stability_margin=margin)
    overlap = float(abs(witness@residual))
    sensitivity = float(np.linalg.norm(design.T@witness))
    allowance = sensitivity*np.linalg.norm(residual)*np.sqrt(updates*eta/margin)
    return dict(status='fp64_evaluation', lower_norm=max(0., overlap-allowance)/norm,
                witness_norm=norm, initial_overlap=overlap, witness_sensitivity=sensitivity,
                movement_allowance=float(allowance), stability_margin=margin)


def run(args):
    with np.load(args.input) as data:
        pack = {k: data[k].copy() for k in ('p', 'x', 'y')}
    manifest = json.loads((args.full_run/'manifest.json').read_text())
    if digest(args.input) != manifest['input_sha256']:
        raise ValueError('Prepared inputs differ from the full-run manifest')
    eta = float(manifest['eta'])
    choices = [(i, c) for i, c in enumerate(manifest['cases'])
               if str(c.get('seed')) in args.seeds.split(',') and c['arm'] in args.arms.split(',')]
    args.output.mkdir(parents=True, exist_ok=False)
    rows, saved = [], []
    best_cases = {}
    for local, (index, case) in enumerate(choices):
        p = pack['p'][case['input_index']]
        a, b, c = p[:-1].reshape(3, -1)
        x, y = pack['x'], pack['y'][case['input_index']]
        design = np.c_[np.tanh(x[:, None]*a+b), np.ones(len(x))]/np.sqrt(len(x))
        target = y/np.sqrt(len(x))
        target_norm = float(np.linalg.norm(target))
        readout = np.r_[c, p[-1]]
        residual = design@readout-target
        spectral = setup(design, target, readout)
        metadata = {k: case.get(k) for k in ('target', 'seed', 'start', 'arm', 'scale', 'reference', 'cohort')}
        candidates = []
        for steps in (20000, 100000):
            fitted, predicted, _ = propagate(spectral, eta, steps)
            direct = design@fitted-target
            for cutoff in CUTOFFS:
                slow = eta*spectral['singular']**2 <= cutoff
                v = spectral['orthogonal']+spectral['U'][:, slow]@spectral['coefficients'][slow]
                norm = np.linalg.norm(v)
                if norm > 0:
                    v /= norm
                old_floor = np.sqrt(float(spectral['orthogonal']@spectral['orthogonal'])
                    +np.exp(2*steps*np.log1p(-cutoff))*float(np.sum(spectral['coefficients'][slow]**2)))/target_norm
                for label, L in (('analytic_width', len(a)+1.), ('fp64_frobenius', float(np.sum(design**2)))):
                    result = witness_bound(design, residual, v, eta, steps, L)
                    row = dict(metadata, local_case=local, full_index=index, width=len(a), eta=eta,
                        updates=steps, cutoff_eta_eigenvalue=cutoff, L_source=label, L_upper=L,
                        original_spectral_cutoff=cutoff in ORIGINAL_CUTOFFS,
                        relative_witness_floor=result['lower_norm']/target_norm,
                        relative_spectral_floor=old_floor if cutoff in ORIGINAL_CUTOFFS else None,
                        frozen_relative_l2=float(np.linalg.norm(direct))/target_norm,
                        residual_reconstruction_error=float(np.linalg.norm(direct-predicted)),
                        witness_endpoint_overlap=float(abs(v@direct)), **result)
                    rows.append(row)
                    if steps == 100000 and label == 'analytic_width' and row['relative_witness_floor'] > .01:
                        candidates.append((row['relative_witness_floor'], row, v.copy()))
        if candidates:
            best = max(candidates, key=lambda item: item[0])
            arm = case['arm']
            if arm == 'repaired' or arm.startswith('s100_'):
                key = (arm, case['target'])
                if key not in best_cases or best[0] > best_cases[key][0]:
                    best_cases[key] = (*best, p.copy(), x.copy(), y.copy())
        write_csv(args.output/'witnesses.csv', rows)
        print(json.dumps(dict(case=local, **metadata)), flush=True)
    for (arm, target), (_, row, v, p, x, y) in best_cases.items():
        path = args.output/f'witness_{arm}_{target}.npz'
        a, b, c = p[:-1].reshape(3, -1)
        np.savez_compressed(path, p=p, a=a, b=b, c=c, d=p[-1], x=x, y=y,
                            v=v, witness=v, eta=np.array(eta), updates=np.array(row['updates']),
                            metadata=np.array(json.dumps(clean(row))))
        saved.append(dict(row, file=path.name, sha256=digest(path)))
    summary = dict(cases=len(choices), rows=len(rows), cutoffs=CUTOFFS, saved_witnesses=saved,
                   source_sha256=digest(args.input), full_manifest_sha256=digest(args.full_run/'manifest.json'),
                   helper_sha256=digest(Path(__file__)),
                   scope='Frozen geometry only; all n through budget in exact arithmetic, FP64 evaluation here.',
                   spectral_role='SVD chooses arbitrary witness; direct bound does not require certified singular vectors.',
                   selection='Largest analytic-width witness floor at100k per repaired/s100 reference arm and target, exceeding1%')
    (args.output/'manifest.json').write_text(json.dumps(clean(summary), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--full-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', default='30')
    parser.add_argument('--arms', default='repaired,s10_primary,s10_inverse,s100_primary,s100_inverse')
    run(parser.parse_args())
