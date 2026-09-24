"""Recompute gamma-dependent Section 3.4 error bounds on the actual W512 geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from .direct_ratio_interval import calculate, error_curve

LAMBDAS = [.03125, .0625, .125, .25]
INDICES = [0, 1, 3, 4]


def curves(summary, arrays, steps):
    weights = arrays['actual_target_weights']
    resolved = arrays['resolved']
    count = len(weights)
    lower = error_curve(.5*arrays['rho_upper'][:count], weights*resolved, 0., steps)
    reference = error_curve(.5*arrays['actual_rho'], weights, summary['projection_remainder'], steps)
    upper = error_curve(.5*arrays['rho_lower'][:count]*resolved, weights, summary['projection_remainder'], steps)
    return lower, reference, upper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2_000_000)
    p.add_argument('--gd', type=Path, help='Optional executed GD directory for trajectory checks')
    args = p.parse_args()
    data = np.load(args.input)
    x, y = data['train_x'], data['target']
    args.output.mkdir(parents=True, exist_ok=True)
    steps = np.unique(np.r_[0, np.geomspace(1, args.steps, 2000).astype(int), np.linspace(0, args.steps, 2001).astype(int)])
    all_summary = []
    for index, lam in zip(INDICES, LAMBDAS):
        a, b = data['a'][index], data['b'][index]
        gamma = float(a[0])
        centers = -b/a
        np.testing.assert_allclose(a, gamma)
        np.testing.assert_allclose(gamma*(centers[1]-centers[0]), lam)
        passes = []
        for order, padding in [(10, 20.), (16, 24.)]:
            stem = f'lambda{lam:g}_q{order}_p{padding:g}'
            started = time.monotonic()
            if (args.output/f'{stem}.json').exists() and (args.output/f'{stem}.npz').exists():
                summary = json.loads((args.output/f'{stem}.json').read_text())
                with np.load(args.output/f'{stem}.npz') as saved:
                    arrays = {k: saved[k] for k in saved.files}
                if summary['input_sha256'] != hashlib.sha256(args.input.read_bytes()).hexdigest():
                    raise ValueError('Existing results use a different input')
            else:
                summary, arrays = calculate(x, centers, gamma, y, order=order, padding=padding)
            lower, reference, upper = curves(summary, arrays, steps)
            arrays.update(steps=steps, lower_error=lower, reference_error=reference, upper_error=upper)
            summary.update(relative_bandwidth=lam, input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), seconds=time.monotonic()-started, horizon=args.steps, final_lower_error=float(lower[-1]), final_reference_error=float(reference[-1]), final_upper_error=float(upper[-1]))
            np.savez_compressed(args.output/f'{stem}.npz', **arrays)
            (args.output/f'{stem}.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
            print(json.dumps({'case': stem, 'seconds': summary['seconds'], 'lower': lower[-1], 'reference': reference[-1]}), flush=True)
            passes.append((summary, arrays))
        base, fine = passes
        item = dict(relative_bandwidth=lam, gamma=gamma, coarse=base[0], refined=fine[0], refinement_max_absolute_lower_curve_difference=float(np.max(np.abs(base[1]['lower_error']-fine[1]['lower_error']))), refinement_max_absolute_ratio_endpoint_difference=float(np.max(np.abs(base[1]['rho_upper']-fine[1]['rho_upper']))))
        if args.gd:
            observed = np.load(args.gd/'raw_error.npy', mmap_mode='r')
            if len(observed) <= args.steps:
                raise ValueError('Executed GD does not cover the bound horizon')
            gi = INDICES.index(index)
            # Validate every recorded update in bounded-memory blocks.
            max_lower_violation, max_reference_discrepancy = 0., 0.
            for begin in range(0, args.steps+1, 4096):
                ns = np.arange(begin, min(begin+4096, args.steps+1))
                lower, reference, _ = curves(fine[0], fine[1], ns)
                actual = observed[ns, gi]
                max_lower_violation = max(max_lower_violation, float(np.max(lower-actual)))
                max_reference_discrepancy = max(max_reference_discrepancy, float(np.max(np.abs(reference-actual))))
            item.update(executed_max_lower_violation=max_lower_violation, executed_max_absolute_reference_discrepancy=max_reference_discrepancy, executed_final_error=float(observed[args.steps, gi]))
        all_summary.append(item)
        (args.output/'summary.json').write_text(json.dumps({'cases': all_summary, 'horizon': args.steps, 'numerical_status': 'FP64 checked evaluations, with quadrature refinement; not interval-arithmetic certificates'}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
