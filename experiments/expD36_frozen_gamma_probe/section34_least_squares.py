"""Measure executed FP64 least-squares fits, with explicit truncation sensitivity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits


def features(x, a, b):
    return np.column_stack((np.tanh(x[:, None] * a + b), np.ones(len(x))))


def run(base, output):
    source = base / 'input.npz'
    data = np.load(source)
    manifest = json.loads((base / 'manifest.json').read_text())
    cutoffs = np.array([1e-12, 1e-14, 1e-16])
    xs = [data[key] for key in ('train_x', 'validation_x', 'eval_x')]
    main_targets = [data[key] for key in ('target', 'validation_target', 'eval_target')]
    for x, y in zip(xs, main_targets):
        raw = np.sin(2*np.pi*x) + .5*np.sin(6*np.pi*x) + .25*np.sin(14*np.pi*x)
        np.testing.assert_allclose(y, raw / manifest['target_normalizer'], rtol=2e-15, atol=2e-15)
    quadratic_scale = float(np.sqrt(np.mean(xs[0]**4)))
    target_sets = [main_targets, [x*x / quadratic_scale for x in xs]]
    target_names = ['mixed_sine', 'quadratic']
    rows, coefficients, spectra = [], [], []
    prediction_checks = []
    for index, geometry in enumerate(manifest['geometries']):
        a, b = data['a'][index], data['b'][index]
        phi = features(xs[0], a, b)
        np.testing.assert_array_equal(phi, data['features'][index])
        u, s, vt = svd(phi, full_matrices=False, lapack_driver='gesvd')
        spectra.append(s)
        other_features = [features(x, a, b) for x in xs[1:]]
        for target_name, targets in zip(target_names, target_sets):
            y = targets[0]
            for cutoff in cutoffs:
                keep = s > cutoff * s[0]
                weights = vt[keep].T @ ((u[:, keep].T @ y) / s[keep])
                predictions = [matrix @ weights for matrix in [phi, *other_features]]
                errors = [float(np.linalg.norm(p - t) / np.linalg.norm(t))
                          for p, t in zip(predictions, targets)]
                # Independent columnwise accumulation retains the actual FP64
                # activations, isolating cancellation in the readout arithmetic.
                check = np.full(len(xs[0]), np.longdouble(weights[-1]))
                for j in range(len(a)):
                    check += np.tanh(xs[0]*a[j] + b[j]).astype(np.longdouble) * np.longdouble(weights[j])
                difference = float(np.linalg.norm(np.asarray(check - predictions[0], dtype=float)) / np.linalg.norm(y))
                projected = u[:, keep] @ (u[:, keep].T @ y)
                row = dict(target=target_name, geometry=geometry['name'],
                           lambda_value=geometry['lambda_rms'], relative_cutoff=float(cutoff),
                           retained_rank=int(keep.sum()), train_error=errors[0],
                           validation_error=errors[1], eval_error=errors[2],
                           projected_train_residual=float(np.linalg.norm(projected-y)/np.linalg.norm(y)),
                           independent_accumulation_difference=difference,
                           independent_accumulation_train_error=float(np.sqrt(np.sum((check-y)**2))/np.linalg.norm(y)),
                           coefficient_l1=float(np.linalg.norm(weights, 1)),
                           coefficient_l2=float(np.linalg.norm(weights)),
                           coefficient_max=float(np.max(np.abs(weights))))
                rows.append(row)
                coefficients.append(weights)
                prediction_checks.append(difference)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'fits.npz', coefficients=np.array(coefficients),
                        singular_values=np.array(spectra), cutoffs=cutoffs)
    result = dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  solver='scipy.linalg.svd, gesvd; explicit retained-singular-value solution',
                  precision='float64', blas_threads=2, quadratic_target_normalizer=quadratic_scale,
                  longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant),
                  interpretation='Executed fits at three numerical truncations; no residual is certified as an irreducible capacity floor. The 1e-16 cutoff is at FP64 spectral resolution and is a sensitivity check.',
                  coefficient_layout='rows follow summary rows; final coefficient is output bias', rows=rows)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(f"{row['target']:10s} {row['lambda_value']:<8g} {row['relative_cutoff']:.0e} "
              f"rank={row['retained_rank']:3d} train={row['train_error']:.5g} "
              f"eval={row['eval_error']:.5g} l1={row['coefficient_l1']:.5g}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.base, args.output)
