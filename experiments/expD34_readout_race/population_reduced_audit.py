"""Unanchored own-state polynomial force audit of archived natural GD states."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import mechanism_persistence_kernel as kernel
from . import mechanism_polynomial as polynomial

HORIZONS = (0, 1, 2, 10, 100, 1000, 10000, 20000)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(numerator, denominator):
    """Undefined ratios stay missing; no force-size cutoff changes the result."""
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return None
    return float(numerator / denominator)


def comparison(a, exact, approximate, tracking):
    exact, approximate, tracking = map(np.asarray, (exact, approximate, tracking))
    en, an = np.linalg.norm(exact), np.linalg.norm(approximate)
    positive = np.maximum(-np.sign(a)*exact, 0.)
    predicted_positive = np.maximum(-np.sign(a)*approximate, 0.)
    error = np.linalg.norm(approximate-exact)
    pn = np.linalg.norm(positive)
    # At a=0, |a|' is the absolute velocity, rather than sign(a)*velocity.
    positive = np.where(a == 0, np.abs(exact), positive)
    predicted_positive = np.where(a == 0, np.abs(approximate), predicted_positive)
    pn = np.linalg.norm(positive)
    return dict(exact_norm=float(en), polynomial_norm=float(an),
                error_norm=float(error), relative_error=ratio(error, en),
                cosine=ratio(exact@approximate, en*an),
                exact_positive_norm=float(pn),
                positive_error_norm=float(np.linalg.norm(predicted_positive-positive)),
                positive_relative_error=ratio(np.linalg.norm(predicted_positive-positive), pn),
                signed_positive_discrepancy=float(np.sum(predicted_positive-positive)),
                signed_positive_discrepancy_over_exact_positive_norm=ratio(np.sum(predicted_positive-positive), pn),
                tracking_norm=float(np.linalg.norm(tracking)),
                tracking_relative=ratio(np.linalg.norm(tracking), en))


def gram_status(gram):
    if not np.isfinite(gram).all():
        return dict(gram_min_eigenvalue=None, gram_condition=None, gram_positive=False)
    eig = np.linalg.eigvalsh(gram)
    return dict(gram_min_eigenvalue=float(eig[0]),
                gram_condition=ratio(eig[-1], eig[0]) if eig[0] > 0 else None,
                gram_positive=bool(eig[0] > 0))


@jax.jit
def exact_fields(p, x, y):
    state = kernel.decomposition(p, x, y)
    return state['F'], state['R'], state['gram']


def clean(row):
    return {k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
            for k, v in row.items()}


def audit(base, output):
    if not jax.config.x64_enabled or any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('FP64 CPU required; launch in the campaign Slurm CPU allocation')
    root = base/'evidence/persistence_1bf7138'
    output.mkdir(parents=True, exist_ok=False)
    fields = {degree: jax.jit(lambda p, transform, target, degree=degree:
                              polynomial.field(p, transform, target, degree)) for degree in (3, 5)}
    sources, missing, rows, case_count = {}, [], 0, 0
    with (output/'reduced_fields.csv').open('w', newline='') as stream:
        writer = None
        for cohort in ('development', 'confirmation'):
            for panel in ('N128', 'N512', 'N1024', 'late'):
                prediction = root/f'feedback_{cohort}_{panel}'
                input_path = prediction/'inputs.npz'
                if not input_path.exists():
                    missing.append(str(input_path)); continue
                manifest_path = prediction/'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                if manifest['input_sha256'] != digest(input_path):
                    raise ValueError(f'Changed issued input: {input_path}')
                sources[str(input_path)] = digest(input_path)
                sources[str(manifest_path)] = digest(manifest_path)
                run = prediction.with_name(prediction.name+'_run')
                run_manifest = run/'manifest.json'
                if run_manifest.exists():
                    if json.loads(run_manifest.read_text())['prediction_sha256'] != digest(manifest_path):
                        raise ValueError(f'Mismatched continuation: {run}')
                    sources[str(run_manifest)] = digest(run_manifest)
                with np.load(input_path) as data:
                    x, yy = data['x'].copy(), data['y'].copy()
                    cases = json.loads(str(data['cases']))
                indices = [i for i, c in enumerate(cases) if c['arm'] == 'natural']
                case_count += len(indices)
                setup = {(i, degree): tuple(map(jnp.asarray, polynomial.modal_setup(x, yy[i], degree)))
                         for i in indices for degree in (3, 5)}
                xj = jnp.asarray(x)
                for horizon in HORIZONS:
                    snapshot = run/'snapshots'/f'{horizon:09d}.npz'
                    if not snapshot.exists():
                        missing.append(str(snapshot)); continue
                    sources[str(snapshot)] = digest(snapshot)
                    with np.load(snapshot) as data:
                        points, counts, failed = data['p'].copy(), data['count'].copy(), data['failed'].copy()
                    for i in indices:
                        p, y = jnp.asarray(points[i]), jnp.asarray(yy[i])
                        width = (len(p)-1)//3
                        a = points[i, :width]
                        exact, tracking, gram = map(np.asarray, exact_fields(p, xj, y))
                        exact_status = {'exact_'+k: v for k, v in gram_status(gram).items()}
                        top = np.argsort(np.abs(a), kind='stable')[-max(1, int(np.ceil(.1*width))):]
                        for degree in (3, 5):
                            force, coarse = map(np.asarray, fields[degree](p, *setup[i, degree]))
                            status = {'polynomial_'+k: v for k, v in gram_status(coarse).items()}
                            for name, selection in (('all', np.arange(width)), ('top10pct_abs_slope', top)):
                                row = {k: cases[i].get(k) for k in ('cohort', 'target', 'seed', 'nref', 'source_index')}
                                row.update(panel=prediction.name, horizon=horizon, width=width,
                                           actual_updates=int(counts[i]), failed=bool(failed[i]), degree=degree,
                                           subset=name, subset_size=len(selection),
                                           finite_fields=bool(np.isfinite(exact).all() and np.isfinite(force).all()),
                                           **exact_status, **status,
                                           **comparison(a[selection], exact[:width][selection],
                                                        force[:width][selection], tracking[:width][selection]))
                                if writer is None:
                                    writer = csv.DictWriter(stream, fieldnames=list(row))
                                    writer.writeheader()
                                writer.writerow(clean(row)); rows += 1
                        stream.flush()
                print(json.dumps(dict(panel=prediction.name, rows=rows)), flush=True)
    result = dict(role='Retrospective actual-state force audit; no anchoring, fitting, or integration',
                  rows=rows, natural_cases=case_count, horizons=HORIZONS, missing=missing, sources=sources,
                  code_sha256={Path(p).name: digest(p) for p in (__file__, kernel.__file__, polynomial.__file__)},
                  force_definition='Full complement of affine sample-space projection',
                  surrogate_definition='Projected degree-3/5 tanh Taylor network at the actual parameters',
                  top_subset='ceil(width/10) largest current absolute slopes; stable index tie break',
                  caveat='Positive eigenvalues and finite solves are diagnostics, not interval certificates')
    (output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(rows=rows, natural_cases=case_count, missing=len(missing))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.base, args.output)
