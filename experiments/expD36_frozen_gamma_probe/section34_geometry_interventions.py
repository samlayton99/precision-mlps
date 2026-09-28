"""Separate learned center placement from slope magnitudes in the W512 probe."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def center_coverage(centers):
    """Report numerical coalescence and domain occupancy, not effective rank."""
    counts = np.histogram(centers, bins=np.linspace(-1, 1, 33))[0]
    return dict(count=len(centers),
        center_quantiles=np.quantile(centers, [0, .05, .25, .5, .75, .95, 1]).tolist(),
        distinct_centers_rounded_12dp=int(len(np.unique(np.round(centers, 12)))),
        occupied_32_equal_interval_bins=int(np.count_nonzero(counts)),
        maximum_count_in_one_bin=int(counts.max())) if len(centers) else dict(count=0)


def prepare(learned, base, output):
    source = np.load(learned/'input.npz')
    reference = np.load(base/'input.npz')
    manifest = json.loads((learned/'manifest.json').read_text())
    base_manifest = json.loads((base/'manifest.json').read_text())
    assert digest(learned/'input.npz') == manifest['input_sha256']
    assert digest(base/'input.npz') == base_manifest['input_sha256']
    assert manifest['width'] == base_manifest['width'] == 512
    h = manifest['spacing']
    assert h == base_manifest['spacing']
    shared = ['target', 'train_x', 'validation_x', 'eval_x',
              'validation_target', 'eval_target']
    for key in shared:
        np.testing.assert_array_equal(source[key], reference[key])
    centers = -reference['b'][0]/reference['a'][0]
    np.testing.assert_allclose(np.diff(centers), h, rtol=1e-12, atol=1e-15)
    candidates = [(i, g) for i, g in enumerate(manifest['geometries'])
                  if g['slope_multiplier'] == 1 and g['snapshot_step'] > 0]
    final_step = max(g['snapshot_step'] for _, g in candidates)
    selected = [(i, g) for i, g in candidates if g['snapshot_step'] == final_step]
    assert {(g['optimizer'], g['seed']) for _, g in selected} == {
        (optimizer, seed) for optimizer in ('adam', 'gd') for seed in range(5)}
    assert len(selected) == 10
    aa, bb, rows, diagnostics = [], [], [], []
    for index, geometry in selected:
        a, b = source['a'][index], source['b'][index]
        assert np.all(a != 0)
        learned_centers = -b/a
        assert np.all(np.isfinite(learned_centers))
        optimizer, seed = geometry['optimizer'], geometry['seed']
        random_seed = [34, 20260924, seed, 0 if optimizer == 'adam' else 1]
        permutation = np.random.default_rng(random_seed).permutation(512)
        permuted_a = a[permutation]
        uniform_b = -permuted_a*centers
        common_a = np.copysign(.25/h, a)
        common_b = b*(common_a/a)
        np.testing.assert_array_equal(np.sort(permuted_a), np.sort(a))
        np.testing.assert_allclose(-uniform_b/permuted_a, centers, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(-common_b/common_a, learned_centers, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(np.signbit(common_a), np.signbit(a))
        np.testing.assert_allclose(abs(common_a)*h, .25, rtol=1e-14)
        for family, new_a, new_b in [
                ('uniform_centers_learned_slopes', permuted_a, uniform_b),
                ('learned_centers_common_slope', common_a, common_b)]:
            aa.append(new_a); bb.append(new_b)
            rows.append(dict(name=f'{optimizer}_seed{seed}_{family}', family=family,
                optimizer=optimizer, seed=seed, snapshot_step=final_step,
                gamma=.25/h if family == 'learned_centers_common_slope' else None,
                lambda_rms=float(h*np.sqrt(np.mean(new_a**2))),
                source_error=geometry['source_error'], source_geometry=geometry,
                source_geometry_index=index,
                permutation=permutation.tolist() if family == 'uniform_centers_learned_slopes' else None,
                permutation_seed=random_seed if family == 'uniform_centers_learned_slopes' else None))
        scaled = h*abs(a)
        thresholds = [.001, .01, .05, .1, .25]
        diagnostics.append(dict(optimizer=optimizer, seed=seed,
            source_geometry=geometry['name'],
            centers_inside_interval=int(np.count_nonzero(abs(learned_centers) <= 1)),
            fraction_centers_inside_interval=float(np.mean(abs(learned_centers) <= 1)),
            scaled_slope_thresholds=thresholds,
            counts_above_threshold=[int(np.count_nonzero(scaled >= t)) for t in thresholds],
            inside_center_counts_above_threshold=[int(np.count_nonzero((scaled >= t) & (abs(learned_centers) <= 1))) for t in thresholds],
            scaled_slope_quantiles=np.quantile(scaled, [0, .5, .9, .99, 1]).tolist(),
            all_center_coverage=center_coverage(learned_centers),
            active_inside_center_coverage=center_coverage(
                learned_centers[(scaled >= .01) & (abs(learned_centers) <= 1)])))
    a, b = np.array(aa), np.array(bb)
    x = source['train_x']
    features = np.concatenate((np.tanh(x[None, :, None]*a[:, None, :]+b[:, None, :]),
                               np.ones((len(a), len(x), 1))), axis=-1)
    assert a.shape == b.shape == (20, 512)
    assert features.shape == (20, len(x), 513)
    assert np.isfinite(features).all()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'input.npz').exists():
        raise FileExistsError('Use a fresh output directory')
    np.savez_compressed(output/'input.npz', a=a, b=b, features=features,
                        **{key: source[key] for key in shared})
    result = dict(manifest)
    result.update(geometries=rows, input_sha256=digest(output/'input.npz'),
                  intervention_source=str(learned), intervention_source_sha256=digest(learned/'input.npz'),
                  uniform_reference_source=str(base), uniform_reference_sha256=digest(base/'input.npz'),
                  intervention_code_sha256=digest(Path(__file__)),
                  parameter_initialization='zero readout and optimizer moments for all frozen interventions',
                  intervention_scope='Final learned dictionaries only; one seeded slope permutation per optimizer and seed; common replacement slope magnitude 0.25/h.')
    (output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'source_geometry_diagnostics.json').write_text(json.dumps(
        dict(cases=diagnostics, count_definition='Magnitude thresholds on h*abs(a); all thresholds and counts reported, not a fitted definition of active features.'),
        indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), feature_shape=features.shape, final_step=final_step)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--learned', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.learned, args.base, args.output)
